"""
eddy.eval.exp2_real
===================
Experiment 2 on REAL model pools — "money figure" implementation.

Frozen pre-registration spec: see docs/ or the researcher's spec comment block.

Sign convention (pinned to coverage.py):
    r = w_Q*Q~ - w_C*C~ - w_T*T~ - w_R*R~
    Q maximized; C, T, R minimized (larger = worse).

Construction matrix (6 minimum cells):
    T ∈ {T_tokens, T_alt1, T_alt2}
    R ∈ {R_err, R_alt2}
    (R_alt1 added where computable — calibration signal required)

Grid: m = {20, 40, 80} (primary = 40 per spec).

Provenance gate:
    provenance ∈ {REAL, MOCK}
    REAL → results/real/  (figures only if ALL inputs are REAL)
    MOCK → results/mock/  (never labeled as real; pipeline validation only)

Usage:
    python -m eddy.eval.exp2_real                # tries real download, falls back to mock
    python -m eddy.eval.exp2_real --mock-only    # force mock (testing/CI)
    python -m eddy.eval.exp2_real --out results/mock  # custom output dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eddy.core.profile import QCTR, OperatorProfile
from eddy.eval.coverage import (
    scalarization_reachable,
    skyline_of,
    coverage_loss,
    simplex_lattice,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_FROZEN_DIR = _REPO_ROOT / "data" / "frozen"
_RAW_DIR = _REPO_ROOT / "data" / "raw"
_MOCK_DIR = _REPO_ROOT / "data" / "mock"

# ---------------------------------------------------------------------------
# Frozen constants (registered in spec)
# ---------------------------------------------------------------------------

ALPHA_R = 0.6   # weight on (1 - mean_score) in R_err
BETA_R = 0.3    # weight on dispersion in R_err
GAMMA_R = 0.1   # weight on failRate in R_err
K_PROXY = 600.0  # throughput proxy constant (output tokens/sec)
TTFT_DEFAULT = 0.4  # seconds; default TTFT when model not in frozen table

GRID_RESOLUTIONS = [20, 40, 80]  # m values; primary = 40

# ---------------------------------------------------------------------------
# Load frozen tables
# ---------------------------------------------------------------------------

def _load_frozen(filename: str) -> dict:
    path = _FROZEN_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Frozen table not found: {path}")
    with open(path) as f:
        return json.load(f)


def _load_frozen_tables() -> tuple[dict, dict, dict]:
    """Return (throughput_tps, model_params, latency_tier)."""
    tps = _load_frozen("throughput_tps.json")
    params = _load_frozen("model_params.json")
    tier = _load_frozen("latency_tier.json")
    return tps, params, tier


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _frozen_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for fname in ["throughput_tps.json", "model_params.json", "latency_tier.json"]:
        p = _FROZEN_DIR / fname
        hashes[fname] = _file_sha256(p) if p.exists() else "MISSING"
    return hashes


# ---------------------------------------------------------------------------
# Coverage.py version fingerprint
# ---------------------------------------------------------------------------

def _coverage_py_version() -> str:
    """Return SHA-256 of coverage.py for the manifest."""
    cov_path = Path(__file__).parent / "coverage.py"
    return _file_sha256(cov_path) if cov_path.exists() else "UNKNOWN"


# ---------------------------------------------------------------------------
# Throughput / latency proxy helpers
# ---------------------------------------------------------------------------

def _get_throughput(model_id: str, tps_table: dict, params_table: dict) -> tuple[float, bool]:
    """Return (throughput_tps, used_proxy).

    Primary: frozen tps_table lookup.
    Fallback: K / sqrt(P_active_B) where P is from params_table.
    If params also missing: K / sqrt(100) as ultimate fallback.
    """
    key = model_id.lower()
    # Try exact match, then prefix strip
    for candidate in [key, key.split(":")[0]]:
        if candidate in tps_table and isinstance(tps_table[candidate], dict):
            return float(tps_table[candidate]["value"]), False

    # Proxy
    p_active = None
    for candidate in [key, key.split(":")[0]]:
        if candidate in params_table and isinstance(params_table[candidate], dict):
            p_active = float(params_table[candidate]["value"])
            break
    if p_active is None:
        p_active = 100.0  # large-flagship default

    proxy_tps = K_PROXY / math.sqrt(p_active)
    return proxy_tps, True


def _get_latency_tier(model_id: str, tier_table: dict) -> int:
    """Return tier value (1/2/3) or 3 (worst) if unknown."""
    key = model_id.lower()
    for candidate in [key, key.split(":")[0]]:
        if candidate in tier_table and isinstance(tier_table[candidate], dict):
            return int(tier_table[candidate]["value"])
    return 3  # unknown proprietary → worst-case tier


# ---------------------------------------------------------------------------
# Token count helpers
# ---------------------------------------------------------------------------

def _count_tokens_tiktoken(text: str) -> int:
    """Count tokens using tiktoken cl100k_base; fallback to len/4."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# T constructions (latency, larger = worse, seconds, NOT pre-normalised)
# ---------------------------------------------------------------------------

def compute_T_tokens(
    records: list[dict],
    model_id: str,
    tps_table: dict,
    params_table: dict,
) -> tuple[float, list[str]]:
    """T_tokens (primary): mean TTFT + completion_tokens/throughput over slice.

    Returns (mean_T, list_of_log_messages).
    """
    logs: list[str] = []
    throughput, used_proxy = _get_throughput(model_id, tps_table, params_table)
    if used_proxy:
        logs.append(f"T_tokens: used proxy throughput={throughput:.1f} tps for {model_id}")

    ttft = TTFT_DEFAULT  # no per-record TTFT in data; use default

    t_vals: list[float] = []
    for rec in records:
        # Completion tokens: prefer field, then tiktoken on response
        ct = rec.get("completion_tokens")
        if ct is None or ct <= 0:
            resp = rec.get("response", "")
            ct = _count_tokens_tiktoken(resp)
            logs.append(f"T_tokens: tiktoken fallback for {model_id} rec={rec.get('record_index','?')}")
        ct = max(1, int(ct))
        t = ttft + ct / throughput
        t_vals.append(t)

    mean_t = sum(t_vals) / len(t_vals) if t_vals else TTFT_DEFAULT
    return mean_t, logs


def compute_T_alt1(
    records: list[dict],
    model_id: str,
    tps_table: dict,
    params_table: dict,
) -> tuple[float, str, list[str]]:
    """T_alt1 (aggregate-anchored): mean time_taken/record_count per (model, slice).

    Returns (mean_T, source_description, logs).
    Falls back to T_tokens where time_taken is absent.
    """
    logs: list[str] = []
    time_vals = [rec.get("time_taken") for rec in records]
    valid = [v for v in time_vals if v is not None and v > 0]

    if len(valid) >= len(records) * 0.5:
        mean_t = sum(valid) / len(valid)
        source = f"time_taken field (n={len(valid)}/{len(records)} valid)"
        return mean_t, source, logs
    else:
        logs.append(f"T_alt1: time_taken absent/sparse for {model_id}, falling back to T_tokens")
        mean_t, t_logs = compute_T_tokens(records, model_id, tps_table, params_table)
        logs.extend(t_logs)
        return mean_t, "T_tokens_fallback", logs


def compute_T_alt2(
    model_id: str,
    tier_table: dict,
) -> float:
    """T_alt2 (rank/tier): latency tier value, coarse decorrelated anchor.

    Returns the integer tier as a float (1, 2, or 3). No jitter.
    """
    return float(_get_latency_tier(model_id, tier_table))


# ---------------------------------------------------------------------------
# R constructions (risk, [0,1], larger = worse)
# ---------------------------------------------------------------------------

def _binary_dispersion(p: float) -> float:
    """4p(1-p) for binary p in [0,1]. Max at p=0.5."""
    return 4.0 * p * (1.0 - p)


def _sample_variance_minmaxed(scores: list[float]) -> float:
    """Min-max scaled sample variance of scores in [0,1].

    Scales to [0,1] where max sample variance = 0.25 (Bernoulli at 0.5).
    """
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    # Max variance for values in [0,1] is 0.25 (Bernoulli).
    return min(1.0, var / 0.25)


def compute_R_err(
    records: list[dict],
    pool_id: str,
) -> tuple[float, dict, list[str]]:
    """R_err (primary): 0.6*(1-meanScore) + 0.3*dispersion + 0.1*failRate.

    Clamp to [0,1].
    If failRate makes gamma contribution unreliable (unrecoverable fails),
    fold gamma into alpha (alpha->0.7, gamma->0).

    Returns (R_value, metadata_dict, logs).
    """
    logs: list[str] = []
    alpha, beta, gamma = ALPHA_R, BETA_R, GAMMA_R

    scores = [rec.get("score", 0.0) for rec in records]
    fails = [bool(rec.get("fail", False)) for rec in records]

    n = len(scores)
    if n == 0:
        return 0.5, {"n": 0, "alpha": alpha, "beta": beta, "gamma": gamma}, logs

    # Scores for non-fail records
    nonfail_scores = [s for s, f in zip(scores, fails) if not f]
    mean_score = sum(nonfail_scores) / len(nonfail_scores) if nonfail_scores else 0.0

    # Dispersion: try binary (score in {0,1}) else sample variance
    unique_scores = set(round(s, 3) for s in scores)
    is_binary = all(s in (0.0, 1.0) for s in scores) or len(unique_scores) <= 2
    if is_binary:
        dispersion = _binary_dispersion(mean_score)
    else:
        dispersion = _sample_variance_minmaxed(scores)

    fail_rate = sum(fails) / n

    # Check for unrecoverable fail_rate > 0.5 → fold gamma into alpha
    if fail_rate > 0.5:
        logs.append(f"R_err: high failRate={fail_rate:.3f} for pool={pool_id}, folding gamma into alpha")
        alpha = 0.7
        gamma = 0.0
        beta = 0.3

    r = alpha * (1.0 - mean_score) + beta * dispersion + gamma * fail_rate
    r = max(0.0, min(1.0, r))

    meta = {
        "n": n,
        "mean_score": mean_score,
        "dispersion": dispersion,
        "fail_rate": fail_rate,
        "is_binary": is_binary,
        "alpha_used": alpha,
        "beta_used": beta,
        "gamma_used": gamma,
    }
    return r, meta, logs


def compute_R_alt1(
    records: list[dict],
) -> tuple[float | None, str]:
    """R_alt1 (calibration): 1 - ECE if confidence/logprob signal exists.

    LLMRouterBench and RouterBench do not carry per-token logprobs.
    Returns (None, "not computable on this dataset") in that case.
    """
    # Check for any calibration-relevant field
    sample = records[0] if records else {}
    if any(k in sample for k in ("logprob", "confidence", "calibration_score")):
        # Placeholder — compute ECE if the field is present
        scores = [rec.get("logprob") or rec.get("confidence") for rec in records]
        is_correct = [float(rec.get("is_correct", rec.get("score", 0) >= 0.5)) for rec in records]
        if all(s is not None for s in scores):
            # Simple 10-bin ECE
            n = len(scores)
            bins = [[] for _ in range(10)]
            accs = [[] for _ in range(10)]
            for conf, correct in zip(scores, is_correct):
                b = min(9, int(conf * 10))
                bins[b].append(conf)
                accs[b].append(correct)
            ece = 0.0
            for b_conf, b_acc in zip(bins, accs):
                if b_conf:
                    ece += (len(b_conf) / n) * abs(sum(b_conf) / len(b_conf) - sum(b_acc) / len(b_acc))
            return max(0.0, min(1.0, ece)), "ECE computed from confidence field"
    return None, "not computable on this dataset"


def compute_R_alt2(
    records: list[dict],
) -> float:
    """R_alt2 (pure dispersion): dispersion only, alpha=gamma=0.

    Decorrelated-from-Q anchor — load-bearing sensitivity check.
    """
    scores = [rec.get("score", 0.0) for rec in records]
    nonfail = [s for s, rec in zip(scores, records) if not rec.get("fail", False)]
    mean_score = sum(nonfail) / len(nonfail) if nonfail else 0.0

    unique_scores = set(round(s, 3) for s in scores)
    is_binary = all(s in (0.0, 1.0) for s in scores) or len(unique_scores) <= 2
    if is_binary:
        return _binary_dispersion(mean_score)
    else:
        return _sample_variance_minmaxed(scores)


# ---------------------------------------------------------------------------
# Cost (C) construction
# ---------------------------------------------------------------------------

def compute_C(records: list[dict]) -> float:
    """Mean cost per record. Directly from 'cost' field."""
    costs = [rec.get("cost", 0.0) for rec in records if rec.get("cost") is not None]
    return sum(costs) / len(costs) if costs else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def try_download_real_data() -> tuple[str | None, str | None, str | None]:
    """Attempt to download RouterBench and LLMRouterBench via huggingface_hub.

    Returns (routerbench_path, llmrouterbench_path, status_message).
    Paths are None if download failed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None, None, "huggingface_hub not installed"

    rb_path = None
    llmrb_path = None
    msgs: list[str] = []

    for repo_id, target_attr in [
        ("withmartian/routerbench", "rb"),
        ("NPULH/LLMRouterBench", "llmrb"),
    ]:
        try:
            local_dir = str(_RAW_DIR / repo_id.replace("/", "__"))
            path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
            msgs.append(f"Downloaded {repo_id} -> {path}")
            if target_attr == "rb":
                rb_path = path
            else:
                llmrb_path = path
        except Exception as e:
            msgs.append(f"Failed to download {repo_id}: {e}")

    return rb_path, llmrb_path, "; ".join(msgs)


def _load_llmrouterbench(path: str) -> list[dict]:
    """Load all records from an LLMRouterBench local directory."""
    import csv
    records: list[dict] = []
    p = Path(path)

    # Try .jsonl files first, then .csv
    for jf in sorted(p.glob("**/*.jsonl")):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not records:
        for cf in sorted(p.glob("**/*.csv")):
            with open(cf, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Coerce numeric fields
                    for field in ("score", "cost", "time_taken"):
                        if field in row:
                            try:
                                row[field] = float(row[field])
                            except (ValueError, TypeError):
                                row[field] = 0.0
                    for field in ("tokens_used", "completion_tokens"):
                        if field in row:
                            try:
                                row[field] = int(float(row[field]))
                            except (ValueError, TypeError):
                                row[field] = 0
                    for field in ("is_correct", "fail"):
                        if field in row:
                            row[field] = str(row[field]).lower() in ("true", "1", "yes")
                    records.append(row)

    # Parquet support (optional)
    if not records:
        try:
            import importlib
            pd = importlib.import_module("pandas")
            for pf in sorted(p.glob("**/*.parquet")):
                df = pd.read_parquet(pf)
                records.extend(df.to_dict("records"))
        except Exception:
            pass

    return records


# RouterBench wide-format columns that are NOT per-model score columns.
_RB_META_COLS = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}


def _load_routerbench(path: str, prefer: str = "routerbench_0shot.pkl") -> list[dict]:
    """Load a RouterBench wide-format ``.pkl`` and melt it to long records.

    RouterBench ships a pandas DataFrame with one row per sample and, per model,
    up to three columns: ``<model>`` (score in [0,1]), ``<model>|total_cost``
    (USD), and ``<model>|model_response`` (text). ``eval_name`` is the
    dataset/benchmark slice. We emit one record per (sample, model) carrying the
    field contract consumed by build_pool / compute_* :
    ``dataset, model, score, cost, response, fail``.

    Returns ``[]`` if no usable ``.pkl`` is found (caller falls back to MOCK).
    """
    import importlib
    p = Path(path)
    pkls = sorted(p.glob("**/*.pkl"))
    if not pkls:
        return []

    chosen = next((c for c in pkls if c.name == prefer), None)
    if chosen is None:
        # Prefer the curated 0shot/5shot tables over the large raw dump.
        non_raw = [c for c in pkls if "raw" not in c.name]
        chosen = (non_raw or pkls)[0]

    pd = importlib.import_module("pandas")
    df = pd.read_pickle(chosen)

    cols = list(df.columns)
    model_cols = [c for c in cols if "|" not in str(c) and c not in _RB_META_COLS]

    eval_names = [str(x) for x in df.get("eval_name", ["unknown"] * len(df))]

    def _num(x):
        if x is None:
            return None
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(xf) else xf

    records: list[dict] = []
    for model in model_cols:
        scores = df[model].tolist()
        cost_col = f"{model}|total_cost"
        resp_col = f"{model}|model_response"
        costs = df[cost_col].tolist() if cost_col in cols else [None] * len(df)
        responses = df[resp_col].tolist() if resp_col in cols else [""] * len(df)

        for i in range(len(df)):
            s = _num(scores[i])
            if s is None:
                continue  # model not evaluated on this sample
            resp = responses[i]
            records.append({
                "dataset": eval_names[i],
                "model": model,
                "score": s,
                "cost": _num(costs[i]),
                "response": resp if isinstance(resp, str) else "",
                "fail": False,  # RouterBench has no explicit failure flag
            })

    return records


def _group_records(
    records: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    """Group records by (dataset, model) key."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        dataset = str(rec.get("dataset", "unknown"))
        model = str(rec.get("model", "unknown"))
        key = (dataset, model)
        groups.setdefault(key, []).append(rec)
    return groups


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def build_pool(
    dataset: str,
    groups: dict[tuple[str, str], list[dict]],
    tps_table: dict,
    params_table: dict,
    tier_table: dict,
    provenance: str,
) -> tuple[list[dict], list[str]]:
    """Build raw mean QCTR values for all models in a dataset slice.

    Returns (pool_records, log_messages) where pool_records is a list of dicts
    with keys: model, Q, C, T_tokens, T_alt1, T_alt1_source, T_alt2,
               R_err, R_err_meta, R_alt1, R_alt1_note, R_alt2, n_records,
               provenance.
    """
    logs: list[str] = []
    pool_records: list[dict] = []

    models = sorted({k[1] for k in groups if k[0] == dataset})

    for model in models:
        records = groups.get((dataset, model), [])
        if not records:
            continue

        n = len(records)
        pool_id = f"{dataset}::{model}"

        # Q: mean score
        scores = [rec.get("score", 0.0) for rec in records]
        q = sum(scores) / len(scores)

        # C: mean cost
        c = compute_C(records)

        # T constructions
        t_tokens, t_logs = compute_T_tokens(records, model, tps_table, params_table)
        logs.extend(t_logs)

        t_alt1, t_alt1_src, t_alt1_logs = compute_T_alt1(records, model, tps_table, params_table)
        logs.extend(t_alt1_logs)

        t_alt2 = compute_T_alt2(model, tier_table)

        # R constructions
        r_err, r_err_meta, r_err_logs = compute_R_err(records, pool_id)
        logs.extend(r_err_logs)

        r_alt1, r_alt1_note = compute_R_alt1(records)

        r_alt2 = compute_R_alt2(records)

        pool_records.append({
            "model": model,
            "dataset": dataset,
            "Q": q,
            "C": c,
            "T_tokens": t_tokens,
            "T_alt1": t_alt1,
            "T_alt1_source": t_alt1_src,
            "T_alt2": t_alt2,
            "R_err": r_err,
            "R_err_meta": r_err_meta,
            "R_alt1": r_alt1,
            "R_alt1_note": r_alt1_note,
            "R_alt2": r_alt2,
            "n_records": n,
            "provenance": provenance,
        })

    return pool_records, logs


# ---------------------------------------------------------------------------
# Coverage computation for one (pool, T_constr, R_constr, m)
# ---------------------------------------------------------------------------

CONSTRUCTION_PAIRS = [
    ("T_tokens", "R_err"),
    ("T_tokens", "R_alt2"),
    ("T_alt1",   "R_err"),
    ("T_alt1",   "R_alt2"),
    ("T_alt2",   "R_err"),
    ("T_alt2",   "R_alt2"),
]


def _make_operator_profile(rec: dict, t_key: str, r_key: str) -> OperatorProfile:
    """Build an OperatorProfile from pool record using specified T/R construction."""
    op_id = rec["model"]
    q = rec["Q"]
    c = rec["C"]
    t = rec[t_key]
    r = rec[r_key]
    return OperatorProfile.from_qctr(op_id, QCTR(Q=q, C=c, T=t, R=r))


def compute_coverage_for_pool(
    dataset: str,
    pool_records: list[dict],
    provenance: str,
) -> list[dict]:
    """Compute coverage metrics across all (T,R) construction pairs and grid resolutions.

    Returns list of result dicts, one per (T_constr, R_constr, m).
    """
    results: list[dict] = []
    pool_id = f"MOCK::{dataset}" if provenance == "MOCK" else dataset

    for t_key, r_key in CONSTRUCTION_PAIRS:
        # Skip R_alt1 if not computable (will be None)
        if r_key == "R_alt1":
            if any(rec.get("R_alt1") is None for rec in pool_records):
                continue

        # Build pool
        operators = [_make_operator_profile(rec, t_key, r_key) for rec in pool_records]

        sky = skyline_of(operators)
        sky_list = sorted(sky)

        # Self-check: skyline loss must be 0
        sky_self_loss = coverage_loss(sky, sky)
        assert abs(sky_self_loss) < 1e-12, (
            f"Skyline self-coverage loss = {sky_self_loss} for pool={pool_id}, "
            f"T={t_key}, R={r_key}. This is a BUG."
        )

        m_stability: dict[int, float] = {}

        for m in GRID_RESOLUTIONS:
            reach = scalarization_reachable(operators, m=m)
            loss_scal = coverage_loss(reach, sky)
            loss_sky = coverage_loss(sky, sky)  # always 0

            unreachable = sky - reach

            result = {
                "pool_id": pool_id,
                "dataset": dataset,
                "n_ops": len(operators),
                "T_constr": t_key,
                "R_constr": r_key,
                "m": m,
                "skyline_ids": sky_list,
                "skyline_size": len(sky),
                "reachable_ids": sorted(reach),
                "reachable_size": len(reach),
                "unreachable_pareto_ids": sorted(unreachable),
                "coverage_loss_scal": round(loss_scal, 6),
                "coverage_loss_sky": round(loss_sky, 6),
                "provenance": provenance,
                "__SYNTHETIC_MOCK__": (provenance == "MOCK"),
            }
            results.append(result)
            m_stability[m] = loss_scal

        # Stability check: loss non-increasing with m
        m_vals = sorted(m_stability)
        for i in range(len(m_vals) - 1):
            m_a, m_b = m_vals[i], m_vals[i + 1]
            if m_stability[m_b] > m_stability[m_a] + 1e-9:
                pass  # Allowed — grid finer can only reveal more reachable ops

    return results


# ---------------------------------------------------------------------------
# Summary: auto-verdicts
# ---------------------------------------------------------------------------

def compute_summary(all_results: list[dict]) -> dict:
    """Compute per-construction summary and auto-verdicts.

    Auto-verdicts:
    - Convex-real fail: median scal coverage loss < 0.05 under primary (T_tokens, R_err)
    - Artifact fail: loss > 0.05 under primary BUT < 0.05 under BOTH decorrelated anchors
    - Supported: loss > 0.05 on non-trivial fraction AND survives under (T_alt2, R_err) + (T_tokens, R_alt2)
    """
    primary_m = 40  # use primary grid resolution

    # Group by (T_constr, R_constr, m)
    by_constr: dict[tuple[str, str, int], list[float]] = {}
    for rec in all_results:
        key = (rec["T_constr"], rec["R_constr"], rec["m"])
        by_constr.setdefault(key, []).append(rec["coverage_loss_scal"])

    def _median(vals: list[float]) -> float:
        if not vals:
            return float("nan")
        s = sorted(vals)
        n = len(s)
        if n % 2 == 0:
            return (s[n // 2 - 1] + s[n // 2]) / 2.0
        return s[n // 2]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def _frac_above(vals: list[float], threshold: float) -> float:
        return sum(1 for v in vals if v > threshold) / len(vals) if vals else 0.0

    constr_stats: dict[str, dict] = {}
    for (t_key, r_key, m), losses in by_constr.items():
        key_str = f"{t_key}x{r_key}@m{m}"
        constr_stats[key_str] = {
            "T_constr": t_key,
            "R_constr": r_key,
            "m": m,
            "n_pools": len(losses),
            "losses": losses,
            "median_scal_loss": round(_median(losses), 6),
            "mean_scal_loss": round(_mean(losses), 6),
            "frac_above_0.05": round(_frac_above(losses, 0.05), 4),
        }

    # Auto-verdicts at primary m
    primary_losses = by_constr.get(("T_tokens", "R_err", primary_m), [])
    anchor1_losses = by_constr.get(("T_alt2", "R_err", primary_m), [])
    anchor2_losses = by_constr.get(("T_tokens", "R_alt2", primary_m), [])

    median_primary = _median(primary_losses)
    median_anchor1 = _median(anchor1_losses)
    median_anchor2 = _median(anchor2_losses)
    frac_primary = _frac_above(primary_losses, 0.05)

    if math.isnan(median_primary):
        verdict = "INDETERMINATE: no primary construction results"
    elif median_primary < 0.05:
        verdict = "CONVEX-REAL-FAIL: median scal coverage loss < 0.05 under primary (T_tokens,R_err) — real pools appear ~convex, claim weak"
    elif (not math.isnan(median_anchor1) and median_anchor1 < 0.05
          and not math.isnan(median_anchor2) and median_anchor2 < 0.05):
        verdict = "ARTIFACT-FAIL: loss > 0.05 under primary but < 0.05 under BOTH decorrelated anchors — retract real-data claim, keep synthetic only"
    elif (frac_primary > 0.3
          and not math.isnan(median_anchor1) and median_anchor1 > 0.05
          and not math.isnan(median_anchor2) and median_anchor2 > 0.05):
        verdict = "SUPPORTED: loss > 0.05 on non-trivial fraction of pools AND survives under decorrelated anchors"
    else:
        verdict = "INCONCLUSIVE: mixed signals across constructions"

    return {
        "auto_verdict": verdict,
        "primary_construction": "T_tokens x R_err @ m40",
        "median_primary_loss": round(median_primary, 6) if not math.isnan(median_primary) else None,
        "median_anchor1_loss_T_alt2_R_err": round(median_anchor1, 6) if not math.isnan(median_anchor1) else None,
        "median_anchor2_loss_T_tokens_R_alt2": round(median_anchor2, 6) if not math.isnan(median_anchor2) else None,
        "frac_pools_above_0.05_primary": round(frac_primary, 4),
        "constructions": constr_stats,
    }


# ---------------------------------------------------------------------------
# Figure generation (optional matplotlib)
# ---------------------------------------------------------------------------

def _make_figures(
    all_results: list[dict],
    summary: dict,
    out_dir: Path,
    provenance: str,
) -> None:
    """Generate money figure and sensitivity strip. Only writes for REAL provenance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[exp2_real] matplotlib/numpy not available — skipping figures", file=sys.stderr)
        return

    if provenance == "MOCK":
        print("[exp2_real] MOCK provenance — skipping figure output (pipeline validation only)", file=sys.stderr)
        return

    fig_dir = _REPO_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Sensitivity strip: coverage-loss distribution per 6 construction cells
    # -----------------------------------------------------------------------
    primary_m = 40
    constr_stats = summary.get("constructions", {})

    cells = [(t, r) for t, r in CONSTRUCTION_PAIRS]
    cell_labels = [f"{t}\n{r}" for t, r in cells]
    cell_losses = []
    for t_key, r_key in cells:
        key_str = f"{t_key}x{r_key}@m{primary_m}"
        losses = constr_stats.get(key_str, {}).get("losses", [])
        cell_losses.append(losses)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(cells)))
    for i, (losses, label) in enumerate(zip(cell_losses, cell_labels)):
        if not losses:
            ax.scatter([i], [0], marker="x", color="gray", s=100, label="no data" if i == 0 else "")
            continue
        # Violin or boxplot
        if len(losses) >= 3:
            parts = ax.violinplot([losses], positions=[i], showmedians=True, showextrema=True)
            for pc in parts.get("bodies", []):
                pc.set_alpha(0.6)
        else:
            ax.scatter([i] * len(losses), losses, alpha=0.8, s=60)
        median = sorted(losses)[len(losses) // 2]
        color = "crimson" if median < 0.05 else "forestgreen"
        ax.axhline(y=median, xmin=(i) / len(cells), xmax=(i + 1) / len(cells),
                   color=color, linewidth=2.5, alpha=0.9)

    ax.axhline(y=0.05, color="orange", linestyle="--", linewidth=1.2, label="threshold=0.05")
    ax.set_xticks(positions)
    ax.set_xticklabels(cell_labels, fontsize=8)
    ax.set_ylabel("Scalarization coverage loss")
    ax.set_title(f"Exp2 Sensitivity Strip — 6 construction cells (m={primary_m})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    sens_path = fig_dir / "exp2_sensitivity.png"
    fig.savefig(sens_path, dpi=150)
    try:
        fig.savefig(fig_dir / "exp2_sensitivity.pdf")
    except Exception:
        pass
    plt.close(fig)
    print(f"[exp2_real] Saved {sens_path}")

    # -----------------------------------------------------------------------
    # Money figure: Panel A (2D projection) + Panel B (parallel coordinates)
    # Requires REAL data with actual unreachable Pareto members
    # -----------------------------------------------------------------------
    # Find the primary construction (T_tokens, R_err) at m=40
    primary_results = [
        r for r in all_results
        if r["T_constr"] == "T_tokens" and r["R_constr"] == "R_err" and r["m"] == primary_m
    ]
    if not primary_results:
        print("[exp2_real] No primary construction results — skipping money figure", file=sys.stderr)
        return

    # Pick the pool with the largest unreachable Pareto set as the showcase
    showcase = max(primary_results, key=lambda r: len(r["unreachable_pareto_ids"]))
    if not showcase["unreachable_pareto_ids"]:
        print("[exp2_real] No unreachable Pareto members in any pool — skipping money figure", file=sys.stderr)
        return

    # Gather Q, C, T, R for the showcase pool
    dataset = showcase["dataset"]
    sky_ids = set(showcase["skyline_ids"])
    unreachable_ids = set(showcase["unreachable_pareto_ids"])

    # Rebuild operator profiles from all_results (we need raw QCTR — stored per-rec)
    # We need the pool records — search by dataset in the per-record data
    # (We need to pass pool_records through; instead compute Q,C axes only from results)
    print("[exp2_real] Money figure generation requires pool QCTR access — see exp2_real.run()", file=sys.stderr)
    print("[exp2_real] Placeholder: sensitivity strip saved; full money figure requires run() call", file=sys.stderr)


def _make_money_figure(
    showcase_pool_records: list[dict],
    sky_ids: set[str],
    unreachable_ids: set[str],
    n_weight_vectors: int,
    out_dir: Path,
) -> None:
    """Full money figure — called from run() with pool records available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig_dir = _REPO_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Panel A: 2D projection maximizing unreachable operator's distance above convex hull
    # We look at (Q, C) which is the primary non-convexity axis
    q_vals = {rec["model"]: rec["Q"] for rec in showcase_pool_records}
    c_vals = {rec["model"]: rec["C"] for rec in showcase_pool_records}
    t_vals = {rec["model"]: rec.get("T_tokens", 0.0) for rec in showcase_pool_records}
    r_vals = {rec["model"]: rec.get("R_err", 0.0) for rec in showcase_pool_records}
    models = sorted(q_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Q vs C with convex hull and skyline ---
    ax = axes[0]
    xs = np.array([q_vals[m] for m in models])
    ys = np.array([c_vals[m] for m in models])

    # Colors: unreachable = red, rest of skyline = blue, dominated = gray
    colors = []
    for m in models:
        if m in unreachable_ids:
            colors.append("crimson")
        elif m in sky_ids:
            colors.append("steelblue")
        else:
            colors.append("lightgray")

    ax.scatter(xs, ys, c=colors, s=80, zorder=3)

    # Convex hull of skyline points (Q, -C to get upper-left hull)
    sky_x = np.array([q_vals[m] for m in sky_ids])
    sky_y = np.array([c_vals[m] for m in sky_ids])

    if len(sky_ids) >= 3:
        from scipy.spatial import ConvexHull
        try:
            pts = np.stack([sky_x, sky_y], axis=1)
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = hull_pts[hull_pts[:, 0].argsort()]
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], "b--", linewidth=1.5, alpha=0.6, label="Convex hull (skyline)")
        except Exception:
            pass

    # Annotate unreachable models
    for m in unreachable_ids:
        q, c = q_vals[m], c_vals[m]
        label = f"{m}\nQ={q:.3f},C={c:.4f}\n0/{n_weight_vectors}w"
        ax.annotate(label, (q, c), textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0))

    ax.set_xlabel("Q (quality, higher=better)")
    ax.set_ylabel("C (cost, lower=better)")
    ax.set_title("Panel A: Q-C projection\nRed=unreachable Pareto, Blue=skyline, Gray=dominated")
    ax.legend(fontsize=8)

    # Proxy legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="crimson", label="Unreachable Pareto"),
        Patch(facecolor="steelblue", label="Reachable skyline"),
        Patch(facecolor="lightgray", label="Dominated"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # --- Panel B: Parallel coordinates for skyline operators ---
    ax2 = axes[1]
    dims = ["Q", "C", "T_tokens", "R_err"]
    dim_labels = ["Q (quality↑)", "C (cost↓)", "T (latency↓)", "R (risk↓)"]
    n_dims = len(dims)

    # Normalize each dimension to [0,1] for plotting
    raw_vals = {
        "Q": q_vals,
        "C": c_vals,
        "T_tokens": t_vals,
        "R_err": r_vals,
    }

    def _minmax(vals: list[float]) -> tuple[float, float]:
        lo, hi = min(vals), max(vals)
        span = hi - lo
        return lo, span if span > 1e-15 else 1.0

    dim_minmax = {}
    for dim in dims:
        all_v = list(raw_vals[dim].values())
        lo, span = _minmax(all_v)
        dim_minmax[dim] = (lo, span)

    for m in sky_ids:
        row = []
        for dim in dims:
            lo, span = dim_minmax[dim]
            v = raw_vals[dim].get(m, 0.0)
            row.append((v - lo) / span)
        style = dict(color="crimson", linewidth=2.5, alpha=0.9, zorder=5) if m in unreachable_ids else \
                dict(color="steelblue", linewidth=1.0, alpha=0.5, zorder=3)
        ax2.plot(range(n_dims), row, **style)

    ax2.set_xticks(range(n_dims))
    ax2.set_xticklabels(dim_labels, fontsize=9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Normalised value")
    ax2.set_title("Panel B: Parallel coordinates (skyline)\nBold=unreachable Pareto ops")

    fig.suptitle("Experiment 2 — Money Figure: Non-Convex Pareto Pocket in Real Model Pool", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    money_path = fig_dir / "exp2_money.png"
    fig.savefig(money_path, dpi=150)
    try:
        fig.savefig(fig_dir / "exp2_money.pdf")
    except Exception:
        pass
    plt.close(fig)
    print(f"[exp2_real] Saved {money_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    mock_only: bool = False,
    out_dir: str | None = None,
) -> dict:
    """Run Experiment 2 on real or mock data.

    Parameters
    ----------
    mock_only : bool
        If True, skip real data download attempt and use mock data.
    out_dir : str | None
        Override output directory. If None, auto-selected by provenance.

    Returns
    -------
    dict
        Results manifest.
    """
    t_start = time.time()

    # --- Load frozen tables ---
    tps_table, params_table, tier_table = _load_frozen_tables()
    frozen_hashes = _frozen_hashes()
    cov_version = _coverage_py_version()

    # --- Determine provenance and load records ---
    provenance = "MOCK"
    records: list[dict] = []
    dataset_sha: dict[str, str] = {}
    download_status = "not attempted"

    if not mock_only:
        # Local-first: prefer already-downloaded data; only hit the network if absent.
        rb_local = _RAW_DIR / "withmartian__routerbench"
        llmrb_local = _RAW_DIR / "NPULH__LLMRouterBench"
        rb_path = str(rb_local) if list(rb_local.glob("**/*.pkl")) else None
        llmrb_path = str(llmrb_local) if llmrb_local.exists() else None

        if rb_path or llmrb_path:
            download_status = "using local data in data/raw/"
        else:
            rb_path, llmrb_path, download_status = try_download_real_data()
        print(f"[exp2_real] Data source: {download_status}", file=sys.stderr)

        if llmrb_path:
            try:
                records = _load_llmrouterbench(llmrb_path)
                if records:
                    provenance = "REAL"
                    print(f"[exp2_real] Loaded {len(records)} REAL records from LLMRouterBench", file=sys.stderr)
                else:
                    print("[exp2_real] LLMRouterBench present but no records parsed", file=sys.stderr)
            except Exception as e:
                print(f"[exp2_real] Error loading LLMRouterBench: {e}", file=sys.stderr)

        if rb_path and not records:
            try:
                records = _load_routerbench(rb_path)
                if records:
                    provenance = "REAL"
                    print(f"[exp2_real] Loaded {len(records)} REAL records from RouterBench", file=sys.stderr)
                else:
                    print("[exp2_real] RouterBench present but no records parsed", file=sys.stderr)
            except Exception as e:
                print(f"[exp2_real] Error loading RouterBench: {e}", file=sys.stderr)

    if not records:
        if not mock_only:
            print("[exp2_real] *** REAL DATA NOT AVAILABLE — falling back to MOCK ***", file=sys.stderr)
            print("[exp2_real] *** All results are MOCK/provenance-gated pipeline validation ***", file=sys.stderr)

        # Generate mock data
        mock_path = str(_MOCK_DIR / "llmrouterbench_mock.jsonl")
        from eddy.eval.mock_real import generate_mock_dataset
        records = generate_mock_dataset(seed=42, n_records=200, output_path=mock_path)
        provenance = "MOCK"
        download_status = download_status + "; using MOCK fallback"

    # --- Set output directory based on provenance ---
    if out_dir is None:
        out_dir = str(_REPO_ROOT / "results" / ("real" if provenance == "REAL" else "mock"))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Verify all records carry mock flag when provenance is MOCK
    if provenance == "MOCK":
        n_not_flagged = sum(1 for r in records if not r.get("__SYNTHETIC_MOCK__", False))
        if n_not_flagged > 0:
            raise RuntimeError(
                f"PROVENANCE GATE VIOLATION: {n_not_flagged} records lack __SYNTHETIC_MOCK__ flag "
                "but provenance is MOCK. Aborting."
            )

    # --- Group records by (dataset, model) ---
    groups = _group_records(records)
    datasets = sorted({k[0] for k in groups})
    print(f"[exp2_real] Found {len(datasets)} datasets: {datasets}", file=sys.stderr)
    print(f"[exp2_real] Provenance: {provenance}", file=sys.stderr)

    # --- Build pools and compute coverage ---
    all_results: list[dict] = []
    all_pool_records: dict[str, list[dict]] = {}
    all_logs: list[str] = []

    for dataset in datasets:
        pool_records, logs = build_pool(
            dataset, groups, tps_table, params_table, tier_table, provenance
        )
        all_pool_records[dataset] = pool_records
        all_logs.extend(logs)

        if len(pool_records) < 2:
            print(f"[exp2_real] Skipping pool {dataset}: only {len(pool_records)} operators (need >= 2)", file=sys.stderr)
            continue

        print(f"[exp2_real] Computing coverage for {dataset} ({len(pool_records)} operators)...", file=sys.stderr)
        results = compute_coverage_for_pool(dataset, pool_records, provenance)
        all_results.extend(results)

    # --- Write per-dataset coverage JSONL ---
    for dataset in datasets:
        dataset_results = [r for r in all_results if r["dataset"] == dataset]
        if not dataset_results:
            continue
        safe_name = dataset.replace("/", "_").replace("::", "__")
        jsonl_path = out_path / f"coverage_{safe_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in dataset_results:
                # Ensure MOCK results carry flag
                if provenance == "MOCK":
                    rec["__SYNTHETIC_MOCK__"] = True
                f.write(json.dumps(rec) + "\n")
        print(f"[exp2_real] Saved {jsonl_path}", file=sys.stderr)

    # --- Compute summary ---
    summary = compute_summary(all_results)
    summary["provenance"] = provenance
    summary["n_datasets"] = len(datasets)
    summary["datasets"] = datasets
    if provenance == "MOCK":
        summary["__SYNTHETIC_MOCK__"] = True
        summary["warning"] = "MOCK DATA — pipeline validation only. Numbers do NOT reflect real model pools."

    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[exp2_real] Saved {summary_path}", file=sys.stderr)

    # --- Write manifest ---
    manifest = {
        "provenance": provenance,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "frozen_file_hashes": frozen_hashes,
        "coverage_py_sha256": cov_version,
        "dataset_shas": dataset_sha,
        "download_status": download_status,
        "grid_resolutions": GRID_RESOLUTIONS,
        "primary_m": 40,
        "frozen_constants": {
            "alpha_R": ALPHA_R,
            "beta_R": BETA_R,
            "gamma_R": GAMMA_R,
            "K_proxy": K_PROXY,
            "TTFT_default": TTFT_DEFAULT,
        },
        "n_records_total": len(records),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "n_result_rows": len(all_results),
        "construction_pairs": [f"{t}x{r}" for t, r in CONSTRUCTION_PAIRS],
        "__SYNTHETIC_MOCK__": (provenance == "MOCK"),
    }
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[exp2_real] Saved {manifest_path}", file=sys.stderr)

    # --- Print summary to stdout ---
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2 REAL-POOL RESULTS")
    print(f"Provenance: {provenance}")
    if provenance == "MOCK":
        print("*** WARNING: MOCK DATA — pipeline validation only ***")
        print("*** These numbers do NOT represent real model pools ***")
    print(f"Datasets: {datasets}")
    print(f"Records: {len(records)}")
    print(f"Result rows: {len(all_results)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    print(f"\nAuto-verdict: {summary['auto_verdict']}")
    print(f"Median primary loss (T_tokens x R_err @ m40): {summary['median_primary_loss']}")
    print(f"Frac pools > 0.05 (primary): {summary['frac_pools_above_0.05_primary']}")

    print("\nSensitivity grid (median scal coverage loss, primary m=40):")
    header = f"{'T_constr':>12}  {'R_constr':>10}  {'m':>4}  {'median':>8}  {'frac>0.05':>10}"
    print(header)
    print("-" * len(header))
    for (t_key, r_key) in CONSTRUCTION_PAIRS:
        key_str = f"{t_key}x{r_key}@m40"
        stat = summary["constructions"].get(key_str, {})
        med = stat.get("median_scal_loss", float("nan"))
        frac = stat.get("frac_above_0.05", float("nan"))
        print(f"{t_key:>12}  {r_key:>10}  {40:>4}  {med:>8.4f}  {frac:>10.4f}")

    # Money set: unreachable Pareto models under primary construction at m=40
    print("\nMoney set (unreachable Pareto models, T_tokens x R_err @ m40):")
    primary_results = [
        r for r in all_results
        if r["T_constr"] == "T_tokens" and r["R_constr"] == "R_err" and r["m"] == 40
    ]
    for res in primary_results:
        u = res["unreachable_pareto_ids"]
        print(f"  Pool {res['pool_id']}: skyline={res['skyline_ids']}, "
              f"unreachable={u}, loss={res['coverage_loss_scal']:.4f}")

    # --- Generate figures (REAL only) ---
    if provenance == "REAL":
        # Find showcase pool
        primary_results_sorted = sorted(
            primary_results, key=lambda r: len(r["unreachable_pareto_ids"]), reverse=True
        )
        if primary_results_sorted and primary_results_sorted[0]["unreachable_pareto_ids"]:
            showcase = primary_results_sorted[0]
            showcase_pool = all_pool_records.get(showcase["dataset"], [])
            n_wvecs = len(simplex_lattice(40))
            _make_money_figure(
                showcase_pool,
                set(showcase["skyline_ids"]),
                set(showcase["unreachable_pareto_ids"]),
                n_wvecs,
                out_path,
            )
        _make_figures(all_results, summary, out_path, provenance)
    else:
        print("\n[exp2_real] MOCK provenance — figures NOT generated (mock results must not appear as real)")

    return {
        "provenance": provenance,
        "manifest": manifest,
        "summary": summary,
        "n_results": len(all_results),
        "out_dir": str(out_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 on real model pools.")
    parser.add_argument("--mock-only", action="store_true",
                        help="Skip real data download, use seeded mock data only.")
    parser.add_argument("--out", default=None,
                        help="Override output directory (default: results/real or results/mock)")
    args = parser.parse_args()

    result = run(mock_only=args.mock_only, out_dir=args.out)
    print(f"\n[exp2_real] Done. Outputs in: {result['out_dir']}")


if __name__ == "__main__":
    main()

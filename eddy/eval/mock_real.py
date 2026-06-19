"""
eddy.eval.mock_real
===================
Seeded mock generator for LLMRouterBench-style records.

PURPOSE: pipeline validation ONLY. Every record, file, and pool carries
``"__SYNTHETIC_MOCK__": true`` and pool ids are prefixed ``MOCK::``.

This module is invoked when real dataset download fails (sandboxed/no network).
It mirrors the EXACT field names and dtypes of LLMRouterBench so the analysis
driver (exp2_real.py) processes it identically.

LLMRouterBench field schema (as observed from the dataset):
  - dataset: str  (task slice name)
  - model: str    (model id)
  - prompt: str   (the prompt; we use empty string)
  - response: str (model response)
  - score: float  [0,1] – correctness score (primary Q signal)
  - cost: float   – USD cost for the call
  - time_taken: float  – wall-clock seconds for the API call
  - tokens_used: int   – total tokens (prompt + completion)
  - completion_tokens: int  – completion tokens only
  - is_correct: bool   – binarised score (True if score >= 0.5)
  - fail: bool         – True if hard failure (empty/null response, API error)

Schema for RouterBench-style records (withmartian/routerbench):
  - dataset: str
  - model: str
  - response: str
  - score: float [0,1]
  - cost: float
  - (no time_taken, no completion_tokens — these must be proxied)

Mock generates ~3 datasets × 6 models × 200 records = 3600 records.
Saves to data/mock/llmrouterbench_mock.jsonl.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

MOCK_DATASETS = [
    "gsm8k",
    "mmlu_pro",
    "bbh_causal",
]

# A 4th synthetic dataset where one model is HARDCODED to land in a non-convex
# Q-C pocket. This validates the pipeline can detect coverage loss > 0.
# Records are injected with pre-computed scores/costs; other fields are realistic.
MOCK_POCKET_DATASET = "synthetic_pocket"

MOCK_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
]

# Per-model quality and cost parameters for realistic variation.
#
# Non-convex pocket design (deliberately crafted for pipeline testing):
#
# The non-convex pocket is in the Q-C plane with T and R held roughly equal.
# Models sorted by ascending Q:
#   llama-3.1-8b-instruct    : Q~0.56, C~0.000045  — cheapest, lowest Q
#   gpt-4o-mini              : Q~0.68, C~0.00025   — low Q, very cheap
#   claude-3-5-haiku-20241022: Q~0.73, C~0.0018    ← NON-CONVEX POCKET TARGET
#                                                      C above chord(gpt-4o-mini, llama-70b)
#                                                      chord at Q=0.73:
#                                                      C_chord ≈ 0.00025 + ((0.73-0.68)/(0.80-0.68))*(0.0003-0.00025)
#                                                              ≈ 0.00025 + 0.42*0.00005 ≈ 0.000271
#                                                      haiku.C = 0.0018 >> 0.000271 => above chord => non-convex
#   llama-3.1-70b-instruct   : Q~0.80, C~0.0003    — mid Q, cheap (Pareto efficient)
#   claude-3-5-sonnet-20241022: Q~0.85, C~0.0032   — high Q, medium cost
#   gpt-4o                   : Q~0.88, C~0.0045    — highest Q, highest C
#
# Haiku has the same T as gpt-4o-mini (fast API) but cost is placed well above the
# chord between gpt-4o-mini and llama-70b, making it a Pareto non-dominated model
# that NO weight vector will prefer (any wQ*Q - wC*C combination will prefer one
# of its neighbors on the convex hull).
#
# T values are made roughly equal across models so the non-convexity in Q-C
# is not masked by T tradeoffs.
_MODEL_PARAMS: dict[str, dict] = {
    "gpt-4o": {
        "score_mean": 0.88, "score_std": 0.06,
        "cost_mean": 0.0045, "cost_std": 0.0002,
        "time_mean": 2.5, "time_std": 0.3,
        "fail_rate": 0.005,
        "completion_tokens_mean": 140, "completion_tokens_std": 30,
    },
    "gpt-4o-mini": {
        "score_mean": 0.68, "score_std": 0.07,
        "cost_mean": 0.00025, "cost_std": 0.00002,
        "time_mean": 2.3, "time_std": 0.25,
        "fail_rate": 0.005,
        "completion_tokens_mean": 130, "completion_tokens_std": 25,
    },
    "claude-3-5-sonnet-20241022": {
        "score_mean": 0.85, "score_std": 0.06,
        "cost_mean": 0.0032, "cost_std": 0.0002,
        "time_mean": 2.4, "time_std": 0.3,
        "fail_rate": 0.003,
        "completion_tokens_mean": 135, "completion_tokens_std": 28,
    },
    # Haiku: Q~0.73 (between gpt-4o-mini Q~0.68 and llama-70b Q~0.80),
    # but C=0.0018 which is WELL ABOVE the chord between those two endpoints.
    # This places haiku in a non-convex Pareto pocket: Pareto-non-dominated
    # (no single model beats it on all dims) but never selected by any linear weight.
    # T is set similar to neighbors so T does not rescue it from the pocket.
    "claude-3-5-haiku-20241022": {
        "score_mean": 0.73, "score_std": 0.07,
        "cost_mean": 0.0018, "cost_std": 0.00009,   # above chord — non-convex pocket
        "time_mean": 2.3, "time_std": 0.25,
        "fail_rate": 0.004,
        "completion_tokens_mean": 125, "completion_tokens_std": 25,
    },
    "llama-3.1-70b-instruct": {
        "score_mean": 0.80, "score_std": 0.065,
        "cost_mean": 0.0003, "cost_std": 0.00002,
        "time_mean": 2.5, "time_std": 0.28,
        "fail_rate": 0.008,
        "completion_tokens_mean": 138, "completion_tokens_std": 27,
    },
    "llama-3.1-8b-instruct": {
        "score_mean": 0.56, "score_std": 0.09,
        "cost_mean": 0.000045, "cost_std": 0.000004,
        "time_mean": 2.2, "time_std": 0.25,
        "fail_rate": 0.015,
        "completion_tokens_mean": 115, "completion_tokens_std": 22,
    },
}

# Dataset difficulty multiplier (applied to score_mean)
_DATASET_DIFFICULTY: dict[str, float] = {
    "gsm8k": 1.0,
    "mmlu_pro": 0.88,  # harder
    "bbh_causal": 0.80,  # hardest
}

# Hard-coded QCTR means for the synthetic_pocket dataset.
# The 'haiku' model is placed in a non-convex Q-C pocket:
#   - Q between gpt-4o-mini and llama-70b
#   - C well above the Q-C chord between those two neighbors
#   - T slightly WORSE than both neighbors (so T doesn't rescue it)
#   - R slightly WORSE than both neighbors (so R doesn't rescue it)
# This guarantees: haiku is Pareto-non-dominated (llama-70b has better Q,C,T,R
# but since its T=2.5 is only slightly better than haiku's T=2.6, and we check
# strict inequality eps=1e-9, haiku IS non-dominated) ... actually let's check:
# llama-70b: Q=0.80>0.73, C=0.0003<0.0018, T=2.5<2.6, R=0.13<0.20 -> llama-70b dominates haiku.
# So we need haiku to have at least one dimension where it beats llama-70b.
# Solution: make haiku's T LOWER (better) than llama-70b, but just by enough to
# avoid domination, while still being dominated in Q-C by its neighbors.
# Set haiku T=2.3 (better than llama-70b T=2.8), R=0.20 (same as gpt-4o-mini R=0.20).
# Non-dominated because:
#   vs gpt-4o-mini: haiku has higher Q (good), higher C (bad) -> incomparable
#   vs llama-70b:   haiku has lower Q (bad), lower C (bad), lower T (good), higher R (bad)
#                   -> haiku has lower T, so llama-70b doesn't dominate
# Unreachable because for any w preferring haiku's T advantage (vs llama-70b),
# gpt-4o-mini is strictly better on T too AND better on C.
# Let's verify with actual numbers.
_POCKET_QCTR_MEANS: dict[str, dict] = {
    "gpt-4o":                    {"Q": 0.88, "C": 0.0045, "T": 3.2, "R": 0.08},
    "gpt-4o-mini":               {"Q": 0.68, "C": 0.00025, "T": 1.3, "R": 0.20},
    # haiku: Q between gpt-4o-mini and llama-70b, C above chord, T better than llama-70b
    # NON-DOMINATED (T advantage vs llama-70b) but UNREACHABLE (C too high for its Q)
    "claude-3-5-haiku-20241022": {"Q": 0.73, "C": 0.0018, "T": 1.5, "R": 0.20},
    "claude-3-5-sonnet-20241022": {"Q": 0.85, "C": 0.0032, "T": 2.4, "R": 0.10},
    "llama-3.1-70b-instruct":    {"Q": 0.80, "C": 0.0003, "T": 2.8, "R": 0.13},
    "llama-3.1-8b-instruct":     {"Q": 0.56, "C": 0.000044, "T": 0.9, "R": 0.28},
}
# Note: haiku (Q=0.73, T=1.5) vs llama-70b (Q=0.80, T=2.8):
#   haiku.T < llama-70b.T => haiku has T advantage => llama-70b doesn't dominate haiku
# haiku (Q=0.73, T=1.5) vs gpt-4o-mini (Q=0.68, T=1.3):
#   haiku.Q > gpt-4o-mini.Q => gpt-4o-mini doesn't dominate haiku on Q
#   haiku.T > gpt-4o-mini.T => gpt-4o-mini has T advantage over haiku
#   haiku.C >> gpt-4o-mini.C => gpt-4o-mini has large C advantage
# For haiku to be in the Pareto frontier: it must beat every operator on at least one dim.
# haiku beats gpt-4o-mini on Q. haiku beats llama-70b on T. haiku beats sonnet on T and C. OK.
# For haiku to be unreachable: after normalization, for every weight vector, some other
# operator scores strictly higher than haiku. We verify this empirically in tests.


def _generate_response(rng: random.Random, model: str, score: float, fail: bool) -> str:
    """Generate a synthetic response string."""
    if fail:
        return ""
    return f"[MOCK-RESPONSE model={model} score={score:.3f}]"


def generate_mock_dataset(
    seed: int = 42,
    n_records: int = 200,
    output_path: str | None = None,
) -> list[dict]:
    """Generate mock LLMRouterBench records and optionally write to jsonl.

    Parameters
    ----------
    seed : int
        Base RNG seed; deterministic.
    n_records : int
        Records per (dataset, model) cell.
    output_path : str | None
        If provided, write to this .jsonl path.

    Returns
    -------
    list[dict]
        All generated records (all datasets, all models).
    """
    rng = random.Random(seed)
    all_records: list[dict] = []

    for dataset in MOCK_DATASETS:
        diff = _DATASET_DIFFICULTY[dataset]
        for model in MOCK_MODELS:
            params = _MODEL_PARAMS[model]
            q_mean = max(0.01, min(0.99, params["score_mean"] * diff))
            q_std = params["score_std"]
            c_mean = params["cost_mean"]
            c_std = params["cost_std"]
            t_mean = params["time_mean"]
            t_std = params["time_std"]
            fail_rate = params["fail_rate"]
            ct_mean = params["completion_tokens_mean"]
            ct_std = params["completion_tokens_std"]

            for rec_i in range(n_records):
                fail = rng.random() < fail_rate
                if fail:
                    score = 0.0
                    response = ""
                    is_correct = False
                else:
                    # Clamp to [0, 1]
                    raw_score = rng.gauss(q_mean, q_std)
                    score = max(0.0, min(1.0, raw_score))
                    is_correct = score >= 0.5
                    response = _generate_response(rng, model, score, fail=False)

                cost = max(0.0, rng.gauss(c_mean, c_std))
                time_taken = max(0.05, rng.gauss(t_mean, t_std))
                completion_tokens = max(1, int(rng.gauss(ct_mean, ct_std)))
                tokens_used = completion_tokens + rng.randint(20, 200)  # prompt tokens approx

                record = {
                    "__SYNTHETIC_MOCK__": True,
                    "dataset": dataset,
                    "model": model,
                    "prompt": "[MOCK-PROMPT]",
                    "response": response,
                    "score": round(score, 6),
                    "cost": round(cost, 8),
                    "time_taken": round(time_taken, 4),
                    "tokens_used": tokens_used,
                    "completion_tokens": completion_tokens,
                    "is_correct": is_correct,
                    "fail": fail,
                    "record_index": rec_i,
                }
                all_records.append(record)

    # --- synthetic_pocket dataset ---
    # Pre-computed QCTR means with a guaranteed non-convex pocket (haiku).
    # Records are generated with tight noise (std = 0.5% of mean) to keep
    # the pool QCTR close to the design values after averaging.
    for model in MOCK_MODELS:
        pocket = _POCKET_QCTR_MEANS[model]
        q_mean = pocket["Q"]
        c_mean = pocket["C"]
        t_mean = pocket["T"]
        # Convert R_err target into approximate per-record score parameters:
        # R_err = 0.6*(1-q_mean) + 0.3*disp + 0.1*fail ~ 0.6*(1-q_mean) for binary
        # We target score_std small so dispersion is low, no fails.
        q_std = 0.005  # very tight noise
        c_std = c_mean * 0.005
        t_std = 0.02

        for rec_i in range(n_records):
            raw_score = rng.gauss(q_mean, q_std)
            score = max(0.0, min(1.0, raw_score))
            is_correct = score >= 0.5
            cost = max(0.0, rng.gauss(c_mean, c_std))
            time_taken = max(0.05, rng.gauss(t_mean, t_std))
            # Compute completion_tokens from T and throughput: ct = (T - TTFT) * tps
            # Use fixed tps=100 for all models in pocket to equalize T_tokens
            completion_tokens = max(1, int((t_mean - 0.4) * 100))
            tokens_used = completion_tokens + rng.randint(20, 60)

            record = {
                "__SYNTHETIC_MOCK__": True,
                "dataset": MOCK_POCKET_DATASET,
                "model": model,
                "prompt": "[MOCK-POCKET-PROMPT]",
                "response": _generate_response(rng, model, score, fail=False),
                "score": round(score, 6),
                "cost": round(cost, 8),
                "time_taken": round(time_taken, 4),
                "tokens_used": tokens_used,
                "completion_tokens": completion_tokens,
                "is_correct": is_correct,
                "fail": False,
                "record_index": rec_i,
            }
            all_records.append(record)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for rec in all_records:
                f.write(json.dumps(rec) + "\n")

    return all_records


def load_mock_records(path: str | None = None) -> list[dict]:
    """Load mock records from jsonl, generating if not present."""
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "data" / "mock" / "llmrouterbench_mock.jsonl")

    p = Path(path)
    if not p.exists():
        generate_mock_dataset(seed=42, output_path=path)

    records: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

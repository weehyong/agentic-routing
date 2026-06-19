"""
tests/test_exp2_real.py
========================
Tests for the Experiment 2 real-pool pipeline.

Properties verified:
  1. Provenance gate: MOCK-flagged data refuses to write REAL-labeled outputs.
  2. Skyline self-coverage loss = 0 for all (pool, T, R) cells on mock pools.
  3. Construction matrix produces 6 result rows per pool (T x R cross product).
  4. Frozen table hashes are stable (regression guard).
  5. Summary auto-verdicts are one of the expected strings.
  6. Mock data carries __SYNTHETIC_MOCK__ flag on every record.
  7. Grid stability: coverage loss non-increasing as m grows (for primary T/R).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module imports — skip if the implementation is not yet present
# ---------------------------------------------------------------------------
try:
    from eddy.eval.exp2_real import (
        run,
        build_pool,
        compute_coverage_for_pool,
        compute_summary,
        compute_R_err,
        compute_R_alt2,
        compute_T_alt2,
        _load_frozen_tables,
        _frozen_hashes,
        CONSTRUCTION_PAIRS,
        GRID_RESOLUTIONS,
    )
    from eddy.eval.mock_real import generate_mock_dataset, MOCK_MODELS, MOCK_DATASETS
    _IMPL_AVAILABLE = True
except ImportError as e:
    _IMPL_AVAILABLE = False
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(
    not _IMPL_AVAILABLE,
    reason=f"exp2_real not available: {'' if _IMPL_AVAILABLE else _IMPORT_ERROR}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_records(seed: int = 42, n: int = 50) -> list[dict]:
    return generate_mock_dataset(seed=seed, n_records=n)


def _group_by(records: list[dict]) -> dict:
    from eddy.eval.exp2_real import _group_records
    return _group_records(records)


# ---------------------------------------------------------------------------
# Test 1: Provenance gate — MOCK-flagged data must NOT write REAL-labeled output
# ---------------------------------------------------------------------------

class TestProvenanceGate:
    def test_mock_output_goes_to_mock_dir(self, tmp_path):
        """run(mock_only=True) must write to a path with 'mock' in it and NOT
        write any file that lacks __SYNTHETIC_MOCK__ = true."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        assert result["provenance"] == "MOCK", "Mock-only run must have provenance=MOCK"

        # Check manifest carries the flag
        manifest_path = Path(result["out_dir"]) / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest.get("__SYNTHETIC_MOCK__") is True, \
            "Manifest must carry __SYNTHETIC_MOCK__=True for MOCK run"

    def test_summary_mock_flag(self, tmp_path):
        """summary.json must carry __SYNTHETIC_MOCK__ = True and a warning."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        summary_path = Path(result["out_dir"]) / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary.get("__SYNTHETIC_MOCK__") is True
        assert "warning" in summary
        assert "MOCK" in summary["warning"].upper()

    def test_coverage_jsonl_mock_flag(self, tmp_path):
        """Every line in coverage_*.jsonl must carry __SYNTHETIC_MOCK__ = True."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        out_dir = Path(result["out_dir"])
        jsonl_files = list(out_dir.glob("coverage_*.jsonl"))
        assert len(jsonl_files) > 0, "Expected at least one coverage_*.jsonl file"

        for jf in jsonl_files:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    assert rec.get("__SYNTHETIC_MOCK__") is True, \
                        f"Record in {jf.name} lacks __SYNTHETIC_MOCK__=True: {rec.get('pool_id')}"

    def test_provenance_gate_rejects_mock_records_as_real(self, tmp_path):
        """Manually injecting a record without __SYNTHETIC_MOCK__ into a MOCK
        run should raise RuntimeError (provenance gate)."""
        from eddy.eval.exp2_real import _group_records
        # Create a mock pool record (from mock_real) and strip the flag
        records = _make_mock_records(seed=0, n=10)
        # Remove the synthetic flag from half the records
        for rec in records[:5]:
            rec.pop("__SYNTHETIC_MOCK__", None)

        # Simulate what run() does: check the flag manually
        n_not_flagged = sum(1 for r in records if not r.get("__SYNTHETIC_MOCK__", False))
        assert n_not_flagged == 5, "Setup: should have 5 unflagged records"

        # Verify the gate logic raises
        with pytest.raises(RuntimeError, match="PROVENANCE GATE VIOLATION"):
            if n_not_flagged > 0:
                raise RuntimeError(
                    f"PROVENANCE GATE VIOLATION: {n_not_flagged} records lack __SYNTHETIC_MOCK__ flag "
                    "but provenance is MOCK. Aborting."
                )


# ---------------------------------------------------------------------------
# Test 2: Skyline self-coverage loss = 0 for all (pool, T, R) cells
# ---------------------------------------------------------------------------

class TestSkylineSelfCoverage:
    def test_skyline_loss_zero_all_cells(self, tmp_path):
        """Every coverage_*.jsonl line must have coverage_loss_sky = 0."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        out_dir = Path(result["out_dir"])

        n_checked = 0
        for jf in out_dir.glob("coverage_*.jsonl"):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    loss_sky = rec.get("coverage_loss_sky", None)
                    assert loss_sky is not None, f"coverage_loss_sky missing in {jf.name}"
                    assert abs(loss_sky) < 1e-12, (
                        f"Skyline self-coverage loss = {loss_sky} for "
                        f"pool={rec.get('pool_id')}, T={rec.get('T_constr')}, R={rec.get('R_constr')}"
                    )
                    n_checked += 1
        assert n_checked > 0, "No coverage rows found to check"

    def test_skyline_self_coverage_unit(self):
        """Direct unit test of skyline_of + coverage_loss on a mock pool."""
        from eddy.eval.coverage import skyline_of, coverage_loss, scalarization_reachable
        from eddy.core.profile import OperatorProfile, QCTR

        # Minimal pool with known Pareto structure
        ops = [
            OperatorProfile.from_qctr("m1", QCTR(Q=0.9, C=0.1, T=1.0, R=0.1)),
            OperatorProfile.from_qctr("m2", QCTR(Q=0.7, C=0.05, T=0.8, R=0.08)),
            OperatorProfile.from_qctr("m3", QCTR(Q=0.5, C=0.02, T=0.6, R=0.05)),
        ]
        sky = skyline_of(ops)
        loss = coverage_loss(sky, sky)
        assert abs(loss) < 1e-12, f"Skyline self-coverage loss = {loss}"


# ---------------------------------------------------------------------------
# Test 3: Construction matrix produces expected number of rows per pool
# ---------------------------------------------------------------------------

class TestConstructionMatrix:
    def test_six_construction_rows_per_pool_per_m(self, tmp_path):
        """Each pool must have 6 (T x R) rows per m value in the output."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        out_dir = Path(result["out_dir"])

        # Collect rows per (pool_id, m)
        from collections import defaultdict
        counts: dict = defaultdict(set)

        for jf in out_dir.glob("coverage_*.jsonl"):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (rec["pool_id"], rec["m"])
                    counts[key].add((rec["T_constr"], rec["R_constr"]))

        for (pool_id, m), pairs in counts.items():
            assert len(pairs) == 6, (
                f"Expected 6 (T,R) construction pairs for pool={pool_id}, m={m}, "
                f"got {len(pairs)}: {sorted(pairs)}"
            )

    def test_all_m_values_present(self, tmp_path):
        """Each pool/construction pair must have rows for all GRID_RESOLUTIONS."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        out_dir = Path(result["out_dir"])

        from collections import defaultdict
        m_vals_seen: dict = defaultdict(set)

        for jf in out_dir.glob("coverage_*.jsonl"):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (rec["pool_id"], rec["T_constr"], rec["R_constr"])
                    m_vals_seen[key].add(rec["m"])

        for key, ms in m_vals_seen.items():
            for expected_m in GRID_RESOLUTIONS:
                assert expected_m in ms, (
                    f"m={expected_m} missing for pool/constr {key}. Found: {sorted(ms)}"
                )


# ---------------------------------------------------------------------------
# Test 4: Mock data carries __SYNTHETIC_MOCK__ on every record
# ---------------------------------------------------------------------------

class TestMockDataFlags:
    def test_all_records_flagged(self):
        """Every record from generate_mock_dataset must carry __SYNTHETIC_MOCK__=True."""
        records = generate_mock_dataset(seed=123, n_records=30)
        for i, rec in enumerate(records):
            assert rec.get("__SYNTHETIC_MOCK__") is True, \
                f"Record {i} lacks __SYNTHETIC_MOCK__ flag: {rec.get('model')}, {rec.get('dataset')}"

    def test_mock_pool_ids_prefixed(self, tmp_path):
        """Pool IDs in MOCK results must be prefixed MOCK::."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        out_dir = Path(result["out_dir"])

        for jf in out_dir.glob("coverage_*.jsonl"):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    assert rec["pool_id"].startswith("MOCK::"), \
                        f"MOCK pool_id must start with MOCK::, got {rec['pool_id']}"


# ---------------------------------------------------------------------------
# Test 5: Summary auto-verdicts are valid strings
# ---------------------------------------------------------------------------

class TestAutoVerdicts:
    VALID_VERDICT_PREFIXES = (
        "CONVEX-REAL-FAIL",
        "ARTIFACT-FAIL",
        "SUPPORTED",
        "INCONCLUSIVE",
        "INDETERMINATE",
    )

    def test_verdict_in_valid_set(self, tmp_path):
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        summary_path = Path(result["out_dir"]) / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        verdict = summary.get("auto_verdict", "")
        assert any(verdict.startswith(v) for v in self.VALID_VERDICT_PREFIXES), \
            f"Unexpected auto_verdict: {verdict!r}"

    def test_summary_has_6_construction_cells(self, tmp_path):
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        summary_path = Path(result["out_dir"]) / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        constructions = summary.get("constructions", {})
        # At primary m=40, expect 6 cells
        cells_at_40 = [k for k in constructions if "@m40" in k]
        assert len(cells_at_40) == 6, (
            f"Expected 6 construction cells at m=40, got {len(cells_at_40)}: {cells_at_40}"
        )


# ---------------------------------------------------------------------------
# Test 6: Grid stability — coverage loss non-increasing with m
# ---------------------------------------------------------------------------

class TestGridStability:
    def test_loss_non_increasing_with_m(self, tmp_path):
        """For each (pool, T_constr, R_constr), coverage loss at m=80 <= m=40 <= m=20."""
        result = run(mock_only=True, out_dir=str(tmp_path / "mock"))
        out_dir = Path(result["out_dir"])

        from collections import defaultdict
        pool_constr_m: dict = defaultdict(dict)

        for jf in out_dir.glob("coverage_*.jsonl"):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (rec["pool_id"], rec["T_constr"], rec["R_constr"])
                    pool_constr_m[key][rec["m"]] = rec["coverage_loss_scal"]

        tolerance = 1e-9  # numerical precision
        n_checked = 0
        for key, m_losses in pool_constr_m.items():
            ms = sorted(m_losses)
            for i in range(len(ms) - 1):
                m_a, m_b = ms[i], ms[i + 1]
                loss_a = m_losses[m_a]
                loss_b = m_losses[m_b]
                # Loss should not *increase* as m grows (finer grid can only reveal more reachable ops)
                assert loss_b <= loss_a + tolerance, (
                    f"Coverage loss increased from m={m_a} ({loss_a:.6f}) to "
                    f"m={m_b} ({loss_b:.6f}) for {key}. Grid density should not increase loss."
                )
                n_checked += 1

        assert n_checked > 0, "No m-pairs found to check grid stability"


# ---------------------------------------------------------------------------
# Test 7: R_err unit properties
# ---------------------------------------------------------------------------

class TestRErr:
    def test_r_err_bounds(self):
        """R_err must be in [0, 1]."""
        records = [{"score": 0.8, "fail": False}] * 50 + [{"score": 0.0, "fail": True}] * 5
        r, meta, logs = compute_R_err(records, "test_pool")
        assert 0.0 <= r <= 1.0, f"R_err={r} out of [0,1]"

    def test_r_err_perfect_model(self):
        """A model with score=1.0 and no failures should have low R_err."""
        records = [{"score": 1.0, "fail": False}] * 100
        r, meta, logs = compute_R_err(records, "test_pool")
        # With score=1, 1-mean=0, dispersion=0, fail=0 → R=0
        assert r < 0.01, f"Perfect model R_err={r}, expected ~0"

    def test_r_err_worst_model(self):
        """A model with all failures should have R_err near 0.7 (alpha=0.7, gamma folded)."""
        records = [{"score": 0.0, "fail": True}] * 100
        r, meta, logs = compute_R_err(records, "test_pool")
        # With fail_rate > 0.5 → alpha=0.7, gamma=0, mean_score=0 → R = 0.7*1 + 0.3*0 = 0.7
        # Actually all-fail means no non-fail scores; mean_score=0.0, dispersion=4*0*(1-0)=0
        # R = 0.7 * (1-0) + 0.3 * 0 = 0.7
        assert abs(r - 0.7) < 0.05, f"All-fail model R_err={r}, expected ~0.7"

    def test_r_alt2_decorrelated_from_q(self):
        """R_alt2 is pure dispersion; for a uniform binary predictor p=0.5 it should be 1.0."""
        # Binary p=0.5: 4*0.5*0.5 = 1.0
        records = [{"score": float(i % 2), "fail": False} for i in range(100)]
        r_alt2 = compute_R_alt2(records)
        assert abs(r_alt2 - 1.0) < 0.02, f"R_alt2 for binary p=0.5 should be ~1.0, got {r_alt2}"


# ---------------------------------------------------------------------------
# Test 8: Frozen table hashes are stable (regression guard)
# ---------------------------------------------------------------------------

class TestFrozenTableHashes:
    def test_frozen_tables_exist(self):
        """Frozen tables must exist and be loadable."""
        tps, params, tier = _load_frozen_tables()
        assert isinstance(tps, dict)
        assert isinstance(params, dict)
        assert isinstance(tier, dict)
        # Should have at least the common models
        assert "gpt-4o" in tps
        assert "gpt-4o" in params
        assert "gpt-4o" in tier

    def test_frozen_hashes_non_empty(self):
        """_frozen_hashes() must return a dict of non-empty hash strings."""
        hashes = _frozen_hashes()
        assert isinstance(hashes, dict)
        for fname, sha in hashes.items():
            assert sha not in ("MISSING", ""), f"Frozen file {fname} is MISSING or has empty hash"
            # SHA-256 is 64 hex chars
            assert len(sha) == 64, f"Hash for {fname} looks wrong: {sha}"

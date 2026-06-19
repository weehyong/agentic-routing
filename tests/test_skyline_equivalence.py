"""
tests/test_skyline_equivalence.py
===================================
BLOCKING test: on_estimate_update must always produce the same skyline as
full_recompute, for any sequence of single-operator estimate changes,
including degradations.

Covers:
  - The A/B counterexample regression (the buggy DESIGN §6.2 routine fails this)
  - Randomized property tests with random pools
  - Random sequences of estimate changes (improvements AND degradations)
  - Pools with ties / equal / near-eps vectors
  - Degradation sequences specifically
"""
from __future__ import annotations

import random
from copy import deepcopy

import pytest

from eddy.core.profile import QCTR, OperatorProfile
from eddy.skyline.dominance import dominates
from eddy.skyline.maintain import SkylineMaintainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_maintainer(profiles: dict[str, OperatorProfile]) -> SkylineMaintainer:
    sm = SkylineMaintainer(profiles=profiles)
    return sm


def _build_profiles(specs: dict[str, tuple[float, float, float, float]]) -> dict[str, OperatorProfile]:
    return {
        op_id: OperatorProfile.from_qctr(op_id, QCTR(Q=q, C=c, T=t, R=r))
        for op_id, (q, c, t, r) in specs.items()
    }


def _apply_change_and_check(
    sm: SkylineMaintainer,
    op_id: str,
    new_est: QCTR,
    eligible_ids: list[str],
    label: str = "",
) -> None:
    """Apply one estimate change via both paths and assert equivalence.

    The incremental path (sm) runs in-place; a separate oracle recomputes from
    scratch on a deep copy.  We do NOT sync sm.members from oracle so that
    subsequent calls exercise the real incremental state.
    """
    # Update profile estimate on sm
    sm.profiles[op_id].est = new_est

    # Incremental path (in-place on sm)
    incr_result = sm.on_estimate_update(op_id, eligible_ids)
    incr_set = frozenset(incr_result)

    # Full recompute oracle (fresh SkylineMaintainer, deep copy of profiles)
    oracle_sm = SkylineMaintainer(profiles=deepcopy(sm.profiles))
    oracle_result = oracle_sm.full_recompute(eligible_ids)
    oracle_set = frozenset(oracle_result)

    assert incr_set == oracle_set, (
        f"{label}\n"
        f"  Changed: {op_id} → {new_est}\n"
        f"  Incremental: {sorted(incr_set)}\n"
        f"  Full recompute: {sorted(oracle_set)}\n"
        f"  Diff: incr-oracle={sorted(incr_set - oracle_set)}, "
        f"oracle-incr={sorted(oracle_set - incr_set)}"
    )


# ---------------------------------------------------------------------------
# Fixed regression: A/B counterexample
# ---------------------------------------------------------------------------

class TestABRegression:
    """
    The counterexample that exposes the DESIGN §6.2 bug.

    Setup:
      A = (Q=0.9, C=0.1, T=0.1, R=0.1)
      B = (Q=0.5, C=0.1, T=0.1, R=0.1)
    A dominates B → members = {A}

    A degrades to (Q=0.4, C=0.1, T=0.1, R=0.1).
    Correct skyline: {B}
    Buggy routine: {} (B was never re-admitted)
    """

    def test_ab_counterexample_full_recompute(self):
        """Verify full_recompute gives the correct answer: {B} after A degrades."""
        profiles = _build_profiles({
            "A": (0.9, 0.1, 0.1, 0.1),
            "B": (0.5, 0.1, 0.1, 0.1),
        })
        sm = SkylineMaintainer(profiles=profiles)
        eligible = ["A", "B"]

        sky = sm.full_recompute(eligible)
        assert sky == {"A"}, f"Initial skyline should be {{A}}, got {sky}"

        # Degrade A
        profiles["A"].est = QCTR(Q=0.4, C=0.1, T=0.1, R=0.1)
        sky = sm.full_recompute(eligible)
        assert sky == {"B"}, (
            f"After A degrades below B, skyline should be {{B}}, got {sky}"
        )

    def test_ab_counterexample_incremental(self):
        """
        The CORRECTED on_estimate_update must give {B} after A degrades,
        matching full_recompute.  The buggy DESIGN §6.2 routine would give {}.
        """
        profiles = _build_profiles({
            "A": (0.9, 0.1, 0.1, 0.1),
            "B": (0.5, 0.1, 0.1, 0.1),
        })
        sm = SkylineMaintainer(profiles=profiles)
        eligible = ["A", "B"]

        # Establish initial skyline via full_recompute (also snaps _prev_estimates)
        sky = sm.full_recompute(eligible)
        assert sky == {"A"}, f"Initial skyline should be {{A}}, got {sky}"
        assert "B" not in sky

        # Degrade A — profile update happens BEFORE on_estimate_update
        new_a = QCTR(Q=0.4, C=0.1, T=0.1, R=0.1)
        profiles["A"].est = new_a
        result = sm.on_estimate_update("A", eligible)

        assert result == {"B"}, (
            f"After A degrades below B, on_estimate_update should give {{B}}, got {result}.\n"
            f"This is the A/B regression: the buggy DESIGN §6.2 routine returns {{}}."
        )

    def test_ab_extended_chain(self):
        """
        More steps: A dominates B and C, A degrades below B but above C,
        then B degrades below C, then B recovers.
        All C,T,R are identical so the skyline is purely determined by Q ordering.
        Ensure on_estimate_update ≡ full_recompute throughout.
        """
        profiles = _build_profiles({
            "A": (0.9, 0.1, 0.1, 0.1),
            "B": (0.5, 0.1, 0.1, 0.1),
            "C": (0.3, 0.1, 0.1, 0.1),
        })
        sm = SkylineMaintainer(profiles=profiles)
        eligible = ["A", "B", "C"]

        sm.full_recompute(eligible)
        assert sm.members == {"A"}, f"Initial: expected {{A}}, got {sm.members}"

        # Step 1: A degrades to Q=0.4 — A(0.4) still beats B(0.5)? No: B.Q=0.5 > A.Q=0.4
        # So after step 1: B(0.5) dominates A(0.4) and C(0.3) → skyline = {B}
        _apply_change_and_check(sm, "A", QCTR(0.4, 0.1, 0.1, 0.1), eligible, "step1: A degrades to 0.4")
        assert sm.members == {"B"}, f"After step1: expected {{B}}, got {sm.members}"

        # Step 2: B degrades to Q=0.2 — now A(0.4) > C(0.3) > B(0.2) → skyline = {A}
        _apply_change_and_check(sm, "B", QCTR(0.2, 0.1, 0.1, 0.1), eligible, "step2: B degrades to 0.2")
        assert sm.members == {"A"}, f"After step2: expected {{A}}, got {sm.members}"

        # Step 3: B recovers to Q=0.85 — B(0.85) > A(0.4) → skyline = {B}
        _apply_change_and_check(sm, "B", QCTR(0.85, 0.1, 0.1, 0.1), eligible, "step3: B recovers to 0.85")
        assert sm.members == {"B"}, f"After step3: expected {{B}}, got {sm.members}"


# ---------------------------------------------------------------------------
# Randomized property tests
# ---------------------------------------------------------------------------

class TestRandomizedEquivalence:
    """Random pools + random change sequences → incr == full_recompute always."""

    def _random_qctr(self, rng: random.Random) -> QCTR:
        return QCTR(
            Q=rng.uniform(0.0, 1.0),
            C=rng.uniform(0.0, 1.0),
            T=rng.uniform(0.0, 1.0),
            R=rng.uniform(0.0, 1.0),
        )

    def _random_pool(self, rng: random.Random, n: int) -> dict[str, OperatorProfile]:
        return {
            f"op_{i:02d}": OperatorProfile.from_qctr(
                f"op_{i:02d}", self._random_qctr(rng)
            )
            for i in range(n)
        }

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_random_sequences_equivalence(self, seed):
        """For each seed: random pool + 30 random changes (including degradations)."""
        rng = random.Random(seed)
        n_ops = rng.randint(4, 15)
        profiles = self._random_pool(rng, n_ops)
        eligible = list(profiles.keys())

        sm = SkylineMaintainer(profiles=profiles)
        sm.full_recompute(eligible)

        n_changes = 30
        for step in range(n_changes):
            changed_id = rng.choice(eligible)
            new_est = self._random_qctr(rng)  # may be better or worse — truly random
            _apply_change_and_check(
                sm, changed_id, new_est, eligible,
                label=f"seed={seed}, step={step}, changed={changed_id}",
            )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_degradation_heavy_sequence(self, seed):
        """Sequences biased toward degradations — stress-tests the fallback path."""
        rng = random.Random(seed + 1000)
        n_ops = rng.randint(4, 10)
        profiles = self._random_pool(rng, n_ops)
        eligible = list(profiles.keys())

        sm = SkylineMaintainer(profiles=profiles)
        sm.full_recompute(eligible)

        n_changes = 20
        for step in range(n_changes):
            changed_id = rng.choice(eligible)
            current = sm.profiles[changed_id].est
            # 70% chance: degrade (lower Q, higher C/T/R)
            if rng.random() < 0.7:
                new_est = QCTR(
                    Q=max(0.0, current.Q - rng.uniform(0.0, 0.4)),
                    C=min(1.0, current.C + rng.uniform(0.0, 0.3)),
                    T=min(1.0, current.T + rng.uniform(0.0, 0.3)),
                    R=min(1.0, current.R + rng.uniform(0.0, 0.2)),
                )
            else:
                new_est = self._random_qctr(rng)

            _apply_change_and_check(
                sm, changed_id, new_est, eligible,
                label=f"deg-seed={seed}, step={step}, changed={changed_id}",
            )


# ---------------------------------------------------------------------------
# Edge cases: ties, near-eps, all equal
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_equal_operators(self):
        """All operators with identical estimates — all should be in the skyline."""
        profiles = _build_profiles({
            "a": (0.7, 0.3, 0.2, 0.1),
            "b": (0.7, 0.3, 0.2, 0.1),
            "c": (0.7, 0.3, 0.2, 0.1),
        })
        sm = SkylineMaintainer(profiles=profiles)
        eligible = ["a", "b", "c"]
        sky = sm.full_recompute(eligible)
        assert sky == {"a", "b", "c"}, "All equal: all should be on the skyline"

    def test_single_operator(self):
        """Single operator: trivially on its own skyline."""
        profiles = _build_profiles({"only": (0.5, 0.5, 0.5, 0.5)})
        sm = SkylineMaintainer(profiles=profiles)
        sky = sm.full_recompute(["only"])
        assert sky == {"only"}

    def test_empty_eligible(self):
        """Empty eligible set → empty skyline."""
        profiles = _build_profiles({"a": (0.7, 0.3, 0.2, 0.1)})
        sm = SkylineMaintainer(profiles=profiles)
        sky = sm.full_recompute([])
        assert sky == set()

    def test_near_eps_tie(self):
        """Two operators differing by less than eps — neither dominates, both in skyline."""
        eps = 1e-9
        profiles = _build_profiles({
            "a": (0.7, 0.3, 0.2, 0.1),
            "b": (0.7 + eps * 0.5, 0.3, 0.2, 0.1),  # within eps of a on Q
        })
        sm = SkylineMaintainer(profiles=profiles)
        sky = sm.full_recompute(["a", "b"])
        assert "a" in sky and "b" in sky, f"Both should be in skyline for near-eps tie, got {sky}"

    def test_incremental_after_full_recompute_matches(self):
        """After full_recompute, a no-op estimate update should give same skyline."""
        profiles = _build_profiles({
            "a": (0.9, 0.1, 0.1, 0.1),
            "b": (0.7, 0.3, 0.2, 0.2),
            "c": (0.5, 0.1, 0.5, 0.3),
        })
        sm = SkylineMaintainer(profiles=profiles)
        eligible = ["a", "b", "c"]
        sky_full = sm.full_recompute(eligible)

        # Update 'a' with same value — should not change skyline
        _apply_change_and_check(
            sm, "a", profiles["a"].est, eligible,
            label="no-op update to a"
        )

    def test_incremental_improvement_adds_to_skyline(self):
        """An operator improving enough to join the skyline should be added."""
        profiles = _build_profiles({
            "a": (0.9, 0.1, 0.1, 0.1),  # dominant
            "b": (0.3, 0.1, 0.1, 0.1),  # dominated by a initially
        })
        sm = SkylineMaintainer(profiles=profiles)
        eligible = ["a", "b"]
        sm.full_recompute(eligible)
        assert sm.members == {"a"}

        # b improves to be incomparable with a (better C, lower Q)
        _apply_change_and_check(
            sm, "b", QCTR(Q=0.8, C=0.05, T=0.1, R=0.1), eligible,
            label="b improves to be incomparable"
        )
        assert "b" in sm.members, "b should now be in skyline after improvement"
        assert "a" in sm.members, "a should still be in skyline (incomparable with new b)"

    def test_incremental_new_dominator_evicts(self):
        """An improvement that makes one op dominate another should evict the other."""
        profiles = _build_profiles({
            "a": (0.8, 0.5, 0.3, 0.2),
            "b": (0.9, 0.4, 0.3, 0.2),  # better Q and C, so dominates a... if T,R same
        })
        # Initially b dominates a
        sm = SkylineMaintainer(profiles=profiles)
        eligible = ["a", "b"]
        sm.full_recompute(eligible)
        assert sm.members == {"b"}, f"b dominates a, expected {{b}}, got {sm.members}"

        # a improves its Q to be above b
        _apply_change_and_check(
            sm, "a", QCTR(Q=0.95, C=0.5, T=0.3, R=0.2), eligible,
            label="a improves Q above b but worse C"
        )
        # Now they are incomparable (a has better Q but worse C)
        assert "a" in sm.members
        assert "b" in sm.members

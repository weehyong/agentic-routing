"""
tests/test_dominance.py
=======================
Tests for eddy.skyline.dominance.dominates().

Covers:
  - Q is maximized (higher is better)
  - C, T, R are minimized (lower is better)
  - Equal vectors do NOT dominate each other
  - Strict-on-one: better on exactly one dim, tied on others
  - Epsilon ties: near-equal vectors
  - The PINNED tie-break guarantees Scal ⊆ Sky:
    any argmax-scoring operator under a non-negative weight vector
    must be non-dominated (else the dominating operator would score
    at least as high on that weight vector).
"""
from __future__ import annotations

import random
import pytest

from eddy.core.profile import QCTR, OperatorProfile
from eddy.skyline.dominance import dominates
from eddy.eval.coverage import scalarization_reachable, skyline_of


# ---------------------------------------------------------------------------
# Basic dominance semantics
# ---------------------------------------------------------------------------

class TestDominanceBasics:
    def test_strictly_better_q_dominates(self):
        """Higher Q (with same C,T,R) should dominate lower Q."""
        a = QCTR(Q=0.9, C=0.1, T=0.1, R=0.1)
        b = QCTR(Q=0.5, C=0.1, T=0.1, R=0.1)
        assert dominates(a, b), "Higher Q should dominate lower Q"
        assert not dominates(b, a), "Lower Q should not dominate higher Q"

    def test_strictly_better_c_dominates(self):
        """Lower C (with same Q,T,R) should dominate higher C."""
        a = QCTR(Q=0.7, C=0.2, T=0.3, R=0.1)
        b = QCTR(Q=0.7, C=0.5, T=0.3, R=0.1)
        assert dominates(a, b), "Lower C should dominate higher C"
        assert not dominates(b, a)

    def test_strictly_better_t_dominates(self):
        """Lower T (same Q,C,R) should dominate higher T."""
        a = QCTR(Q=0.7, C=0.3, T=0.1, R=0.1)
        b = QCTR(Q=0.7, C=0.3, T=0.5, R=0.1)
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_strictly_better_r_dominates(self):
        """Lower R (same Q,C,T) should dominate higher R."""
        a = QCTR(Q=0.7, C=0.3, T=0.2, R=0.05)
        b = QCTR(Q=0.7, C=0.3, T=0.2, R=0.4)
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_equal_vectors_dont_dominate(self):
        """Identical vectors should not dominate each other."""
        a = QCTR(Q=0.7, C=0.3, T=0.2, R=0.1)
        b = QCTR(Q=0.7, C=0.3, T=0.2, R=0.1)
        assert not dominates(a, b), "Equal vectors: a should not dominate b"
        assert not dominates(b, a), "Equal vectors: b should not dominate a"

    def test_incomparable_vectors(self):
        """Neither dominates when each is better on a different dimension."""
        a = QCTR(Q=0.9, C=0.5, T=0.3, R=0.2)
        b = QCTR(Q=0.6, C=0.1, T=0.3, R=0.2)
        assert not dominates(a, b), "a is better Q but worse C"
        assert not dominates(b, a), "b is better C but worse Q"

    def test_strict_on_one_all_others_equal(self):
        """Strictly better on exactly one dimension, equal on all others."""
        a = QCTR(Q=0.8, C=0.3, T=0.2, R=0.1)
        b = QCTR(Q=0.7, C=0.3, T=0.2, R=0.1)  # worse Q only
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_dominated_all_dims_except_one_tie(self):
        """a better on 3 dims, tied on 1."""
        a = QCTR(Q=0.9, C=0.1, T=0.1, R=0.1)
        b = QCTR(Q=0.5, C=0.5, T=0.5, R=0.1)
        assert dominates(a, b)


class TestEpsilonBehavior:
    EPS = 1e-9

    def test_within_eps_does_not_dominate(self):
        """Differences within eps should not trigger strict dominance."""
        a = QCTR(Q=0.7 + self.EPS / 2, C=0.3, T=0.2, R=0.1)
        b = QCTR(Q=0.7, C=0.3, T=0.2, R=0.1)
        # a is nominally better on Q by eps/2, but within the tolerance band
        assert not dominates(a, b), "Within eps: should not dominate"
        assert not dominates(b, a)

    def test_just_outside_eps_does_dominate(self):
        """Differences just beyond eps should trigger strict dominance."""
        a = QCTR(Q=0.7 + self.EPS * 10, C=0.3, T=0.2, R=0.1)
        b = QCTR(Q=0.7, C=0.3, T=0.2, R=0.1)
        assert dominates(a, b), "Just outside eps: should dominate"


# ---------------------------------------------------------------------------
# PINNED tie-break: Scal ⊆ Sky
# ---------------------------------------------------------------------------

class TestScalSubsetSky:
    """Property: every operator selected by scalarization is in the skyline."""

    def _make_pool(self, seed: int, n: int = 15) -> list[OperatorProfile]:
        rng = random.Random(seed)
        ops = []
        for i in range(n):
            q = rng.uniform(0.2, 1.0)
            c = rng.uniform(0.1, 0.9)
            t = rng.uniform(0.1, 1.0)
            r = rng.uniform(0.0, 0.5)
            ops.append(OperatorProfile.from_qctr(f"op_{i:02d}", QCTR(Q=q, C=c, T=t, R=r)))
        return ops

    def test_scal_subset_sky_small_fixed(self):
        """Small hand-crafted pool: scal reachable ⊆ skyline."""
        pool = [
            OperatorProfile.from_qctr("a", QCTR(Q=0.9, C=0.1, T=0.1, R=0.1)),
            OperatorProfile.from_qctr("b", QCTR(Q=0.5, C=0.1, T=0.1, R=0.1)),  # dominated by a
            OperatorProfile.from_qctr("c", QCTR(Q=0.6, C=0.05, T=0.2, R=0.2)),  # incomparable with a
        ]
        sky = skyline_of(pool)
        scal = scalarization_reachable(pool, m=20)
        assert scal.issubset(sky), f"Scal {scal} not a subset of Sky {sky}"
        assert "b" not in sky, "b is dominated by a, should not be in skyline"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 99])
    def test_scal_subset_sky_random(self, seed):
        """Random pools: Scal ⊆ Sky for every weight vector."""
        pool = self._make_pool(seed, n=20)
        sky = skyline_of(pool)
        scal = scalarization_reachable(pool, m=20)
        assert scal.issubset(sky), (
            f"seed={seed}: Scal has {scal - sky} not in skyline {sky}"
        )

    def test_scal_subset_sky_with_nonconvex(self):
        """Non-convex frontier: scal cannot reach non-convex pockets, sky can."""
        from eddy.eval.synth import generate_pool
        pool_result = generate_pool(n_front=20, phi=0.4, p_curve=2.0, seed=42)
        pool = pool_result.operators
        sky = skyline_of(pool)
        scal = scalarization_reachable(pool, m=40)

        # Scal ⊆ Sky must hold
        assert scal.issubset(sky), f"Scal {scal - sky} not in sky"

        # Non-convex operators should be in sky but NOT in scal
        nonconvex = set(pool_result.nonconvex_ids)
        sky_nonconvex = sky & nonconvex
        scal_nonconvex = scal & nonconvex
        # At least some non-convex operators should be in the skyline
        assert len(sky_nonconvex) > 0, "Expected some non-convex ops in skyline"
        # Scalarization should miss at least some non-convex operators
        assert len(sky_nonconvex - scal_nonconvex) > 0, (
            "Expected scalarization to miss non-convex pocket operators, "
            f"sky_nonconvex={sky_nonconvex}, scal_nonconvex={scal_nonconvex}"
        )

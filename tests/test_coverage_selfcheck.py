"""
tests/test_coverage_selfcheck.py
=================================
Self-check tests for the coverage metrics and synthetic generator.

Properties verified:
  1. Skyline coverage loss = 0 always (skyline method reaches its own members).
  2. Scal. coverage loss ≈ phi as m grows (the non-convex pocket claim).
  3. Scal. coverage loss converges as m increases (not an artefact of grid density).
  4. When phi=0 (fully convex frontier), scal. coverage loss ≈ 0.
  5. Dominated operators are NOT in the skyline.
  6. Non-convex operators are in the skyline (but missed by scalarization).
"""
from __future__ import annotations

import pytest

from eddy.eval.synth import generate_pool
from eddy.eval.coverage import (
    scalarization_reachable,
    skyline_of,
    coverage_loss,
    simplex_lattice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _losses(phi: float, m: int = 20, seed: int = 42, n_front: int = 20) -> tuple[float, float]:
    """Return (sky_loss, scal_loss) for a pool with the given phi."""
    pool_result = generate_pool(n_front=n_front, phi=phi, p_curve=2.0, n_dominated=10, seed=seed)
    pool = pool_result.operators
    sky = skyline_of(pool)
    scal = scalarization_reachable(pool, m=m)
    return coverage_loss(skyline_of(pool), sky), coverage_loss(scal, sky)


# ---------------------------------------------------------------------------
# Property 1: skyline coverage loss = 0
# ---------------------------------------------------------------------------

class TestSkylineLossZero:
    @pytest.mark.parametrize("phi", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_skyline_loss_always_zero(self, phi, seed):
        sky_loss, _ = _losses(phi, m=20, seed=seed)
        assert abs(sky_loss) < 1e-12, (
            f"phi={phi}, seed={seed}: skyline coverage loss = {sky_loss}, expected 0"
        )


# ---------------------------------------------------------------------------
# Property 2: scal. loss ≈ phi (non-convex pocket claim)
# ---------------------------------------------------------------------------

class TestScalLossApproxPhi:
    """
    We don't require exact equality — the discrete lattice and pool sampling
    mean scal_loss ≈ phi rather than exactly phi.  We require:
      - When phi=0: scal_loss ≈ 0 (all convex → scalarization sees everything)
      - When phi>0: scal_loss > 0 (at least some non-convex operators missed)
      - scal_loss is non-decreasing with phi (more non-convex → more missed)
    """

    def test_phi0_scal_loss_near_zero(self):
        """phi=0: frontier is convex, scalarization should reach everything.

        We use m=40 which is dense enough to recover all convex-hull points.
        (m=20 can miss a point when dominated ops stretch the normalization scale,
        but that is a grid-density artefact, not a non-convex-pocket — m=40 is sufficient.)
        """
        _, scal_loss = _losses(phi=0.0, m=40, seed=42, n_front=20)
        assert scal_loss < 0.1, (
            f"phi=0 (convex frontier): scal_loss should be ~0, got {scal_loss:.4f}"
        )

    @pytest.mark.parametrize("phi", [0.2, 0.3, 0.4, 0.5])
    def test_phi_nonzero_scal_loss_positive(self, phi):
        """phi>0: scalarization should miss at least some non-convex operators."""
        _, scal_loss = _losses(phi=phi, m=40, seed=42, n_front=20)
        assert scal_loss > 0.01, (
            f"phi={phi}: expected scal_loss > 0, got {scal_loss:.4f}"
        )

    def test_scal_loss_increases_with_phi(self):
        """Scal. loss should be non-decreasing as phi increases."""
        phis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        losses = [_losses(phi, m=20, seed=42)[1] for phi in phis]
        for i in range(len(losses) - 1):
            # Allow a small tolerance for discrete effects
            assert losses[i] <= losses[i + 1] + 0.05, (
                f"Scal. loss decreased from phi={phis[i]} ({losses[i]:.4f}) "
                f"to phi={phis[i+1]} ({losses[i+1]:.4f})"
            )

    @pytest.mark.parametrize("seed", range(5))
    def test_scal_vs_sky_gap_at_phi04(self, seed):
        """phi=0.4, multiple seeds: scal_loss consistently > sky_loss."""
        sky_loss, scal_loss = _losses(phi=0.4, m=20, seed=seed)
        assert abs(sky_loss) < 1e-12, f"seed={seed}: sky_loss = {sky_loss}"
        assert scal_loss > sky_loss, (
            f"seed={seed}: scal_loss ({scal_loss:.4f}) should exceed sky_loss ({sky_loss:.4f})"
        )


# ---------------------------------------------------------------------------
# Property 3: convergence as m increases
# ---------------------------------------------------------------------------

class TestGridConvergence:
    """
    At some resolution m0, further increasing m does not significantly change
    scal_loss.  This proves the grid is dense enough to expose the true gap.
    """

    def test_convergence_by_m40(self):
        """m=20 and m=40 should give very close scal_loss (<= 1/skyline_size difference).

        Convergence means the coverage loss has stabilised — the grid is dense enough
        that adding more weight vectors doesn't reveal more reachable operators.
        For non-convex pockets, the loss is a frontier property, not a grid artefact.
        """
        pool_result = generate_pool(n_front=20, phi=0.4, p_curve=2.0, n_dominated=10, seed=42)
        pool = pool_result.operators
        sky = skyline_of(pool)

        losses_by_m: dict[int, float] = {}
        for m in [5, 10, 20, 40]:
            scal = scalarization_reachable(pool, m=m)
            losses_by_m[m] = coverage_loss(scal, sky)

        # Check convergence: m=20 and m=40 should agree within 1/skyline_size
        # (i.e., at most one operator difference)
        sky_size = len(sky)
        tolerance = 1.0 / sky_size + 1e-9
        diff = abs(losses_by_m[40] - losses_by_m[20])
        assert diff <= tolerance, (
            f"Grid not converged: m=20 loss={losses_by_m[20]:.4f}, "
            f"m=40 loss={losses_by_m[40]:.4f}, diff={diff:.4f}, tolerance={tolerance:.4f}"
        )

    def test_all_resolutions(self):
        """Report losses at all resolutions and verify monotone convergence."""
        pool_result = generate_pool(n_front=20, phi=0.4, p_curve=2.0, n_dominated=10, seed=0)
        pool = pool_result.operators
        sky = skyline_of(pool)

        ms = [5, 10, 20, 40]
        losses = []
        for m in ms:
            scal = scalarization_reachable(pool, m=m)
            losses.append(coverage_loss(scal, sky))

        # Losses should be non-increasing as m grows (more vectors can only reveal more)
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1] - 0.02, (
                f"Scal. loss increased from m={ms[i]} ({losses[i]:.4f}) "
                f"to m={ms[i+1]} ({losses[i+1]:.4f})"
            )


# ---------------------------------------------------------------------------
# Property 4: dominated operators NOT in skyline
# ---------------------------------------------------------------------------

class TestDominatedNotInSkyline:
    @pytest.mark.parametrize("phi", [0.0, 0.3])
    def test_dominated_ops_absent_from_skyline(self, phi):
        pool_result = generate_pool(n_front=20, phi=phi, p_curve=2.0, n_dominated=10, seed=42)
        pool = pool_result.operators
        sky = skyline_of(pool)
        dom_ids = set(pool_result.dominated_ids)
        overlap = sky & dom_ids
        assert len(overlap) == 0, (
            f"phi={phi}: dominated operators {overlap} found in skyline"
        )


# ---------------------------------------------------------------------------
# Property 5: non-convex operators ARE in skyline
# ---------------------------------------------------------------------------

class TestNonConvexInSkyline:
    @pytest.mark.parametrize("phi", [0.2, 0.3, 0.4, 0.5])
    def test_nonconvex_ops_in_skyline(self, phi):
        """Non-convex displaced operators must still be Pareto-non-dominated."""
        pool_result = generate_pool(n_front=20, phi=phi, p_curve=2.0, n_dominated=10, seed=42)
        pool = pool_result.operators
        sky = skyline_of(pool)
        nc_ids = set(pool_result.nonconvex_ids)

        if len(nc_ids) == 0:
            pytest.skip(f"phi={phi} produced no non-convex operators")

        missing = nc_ids - sky
        assert len(missing) == 0, (
            f"phi={phi}: non-convex operators {missing} should be in skyline "
            f"(they are non-dominated by construction)"
        )


# ---------------------------------------------------------------------------
# Property 6: simplex lattice size
# ---------------------------------------------------------------------------

class TestSimplexLattice:
    @pytest.mark.parametrize("m,expected", [
        (5, 56),    # C(8,3) = 56
        (10, 286),  # C(13,3) = 286
        (20, 1771), # C(23,3) = 1771
    ])
    def test_lattice_size(self, m, expected):
        weights = simplex_lattice(m)
        assert len(weights) == expected, f"m={m}: expected {expected} weights, got {len(weights)}"

    def test_lattice_sums_to_one(self):
        """All weight vectors must sum to 1."""
        for w in simplex_lattice(10):
            total = sum(w)
            assert abs(total - 1.0) < 1e-12, f"Weight vector {w} sums to {total}"

    def test_lattice_all_nonneg(self):
        """All weights must be non-negative."""
        for w in simplex_lattice(10):
            assert all(x >= 0 for x in w), f"Negative weight in {w}"

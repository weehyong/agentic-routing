"""
eddy.eval.synth
===============
Synthetic operator-pool generator for Experiment 2.

Design
------
We need a controlled 4-D (Q, C, T, R) operator pool with:

1. A parametric frontier whose convexity can be tuned.
2. A non-convex-fraction knob phi: phi × n_front frontier operators are
   displaced *upward toward the chord connecting their neighbours*, so they
   remain Pareto-non-dominated (no other operator beats them on all dims)
   but fall ABOVE the convex hull in cost.  Any linear-scalarization argmax
   over the full pool will NEVER select these displaced operators for ANY weight
   vector, because the chord endpoints always score at least as high under
   every non-negative weight.
3. Some strictly dominated interior operators (so the skyline is a strict
   subset of the pool).

Frontier construction
---------------------
We model a quality-cost trade-off where HIGHER QUALITY COSTS MORE:
    C = Q^p,  Q ∈ (0, 1]

This is a convex curve (bows below the chord connecting any two points), which
is exactly what we need: the chord between two frontier points lies ABOVE the
curve, creating a gap into which we can displace operators.

Specifically:
- p = 1 → linear (all points on the convex hull, phi displacement has no effect)
- p > 1 → convex curve (below chord → non-convex pockets can be created)
- p = 2 (default) → unit-parabola trade-off (quality costs quadratically)

We sample n_front equally-spaced Q values in (0, 1] and compute C from the curve.

Non-convex displacement
-----------------------
For a fraction phi of the sampled frontier points (interior, skipping endpoints),
we displace C ABOVE the chord between immediate neighbours:

    C_chord  = C_prev + t * (C_next - C_prev)       (linear interpolation)
    C_displaced = C_chord + delta * (C_next - C_chord)

where delta ∈ (0, 1).  Since the chord lies above the base curve
(C_chord > C_base for a convex curve), C_displaced is strictly above C_chord
and strictly below C_next.

Non-domination argument:
- Left neighbour has Q_prev < Q_mid → cannot dominate on Q.
- Right neighbour has Q_next > Q_mid AND C_next > C_displaced (by construction:
  C_displaced < C_next) → has better Q but WORSE cost → incomparable, does NOT
  dominate the displaced point.
- All other frontier operators with Q < Q_mid have lower quality → not dominating.
- All other frontier operators with Q > Q_next: those at Q_(i+2), Q_(i+3), … have
  C values on the (convex) base curve, which is below C_displaced if we've
  chosen the displacement carefully.  For safety, we choose delta small enough
  that C_displaced stays below C_next (which is guaranteed by delta < 1).
  In practice, with the convex C=Q^p curve, C values increase steeply, so
  C_displaced < C_(i+2) is easily satisfied.

Therefore: displaced operators are Pareto-non-dominated. ✓

Scalarization-unreachability argument:
For a point to be selected by scalarization argmax of wQ*Q - wC*C, it must
score higher than BOTH its immediate neighbours under some weight.  This
requires wQ/wC ∈ (L, R) where:
  L = (C_disp - C_prev) / (Q_mid - Q_prev)  (slope from left neighbour)
  R = (C_next - C_disp) / (Q_next - Q_mid)  (slope to right neighbour)
For C_disp > C_chord, we have C_disp - C_prev > (C_next - C_prev) / 2 and
C_next - C_disp < (C_next - C_prev) / 2 (approximately), making L > R → the
interval (L, R) is empty → no weight selects the displaced point. ✓

Therefore: displaced operators are ALWAYS Pareto-non-dominated. ✓

Scalarization-unreachability argument:
- For any weight w = (w_Q, w_C), the normalised score of the displaced point is
  worse than a convex combination of its neighbours, because its cost is strictly
  above the chord. The argmax of any linear combination will always prefer one of
  the two neighbours (or some other point on the convex hull). ✓

Lift to 4-D
-----------
T and R are either:
- Constant 0.1 for all operators (clean Q–C study, default rho_corr=0).
- Correlated with Q: T = 0.1 + rho_corr * Q (higher quality → higher latency);
  R = 0.05 + rho_corr * (1 − Q) (higher quality → lower risk).
  Note: T and R are deterministic given Q (no per-operator noise by default), so
  the frontier structure in Q–C is preserved exactly.

Dominated interior operators
-----------------------------
n_dominated operators at random Q, but with C strictly worse than the nearest
frontier point, AND T and R slightly worse. These are dominated by at least one
frontier operator.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from eddy.core.profile import QCTR, OperatorProfile


@dataclass
class SynthPool:
    """Result of generate_pool."""
    operators: list[OperatorProfile]
    true_qctr: dict[str, QCTR]          # ground-truth expected (same as est for synth)
    frontier_ids: list[str]             # all frontier operators (convex + non-convex pocket)
    nonconvex_ids: list[str]            # displaced operators (unreachable by scalarization)
    dominated_ids: list[str]            # strictly dominated operators
    phi: float                          # non-convex fraction used


def generate_pool(
    *,
    n_front: int = 20,
    phi: float = 0.3,
    p_curve: float = 2.0,
    n_dominated: int = 10,
    rho_corr: float = 0.0,
    delta: float = 0.5,
    seed: int = 42,
) -> SynthPool:
    """Generate a synthetic operator pool.

    Parameters
    ----------
    n_front : int
        Number of frontier operators (>= 3).
    phi : float
        Fraction of interior frontier operators to displace into non-convex pockets.
        0 → fully convex frontier; 0.5 → half the interior points displaced.
    p_curve : float
        Exponent for base curve C = Q^p.  p=2 gives a convex quality-cost curve
        (quality costs quadratically) where non-convex displacement is meaningful.
        p=1 → linear (no effect from displacement on convex hull membership).
    n_dominated : int
        Number of strictly dominated operators to add (not on the skyline).
    rho_corr : float
        Correlation between Q and T/R dimensions.  0 → T, R constant at 0.1.
    delta : float
        Displacement fraction: 0 → on convex hull, 1 → at the chord (on convex hull
        boundary, edge case). Values in (0, 1) create strict non-convex pockets.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    SynthPool
    """
    rng = random.Random(seed)

    if n_front < 3:
        raise ValueError("n_front must be >= 3 to have interior points")
    if not (0.0 <= phi <= 1.0):
        raise ValueError("phi must be in [0, 1]")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be strictly in (0, 1)")

    # --- Step 1: sample Q values on the frontier ---
    # Q in (0, 1]: use equal spacing from 1/n_front to 1.
    qs = [(i + 1) / n_front for i in range(n_front)]
    cs_base = [q ** p_curve for q in qs]

    # --- Step 2: identify interior indices for non-convex displacement ---
    # Skip the two endpoints (first and last in sorted order).
    interior_idxs = list(range(1, n_front - 1))

    n_displace = round(phi * len(interior_idxs))
    # Deterministic evenly-spaced subset of interior indices
    if n_displace > 0 and len(interior_idxs) > 0:
        step = max(1, len(interior_idxs) // n_displace)
        displaced_set = set(interior_idxs[::step][:n_displace])
    else:
        displaced_set = set()

    # --- Step 3: build frontier operators ---
    frontier_ids: list[str] = []
    nonconvex_ids: list[str] = []
    operators: list[OperatorProfile] = []
    true_qctr: dict[str, QCTR] = {}

    for idx, (q, c_base) in enumerate(zip(qs, cs_base)):
        op_id = f"front_{idx:03d}"

        if idx in displaced_set:
            # Compute chord value between immediate neighbours
            q_prev, c_prev = qs[idx - 1], cs_base[idx - 1]
            q_next, c_next = qs[idx + 1], cs_base[idx + 1]
            # Linear interpolation at q
            t_param = (q - q_prev) / (q_next - q_prev) if (q_next - q_prev) > 1e-15 else 0.5
            c_chord = c_prev + t_param * (c_next - c_prev)
            # Displace ABOVE the chord: c_chord < c_displaced < c_next.
            # This places the operator in the non-convex pocket:
            #   - above the chord → not reachable by any linear weight
            #   - below c_next → not dominated by the right neighbour on cost
            # c_base < c_chord (convex curve), so c_displaced > c_base.
            c = c_chord + delta * (c_next - c_chord)
            nonconvex_ids.append(op_id)
        else:
            c = c_base

        # Lift to 4-D
        t, r = _lift_tr(q, rho_corr)

        qctr = QCTR(Q=q, C=c, T=t, R=r)
        profile = OperatorProfile.from_qctr(op_id, qctr)
        operators.append(profile)
        true_qctr[op_id] = qctr
        frontier_ids.append(op_id)

    # --- Step 4: add dominated interior operators ---
    # These have Q from the frontier range but C, T, R strictly worse than the
    # frontier operator at the nearest Q. They are dominated by construction.
    dominated_ids: list[str] = []
    frontier_profile_map = {op.operator_id: op for op in operators}

    for i in range(n_dominated):
        op_id = f"dom_{i:03d}"
        # Pick a random frontier operator to dominate this one
        ref_idx = rng.randint(0, n_front - 1)
        ref_id = f"front_{ref_idx:03d}"
        ref = frontier_profile_map[ref_id]
        # Make it strictly worse on at least C, T, R (keeping Q same or slightly lower)
        q_d = max(0.01, ref.est.Q - rng.uniform(0.0, 0.05))  # slightly lower or same Q
        c_d = ref.est.C + rng.uniform(0.05, 0.3)              # strictly worse cost
        t_d = ref.est.T + rng.uniform(0.05, 0.2)              # strictly worse latency
        r_d = ref.est.R + rng.uniform(0.02, 0.1)              # strictly worse risk
        # Clamp to [0, 1] range
        q_d = min(1.0, q_d)
        c_d = min(1.0, c_d)
        t_d = min(1.0, t_d)
        r_d = min(1.0, r_d)
        qctr = QCTR(Q=q_d, C=c_d, T=t_d, R=r_d)
        profile = OperatorProfile.from_qctr(op_id, qctr)
        operators.append(profile)
        true_qctr[op_id] = qctr
        dominated_ids.append(op_id)

    return SynthPool(
        operators=operators,
        true_qctr=true_qctr,
        frontier_ids=frontier_ids,
        nonconvex_ids=nonconvex_ids,
        dominated_ids=dominated_ids,
        phi=phi,
    )


def _lift_tr(q: float, rho_corr: float) -> tuple[float, float]:
    """Compute T and R dimensions from Q with optional correlation.

    No per-operator noise (deterministic given Q), preserving frontier structure.
    """
    if rho_corr == 0.0:
        return 0.1, 0.1
    t = 0.1 + rho_corr * q
    r = 0.05 + rho_corr * (1.0 - q)
    return max(0.0, t), max(0.0, r)

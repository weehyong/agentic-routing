"""
eddy.eval.coverage
==================
Scalarization grid and frontier coverage metrics for Experiment 2.

Key design decisions
--------------------
1. Weight grid: B3 simplex lattice at resolution m over (w_Q, w_C, w_T, w_R).
   All (i, j, k, l) / m with i+j+k+l=m.  m=20 → 1771 weight vectors.

2. Min-max normalisation per objective axis over the pool before scoring,
   so every objective contributes on the same [0,1] scale regardless of
   units.  The normalised score is:
       r = w_Q·Q̃ − w_C·C̃ − w_T·T̃ − w_R·R̃
   where Q̃, C̃, T̃, R̃ are the normalised values.

3. Tie-breaking: PINNED — when two operators have the same score we always
   take the one with the lexicographically smallest operator_id.  This is
   the same deterministic tie-break used in dominance.py, which guarantees
   Scal ⊆ Sky: the selected operator must be non-dominated, because any
   operator that dominates it would score at least as high under any
   non-negative weight vector and would be selected instead (or tied, in
   which case the tie-break makes the choice deterministic but the
   property still holds because every tied op is also non-dominated).

Coverage metrics
----------------
scalarization_reachable(pool, m):
    Union over all weight vectors of the argmax-scoring operator.
    Returns a set[str] of operator_ids.

skyline_of(pool):
    Full O(k²) skyline computation.  Returns a set[str].

coverage_loss(reachable, skyline):
    1 − |reachable ∩ skyline| / |skyline|.
    Skyline method: loss = 0 always (every skyline member is reachable by
    definition via full_recompute).
    Scalarization: loss > 0 when non-convex pockets exist.
"""
from __future__ import annotations

import itertools
from typing import Sequence

from eddy.core.profile import QCTR, OperatorProfile
from eddy.skyline.dominance import dominates


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _minmax_normalize(
    pool: list[OperatorProfile],
) -> dict[str, tuple[float, float, float, float]]:
    """Return {op_id: (Q̃, C̃, T̃, R̃)} with each axis scaled to [0,1] over pool.

    If all values on an axis are identical the axis is mapped to 0.5 for all
    operators (degenerate but stable).
    """
    ids = [p.operator_id for p in pool]
    qs = [p.est.Q for p in pool]
    cs = [p.est.C for p in pool]
    ts = [p.est.T for p in pool]
    rs = [p.est.R for p in pool]

    def _scale(vals: list[float]) -> list[float]:
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span < 1e-15:
            return [0.5] * len(vals)
        return [(v - lo) / span for v in vals]

    qs_n = _scale(qs)
    cs_n = _scale(cs)
    ts_n = _scale(ts)
    rs_n = _scale(rs)

    return {op_id: (qn, cn, tn, rn)
            for op_id, qn, cn, tn, rn in zip(ids, qs_n, cs_n, ts_n, rs_n)}


# ---------------------------------------------------------------------------
# Weight-vector lattice
# ---------------------------------------------------------------------------

def simplex_lattice(m: int) -> list[tuple[float, float, float, float]]:
    """Generate all B3 simplex lattice points at resolution m.

    Returns weight vectors (w_Q, w_C, w_T, w_R) with w_Q+w_C+w_T+w_R = 1,
    each component a multiple of 1/m.  Total count = C(m+3, 3).

    m=5  →  56 vectors
    m=10 → 286 vectors
    m=20 → 1771 vectors
    m=40 → 12341 vectors
    """
    weights: list[tuple[float, float, float, float]] = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            for k in range(m + 1 - i - j):
                l = m - i - j - k
                weights.append((i / m, j / m, k / m, l / m))
    return weights


# ---------------------------------------------------------------------------
# Scalarization reachability
# ---------------------------------------------------------------------------

def scalarization_reachable(
    pool: list[OperatorProfile],
    m: int = 20,
) -> set[str]:
    """Operators reachable by ANY weight vector on the B3 simplex lattice.

    For each weight vector w = (w_Q, w_C, w_T, w_R), compute the normalised
    score r = w_Q·Q̃ − w_C·C̃ − w_T·T̃ − w_R·R̃ for every operator and take
    the argmax.  Tie-break: lexicographically smallest operator_id.

    Returns the union of all argmax selections.
    """
    if not pool:
        return set()

    norms = _minmax_normalize(pool)
    ids = sorted(p.operator_id for p in pool)  # sorted for pinned tie-break

    reachable: set[str] = set()
    for wq, wc, wt, wr in simplex_lattice(m):
        best_id: str | None = None
        best_score = float("-inf")
        for op_id in ids:
            qn, cn, tn, rn = norms[op_id]
            score = wq * qn - wc * cn - wt * tn - wr * rn
            if score > best_score + 1e-12:
                best_score = score
                best_id = op_id
            # equal scores: lexicographic tie-break — lower id wins (ids are sorted)
            # no action needed; ids is already sorted and we keep first maximum
        if best_id is not None:
            reachable.add(best_id)

    return reachable


# ---------------------------------------------------------------------------
# Skyline computation (standalone; wraps full_recompute logic)
# ---------------------------------------------------------------------------

def skyline_of(pool: list[OperatorProfile]) -> set[str]:
    """Compute the Pareto skyline of a pool.  O(k²).

    Standalone convenience — does not mutate any SkylineMaintainer.
    """
    if not pool:
        return set()
    ids = [p.operator_id for p in pool]
    sky: set[str] = set()
    for i in ids:
        est_i = next(p.est for p in pool if p.operator_id == i)
        dominated = False
        for j in ids:
            if j == i:
                continue
            est_j = next(p.est for p in pool if p.operator_id == j)
            if dominates(est_j, est_i):
                dominated = True
                break
        if not dominated:
            sky.add(i)
    return sky


# ---------------------------------------------------------------------------
# Coverage loss
# ---------------------------------------------------------------------------

def coverage_loss(reachable: set[str], skyline: set[str]) -> float:
    """Fraction of skyline members unreachable by the method.

    coverage_loss = 1 − |reachable ∩ skyline| / |skyline|

    Returns 0.0 if skyline is empty (undefined case; treat as perfect coverage).
    """
    if not skyline:
        return 0.0
    covered = len(reachable & skyline)
    return 1.0 - covered / len(skyline)

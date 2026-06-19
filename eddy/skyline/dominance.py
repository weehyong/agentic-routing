"""
eddy.skyline.dominance
======================
Pareto dominance predicate for (Q↑, C↓, T↓, R↓) operator profiles.

Convention
----------
as_tuple_for_dominance() negates Q, making every dimension "lower is better".
dominates(a, b) means "a is at least as good as b on every dim, and strictly
better on at least one".

Tie-breaking (PINNED)
---------------------
Equal vectors (|ta_i - tb_i| <= eps for all i) do NOT satisfy the "strictly
better on >=1" condition, so neither dominates the other.  This means equal
operators both appear in the skyline, which is the correct semantic: when two
operators tie exactly, you want access to both.

The PINNED property (Scal ⊆ Sky) is guaranteed by construction:
- scalarization argmax over the skyline finds the best linear combination;
- any operator reachable by scalarization must be non-dominated (otherwise a
  dominating operator scores at least as high on every weighted sum);
- therefore the scalarization-selected operator is in the skyline.
The converse (Sky ⊄ Scal for non-convex frontiers) is what Experiment 2 shows.
"""
from __future__ import annotations

from eddy.core.profile import QCTR


def dominates(a: QCTR, b: QCTR, eps: float = 1e-9) -> bool:
    """Return True iff *a* dominates *b*.

    Dominance: a is no worse than b on ALL objectives and strictly better on
    at least one.  We use a symmetric epsilon band so floating-point noise
    does not accidentally flip comparisons (especially the strict test).

    Parameters
    ----------
    a, b : QCTR
        Operator estimates to compare.
    eps : float
        Tolerance.  Values within eps are treated as equal; strictly better
        means a margin > eps.

    Returns
    -------
    bool
        True iff a dominates b.
    """
    ta = a.as_tuple_for_dominance()
    tb = b.as_tuple_for_dominance()
    no_worse = all(x <= y + eps for x, y in zip(ta, tb))
    strictly = any(x < y - eps for x, y in zip(ta, tb))
    return no_worse and strictly

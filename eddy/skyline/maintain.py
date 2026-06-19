"""
eddy.skyline.maintain
=====================
SkylineMaintainer: keeps the set of non-dominated operators current.

Two public update paths
-----------------------
full_recompute(eligible_ids) -- O(k²)
    Recomputes the skyline from scratch.  Used when the *shape* of the
    eligible set changes (different ready/done mask → different candidate set).
    Also the ground-truth oracle used by tests.

on_estimate_update(changed_id, eligible_ids) -- O(k) common / O(k²) worst-case
    Incremental update after ONE operator's EWMA estimate changes.

Correctness fix vs. DESIGN §6.2
---------------------------------
The routine in DESIGN §6.2 is BUGGY on DEGRADATION:

    operators A = (Q=0.9, C=0.1, T=0.1, R=0.1),
              B = (Q=0.5, C=0.1, T=0.1, R=0.1)

    After full_recompute: members = {A}  (A dominates B)
    A degrades to (Q=0.4, ...).
    Correct skyline: {B}.
    Buggy routine: checks "is A now dominated?" → yes → members.discard(A) → {}
                  (B was never re-admitted because the fast path skips
                   re-admission of previously-excluded operators.)

Root cause: the fast path only handles the improvement case (changed op becomes
non-dominated and adds dominations), but NOT the degradation case (changed op
was a skyline member and is now dominated, OR it stopped dominating some
operator it previously dominated — those operators may now be non-dominated and
must be re-admitted).

Safe fix (implemented here):
    1. If changed_id was NOT in members and is now non-dominated
       → O(k): it is safe to add it and evict newly dominated members
         (adding a better op can only remove others, never re-admit excluded ones).
    2. If changed_id IS in members and it is STILL non-dominated
       → O(k): refresh which other members it dominates (some it may no longer).
         Re-check members it previously dominated — they may now survive.
         This is still O(k) if we just re-sweep members for eviction.
       NOTE: we must also check whether non-members might now be non-dominated
         because changed_id stopped dominating them.  To handle this correctly
         we fall back to full_recompute if changed_id was previously dominating
         any non-member.  The cheap O(k) shortcut: if changed_id still dominates
         every op it used to dominate, we only need a members sweep.  But
         tracking "what did it used to dominate" requires extra state; instead
         we detect the case conservatively: if the changed op *improved* (its
         new estimate is ≤ old on all dims after negation), take the O(k) path;
         otherwise fall back to full_recompute (O(k²)).
    3. If changed_id IS in members and is NOW DOMINATED → fall back to
       full_recompute, because previously dominated ops may need re-admission.

Complexity:
    - O(k)  — improvement case (changed op gets better or is new non-member)
    - O(k²) — degradation case (changed op was a member and degraded, or
               became dominated)

This is the honest cost; the hot path is the improvement/new-non-member case,
which covers "just observed a good result and EWMA moves up" events.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eddy.core.profile import QCTR, OperatorProfile
from eddy.skyline.dominance import dominates


@dataclass(slots=True)
class SkylineMaintainer:
    """Tracks the Pareto-non-dominated (skyline) subset of a pool of operators.

    Attributes
    ----------
    members : set[str]
        Current skyline — operator_ids of non-dominated operators.
    profiles : dict[str, OperatorProfile]
        Reference to the shared profile map.  Must be updated by the caller
        *before* calling on_estimate_update.
    _prev_estimates : dict[str, QCTR]
        Internal snapshot of each operator's estimate as of the last update.
        Used to detect improvement vs. degradation without extra caller burden.
    """

    members: set[str] = field(default_factory=set)
    profiles: dict[str, OperatorProfile] = field(default_factory=dict)
    _prev_estimates: dict[str, QCTR] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def full_recompute(self, eligible_ids: list[str]) -> set[str]:
        """Recompute the skyline from scratch.  O(k²).

        After this call ``members`` equals the non-dominated subset of
        ``eligible_ids`` under the current profile estimates.
        """
        sky: set[str] = set()
        for i in eligible_ids:
            dominated = False
            for j in eligible_ids:
                if j != i and dominates(self.profiles[j].est, self.profiles[i].est):
                    dominated = True
                    break
            if not dominated:
                sky.add(i)
        self.members = sky
        # Snapshot estimates for future incremental updates
        for i in eligible_ids:
            self._prev_estimates[i] = self.profiles[i].est
        return sky

    def on_estimate_update(self, changed_id: str, eligible_ids: list[str]) -> set[str]:
        """Incrementally update the skyline after one operator's estimate changed.

        The caller must update ``self.profiles[changed_id].est`` *before*
        calling this method.

        Complexity
        ----------
        O(k)   — common / improvement path (changed op improved or is new)
        O(k²)  — degradation path (changed op degraded or became dominated)

        See module docstring for the correctness argument.
        """
        new_est = self.profiles[changed_id].est
        old_est = self._prev_estimates.get(changed_id)

        # Record new snapshot immediately
        self._prev_estimates[changed_id] = new_est

        was_member = changed_id in self.members

        # Detect improvement: new estimate is at least as good as old on every
        # (negated) dimension.  If old_est is unknown, treat as improvement.
        if old_est is not None:
            old_t = old_est.as_tuple_for_dominance()
            new_t = new_est.as_tuple_for_dominance()
            improved = all(n <= o + 1e-15 for n, o in zip(new_t, old_t))
        else:
            improved = True

        # Determine if changed_id is currently dominated by anyone
        ec = new_est
        is_dominated = any(
            j != changed_id and dominates(self.profiles[j].est, ec)
            for j in eligible_ids
        )

        if is_dominated:
            # changed_id cannot be in the skyline.
            if was_member:
                # DEGRADATION CASE: it was a member and is now dominated.
                # Non-members it used to dominate may now be non-dominated.
                # Must fall back to full_recompute.
                return self.full_recompute(eligible_ids)
            else:
                # It wasn't a member and still isn't.  No skyline change.
                return self.members
        else:
            # changed_id is non-dominated.
            if not was_member and improved:
                # IMPROVEMENT (new non-member becoming non-dominated): O(k) path.
                # Adding changed_id can only evict existing members it dominates;
                # it cannot cause previously excluded ops to re-enter (because
                # it is at least as good as before, so it still dominates anyone
                # it used to dominate — no evictions it held are released).
                self.members.add(changed_id)
                for m in list(self.members):
                    if m != changed_id and dominates(ec, self.profiles[m].est):
                        self.members.discard(m)
                return self.members

            elif was_member and improved:
                # Still a member and improved: O(k) path.
                # Re-sweep to evict members it newly dominates.
                # Improvement means it still dominates anyone it dominated before,
                # so no non-member re-admission is needed.
                for m in list(self.members):
                    if m != changed_id and dominates(ec, self.profiles[m].est):
                        self.members.discard(m)
                return self.members

            else:
                # NOT improved (degraded) but still non-dominated.
                # The changed op stayed on the skyline but may have stopped
                # dominating some ops it previously dominated — those non-members
                # might now be non-dominated.  Must fall back to full_recompute.
                return self.full_recompute(eligible_ids)

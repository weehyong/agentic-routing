"""
eddy.runtime.completion
=======================
Completion predicate φ(w) and the emission gate — Invariant 3 (DESIGN §11).

``satisfied`` is true iff every ``required`` bit is ``done`` *and* the item is not
parked awaiting approval.  ``_emit`` (in the runtime) re-checks the gate chain;
a gate failure at emission parks the item in AWAITING_APPROVAL rather than
emitting.
"""
from __future__ import annotations

from dataclasses import dataclass

from eddy.core.workitem import ItemStatus, WorkItem


@dataclass(slots=True)
class CompletionPredicate:
    def satisfied(self, item: WorkItem) -> bool:
        if item.status == ItemStatus.AWAITING_APPROVAL:
            return False
        req, done = item.required, item.done
        if not req:
            return False                     # nothing required ⇒ never auto-completes
        for i, bit in enumerate(req):
            if bit == 1 and (i >= len(done) or done[i] != 1):
                return False
        return True

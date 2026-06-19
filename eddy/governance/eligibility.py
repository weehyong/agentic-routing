"""
eddy.governance.eligibility
===========================
Per-item ready-mask computation (DESIGN §8) — feeds Invariant 1.

Eligibility is recomputed fresh per item per tick from three inputs: the
operator's static ``Precondition``, the item's ``done`` mask + ``signals``, and
the active gate chain.  An operator's ``ready`` bit is 1 iff its preconditions
are met *and* every gate allows it.

``requires_approval`` denials are surfaced separately so the runtime can park an
item in AWAITING_APPROVAL instead of failing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eddy.core.workitem import WorkItem
from eddy.governance.gates import GovernanceGate, evaluate_chain
from eddy.governance.permissions import PermissionScope

if TYPE_CHECKING:  # pragma: no cover
    from eddy.runtime.registry import OperatorRegistry


@dataclass(slots=True)
class EligibilityEvaluator:
    registry: "OperatorRegistry"
    scope: PermissionScope
    gates: list[GovernanceGate] = field(default_factory=list)

    # The most recent set of operator_ids whose denial requested approval — the
    # runtime reads this to decide whether to park an item in AWAITING_APPROVAL.
    pending_approval: set[str] = field(default_factory=set)

    def compute_ready_mask(self, item: WorkItem) -> list[int]:
        mask = [0] * len(self.registry)
        self.pending_approval = set()
        for i, op in enumerate(self.registry.operators):
            if not self._preconditions_met(op, item):
                continue
            # Operators may declare a dynamic eligibility predicate (e.g. the
            # approval operator is eligible only while AWAITING_APPROVAL, §8).
            is_elig = getattr(op, "is_eligible", None)
            if callable(is_elig) and not is_elig(item):
                continue
            res = evaluate_chain(self.gates, item, op, self.scope)
            if res.allowed:
                mask[i] = 1
            elif res.requires_approval:
                self.pending_approval.add(op.operator_id)
        return mask

    # ------------------------------------------------------------------
    def _preconditions_met(self, op, item: WorkItem) -> bool:
        pre = op.precondition
        if pre.task_types and item.task_type not in pre.task_types:
            return False
        # All prerequisite operators must be done.
        for dep_id in pre.requires_done:
            idx = self.registry.index(dep_id)
            if idx is None or idx >= len(item.done) or item.done[idx] != 1:
                return False
        # All required signal keys must be present.
        for key in pre.requires_signals:
            if key not in item.signals.quality and key not in item.signals.metadata:
                return False
        return True

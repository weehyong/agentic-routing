"""
eddy.governance.approval
========================
Human-in-the-loop approval, modeled as an operator (DESIGN §8).

The approval operator keeps approvals inside the same routing fabric: it is
eligible only while the item is AWAITING_APPROVAL, and its ``execute`` grants the
configured permission scope to the session — flipping the blocked operator's gate
state so it becomes ``ready`` on the next tick.

For offline / reproducible runs an ``auto_approve`` flag decides instantly; a real
deployment would block on an external decision instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eddy.core.operator import Idempotency, OperatorResult, Precondition, SideEffectClass
from eddy.core.profile import OperatorProfile, QCTR
from eddy.core.workitem import ItemStatus, WorkItem
from eddy.governance.permissions import PermissionScope

if TYPE_CHECKING:  # pragma: no cover
    from eddy.adapters.base import ExecContext


@dataclass(slots=True)
class ApprovalOperator:
    """Grants ``grants_scope`` to the session when run, then re-activates the item."""

    operator_id: str = "gov.approval"
    grants_scope: str = "write"
    scope: PermissionScope = field(default_factory=PermissionScope)
    auto_approve: bool = True
    side_effect: SideEffectClass = SideEffectClass.READ
    idempotency: Idempotency = Idempotency.AT_MOST_ONCE
    precondition: Precondition = field(default_factory=Precondition)
    _profile: OperatorProfile | None = None

    def __post_init__(self) -> None:
        if self._profile is None:
            self._profile = OperatorProfile.from_qctr(self.operator_id, self.declared_profile())

    @property
    def profile(self) -> OperatorProfile:
        return self._profile

    def declared_profile(self) -> QCTR:
        # Cheap, fast, low-risk: a human click, not a model call.
        return QCTR(Q=1.0, C=0.0, T=0.1, R=0.0)

    def is_eligible(self, item: WorkItem) -> bool:
        return item.status == ItemStatus.AWAITING_APPROVAL

    def execute(self, item: WorkItem, ctx: "ExecContext") -> OperatorResult:
        if self.auto_approve:
            self.scope.granted.add(self.grants_scope)
            item.status = ItemStatus.ACTIVE
            succeeded = True
        else:  # pragma: no cover - real deployments block on an external decision
            succeeded = False
        return OperatorResult(
            item=item,
            observed=self.declared_profile(),
            sets_done=[],
            succeeded=succeeded,
        )

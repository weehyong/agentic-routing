"""
eddy.governance.gates
======================
Governance gate chain (DESIGN §8).

Gates are composed as a chain; **any deny zeroes the operator's ``ready`` bit**.
A denial may also flag ``requires_approval`` to route the item through the
human-in-the-loop approval operator rather than failing it.

Concrete gates:
- PermissionGate          — operator's requires_permission ⊆ scope.granted
- WriteCapabilityGate     — WRITE operators blocked until "write" granted
- DataClassificationGate  — TRANSMIT operators blocked for non-allowed classes
- ToolArgGate             — schema/allowlist validation of tool calls (threat model)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from eddy.core.operator import Operator, SideEffectClass
from eddy.core.workitem import WorkItem
from eddy.governance.permissions import PermissionScope


@dataclass(frozen=True, slots=True)
class GateResult:
    allowed: bool
    reason: str = ""
    requires_approval: bool = False


@runtime_checkable
class GovernanceGate(Protocol):
    def evaluate(
        self, item: WorkItem, op: Operator, scope: PermissionScope
    ) -> GateResult: ...


_ALLOW = GateResult(allowed=True)


@dataclass(slots=True)
class PermissionGate:
    """Operator's required permission scopes must be a subset of the granted set."""

    def evaluate(self, item: WorkItem, op: Operator, scope: PermissionScope) -> GateResult:
        needed = op.precondition.requires_permission
        if needed <= scope.granted:
            return _ALLOW
        missing = needed - scope.granted
        return GateResult(False, f"missing permissions: {sorted(missing)}")


@dataclass(slots=True)
class WriteCapabilityGate:
    """WRITE operators are disabled until the session holds the 'write' scope.

    Denial flags ``requires_approval`` so the item can be parked in
    AWAITING_APPROVAL and the approval operator becomes the eligible path.
    """

    scope_name: str = "write"

    def evaluate(self, item: WorkItem, op: Operator, scope: PermissionScope) -> GateResult:
        if op.side_effect is not SideEffectClass.WRITE:
            return _ALLOW
        if self.scope_name in scope.granted:
            return _ALLOW
        return GateResult(False, "write capability not granted", requires_approval=True)


@dataclass(slots=True)
class DataClassificationGate:
    """TRANSMIT (external egress) operators blocked when the item's data
    classification is not in the session's transmit-allowed classes."""

    def evaluate(self, item: WorkItem, op: Operator, scope: PermissionScope) -> GateResult:
        if op.side_effect is not SideEffectClass.TRANSMIT:
            return _ALLOW
        if item.data_classification in scope.transmit_allowed_classes:
            return _ALLOW
        return GateResult(
            False,
            f"data_classification={item.data_classification!r} not allowed for transmit",
        )


@dataclass(slots=True)
class ToolArgGate:
    """Allowlist domains/paths for transmit/write operators (threat-model defense).

    The operator advertises the domain/path it intends to touch via
    ``item.payload['tool_domain']`` / ``['tool_path']`` (set by the planner);
    if advertised, it must be on the corresponding allowlist.  Operators that
    advertise nothing pass (nothing to validate).
    """

    def evaluate(self, item: WorkItem, op: Operator, scope: PermissionScope) -> GateResult:
        domain = item.payload.get("tool_domain")
        if domain is not None and scope.domain_allowlist and domain not in scope.domain_allowlist:
            return GateResult(False, f"domain {domain!r} not on allowlist")
        path = item.payload.get("tool_path")
        if path is not None and scope.path_allowlist and path not in scope.path_allowlist:
            return GateResult(False, f"path {path!r} not on allowlist")
        return _ALLOW


def evaluate_chain(
    gates: list[GovernanceGate],
    item: WorkItem,
    op: Operator,
    scope: PermissionScope,
) -> GateResult:
    """Run the chain; first deny wins (and propagates its requires_approval flag)."""
    for gate in gates:
        res = gate.evaluate(item, op, scope)
        if not res.allowed:
            return res
    return _ALLOW

"""
eddy.core.operator
==================
Operator Protocol, preconditions, and result (DESIGN §5.3).

An operator is the orchestration analogue of a relational operator.  It declares
a static ``Precondition`` (prerequisites, applicable task types, permissions), a
``side_effect`` class and ``idempotency`` (governance inputs, §8), a declared
prior to seed the EWMA, and an ``execute`` that returns an ``OperatorResult``.

``ExecContext`` is defined in ``eddy.adapters.base`` (the execution-substrate
seam, §9); it is referenced here only under ``TYPE_CHECKING`` so ``core`` never
imports outward at runtime (DESIGN §4 dependency direction).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from eddy.core.profile import OperatorProfile, QCTR
from eddy.core.workitem import WorkItem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eddy.adapters.base import ExecContext


class SideEffectClass(str, Enum):
    READ = "read"
    WRITE = "write"
    TRANSMIT = "transmit"          # transmit = external egress


class Idempotency(str, Enum):
    IDEMPOTENT = "idempotent"
    AT_MOST_ONCE = "at_most_once"
    NON_IDEMPOTENT = "non_idempotent"


@dataclass(slots=True)
class Precondition:
    requires_done: list[str] = field(default_factory=list)      # operator_ids that must be done
    requires_signals: list[str] = field(default_factory=list)   # signal keys that must exist
    task_types: set[str] = field(default_factory=set)           # applies only to these (empty = all)
    requires_permission: set[str] = field(default_factory=set)  # scope names needed


@dataclass(slots=True)
class OperatorResult:
    item: WorkItem
    observed: QCTR                                  # measured (Q,C,T,R) for EWMA + reward
    sets_done: list[str] = field(default_factory=list)
    spawned: list[WorkItem] = field(default_factory=list)   # split outputs
    succeeded: bool = True
    dollars: float = 0.0                            # charged to the budget
    tokens: int = 0


@runtime_checkable
class Operator(Protocol):
    operator_id: str
    side_effect: SideEffectClass
    idempotency: Idempotency
    precondition: Precondition

    @property
    def profile(self) -> OperatorProfile: ...

    def declared_profile(self) -> QCTR: ...                 # prior to seed the EWMA

    def execute(self, item: WorkItem, ctx: "ExecContext") -> OperatorResult: ...

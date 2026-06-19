"""
eddy.adapters.base
==================
The execution-substrate seam (DESIGN §9).

A single Protocol separates routing logic from execution substrate.  The router
never knows which backend an operator uses; a concrete operator owns one backend
and forwards ``execute`` to it.  ``ExecContext`` carries everything a backend
needs to charge the budget, respect the scope, log lineage, and stay deterministic
under the seeded ``rng``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from eddy.core.budget import BudgetTracker
from eddy.core.lineage import LineageRecord
from eddy.core.operator import OperatorResult
from eddy.core.workitem import WorkItem
from eddy.governance.permissions import PermissionScope


@dataclass(slots=True)
class ExecContext:
    budget: BudgetTracker
    scope: PermissionScope
    lineage: LineageRecord
    replay_id: str
    rng: random.Random


@runtime_checkable
class OperatorBackend(Protocol):
    def run(self, item: WorkItem, ctx: ExecContext) -> OperatorResult: ...

"""
eddy.routing.policy
===================
Routing-policy Protocol and the scalar reward (DESIGN §7).

Every policy selects from ``candidates = Sky(w) ∩ E(w)`` handed in by the router,
so Invariant 4 (never select a dominated operator) is enforced *upstream* and a
policy cannot violate it.  Scalarized routing is the special case: maximize
``reward`` within a skyline tier (still skyline-restricted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from eddy.core.budget import BudgetTracker
from eddy.core.profile import OperatorProfile, QCTR


@dataclass(slots=True)
class RoutingContext:
    item: object                                  # WorkItem (avoid import cycle)
    budget: BudgetTracker
    queue_lengths: dict[str, int] = field(default_factory=dict)
    profiles: dict[str, OperatorProfile] = field(default_factory=dict)


@runtime_checkable
class RoutingPolicy(Protocol):
    def select(self, candidates: list[str], ctx: RoutingContext) -> str: ...
    def update(self, operator_id: str, reward: float, ctx: RoutingContext) -> None: ...


def reward(est: QCTR, budget: BudgetTracker, lam: float = 1.0, mu: float = 1.0, nu: float = 1.0) -> float:
    """r = Q − λC − μT − νR  (DESIGN §7).  C uses the raw estimate; the router
    may pre-shift C via ``budget.cost`` before this is called."""
    return est.Q - lam * est.C - mu * est.T - nu * est.R

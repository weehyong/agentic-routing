"""
eddy.routing.lottery
====================
P2 — windowed lottery, Eddies-inspired ticket scheduling (DESIGN §7).

Each operator carries a sliding-window utility ``uᵢ ← (1−ρ)·uᵢ + ρ·r`` and is
assigned tickets ``tᵢ ∝ exp(β·uᵢ)``.  Selection samples ``∝ tᵢ`` over the skyline
candidates — probabilistic, naturally load-balancing, mirroring Eddies tickets.

A softmax over a *bounded* utility window makes the policy adapt to
non-stationarity: when an operator degrades, its windowed utility decays and its
ticket share shrinks within a few observations.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from eddy.routing.policy import RoutingContext


@dataclass(slots=True)
class WindowedLottery:
    beta: float = 4.0
    rho: float = 0.3                                   # window update rate
    rng: random.Random = field(default_factory=random.Random)
    utility: dict[str, float] = field(default_factory=dict)

    def select(self, candidates: list[str], ctx: RoutingContext) -> str:
        # Stable softmax over candidate utilities (default 0 for unseen ops).
        us = [self.utility.get(a, 0.0) for a in candidates]
        m = max(us)
        weights = [math.exp(self.beta * (u - m)) for u in us]
        total = sum(weights)
        if total <= 0.0:
            return self.rng.choice(candidates)
        r = self.rng.random() * total
        acc = 0.0
        for a, wt in zip(candidates, weights):
            acc += wt
            if r <= acc:
                return a
        return candidates[-1]

    def update(self, operator_id: str, reward: float, ctx: RoutingContext) -> None:
        u = self.utility.get(operator_id, 0.0)
        self.utility[operator_id] = (1.0 - self.rho) * u + self.rho * reward

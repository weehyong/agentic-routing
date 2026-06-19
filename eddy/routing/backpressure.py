"""
eddy.routing.backpressure
=========================
P1 — backpressure-aware greedy (DESIGN §7).

Picks ``argmin (T̂(a) + β·q_a)`` over the skyline candidates, where ``q_a`` is the
operator's queue length.  Reacts to backlog; no exploration.  Deterministic ties
broken by operator_id for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass

from eddy.routing.policy import RoutingContext


@dataclass(slots=True)
class BackpressureGreedy:
    beta: float = 1.0

    def select(self, candidates: list[str], ctx: RoutingContext) -> str:
        profiles = ctx.profiles
        ql = ctx.queue_lengths

        def score(a: str) -> tuple[float, str]:
            t_hat = profiles[a].est.T
            q = ql.get(a, 0)
            return (t_hat + self.beta * q, a)

        return min(candidates, key=score)

    def update(self, operator_id: str, reward: float, ctx: RoutingContext) -> None:
        # Greedy policy keeps no learned state.
        return None

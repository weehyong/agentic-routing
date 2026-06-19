"""
eddy.routing.bandit
===================
P3 — contextual bandit over the skyline (DESIGN §7).

Operators in ``Sky(w)`` are arms; the context is ``(task_type, sla_tier,
safety_tier)``.  Selection is **discounted UCB**: per (context, arm) we keep a
γ-discounted pull count and reward sum, score ``mean + c·√(ln N / n)``, and pull
any unsampled arm first (optimism).  The discount γ<1 down-weights stale
observations so the bandit re-explores when an arm degrades mid-run — the
non-stationary regime of Experiment 1 (Garivier & Moulines, discounted-UCB).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from eddy.routing.policy import RoutingContext


@dataclass(slots=True)
class ContextualBandit:
    c: float = 1.0                                     # exploration coefficient
    gamma: float = 0.95                                # discount (non-stationarity)
    counts: dict[tuple[str, str], float] = field(default_factory=dict)   # (ctx, arm) -> n
    sums: dict[tuple[str, str], float] = field(default_factory=dict)     # (ctx, arm) -> Σr
    totals: dict[str, float] = field(default_factory=dict)               # ctx -> N

    @staticmethod
    def _ctx_key(ctx: RoutingContext) -> str:
        it = ctx.item
        return f"{it.task_type}|{it.sla_tier}|{it.safety_tier}"

    def select(self, candidates: list[str], ctx: RoutingContext) -> str:
        key = self._ctx_key(ctx)
        n_total = self.totals.get(key, 0.0)
        log_n = math.log(max(n_total, math.e))

        best, best_val = candidates[0], -math.inf
        for a in candidates:
            n = self.counts.get((key, a), 0.0)
            if n < 1e-9:
                return a                               # optimism: pull unsampled arm first
            mean = self.sums[(key, a)] / n
            bonus = self.c * math.sqrt(log_n / n)
            val = mean + bonus
            if val > best_val:
                best, best_val = a, val
        return best

    def update(self, operator_id: str, reward: float, ctx: RoutingContext) -> None:
        key = self._ctx_key(ctx)
        # γ-discount every arm in this context, then fold in the new observation.
        for (k, a) in list(self.counts):
            if k == key:
                self.counts[(k, a)] *= self.gamma
                self.sums[(k, a)] *= self.gamma
        self.totals[key] = self.totals.get(key, 0.0) * self.gamma + 1.0
        self.counts[(key, operator_id)] = self.counts.get((key, operator_id), 0.0) + 1.0
        self.sums[(key, operator_id)] = self.sums.get((key, operator_id), 0.0) + reward

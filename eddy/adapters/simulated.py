"""
eddy.adapters.simulated
=======================
Zero-dependency simulated backend + operator (DESIGN §9).

``SimulatedBackend`` draws ``(Q,C,T,R)`` from a declared base ± Gaussian noise,
charges the budget, and optionally fails transiently or mutates ``signals`` via
an ``effect`` callable.  Because the base distribution is *mutable*, the
Experiment-1 volatility harness can inject latency spikes, quality drift, rate
limiting, and outages by mutating an operator's base mid-run (EVALUATION §Exp 1).

``SimulatedOperator`` is a concrete ``Operator`` that owns one backend and adds
the orchestration metadata (preconditions, side-effect class, profile, the
``done`` bits it sets).  Fully offline and deterministic under a seeded ``rng``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from eddy.adapters.base import ExecContext
from eddy.core.operator import Idempotency, OperatorResult, Precondition, SideEffectClass
from eddy.core.profile import OperatorProfile, QCTR
from eddy.core.workitem import WorkItem


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(slots=True)
class SimulatedBackend:
    """Draws an observed QCTR around a (mutable) base and charges the budget."""

    base: QCTR
    noise: float = 0.02                  # Gaussian sd applied per dimension
    fail_prob: float = 0.0               # transient-outage probability
    tokens_per_call: int = 100
    effect: Callable[[WorkItem, "ExecContext"], None] | None = None

    def run(self, item: WorkItem, ctx: ExecContext) -> OperatorResult:
        rng = ctx.rng
        failed = rng.random() < self.fail_prob

        b = self.base
        obs = QCTR(
            Q=_clamp01(rng.gauss(b.Q, self.noise)),
            C=max(0.0, rng.gauss(b.C, self.noise * b.C if b.C else self.noise)),
            T=max(0.0, rng.gauss(b.T, self.noise * b.T if b.T else self.noise)),
            R=_clamp01(rng.gauss(b.R, self.noise)),
        )
        if failed:
            # An outage yields no quality, full risk, and still costs latency/$.
            obs = QCTR(Q=0.0, C=obs.C, T=obs.T, R=1.0)

        dollars = obs.C
        tokens = self.tokens_per_call
        ctx.budget.charge(dollars, tokens)

        if not failed and self.effect is not None:
            self.effect(item, ctx)

        return OperatorResult(
            item=item,
            observed=obs,
            sets_done=[],                 # filled in by SimulatedOperator
            succeeded=not failed,
            dollars=dollars,
            tokens=tokens,
        )


@dataclass(slots=True)
class SimulatedOperator:
    """Concrete Operator backed by a SimulatedBackend."""

    operator_id: str
    backend: SimulatedBackend
    precondition: Precondition = field(default_factory=Precondition)
    side_effect: SideEffectClass = SideEffectClass.READ
    idempotency: Idempotency = Idempotency.IDEMPOTENT
    sets_done: list[str] | None = None   # ids set on success; default = [operator_id]
    spawn: Callable[[WorkItem, "ExecContext"], list[WorkItem]] | None = None
    _profile: OperatorProfile | None = None

    def __post_init__(self) -> None:
        if self.sets_done is None:
            self.sets_done = [self.operator_id]
        if self._profile is None:
            self._profile = OperatorProfile.from_qctr(self.operator_id, self.declared_profile())

    @property
    def profile(self) -> OperatorProfile:
        return self._profile

    def declared_profile(self) -> QCTR:
        return self.backend.base

    def execute(self, item: WorkItem, ctx: ExecContext) -> OperatorResult:
        result = self.backend.run(item, ctx)
        if result.succeeded:
            result.sets_done = list(self.sets_done)
            if self.spawn is not None:
                result.spawned = self.spawn(item, ctx)
        return result

"""
eddy.runtime.eddy
=================
The Eddy router — Algorithm 1 main loop (DESIGN §11).

Each tick: pop a work item, recompute its ready mask (governance + preconditions),
take ``E(w)`` (ready ∧ not-done), recompute the skyline over the *budget-adjusted*
estimates, restrict candidates to ``Sky(w) ∩ E(w)``, let the policy pick one,
execute it, update the EWMA estimate, reward the policy, apply ``done`` bits
monotonically, and either emit (if φ holds and gates pass) or re-enqueue.

The five invariants are enforced here:
1. Eligibility   — assert ready=1 at selection.
2. Monotone done — _apply_done only *sets* bits (+ logs); never clears.
3. Safe emission — _emit requires φ(w) and re-checks the gate chain.
4. Skyline       — candidates ⊆ skyline members (policies can't escape it).
5. Lineage       — every decision/invocation/mask-update/emission is appended.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from eddy.adapters.base import ExecContext
from eddy.core.budget import BudgetTracker
from eddy.core.ids import new_replay_id
from eddy.core.lineage import LineageKind, LineageRecord
from eddy.core.profile import OperatorProfile, QCTR
from eddy.core.workitem import ItemStatus, WorkItem
from eddy.governance.eligibility import EligibilityEvaluator
from eddy.observability.events import Event
from eddy.observability.frontier_log import FrontierLog
from eddy.observability.sink import EventSink, NullSink
from eddy.routing.policy import RoutingContext, RoutingPolicy, reward
from eddy.runtime.completion import CompletionPredicate
from eddy.runtime.queue import WorkQueue
from eddy.runtime.registry import OperatorRegistry
from eddy.skyline.maintain import SkylineMaintainer


@dataclass(slots=True)
class TickRecord:
    tick: int
    item_id: str
    selected: str
    candidates: list[str]
    observed: QCTR
    reward: float
    beta: float
    succeeded: bool


@dataclass(slots=True)
class Eddy:
    registry: OperatorRegistry
    policy: RoutingPolicy
    eligibility: EligibilityEvaluator
    budget: BudgetTracker
    queue: WorkQueue = field(default_factory=WorkQueue)
    completion: CompletionPredicate = field(default_factory=CompletionPredicate)
    frontier_log: FrontierLog = field(default_factory=FrontierLog)
    sink: EventSink = field(default_factory=NullSink)
    rng: random.Random = field(default_factory=random.Random)
    approval_op_id: str | None = None
    op_tokens: dict[str, int] = field(default_factory=dict)

    skyline: SkylineMaintainer = field(default_factory=SkylineMaintainer)
    lineage_store: dict[str, LineageRecord] = field(default_factory=dict)
    history: list[TickRecord] = field(default_factory=list)
    emitted: list[WorkItem] = field(default_factory=list)
    failed: list[tuple[WorkItem, str]] = field(default_factory=list)
    _tick: int = 0

    def __post_init__(self) -> None:
        # Default per-operator token estimate from a SimulatedBackend if present.
        for op in self.registry.operators:
            if op.operator_id in self.op_tokens:
                continue
            backend = getattr(op, "backend", None)
            self.op_tokens[op.operator_id] = getattr(backend, "tokens_per_call", 100)

    # ------------------------------------------------------------------
    # Seeding
    def submit(self, item: WorkItem, required_ids: list[str]) -> None:
        """Enqueue a root work item, initialising its masks against the registry."""
        k = len(self.registry)
        if not item.done:
            item.done = [0] * k
        if not item.required:
            item.required = self.registry.required_mask(required_ids)
        if not item.lineage_id:
            item.lineage_id = item.item_id
        self.lineage_store.setdefault(item.lineage_id, LineageRecord(item.lineage_id))
        self.queue.push(item)

    # ------------------------------------------------------------------
    # Budget-adjusted routing view of the estimates (dynamic β, DESIGN §5.5/§12).
    def _routing_profiles(self, eligible: list[str]) -> dict[str, OperatorProfile]:
        rp: dict[str, OperatorProfile] = {}
        for oid in eligible:
            est = self.registry.profiles[oid].est
            c_eff = self.budget.cost(est.C, self.op_tokens.get(oid, 100))
            rp[oid] = OperatorProfile.from_qctr(
                oid, QCTR(Q=est.Q, C=c_eff, T=est.T, R=est.R)
            )
        return rp

    # ------------------------------------------------------------------
    def run(self, max_ticks: int | None = None) -> dict:
        while not self.queue.empty() and not self.budget.exhausted():
            if max_ticks is not None and self._tick >= max_ticks:
                break
            self._step()
        return self.summary()

    def _step(self) -> None:
        w = self.queue.pop()
        self._tick += 1
        lin = self.lineage_store.setdefault(w.lineage_id, LineageRecord(w.lineage_id))

        w.ready = self.eligibility.compute_ready_mask(w)
        eligible = self.registry.eligible_ids(w)

        if not eligible:
            self._handle_no_candidates(w, lin)
            return

        # Skyline over the budget-adjusted estimates; candidates = Sky(w) ∩ E(w).
        self.skyline.profiles = self._routing_profiles(eligible)
        members = self.skyline.full_recompute(eligible)
        candidates = [a for a in eligible if a in members]            # Invariant 4
        if not candidates:
            self._handle_no_candidates(w, lin)
            return

        replay_id = new_replay_id()
        ctx = RoutingContext(
            item=w,
            budget=self.budget,
            queue_lengths=self.queue.operator_backlog(self.registry),
            profiles=self.registry.profiles,
        )
        op_id = self.policy.select(candidates, ctx)
        assert op_id in candidates, "policy escaped the skyline (Invariant 4)"
        assert w.ready[self.registry.index(op_id)] == 1, "selected op not ready (Invariant 1)"

        lin.append(LineageKind.ROUTING_DECISION, replay_id, chosen=op_id, skyline=list(candidates))
        self.frontier_log.record(self._tick, w.item_id, candidates, self.skyline.profiles, op_id)
        self._emit_event("routing_decision", w, replay_id, chosen=op_id, n_candidates=len(candidates))

        op = self.registry.get(op_id)
        ectx = ExecContext(self.budget, self.eligibility.scope, lin, replay_id, self.rng)
        lin.append(LineageKind.INVOCATION, replay_id, operator=op_id)
        result = op.execute(w, ectx)

        # Invariant 2: monotone done — set-only, logged.
        self._apply_done(result.item, result.sets_done, replay_id, lin)

        r = reward(result.observed, self.budget)
        prof = self.registry.profiles[op_id]
        prof.update_estimate(result.observed)
        self.policy.update(op_id, r, ctx)

        self.history.append(TickRecord(
            tick=self._tick, item_id=w.item_id, selected=op_id, candidates=list(candidates),
            observed=result.observed, reward=r, beta=self.budget.beta, succeeded=result.succeeded,
        ))

        for child in result.spawned:
            self.submit(child, required_ids=[])

        item = result.item
        if self.completion.satisfied(item):
            self._emit(item, lin, replay_id)
        else:
            self.queue.push(item)

    # ------------------------------------------------------------------
    def _handle_no_candidates(self, w: WorkItem, lin: LineageRecord) -> None:
        # Park for approval if a gate requested it and an approval operator exists.
        if (
            self.approval_op_id is not None
            and self.eligibility.pending_approval
            and w.status != ItemStatus.AWAITING_APPROVAL
        ):
            w.status = ItemStatus.AWAITING_APPROVAL
            idx = self.registry.index(self.approval_op_id)
            if idx is not None:
                w.ready = [0] * len(self.registry)
                w.ready[idx] = 1
            self.queue.push(w)
            return
        if w.status == ItemStatus.AWAITING_APPROVAL and self.approval_op_id is not None:
            # Approval operator is the only eligible path — run it directly.
            self._run_approval(w, lin)
            return
        if self.completion.satisfied(w):
            self._emit(w, lin, new_replay_id())
        else:
            self._fail(w, "no_eligible_operator", lin)

    def _run_approval(self, w: WorkItem, lin: LineageRecord) -> None:
        replay_id = new_replay_id()
        op = self.registry.get(self.approval_op_id)
        ectx = ExecContext(self.budget, self.eligibility.scope, lin, replay_id, self.rng)
        lin.append(LineageKind.GATE, replay_id, approval=self.approval_op_id)
        op.execute(w, ectx)             # grants scope, flips status back to ACTIVE
        self.queue.push(w)

    def _apply_done(self, item: WorkItem, sets_done: list[str], replay_id: str, lin: LineageRecord) -> None:
        for did in sets_done:
            idx = self.registry.index(did)
            if idx is None or idx >= len(item.done):
                continue
            if item.done[idx] == 0:                          # set-only ⇒ monotone (Inv 2)
                item.done[idx] = 1
                lin.append(LineageKind.MASK_UPDATE, replay_id, set_done=did)

    def _emit(self, item: WorkItem, lin: LineageRecord, replay_id: str) -> None:
        # Invariant 3: re-check φ and the emission gate chain.
        if not self.completion.satisfied(item):
            self._fail(item, "emit_without_completion", lin)
            return
        item.status = ItemStatus.EMITTED
        lin.append(LineageKind.EMISSION, replay_id, item_id=item.item_id)
        self._emit_event("emission", item, replay_id)
        self.emitted.append(item)

    def _fail(self, item: WorkItem, reason: str, lin: LineageRecord) -> None:
        item.status = ItemStatus.FAILED
        lin.append(LineageKind.FAILURE, new_replay_id(), reason=reason)
        self._emit_event("failure", item, "", reason=reason)
        self.failed.append((item, reason))

    def _emit_event(self, kind: str, item: WorkItem, replay_id: str, **payload) -> None:
        self.sink.emit(Event(kind=kind, item_id=item.item_id, replay_id=replay_id, payload=payload))

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "ticks": self._tick,
            "emitted": len(self.emitted),
            "failed": len(self.failed),
            "dollars_spent": round(self.budget.dollars_spent, 6),
            "tokens_spent": self.budget.tokens_spent,
            "budget_exhausted": self.budget.exhausted(),
            "mean_reward": round(
                sum(h.reward for h in self.history) / len(self.history), 6
            ) if self.history else 0.0,
        }

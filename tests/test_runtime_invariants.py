"""
Invariant-focused runtime tests (DESIGN §14).

Covers Invariants 1–5, governance gates (§8), budget/dynamic-β (§5.5),
the routing policies (§7), and safe emission (§11), plus a fuzz over random
operator chains × policies asserting all five invariants hold every tick.
"""
from __future__ import annotations

import random

import pytest

from eddy.adapters.simulated import SimulatedBackend, SimulatedOperator
from eddy.core.budget import BudgetTracker
from eddy.core.ids import reset
from eddy.core.operator import Precondition, SideEffectClass
from eddy.core.profile import OperatorProfile, QCTR
from eddy.core.workitem import ItemStatus, WorkItem
from eddy.governance.approval import ApprovalOperator
from eddy.governance.eligibility import EligibilityEvaluator
from eddy.governance.gates import (
    DataClassificationGate,
    PermissionGate,
    ToolArgGate,
    WriteCapabilityGate,
)
from eddy.governance.permissions import PermissionScope
from eddy.routing.backpressure import BackpressureGreedy
from eddy.routing.bandit import ContextualBandit
from eddy.routing.lottery import WindowedLottery
from eddy.routing.policy import RoutingContext
from eddy.runtime.eddy import Eddy
from eddy.runtime.registry import OperatorRegistry


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _chain_ops(n: int, task="t", write_last=False, transmit_last=False):
    ops = []
    prev = None
    for i in range(n):
        oid = f"op{i}"
        se = SideEffectClass.READ
        if i == n - 1 and write_last:
            se = SideEffectClass.WRITE
        if i == n - 1 and transmit_last:
            se = SideEffectClass.TRANSMIT
        base = QCTR(Q=0.6 + 0.03 * i, C=0.02 + 0.01 * i, T=0.2 + 0.05 * i, R=0.05)
        ops.append(
            SimulatedOperator(
                oid,
                SimulatedBackend(base, noise=0.0, tokens_per_call=50),
                precondition=Precondition(
                    requires_done=[prev] if prev else [], task_types={task}
                ),
                side_effect=se,
            )
        )
        prev = oid
    return ops


def _make_eddy(ops, scope=None, gates=None, policy=None, budget=None, approval=None, seed=1):
    reg = OperatorRegistry(ops)
    scope = scope if scope is not None else PermissionScope()
    gates = gates if gates is not None else [
        PermissionGate(), WriteCapabilityGate(), DataClassificationGate(), ToolArgGate()
    ]
    elig = EligibilityEvaluator(reg, scope, gates)
    return Eddy(
        registry=reg,
        policy=policy or BackpressureGreedy(),
        eligibility=elig,
        budget=budget or BudgetTracker(1e6, 10**9),
        rng=random.Random(seed),
        approval_op_id=approval,
    )


# ---------------------------------------------------------------------------
# Invariant 1 — eligibility / governance
# ---------------------------------------------------------------------------

def test_selected_op_always_ready():
    reset()
    eddy = _make_eddy(_chain_ops(4))
    eddy.submit(WorkItem("w", "t"), [f"op{i}" for i in range(4)])
    eddy.run()
    # Every routing decision must have chosen a then-ready op (asserted in loop);
    # reaching here without assertion error proves Invariant 1 held each tick.
    assert len(eddy.emitted) == 1


def test_write_op_blocked_without_scope_then_approved():
    reset()
    ops = _chain_ops(2, write_last=True)
    scope = PermissionScope(granted=set())
    approval = ApprovalOperator(operator_id="gov.approval", grants_scope="write", scope=scope)
    eddy = _make_eddy(ops + [approval], scope=scope, approval="gov.approval")
    eddy.submit(WorkItem("w", "t"), ["op0", "op1"])
    eddy.run()
    # op1 is a WRITE op; without "write" it parks for approval, the approval op
    # grants the scope, and then op1 runs and the item emits.
    assert "write" in scope.granted
    assert len(eddy.emitted) == 1
    w = eddy.emitted[0]
    assert w.done[eddy.registry.index("op1")] == 1


def test_transmit_op_blocked_for_regulated_data():
    reset()
    ops = _chain_ops(2, transmit_last=True)
    eddy = _make_eddy(ops)
    w = WorkItem("w", "t", data_classification="regulated")
    eddy.submit(w, ["op0"])   # op1 not required → op0 alone completes it
    eddy.run()
    # op1 (transmit) must never have been selected for regulated data.
    chosen = {h.selected for h in eddy.history}
    assert "op1" not in chosen
    assert w.status in (ItemStatus.EMITTED, ItemStatus.FAILED)


def test_permission_gate_blocks_missing_scope():
    reset()
    op = SimulatedOperator(
        "needs_perm", SimulatedBackend(QCTR(0.8, 0.02, 0.2, 0.05), noise=0.0),
        precondition=Precondition(task_types={"t"}, requires_permission={"special"}),
    )
    eddy = _make_eddy([op])
    eddy.submit(WorkItem("w", "t"), ["needs_perm"])
    eddy.run()
    assert eddy.history == []                     # never eligible
    assert len(eddy.failed) == 1


# ---------------------------------------------------------------------------
# Invariant 2 — monotone done
# ---------------------------------------------------------------------------

def test_done_bits_monotone_nondecreasing():
    reset()
    eddy = _make_eddy(_chain_ops(5))
    w = WorkItem("w", "t")
    eddy.submit(w, [f"op{i}" for i in range(5)])
    # Step manually, asserting done never decreases.
    prev = list(w.done)
    while not eddy.queue.empty():
        eddy._step()
        cur = w.done
        assert all(c >= p for c, p in zip(cur, prev)), "done bit cleared (Invariant 2 violated)"
        prev = list(cur)
    assert sum(w.done) == 5


# ---------------------------------------------------------------------------
# Invariant 3 — safe emission
# ---------------------------------------------------------------------------

def test_no_emit_until_required_done():
    reset()
    ops = _chain_ops(3)
    eddy = _make_eddy(ops)
    w = WorkItem("w", "t")
    eddy.submit(w, ["op0", "op1", "op2"])
    # Drive one step at a time; item must not be EMITTED before all required done.
    while not eddy.queue.empty():
        eddy._step()
        if w.status == ItemStatus.EMITTED:
            assert sum(w.done) == 3
    assert w.status == ItemStatus.EMITTED


# ---------------------------------------------------------------------------
# Invariant 4 — skyline restriction (policy cannot escape)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy_factory", [
    lambda: BackpressureGreedy(),
    lambda: WindowedLottery(rng=random.Random(0)),
    lambda: ContextualBandit(),
])
def test_policy_selection_within_skyline(policy_factory):
    reset()
    eddy = _make_eddy(_chain_ops(4), policy=policy_factory())
    eddy.submit(WorkItem("w", "t"), [f"op{i}" for i in range(4)])
    eddy.run()
    for h in eddy.history:
        assert h.selected in h.candidates           # Invariant 4


# ---------------------------------------------------------------------------
# Invariant 5 — lineage
# ---------------------------------------------------------------------------

def test_lineage_event_count_ge_invocations():
    reset()
    eddy = _make_eddy(_chain_ops(4))
    eddy.submit(WorkItem("w", "t"), [f"op{i}" for i in range(4)])
    eddy.run()
    from eddy.core.lineage import LineageKind
    lin = eddy.lineage_store["w"]
    n_inv = lin.count(LineageKind.INVOCATION)
    assert lin.count() >= n_inv >= 1
    assert lin.count(LineageKind.ROUTING_DECISION) == n_inv     # one decision per invocation
    assert lin.count(LineageKind.EMISSION) == 1


# ---------------------------------------------------------------------------
# Budget / dynamic-β (§5.5)
# ---------------------------------------------------------------------------

def test_budget_halts_loop_on_exhaustion():
    reset()
    op = SimulatedOperator("loop", SimulatedBackend(QCTR(0.5, 0.0, 0.1, 0.1), noise=0.0,
                                                    tokens_per_call=100),
                           precondition=Precondition(task_types={"t"}),
                           sets_done=[])             # never sets done → would loop forever
    eddy = _make_eddy([op], budget=BudgetTracker(1e6, 1000))   # 1000 tokens / 100 per call
    eddy.submit(WorkItem("w", "t"), ["loop"])
    eddy.run(max_ticks=10_000)
    assert eddy.budget.exhausted()
    assert eddy.budget.tokens_spent <= 1000 + 100        # halts within one call of the cap


def test_beta_increases_monotonically_as_budget_depletes():
    b = BudgetTracker(1e6, 1000)
    betas = []
    for _ in range(9):
        betas.append(b.beta)
        b.charge(0.0, 100)
    assert all(b2 >= b1 for b1, b2 in zip(betas, betas[1:]))
    assert betas[-1] > betas[0]


def test_budget_adjustment_excludes_token_heavy_op():
    """Dynamic-β mechanism (DESIGN §5.5/§12): budget-adjusting the C dimension
    drops a token-heavy operator from the skyline in favour of an equal-quality
    token-light one, and the effective-cost gap *widens* as budget depletes —
    "the frontier shifts toward cheaper operators as budget depletes."
    """
    from eddy.skyline.maintain import SkylineMaintainer
    reset()
    base = QCTR(0.90, 0.02, 0.30, 0.10)            # A and B identical except token cost
    a = SimulatedOperator("A", SimulatedBackend(base, noise=0.0, tokens_per_call=5000),
                          precondition=Precondition(task_types={"t"}), sets_done=[])
    b = SimulatedOperator("B", SimulatedBackend(base, noise=0.0, tokens_per_call=10),
                          precondition=Precondition(task_types={"t"}), sets_done=[])
    eddy = _make_eddy([a, b], budget=BudgetTracker(1e9, 20000))

    # Raw (token-agnostic) estimates are identical ⇒ neither dominates ⇒ both members.
    raw = SkylineMaintainer(profiles=dict(eddy.registry.profiles))
    assert raw.full_recompute(["A", "B"]) == {"A", "B"}

    # Budget-adjusted: A is token-heavy ⇒ strictly higher effective C ⇒ B dominates A.
    rp_full = eddy._routing_profiles(["A", "B"])
    assert SkylineMaintainer(profiles=rp_full).full_recompute(["A", "B"]) == {"B"}
    gap_full = rp_full["A"].est.C - rp_full["B"].est.C
    assert gap_full > 0

    # Deplete the budget; β rises; the effective-cost gap widens further.
    eddy.budget.charge(0.0, 18000)
    rp_low = eddy._routing_profiles(["A", "B"])
    gap_low = rp_low["A"].est.C - rp_low["B"].est.C
    assert gap_low > gap_full
    assert SkylineMaintainer(profiles=rp_low).full_recompute(["A", "B"]) == {"B"}


# ---------------------------------------------------------------------------
# Policy unit behaviour (§7)
# ---------------------------------------------------------------------------

def test_lottery_distribution_tracks_utility():
    lot = WindowedLottery(beta=4.0, rho=0.3, rng=random.Random(0))
    ctx = RoutingContext(WorkItem("w", "t"), BudgetTracker(1, 1), {}, {})
    for _ in range(20):
        lot.update("hi", 1.0, ctx)
        lot.update("lo", 0.0, ctx)
    from collections import Counter
    c = Counter(lot.select(["hi", "lo"], ctx) for _ in range(3000))
    assert c["hi"] > c["lo"] * 5          # strong preference for the high-utility arm


def test_ucb_prefers_undersampled_on_ties():
    bandit = ContextualBandit(c=1.0)
    ctx = RoutingContext(WorkItem("w", "t"), BudgetTracker(1, 1), {}, {})
    for _ in range(8):
        bandit.update("a", 0.5, ctx)
    bandit.update("b", 0.5, ctx)          # same mean, fewer pulls
    assert bandit.select(["a", "b"], ctx) == "b"


# ---------------------------------------------------------------------------
# Fuzz — all five invariants over random chains × policies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_fuzz_all_invariants(seed):
    reset()
    rng = random.Random(seed)
    n = rng.randint(2, 6)
    ops = _chain_ops(n)
    policy = rng.choice([
        BackpressureGreedy(),
        WindowedLottery(rng=random.Random(seed)),
        ContextualBandit(),
    ])
    eddy = _make_eddy(ops, policy=policy, seed=seed)
    w = WorkItem("w", "t")
    eddy.submit(w, [f"op{i}" for i in range(n)])

    prev_done = list(w.done)
    while not eddy.queue.empty() and not eddy.budget.exhausted():
        eddy._step()
        # Inv 2: monotone done
        assert all(c >= p for c, p in zip(w.done, prev_done))
        prev_done = list(w.done)
    # Inv 4: every selection inside its skyline
    for h in eddy.history:
        assert h.selected in h.candidates
    # Inv 3: emitted ⇒ all required done
    if w.status == ItemStatus.EMITTED:
        assert all(d == 1 for r, d in zip(w.required, w.done) if r == 1)
    # Inv 5: lineage ≥ invocations ≥ 1
    from eddy.core.lineage import LineageKind
    lin = eddy.lineage_store["w"]
    assert lin.count() >= lin.count(LineageKind.INVOCATION) >= 1

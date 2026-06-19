# Agent Eddy Router — Design & Architecture

Routing-centric orchestration runtime for agentic data integration.
Target: Python 3.11+, dataclasses + `typing.Protocol`, stdlib-only core (optional adapters
for OpenRouter / Omnigent). Status: **design only** — no code written yet; §13 is the build
plan.

---

## 1. Overview & motivation

For two decades the dominant data-integration execution model has been the **static
pipeline**: a user specifies operators on a canvas or in SQL, the tooling compiles a plan,
and the plan runs. Azure Data Factory, Google Dataform, AWS Glue, and Databricks LakeFlow all
operationalize this model and share its core constraint — *the plan commits before execution
begins*.

That constraint is now untenable for three converging reasons:

1. **Agentic integration is already happening, unsafely.** Analysts and engineers perform
   schema alignment, multi-source reconciliation, and quality repair by conversing with LLMs —
   producing outputs that are unpredictable across runs, ungoverned w.r.t. data policy, and
   untraceable w.r.t. lineage.
2. **MCP connectivity without adaptive orchestration amplifies brittleness.** An LLM that
   selects among integration tools with no structured execution layer is doing ad-hoc routing
   with no correctness guarantees, no backpressure awareness, and no multi-objective control.
3. **The market structure confirms the gap.** The cautions Gartner attaches to every
   data-integration Leader — intermediate quality feedback absent, observability fragmented,
   operator selection fixed — are symptoms of one missing component: a **runtime routing layer
   that observes multi-dimensional performance signals and selects operators adaptively from a
   Pareto-optimal set**.

**Thesis.** Agent Eddy is that layer. It routes each work item *online* from live `(Q, C, T,
R)` signals (adaptive query processing), restricts selection to the *non-dominated skyline*
(Pareto query theory) rather than a fixed-weight sum, enforces *correctness and governance*
through eligibility masks even under aggressive reordering, and records *per-routing-decision
lineage* so the whole run is observable and replayable.

---

## 2. Positioning

Agent Eddy is an **orchestration-layer** component. It does not replace the model router below
it or the agent harness beside it; it is the adaptive multi-objective routing layer both lack.

```
        Agent Eddy  (adaptive Pareto routing · masks · lineage)
              │ routes over                 │ consumes as operators
              ▼                             ▼
   Operator pool                      Omnigent meta-harness
   · simulated                        · YAML agents
   · OpenRouter-backed LLM            · pluggable harnesses
     (model/provider routing)         · server/agent/session policy
```

| Layer | System | Routes on | Granularity | What it does *not* do |
|---|---|---|---|---|
| Model/provider | **OpenRouter** | cost, availability (fallback) | one model call | multi-step orchestration; latency/risk objectives; correctness masks |
| Execution + governance | **Omnigent** | supervisor/delegation | one agent invocation | Pareto-adaptive *mid-run* operator selection |
| **Orchestration (this)** | **Agent Eddy** | live `(Q,C,T,R)` skyline | one work-item step | execute models/agents itself — it *routes to* them |

The seam is uniform: an OpenRouter model and an Omnigent agent are both just `OperatorBackend`
implementations (§9). Governance composes downward — Agent Eddy's gate chain and Omnigent's
policy levels both apply, and the stricter decision wins.

**Routing in the literature (pointer).** Prior LLM routers — RouterBench, RouteLLM,
MasRouter — evaluate a *static* pool with *convex-hull* analysis over *two* objectives. Agent
Eddy is online, four-objective, and non-convex-aware. The evaluation gap and how to close it
is in [`EVALUATION.md`](EVALUATION.md).

---

## 3. Conceptual model

Agent Eddy lifts the Eddies + skyline machinery from relational query processing into
multi-agent orchestration.

| Database concept | Orchestration analogue |
|---|---|
| Tuple | Work item (subtask instance) |
| Operator | Agent operator (DI tool, LLM, verifier) |
| Eddy router | Agent Eddy dispatcher |
| Ready / done bits | Eligibility / completion masks |
| Selectivity | Quality gain or error reduction |
| Moments of symmetry | Interruptible / reorderable steps |
| Skyline | Pareto-optimal agent set `Sky(w)` |
| Dominance | Multi-objective agent superiority |
| Threshold algorithm | Completion confidence bound `φ(w)` |

### The five invariants

The runtime is correct iff, every tick:

1. **Eligibility.** If operator `aᵢ` is selected for item `w`, then `ready_i(w)=1 ∧
   done_i(w)=0` at selection time.
2. **Monotone completion.** `done` bits move monotonically toward completion unless an
   explicit, lineage-logged rollback occurs.
3. **Safe emission.** Emitting an item implies its `required` done-mask is satisfied *and* all
   safety gates currently pass.
4. **Skyline selection.** The router only selects from `Sky(w) ∩ E(w)`; a demonstrably
   dominated operator is never chosen (pending performance recovery).
5. **Auditable lineage.** Every routing decision, invocation, and tool call is appended to
   `L(w)` with timestamps, cost, and outcome.

These map directly to tests (§14).

---

## 4. Architecture

Proposed package layout. Dependency direction is strictly **inward**: `operators/`,
`adapters/`, `examples/` depend on the inner packages; inner packages never import outward.

```
eddies-router/
├── pyproject.toml                  # core has zero runtime deps; adapters are extras
├── eddy/
│   ├── __init__.py                 # public API (WorkItem, Eddy, build_runtime)
│   ├── core/
│   │   ├── workitem.py             # WorkItem, ItemStatus, Signals, masks
│   │   ├── operator.py             # Operator Protocol, OperatorResult, side-effect/idempotency
│   │   ├── profile.py              # OperatorProfile, QCTR vector, EWMA update
│   │   ├── lineage.py              # LineageRecord, LineageEvent (append-only)
│   │   ├── budget.py               # BudgetTracker (token+dollar, dynamic β)
│   │   └── ids.py                  # monotonic / replay-id generation
│   ├── skyline/
│   │   ├── dominance.py            # dominates(a,b) over (Q↑,C↓,T↓,R↓)
│   │   └── maintain.py             # SkylineMaintainer: full O(k²) + incremental O(k)
│   ├── routing/
│   │   ├── policy.py               # RoutingPolicy Protocol + reward()
│   │   ├── backpressure.py         # P1: backpressure-aware greedy
│   │   ├── lottery.py              # P2: windowed lottery (Eddies tickets)
│   │   └── bandit.py               # P3: contextual bandit (UCB / Thompson)
│   ├── governance/
│   │   ├── eligibility.py          # EligibilityEvaluator → ready-mask
│   │   ├── gates.py                # GovernanceGate Protocol + concrete gates
│   │   ├── permissions.py          # PermissionScope, allowlists, classification
│   │   └── approval.py             # human-in-the-loop as an operator
│   ├── runtime/
│   │   ├── eddy.py                 # Eddy router: Algorithm 1 main loop
│   │   ├── queue.py                # WorkQueue (FIFO + re-enqueue)
│   │   ├── registry.py             # OperatorRegistry: k operators, index map
│   │   └── completion.py           # CompletionPredicate φ(w) + emission gate
│   ├── adapters/
│   │   ├── base.py                 # OperatorBackend Protocol (the seam)
│   │   ├── simulated.py            # SimulatedBackend (default, zero-dep)
│   │   ├── openrouter.py           # OpenRouterBackend (optional; lazy requests)
│   │   └── omnigent.py             # OmnigentBackend (optional; YAML agent)
│   ├── observability/
│   │   ├── events.py               # structured event schema (JSON)
│   │   ├── sink.py                 # EventSink Protocol; JSONLSink, InMemorySink
│   │   └── frontier_log.py         # Pareto-frontier evolution artifact
│   ├── operators/                  # DI-domain operator library (simulated profiles)
│   │   ├── schema.py               # Profiler, Matcher, Mapper[W]
│   │   ├── identity.py             # Blocker, Matcher, Clusterer
│   │   ├── quality.py              # Profiler, Repairer[W], Detector
│   │   ├── enrichment.py           # Lookup, LLMEnricher, APIEnricher
│   │   └── governance_ops.py       # Provenance, PolicyChecker, ConfAggregator
│   └── examples/
│       └── customer_master.py      # worked episode (§12)
└── tests/                          # invariant-focused (§14)
```

---

## 5. Core data models

Python 3.11+, `dataclasses` + `typing.Protocol`. Signatures below are the intended Phase-0
API.

### 5.1 QCTR vector and operator estimates (`core/profile.py`)

```python
@dataclass(frozen=True, slots=True)
class QCTR:
    """Expected reward dimensions. Q is maximized; C, T, R are minimized."""
    Q: float  # quality / utility,  higher better, in [0,1]
    C: float  # cost (dollars + token-normalized), lower better
    T: float  # latency (seconds),  lower better
    R: float  # risk (impact·p_fail), lower better, in [0,1]

    def as_tuple_for_dominance(self) -> tuple[float, float, float, float]:
        # negate Q so ALL dims become "lower is better" for a uniform test
        return (-self.Q, self.C, self.T, self.R)


@dataclass(slots=True)
class OperatorProfile:
    """Per-operator EWMA estimate of (Q,C,T,R), updated after each invocation."""
    operator_id: str
    est: QCTR                  # current (Q̂, Ĉ, T̂, R̂)
    alpha: float = 0.3         # EWMA smoothing factor
    n_invocations: int = 0
    reward_mean: float = 0.0   # bandit bookkeeping (P3)
    reward_m2: float = 0.0     # Welford running variance numerator

    def update_estimate(self, obs: QCTR) -> None:
        a, e = self.alpha, self.est
        self.est = QCTR(Q=(1-a)*e.Q + a*obs.Q, C=(1-a)*e.C + a*obs.C,
                        T=(1-a)*e.T + a*obs.T, R=(1-a)*e.R + a*obs.R)
        self.n_invocations += 1
```

### 5.2 Work item, masks, state (`core/workitem.py`)

```python
class ItemStatus(str, Enum):
    ACTIVE = "active"; EMITTED = "emitted"; FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"

@dataclass(slots=True)
class Signals:
    quality: dict[str, float] = field(default_factory=dict)   # {"completeness":0.8}
    metadata: dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass(slots=True)
class WorkItem:
    item_id: str
    task_type: str                         # "schema_profile", "record_match", ...
    payload: dict[str, object]
    signals: Signals = field(default_factory=Signals)
    ready: list[int] = field(default_factory=list)     # ready(w) ∈ {0,1}^k
    done: list[int]  = field(default_factory=list)     # done(w)  ∈ {0,1}^k
    required: list[int] = field(default_factory=list)  # steps required for φ
    status: ItemStatus = ItemStatus.ACTIVE
    lineage_id: str = ""
    parent_id: str | None = None
    sla_tier: str = "standard"             # routing context
    safety_tier: str = "normal"
    data_classification: str = "internal"  # governance-gate input
```

Masks are length-`k` 0/1 lists aligned to the registry's operator index, so dominance and
eligibility stay cheap and invariants are trivially checkable. `required` decouples "this
operator ran" from "this operator was required to run before emission".

### 5.3 Operator Protocol and result (`core/operator.py`)

```python
class SideEffectClass(str, Enum):
    READ = "read"; WRITE = "write"; TRANSMIT = "transmit"  # transmit = external egress

class Idempotency(str, Enum):
    IDEMPOTENT = "idempotent"; AT_MOST_ONCE = "at_most_once"; NON_IDEMPOTENT = "non_idempotent"

@dataclass(slots=True)
class Precondition:
    requires_done: list[str] = field(default_factory=list)     # operator_ids that must be done
    requires_signals: list[str] = field(default_factory=list)  # signal keys that must exist
    task_types: set[str] = field(default_factory=set)          # applies only to these
    requires_permission: set[str] = field(default_factory=set) # scope names needed

@dataclass(slots=True)
class OperatorResult:
    item: WorkItem
    observed: QCTR                              # measured (Q,C,T,R) for EWMA + reward
    sets_done: list[str] = field(default_factory=list)
    spawned: list[WorkItem] = field(default_factory=list)  # split outputs
    succeeded: bool = True

@runtime_checkable
class Operator(Protocol):
    operator_id: str
    side_effect: SideEffectClass
    idempotency: Idempotency
    precondition: Precondition
    @property
    def profile(self) -> OperatorProfile: ...
    def declared_profile(self) -> QCTR: ...               # prior to seed the EWMA
    def execute(self, item: WorkItem, ctx: "ExecContext") -> OperatorResult: ...
```

### 5.4 Lineage (`core/lineage.py`) — Invariant 5

```python
class LineageKind(str, Enum):
    ROUTING_DECISION="routing_decision"; INVOCATION="invocation"; TOOL_CALL="tool_call"
    MASK_UPDATE="mask_update"; EMISSION="emission"; FAILURE="failure"; GATE="gate"

@dataclass(frozen=True, slots=True)
class LineageEvent:
    seq: int; kind: LineageKind; item_id: str; replay_id: str
    ts: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)            # JSON-serializable

@dataclass(slots=True)
class LineageRecord:
    """Append-only L(w). Only public mutator is append()."""
    item_id: str
    events: list[LineageEvent] = field(default_factory=list)
    _seq: int = 0
    def append(self, kind: LineageKind, replay_id: str, **detail) -> LineageEvent:
        ev = LineageEvent(self._seq, kind, self.item_id, replay_id, detail=detail)
        self.events.append(ev); self._seq += 1
        return ev
```

### 5.5 Budget with dynamic β (`core/budget.py`)

```python
@dataclass(slots=True)
class BudgetTracker:
    dollar_budget: float; token_budget: int
    dollars_spent: float = 0.0; tokens_spent: int = 0
    alpha: float = 1.0; beta_floor: float = 0.5

    @property
    def tokens_remaining(self) -> int: return max(1, self.token_budget - self.tokens_spent)

    @property
    def beta(self) -> float:
        # β rises as budget depletes: frontier shifts toward cheaper operators
        return self.beta_floor / max(1e-3, self.tokens_remaining / self.token_budget)

    def cost(self, dollars: float, tokens: int) -> float:
        return self.alpha*dollars + self.beta*(tokens / self.tokens_remaining)

    def charge(self, dollars: float, tokens: int) -> None:
        self.dollars_spent += dollars; self.tokens_spent += tokens

    def exhausted(self) -> bool:
        return self.dollars_spent >= self.dollar_budget or self.tokens_spent >= self.token_budget
```

The router recomputes each candidate's `C` dimension via `cost(...)` at routing time, so the
Pareto frontier literally shifts toward cheaper operators as budget depletes.

---

## 6. Skyline maintenance

### 6.1 Dominance (`skyline/dominance.py`)

```python
def dominates(a: QCTR, b: QCTR, eps: float = 1e-9) -> bool:
    """a dominates b iff no-worse on all dims and strictly better on ≥1.
    as_tuple_for_dominance() negates Q so every dim is 'lower is better'."""
    ta, tb = a.as_tuple_for_dominance(), b.as_tuple_for_dominance()
    no_worse  = all(x <= y + eps for x, y in zip(ta, tb))
    strictly  = any(x <  y - eps for x, y in zip(ta, tb))
    return no_worse and strictly
```

### 6.2 Maintainer (`skyline/maintain.py`) — full + incremental

#### Correctness note — the incremental path has a known bug in early drafts

The naive O(k) incremental routine is BUGGY on the degradation path.  Concrete
counterexample:

```
A = (Q=0.9, C=0.1, T=0.1, R=0.1)
B = (Q=0.5, C=0.1, T=0.1, R=0.1)
```

After `full_recompute`: `members = {A}` (A dominates B).  A's estimate then
degrades to `(Q=0.4, ...)`.  Correct skyline: `{B}`.  Buggy O(k) routine
returns `{}` — it discards A (now dominated) but never re-admits B (which was
excluded only because A dominated it).

Root cause: the fast path only handles the *improvement* case (changed op
becomes/stays non-dominated and only adds dominations); it does not handle the
*degradation* case where the changed op was dominating other ops that may now be
non-dominated.

#### Corrected implementation

```python
@dataclass(slots=True)
class SkylineMaintainer:
    members: set[str] = field(default_factory=set)
    profiles: dict[str, OperatorProfile] = field(default_factory=dict)
    _prev_estimates: dict[str, QCTR] = field(default_factory=dict)  # snapshot for improvement detection

    def full_recompute(self, eligible_ids: list[str]) -> set[str]:    # O(k²)
        sky = {i for i in eligible_ids
               if not any(j != i and dominates(self.profiles[j].est, self.profiles[i].est)
                          for j in eligible_ids)}
        self.members = sky
        for i in eligible_ids:                                        # snapshot for incremental
            self._prev_estimates[i] = self.profiles[i].est
        return sky

    def on_estimate_update(self, changed_id: str, eligible_ids: list[str]) -> set[str]:
        """Incremental update after ONE operator's estimate changed.

        Complexity:
          O(k)   — improvement path (changed op got better or was not a member)
          O(k²)  — degradation path (changed op degraded or became dominated while
                   a member, because previously dominated ops may need re-admission)
        """
        new_est = self.profiles[changed_id].est
        old_est = self._prev_estimates.get(changed_id)
        self._prev_estimates[changed_id] = new_est

        was_member = changed_id in self.members

        # Detect improvement (new estimate ≤ old on every negated dimension)
        if old_est is not None:
            old_t = old_est.as_tuple_for_dominance()
            new_t = new_est.as_tuple_for_dominance()
            improved = all(n <= o + 1e-15 for n, o in zip(new_t, old_t))
        else:
            improved = True

        is_dominated = any(
            j != changed_id and dominates(self.profiles[j].est, new_est)
            for j in eligible_ids
        )

        if is_dominated:
            if was_member:
                # DEGRADATION: was a member, now dominated.  Re-admit candidates.
                return self.full_recompute(eligible_ids)         # O(k²)
            return self.members                                  # not a member, no change

        # changed_id is non-dominated
        if improved or not was_member:
            # O(k) safe path: adding/improving a non-dominated op can only evict
            # existing members it now dominates; it cannot release excluded ops.
            self.members.add(changed_id)
            for m in list(self.members):
                if m != changed_id and dominates(new_est, self.profiles[m].est):
                    self.members.discard(m)
            return self.members
        else:
            # Degraded but still non-dominated: may have stopped dominating some
            # previously excluded ops → fall back to full_recompute.
            return self.full_recompute(eligible_ids)             # O(k²)
```

#### Honest complexity

| Case | Trigger | Complexity |
|---|---|---|
| New non-member becomes non-dominated | EWMA improvement | O(k) |
| Existing member improves | EWMA improvement | O(k) |
| Existing member degrades but stays non-dominated | EWMA degradation | O(k²) |
| Existing member degrades and becomes dominated | EWMA degradation | O(k²) |
| Non-member stays dominated | EWMA change | O(k) |

The O(k) fast path covers the most frequent hot-path case (just observed a good
result; EWMA moves upward).  Degradation is rarer and triggers the O(k²) fallback.
The regression test `test_skyline_equivalence.py::TestABRegression` encodes the
counterexample above.

`full_recompute` runs when the *shape* of the eligible set changes (a new item with
a different `ready` mask) and always serves as the ground-truth oracle.

Membership is maintained **globally per operator**; eligibility is **per item**.  The
router takes `Sky(w) = {i ∈ members : eligible_i(w)}` each tick.

---

## 7. Routing policies

```python
@dataclass(slots=True)
class RoutingContext:
    item: WorkItem; budget: BudgetTracker
    queue_lengths: dict[str, int]; profiles: dict[str, OperatorProfile]

@runtime_checkable
class RoutingPolicy(Protocol):
    def select(self, candidates: list[str], ctx: RoutingContext) -> str: ...   # from Sky∩E
    def update(self, operator_id: str, reward: float, ctx: RoutingContext) -> None: ...

def reward(est: QCTR, budget: BudgetTracker, lam=1.0, mu=1.0, nu=1.0) -> float:
    return est.Q - lam*est.C - mu*est.T - nu*est.R          # r = Q − λC − μT − νR
```

- **P1 Backpressure-aware greedy.** `argmin (T̂(w,a) + β·q_a)` where `q_a` is queue length.
  Reacts to backlog, ignores exploration.
- **P2 Windowed lottery (Eddies-inspired).** Per-operator tickets `tᵢ ∝ exp(β·uᵢ)` on a
  sliding-window utility `uᵢ ← (1−ρ)uᵢ + ρ·r`; sample `∝ tᵢ` over `Sky(w) ∩ E(w)`.
  Probabilistic, naturally load-balancing, mirrors Eddies ticket scheduling.
- **P3 Contextual bandit.** Operators in `Sky(w)` are arms; context = `(task_type, sla_tier,
  safety_tier)` + current Pareto estimates; UCB bonus `c·√(ln N / nₐ)` or Thompson sampling
  `~ Normal(reward_mean, reward_var/nₐ)`. Explores within the skyline.

All three operate strictly on `candidates = Sky(w) ∩ E(w)` handed in by the router, so
**Invariant 4** is enforced upstream and policies cannot violate it. **Scalarized routing is
the special case**: P1/P3 maximizing `reward(...)` within a skyline tier with tier-specific
`λ, μ, ν`. Skyline restriction is preserved, so scalarization never reaches a dominated
operator — but it also never *misses* a non-convex-pocket operator the way unconstrained
scalarization does (see [`EVALUATION.md`](EVALUATION.md) Experiment 2).

---

## 8. Governance & eligibility

Eligibility is computed fresh per item per tick from three inputs: the operator's static
`Precondition`, the item's `done` mask + `signals`, and the active gate chain.

```python
@dataclass(slots=True)
class PermissionScope:
    granted: set[str] = field(default_factory=set)             # scopes the session holds
    tool_allowlist: set[str] = field(default_factory=set)
    domain_allowlist: set[str] = field(default_factory=set)
    path_allowlist: set[str] = field(default_factory=set)
    transmit_allowed_classes: set[str] = field(default_factory=lambda: {"public","internal"})

@dataclass(frozen=True, slots=True)
class GateResult:
    allowed: bool; reason: str = ""; requires_approval: bool = False

@runtime_checkable
class GovernanceGate(Protocol):
    def evaluate(self, item: WorkItem, op: Operator, scope: PermissionScope) -> GateResult: ...
```

Concrete gates (composed as a chain; any deny zeroes the operator's `ready` bit):

- **PermissionGate** — operator's `requires_permission ⊆ scope.granted`.
- **WriteCapabilityGate** — `WRITE` operators disabled (`ready=0`) until `"write"` granted;
  denial flags `requires_approval`.
- **DataClassificationGate** — `TRANSMIT` operators blocked when
  `item.data_classification ∉ scope.transmit_allowed_classes` (regulated data egress).
- **ToolArgGate** — schema-validate tool calls; allowlist domains / paths / operations
  (threat-model defense).

The ready-mask computation:

```python
@dataclass(slots=True)
class EligibilityEvaluator:
    registry: "OperatorRegistry"; scope: PermissionScope; gates: list[GovernanceGate]

    def compute_ready_mask(self, item: WorkItem) -> list[int]:
        mask = [0] * len(self.registry)
        for i, op in enumerate(self.registry.operators):
            if self._preconditions_met(op, item) and self._gates_allow(op, item):
                mask[i] = 1
        return mask
```

**Where things live.** Prerequisites are declared once per operator (`Precondition`).
Permission scope + allowlists are session-scoped (`PermissionScope`). Side-effect class +
idempotency are operator attributes. The `done` mask is the only per-item mutable prerequisite
state, and it moves monotonically (**Invariant 2**) because the runtime only *sets* bits via
`OperatorResult.sets_done` — the sole way to clear a bit is an explicit rollback API that
re-logs to lineage.

**Human-in-the-loop** (`governance/approval.py`) is modeled as a special operator whose
`ready` bit is 1 only when `item.status == AWAITING_APPROVAL`; its `execute` records the
approval and flips the blocked operator's gate state, keeping approvals inside the same
routing fabric.

**Threat model.** Prompt-injection containment (segregate instruction vs data channels; label
retrieved content; disallow tool control via retrieved text); tool-argument validation
(schema + allowlists); least privilege (per-agent scopes, default read-only, elevate only
after approval); auditability (immutable lineage with replay IDs; Pareto curve logged per
run).

---

## 9. Extension points — the adapter seam

A single Protocol separates routing logic from execution substrate. The router never knows
which backend an operator uses.

```python
@dataclass(slots=True)
class ExecContext:
    budget: BudgetTracker; scope: PermissionScope
    lineage: LineageRecord; replay_id: str; rng: random.Random

@runtime_checkable
class OperatorBackend(Protocol):
    def run(self, item: WorkItem, ctx: ExecContext) -> OperatorResult: ...
```

A concrete operator owns one backend and forwards `execute` to it. Three backends:

- **`SimulatedBackend`** (default, zero-dep) — draws `(Q,C,T,R)` from a declared base ±
  Gaussian noise, charges the budget, optionally mutates `signals`/`done` via an `effect`
  callable. Enables fully offline, deterministic (seeded) runs and the entire eval harness.
- **`OpenRouterBackend`** (optional, lazy `import requests`) — POSTs to OpenRouter's
  OpenAI-compatible endpoint; reads `usage.total_tokens` / `usage.cost`; charges budget; logs
  a `TOOL_CALL`. Agent Eddy sits above OpenRouter — model/provider fallback happens inside the
  backend; Pareto operator-layer routing happens above it.
- **`OmnigentBackend`** (optional) — invokes a YAML-defined Omnigent agent via a duck-typed
  client; maps the returned `(quality, cost, latency, risk)` back into a `QCTR`. Omnigent's
  own spend caps / approvals enforce *underneath* the gate chain; the stricter wins.

Adding a real provider = a new `OperatorBackend` + a registry entry. No router changes.

---

## 10. Observability

```python
@dataclass(frozen=True, slots=True)
class Event:
    ts: float; kind: str; item_id: str; replay_id: str; payload: dict

@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...
# InMemorySink (tests/dashboards) · JSONLSink (immutable append-only audit log)
```

The runtime emits to both the per-item `LineageRecord` (replayable, item-scoped) and the
global `EventSink` (chronological, governance-facing). Each event carries a `replay_id` so any
decision can be deterministically re-executed under the seeded `rng`.

`FrontierLog` snapshots the skyline each tick — `{members, per-op (Q,C,T,R) estimates,
selected}`. `FrontierLog.dump()` is the **governance/monitoring artifact**: it shows the
Pareto frontier evolving — which operators entered/left the skyline, their estimates, and
which was chosen — making the "frontier shifts toward cheaper operators as budget depletes"
claim *observable*, not merely asserted.

---

## 11. The main loop (Algorithm 1)

```python
def run(self) -> None:
    while not self.queue.empty() and not self.budget.exhausted():
        w = self.queue.pop(); self._tick += 1

        w.ready = self.eligibility.compute_ready_mask(w)        # ready_i(w)
        eligible = self.registry.eligible_ids(w)                # ready=1 ∧ done=0
        self.skyline.full_recompute(eligible)                  # shape changed → recompute
        candidates = [a for a in eligible if a in self.skyline.members]   # Sky(w) ∩ E(w)

        if not candidates:
            self._emit(w) if self.completion.satisfied(w) else self._fail(w, "skyline_empty")
            continue

        lin = self.lineage_store[w.lineage_id]; replay_id = new_replay_id()
        ctx = RoutingContext(w, self.budget, self.queue.lengths(), self.registry.profiles)
        op_id = self.policy.select(candidates, ctx)            # Invariant 4
        assert w.ready[self.registry.index(op_id)] == 1        # Invariant 1
        lin.append(ROUTING_DECISION, replay_id, chosen=op_id, skyline=candidates)
        self.frontier_log.record(self._tick, w, candidates, self.registry.profiles, op_id)

        op = self.registry.get(op_id)
        ectx = ExecContext(self.budget, self.eligibility.scope, lin, replay_id, self.rng)
        result = op.execute(w, ectx)                           # may call tools/LLMs

        self._apply_done(result.item, result.sets_done)        # Invariant 2 (set-only)
        r = reward(result.observed, self.budget)
        self.registry.profiles[op_id].update_estimate(result.observed)
        self.skyline.on_estimate_update(op_id, eligible)       # O(k) incremental
        self.policy.update(op_id, r, ctx)                      # θ ← π.update(...)

        for child in result.spawned:
            self._enqueue(child)
        self._emit(result.item) if self.completion.satisfied(result.item) \
            else self._enqueue(result.item)                    # Invariant 3 gated in _emit
```

`CompletionPredicate.satisfied` enforces **Invariant 3**: true only if every `required` bit is
`done` *and* all safety gates currently pass. `_emit` re-checks the emission gate and logs
`EMISSION`; a gate failure at emission parks the item in `AWAITING_APPROVAL` rather than
emitting.

---

## 12. Worked example — customer-master integration

Goal: integrate `crm`, `ecomm`, `support` into `customer_master` with lineage certification.
Operator pool (k≈12), declared priors and preconditions:

| Operator (id) | task_types | requires_done | side-effect | declared (Q,C,T,R) |
|---|---|---|---|---|
| `schema.profiler` | schema_profile | — | read | (.70, .02, .30, .05) |
| `schema.matcher` | schema_match | schema.profiler | read | (.75, .03, .50, .10) |
| `schema.mapper` | schema_map | schema.matcher | **write** | (.80, .04, .60, .30) |
| `identity.blocker` | record_block | schema.mapper | read | (.60, .05, .40, .10) |
| `identity.matcher` | record_match | identity.blocker | read | (.78, .08, .70, .15) |
| `identity.clusterer` | record_cluster | identity.matcher | read | (.82, .06, .60, .10) |
| `quality.profiler` | quality_profile | schema.mapper | read | (.70, .02, .30, .05) |
| `quality.repairer` | quality_repair | quality.profiler | **write** | (.75, .05, .50, .35) |
| `enrich.lookup` | enrich | identity.clusterer | read | (.65, .01, .20, .05) |
| `enrich.llm` | enrich | identity.clusterer | **transmit** | (.85, .09, .90, .40) |
| `gov.provenance` | certify | identity.clusterer | read | (.90, .02, .20, .02) |
| `gov.policy_checker` | certify | — | read | (.95, .01, .10, .00) |

**Flow.**

1. Seed one root work item per source-pair task. Only `schema.profiler` is eligible (no
   prerequisites). `Sky = {schema.profiler}`; routed; sets `done[schema.profiler]=1`,
   populates `signals.metadata["schema"]`.
2. `schema.matcher` becomes eligible. Routed → done. Then `schema.mapper` (**write**) is
   gated by `WriteCapabilityGate`: without `"write"` in scope, `ready=0`, item parks in
   `AWAITING_APPROVAL`, and the approval operator becomes the eligible path. After approval,
   `schema.mapper` runs and writes the mapping.
3. Identity and quality branches open in parallel after `schema.mapper`. The skyline now holds
   several non-dominated operators (`identity.blocker` cheap/low-Q vs `quality.profiler`
   cheap/fast); the policy picks within `Sky(w)`. After clustering, `enrich.llm` (transmit,
   expensive, high-Q) and `enrich.lookup` (cheap, lower-Q) sit on the skyline together —
   neither dominates — and P3 explores. If `data_classification == "regulated"`, the
   `DataClassificationGate` zeroes `enrich.llm`'s `ready` bit, leaving only `enrich.lookup`.
4. As budget depletes, `BudgetTracker.beta` rises → the `C` dimension of token-heavy operators
   (`enrich.llm`, `identity.matcher`) grows → they become dominated and **leave the skyline** →
   the frontier shifts to `enrich.lookup` / `identity.blocker`. Visible in `FrontierLog.dump()`.
5. `required = {schema.mapper, identity.clusterer, quality.profiler, gov.provenance,
   gov.policy_checker}`. `φ(w)` fires only when all required bits are done and safety gates
   pass; `gov.provenance` + `gov.policy_checker` run last, then `_emit` writes `customer_master`
   and appends an `EMISSION` event with the full replay trail.

This exercises skyline restriction, governance-driven mask zeroing, write-gating with
human-in-the-loop, budget-driven frontier shift, and audited emission — all from one loop.

---

## 13. Build phases

| Phase | Deliverable | Modules |
|---|---|---|
| **0** | Minimal runnable core (Algorithm 1 end-to-end, 3-op linear chain) | `core/*`, `skyline/dominance`, `skyline/maintain` (full only), `routing/policy`+`backpressure`, `runtime/*`, `adapters/simulated` |
| **1** | Governance + eligibility | `governance/*`; wire write-gating + data-classification + human-approval into the loop |
| **2** | Full skyline + policies | incremental `on_estimate_update`; P2 lottery, P3 bandit; EWMA + dynamic-β cost in routing |
| **3** | Observability | `observability/*`; `FrontierLog`, JSONL audit sink, replay |
| **4** | DI operator library + worked episode | `operators/*`, `examples/customer_master` |
| **5** | Real backends | `adapters/openrouter`, `adapters/omnigent` behind optional extras |

---

## 14. Testing strategy (invariant-focused)

- **`test_dominance.py`** — Q-maximize / C,T,R-minimize semantics; equal vectors don't
  dominate; strict-on-one; epsilon ties.
- **`test_skyline_maintain.py`** — property test: `on_estimate_update` after a single estimate
  change ≡ `full_recompute` (incremental = oracle); membership-bit consistency; empty-eligible
  → empty skyline.
- **`test_masks_monotonic.py`** (Inv. 2) — random operator graphs; `done` bits non-decreasing
  across ticks; rollback is the only clearing path and logs a lineage event.
- **`test_eligibility_governance.py`** (Inv. 1 + §8) — selected op always `ready=1 ∧ done=0`;
  write op stays `ready=0` without `"write"`; transmit op `ready=0` for regulated data;
  approval flips it.
- **`test_safe_emission.py`** (Inv. 3) — never emit unless all `required` done + gates pass;
  failing gate at emission parks in `AWAITING_APPROVAL`.
- **`test_budget.py`** (§5.5) — `dollars_spent ≤ budget`, `tokens_spent ≤ budget`; loop halts
  on exhaustion; `beta` increases monotonically; an early-skyline high-token op leaves the
  skyline once `beta` crosses the dominance threshold.
- **`test_policies.py`** — each policy only returns an id from its `candidates` (Inv. 4);
  lottery distribution ∝ `exp(β·u)` within tolerance; UCB picks the under-sampled arm on ties.
- **`test_main_loop_invariants.py`** — fuzz random graphs × policies; assert all five
  invariants every tick; lineage event count ≥ invocation count (Inv. 5).
- **`test_customer_master_episode.py`** — golden seed → expected emission, required bits done,
  non-empty provenance, `FrontierLog` showing ≥1 operator leaving the skyline as budget
  depletes.

Determinism via a seeded `random.Random` threaded through `ExecContext` and the policies;
`replay_id` enables exact re-execution.

---

## 15. Open questions & future work

- **Correlated-objective skylines.** `(Q,C,T,R)` are not independent (higher Q often costs
  more tokens); skyline maintenance under correlated noise needs study.
- **Formal safety under adversarial injection.** Guarantees for the gate chain when an
  attacker controls retrieved content / tool outputs (threat model §8).
- **Automatic operator decomposition.** Lowering a LakeFlow-style declarative pipeline into
  registered operators automatically.
- **Evaluation.** The full publication evaluation — metrics, baselines, and three experiments
  (robustness under non-stationarity; skyline vs. scalarization on non-convex frontiers;
  governance-correctness under reordering) — is in [`EVALUATION.md`](EVALUATION.md), with the
  paper's DI-1..DI-6 benchmarks and the Pareto-regret metric.

---

## Appendix — paper-coverage checklist

| Paper concept | Section here |
|---|---|
| Eddy router / Algorithm 1 | §3, §11 |
| Work-item state, ready/done masks, φ | §5.2, §8, §11 |
| Skyline / Pareto dominance + dynamic maintenance | §6 |
| Reward `Q−λC−μT−νR`, scalarization as special case | §7 |
| Routing policies P1/P2/P3 | §7 |
| Five invariants | §3, enforced in §6/§8/§11, tested §14 |
| Governance gates, classification, write-gating, HITL | §8 |
| Token efficiency / dynamic β | §5.5, §12 |
| Threat model & safety | §8 |
| DI operator taxonomy | §4 (`operators/`), §12 |
| Worked customer-master episode + lineage | §12 |
| Lineage / observability / Pareto-frontier artifact | §5.4, §10 |
| OpenRouter / Omnigent positioning | §2, §9 |
| Evaluation suite, Pareto regret | §15 → `EVALUATION.md` |

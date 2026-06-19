# Agent Eddy Router — Diagrams

Mermaid + ASCII diagrams for [`DESIGN.md`](DESIGN.md). Mermaid blocks render on GitHub and in
most Markdown previewers.

---

## 1. Layering — where Agent Eddy sits

```mermaid
flowchart TD
    intent["User integration intent<br/>(sources, target schema, quality, budget, governance)"]
    eddy["<b>Agent Eddy</b><br/>adaptive Pareto routing · eligibility masks · lineage"]
    pool["Operator pool"]
    sim["Simulated operators"]
    orr["OpenRouter-backed LLM operators<br/>(model/provider routing + fallback)"]
    omni["Omnigent agents<br/>(YAML · harnesses · server/agent/session policy)"]

    intent --> eddy
    eddy -->|routes each work item to| pool
    pool --> sim
    pool --> orr
    eddy -->|consumes agents as operators| omni
```

- OpenRouter = model/provider layer (below). Omnigent = execution + governance substrate
  (beside). Agent Eddy = the adaptive multi-objective routing layer both lack.

---

## 2. Main loop (Algorithm 1)

```mermaid
flowchart TD
    pop["pop work item w"] --> ready["compute ready(w)<br/>preconditions + governance gates"]
    ready --> elig["E(w) = ready=1 AND done=0"]
    elig --> sky["Sky(w) = skyline(E(w))<br/>non-dominated over (Q,C,T,R)"]
    sky --> empty{"Sky(w) empty?"}
    empty -->|yes| phi{"phi(w) satisfied?"}
    phi -->|yes| emit1["emit (Inv. 3)"]
    phi -->|no| fail["mark_failed"]
    empty -->|no| select["policy.select(Sky∩E)  (Inv. 1, 4)"]
    select --> log["append ROUTING_DECISION to lineage (Inv. 5)<br/>snapshot FrontierLog"]
    log --> exec["operator.execute(w)  — may call tools/LLMs"]
    exec --> upd["apply sets_done (Inv. 2)<br/>update EWMA estimate<br/>on_estimate_update (O(k))<br/>policy.update(theta)"]
    upd --> done{"phi(w') satisfied?"}
    done -->|yes| emit2["emit (Inv. 3)"]
    done -->|no| push["re-enqueue w'"]
    push --> pop
```

---

## 3. Skyline over (Q, C, T, R)

Two dimensions shown (Quality ↑ vs. Cost ↓); the runtime maintains all four. An operator is on
the skyline iff no eligible operator is no-worse on all dims and strictly better on one.

```
 Quality Q
   ^
   |            x A3 (LLM-fast)         . = dominated (off skyline)
   |          /                          x = skyline member
   |        x A2 (embed)
   |       /
   |     x A1 (rule-based)
   |    /        . A4 (LLM-slow: same Q,C as A3 but worse T -> dominated)
   |   /
   +----------------------------------> Cost C  (lower is better -> left)

 Sky(w) = { A1, A2, A3 };   A4 excluded (dominated by A3).
 As budget depletes, beta rises -> C of token-heavy A3 grows -> A3 can leave the skyline.
```

```mermaid
flowchart LR
    subgraph eligible["Eligible operators E(w)"]
        A1["A1 rule-based<br/>Q.71 C.02 T120 R.01"]
        A2["A2 embed<br/>Q.84 C.15 T340 R.02"]
        A3["A3 LLM-fast<br/>Q.93 C1.20 T2100 R.05"]
        A4["A4 LLM-slow<br/>Q.93 C1.20 T6800 R.05"]
    end
    A3 -->|dominates| A4
    sky["Sky(w) = {A1, A2, A3}"]
    A1 --> sky
    A2 --> sky
    A3 --> sky
```

---

## 4. Core data model relationships

```mermaid
classDiagram
    class WorkItem {
        item_id, task_type, payload
        signals: Signals
        ready[], done[], required[]  (len k)
        status, lineage_id, sla_tier
        data_classification
    }
    class Operator {
        <<Protocol>>
        operator_id
        side_effect, idempotency
        precondition: Precondition
        execute(item, ctx) OperatorResult
    }
    class OperatorProfile {
        est: QCTR  (EWMA)
        n_invocations
        update_estimate(obs)
    }
    class QCTR {
        Q (max) · C, T, R (min)
        as_tuple_for_dominance()
    }
    class SkylineMaintainer {
        members: set
        full_recompute() O(k^2)
        on_estimate_update() O(k)
    }
    class RoutingPolicy {
        <<Protocol>>
        select(candidates, ctx)
        update(op_id, reward, ctx)
    }
    class OperatorBackend {
        <<Protocol>>
        run(item, ctx) OperatorResult
    }
    class LineageRecord {
        events[] (append-only)
        append(kind, replay_id)
    }
    class BudgetTracker {
        dynamic beta
        cost(), charge(), exhausted()
    }

    Operator --> OperatorProfile : has
    OperatorProfile --> QCTR : est
    Operator --> OperatorBackend : delegates execute
    SkylineMaintainer --> OperatorProfile : reads
    SkylineMaintainer --> QCTR : dominance
    RoutingPolicy --> QCTR : reward Q-lC-mT-nR
    WorkItem --> LineageRecord : lineage_id
    RoutingPolicy --> BudgetTracker : recompute C
```

---

## 5. Governance gate chain (computing ready(w))

```mermaid
flowchart LR
    op["operator a_i"] --> pc{"preconditions met?<br/>requires_done ⊆ done<br/>requires_signals present"}
    pc -->|no| zero["ready_i = 0"]
    pc -->|yes| g1{"PermissionGate"}
    g1 -->|deny| zero
    g1 -->|allow| g2{"WriteCapabilityGate"}
    g2 -->|deny → approval| approval["status = AWAITING_APPROVAL<br/>ready_i = 0"]
    g2 -->|allow| g3{"DataClassificationGate"}
    g3 -->|deny → approval| approval
    g3 -->|allow| g4{"ToolArgGate"}
    g4 -->|deny| zero
    g4 -->|allow| one["ready_i = 1"]
```

# Agent Eddy Router

A **routing-centric orchestration runtime** for agentic data integration — the missing
adaptive routing layer between a queue of *work items* (subtasks) and a pool of *agent
operators* (data-integration tools, LLMs, verifiers).

Based on the position paper *Beyond Eddies: Agentic Data Integration with intelligent
orchestration* (Wee Hyong Tok, 2026). This repository currently contains the **design and
architecture** for the runtime; the implementation is described as a phased build in
[`docs/DESIGN.md`](docs/DESIGN.md) §13 and is not yet written.

## The idea in one paragraph

Today's data-integration platforms commit to a static execution plan at design time. That is
the wrong execution model for agentic workloads, where operator quality, cost, latency, and
risk all move during a run. Agent Eddy evolves two lines of database research into the
agentic setting:

- **Eddies / adaptive query processing** — route each work item *online* from live
  performance signals instead of following a precomputed plan.
- **Skyline (Pareto) queries** — restrict operator selection to the **non-dominated
  frontier** over Quality / Cost / Latency / Risk `(Q, C, T, R)`, instead of collapsing them
  into a fixed-weight sum.

It adds **eligibility masks** (semantic preconditions + governance/permission gates enforced
even under aggressive reordering) and **per-routing-decision lineage** for auditability.

## Where it sits relative to OpenRouter and Omnigent

```
            ┌─────────────────────────────────────────────┐
            │     Agent Eddy  —  adaptive Pareto routing    │   ← this project
            │  (skyline selection · eligibility masks ·     │
            │   live (Q,C,T,R) estimates · lineage)         │
            └───────────────┬──────────────────┬────────────┘
                            │ routes over      │ consumes as operators
                            ▼                  ▼
        ┌───────────────────────────┐  ┌──────────────────────────────┐
        │  Operator pool            │  │  Omnigent  (meta-harness)      │
        │  · simulated operators    │  │  · YAML-defined agents         │
        │  · OpenRouter-backed LLMs │  │  · pluggable harnesses         │
        │    (model/provider        │  │  · server/agent/session policy │
        │     routing + fallback)   │  │    (execution + governance)    │
        └───────────────────────────┘  └──────────────────────────────┘
```

- **OpenRouter** routes at the *model/provider* layer (cost + fallback across 400+ models).
  Agent Eddy sits **above** it: an OpenRouter-backed LLM is just one operator in the pool.
- **Omnigent** is an agent *meta-harness* (execution + governance substrate). Agent Eddy
  consumes Omnigent agents **as operators** and adds the Pareto-adaptive routing its
  supervisor/delegation model lacks. The two governance layers compose; the stricter wins.

Prior LLM routers (RouterBench, RouteLLM, MasRouter) evaluate a *static* pool with
*convex-hull* analysis over *two* objectives. Agent Eddy targets *online* selection on
*non-convex four-objective* frontiers under *non-stationarity* — see
[`docs/EVALUATION.md`](docs/EVALUATION.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture: data models, skyline maintenance, routing policies, governance, adapters, the main loop, worked example, build phases, testing strategy. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Position-paper evaluation: how routing effectiveness is measured today, adopted metrics, and three concrete experiments (ICML/NeurIPS-targeted). |
| [`docs/diagrams.md`](docs/diagrams.md) | Mermaid + ASCII diagrams of the layering, main loop, skyline, and data model. |

## Status

Design phase. No runtime code yet — `docs/DESIGN.md` §13 lays out the phased implementation
(Phase 0 minimal runnable core → Phase 5 real OpenRouter / Omnigent backends).

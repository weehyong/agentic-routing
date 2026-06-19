# Agent Eddy Router

A **routing-centric orchestration runtime** for agentic data integration — the missing
adaptive routing layer between a queue of *work items* (subtasks) and a pool of *agent
operators* (data-integration tools, LLMs, verifiers).

Based on the position paper *Beyond Eddies: Agentic Data Integration with intelligent
orchestration* (Wee Hyong Tok, 2026). This repository contains both the **design**
([`docs/DESIGN.md`](docs/DESIGN.md)) and a working **runtime + evaluation harness**
(the `eddy` package). See [Installation](#installation) and [Running](#running) to get started.

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

## Installation

Requires **Python ≥ 3.11**. The core runtime is **stdlib-only**; extras pull in the
scientific stack used by the evaluation harness and figures.

```bash
git clone <repo-url> eddies-router
cd eddies-router

# create + activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install the package (editable) with the extras you need
pip install -e .                   # core runtime only (no third-party deps)
pip install -e ".[dev]"            # + pytest, numpy, matplotlib  (run the tests)
pip install -e ".[eval]"           # + pandas/scipy/tiktoken/huggingface_hub (full experiments)
```

Dependency extras are declared in [`pyproject.toml`](pyproject.toml):

| Extra  | Pulls in | Use for |
|--------|----------|---------|
| (none) | stdlib only | importing and using the `eddy` runtime |
| `dev`  | `pytest`, `numpy`, `matplotlib` | running the test suite |
| `viz`  | `matplotlib` | regenerating figures |
| `eval` | `numpy`, `matplotlib`, `pandas`, `pyarrow`, `tiktoken`, `scipy`, `huggingface_hub` | the real/synthetic coverage experiments |

## Running

### Test suite (verifies the five runtime invariants + skyline/coverage equivalence)

```bash
pip install -e ".[dev]"
pytest                             # 145 tests
```

### Experiment 2 — synthetic skyline vs. scalarization sweep

Shows that skyline routing reaches the full Pareto frontier while fixed-weight
scalarization loses coverage on non-convex frontiers. Writes JSON/CSV/figures to
`results/exp2/`.

```bash
pip install -e ".[eval]"
python -m eddy.eval.run_exp2                 # default sweep (5 seeds, m=20)
python -m eddy.eval.run_exp2 --seeds 3 --m 40 --out results/exp2
```

### Experiment 2 — "money figure" on real model pools

Downloads RouterBench / LLMRouterBench via `huggingface_hub` (or uses any data already
present under `data/raw/`) and computes coverage loss across six (T, R) construction
cells. Output is **provenance-gated**: real data → `results/real/`, mock fallback →
`results/mock/` (mock numbers are never written as figures).

```bash
pip install -e ".[eval]"
python -m eddy.eval.exp2_real                # try real data, fall back to mock
python -m eddy.eval.exp2_real --mock-only    # force seeded mock (offline / CI)
python -m eddy.eval.exp2_real --out results/real
```

### Using the runtime as a library

```python
from eddy import Eddy, OperatorRegistry, WorkItem, OperatorProfile, QCTR

# register operators with (Quality, Cost, Latency, Risk) profiles, enqueue work
# items, then tick the router — it routes each item over the live Pareto skyline.
# See docs/DESIGN.md §11 (Algorithm 1) and tests/test_runtime_invariants.py for
# a complete, runnable construction.
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture: data models, skyline maintenance, routing policies, governance, adapters, the main loop, worked example, build phases, testing strategy. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Position-paper evaluation: how routing effectiveness is measured today, adopted metrics, and three concrete experiments (ICML/NeurIPS-targeted). |
| [`docs/diagrams.md`](docs/diagrams.md) | Mermaid + ASCII diagrams of the layering, main loop, skyline, and data model. |

## Status

Runtime core + evaluation harness implemented and tested (145 passing tests). The phased
build is tracked in `docs/DESIGN.md` §13 (Phase 0 minimal runnable core → Phase 5 real
OpenRouter / Omnigent backends); the operator pool currently ships with simulated adapters.

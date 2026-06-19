# Agent Eddy Router — Evaluation Plan

Evaluation design for a **position paper** targeting ICML / NeurIPS. Companion to
[`DESIGN.md`](DESIGN.md). The goal is to make each of Agent Eddy's novel claims *falsifiable*
with cheap, reproducible experiments, and to position the contribution precisely against the
existing routing-evaluation literature.

---

## 0. Framing — why a new evaluation is needed

Every existing LLM-router benchmark evaluates **one model selection step** over a **static
pool**, with **convex-hull** (scalarization-friendly) analysis over **two objectives** (cost,
quality). Agent Eddy makes three claims that none of them can test:

1. **Skyline > scalarization** on *non-convex, four-objective* `(Q,C,T,R)` frontiers.
2. **Adaptive routing wins under non-stationarity** — latency spikes, quality/schema drift,
   rate-limit cost rises occurring *mid-run*.
3. **Governance masks preserve correctness** under aggressive reordering, at negligible
   quality cost.

The evaluation below isolates each claim and, wherever possible, reuses established
infrastructure (RouterBench-style precomputed records; DeepMatcher/Magellan entity-resolution
datasets) so runs are cheap, deterministic, and reproducible — a property reviewers weight
heavily for a position paper.

---

## 1. Related work — how routing effectiveness is measured today

| System / line | What it measures | Key metrics | Gap for Agent Eddy |
|---|---|---|---|
| **RouterBench** (arXiv:2403.12031) | multi-LLM routing on 405K precomputed inference records; cost–quality convex hull | **AIQ** (area between router curve and individual models), cost–accuracy curve | static pool; 2 objectives; no latency/risk; no non-stationarity |
| **RouteLLM** (lm-sys, arXiv:2406.18665) | strong/weak binary routing; trained routers | **PGR** (performance-gap recovered), **CPT** (call-perf threshold), % cost cut at fixed quality | binary; convex-hull; single-step; static |
| **LLMRouterBench** (arXiv:2601.07206) | 400K instances, 21 datasets, 33 models; perf and perf-cost settings | accuracy, cost-normalized accuracy | single-step model selection, not multi-step orchestration |
| **OptLLM** (arXiv:2405.15130) | optimal query→model assignment | Pareto cost/accuracy | offline assignment; no online adaptation |
| **MasRouter** (arXiv:2502.11133) | multi-agent system routing; learns a config | task accuracy, **overhead reduction %**, utility–cost | learns a *fixed* config; not adaptive *mid-run* under drift |
| **MOO metrics** (general) | quality of a solution set vs. the true Pareto front | **Hypervolume (HV)**, **IGD**, **ParetoDist**, **Δ-spread** | rarely applied to *online* LLM routing |
| **Non-stationary bandits** (arXiv:1708.01799) | regret in changing worlds | **dynamic / interval / switching regret**, path-length `P_T`, `O(√T(1+P_T))` | theory unconnected to agentic-routing benchmarks |

**Takeaway.** The field has rich *static, 2-objective, convex-hull* router evaluation and
rich *non-stationary regret theory*, but nothing that joins them for *online, multi-objective,
governed* agentic routing. That join is the paper's evaluation contribution.

---

## 2. Metrics adopted

**Primary (routing quality).**

- **Pareto regret** (paper §11). For an observed outcome vector `o = (Qₒ,Cₒ,Tₒ,Rₒ)` and the
  offline-computed Pareto frontier `F`: `ParetoRegret = min_{f∈F} ‖ô − f̂‖₂` under normalized
  objectives. Measures whether the system operates near the efficient frontier or wastefully
  inside it — strictly more informative than scalar task success.
- **Dynamic regret** — cumulative gap vs. the *per-tick best* operator (the right benchmark
  for non-stationary environments; fixed-best regret is misleading here). Report path-length
  `P_T` of the environment to contextualize.
- **Hypervolume (HV)** of the achieved `(Q,C,T,R)` outcome set vs. a fixed reference point —
  standard MOO solution-set quality.
- **Recovery latency** — ticks to return to pre-event Pareto regret after a volatility event.
- **Frontier coverage** — fraction of Pareto-optimal operators a method can actually select.

**Secondary (data-integration quality).** entity-resolution **F1**, schema-match accuracy
(before/after drift), contradiction-detection precision/recall, provenance coverage.

**Safety.** policy violations reaching output (**target 0**), unsafe tool invocations blocked,
**gate-trigger precision** (no unnecessary human approvals), data-residency violations.

**Efficiency.** total tokens, dollar cost, token-budget utilization, wall-clock p50/p95.

All metrics: multiple seeds, **95% CIs via paired bootstrap**; for non-stationary episodes also
report **learning curves** (utility vs. time) and adaptation latency.

---

## 3. Baselines

- **B1 — Static workflow graph.** Fixed operator order + retry rules (ADF / Glue style). No
  rerouting; retries go through the same operator.
- **B2 — Supervisor router.** A manager LLM picks the next specialist step from prompting +
  tool observations (AutoGen / supervisor pattern). Adaptive but not Pareto-constrained.
- **B3 — Fixed-weight scalarized router.** Eq. 1 with fixed `(λ,μ,ν)` over the *full* eligible
  set (no skyline restriction). The standard multi-objective query-optimization approach.
- **Agent Eddy variants** — P1 backpressure-greedy, P2 windowed lottery, P3 contextual bandit,
  all skyline-constrained, with dynamic skyline maintenance.

Ablations (toggle one component): disable skyline (route from all eligible); disable ready/done
masks (unconstrained reordering); disable ticket windowing; disable governance masking; disable
lineage-based credit assignment; replace Pareto tracking with fixed-weight scalarization.

---

## Experiment 1 — Robustness under non-stationarity  *(headline; RQ1, RQ3)*

**Claim tested.** Adaptive, skyline-constrained routing bounds Pareto regret under volatility,
where static and fixed-weight baselines degrade and recover slowly or not at all.

**Setup.** A **RouterBench-style offline harness**: precompute each operator's `(Q,C,T,R)`
outcome distribution on a workload, then route over the precomputed records while injecting
volatility — **no live API calls**, so runs are cheap and exactly reproducible. Volatility
injections (paper §11.3):

- **Latency spike** — 3× an operator's `T` for a timed interval.
- **Quality drift** — swap a matching operator's backend to a weaker/stronger model mid-run.
- **Rate limiting** — throttle an operator to inflate effective `T`/`C` (dynamic-β interacts).
- **Partial outage** — 20% transient failure rate on one operator.
- **Task-mix shift** — change difficulty distribution easy→hard to test frontier adaptation.

**Compare.** Agent Eddy {P1, P2, P3} vs. B1, B2, B3.

**Metrics.** Pareto regret (primary), dynamic regret, **recovery latency** per event,
hypervolume; learning curves. Report `P_T` per episode. Multiple seeds + 95% CIs.

**Hypothesis (falsifiable).** At each injected event the static (B1) and scalarized (B3)
baselines show a Pareto-regret spike and slow/no recovery; the supervisor (B2) reacts but
without Pareto guarantees. Agent Eddy's dynamic skyline detects the shift (an operator leaves
the skyline) and re-routes, producing **lower peak Pareto regret and shorter recovery
latency**. If Agent Eddy does *not* improve recovery latency vs. B2 at p<0.05, the adaptivity
claim fails.

---

## Experiment 2 — Skyline vs. scalarization on non-convex frontiers  *(ablation; RQ2, RQ6)*

**Claim tested.** Fixed-weight scalarization provably cannot reach operators in non-convex
pockets of the Pareto frontier *for any weights*; skyline selection reaches all of them.

**Setup.** Controlled operator pools sweeping **frontier convexity** — vary the fraction of
the Pareto set lying in non-convex pockets. Ground the construction by first **demonstrating
that real model pools are non-convex**: take RouterBench / LLMRouterBench precomputed records,
add latency and risk dimensions, and show the four-objective frontier has non-convex regions
(the paper's Table-2 `A₄` case — an operator excluded from the skyline yet unreachable by
scalarization — generalized).

**Compare.** Skyline-restricted selection vs. fixed-weight scalarization swept over a **dense
grid of `(λ,μ,ν)`** (the best any single weight vector can do).

**Metrics.**

- **Frontier coverage loss** — fraction of Pareto-optimal operators unreachable by *any*
  weight vector in the grid (the classical "scalarization finds only convex-hull points"
  result, measured empirically).
- **Selection regret** when the optimal operator for a context sits in a non-convex pocket.

**Hypothesis (falsifiable).** Coverage loss for scalarization grows monotonically with the
non-convex fraction and is **> 0** whenever non-convex pockets exist; skyline coverage loss is
**0**. If real model pools turn out to be convex (coverage loss ≈ 0 even for scalarization),
the skyline contribution is weakened — this experiment is designed to be able to show that.

This experiment isolates the **skyline** contribution from the **adaptivity** contribution of
Experiment 1.

---

## Experiment 3 — Governance-correctness under reordering  *(safety; RQ4, RQ5)*

**Claim tested.** Eligibility masks + the governance gate chain drive policy violations to ~0
under aggressive reordering, with negligible quality loss vs. an ungoverned adaptive router —
the unique agentic-DI value (correctness *and* adaptivity together).

**Setup.** A **real DI workload** built on DeepMatcher / Magellan entity-resolution and
schema-matching datasets — **Abt-Buy**, **Amazon-Google**, **DBLP-ACM (structured + dirty)**,
**DBLP-Scholar** (fixed train/val/test splits). Inject:

- **Schema drift** — rename/retype 10% of columns at a random point in a batch run.
- **Regulated data** — mark a subset of items `data_classification=regulated`, requiring the
  `DataClassificationGate` to block external-transmission operators.
- **Write-gated operators** — `schema.mapper` / `quality.repairer` require approval.
- **Prompt-injection payloads** — embed tool-control instructions in retrieved content to test
  containment (threat model, DESIGN §8).

**Compare.** Agent Eddy (masks + gate chain) vs. an **ungoverned adaptive router** (same
routing, gates disabled) vs. a **static pipeline** (B1).

**Metrics.**

- Quality: entity-resolution **F1**, schema-match accuracy before/after drift, **adaptation
  latency** (masks must not cost quality or speed).
- Safety: **policy violations reaching output (target 0)**, unsafe tool invocations blocked,
  **gate-trigger precision** (penalize unnecessary approvals), data-residency violations.
- Efficiency: token-budget adherence at **100 / 75 / 50 / 25%** of nominal budget (does the
  router degrade gracefully?).

**Hypothesis (falsifiable).** Agent Eddy reduces policy-violations-reaching-output to ~0 with
F1 within noise of the ungoverned router, while the static pipeline fails to adapt to schema
drift (large F1 drop, no recovery). If governance costs > a small, pre-registered F1 delta vs.
the ungoverned router, the "negligible quality cost" claim fails.

---

## 4. Methodology notes (reviewer-facing)

- **Reproducibility.** Offline precomputed-record evaluation (RouterBench-style) for the
  routing experiments; seeded `random.Random` threaded through execution and policies; every
  routing decision carries a `replay_id` for exact re-execution. Release the harness + the
  injected-volatility schedules as a community artifact.
- **Right metrics.** Pareto-relative metrics (Pareto regret, HV, IGD) over scalar accuracy;
  **dynamic** regret (not fixed-best) for non-stationary episodes; report `P_T`.
- **Statistics.** Multiple seeds; paired bootstrap 95% CIs for per-task comparisons; learning
  curves + adaptation latency for non-stationary episodes; pre-register the F1-delta threshold
  for Experiment 3.
- **Ablations.** One per contribution — skyline, ready/done masks, ticket windowing,
  governance masking, dynamic-β cost, lineage-based credit assignment — so each claimed
  component earns its place.

---

## 5. Mapping to the paper's research questions

| RQ (paper §11.1) | Covered by |
|---|---|
| RQ1 — robustness vs. static graphs / supervisors under non-stationarity | Experiment 1 |
| RQ2 — Pareto routing reduces cost at fixed quality vs. scalarization | Experiment 2 |
| RQ3 — skyline eliminates provably suboptimal selections | Experiment 1 + 2 |
| RQ4 — semantic masks preserve correctness under reordering | Experiment 3 |
| RQ5 — governance-aware routing reduces safety violations | Experiment 3 |
| RQ6 — Pareto frontier as a useful governance artifact | Experiment 2 (frontier analysis) + `FrontierLog` |

The paper's benchmark suite (DI-1 multi-source entity resolution under source volatility; DI-2
schema matching under drift; DI-3 quality repair with governance gates; DI-4 token-budget
adherence; DI-5 multi-source research synthesis; DI-6 code-and-test loop) maps onto these three
experiments: DI-1/DI-2/DI-4 → Exp 1, the frontier analysis → Exp 2, DI-2/DI-3/DI-4 → Exp 3.

---

## References (resolve before submission)

- RouterBench — arXiv:2403.12031
- RouteLLM — arXiv:2406.18665 · github.com/lm-sys/RouteLLM
- LLMRouterBench — arXiv:2601.07206
- OptLLM — arXiv:2405.15130
- MasRouter — arXiv:2502.11133 (ACL 2025)
- Efficient Contextual Bandits in Non-stationary Worlds — arXiv:1708.01799
- Deep Learning for Entity Matching (DeepMatcher) — SIGMOD 2018
- Magellan entity-matching datasets · Ditto (arXiv:2004.00584) · Machamp (arXiv:2106.08455)
- Skyline operator — Börzsönyi et al., ICDE 2001 · Eddies — Avnur & Hellerstein, SIGMOD 2000

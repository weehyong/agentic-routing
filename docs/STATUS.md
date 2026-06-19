# Project Status — Checkpoint

**Last updated:** 2026-06-19 — **M2 "money figure" COMPLETE on REAL data: verdict = SUPPORTED.**
**Prev pause:** 2026-06-18 (API credits ran out mid-task).

---

## Where we are

Building the **Agent Eddy Router** — a routing-centric orchestration runtime for agentic data
integration — toward a **NeurIPS position+proof-of-concept paper** (Datasets & Benchmarks as
fallback). See `docs/DESIGN.md` (architecture) and `docs/EVALUATION.md` (experiment plan).

**Team agents** (definitions in `.claude/agents/`): `professor` (strategy), `researcher`
(technical rigor), `research-assistant` (Python implementation), `phd-student` (literature/gaps).
NOTE: these only register as selectable subagent types in a *fresh* session; this session ran
them as `claude` agents that read their persona file first.

### Done

1. **Docs** — `README.md`, `docs/DESIGN.md`, `docs/EVALUATION.md`, `docs/diagrams.md`.
2. **Planning round** (professor + researcher):
   - Framing = hybrid position+PoC, target **NeurIPS** (D&B fallback). Thesis: *agentic data
     integration is online Pareto-constrained routing, not static planning.*
   - **Headline = Experiment 2** (skyline vs. scalarization on non-convex frontiers), overriding
     EVALUATION.md's choice of Exp 1. Experiment 3 (governance) demoted to appendix/follow-on.
   - Researcher caught a **real bug** in DESIGN §6.2 incremental skyline update (now fixed) and
     pinned down metric definitions (Pareto/dynamic regret vs. ground-truth *expected* vectors).
3. **Milestone M1 — Experiment-2 de-risk on SYNTHETIC data: COMPLETE & VERIFIED.**
   - `eddy/core/profile.py`, `eddy/skyline/{dominance,maintain}.py` (corrected incremental
     update), `eddy/eval/{synth,coverage,run_exp2}.py`, tests in `tests/`.
   - 99/99 tests pass (in venv). Independently re-verified with stdlib:
     **skyline coverage loss = 0 across φ; scalarization loss grows with φ (≈0.20 @ φ=0.2,
     ≈0.35 @ φ=0.4).** Central separation claim HOLDS on synthetic data. Kill criterion not triggered.
   - DESIGN.md §6.2 corrected in place (honest O(k)/O(k²) complexity + A/B regression).
4. **Researcher pre-registration** for the real-data T/R axis synthesis (FROZEN): 3 latency
   constructions (T_tokens primary, T_alt1 aggregate, T_alt2 tier), 3 risk constructions
   (R_err primary, R_alt1 calibration, R_alt2 pure-dispersion decorrelated anchor), 6-cell
   matrix, auto-verdict thresholds, hard MOCK/REAL provenance gate. (Captured in this session's
   transcript; the implementation in `eddy/eval/exp2_real.py` follows it.)

### Milestone M2 — the "money figure": Experiment 2 on REAL model pools — **COMPLETE & VERIFIED**

**Verdict on real RouterBench (0shot): `SUPPORTED`.** 401,467 real records → **86 real pools**
(benchmark slices × up to 11 models). Primary (T_tokens × R_err @ m40): **median scal coverage
loss = 0.125, 73.3% of pools above the 0.05 threshold.** Survives BOTH decorrelated anchors:
(T_alt2 × R_err) = 0.143, (T_tokens × R_alt2) = 0.20. → The non-convex Pareto pocket is real on
real model pools; **the paper has its headline result.**
- Outputs: `results/real/summary.json` + 86 `coverage_*.jsonl` + `manifest.json` (all
  provenance=REAL); figures `figures/exp2_money.{png,pdf}` + `figures/exp2_sensitivity.{png,pdf}`.
- 19/19 `tests/test_exp2_real.py` pass.

**Bug found & fixed (the RA's cut-off bug):** `run()` loaded RouterBench via
`_load_llmrouterbench()`, which only parses jsonl/csv/parquet — but RouterBench ships **wide-format
`.pkl`**. It would have parsed 0 records and **silently fallen back to MOCK**. Fix: added
`_load_routerbench()` (melts the wide DataFrame → long records: dataset/model/score/cost/response/fail)
and made `run()` **local-first** (uses existing `data/raw/` without a network re-download). Changes
in `eddy/eval/exp2_real.py`.

**Env deps added to `.venv`:** `pandas`, `pyarrow` (read pkl), `tiktoken` (primary T_tokens token
counts — was falling back to len/4), `scipy` + `matplotlib` (figure generation). NOTE: these are
NOT yet in `pyproject.toml [eval]` — add them so the env is reproducible.

- ❌ **LLMRouterBench still INCOMPLETE** — never downloaded (`huggingface_hub`, `NPULH/LLMRouterBench`).
  Optional now that RouterBench gives SUPPORTED; would add more pools + native per-record tokens
  (T_tokens) and aggregate time_taken (T_alt1) instead of tiktoken/tier fallbacks.

---

### Phase 0 runtime — **BUILT & VERIFIED (2026-06-19)**

The full Agent Eddy runtime is now implemented end-to-end (DESIGN §4–§11), not just
the skyline/eval pieces. New modules:
- `core/{ids,workitem,lineage,budget,operator}.py` (§5) — masks, append-only lineage,
  dynamic-β BudgetTracker, Operator Protocol.
- `governance/{permissions,gates,eligibility,approval}.py` (§8) — PermissionGate,
  WriteCapabilityGate, DataClassificationGate, ToolArgGate; per-item ready-mask eval;
  HITL approval-as-operator.
- `adapters/{base,simulated}.py` (§9) — ExecContext seam + zero-dep SimulatedBackend
  (mutable base → the volatility harness injects spikes/drift/outage by mutating it).
- `routing/{policy,backpressure,lottery,bandit}.py` (§7) — reward Q−λC−μT−νR; P1
  backpressure-greedy, P2 windowed lottery, P3 **discounted-UCB** contextual bandit
  (γ<1 → re-explores under drift, the Exp 1 regime).
- `observability/{events,sink,frontier_log}.py` (§10) — InMemory/JSONL sinks, FrontierLog.
- `runtime/{registry,queue,completion,eddy}.py` (§11) — **Algorithm 1 main loop**
  enforcing all five invariants; budget-adjusted skyline each tick (dynamic-β frontier shift).
- Public API exported from `eddy/__init__.py` (Eddy, WorkItem, OperatorRegistry, ...).

**Tests: 145/145 pass** (118 prior + `tests/test_runtime_invariants.py` = 27 new):
Inv 1–5, governance gates, write-gating+approval flow, budget halt + β-monotonicity,
budget-adjusted skyline exclusion, policy-within-skyline for all 3 policies, and a
12-seed fuzz over random chains × policies asserting all invariants every tick.

NOTE: design §13 calls this "Phase 0+1+2" (core + governance + full policies). The
incremental skyline `on_estimate_update` exists & is tested, but the main loop uses
`full_recompute` each tick (correct, O(k²), fine for experiments).

---

## RESUME HERE (next session)

1. **Build the Experiment-1 harness** (`eddy/eval/exp1_*.py`) on top of the runtime:
   - Operator pool from RouterBench-style precomputed (Q,C,T,R) records.
   - Volatility injections (EVALUATION §Exp 1): latency spike (3×T), quality drift,
     rate limiting, partial outage (20%), task-mix shift — implemented by mutating
     `SimulatedBackend.base`/`fail_prob` mid-run on a seeded schedule.
   - Baselines **B1** static graph, **B2** supervisor router, **B3** fixed-weight scalarized.
   - Metrics: **Pareto regret** (primary), dynamic regret, **recovery latency**, hypervolume,
     frontier coverage; multi-seed + paired-bootstrap 95% CIs.
   - Falsifiable gate: Eddy must beat B2 on recovery latency at p<0.05 (else adaptivity fails).
2. **Add deps to `pyproject.toml [eval]`** — DONE (pandas, pyarrow, tiktoken, scipy, matplotlib).
3. **Write up M2** money figure in the paper draft (median loss 0.125 / 73% of pools / both
   anchors survive); the runtime now exists to back the Exp-1 adaptivity story too.
4. **(Optional) Finish LLMRouterBench** for a second real corpus.

## Open risks to remember
- The real-data verdict is the paper's make-or-break (professor's M1 gate). Do not over-claim
  until `results/real/summary.json` exists and passes the frozen thresholds.
- `R_err` and `T_tokens` carry some Q-correlation by construction — the decorrelated anchors
  (T_alt2, R_alt2) are load-bearing for credibility, not optional.
- DESIGN §6.2 bug is fixed but the rest of the runtime (main loop, policies, governance) is
  NOT built yet — that's Phase 0 for Experiment 1, after the money figure.

## Housekeeping
- `data/raw/` holds ~1.5GB; consider gitignoring it (not yet a git repo).
- Tests run in a venv (`pip install -e .[eval]`); system Python lacks pytest.

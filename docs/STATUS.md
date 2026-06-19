# Project Status — Checkpoint

**Last updated:** 2026-06-18 (end of session; paused — resume tomorrow)
**Reason for pause:** Anthropic API credit balance ran out mid-task; user stopping for the day.

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

### In progress / INTERRUPTED (research assistant, ran out of credits)

**Milestone M2 — the "money figure": Experiment 2 on REAL model pools.** Partially built:
- ✅ `eddy/eval/exp2_real.py`, `eddy/eval/mock_real.py`, `tests/test_exp2_real.py`.
- ✅ Frozen lookup tables: `data/frozen/{throughput_tps,model_params,latency_tier}.json`.
- ✅ **RouterBench REAL data downloaded**: `data/raw/withmartian__routerbench/*.pkl`
  (0shot 95M, 5shot 164M, raw 1.2G).
- ❌ **LLMRouterBench download INCOMPLETE** — only `download.md` stub; tarball left
  `.incomplete`/locked. Need to re-download (`huggingface_hub`, dataset `NPULH/LLMRouterBench`).
- ✅ MOCK pipeline ran end-to-end → `results/mock/` (validates code + verdict logic + provenance
  gate). **MOCK verdict was CONVEX-REAL-FAIL but this is MOCK data only — says NOTHING about real
  pools.**
- ❌ **`results/real/` DOES NOT EXIST** — the analysis never ran on the downloaded RouterBench
  data. **THE REAL-DATA RESULT IS STILL UNKNOWN.**

---

## RESUME HERE (tomorrow's first task)

1. **Run the real analysis on the already-downloaded RouterBench pkl:**
   `python -m eddy.eval.exp2_real` (or its documented entrypoint) pointed at
   `data/raw/withmartian__routerbench/`. Expect outputs in `results/real/` + `figures/` (figs
   only emit if all inputs are provenance=REAL).
2. **Inspect `results/real/summary.json`** for the auto-verdict across the 6 construction cells:
   - SUPPORTED = scal coverage loss > 0.05 on a non-trivial fraction of pools AND survives the
     decorrelated anchors (T_alt2, R_alt2). → real-data money figure is real; paper has its
     headline result.
   - CONVEX-REAL-FAIL = real pools ~convex → pivot per professor's kill-criterion (re-center on
     Exp 1 adaptivity, or NeurIPS D&B benchmark framing).
   - ARTIFACT-FAIL = non-convexity dies under decorrelated anchors → retract real claim, keep
     synthetic only.
3. **Finish LLMRouterBench download** and repeat (it has per-record tokens → feeds T_tokens, and
   aggregate time_taken → feeds T_alt1; more pools = stronger evidence).
4. **Verify** the RA's `exp2_real.py` faithfully implements the frozen pre-registration (esp.
   the decorrelated anchors and the provenance gate) before trusting any verdict — the RA was
   cut off mid-run and `test_exp2_real.py` should be run first (`pytest`).

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

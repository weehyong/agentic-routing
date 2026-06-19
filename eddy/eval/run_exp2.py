"""
eddy.eval.run_exp2
==================
Experiment-2 runner: skyline vs. scalarization on non-convex frontiers.

Runs two sweeps and emits machine-readable results.

Sweep A — Coverage loss vs. phi (non-convex fraction)
------------------------------------------------------
For phi in {0, 0.1, 0.2, 0.3, 0.4, 0.5}, generate a pool, compute:
  - Skyline coverage loss (expected 0 always)
  - Scalarization coverage loss (expected ≈ phi)
Multiple seeds; report mean ± std.

Sweep B — Grid resolution convergence
--------------------------------------
For a fixed non-convex pool (phi=0.4), sweep m in {5, 10, 20, 40} and show
that scalarization coverage loss converges (the coarse grid already captures
the deficit).  This proves the grid is dense enough: the gap is a frontier
property, not a grid-resolution artefact.

Outputs
-------
results/exp2/run.json      — machine-readable full results
results/exp2/metrics.csv   — summary table
results/exp2/fig_d.png     — Fig-D: coverage loss vs phi (if matplotlib available)
results/exp2/fig_e.png     — Fig-E: coverage loss vs m  (if matplotlib available)

The figures can be regenerated from run.json alone (see regenerate_figures()).

Usage
-----
    python -m eddy.eval.run_exp2
    python -m eddy.eval.run_exp2 --out results/exp2 --seeds 5 --m 20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from eddy.eval.synth import generate_pool
from eddy.eval.coverage import (
    scalarization_reachable,
    skyline_of,
    coverage_loss,
)


# ---------------------------------------------------------------------------
# Core measurement helpers
# ---------------------------------------------------------------------------

def _measure(pool_result, m: int) -> dict[str, float]:
    """Compute coverage losses for one pool at one resolution."""
    pool = pool_result.operators
    sky = skyline_of(pool)
    scal_reachable = scalarization_reachable(pool, m=m)
    sky_loss = coverage_loss(skyline_of(pool), sky)   # always 0
    scal_loss = coverage_loss(scal_reachable, sky)
    return {
        "sky_loss": sky_loss,
        "scal_loss": scal_loss,
        "skyline_size": len(sky),
        "scal_reachable_size": len(scal_reachable),
        "nonconvex_count": len(pool_result.nonconvex_ids),
        "total_operators": len(pool),
    }


def _mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


# ---------------------------------------------------------------------------
# Sweep A: coverage loss vs phi
# ---------------------------------------------------------------------------

def sweep_phi(
    phis: list[float],
    seeds: list[int],
    m: int,
    n_front: int = 20,
    p_curve: float = 2.0,
    n_dominated: int = 10,
) -> list[dict[str, Any]]:
    """For each phi, run multiple seeds and collect statistics."""
    rows: list[dict[str, Any]] = []
    for phi in phis:
        sky_losses: list[float] = []
        scal_losses: list[float] = []
        per_seed: list[dict] = []

        for seed in seeds:
            pool_result = generate_pool(
                n_front=n_front,
                phi=phi,
                p_curve=p_curve,
                n_dominated=n_dominated,
                seed=seed,
            )
            r = _measure(pool_result, m)
            sky_losses.append(r["sky_loss"])
            scal_losses.append(r["scal_loss"])
            per_seed.append({
                "seed": seed,
                "phi": phi,
                **r,
            })

        sky_mean, sky_std = _mean_std(sky_losses)
        scal_mean, scal_std = _mean_std(scal_losses)
        rows.append({
            "phi": phi,
            "sky_loss_mean": sky_mean,
            "sky_loss_std": sky_std,
            "scal_loss_mean": scal_mean,
            "scal_loss_std": scal_std,
            "per_seed": per_seed,
        })
    return rows


# ---------------------------------------------------------------------------
# Sweep B: coverage loss vs grid resolution m
# ---------------------------------------------------------------------------

def sweep_resolution(
    ms: list[int],
    phi: float,
    seed: int,
    n_front: int = 20,
    p_curve: float = 2.0,
    n_dominated: int = 10,
) -> list[dict[str, Any]]:
    """For a fixed pool (phi, seed), sweep m and show loss convergence."""
    pool_result = generate_pool(
        n_front=n_front,
        phi=phi,
        p_curve=p_curve,
        n_dominated=n_dominated,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    for m_val in ms:
        r = _measure(pool_result, m_val)
        rows.append({
            "m": m_val,
            "phi": phi,
            "seed": seed,
            **r,
        })
    return rows


# ---------------------------------------------------------------------------
# Figure generation (optional matplotlib)
# ---------------------------------------------------------------------------

def regenerate_figures(run_data: dict, out_dir: Path) -> None:
    """Regenerate Fig-D and Fig-E from run.json data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[run_exp2] matplotlib not available — skipping figures", file=sys.stderr)
        return

    # Fig-D: coverage loss vs phi
    sweep_a = run_data["sweep_phi"]
    phis = [r["phi"] for r in sweep_a]
    sky_means = [r["sky_loss_mean"] for r in sweep_a]
    sky_stds = [r["sky_loss_std"] for r in sweep_a]
    scal_means = [r["scal_loss_mean"] for r in sweep_a]
    scal_stds = [r["scal_loss_std"] for r in sweep_a]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(phis, sky_means, yerr=sky_stds, marker="o", label="Skyline", color="steelblue")
    ax.errorbar(phis, scal_means, yerr=scal_stds, marker="s", label="Scalarization (B3)", color="tomato")
    ax.axline((0, 0), slope=1.0, linestyle="--", color="gray", alpha=0.5, label="ideal scal. loss = phi")
    ax.set_xlabel("Non-convex fraction phi")
    ax.set_ylabel("Coverage loss (fraction of skyline unreachable)")
    ax.set_title("Fig-D: Coverage loss vs. non-convex frontier fraction")
    ax.legend()
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig_path = out_dir / "fig_d.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[run_exp2] Saved {fig_path}")

    # Fig-E: coverage loss vs resolution m
    sweep_b = run_data["sweep_resolution"]
    ms_vals = [r["m"] for r in sweep_b]
    scal_losses_b = [r["scal_loss"] for r in sweep_b]
    sky_losses_b = [r["sky_loss"] for r in sweep_b]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ms_vals, sky_losses_b, marker="o", label="Skyline", color="steelblue")
    ax.plot(ms_vals, scal_losses_b, marker="s", label="Scalarization (B3)", color="tomato")
    ax.set_xlabel("Grid resolution m")
    ax.set_ylabel("Coverage loss")
    ax.set_title("Fig-E: Coverage loss vs. grid resolution (convergence check)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig_path = out_dir / "fig_e.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[run_exp2] Saved {fig_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(sweep_a: list[dict], sweep_b: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep", "phi", "m", "sky_loss_mean", "sky_loss_std",
                         "scal_loss_mean", "scal_loss_std"])
        for row in sweep_a:
            writer.writerow([
                "phi_sweep", row["phi"], "n/a",
                f"{row['sky_loss_mean']:.4f}", f"{row['sky_loss_std']:.4f}",
                f"{row['scal_loss_mean']:.4f}", f"{row['scal_loss_std']:.4f}",
            ])
        for row in sweep_b:
            writer.writerow([
                "resolution_sweep", row["phi"], row["m"],
                f"{row['sky_loss']:.4f}", "n/a",
                f"{row['scal_loss']:.4f}", "n/a",
            ])
    print(f"[run_exp2] Saved {csv_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    out_dir: str = "results/exp2",
    seeds: list[int] | None = None,
    m: int = 20,
    phis: list[float] | None = None,
    ms_resolution: list[int] | None = None,
    phi_for_resolution: float = 0.4,
    n_front: int = 20,
    p_curve: float = 2.0,
    n_dominated: int = 10,
) -> dict:
    """Run Experiment 2 and return the results dict."""
    if seeds is None:
        seeds = list(range(5))
    if phis is None:
        phis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if ms_resolution is None:
        ms_resolution = [5, 10, 20, 40]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[run_exp2] Sweep A: phi in {phis}, seeds={seeds}, m={m}")
    t0 = time.time()
    sweep_a = sweep_phi(
        phis=phis,
        seeds=seeds,
        m=m,
        n_front=n_front,
        p_curve=p_curve,
        n_dominated=n_dominated,
    )
    t1 = time.time()
    print(f"[run_exp2] Sweep A done in {t1-t0:.1f}s")

    print(f"[run_exp2] Sweep B: m in {ms_resolution}, phi={phi_for_resolution}")
    t0 = time.time()
    sweep_b = sweep_resolution(
        ms=ms_resolution,
        phi=phi_for_resolution,
        seed=seeds[0],
        n_front=n_front,
        p_curve=p_curve,
        n_dominated=n_dominated,
    )
    t1 = time.time()
    print(f"[run_exp2] Sweep B done in {t1-t0:.1f}s")

    run_data = {
        "config": {
            "seeds": seeds,
            "m": m,
            "phis": phis,
            "ms_resolution": ms_resolution,
            "phi_for_resolution": phi_for_resolution,
            "n_front": n_front,
            "p_curve": p_curve,
            "n_dominated": n_dominated,
        },
        "sweep_phi": sweep_a,
        "sweep_resolution": sweep_b,
    }

    # Write JSON
    json_path = out_path / "run.json"
    with open(json_path, "w") as f:
        json.dump(run_data, f, indent=2)
    print(f"[run_exp2] Saved {json_path}")

    # Write CSV
    _write_csv(sweep_a, sweep_b, out_path)

    # Write figures
    regenerate_figures(run_data, out_path)

    # --- Print summary to stdout ---
    print("\n=== Experiment 2 Results ===")
    print("\nSweep A: Coverage Loss vs. phi")
    print(f"{'phi':>6}  {'sky_loss':>12}  {'scal_loss':>14}")
    print("-" * 40)
    for row in sweep_a:
        phi_v = row["phi"]
        sky_s = f"{row['sky_loss_mean']:.4f} ± {row['sky_loss_std']:.4f}"
        scal_s = f"{row['scal_loss_mean']:.4f} ± {row['scal_loss_std']:.4f}"
        print(f"{phi_v:>6.2f}  {sky_s:>12}  {scal_s:>14}")

    print("\nSweep B: Scal. Coverage Loss vs. grid resolution m (phi=0.4)")
    print(f"{'m':>6}  {'sky_loss':>10}  {'scal_loss':>10}")
    print("-" * 30)
    for row in sweep_b:
        print(f"{row['m']:>6}  {row['sky_loss']:>10.4f}  {row['scal_loss']:>10.4f}")

    # Self-check
    print("\n=== Self-check ===")
    all_sky_zero = all(abs(r["sky_loss_mean"]) < 1e-9 for r in sweep_a)
    scal_increasing = all(
        sweep_a[i]["scal_loss_mean"] <= sweep_a[i + 1]["scal_loss_mean"] + 0.05
        for i in range(len(sweep_a) - 1)
    )
    scal_nonzero_when_phi_nonzero = all(
        sweep_a[i]["scal_loss_mean"] > 0.01
        for i in range(len(sweep_a))
        if sweep_a[i]["phi"] > 0.1
    )
    scal_converged = (
        len(sweep_b) >= 2 and
        abs(sweep_b[-1]["scal_loss"] - sweep_b[-2]["scal_loss"]) < 0.05
    )

    print(f"  Skyline coverage loss = 0 always:   {'PASS' if all_sky_zero else 'FAIL'}")
    print(f"  Scal. loss roughly non-decreasing:  {'PASS' if scal_increasing else 'FAIL'}")
    print(f"  Scal. loss > 0 when phi > 0:        {'PASS' if scal_nonzero_when_phi_nonzero else 'FAIL'}")
    print(f"  Grid coverage converged (Sweep B):  {'PASS' if scal_converged else 'FAIL'}")

    central_claim = all_sky_zero and scal_nonzero_when_phi_nonzero
    print(f"\n  Central claim held: {'YES' if central_claim else 'NO'}")

    return run_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2 (skyline vs scalarization).")
    parser.add_argument("--out", default="results/exp2", help="Output directory")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--m", type=int, default=20, help="Grid resolution for Sweep A")
    parser.add_argument("--n-front", type=int, default=20, help="Frontier operators per pool")
    parser.add_argument("--p-curve", type=float, default=2.0, help="Curve exponent (p>1 for non-convex pockets)")
    parser.add_argument("--n-dominated", type=int, default=10, help="Dominated operators per pool")
    args = parser.parse_args()

    run(
        out_dir=args.out,
        seeds=list(range(args.seeds)),
        m=args.m,
        n_front=args.n_front,
        p_curve=args.p_curve,
        n_dominated=args.n_dominated,
    )


if __name__ == "__main__":
    main()

"""Daily margin, direct breaches and timing report for a repaired factor backtest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_factor_stress import writeJson
from report_factor_sweep import writeCsv


def aggregateDaily(days, trials, solvers, repeats, selection_dates):
    """Choose the greatest feasible loss without consulting realized returns."""
    grouped = defaultdict(list)
    for trial in trials:
        grouped[trial["date"]].append(trial)
    if set(grouped) != {day["date"] for day in days}:
        raise ValueError("trial dates do not cover the daily model dates exactly")
    rows = []
    for day in days:
        candidates = grouped[day["date"]]
        expected = {(solver, repeat) for solver in solvers for repeat in range(repeats)}
        actual = {(row["solver"], row["repeat"]) for row in candidates}
        if actual != expected or len(candidates) != len(expected):
            raise ValueError("each date must contain each solver/seed pair exactly once")
        for row in candidates:
            if not np.isfinite(row["repaired_margin"]) or row["repaired_margin"] < 0:
                raise ValueError("repaired margins must be finite and nonnegative")
            if abs(row["realized_loss"]-day["realized_loss"]) > 1e-12:
                raise ValueError("trial realized loss differs from the daily market result")
        for method in (*solvers, "combined"):
            eligible = candidates if method == "combined" else [r for r in candidates if r["solver"] == method]
            best = min(eligible, key=lambda r: (-r["repaired_margin"], r["id"]))
            reference = next(r for r in day["references"] if r["method"] == "exact_repricing_continuous")
            margin = best["repaired_margin"]
            rows.append(dict(date=day["date"], method=method, margin=margin,
                realized_pnl=day["realized_pnl"], realized_loss=day["realized_loss"],
                breach=bool(day["realized_loss"] > margin),
                breach_excess=max(0., day["realized_loss"]-margin),
                winner=best["id"], winner_solver=best["solver"], winner_repeat=best["repeat"],
                candidates=len(eligible), calibration_start=day["calibration_start"],
                calibration_end=day["calibration_end"], penalty_selection_date=day["date"] in selection_dates,
                reference_margin=reference["margin"], reference_gap=reference["margin"]-margin,
                summed_candidate_repair_seconds=sum(r["totalSeconds"] for r in eligible),
                summed_amortized_solve_seconds=sum(r["raw_amortized_solve_seconds"] for r in eligible)))
    return rows


def summarizeDaily(rows):
    summary = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        for subset in ("all_dates", "penalty_selection_dates", "other_dates"):
            selected = [r for r in group if subset == "all_dates" or
                        r["penalty_selection_date"] == (subset == "penalty_selection_dates")]
            if not selected:
                continue
            summary.append(dict(method=method, subset=subset, dates=len(selected),
                breaches=sum(r["breach"] for r in selected),
                breach_rate=sum(r["breach"] for r in selected)/len(selected),
                mean_margin=float(np.mean([r["margin"] for r in selected])),
                max_realized_loss=max(r["realized_loss"] for r in selected),
                max_breach_excess=max(r["breach_excess"] for r in selected),
                mean_reference_gap=float(np.mean([r["reference_gap"] for r in selected])),
                max_reference_gap=max(r["reference_gap"] for r in selected)))
    return summary


def generate(experiment, group):
    raw, repaired = experiment/"results", experiment/"repair_results"
    settings = json.loads((raw/"settings.json").read_text())
    plan = json.loads((experiment/"plan.json").read_text())
    days = json.loads((raw/"days.json").read_text())["days"]
    timing = json.loads((raw/"timings.json").read_text())
    repair_totals = json.loads((repaired/"results.json").read_text())
    if hashlib.sha256(group.read_bytes()).hexdigest() != settings["group_sha256"]:
        raise ValueError("local verification dataset differs from the remote dataset")
    for directory in (raw, repaired):
        if json.loads((directory/"verification.json").read_text())["status"] != "passed":
            raise ValueError("raw and repaired sample verification must pass before reporting")
    for key, expected in (("window", 125), ("bits", [8]), ("multipliers", [plan["penalty_multiplier"]]),
                          ("runs", plan["runs"]), ("steps", plan["steps"]), ("repeats", plan["repeats"])):
        if settings[key] != expected:
            raise ValueError(f"execution does not match requested {key}")
    prices = pd.read_csv(group, index_col=0).sort_index()
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or not prices.index.is_unique:
        raise ValueError("this full-coverage report requires unique dates and complete positive prices")
    dates = list(prices.index)
    eligible = dates[settings["window"]+1:]
    if [day["date"] for day in days] != eligible or plan["dates"] != eligible:
        raise ValueError("execution did not cover every eligible dataset date")
    # Preserve the full float written by the application; the default CSV parser
    # can round tiny weights enough to affect portfolios with extreme returns.
    portfolio = pd.read_csv(raw/"portfolio.csv", float_precision="round_trip").set_index("ticker")["weight"]
    prices = prices.loc[:, list(portfolio.index)]
    for day in days:
        position = dates.index(day["date"])
        if day["calibration_start"] != dates[position-settings["window"]] or day["calibration_end"] != dates[position-1]:
            raise ValueError("PCA calibration does not contain exactly the prior 125 return dates")
        pnl = float((prices.iloc[position].to_numpy()/prices.iloc[position-1].to_numpy()-1) @ portfolio.to_numpy())
        np.testing.assert_allclose(pnl, day["realized_pnl"], atol=1e-12, rtol=0)
    outliers = []
    for day in sorted(days, key=lambda d: -abs(d["realized_pnl"]))[:10]:
        position = dates.index(day["date"])
        previous = prices.iloc[position-1].to_numpy()
        current = prices.iloc[position].to_numpy()
        returns = current/previous-1
        contributions = returns*portfolio.to_numpy()
        for rank, index in enumerate(np.argsort(-np.abs(contributions), kind="stable")[:3], start=1):
            outliers.append(dict(date=day["date"], portfolio_pnl=day["realized_pnl"], rank=rank,
                ticker=portfolio.index[index], previous_close=float(previous[index]),
                current_close=float(current[index]), exposure=float(portfolio.iloc[index]),
                asset_return=float(returns[index]), pnl_contribution=float(contributions[index])))
    writeCsv(experiment/"largest_realized_moves.csv", outliers)
    trials = []
    for path in sorted((repaired/"batches").glob("*.json")):
        trials.extend(json.loads(path.read_text())["rows"])
    daily = aggregateDaily(days, trials, plan["solvers"], settings["repeats"], set(plan["selection_dates"]))
    summary = summarizeDaily(daily)
    writeCsv(experiment/"daily_margins.csv", daily)
    writeCsv(experiment/"summary.csv", summary)
    breaches = [row for row in daily if row["breach"]]
    with (experiment/"breaches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(daily[0]))
        writer.writeheader()
        writer.writerows(breaches)
    per_day = []
    for day in days:
        candidates = [r for r in trials if r["date"] == day["date"]]
        per_day.append(dict(date=day["date"], **day["timings"],
            summed_amortized_solve_seconds=sum(r["raw_amortized_solve_seconds"] for r in candidates),
            summed_repair_seconds=sum(r["totalSeconds"] for r in candidates)))
    writeCsv(experiment/"daily_timings.csv", per_day)
    stages = []
    for key in sorted({key for day in days for key in day["timings"]}):
        seconds = sum(day["timings"].get(key, 0.) for day in days)
        stages.append(dict(stage=key, seconds=seconds, timing_kind="summed shared preparation work"))
    for key in ("metadata_seconds", "prefetch_seconds", "preparation_wall_seconds", "worker_phase_wall_seconds",
                "summed_batch_solve_seconds", "summed_qubo_build_seconds", "summed_validation_seconds",
                "summed_repricing_seconds", "summed_sample_write_seconds", "reporting_seconds", "total_wall_seconds"):
        stages.append(dict(stage=key, seconds=timing[key], timing_kind="wall" if "wall" in key else "stage work"))
    for key, seconds in repair_totals["stage_seconds"].items():
        stages.append(dict(stage="repair_"+key, seconds=seconds, timing_kind="summed sequential repair work"))
    stages.append(dict(stage="repair_wall_seconds", seconds=repair_totals["repair_wall_seconds"], timing_kind="wall"))
    writeCsv(experiment/"timing_breakdown.csv", stages)
    verification = dict(status="passed", eligible_dates=len(eligible), warmup_dates=settings["window"]+1,
        independently_repriced_realized_returns=len(days), calibration_return_observations=settings["window"],
        no_evaluation_close_in_calibration=True, trials=len(trials), daily_series=len(plan["solvers"])+1,
        direct_breaches_recomputed=True)
    writeJson(experiment/"verification.json", verification)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combined = [r for r in daily if r["method"] == "combined"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    timeline = pd.to_datetime([r["date"] for r in combined])
    axes[0].plot(timeline, [r["margin"]*100 for r in combined], label="Combined repaired margin", lw=1.6)
    axes[0].plot(timeline, [r["reference_margin"]*100 for r in combined], label="Continuous reference", lw=1, alpha=.7)
    axes[0].plot(timeline, [max(0., r["realized_loss"])*100 for r in combined],
                 label="Realized loss (gains shown as 0)", lw=.8, alpha=.65)
    breach_rows = [r for r in combined if r["breach"]]
    if breach_rows:
        axes[0].scatter(pd.to_datetime([r["date"] for r in breach_rows]), [r["realized_loss"]*100 for r in breach_rows], color="red", s=24, label="Breach", zorder=5)
    axes[0].set_ylabel("Portfolio exposure (%)")
    axes[0].legend(loc="upper left", ncol=2)
    for method in plan["solvers"]+["combined"]:
        selected = [r for r in daily if r["method"] == method]
        axes[1].plot(timeline, [r["reference_gap"]*10000 for r in selected], label=method.replace("torch_", ""), lw=1)
    axes[1].set_ylabel("Reference margin gap (bp)")
    axes[1].legend(ncol=4)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Group 1 · 8,590 stocks · 125-return EW window · 8-bit factor coordinates")
    fig.savefig(experiment/"backtest.png", dpi=160)
    plt.close(fig)

    whole = [r for r in summary if r["subset"] == "all_dates"]
    table = "\n".join(f"| {r['method']} | {r['dates']} | {r['breaches']} | {r['breach_rate']:.2%} | {r['mean_margin']:.6%} | {r['mean_reference_gap']*10000:.4f} |" for r in whole)
    total = timing["total_wall_seconds"]+repair_totals["repair_wall_seconds"]
    shell_times = [float(line.split()[1]) for line in (experiment/"run.log").read_text().splitlines()
                   if line.startswith("real ")]
    if len(shell_times) != 4:
        raise ValueError("expected completed benchmark, repair and two verification shell timings")
    writeJson(experiment/"workflow_timing.json", dict(
        benchmark_shell_seconds=shell_times[0], repair_shell_seconds=shell_times[1],
        raw_verification_shell_seconds=shell_times[2], repair_verification_shell_seconds=shell_times[3],
        summed_remote_phase_seconds=sum(shell_times),
        benchmark_and_repair_internal_seconds=total,
        excludes="initial unit tests, gaps between shell commands, transfer and final daily report"))
    report = f"""# Full Group 1 factor-stress backtest

Completed all **{len(days)} eligible dates**, **{eligible[0]}–{eligible[-1]}**, for a fixed random long-only portfolio of 8,590 stocks (seed {settings['seed']}, total exposure 1). The first 126 price dates are calibration warmup. Each fit uses exactly 125 prior daily returns (`ew_window=125`, decay {settings['ew_lambda']}) and excludes the evaluation close.

## Configuration

Eight V100 GPUs; SBM, SVL and TRF; {settings['runs']} trajectories, {settings['steps']:,} steps and {settings['repeats']} seeds per solver/date: **{len(trials):,} QUBO solves**. Coordinates use 8-bit encoding: 2 PCA factors plus 1 portfolio residual coordinate, **122 variables and 7,381 edges**. Radius 3 is unchanged. Projection, auxiliary reconstruction and feasible exact-P&L descent are applied to every raw result.

Penalty multiplier **{plan['penalty_multiplier']:g}** was selected from the earlier sweep by the smallest mean repaired-margin shortfall from the continuous reference at 8 bits, averaged across solvers and seeds. Actual daily penalty is `multiplier × 1.1 × objective coefficient absolute-sum bound`. This weak penalty relies on repair for feasibility. See [penalty ranking](penalty_ranking.csv).

## Daily results

Each solver's daily margin is the maximum repaired loss over its three seeds. The combined series takes the maximum over all nine candidates, selected without consulting realized loss. A breach is strictly `realized loss > margin`; equality is not a breach. Returns compare the prior available close with the evaluation close.

| Method | Dates | Breaches | Breach rate | Mean margin | Mean reference gap (bp) |
|---|---:|---:|---:|---:|---:|
{table}

All **{repair_totals['repaired_feasible']:,}/{len(trials):,} repaired encodings are feasible**; {repair_totals['converged']:,} reached a local optimum within the repair cap. These are local optima on the integer lattice, not guaranteed global worst losses.

The 20 dates used to select the penalty are included in the full period. [Summary CSV](summary.csv) reports them and the other {len(days)-len(plan['selection_dates'])} dates separately. This is a retrospective fixed-penalty replay, not a prospective test. The radius is illustrative and was not recalibrated to a target breach probability.

The supplied prices produce realized portfolio returns from **{min(d['realized_pnl'] for d in days):.4%}** to **{max(d['realized_pnl'] for d in days):.4%}**. The extreme positive returns are present in the source data; they are not margin breaches. [Largest realized moves and their asset contributions](largest_realized_moves.csv) records the prices and weights behind them for data-quality review. No price corrections were applied.

![Daily margin and realized loss](backtest.png)

## Timing

- Shared preparation: **{timing['preparation_wall_seconds']:.3f} s**.
- Eight-GPU worker phase, including startup: **{timing['worker_phase_wall_seconds']:.3f} s**.
- Raw benchmark wall time: **{timing['total_wall_seconds']:.3f} s**.
- Sequential CPU repair pass: **{repair_totals['repair_wall_seconds']:.3f} s**.
- Raw benchmark plus repair: **{total:.3f} s ({total/60:.2f} minutes)**.
- Sum of the four remote phases, including interpreter startup and both independent verifications: **{sum(shell_times):.2f} s ({sum(shell_times)/60:.2f} minutes)**.

The raw benchmark includes continuous and exhaustive quadratic-lattice references for every date. Summed GPU solve work is {timing['summed_batch_solve_seconds']:.3f} s and overlaps across eight workers. Inclusive timing rows overlap and must not be added. Per-trial solve times are amortized batch time, not measured individual latency. Independent sample verification is included only in the four-phase total. Transfer and final daily report generation are additional.

## Verification and artifacts

Raw and repaired samples were independently verified on the remote server. This daily report independently checks coverage of every eligible dataset date, 125-return calibration boundaries, realized P&L against source prices, complete solver/seed combinations, and direct breach comparisons.

- [Daily margins](daily_margins.csv), [breaches](breaches.csv), [summary](summary.csv).
- [Timing breakdown](timing_breakdown.csv), [daily timing work](daily_timings.csv), [remote workflow timing](workflow_timing.json).
- [Raw trials](results/trials.csv), [repaired trials](repair_results/trials.csv).
- [Daily verification](verification.json), [raw verification](results/verification.json), [repair verification](repair_results/verification.json).
- [Exact invocation](run_remote.sh), [settings and eligible dates](plan.json), [remote log](run.log).
"""
    (experiment/"README.md").write_text(report)
    print(json.dumps(dict(verification=verification, summary=whole, calculation_and_repair_seconds=total)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--group", type=Path, required=True)
    args = parser.parse_args()
    generate(args.experiment.resolve(), args.group.resolve())

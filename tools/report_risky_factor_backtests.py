"""Verify risky-portfolio samples and report daily confidence/penalty sensitivities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from benchmark_factor_stress import writeJson
from report_factor_sweep import writeCsv
from sweep_factor_stress import SOLVERS, buildTrial, loadModel
from risky_factor_portfolios import breachUpperBound


def dailyRows(days, trials, configurations):
    result = []
    for config in configurations:
        for analysis in (["paired_seed0", "primary_best3"] if config["primary"] else ["paired_seed0"]):
            for day in days:
                group = [r for r in trials if r["day"] == day["index"] and r["setting"] == config["id"]
                         and (analysis == "primary_best3" or r["repeat"] == 0)]
                reference = next(r for r in day["references"] if r["method"] == "exact_repricing_continuous"
                                 and r["coverage"] == config["coverage"])
                for solver in (*SOLVERS, "combined"):
                    selected = group if solver == "combined" else [r for r in group if r["solver"] == solver]
                    winner = min(selected, key=lambda r: (-r["repaired_margin"], r["id"]))
                    margin = winner["repaired_margin"]
                    result.append(dict(date=day["date"], setting=config["id"], analysis=analysis,
                        coverage=config["coverage"], radius=config["radius"], multiplier=config["multiplier"],
                        solver=solver, margin=margin, realized_loss=day["realized_loss"],
                        realized_pnl=day["realized_pnl"], breach=bool(day["realized_loss"] > margin),
                        breach_excess=max(0., day["realized_loss"]-margin), winner=winner["id"],
                        reference_margin=reference["margin"], reference_success=reference["success"],
                        reference_global_certificate=reference.get("global_certificate", False),
                        gap_to_local_reference=None if reference["margin"] is None else reference["margin"]-margin))
    return result


def loadGreedy(directory, primary):
    """Load a complete baseline with identical dates and realized losses."""
    path = directory/"greedy_pca.csv"
    if not path.exists():
        return None
    greedy = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    reference = primary[primary.solver == "combined"].sort_values("date").reset_index(drop=True)
    if greedy.date.duplicated().any() or greedy.date.tolist() != reference.date.astype(str).tolist():
        raise ValueError("greedy PCA dates do not match the plotted primary series")
    if not np.isfinite(greedy.margin).all() or (greedy.margin < 0).any():
        raise ValueError("greedy PCA margins must be finite and nonnegative")
    np.testing.assert_allclose(greedy.realized_loss, reference.realized_loss, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(greedy.breach, greedy.realized_loss > greedy.margin)
    return greedy


def reportPortfolio(root, identifier):
    settings = json.loads((root/"settings.json").read_text())
    directory = root/identifier
    days = json.loads((directory/"days.json").read_text())["days"]
    expected = {trial["id"]: trial for batch in json.loads((directory/"schedule.json").read_text())["batches"]
                for trial in batch["trials"]}
    models, encodings, rows = {}, {}, []
    for path in sorted((directory/"batches").glob("*.json")):
        batch = json.loads(path.read_text())
        for row in batch["rows"]:
            trial = expected.pop(row["id"])
            if any(row[k] != v for k, v in trial.items()):
                raise ValueError("trial differs from its predetermined schedule")
            if row["parameters"]["runs"] != settings["runs"] or row["parameters"]["steps"] != settings["steps"]:
                raise ValueError("solver execution parameters differ from settings")
            day = days[row["day"]]
            if not day["calibration_end"] < row["date"] or day["calibration_observations"] != 125:
                raise ValueError("invalid calibration window")
            if row["day"] not in models:
                models[row["day"]] = loadModel(directory/"models"/f"{row['day']:03d}.npz")
            model, objective = models[row["day"]]
            key = row["day"], row["radius"], row["multiplier"]
            if key not in encodings:
                encodings[key] = buildTrial(objective, 8, row["multiplier"], row["radius"], row["seed_offset"])[0]
            encoded = encodings[key]
            if not encoded.penalty > encoded.objectiveRangeBound:
                raise ValueError("penalty does not satisfy the sufficient global-optimum bound")
            raw = np.load(directory/"samples"/(row["id"]+".npy"), allow_pickle=False)
            info = encoded.diagnostics(raw)
            if info["encoding_feasible"] != row["encoding_feasible"]:
                raise ValueError("raw feasibility mismatch")
            np.testing.assert_allclose(float(model.pnl(encoded.coordinates(raw))), row["raw_repriced_pnl"],
                                       atol=1e-12, rtol=1e-12)
            sample = np.load(directory/"repaired_samples"/(row["id"]+".npy"), allow_pickle=False)
            if not encoded.diagnostics(sample)["encoding_feasible"]:
                raise ValueError("repair returned an infeasible encoding")
            np.testing.assert_array_equal(encoded.integerCoordinates(sample), row["repaired_integers"])
            pnl = float(model.pnl(encoded.coordinates(sample)))
            np.testing.assert_allclose(pnl, row["repaired_pnl"], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(max(0., -pnl), row["repaired_margin"], atol=1e-12, rtol=1e-12)
            if (day["realized_loss"] > max(0., -pnl)) != row["repaired_breach"]:
                raise ValueError("direct breach classification mismatch")
            rows.append(row)
    if expected or len(rows) != settings["expected_trials_per_portfolio"]:
        raise ValueError("incomplete portfolio trial coverage")
    daily = dailyRows(days, rows, settings["settings"])
    writeCsv(directory/"daily_margins.csv", daily)
    frame = pd.DataFrame(daily)
    summary = []
    for keys, group in frame.groupby(["analysis", "coverage", "multiplier", "solver"], sort=True):
        analysis, coverage, multiplier, solver = keys
        breaches = int(group.breach.sum())
        summary.append(dict(portfolio=identifier, analysis=analysis, coverage=coverage, multiplier=multiplier,
            solver=solver, dates=len(group), breaches=breaches, breach_rate=breaches/len(group),
            mean_margin=float(group.margin.mean()), max_loss=float(group.realized_loss.max()),
            max_breach_excess=float(group.breach_excess.max()),
            breach_rate_upper_95=breachUpperBound(breaches, len(group)),
            target_breach_rate=1-coverage,
            binomial_excess_pvalue=binomtest(breaches, len(group), 1-coverage, alternative="greater").pvalue,
            mean_gap_to_local_reference=None if group.gap_to_local_reference.isna().all() else float(group.gap_to_local_reference.mean()),
            reference_failures=int((~group.reference_success).sum())))
    writeCsv(directory/"summary.csv", summary)
    trial_table = [{k: r[k] for k in ("id", "date", "setting", "coverage", "radius", "multiplier", "solver", "repeat", "gpu",
        "variables", "edges", "penalty", "sufficient_penalty", "encoding_feasible", "scenario_feasible",
        "repaired_margin", "repaired_breach", "projected_margin", "repair_converged", "repair_steps", "repair_seconds",
        "realized_loss", "realized_pnl", "amortized_solve_seconds", "qubo_build_seconds", "validation_seconds", "repricing_seconds")} for r in rows]
    writeCsv(directory/"trials.csv", trial_table)
    verification = dict(status="passed", trials=len(rows), dates=len(days), raw_feasible=sum(r["encoding_feasible"] for r in rows),
        repaired_feasible=len(rows), sufficient_penalties=True, runs=settings["runs"], steps=settings["steps"],
        gpu_indices=sorted({r["gpu"] for r in rows}), repair_converged=sum(r["repair_converged"] for r in rows),
        total_repair_work_seconds=sum(r["repair_seconds"] for r in rows),
        min_variables=min(r["variables"] for r in rows), max_variables=max(r["variables"] for r in rows),
        min_edges=min(r["edges"] for r in rows), max_edges=max(r["edges"] for r in rows))
    writeJson(directory/"verification.json", verification)
    writeJson(directory/"status.json", dict(stage="complete", **verification))
    plotPortfolio(root, identifier)


def plotPortfolio(root, identifier):
    """Refresh charts from verified daily records without rerunning solvers."""
    directory = root/identifier
    frame = pd.read_csv(directory/"daily_margins.csv")
    summary = pd.read_csv(directory/"summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = frame[frame.analysis == "primary_best3"]
    greedy = loadGreedy(directory, primary)
    fig, axis = plt.subplots(figsize=(13, 5), layout="constrained")
    for solver in (*SOLVERS, "combined"):
        data = primary[primary.solver == solver].sort_values("date")
        axis.plot(pd.to_datetime(data.date), data.margin*100, label=solver.replace("torch_", ""), lw=1.2)
    data = primary[primary.solver == "combined"].sort_values("date")
    if greedy is not None:
        axis.plot(pd.to_datetime(greedy.date), greedy.margin*100, label="Greedy PCA (105 scenarios)",
                  color="#be185d", lw=1.5, linestyle="--")
        greedy_breaches = greedy[greedy.breach]
        axis.scatter(pd.to_datetime(greedy_breaches.date), greedy_breaches.realized_loss*100,
                     edgecolors="#be185d", facecolors="none", marker="s", s=35,
                     label="Greedy breach", zorder=6)
        merged = primary.copy()
        baseline = greedy.assign(solver="greedy_pca", analysis="original_grid_baseline")
        pd.concat([merged, baseline], ignore_index=True).to_csv(directory/"margins_with_greedy.csv", index=False)
    axis.plot(pd.to_datetime(data.date), data.realized_loss*100, color="black", label="Realized loss (clipped at 0)", lw=.9)
    breached = data[data.breach]
    axis.scatter(pd.to_datetime(breached.date), breached.realized_loss*100, color="red", s=20, label="Combined breach", zorder=5)
    highest = max(data.margin.max(), data.realized_loss.max(), 0 if greedy is None else greedy.margin.max())
    if highest > 10*data.margin.median():
        axis.set_yscale("symlog", linthresh=1.)
        axis.set_ylabel("Margin/loss (%) · symmetric log scale above 1%")
    else:
        axis.set_ylabel("Margin/loss (% of initial capital unit)")
    axis.set_ylim(bottom=0)
    axis.set_title(f"{identifier}: joint-factor nominal 99.95% · 8-bit coordinates"
                   +("\nGreedy PCA: original 105-scenario grid (different scenario region)" if greedy is not None else ""))
    axis.legend(ncol=3, fontsize=9)
    axis.grid(alpha=.2)
    fig.savefig(directory/"margin_loss.png", dpi=160)
    plt.close(fig)
    # Sensitivities use the same seed across settings, unlike the primary
    # three-seed margin. Keep the two comparisons visibly distinct.
    sensitivity = pd.DataFrame(summary)
    paired = sensitivity[(sensitivity.analysis == "paired_seed0") & (sensitivity.solver == "combined")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for column, (subset, key, title) in enumerate((
            (paired[paired.multiplier == 1.].sort_values("coverage"), "coverage", "Nominal Gaussian coverage · penalty multiplier 1"),
            (paired[paired.coverage == .9995].sort_values("multiplier"), "multiplier", "Penalty multiplier · nominal coverage 99.95%"))):
        positions = np.arange(len(subset))
        labels = [f"{v:.2%}" if key == "coverage" else f"{v:g}" for v in subset[key]]
        axes[0, column].plot(positions, subset.mean_margin*100, marker="o", color="#2563eb")
        axes[1, column].bar(positions, subset.breaches, color="#c2410c", width=.55)
        axes[0, column].set_title(title, fontsize=10)
        for row_index in range(2):
            axes[row_index, column].set_xticks(positions, labels)
            axes[row_index, column].grid(axis="y", alpha=.2)
        axes[0, column].set_ylabel("Mean margin (%)")
        axes[1, column].set_ylabel("Breach dates / 294")
    fig.suptitle(f"{identifier} · paired seed-0 sensitivity comparisons")
    fig.savefig(directory/"confidence_penalty.png", dpi=160)
    plt.close(fig)


def reportAll(root):
    settings = json.loads((root/"settings.json").read_text())
    summaries, primary = [], []
    for metadata in settings["portfolios"]:
        path = root/metadata["id"]/"summary.csv"
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        summaries.append(rows)
        selected = rows[(rows.analysis == "primary_best3") & (rows.solver == "combined")].iloc[0].to_dict()
        selected.update(name=metadata["name"], gross=metadata["gross"], net=metadata["net"], positions=metadata["positions"])
        primary.append(selected)
    if not summaries:
        return
    all_summaries = pd.concat(summaries, ignore_index=True)
    all_summaries.to_csv(root/"sweep_summary.csv", index=False)
    writeCsv(root/"summary.csv", primary)
    paired = all_summaries[(all_summaries.analysis == "paired_seed0") & (all_summaries.solver == "combined")]
    penalties = paired[paired.coverage == .9995].pivot(index="portfolio", columns="multiplier", values="breaches")
    unchanged = int((penalties.nunique(axis=1) == 1).sum())
    verification = [json.loads((root/r["portfolio"]/"verification.json").read_text()) for r in primary]
    findings = (f"At nominal 99.95%, penalty multipliers 1, 10 and 100 produced identical paired-seed breach counts "
                f"for **{unchanged}/{len(primary)} portfolios**. Across all settings and solvers, "
                f"**{sum(v['raw_feasible'] for v in verification):,}/{sum(v['trials'] for v in verification):,}** "
                f"saved raw solver outputs were encoding-feasible; "
                f"**{sum(v['repaired_feasible'] for v in verification):,}** outputs were feasible after repair. "
                "These counts concern the saved result of each solve, not every internal trajectory. "
                "Every QUBO had **122 variables and 7,381 edges**.")
    timing_text = ""
    if (root/"timings.json").exists():
        timing = json.loads((root/"timings.json").read_text())
        timing_rows = []
        for item in timing["portfolios"]:
            trials = pd.read_csv(root/item["portfolio"]/"trials.csv")
            row = {key: value for key, value in item.items() if key != "workers"}
            for key in ("qubo_build_seconds", "validation_seconds", "repricing_seconds", "repair_seconds"):
                row["summed_"+key] = float(trials[key].sum())
            timing_rows.append(row)
        writeCsv(root/"timing_summary.csv", timing_rows)
        timing_text = f"""## Execution timing

The complete remote run took **{timing['total_wall_seconds']/60:.2f} minutes** including shared preparation, all ten sequential portfolios, continuous reference calculations, GPU solving, repair, verification and remote reporting. Local transfers, independent source-price audits and local plotting are additional and overlap remote execution.

[Timing breakdown by portfolio](timing_summary.csv) reports reference and worker-phase wall time separately. Summed solve, QUBO construction, validation, repricing and repair times are work across concurrent workers; they overlap the worker-phase wall time and must not be added to it. `portfolio_wall_seconds` includes references and the worker phase but excludes the separately recorded `verification_report_seconds`. Shared preparation is recorded once in `shared_timing.json`.

"""
    greedy_text = ""
    if (root/"greedy_pca"/"summary.csv").exists():
        greedy_summary = pd.read_csv(root/"greedy_pca"/"summary.csv")
        greedy_table = "\n".join(f"| {r.portfolio} | {r.breaches} / {r.dates} | {r.mean_margin:.2%} |"
                                 for r in greedy_summary.itertuples())
        greedy_text = f"""## Greedy PCA comparison

The updated overview and individual margin/loss charts include the original greedy PCA baseline (dashed magenta). It uses full-universe float64 PCA, the same 125-return EW window and fixed portfolio weights, 21×5 PCA scenarios, 21 return bins, residual sigma range 5 and the original distance inflation and empty-bin fallback settings. The PCA axes span ±10 and ±2 component standard deviations respectively. Within each scenario, long positions select the lowest available return and shorts select the highest. The maximum loss across all 105 scenarios is the margin. This independently bounded asset scenario model differs from the joint-factor nominal 99.95% region; no 99.95% confidence claim is attached to the greedy baseline.

| Portfolio | Greedy breaches / dates | Mean greedy margin |
|---|---:|---:|
{greedy_table}

The remote run shares each date's PCA and scenario stream across all ten portfolios. [Greedy daily margins](greedy_pca/daily_margins.csv), [summary](greedy_pca/summary.csv), [timings](greedy_pca/timings.json), [daily stage timings](greedy_pca/daily_timings.csv), and [independent verification](greedy_pca/verification.json) are saved separately. Per-portfolio `margins_with_greedy.csv` contains the joint-factor series and the new baseline; the baseline has no joint-factor penalty or coverage setting. Saved per-scenario return bounds are independently repriced after fetching, with dates and realized losses matched to the original experiment.

"""
    table = "\n".join(f"| [{r['portfolio']}]({r['portfolio']}/margin_loss.png) | {r['name']} | {r['gross']:.1f} | {r['net']:.1f} | {r['breaches']} / {r['dates']} | {r['mean_margin']:.2%} | {r['max_loss']:.2%} |" for r in primary)
    text = f"""# Ten risky Group 1 portfolios

Completed **{len(primary)}/10 portfolios**, sequentially, with eight GPUs per portfolio. Every portfolio uses the same **294 dates, 2025-07-08 through 2026-09-04**, a 125-return EW window (decay 0.93), 8-bit coordinates, and 32 trajectories × 10,000 steps per solve. The holdings are fixed, while return exposures are rebalanced to those weights each observation interval; financing, borrow fees and transaction costs are excluded.

## Primary 99.95% nominal scenario region

| Portfolio | Design | Gross | Net | Breaches / dates | Mean margin | Maximum realized loss |
|---|---|---:|---:|---:|---:|---:|
{table}

Primary daily margin is the maximum feasible repaired loss over SBM, SVL and TRF, each with three seeds. Individual solver results are in each portfolio's `daily_margins.csv` and `summary.csv`. Realized loss is `max(0, -P&L)`; a breach is strictly `loss > margin`. Margin and loss share the same exposure units. Concentration and long/short composition change risk geometry; leverage alone scales both margin and loss and does not automatically create breaches.

{findings}

## Confidence and penalty are separate

For independent standardized Gaussian coordinates, the squared radius follows a chi-square distribution. With three coordinates, `sqrt(chi2.ppf(0.9995, 3)) = 4.2107002065`. This sets **nominal Gaussian scenario-region content**, not verified 99.95% coverage of actual portfolio losses. EW covariance estimation, non-Gaussian tails, omitted residual directions, finite coordinate precision, and heuristic optimization all affect actual coverage. See [NIST chi-square definition](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm).

The primary QUBO penalty is `1.1 × objectiveRangeBound` (multiplier **1**, replacing `1e-11`). It is sufficient for a feasible global QUBO optimum; an approximate solver can still return an infeasible encoding, so repair remains enabled. Penalty probes use multipliers **10 and 100** at the same radius. Coverage probes use **99%, 99.9%, 99.95%, 99.99%** at multiplier 1. Every paired comparison uses seed 0; the primary three-seed result is reported separately. No parameter is selected by minimizing breaches on the evaluation dates.

There are only 294 market dates per portfolio, not 2,940 independent dates across the ten correlated portfolios. Even zero breaches gives an independent-Bernoulli one-sided 95% upper breach-rate bound of **1.014%**, far above the 0.05% target. About **5,990 independent zero-breach observations** would be required for that particular upper bound to reach 0.05%. [Exact binomial interval method](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm). The tables report bounds and one-sided excess-breach p-values; they do not declare 99.95% coverage certified.

## Portfolio construction and numerical checks

All holdings were chosen using only **2025-01-02 through 2025-07-07**. Selection favors high log-return volatility, residual volatility or PCA loading. Pre-period filters require minimum price ≥1, last price ≥5, and no absolute one-day log return exceeding log(2). This reduces known historical tiny-price artifacts without inspecting future outcomes. Subsequent extreme moves are retained and listed in `large_asset_moves.csv`.

PCA is fitted on all 8,590 stocks once per date and shared across portfolios. The residual direction is recomputed for each portfolio. Only then are zero-exposure assets removed from scenario repricing; full and compact models are checked for identical P&L. Every saved repaired sample is checked against exact integer product/slack/radius constraints and repriced independently. Continuous references for signed portfolios are multistart local solutions, not global certificates. Exhaustive quadratic lattice benchmarks are omitted in this experiment.

{greedy_text}{timing_text}## Files

- [Primary combined results](summary.csv), [all confidence and penalty comparisons](sweep_summary.csv), [portfolio definitions](portfolios.csv).
- [Portfolio overview plot](portfolio_overview.png), [realized risk comparison](realized_risk_comparison.csv), [individual breach diagnostics](breach_diagnostics.csv).
- Each portfolio directory contains holdings, fitted models, raw and repaired binary samples, daily margins, trial timings, verification and a margin/loss chart.
- [Shared preparation timing](shared_timing.json), [portfolio timings](portfolio_timings.json), [settings](settings.json), [status](status.json).
- [Independent source-price and raw-sample audit](market_data_verification.json). Large-move flags identify supplied asset prices changing by more than a factor of two; they do not automatically classify a price as erroneous or remove a breach.
"""
    (root/"README.md").write_text(text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--portfolio")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.plots_only:
        for item in json.loads((args.root/"settings.json").read_text())["portfolios"]:
            if args.portfolio is None or args.portfolio == item["id"]:
                plotPortfolio(args.root, item["id"])
    elif args.portfolio:
        reportPortfolio(args.root, args.portfolio)
    reportAll(args.root)

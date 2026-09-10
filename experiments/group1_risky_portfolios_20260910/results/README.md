# Ten risky Group 1 portfolios

Completed **10/10 portfolios**, sequentially, with eight GPUs per portfolio. Every portfolio uses the same **294 dates, 2025-07-08 through 2026-09-04**, a 125-return EW window (decay 0.93), 8-bit coordinates, and 32 trajectories × 10,000 steps per solve. The holdings are fixed, while return exposures are rebalanced to those weights each observation interval; financing, borrow fees and transaction costs are excluded.

## Primary 99.95% nominal scenario region

| Portfolio | Design | Gross | Net | Breaches / dates | Mean margin | Maximum realized loss |
|---|---|---:|---:|---:|---:|---:|
| [p01](p01/margin_loss.png) | High-volatility 50 long | 1.0 | 1.0 | 1 / 294 | 9.56% | 8.16% |
| [p02](p02/margin_loss.png) | High-volatility 10 long | 1.0 | 1.0 | 0 / 294 | 15.77% | 12.53% |
| [p03](p03/margin_loss.png) | Residual-volatility 10 long | 1.0 | 1.0 | 2 / 294 | 12.80% | 17.82% |
| [p04](p04/margin_loss.png) | PCA-factor concentrated 10 long | 1.0 | 1.0 | 1 / 294 | 11.84% | 13.03% |
| [p05](p05/margin_loss.png) | High-volatility 20 short | 1.0 | -1.0 | 4 / 294 | 277.24% | 241.25% |
| [p06](p06/margin_loss.png) | Residual-volatility 10 short | 1.0 | -1.0 | 3 / 294 | 161.91% | 492.11% |
| [p07](p07/margin_loss.png) | High-volatility 20/20 long-short | 2.0 | 0.0 | 5 / 294 | 47.19% | 79.38% |
| [p08](p08/margin_loss.png) | Concentrated 5/5 long-short | 2.0 | 0.0 | 1 / 294 | 41.47% | 76.53% |
| [p09](p09/margin_loss.png) | Residual-volatility 20/20 long-short | 4.0 | -0.0 | 5 / 294 | 425.13% | 479.53% |
| [p10](p10/margin_loss.png) | Short-biased 10/30 long-short | 3.0 | -1.0 | 4 / 294 | 362.82% | 323.32% |

Primary daily margin is the maximum feasible repaired loss over SBM, SVL and TRF, each with three seeds. Individual solver results are in each portfolio's `daily_margins.csv` and `summary.csv`. Realized loss is `max(0, -P&L)`; a breach is strictly `loss > margin`. Margin and loss share the same exposure units. Concentration and long/short composition change risk geometry; leverage alone scales both margin and loss and does not automatically create breaches.

At nominal 99.95%, penalty multipliers 1, 10 and 100 produced identical paired-seed breach counts for **10/10 portfolios**. Across all settings and solvers, **0/70,560** saved raw solver outputs were encoding-feasible; **70,560** outputs were feasible after repair. These counts concern the saved result of each solve, not every internal trajectory. Every QUBO had **122 variables and 7,381 edges**.

## Confidence and penalty are separate

For independent standardized Gaussian coordinates, the squared radius follows a chi-square distribution. With three coordinates, `sqrt(chi2.ppf(0.9995, 3)) = 4.2107002065`. This sets **nominal Gaussian scenario-region content**, not verified 99.95% coverage of actual portfolio losses. EW covariance estimation, non-Gaussian tails, omitted residual directions, finite coordinate precision, and heuristic optimization all affect actual coverage. See [NIST chi-square definition](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm).

The primary QUBO penalty is `1.1 × objectiveRangeBound` (multiplier **1**, replacing `1e-11`). It is sufficient for a feasible global QUBO optimum; an approximate solver can still return an infeasible encoding, so repair remains enabled. Penalty probes use multipliers **10 and 100** at the same radius. Coverage probes use **99%, 99.9%, 99.95%, 99.99%** at multiplier 1. Every paired comparison uses seed 0; the primary three-seed result is reported separately. No parameter is selected by minimizing breaches on the evaluation dates.

There are only 294 market dates per portfolio, not 2,940 independent dates across the ten correlated portfolios. Even zero breaches gives an independent-Bernoulli one-sided 95% upper breach-rate bound of **1.014%**, far above the 0.05% target. About **5,990 independent zero-breach observations** would be required for that particular upper bound to reach 0.05%. [Exact binomial interval method](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm). The tables report bounds and one-sided excess-breach p-values; they do not declare 99.95% coverage certified.

## Portfolio construction and numerical checks

All holdings were chosen using only **2025-01-02 through 2025-07-07**. Selection favors high log-return volatility, residual volatility or PCA loading. Pre-period filters require minimum price ≥1, last price ≥5, and no absolute one-day log return exceeding log(2). This reduces known historical tiny-price artifacts without inspecting future outcomes. Subsequent extreme moves are retained and listed in `large_asset_moves.csv`.

PCA is fitted on all 8,590 stocks once per date and shared across portfolios. The residual direction is recomputed for each portfolio. Only then are zero-exposure assets removed from scenario repricing; full and compact models are checked for identical P&L. Every saved repaired sample is checked against exact integer product/slack/radius constraints and repriced independently. Continuous references for signed portfolios are multistart local solutions, not global certificates. Exhaustive quadratic lattice benchmarks are omitted in this experiment.

## Greedy PCA comparison

The updated overview and individual margin/loss charts include the original greedy PCA baseline (dashed magenta). It uses full-universe float64 PCA, the same 125-return EW window and fixed portfolio weights, 21×5 PCA scenarios, 21 return bins, residual sigma range 5 and the original distance inflation and empty-bin fallback settings. The PCA axes span ±10 and ±2 component standard deviations respectively. Within each scenario, long positions select the lowest available return and shorts select the highest. The maximum loss across all 105 scenarios is the margin. This independently bounded asset scenario model differs from the joint-factor nominal 99.95% region; no 99.95% confidence claim is attached to the greedy baseline.

| Portfolio | Greedy breaches / dates | Mean greedy margin |
|---|---:|---:|
| p01 | 0 / 294 | 42.38% |
| p02 | 0 / 294 | 47.97% |
| p03 | 0 / 294 | 46.14% |
| p04 | 0 / 294 | 38.03% |
| p05 | 0 / 294 | 317746.23% |
| p06 | 0 / 294 | 1437.04% |
| p07 | 0 / 294 | 378.80% |
| p08 | 0 / 294 | 215.29% |
| p09 | 0 / 294 | 634758.00% |
| p10 | 0 / 294 | 423991.28% |

The remote run shares each date's PCA and scenario stream across all ten portfolios. [Greedy daily margins](greedy_pca/daily_margins.csv), [summary](greedy_pca/summary.csv), [timings](greedy_pca/timings.json), [daily stage timings](greedy_pca/daily_timings.csv), and [independent verification](greedy_pca/verification.json) are saved separately. Per-portfolio `margins_with_greedy.csv` contains the joint-factor series and the new baseline; the baseline has no joint-factor penalty or coverage setting. Saved per-scenario return bounds are independently repriced after fetching, with dates and realized losses matched to the original experiment.

## Execution timing

The complete remote run took **41.10 minutes** including shared preparation, all ten sequential portfolios, continuous reference calculations, GPU solving, repair, verification and remote reporting. Local transfers, independent source-price audits and local plotting are additional and overlap remote execution.

[Timing breakdown by portfolio](timing_summary.csv) reports reference and worker-phase wall time separately. Summed solve, QUBO construction, validation, repricing and repair times are work across concurrent workers; they overlap the worker-phase wall time and must not be added to it. `portfolio_wall_seconds` includes references and the worker phase but excludes the separately recorded `verification_report_seconds`. Shared preparation is recorded once in `shared_timing.json`.

## Files

- [Primary combined results](summary.csv), [all confidence and penalty comparisons](sweep_summary.csv), [portfolio definitions](portfolios.csv).
- [Portfolio overview plot](portfolio_overview.png), [realized risk comparison](realized_risk_comparison.csv), [individual breach diagnostics](breach_diagnostics.csv).
- Each portfolio directory contains holdings, fitted models, raw and repaired binary samples, daily margins, trial timings, verification and a margin/loss chart.
- [Shared preparation timing](shared_timing.json), [portfolio timings](portfolio_timings.json), [settings](settings.json), [status](status.json).
- [Independent source-price and raw-sample audit](market_data_verification.json). Large-move flags identify supplied asset prices changing by more than a factor of two; they do not automatically classify a price as erroneous or remove a breach.

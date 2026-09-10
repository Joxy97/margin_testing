# Group 1 factor-stress Cartesian sweep

Completed 8,640 trials on 8 GPUs for all 8,590 stocks. Every solve uses 32 trajectories and 10,000 steps. The sweep combines bits [4, 6, 8], 16 penalty multipliers, three solvers, 3 seeds and 20 evaluation dates (2026-08-10 to 2026-09-04).

The fixed random long-only portfolio uses seed 20260910 and total exposure 1. Each date fits two PCA factors and one portfolio residual direction using 125 strictly prior return observations with EW decay 0.93. Radius 3 remains illustrative.

## Penalties and graph sizes

Penalty multipliers are zero and powers of ten from 1e-12 through 1e2. Both constraint penalties use `P = multiplier * 1.1 * objectiveRangeBound`. P&L coefficients remain fixed. The conservative global-optimum feasibility guarantee applies only when P exceeds the objective range bound. Smaller values are diagnostic.

| Bits/coordinate | Scenario | Product | Slack | Variables | Nonzero-penalty edges | Zero-penalty edges |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 12 | 18 | 6 | 36 | 630 | 66 |
| 6 | 18 | 45 | 10 | 73 | 2628 | 153 |
| 8 | 24 | 84 | 14 | 122 | 7381 | 276 |

Counts are measured after coefficient aggregation. At zero penalty, auxiliary and slack bits remain declared but are disconnected.

## Feasibility and breaches

**10/8,640 returned encodings were fully feasible.** There were 0 direct breaches among valid margins and 8,630 missing margins due to invalid encodings. 0 solver/penalty/resolution/seed series had a valid margin for every evaluation date.

A breach is exactly `realized_loss > margin`, using the preceding available close to the evaluation-date close. Gains have negative realized loss. No historical floor, repair, reference fallback or warm-start margin replaces a solver result. Empty breach fields mean missing margins, not successful coverage. Each row in `breaches_by_seed.csv` is one 20-date series; pooled repeated-date trials are not independent observations and these 20 dates do not establish tail calibration.

| Reference | Dates | Breaches | Mean margin |
|---|---:|---:|---:|
| linear_analytic | 20 | 0 | 0.994447% |
| quadratic_continuous | 20 | 0 | 1.005604% |
| exact_repricing_continuous | 20 | 0 | 1.006147% |

## Timing

Measured total benchmark wall time: **266.120 s**. Shared preparation: **40.347 s**. Worker phase, including worker startup and warmup: **225.008 s**. Reporting: **0.502 s**. The shell timer in `run.log` also includes interpreter/import startup. SSH, transfer and installation are excluded.

Peak concurrency was **256 QUBOs**. The sum of synchronized batch solve times was 1687.516 s across GPUs. Parallel work sums are not wall time. Trial solve times are batch time divided by 32, explicitly amortized throughput costs rather than individual request latency. Candidate scoring is included in the solver call. Constraint validation and exact exponential repricing are measured separately after solving.

| Shared stage, all dates | Seconds |
|---|---:|
| Metadata and configuration | 1.463 |
| Prefetch acquisition | 5.136 |
| data_acquisition_seconds | 0.289 |
| exact_repricing_continuous_seconds | 0.151 |
| factor_model_seconds | 0.184 |
| historical_stress_seconds | 0.168 |
| lattice_4_seconds | 0.098 |
| lattice_6_seconds | 1.756 |
| lattice_8_seconds | 29.230 |
| linear_analytic_seconds | 0.001 |
| model_write_seconds | 0.041 |
| objective_seconds | 0.006 |
| pca_fit_seconds | 1.192 |
| quadratic_continuous_seconds | 0.217 |
| realized_pnl_seconds | 0.355 |

| Trial work, summed across workers | Seconds |
|---|---:|
| summed_qubo_build_seconds | 4.978 |
| summed_validation_seconds | 1.513 |
| summed_repricing_seconds | 0.490 |
| summed_sample_write_seconds | 1.410 |

Exact lattice and continuous reference timings above are benchmark validation overhead, not required work for a heuristic-only margin calculation. Per-date stage details are in `preparation_stages.csv` and `days.json`.

## Files

- [Grouped sweep results](results/summary.csv) and [all trials](results/trials.csv).
- [Breaches by seed](results/breaches_by_seed.csv), [daily reference margins](results/daily_references.csv).
- [Graph sizes](results/graph_sizes.csv), [timings](results/timings.json), [verification](results/verification.json).
- [Model and encoding guide](../../docs/benchmarks/factor_stress.md).

The archive includes fitted models, every binary sample, the fixed portfolio, configuration, complete schedule and batch records. Verification reconstructs every QUBO, checks every saved sample, reprices it, and recomputes breaches.

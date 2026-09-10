# Joint factor stress QUBO

This experimental path replaces the outer PCA scenario grid and per-asset
one-hot choices with a single joint stress model. It is a reusable Python API
and benchmark CLI, not a replacement for the application's existing YAML
margin calculators. Market data still flows through `MarginEngine` acquisition.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python tools/benchmark_factor_stress.py \
  --group yahoo_equities_grouped/groups/US/group_49_102.csv \
  --bits 4 6 8 --radius 3 --steps 2000 --runs 64 --repeats 3 \
  --output experiments/factor_stress_20260910/results
PYTHONPATH=src .venv/bin/python -m unittest tests.test_factor_stress
```

Choose a fresh output directory when repeating a completed experiment. The CLI
prints variable and edge counts before solving each model and writes atomic
status/partial-result JSON. `summary.csv` contains solver feasibility and timing;
`results.json` also contains continuous/lattice references, settings, and Taylor
error diagnostics. It saves the random portfolio, acquisition YAML, fitted
model, and every returned sample. It does not export large QUBO arrays.

## Scenario model

`risk_state_generator.factor_stress_model.FactorStressModel` owns the scenario
math. It consumes the existing EW-standardized log-return `ReturnsPCAGrid` and
canonical portfolio exposures. All calibration observations precede the as-of
date. With log-return center `c` and directions `A`, actual portfolio P&L is

```text
log_return(z) = c + A z
PnL(z) = sum_i exposure_i * expm1(log_return_i(z))
```

The first two columns of `A` are PCA loadings multiplied by the asset log-return
scales and square roots of the component eigenvalues. The third is the residual
direction `R v / sqrt(v.T R v)`, where `v = exposure * exp(c)` and `R` is the EW
covariance of residual log returns. Matrix-vector multiplication is performed
through the observation matrix; no asset-by-asset covariance is materialized.
Zero residual variance produces a zero column.

This direction preserves the fixed portfolio's **local linear** residual
variance. It does not preserve the entire residual distribution or the nonlinear
worst case over all omitted residual directions. The model is portfolio-specific.
Short positions are accepted, but exact-repricing convexity and its certificate
are only claimed for nonnegative exposures.

The stress set is `z.T z <= radius**2`. Radius is independent of binary
resolution. Default radius 3 is an illustrative stress budget, not a calibrated
confidence level or a claim about VaR/ES. The benchmark also measures the worst
observed portfolio loss in the fitting window. That check can exceed the reduced
model's stress margin; it is not an out-of-sample coverage test.

## Objective and exact integer constraint encoding

`margin_calculator.optimization.factor_stress` owns the QUBO representation. The
objective is the second-order Taylor expansion at zero:

```text
F(z) = p0 + g.T z + 0.5 z.T H z
p0 = exposure.T expm1(c)
g = A.T (exposure * exp(c))
H = A.T diag(exposure * exp(c)) A
```

For `k` bits per coordinate, let `m = 2**(k-1)-1`, integer
`t_a = sum_j 2**j b_aj - m`, and `z_a = radius * t_a / m`. This includes zero and
both axis endpoints. The unsigned encoding has one extra level `t=m+1`; it is
excluded by the exact ball constraint. Supported resolutions are 2–8 bits.

The radius inequality is exactly `sum_a t_a**2 <= m**2`, with no rounding of
PCA coefficients or the stress boundary. Each within-coordinate bit product is
replaced by a shared auxiliary bit `y_ij`. The resulting budget expression `q`
is linear in original and product bits. Binary nonnegative slack spans at least
`0..m**2`; surplus representable slack values cannot satisfy the equality.

```text
E(b, y, s) = F(z(b)) + P * (q(b,y) + s - m**2)**2
                      + P * sum_ij [b_i*b_j - 2*b_i*y_ij - 2*b_j*y_ij + 3*y_ij]
```

The product penalty is nonnegative and vanishes exactly when `y_ij=b_i*b_j`.
The squared budget expression has integer residual. Thus every invalid encoding
has total unweighted penalty at least 1; every feasible lattice point has a
zero-penalty extension. `P = penaltySafety * max(objectiveRangeBound, eps)` with
default safety 1.1, where the bound sums the absolute nonconstant QUBO objective
coefficients **before** adding penalties. This bounds objective variation over
all original bitstrings. In exact arithmetic, every global minimizer therefore
satisfies the constraints and minimizes F over the lattice. The optional
`penaltyMultiplier` scales both constraint penalties together without changing
the P&L objective; it defaults to 1. Zero removes the penalties. In a sweep,
the sufficient-bound guarantee applies only when the resulting P exceeds the
objective range bound. Weaker penalties remain diagnostic experiments.

The original P&L objective is not replaced by a likelihood penalty. The radius
is a hard encoded constraint. Solver copies are divided by one positive global
coefficient scale for dynamics; original coefficients and scoring remain intact.

The model has no one-hot groups. Existing candidate repair consequently does
not enforce these product/budget constraints. `FactorStressQUBO.diagnostics`
checks them independently in integer arithmetic. The benchmark accepts a solver
margin only if **all** constraints pass. It reports `scenario_feasible`
separately: a scenario can lie inside the ball even when its auxiliary encoding
is invalid. No repair, reference seeding, or reference fallback is included in
reported heuristic solutions. Existing SAC policies were not transferred.

## References and numerical limits

`factor_stress_reference` supplies:

- The analytical linear minimum `p0 - radius * norm(g)`.
- A continuous quadratic reference on the same ball.
- A continuous reference with exact `expm1` repricing of this reduced model.

Continuous numerical references use SLSQP with analytic gradients. Convex cases
report a global supporting-hyperplane lower bound:
`f(z) - grad(f)(z).T z - radius * norm(grad(f)(z))`. The gap to this bound certifies
the solution to numerical precision. Nonconvex cases use multiple starts and
explicitly return no global certificate.

`solveLattice` is an exact **quadratic-objective** reference for up to three
coordinates. It enumerates the first d-1 integer coordinates and checks the
endpoints and adjacent stationary-point integers along the last coordinate.
This also handles indefinite Hessians. It does not enumerate auxiliary bits,
and it does not claim to minimize exact exponential P&L over the discrete grid.
The chosen scenario is subsequently repriced exactly.

Squaring the budget creates dense couplings and large coefficients. Expanded
float64 QUBO energy can lose precision when penalty terms cancel. Diagnostics
also evaluate the unexpanded objective plus integer penalties and report the
difference. Mathematical penalty equivalence is not a finite-precision or
finite-iteration solver guarantee. Increasing resolution can worsen conditioning
even while improving the exact lattice approximation.

Tests exhaustively compare every bitstring of a small QUBO with its unexpanded
polynomial, verify its global feasible optimum, compare lattice search with full
enumeration, check boundary/slack/product violations, and validate no-lookahead,
local residual variance, repricing derivatives, and continuous certificates.

Measured results and interpretation: [102-stock experiment](../../experiments/factor_stress_20260910/README.md).

## Parallel rolling penalty sweep

`tools/sweep_factor_stress.py` builds one model per evaluation date and runs the
Cartesian product of resolutions, penalty multipliers, solvers and repeated
seeds. All repeated seeds are paired across penalties and independent of GPU
assignment. Each GPU has one persistent process using `solveMany` batches;
solver types are interleaved in the work queue. Float64 dynamics and original
integer constraint validation are retained. A batch is rejected if its solver
working-memory estimate exceeds 70% of free device memory.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/sweep_factor_stress.py \
  --group yahoo_equities_grouped/groups/US/group_1_8590.csv \
  --output experiments/group1_factor_cartesian_20260910/results \
  --bits 4 6 8 --runs 32 --steps 10000 --days 20 --repeats 3 \
  --batch-size 32 --devices 0 1 2 3 4 5 6 7
```

Default multipliers are zero and powers of ten from `1e-12` through `1e2`,
relative to the baseline sufficient penalty. The command creates 8,640 trials.
The portfolio is fixed across dates. `MarginEngine.prepareBacktest` acquires the
union of required history, and per-date access uses `getPortfolioMarketData`.
PCA uses only prices strictly before each evaluation date. Realized P&L spans
the preceding available close to the evaluation-date close, matching the
repository backtest convention. A breach is exactly `realized_loss > margin`.
Invalid encodings have no accepted margin or breach classification.

`days.json` records acquisition, PCA, factor model, objective, reference,
realized-P&L and model-write timings per date. `timings.json` separates total
wall time from summed worker work. Batched solve time divided by batch size is
explicitly reported as amortized time, not individual latency. `trials.csv`
reports source energy, integer feasibility, direct breaches, and per-trial
encoding/validation/repricing timings. `summary.csv` groups by resolution,
penalty and solver; repeated-date observations are not independent backtest
dates. `samples/` and `batches/` retain each result for independent verification.

## Repair saved factor-stress encodings

`optimization.factor_stress_repair.FactorStressRepair` implements a separate
postprocessing path. It does not alter the binary solvers or the shared one-hot
repair policy. Run it on the archived results without additional GPU sampling:

```bash
PYTHONPATH=src:tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/repair_factor_sweep.py \
  experiments/group1_factor_cartesian_20260910/results \
  experiments/group1_factor_cartesian_20260910/repair_results
PYTHONPATH=src python -m unittest tests.test_factor_stress_repair
```

First decode integer stress coordinates. If their squared norm is within the
integer budget, retain them exactly and rebuild product/slack bits. Otherwise,
radially project and round each coordinate toward zero. Projection uses
`isqrt((t_i**2 * m**2) // sum(t_j**2))` with the original sign, so intermediate
arithmetic and final feasibility are exact. Projection can change loss; it is
not a nearest-lattice-point or minimum-loss projection claim.

Optional local improvement examines all unit-neighborhood moves: 26 for three
coordinates, including diagonal moves along the ball boundary. It accepts only
feasible moves that lower the exact exponential portfolio P&L by more than the
configured tolerance (default `1e-12`). Each accepted move is repriced directly.
Fixed exponential increments accelerate neighbor scoring; they are reused per
date/resolution, while solutions are never memoized. `maxSteps=0` gives only
projection and auxiliary repair; the default cap is 1024 accepted moves.
`converged` distinguishes a local optimum from hitting that cap. No global
optimality guarantee is made.

`FactorStressRepairResult` preserves the raw, projected and final integer
coordinates, both feasible encodings, each stage's P&L, movement, iteration
counts and stage timings. All exposed arrays are immutable. The archive runner
stores projection-only and improved samples and reports breaches for each as
`realized_loss > margin`. A raw-file digest before/after protects the input
archive, and repair output must be outside it. Input acquisition, PCA and GPU
sampling are not rerun. Repair wall time is additional CPU postprocessing, not
part of the previously measured GPU sweep interval.

## Full Group 1 daily backtest

`experiments/group1_factor_full_20260910/run_remote.sh` runs the fixed 8-bit
configuration on all 294 eligible dates in the 420-row Group 1 fixture. Here
`ew_window=125` means 125 prior daily return observations, requiring 126 prior
closes; the evaluation close is used only for realized P&L. All 8,590 prices are
finite and positive in this fixture. The first eligible date is 2025-07-08 and
the last is 2026-09-04.

The selected multiplier is `1e-11`, which minimized average repaired-margin
shortfall from the continuous reference in the earlier 8-bit sweep. The archived
`plan.json` and `penalty_ranking.csv` record the selection rule and tuning dates.
SBM, SVL and TRF each use 32 trajectories, 10,000 steps and three seeds per date.
Eight persistent GPU workers handle the 2,646 solves; CPU repair and independent
sample verification follow.

After fetching the complete experiment directory, generate the daily report:

```bash
PYTHONPATH=src:tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/report_factor_backtest.py \
  experiments/group1_factor_full_20260910 \
  --group yahoo_equities_grouped/groups/US/group_1_8590.csv
PYTHONPATH=src python -m unittest tests.test_factor_backtest_report
```

The report verifies complete date/solver/seed coverage, dataset identity,
125-return calibration boundaries, and realized P&L against source prices.
Daily margin is the greatest repaired loss over three seeds per solver; the
combined series uses all nine candidates. Breaches count dates, not repeated
trials, and use strict `realized_loss > margin`. The 20 penalty-selection dates
and the other 274 dates are summarized separately. The full run is retrospective
and the radius remains illustrative. Stage timing tables distinguish shared
preparation, parallel solve work, sequential repair, and inclusive wall times.

## Riskier portfolios and nominal confidence sweeps

`tools/run_risky_factor_backtests.py` constructs ten fixed portfolios using only
the 126 prices preceding the first evaluation date. It varies concentration,
PCA/residual exposure, and long/short balance. Selection excludes stocks with
pre-period minimum price below 1, last price below 5, or absolute one-day log
return above log(2). Residual and PCA concentrated long portfolios avoid the
earlier concentrated selections so the ten portfolios remain distinct.

The universe PCA is shared per date; each portfolio gets its own residual
direction. Zero-exposure assets are removed only after model construction, with
full/compact P&L parity checks. Portfolios then execute sequentially on eight
GPU workers. `tools/sweep_factor_stress.py` supports optional per-trial radii
and in-worker repair for this runner; the existing fixed-radius archive sweep
keeps its previous behavior when `repair_samples` is absent.

Nominal confidence maps to the Gaussian scenario radius, not the QUBO penalty:
`radius = sqrt(chi2.ppf(coverage, 3))`. At 99.95% this is 4.2107002065. Primary
penalty multiplier 1 gives `1.1 * objectiveRangeBound`, sufficient for a feasible
global QUBO optimum; approximate solutions still require validation/repair.
The confidence probes are 99%, 99.9%, 99.95% and 99.99%. Penalty probes at
99.95% use multipliers 1, 10 and 100. Primary results use three seeds, while
paired sensitivity comparisons use seed 0 throughout. Each portfolio runs
7,056 QUBOs at 8 bits, 32 trajectories and 10,000 steps.

This Gaussian probability is model-based scenario-region content, not verified
loss coverage. With 294 dates, zero breaches has an independent-Bernoulli
one-sided 95% upper breach-rate bound of about 1.014%; ten correlated portfolios
do not provide ten times as many independent market dates. The experiment
reports bounds and sensitivity curves without selecting parameters on the
evaluation-period breach counts. Continuous references for signed portfolios
are local multistart solutions; SLSQP status and any convex bounds are retained.

The remote wrapper is
`experiments/group1_risky_portfolios_20260910/run_remote.sh`. After a complete
shared preparation, `--resume` skips verified portfolios and restarts an
unfinished portfolio with the same dataset and settings. The local tracker
`tools/track_risky_factor_backtests.py` fetches each completed portfolio via the
configured SSH control socket, checks archive integrity, and generates margin,
clipped-loss and paired sensitivity plots while the next portfolio runs.

```bash
PYTHONPATH=src python -m unittest tests.test_risky_factor_portfolios
PYTHONPATH=src:tools RUN_FACTOR_GPU_TESTS=1 \
  python -m unittest tests.test_factor_sweep_worker
PYTHONPATH=src:tools python tools/audit_risky_factor_data.py \
  experiments/group1_risky_portfolios_20260910/results \
  --group yahoo_equities_grouped/groups/US/group_1_8590.csv
```

The audit recomputes realized returns from source prices and checks each
125-return calibration interval. The GPU integration test exercises all three
binary solvers with differing radii and in-worker repair; it is opt-in.

### Original greedy PCA comparison

`tools/run_risky_greedy_pca.py` adds the existing state-aware greedy calculator's
returns-grid baseline to a completed ten-portfolio experiment. It retains the
full 8,590-stock float64 PCA universe, 125 prior returns, EW decay 0.93, the
original 21×5 component grid, 21 return bins, residual sigma range 5, distance
inflation, and empty-bin fallback. Each date's scenario stream is evaluated for
all ten original signed portfolios through `StateAwareGreedyRiskStateVisitor`.
Dates run in parallel CPU processes on the remote server, with acquisition
through public `MarginEngine` methods. No additional QUBOs or repairs are used.

The 105-scenario region spans ±10 and ±2 component standard deviations and
independent residual choices. It differs from the joint-factor nominal 99.95%
region and does not inherit its confidence label. A breach remains strictly
`max(0, -realized_pnl) > margin`.

```bash
PYTHONPATH=src:tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/run_risky_greedy_pca.py \
  experiments/group1_risky_portfolios_20260910/results \
  --group yahoo_equities_grouped/groups/US/group_1_8590.csv --workers 8
PYTHONPATH=src:tools python tools/run_risky_greedy_pca.py \
  experiments/group1_risky_portfolios_20260910/results --verify-only
PYTHONPATH=src:tools python tools/report_risky_factor_backtests.py \
  experiments/group1_risky_portfolios_20260910/results --plots-only
PYTHONPATH=src:tools python tools/plot_risky_portfolio_overview.py \
  experiments/group1_risky_portfolios_20260910/results
```

The runner refuses an existing `greedy_pca/` output directory. It saves compact
per-scenario return bounds, independently verifies signed portfolio repricing,
and reports per-stage timings separately from concurrent wall time. Local
`--verify-only` regenerates the greedy CSVs from the fetched bounds and compares
dates, calibration windows and realized P&L with the prior backtest. Plotting
requires exact date alignment; updated charts retain the solver series and add
the greedy baseline. Per-portfolio `margins_with_greedy.csv` provides all plotted
margin series without changing the original QUBO result files.

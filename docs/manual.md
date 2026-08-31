# Beginner's manual for `backtest_orchestrator.py`

This guide explains how to run the rolling margin backtester, what files it
needs, what it produces, and what every command-line option means. You do not
need to understand QUBOs, PCA, or simulated bifurcation to follow the basic
examples.

## 1. What the program does

For each backtest date, the program uses only information from earlier dates to:

1. learn a market model from a rolling window of returns;
2. generate stressed market scenarios;
3. evaluate every selected scenario with the chosen margin method;
4. use the worst result to calculate a margin amount;
5. compare that margin with the return that actually occurred; and
6. save one result row per requested method for that date.

This is a rolling, leakage-safe backtest: the realized return for a date is not
used to construct that date's margin estimate.

There are two margin methods:

- `qubo` builds and solves a plausibility-steered QUBO, repairs every solver
  candidate to categorical convergence, then enforces a hard empirically
  calibrated residual-score cutoff during final decoding;
- `baseline` skips correlations and QUBO solving. Within each scenario, it
  independently chooses the state with the lowest weighted P&L for every
  asset, sums those asset P&Ls, and then retains the worst scenario.

Use `--method both` to compare them from exactly the same PCA fit and generated
scenarios. The baseline intentionally performs no cross-asset compatibility or
feasibility checks, so it is a deliberately conservative independent-asset
reference rather than another QUBO solver.

More precisely, for factor scenario `s`, asset `i`, residual state `j`, position
weight `w_i`, and simple return `r_sij`, the baseline calculates
`B_s = sum_i min_j(w_i * r_sij)`. It then uses `min_s(B_s)` as the worst
portfolio P&L. This correctly handles shorts too: multiplying by a negative
weight makes a large positive asset return a loss for that position.

## 2. Before you run it

Open PowerShell in the project directory, the folder that contains
`requirements.txt`, `src`, and `synthetic_market`.

Install the Python packages:

```powershell
python -m pip install -r requirements.txt
```

Check whether PyTorch can use an NVIDIA GPU:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

CUDA is optional. The program runs on the CPU if CUDA is unavailable, although
large backtests can be much slower.

You can always display the built-in option list with:

```powershell
python src/backtest_orchestrator.py --help
```

## 3. Required input files

The first argument tells the program which market-universe folder to use. For
example, `assets_00100` means:

```text
synthetic_market/assets_00100/
```

That folder must contain:

- `historical_close.csv`: close prices used for the initial history;
- `backtest_close.csv`: an anchor close followed by dates to evaluate; and
- `portfolio.csv`: the portfolio positions or weights.

The asset columns in the historical and backtest price files must match. By
default, the second row of `backtest_close.csv` is the first evaluation date;
the first row is the anchor needed to calculate its return.

The `subfolder` argument accepts any of these forms:

```powershell
# A folder name below synthetic_market
python src/backtest_orchestrator.py assets_00100

# A path beginning with synthetic_market
python src/backtest_orchestrator.py synthetic_market/assets_00100

# An absolute path
python src/backtest_orchestrator.py C:\data\my_market
```

## 4. Start with a small test

Before starting a long run, evaluate one date, one scenario, and only a small
number of solver steps and randomized runs:

```powershell
python src/backtest_orchestrator.py assets_00100 `
  --day-limit 1 `
  --scenario-index 0 `
  --steps 100 `
  --runs 2 `
  --dt 0.1 `
  --output backtest-results/smoke_test.csv
```

The backtick at the end of a PowerShell line means "continue this command on
the next line." You may also write the entire command on one line.

If that succeeds, a normal run using all default settings is:

```powershell
python src/backtest_orchestrator.py assets_00100
```

The program automatically uses CUDA when it is available and otherwise uses
the CPU.

> **Important:** without `--resume`, an existing output CSV and its summary are
> deleted when the run starts. Use a new `--output` name or add `--resume` when
> you want to preserve completed results.

### Reading console progress

Each date is shown as one block. The current implementation reads one client
portfolio from `portfolio.csv`, so the header reports `Portfolios: 1`.

```text
========================================================================================
DAY 1/258 | 2025-01-01 | cuda:0
----------------------------------------------------------------------------------------
Assets: 10,000 | Portfolios: 1 | Scenarios: 105 | Batches: 105 | Steps: 1000 | Runs: 16
Method: qubo | Solver: cuda:0 | PCA: cuda:0 | Correlation: cuda:0
----------------------------------------------------------------------------------------
[cuda:0 2025-01-01] PCA completed in 2.31s | Factor scenarios enumerated: 105/105
----------------------------------------------------------------------------------------
[cuda:0 2025-01-01] Scenario 17/105 | data=0.08s | QUBO=1.12s | solve=2.44s |
decode=0.03s |
progress=17/105 (16.2%) | ETA=319.5s
[cuda:0 2025-01-01] RESULT QUBO | margin=... | realized P&L=... | realized loss=... | SME=... |
selected scenario=...
----------------------------------------------------------------------------------------
[cuda:0 2025-01-01] DAY COMPLETE | PCA=2.31s | scenario data=8.42s | QUBO build=118.69s |
solve=258.13s | baseline=0.0000s | total=481.85s
========================================================================================
```

Progress is reported after completed scenario batches and is throttled by
`--progress-interval`. Consequently, batch size 1 gives the finest progress;
a large packed batch cannot report its internal scenarios while its single
solver call is still running.

## 5. Output files

Unless `--output` is supplied, results go to:

```text
backtest-results/<universe-name>_margin_backtest.csv
```

For `assets_00100`, that is:

```text
backtest-results/assets_00100_margin_backtest.csv
```

The CSV is checkpointed after every completed method. A `both` run writes two
rows per date. Its most useful columns are:

- `date`: evaluated market date;
- `method`: `qubo` or `baseline`;
- `margin`: calculated non-negative margin;
- `worst_scenario_pnl`: worst selected scenario P&L;
- `realized_pnl`: P&L that actually occurred;
- `gross_exposure`: sum of absolute portfolio positions;
- `signed_margin_error`: `(margin + realized_pnl) / gross_exposure`;
- `selected_scenario`: scenario that produced the worst P&L;
- `raw_one_hot_violations`: QUBO solver constraint violations before repair;
  this is blank for the baseline;
- `raw_energy` and `repaired_energy`: energy of the winning raw solver output
  and of that candidate after converged categorical repair;
- `raw_feasible_candidates` and `solver_candidates`: raw one-hot feasibility
  diagnostics across randomized trajectories;
- `plausibility_score`, `plausibility_threshold`, and
  `hard_plausibility_feasible`: the exact final feasibility check;
- `pca_seconds`, `scenario_build_seconds`, `qubo_build_seconds`,
  `solve_seconds`, `baseline_seconds`, and `day_seconds`: timing information;
  and
- `device`, `steps`, `runs`, and `seed`: important run settings.

A negative `signed_margin_error` means the margin did not fully cover that
date's realized loss. A positive value means there was a cushion.

A second file is written beside the CSV:

```text
backtest-results/assets_00100_margin_backtest.summary.json
```

Its `methods` object contains separate error quantiles, under-margin counts and
rates, mean and worst shortfalls, and runtime for each requested method. A
single-method summary also retains the original top-level statistics for
backward compatibility.

A third file records one row of resource diagnostics for every completed day:

```text
backtest-results/assets_00100_margin_backtest.diagnostics.csv
```

It records devices, numeric precision, scenario and run batch counts, build
workers, phase timings, and sampled mean and peak CPU, process RAM, GPU
utilization, and total device VRAM use. CPU usage follows the process
convention: 100% means one fully occupied CPU core, so multithreaded work can
exceed 100%. GPU utilization and VRAM are whole-device measurements and can
include other programs using the GPU. GPU fields are blank for CPU-only runs.
Sampling occurs once per second, so very short runs may have only a few samples.
The `method` field is `qubo`, `baseline`, or `both`. In `both` mode the single
diagnostic row measures the shared run, while its QUBO and baseline timing
columns remain separate.

## 6. Common recipes

### Resume an interrupted run

```powershell
python src/backtest_orchestrator.py assets_00100 --resume
```

The program reads the existing CSV and skips method/date pairs already present.
Use the same universe, output path, and model settings that were used
originally. This also lets you add baseline rows to an existing QUBO result by
resuming the same output with `--method both`.

### Evaluate a date range

```powershell
python src/backtest_orchestrator.py assets_00100 `
  --backtest-start 2025-01-02 `
  --backtest-end 2025-03-31
```

Both endpoints are included when those dates exist in the market data.

### Force CPU or CUDA

```powershell
python src/backtest_orchestrator.py assets_00100 --device cpu
python src/backtest_orchestrator.py assets_00100 --device cuda
python src/backtest_orchestrator.py assets_00100 --device cuda:0
```

Explicitly requesting CUDA produces an error if CUDA is unavailable. `auto` is
the safest default.

### Use several GPUs

```powershell
python src/backtest_orchestrator.py assets_10000 `
  --devices cuda:0,cuda:1,cuda:2,cuda:3 `
  --scenario-batch-size 1 `
  --resume
```

Independent dates are divided among separate GPU processes. The GPUs must be
distinct, but they do not need NVLink. Partial files are merged at the end.

### Try several specific scenarios

Repeat `--scenario-index` once per scenario:

```powershell
python src/backtest_orchestrator.py assets_00100 `
  --scenario-index 0 `
  --scenario-index 10 `
  --scenario-index 52
```

With the default `21,5` grid, valid indices are 0 through 104. If this option is
omitted, all 105 scenarios are evaluated.

### Compare QUBO with the greedy baseline

```powershell
python src/backtest_orchestrator.py assets_00100 `
  --method both `
  --day-limit 1
```

Use `--method baseline` when only the inexpensive reference is needed. It still
runs rolling PCA and creates all requested scenarios, but it does not calculate
conditional correlations, build QUBOs, or invoke simulated bifurcation.

## 7. Complete argument reference

### Required argument

| Argument | Meaning |
|---|---|
| `subfolder` | Market-universe folder. A simple name is looked up under `synthetic_market`; a project-relative or absolute path may also be used. |

### Files and date selection

| Option | Default | Meaning |
|---|---:|---|
| `-h`, `--help` | — | Print the built-in help and exit. |
| `--output PATH` | `backtest-results/<name>_margin_backtest.csv` | Choose the result CSV. Parent folders are created automatically. |
| `--backtest-start DATE` | second backtest row | First evaluation date, normally written as `YYYY-MM-DD`. |
| `--backtest-end DATE` | last backtest row | Last evaluation date, inclusive. |
| `--day-limit N` | all dates | Evaluate only the first `N` dates remaining after the date filters. Must be positive. Useful for testing. |
| `--resume` | off | Keep existing output and skip dates already in its CSV. Without this flag, existing output and summary files are replaced. |
| `--progress-interval SECONDS` | `10` | Print scenario progress, elapsed time, and an estimated time remaining at approximately this interval. Set it to `0` to disable progress lines. In multi-GPU runs every line includes its device and date. |

### Margin method

| Option | Default | Meaning |
|---|---:|---|
| `--method METHOD` | `qubo` | Choose `qubo`, `baseline`, or `both`. `both` reuses the same PCA and scenario data, then writes a clearly labelled row for each method. |

### Rolling model and scenario construction

Beginners should normally leave this group at its defaults.

| Option | Default | Meaning |
|---|---:|---|
| `--window N` | `125` | Number of completed returns used to fit each date's model. Must be at least 2, and enough prior data must exist. |
| `--ew-lambda X` | `0.93` | Exponential-weight decay. Must be greater than 0 and at most 1. Smaller values emphasize recent observations more strongly; 1 gives equal weight. |
| `--grid-points A,B` | `21,5` | Odd, positive grid sizes for the two PCA factors. The number of scenarios is `A × B`. Do not put a space after the comma. |
| `--scenario-index N` | all scenarios | Evaluate one scenario index. Repeat the option to select several. Valid values are 0 through `A × B - 1`. |
| `--z-bins N` | `21` | Number of discrete residual-shock states per asset. Must be positive. Increasing it makes the optimization problem larger. |
| `--nearest N` | `100` | Number of nearby historical factor observations used for conditional shocks. Must be at least 2. |
| `--residual-sigma-range X` | `5.0` | Width of the residual-shock grid in standard deviations. Must be positive. |
| `--distance-inflation-alpha X` | `0.50` | Strength of uncertainty inflation as a scenario moves away from historical observations. Must be non-negative. |
| `--distance-inflation-power X` | `2.0` | Exponent controlling how quickly distance inflation grows. Must be positive. |
| `--max-inflation-factor X` | `5.0` | Upper bound on distance-based inflation. Must be at least 1. |
| `--lambda-one-hot X` | `1.0` | Minimum one-hot penalty. The builder automatically raises it above a conservative safe bound derived from all non-constraint QUBO coefficients. |
| `--lambda-compat X` | `0.1` | Dimensionless strength of SB steering toward low PSD residual scores. It does not define feasibility; the hard cutoff below does. |
| `--top-k-neighbors N` | `5` | Number of strongest residual-correlation neighbors nominated per asset. The signed undirected union is degree-normalized into a positive-semidefinite score. Zero gives an independent standardized-residual score. |
| `--plausibility-confidence X` | `0.998` | Empirical quantile used as the hard PSD residual-score cutoff. Must be greater than 0 and at most 1. |

### Devices, numeric precision, and memory

| Option | Default | Meaning |
|---|---:|---|
| `--device DEVICE` | `auto` | Device for the solver and PCA: `auto`, `cpu`, `cuda`, or a GPU such as `cuda:0`. `gpu` is accepted as an alias for `cuda`. |
| `--devices LIST` | not set | Comma-separated, distinct devices used to shard dates, for example `cuda:0,cuda:1`. This takes precedence over `--device` for execution. |
| `--pca-dtype TYPE` | `float32` | PCA precision. `float32` uses less memory and is faster on typical GPUs; use `float64` for a higher-precision reference comparison. |
| `--correlation-device DEVICE` | `auto` | Device used to calculate conditional correlations. `auto` follows the solver device. |
| `--correlation-block-mib N` | `256` | Approximate MiB limit for each temporary dense correlation row block. Must be positive. Reduce it if correlation construction runs out of memory. |
| `--scenario-batch-size N` | `1` | Number of scenarios materialized as one work chunk and, on the QUBO path, packed into one solver call. Must be positive. Larger values can improve QUBO throughput but use more RAM or VRAM. On the baseline-only path this changes chunk memory, not the calculation. |
| `--build-workers N` | `1` | Threads used for scenario construction and QUBO building. Must be positive. A QUBO run with GPU correlation requires this to remain `1`. More workers can use substantially more memory. |
| `--sbm-dtype TYPE` | `float32` | Solver precision: `float32` or `float64`. `float32` is faster and uses less memory. |
| `--run-batch-size N` | all runs together | Divide randomized solver runs into batches to reduce state memory. Must be positive. Try a smaller value if solving runs out of memory. |

### Simulated-bifurcation solver settings

These options change optimization behavior. For comparable experiments, change
one setting at a time and keep the seed fixed.

| Option | Default | Meaning |
|---|---:|---|
| `--steps N` | `10000` | Solver time steps per randomized run. Must be positive. More steps normally cost more time. |
| `--runs N` | `16` | Number of randomized trajectories. Every candidate is repaired to convergence and the lowest repaired energy is retained. Must be positive. More runs cost more time and memory unless batched. |
| `--dt X` | `1.0` | Integration time step. Must be finite and positive. |
| `--a0 X` | `1.0` | Positive simulated-bifurcation pump/schedule scale. Advanced setting. |
| `--c0 X` | `0.0` | Coupling scale. `0` requests the solver's automatic scale; any explicit value must be non-negative. |
| `--gamma X` | `0.0` | Non-negative damping coefficient. Advanced setting. |
| `--initial-scale X` | `0.05` | Positive scale of the randomized initial state. |
| `--seed N` | `1` | Base random seed. Reuse it for reproducible comparisons with the same devices, precision, and settings. |
| `--decode-sweeps N` | `100` | Safety cap for categorical repair. Repair must converge; reaching a positive cap raises instead of returning a partial result. Zero removes the cap. |

## 8. Performance guidance

### Interactive resource planner

Open [RESOURCE_SCALING_DASHBOARD.html](../dashboards/RESOURCE_SCALING_DASHBOARD.html) in a
browser to explore how assets, z-bins, top-k neighbors, reciprocal nominations,
scenario/run batching, dtype, GPU count, VRAM, and system RAM change the
estimated problem capacity. Its memory model follows `RESOURCE_ESTIMATION.tex`.
Solve time is deliberately calibration-driven: replace the bundled small
diagnostic reference with a representative one-scenario measurement from the
target GPU before relying on a large-universe projection.

For a large universe, begin conservatively:

```powershell
python src/backtest_orchestrator.py assets_10000 `
  --device cuda `
  --correlation-device cuda `
  --scenario-batch-size 1 `
  --build-workers 1 `
  --resume
```

If there is plenty of free VRAM, gradually increase
`--scenario-batch-size`. For smaller universes, values such as 8, 16, or 32 may
improve throughput. If solver state memory is the problem, set
`--run-batch-size`, for example `--run-batch-size 4`. If correlation
construction is the problem, reduce `--correlation-block-mib`.

Do not judge production-quality results from the tiny smoke-test values. Low
`--steps`, low `--runs`, or a single scenario are intended only to confirm that
the program and data pipeline work.

## 9. Common errors

### "market universe folder does not exist"

Check the spelling of the first argument and confirm that the folder is inside
`synthetic_market`, or pass its correct absolute path.

### "has ... prior returns; ... required"

The chosen first date does not have enough history for `--window`. Move
`--backtest-start` later, supply more historical data, or deliberately use a
smaller window.

### CUDA was requested but is unavailable

Use `--device cpu`, or install a CUDA-enabled PyTorch build appropriate for the
machine.

### Out-of-memory error

Try the following, one at a time:

1. set `--scenario-batch-size 1`;
2. set `--run-batch-size 2` or `4`;
3. reduce `--correlation-block-mib`;
4. keep `--build-workers 1`;
5. use `float32` precision; or
6. test a smaller universe.

### GPU correlation requires `build_workers=1`

Remove `--build-workers` or set it to `1` when
`--correlation-device` resolves to CUDA.

### No evaluation dates remain

Check that the requested start and end dates overlap `backtest_close.csv`, that
the start is not after the end, and that `--day-limit` is positive.

### The results changed or disappeared

A run without `--resume` replaces an existing output file. Use `--resume` to
continue a checkpoint, or `--output` to give experiments different filenames.

## 10. Verify the project after changes

The repository's test suite can be run with:

```powershell
python -m unittest discover -s tests -v
```

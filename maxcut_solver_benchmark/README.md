# Torch MaxCut solver benchmark

This benchmark compares `torch_sbm`, `adaptive_torch_sbm`, and `torch_svl` on
the same reproducibly generated unweighted Erdős–Rényi MaxCut instances. It records cut
quality and wall-clock solve time, stops before the next size would exceed the
configured RAM/device-memory allowance, and can resume an interrupted sweep.

The shared comparison knobs are graph sizes, graph density, instances, trials,
four trajectories (`runs`), integration steps, dtype, devices, and batch size.
Solver-specific dynamics retain their implementation defaults; they are not
treated as comparable tuning parameters. All solvers see the same graph and
seed schedule at each `(size, instance, trial)` point.

## Run

Run from the repository root with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python maxcut_solver_benchmark/run_benchmark.py \
  --config maxcut_solver_benchmark/config.example.json
```

Resume safely after an interruption:

```bash
PYTHONPATH=src python maxcut_solver_benchmark/run_benchmark.py \
  --config maxcut_solver_benchmark/config.example.json --resume
```

For several explicitly indexed GPUs, copy the example config and set, for
example, `"devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]`. `runs` is
deliberately fixed at 4. Keep `steps` at 1000 or above; the example uses 2000.
The size list is attempted in ascending order. `memory_fraction` reserves some
headroom, while `memory_limit_gb` places an absolute cap on the host-RAM budget;
the smaller allowance is used. The runner ends the sweep cleanly when its
conservative estimate for the next size no longer fits. Sparse graphs are
strongly recommended for large sizes because the QUBO edge storage grows with
the number of graph edges.

`output_dir` is resolved relative to the config file. The runner appends
completed records to `raw_results.csv` and `raw_results.jsonl`, flushing each
batch before continuing. With `--resume`, already completed
solver/instance/trial keys are skipped. Reference cuts are exact through
`exact_reference_max_vertices`; larger graphs use the same seeded multi-restart
reference procedure for every solver. Independent restarts use
`reference_workers` CPU processes when `fork` is available; reference
generation remains outside all solver timings.

## Plot

Plot either a completed or partial sweep:

```bash
python maxcut_solver_benchmark/plot_results.py \
  maxcut_solver_benchmark/results/default/raw_results.csv
```

The command writes `plots/summary.csv`, `plots/solution_quality.png`, and
`plots/time_to_solution.png` beside the input. A different destination and TTS
confidence are available through `--output-dir` and `--confidence`.

Solution quality is `cut / reference_cut`; the chart shows the median and
interquartile range for every solver/size group present in the CSV. For each
individual graph and solver, empirical time to solution (TTS) uses the observed
success probability `p` across trials and mean per-instance amortized wall time
`t`, then applies

```text
TTS(q) = t * ceil(log(1 - q) / log(1 - p)).
```

At `p = 1`, one call is used. The size-level chart reports the median of these
per-graph estimates. Graphs with zero successes have unbounded empirical TTS;
groups whose median is unbounded remain visible in `summary.csv` and are omitted
from the logarithmic plot with a warning. This makes partial/resumed results
useful without pretending a missing or unsuccessful point is a timing result.

Timing includes the solver call but excludes graph construction and reference
generation. Each solver performs an untimed warm-up on every configured device
before measurements begin. Compare runs made on the same GPU model, device
count, dtype, steps, and batch settings; GPU timings from different hardware
are not apples-to-apples.

## Torch SVL parameter sweep

The dedicated Torch SVL sweep varies every dynamics parameter one at a time
around the benchmark baseline. This avoids an intractable Cartesian product and
keeps each quality/runtime sensitivity directly attributable to one parameter.
It reuses the saved instances and reference cuts from the CPU benchmark, runs
sequentially, checkpoints every trial, and enforces the configured 5 GiB
estimated-memory allowance.

```bash
PYTHONPATH=src .venv/bin/python maxcut_solver_benchmark/run_torch_svl_sweep.py \
  --config maxcut_solver_benchmark/config.torch_svl_sweep.json --resume

.venv/bin/python maxcut_solver_benchmark/plot_torch_svl_sweep.py \
  maxcut_solver_benchmark/results/torch_svl_parameter_sweep/raw_results.csv
```

The plotter writes `parameter_plots/summary.csv` and one two-panel PNG per
parameter, showing trial-median solution quality and solver time for every graph
size.

The follow-up Cartesian runs-by-steps sweep fixes every other parameter at its
highest mean-quality value from the sensitivity sweep and tests runs
`1, 2, 4, 8, 16, 32` against steps `1000, 2000, 3000, 4000, 5000`:

```bash
PYTHONPATH=src .venv/bin/python maxcut_solver_benchmark/run_torch_svl_runs_steps.py \
  --config maxcut_solver_benchmark/config.torch_svl_runs_steps.json --resume

.venv/bin/python maxcut_solver_benchmark/plot_torch_svl_runs_steps.py \
  maxcut_solver_benchmark/results/torch_svl_runs_steps/raw_results.csv
```

### Torch SVL BiqMac steps/dt sweep

The BiqMac sweep fixes Torch SVL at 8 runs and evaluates 1,000--10,000 steps in
increments of 1,000 against `dt` values from 0.005 through 0.030 in increments
of 0.0025. It runs sequentially on six graphs selected to span vertex count,
density, and signed/positive edge weights. Solver timing starts after each graph
has been loaded. Each grid point performs eight independent trajectories in one
solver call. The CSV is flushed after every solve and can be resumed.

```bash
PYTHONPATH=src .venv/bin/python maxcut_solver_benchmark/run_torch_svl_steps_dt_biqmac.py \
  --config maxcut_solver_benchmark/config.torch_svl_steps_dt_biqmac.json --resume

.venv/bin/python maxcut_solver_benchmark/plot_torch_svl_steps_dt_biqmac.py \
  maxcut_solver_benchmark/results/torch_svl_steps_dt_biqmac/raw_results.csv
```

The plotter writes aggregate line plots, separate 3-D quality and runtime
surfaces, and `aggregate_summary.csv`. Quality is normalized by each instance's
published reference cut before results are aggregated across unlike graphs.

### Torch SVL polynomial quality model

The polynomial experiment uses 36 BiqMac instances divided equally among
sparse, medium, dense, and complete strata. Each stratum contributes six
training and three evaluation graphs, so no instance occurs in both sets. An
18-point Latin hypercube varies every Torch SVL dynamics parameter.

```bash
PYTHONPATH=src .venv/bin/python maxcut_solver_benchmark/run_torch_svl_polynomial_data.py \
  --config maxcut_solver_benchmark/config.torch_svl_polynomial.json --resume

.venv/bin/python maxcut_solver_benchmark/fit_torch_svl_polynomial.py \
  maxcut_solver_benchmark/results/torch_svl_polynomial/raw_results.csv
```

The fitter compares degree-one, quadratic, and cubic ridge polynomials using
group cross-validation over whole training instances. It writes the reusable
`quality_polynomial.joblib`, coefficient table, held-out metrics, scheduler
choices and scheduler regret for the untouched evaluation instances.

To fit a graph-adaptive variant with explicit interactions between density and
every solver parameter:

```bash
PYTHONPATH=maxcut_solver_benchmark .venv/bin/python \
  maxcut_solver_benchmark/fit_torch_svl_sparsity_scheduler.py \
  maxcut_solver_benchmark/results/torch_svl_polynomial/raw_results.csv
```

This writes a separate reusable sparsity scheduler, coefficient table, metrics,
and held-out choices without overwriting the unrestricted polynomial model.

The plotter produces one quality/runtime line-chart pair per graph size, with
steps on the x-axis and one line for each trajectory count.

For Torch SBM, the dedicated sensitivity sweep fixes `steps=1000` and
`runs=4`, then varies `dt`, `a0`, `c0`, `gamma`, and `initial_scale` one at a
time:

```bash
PYTHONPATH=src .venv/bin/python maxcut_solver_benchmark/run_torch_sbm_sweep.py \
  --config maxcut_solver_benchmark/config.torch_sbm_sweep.json --resume

.venv/bin/python maxcut_solver_benchmark/plot_torch_svl_sweep.py \
  maxcut_solver_benchmark/results/torch_sbm_parameter_sweep/raw_results.csv
```

## Biq Mac solver comparison

`run_biqmac_benchmark.py` compares tuned Torch SVL (`runs=8`, `steps=5000`),
OR-Tools CP-SAT, SCIP, HiGHS, and CPLEX on a representative subset of the
official Biq Mac MaxCut library. Published optima from `biqmaclib.pdf` are used
as references; the plotter falls back to the best observed cut when a selected
instance has no published optimum. The selected instances contain at most 100
vertices, solvers run sequentially with a 10-second mathematical-programming
limit in the original `results/biqmac` run and a 30-second limit in the current
`results/biqmac_30s` run. Applicable memory controls are set below 6 GiB. OSQP is installed
but excluded because it cannot directly solve binary non-convex MaxCut.

```bash
PYTHONPATH=src .venv/bin/python maxcut_solver_benchmark/run_biqmac_benchmark.py \
  --config maxcut_solver_benchmark/config.biqmac.json --resume

.venv/bin/python maxcut_solver_benchmark/plot_biqmac_benchmark.py \
  maxcut_solver_benchmark/results/biqmac/raw_results.csv
```

`config.biqmac_large.json` selects the library's largest MaxCut families: the
400-vertex 2D torus graphs, 343-vertex 3D torus graphs, and dense 300-vertex
Ising graphs. It excludes unavailable CPLEX, gives CP-SAT, SCIP, and HiGHS 20
seconds of optimizer time per graph, and retains the 6 GiB memory ceiling.
Torch SVL uses a per-instance throughput pilot to increase both runs and steps
toward the same time budget, retaining the best pilot or calibrated candidate.

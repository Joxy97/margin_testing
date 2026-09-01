# Rolling margin backtester

`src/backtest_orchestrator.py` performs a leakage-safe rolling backtest over one
pre-generated market universe. It can run the QUBO method, an independent-asset
greedy baseline, or both from the same scenarios. It does not write intermediate
BQM or QUBO files.

For every evaluation date it:

1. Takes the 125 completed returns ending on the previous market date.
2. Fits window-local exponentially weighted normalization and two-component PCA.
3. Builds the 21 by 5 scenario grid.
4. Creates each scenario's per-asset state grids and portfolio P&L vector once.
5. On the QUBO path, builds a sparse positive-semidefinite residual score from
   the degree-normalized, undirected top-k correlation graph and calibrates an
   empirical plausibility threshold. Portfolio P&L is normalized by gross
   exposure and the one-hot penalty is raised to a coefficient-derived safe
   bound.
6. Packs compact QUBOs into a block-diagonal CSR matrix and solves them with
   simulated bifurcation. Every randomized candidate is repaired and improved
   to categorical convergence before energy comparison. The lowest-energy
   repaired QUBO candidate is used directly; neither a greedy candidate nor a
   scenario-center candidate is injected during decoding. The plausibility
   threshold is reported as a diagnostic and does not replace the QUBO result.
7. On the baseline path, independently selects the minimum weighted-PnL state
   for each asset in each scenario. It applies no cross-asset compatibility or
   feasibility checks and never consumes a QUBO solution.
8. For each requested method, selects the minimum portfolio PnL across
   scenarios and issues `margin = max(0, -worst_pnl)`.
9. Projects each method's selected asset-state vector into the same EW-PCA
   factor space and measures its distance from the selected scenario center in
   factor-coordinate units and local scenario-grid-step units.
10. Reveals that day's realized loss and writes one labelled method row with
   `signed_margin_error = (margin - realized_loss) / gross_exposure`.

The financial result columns are ordered as gross exposure, realized loss,
margin, realized loss as a percentage of gross exposure, margin as a percentage
of gross exposure, and signed margin error. The two percentage columns store
percentage points; for example, `6.4` means `6.4%`. Signed margin error remains
a fraction, so `0.064` means `6.4%`.

## Install

Install the ordinary dependencies first:

```powershell
python -m pip install -r requirements.txt
```

The default PyPI PyTorch package may be CPU-only on some systems. For CUDA,
install the wheel specified by the official PyTorch selector for the CUDA
version installed on the machine. Confirm the environment before a long run:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run

Full default run on the best available backend:

```powershell
python src/backtest_orchestrator.py assets_00100
```

Compare both methods using shared PCA and scenarios:

```powershell
python src/backtest_orchestrator.py assets_00100 --method both
```

Run only the greedy baseline, without correlation, QUBO build, or SBM solve:

```powershell
python src/backtest_orchestrator.py assets_00100 --method baseline
```

Explicit CUDA with scenario batching and resumable output:

```powershell
python src/backtest_orchestrator.py assets_10000 `
  --device cuda `
  --correlation-device cuda `
  --scenario-batch-size 4 `
  --steps 10000 `
  --runs 16 `
  --resume
```

Multiple GPUs shard independent dates without requiring NVLink:

```powershell
python src/backtest_orchestrator.py assets_10000 `
  --devices cuda:0,cuda:1,cuda:2,cuda:3 `
  --scenario-batch-size 1 `
  --resume
```

Small validation run:

```powershell
python src/backtest_orchestrator.py assets_00100 `
  --day-limit 1 `
  --scenario-index 0 `
  --steps 100 `
  --runs 2 `
  --dt 0.1
```

By convention, `backtest_close.csv` contains one anchor close before the first
evaluation date. When `--backtest-start` is omitted, the second row is the first
date evaluated. Pass `--backtest-start YYYY-MM-DD` to use an explicit marker.

## Performance controls

- `--scenario-batch-size`: Number of independent scenario graphs packed into a
  single sparse SBM call. Start at 1 for 10,000 assets; increase while VRAM
  permits. For smaller universes, 8-32 is generally a better starting point.
  In baseline-only mode it controls scenario chunking but does not change the
  greedy calculation.
- `--run-batch-size`: Splits randomized trajectories when state memory is too
  large. Omit it to run all trajectories together.
- `--correlation-block-mib`: Bounds the temporary dense correlation row block.
- `--pca-dtype`: `float32` is the default for GPU throughput and memory use;
  use `float64` for higher-precision reference comparisons.
- `--build-workers`: Scenario/QUBO construction threads. Keep this at 1 when
  GPU correlation is enabled on the QUBO path or for very large models to bound
  memory.
- `--backtest-start`, `--backtest-end`, and `--day-limit`: Divide validation or
  multi-GPU runs into explicit, non-overlapping date ranges.

GPU PCA and exact blockwise conditional correlations use PyTorch. QUBO
coefficient expansion remains on CPU, while the dominant sparse SBM loop is
executed on the selected Torch device. The output is checkpointed after every
method/date pair; `--resume` skips pairs already present in the CSV. A sibling
`.summary.json` groups signed-error quantiles, under-margin rate, shortfall
statistics, and factor-projection errors by method.

The projection uses the asset state selected by each method. If `z` is that
vector in normalized asset space, the factor coordinate is
`(z - weighted_mean) @ loadings.T`. The factor-unit error is the Euclidean
distance from the enumerated scenario center. The grid-unit error first divides
each component difference by that component's local scenario-grid spacing,
then takes the Euclidean norm. A grid error of `1` therefore means one grid step
of total displacement; diagonal displacement can exceed `1` even when each
component is less than one step away.

A sibling `.diagnostics.csv` contains one resource row per completed date. It
records settings and batch counts, phase timings, process CPU/RAM, and—when
CUDA is used—sampled whole-device GPU utilization and VRAM use. CPU percentages
may exceed 100% because 100% represents one fully occupied core. Resource
sampling occurs once per second.

Multi-GPU workers use separate spawned processes and shard dates, so each GPU
owns an isolated solver context and no peer-to-peer interconnect is required.

## Tests

```powershell
python -m unittest discover -s tests -v
```

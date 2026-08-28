# Rolling margin backtester

`src/backtest_orchestrator.py` performs a leakage-safe rolling backtest over one
pre-generated market universe. It does not write intermediate BQM or QUBO
files.

For every evaluation date it:

1. Takes the 125 completed returns ending on the previous market date.
2. Fits window-local exponentially weighted normalization and two-component PCA.
3. Builds the 21 by 5 scenario grid.
4. Creates each scenario's portfolio-independent structure and portfolio linear
   overlay in compact contiguous arrays. The hot path never creates a duplicate
   `dimod` graph or writes an intermediate QUBO.
5. Packs a configurable number of scenario graphs into one block-diagonal CSR
   matrix and solves all randomized trajectories with simulated bifurcation.
6. Repairs every selected sample to exactly one state per asset.
7. Selects the minimum decoded portfolio PnL across scenarios and issues
   `margin = max(0, -worst_pnl)`.
8. Reveals that day's realized PnL and writes
   `signed_margin_error = (margin + realized_pnl) / gross_exposure`.

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
- `--run-batch-size`: Splits randomized trajectories when state memory is too
  large. Omit it to run all trajectories together.
- `--correlation-block-mib`: Bounds the temporary dense correlation row block.
- `--pca-dtype`: Use `float64` for the reference path; validate `float32` before
  using it for production.
- `--build-workers`: CPU QUBO construction threads. Keep this at 1 when GPU
  correlation is enabled or for very large models to bound memory.
- `--backtest-start`, `--backtest-end`, and `--day-limit`: Divide validation or
  multi-GPU runs into explicit, non-overlapping date ranges.

GPU PCA and exact blockwise conditional correlations use PyTorch. QUBO
coefficient expansion remains on CPU, while the dominant sparse SBM loop is
executed on the selected Torch device. The output is checkpointed after every
date; `--resume` skips dates already present in the CSV. A sibling
`.summary.json` contains the signed-error quantiles, under-margin rate, and
shortfall statistics over every completed date.

Multi-GPU workers use separate spawned processes and shard dates, so each GPU
owns an isolated solver context and no peer-to-peer interconnect is required.

## Tests

```powershell
python -m unittest discover -s tests -v
```

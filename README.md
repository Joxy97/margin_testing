# Margin testing

This repository contains the minimal pipeline required to generate synthetic
market universes, construct scenario-conditioned portfolio QUBOs, and run a
rolling margin backtest with simulated bifurcation.

## Project structure

- `src/synthetic_market_generator.py` generates the retained market datasets.
- `src/market_to_qubo.py` creates scenario shocks and QUBO coefficients.
- `src/backtest_orchestrator.py` runs leakage-safe rolling backtests.
- `src/sbm_torch.py` batches simulated-bifurcation trajectories on CPU or CUDA.
- `src/qubo_model.py` stores large QUBOs in compact numeric arrays.
- `include/sbm/` and `src/sbm/` provide the native C++ CPU solver.
- `tools/export_dwave_qubo.py` converts Ocean BQM files to the native format.
- `synthetic_market/` contains the pre-generated market universes.
- `tests/` contains focused backtesting and solver tests.
- `docs/` contains the beginner manual, technical notes, and resource model.
- `dashboards/` contains the interactive backtest and resource-planning dashboards.

## Python setup and backtesting

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/backtest_orchestrator.py assets_00100
```

See [BACKTESTING.md](docs/BACKTESTING.md) for rolling backtest and GPU controls.
The longer [beginner manual](docs/manual.md) covers every command-line option.
Analyze completed runs with the
[backtest results dashboard](dashboards/BACKTEST_RESULTS_DASHBOARD.html), or plan
hardware capacity with the
[resource scaling dashboard](dashboards/RESOURCE_SCALING_DASHBOARD.html).

## Native simulated bifurcation

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
python tools/export_dwave_qubo.py model.bqm model.qubo
build\sbm_solve.exe model.qubo
```

See [SIMULATED_BIFURCATION.md](docs/SIMULATED_BIFURCATION.md) for the QUBO format and
solver parameters.

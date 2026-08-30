# AGENTS.md

## Purpose

This repository is a research-oriented portfolio margin and backtesting application. It acquires historical close prices, builds PCA-conditioned return/volatility stress states, and calculates the greatest portfolio loss either greedily or by encoding each state as a binary quadratic model (BQM/QUBO). The QUBO path supports classical samplers and simulated-bifurcation implementations on Python/Torch, native C++, optional CUDA, and an FPGA simulation/HLS kernel.

Use this file as the default guide for work anywhere in the repository. Keep it current when architecture, commands, configuration, or extension points change.

## Start Here

The project is not installed as a Python package. Run Python code from the repository root with `src` on `PYTHONPATH`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

PYTHONPATH=src python -m margin_engine config/margin.example.yaml
PYTHONPATH=src python -m margin_engine config/options.example.yaml
python run_backtest.py config/backtests/assets_00010.yaml
python plot_backtest.py backtest_results/<portfolio>/breaches.csv
```

All paths in an application YAML file are resolved relative to that YAML file. Prefer copying `config/margin.example.yaml` and changing the copy. The parser is intentionally strict and rejects unknown keys.

For native simulated bifurcation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful CMake options are `SBM_ENABLE_CUDA`, `SBM_BUILD_FPGA_SIM`, and `SBM_NATIVE_OPTIMIZATION`. See `SIMULATED_BIFURCATION.md` for solver algorithms, QUBO file format, accelerator details, and tuning parameters.

## Runtime Architecture

The main calculation flow is:

```text
YAML / typed configs
        |
        v
MarginEngine
  -> RiskStateGenerator.createDataRequest(portfolio, date)
  -> DataManager cache/backing store
  -> DownloadManager + selected DataProvider for missing intervals
  -> RiskStateGenerator.getRiskStates(context), lazily
  -> MarginCalculator
       |-> StateAwareGreedyMarginCalculator
       |    -> StateAwareGreedyRiskStateVisitor dispatches by risk-state type
       `-> BQMMarginCalculator
            -> PortfolioRiskStateBQMVisitor encodes QUBO
            -> BQMExecutionPolicy schedules one or many QUBOs
            -> BQMSolver solves them
            -> manager decodes the greatest loss
  -> MarginReport (margin plus stage timings)
```

`MarginApplicationConfig` in `src/margin_engine/yaml_application.py` is the public YAML composition boundary. It converts primitive, safely loaded YAML into typed configs and runtime collaborators. `MarginEngineConfig` then constructs an independent pipeline; avoid hidden global runtime state.

Backtesting calls only the public `MarginEngine` methods. It prefetches the union of required dates, calculates a daily margin, obtains realized prices, computes close-to-close realized P&L for each configured date, classifies breaches, and reports Basel traffic-light statistics. `run_backtest.py` writes CSV reports; `plot_backtest.py` renders a non-interactive PNG.

## Repository Map

- `src/margin_engine/`: top-level orchestration, strict YAML parsing, reports, and the `python -m margin_engine` CLI.
- `src/portfolio/`: portfolio weights and cash. Weights are `Decimal` values and represent return/P&L exposure, not share counts.
- `src/option_pricing/`: futures/equity option models, forward curves, volatility smiles, smile calibration, and volatility-shock estimation.
- `src/download_unit/`: provider-neutral `DataRequest`, providers, chunkers, single-request and retry/backoff download strategies.
- `src/download_manager/`: provider selection and download-strategy orchestration.
- `src/data_manager/`: in-memory LRU caching, interval-aware missing-data lookup, and optional partitioned pickle backing storage.
- `src/cache/`: small generic cache abstractions used by market data, PCA grids, and QUBO topology caches.
- `src/risk_state_generator/`: exponentially weighted PCA, shock-grid construction, optional cross-asset compatibility/correlation factors, and portfolio risk-state visitors.
- `src/margin_calculator/`: greedy and BQM margin strategies, compact QUBO representation, execution policies, and solver adapters.
- `src/backtesting/`: rolling evaluation, breach/Basel results, timing data, and CSV output.
- `src/sbm/`, `include/sbm/`: native C++17 simulated-bifurcation model, solvers, CLI, and Python C ABI bridge.
- `cuda/`: optional CUDA solver backend.
- `fpga/`: Vitis HLS kernel and declarations; the default C++ build includes a software simulation.
- `src/benchmark/`, `tools/`: native benchmarks and model conversion/summary utilities.
- `src/market_to_qubo.py`: standalone analysis/export pipeline that builds scenario QUBOs and an HTML report from market folders. It is related to, but separate from, `MarginEngine` orchestration.
- `options_margin_benchmark/`: ES/NQ ATP and CME benchmark portfolios, generated option-margin YAMLs, batch runner, result CSV, and matplotlib charts. Run its prepare script before regenerating results.
- `tests/`: Python `unittest` suite and native `tests/sbm_tests.cpp`.
- `config/`: canonical application example and backtest configurations.
- `synthetic_market/`, `vanguard_market/`: input datasets and portfolios. Treat these as data fixtures; do not mechanically reformat or regenerate them during unrelated work.

## Key Design Features and Invariants

### Typed, provider-neutral boundaries

Core services communicate with application types rather than third-party types. `DataRequest`, `RiskState`, `QUBOProblem`, `BQMOptimizationResult`, and `MarginReport` are the important boundaries. Keep pandas/yfinance/dimod/Torch/native-library details inside their adapters.

### Lazy and memory-aware processing

Risk states and encoded problems are iterators so large scenario spaces need not be fully materialized. Preserve streaming behavior. Use `BatchBQMExecutionPolicy` and its `batchSize`, `maxBatchBytes`, and `memoryMultiplier` controls for bounded batching. Compact QUBOs use contiguous numeric arrays, and reusable one-hot topology is cached by state shape and penalty.

### Explicit constraints and deterministic decoding

Each asset contributes one one-hot group to a QUBO. Solvers should prefer valid samples. Decoding defensively handles invalid heuristic output by choosing a deterministic worst candidate for missing or multiply selected states. Do not silently change this behavior: constraint selection directly affects calculated margin.

### Numerical and temporal correctness

- Margin and reported loss are non-negative; portfolio return/P&L can be signed.
- Market-data ranges are inclusive, dates must be unique where consumed, and instrument ordering must remain stable across requests, arrays, QUBO variables, samples, and decoding.
- Validate shapes, finite values, binary samples, one-hot group indices, date bounds, and nonzero price denominators at public boundaries.
- Preserve exact QUBO energy semantics: `offset + linear @ x + sum(bias * x[head] * x[tail])`.
- Backtests must not bypass `MarginEngine` storage/acquisition APIs or introduce look-ahead data into risk-state generation.

### Immutability and isolation

Configuration and request dataclasses are generally frozen. Mapping inputs are copied/proxied, and `QUBOProblem` arrays are made contiguous and read-only. Preserve these defensive copies so one engine or solver cannot mutate another calculation.

### Single-dispatch scenario behavior

`StateAwareGreedyMarginCalculator` owns only worst-P&L aggregation and constructs its `StateAwareGreedyRiskStateVisitor` dependency. The visitor uses `singledispatchmethod` to interpret each risk-state family; returns grids use independent greedy bounds while option states reprice all legs under one shared scenario. `PortfolioRiskStateBQMVisitor` similarly dispatches BQM encoding and decoding directly on raw risk states. Add new behavior at the relevant dispatch boundary instead of putting `isinstance` switches in calculators.

### Option-market conventions

`OptionMarketConvention` implementations own market-specific calibration behavior: pricing-model selection, spot/forward inputs, and historical underlying-price extraction. `VolatilitySmileCalibrator` and `VolatilityShockEstimator` select conventions at the dataframe boundary; derivative contracts remain independent of smiles and pricing models. Add a convention and register it with those services when supporting another option market. Implied volatility uses bounded, damped Newton-Raphson iterations; analytical vegas are used for European models and the pricing-model base class supplies a numerical vega for other models.

### Optional native acceleration

The Python application must remain usable without a compiled native library by selecting another solver. `SBMBQMSolver` loads the shared library at runtime; Torch and D-Wave adapters are separate implementations. CPU, CUDA, Torch, and FPGA-simulation changes should retain the same QUBO convention and be compared by original QUBO energy, not only terminal dynamics.

## Configuration Model

The root YAML contains `marginDate`, `portfolio`, `engine`, and optionally `backtest`.

- Portfolio input is either inline `weights` or `csv`, optionally with `clientId`; CSV supports long `client_id,ticker,weight` and wide `client_id,<ticker>...` forms.
- Providers are `local_csv`, `derivative_csv`, or `yfinance`; current provider selection is `local_first`.
- Long-form option-chain input uses provider `derivative_csv` with data manager type `derivative_quotes`; see `config/options.example.csv`.
- Download algorithms are `single_request` and `exponential_backoff`; chunkers are `date`, `instrument`, or nested `product` chunkers.
- The data backing store type is `partitioned_pickle`.
- Risk generators are `returns_vola_grid` and `correlated_returns_vola_grid`.
- Risk generator `option_scenarios` creates shared underlying-price/volatility stresses for one-symbol derivative portfolios.
- Margin calculators are `greedy`, `state_aware_greedy`, and `bqm`.
- BQM execution policies are `sequential` and `batch`.
- Registered solvers include `simulated_annealing`, `random`, `steepest_descent`, `tabu`, the tree/planar adapters, `sbm`, `torch_sbm`, and `adaptive_torch_sbm`.

Constructor options belong under `solver.constructorParameters`; per-call solve options belong under `solver.solverParameters`. Do not blur those lifecycles. When adding or renaming YAML options, update the strict parser, typed config, `config/margin.example.yaml`, and parser tests together.

## How to Make Changes

1. Read the nearest public interface and its tests before editing an implementation. Trace the composition path back through the relevant config and `yaml_application.py`.
2. Keep changes localized to the owning layer. For example, provider quirks belong in a provider adapter; scheduling belongs in an execution policy; risk-state math belongs in the generator/state types.
3. Add focused tests for success, invalid input, and the invariant most likely to regress. Use small deterministic arrays and stub collaborators; do not require network access in ordinary unit tests.
4. If behavior is configurable, add the typed config and strict YAML parsing in the same change. Add a representative YAML test and update the example.
5. Run the narrow tests first, then the complete Python and/or native suite. For numerical or solver changes, compare energy and decoded loss with a small deterministic reference problem.
6. Update this guide and `SIMULATED_BIFURCATION.md` when commands, architecture, solver semantics, or supported configuration change.

Do not edit unrelated generated outputs, caches, benchmark results, or large market-data files. Never commit `.venv/`, `build/`, `.cache/`, `backtest_results/`, or ad hoc exported QUBOs. Avoid network-dependent tests unless they are explicitly marked/skipped like the existing yfinance integration coverage.

## Feature Extension Playbooks

### Add a data provider

Implement `DataProvider` under `src/download_unit/data_provider/`, export it from the package, declare its supported data types, and translate `DataRequest` without leaking provider commands upstream. Add it to `_YamlConfigParser._provider`, document any provider parameters, and test conversion/filtering with mocks or local fixtures. If provider priority rules change, update `ProviderSelection` separately.

### Add a download or chunking strategy

Implement the corresponding abstract interface, preserve exact request coverage, and register/select it in `DownloadUnitFactory` or the YAML chunker parser. Test boundary dates, partial final chunks, ordering, duplicate avoidance, and error propagation.

### Add a risk-state family

Implement `RiskStateGenerator.createDataRequest` and lazy `getRiskStates`, add the raw risk-state type, and register the visitor operations needed by greedy/BQM calculation. Add a frozen typed config, select it in `_riskStateGenerator`, and test dimensions, instrument order, conditioning, and empty/degenerate history behavior.

### Add a margin calculator

Implement `MarginCalculator.calculateMargin`, add a config with `createMarginCalculator`, extend the `MarginCalculatorConfig` union and YAML parser, then test sign conventions and aggregation across multiple risk states. Keep orchestration out of `MarginEngine` so calculators remain interchangeable.

### Change option margining

Keep contract identity in `portfolio.derivatives`, valuation formulas and market conventions in `option_pricing`, contract dispatch in `OptionScenarioValuator`, calibration/stress construction in `OptionScenarioRiskStateGenerator`, risk-state dispatch in `StateAwareGreedyRiskStateVisitor`, and worst-P&L aggregation in `StateAwareGreedyMarginCalculator`. All option legs must share each scenario before P&L is aggregated. Do not choose independent worst states per leg. The current option path supports one symbol per portfolio and does not yet support derivative backtesting.

### Add a BQM solver

Subclass `BQMSolver`, return `BQMOptimizationResult`, and override `solveMany` only when real batching is supported. Respect variable ordering, original QUBO energy, binary output, `oneHotGroups`, series lifecycle hooks, and the constructor/solve parameter split. Register a stable snake-case name with `BQMSolverFactory`, export the module so registration occurs, add factory and deterministic energy tests, and add YAML coverage. A solver must not mutate `QUBOProblem` arrays.

### Change native SBM code

Keep public declarations in `include/sbm/` aligned with implementations and the C ABI in `src/sbm/python_api.cpp`. Test the native library with CTest and exercise the Python adapter when its ABI changes. Guard optional backends with existing CMake definitions. Changes shared with HLS must remain synthesizable in the HLS path; avoid unsupported dynamic allocation or library facilities there.

## Testing and Verification

Python tests use the standard library test runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python -m unittest tests.test_margin_calculator
```

The full suite requires dependencies from `requirements.txt`. The yfinance integration test is opt-in through its environment guard; normal tests should remain offline and deterministic.

Native tests and an optional optimized build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSBM_NATIVE_OPTIMIZATION=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For configuration changes, also smoke-test a small local/synthetic YAML. For backtesting/reporting changes, validate both CSV schemas and plotting. For performance work, first prove numerical equivalence on a small fixed seed, then use the programs under `src/benchmark/` or `tools/`; do not weaken correctness tests to accommodate a faster result.

Option benchmark generation and execution:

```bash
PYTHONPATH=src python options_margin_benchmark/prepare_benchmarks.py
PYTHONPATH=src python options_margin_benchmark/run_benchmarks.py
PYTHONPATH=src python options_margin_benchmark/plot_results.py
```

## Code Conventions

- Follow the existing Python style: type hints, short module/class docstrings, explicit validation, dataclasses for value/config objects, `pathlib.Path`, and dependency injection through constructors.
- Existing public Python APIs predominantly use camelCase method/field names (`generateReport`, `riskStateGenerator`). Match the surrounding module instead of introducing a second naming convention within the same API.
- Keep imports rooted at packages under `src` (for example, `from portfolio import Portfolio`) and run with `PYTHONPATH=src`.
- Prefer `collections.abc` for runtime container types and `TYPE_CHECKING` for annotation-only heavy imports.
- Keep optional heavy imports local where practical so selecting one backend does not require every accelerator runtime.
- Raise specific `TypeError` for wrong kinds and `ValueError` for invalid values/configuration. Error messages should identify the failing path or invariant.
- Preserve public `__init__.py` exports when introducing a public type.
- In C++, retain C++17 compatibility, RAII, contiguous buffers, explicit size validation, and compile-time backend guards.

## Agent Completion Checklist

Before handing off a change, report:

- what changed and which architectural layer owns it;
- tests/build commands run and their results;
- anything not run, including the concrete reason (missing dependency, accelerator, network opt-in, or time);
- configuration, data migration, ABI, memory, or numerical-accuracy implications.

Do not claim the full suite passed if collection failed or dependencies were absent. Preserve user changes in a dirty worktree and keep unrelated modifications out of the patch.

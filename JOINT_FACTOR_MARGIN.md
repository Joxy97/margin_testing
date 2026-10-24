# Joint factor margin application

`joint_factor_margin` connects the existing factor-stress research implementations
into an installable Python application. It preserves exposure units, EW PCA,
portfolio-aligned residuals, polynomial construction, integer-ball QUBOs, penalties,
repair, exact repricing, and numerical defaults. Existing Python imports and YAML
entry points remain available.

## Install and run

Requires Python 3.12 or later, matching the existing dataclass/mapping defaults.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[reports]'
factor-margin calculate examples/joint_factor_margin/equities.yaml
factor-margin calculate examples/joint_factor_margin/european.yaml --output /tmp/european-margin.json
factor-margin calculate examples/joint_factor_margin/vanilla.yaml --output /tmp/vanilla-margin.json
factor-margin backtest examples/joint_factor_margin/backtest.yaml --output /tmp/factor-backtest --plots
factor-margin backtest examples/joint_factor_margin/backtest.yaml --output /tmp/factor-backtest --resume
```

`python -m joint_factor_margin` has the same commands. Without installation, set
`PYTHONPATH=src`. Paths inside YAML resolve relative to the YAML file; command-line
output paths resolve relative to the working directory. The examples use a small,
fixed synthetic dataset and synthetic option marks. They demonstrate integration,
not model calibration or live market performance.

The base installation runs on CPU. Extras are `torch`, `online` (yfinance), and
`reports` (matplotlib). Install the appropriate PyTorch wheel for the target
hardware. CUDA bounds require a matching CuPy installation. Native CPU/CUDA/FPGA
simulation solvers use the existing CMake build and shared-library settings;
Python wheels do not build or bundle native binaries. No hardware-annealer adapter
is added. Existing registered solver adapters remain selectable.

## Python interface

```python
from joint_factor_margin import ApplicationConfig, BacktestRunner

settings = ApplicationConfig.fromYaml("examples/joint_factor_margin/equities.yaml")
application = settings.createApplication()
result = application.calculate(settings.request)
print(result.margin, result.pnl, result.coordinates)
```

For programmatic composition, construct `EngineConfig`, `ExposurePortfolio` and
`CalculationRequest`, then call `createApplication(engineConfig, configurationIdentity)`.
The supplied identity should identify the caller's settings and input revisions.
Prepared `EuropeanOptionBook` or `OptionBook` values can be supplied directly in a
request. `MarginApplication` also accepts injected market-data, scenario, search
and diagnostic implementations for other callers and tests. An application owns
its runtime caches and solver; concurrent callers should use separate applications.

Calculation returns immutable scenario arrays and diagnostic mappings. It writes
no artifacts. `reporting.writeJson` writes results explicitly, using atomic strict
JSON. Nonfinite diagnostic bounds retain a `nonFinite` tag rather than producing
invalid JSON or implying a finite certificate.

## Configuration and model choices

The strict schema is version 1. Unknown keys and duplicate YAML keys fail.
The examples show every main section; typed settings live in `settings.py`.

* `portfolio`: existing `weights` or portfolio `csv`/`clientId`, plus `cash`.
  Stock weights remain return/P&L exposure amounts, not shares. Cash has no
  scenario P&L under the existing model. All underlyings, including option-only
  underlyings, must occur in weights; use zero exposure when appropriate.
* `engine.downloadManager` and `engine.dataManager`: existing acquisition settings,
  including local-first providers, retry/chunking and optional partitioned storage.
  Local files and yfinance provide underlying closes. Option books are explicit
  dated snapshots; option-chain calibration is not silently inferred.
* `engine.estimation`: `ew_window: 125`, `ew_lambda: 0.93`, `components: 2`, and
  optional `pcaGridProvider` settings, including NumPy/Torch backend selection.
  Acquisition retains the legacy two-calendar-days-per-return request window;
  incomplete history fails rather than changing the fitting window.
* `engine.encoding`: existing `FactorStressQUBOConfig`; defaults are 6 bits,
  radius 3, safety 1.1 and penalty multiplier 1. Radius is an explicit stress
  budget, independently configured from backtest confidence.
* `engine.search`: `single_anchor` uses the existing center polynomial and optional
  factor repair. `boundary_aware` delegates existing seeds, smooth-anchor QUBOs,
  projection and nonlinear ranking to `solveOptionFactorStress`; it requires a
  vanilla option book and always repairs. Repair defaults remain 1024 moves,
  tolerance `1e-12`, `full` neighborhood. More than three coordinates requires
  explicitly selecting `pairwise` repair. Disabling repair rejects invalid raw
  encodings; it never substitutes a reference solution.
* `engine.solver`: existing factory `type`, `constructorParameters` and
  `solverParameters`, with their existing defaults and seed semantics. Specify a
  solver seed for reproducible heuristic runs. Solvers requiring complete one-hot
  groups cannot solve factor-stress QUBOs, which have no such groups.
* `engine.executionPolicy`: omitted retains direct-solve behavior. Explicit
  `sequential` or `batch` uses the existing policies. Batch accepts `batchSize`,
  `maxBatchBytes`, `memoryMultiplier`, and `prefetch`. Multi-anchor problems stream
  through the policy in original order; the policy bounds retained QUBOs and
  supports the solver's existing device sharding. A single-anchor run still has
  only one problem. Candidate results remain retained as in the existing search.
* `engine.volatility`: existing `VolatilityConfig`, omitted by default. EWMA,
  envelope, GARCH/GJR, calendar and jump settings preserve asset-row rescaling and
  the conservative expansion-only default. They do not implement the PDF's
  alternative fixed-factor-direction interpretation.
* `engine.diagnostics`: references and bounds are independently disabled by
  default. Existing continuous references, global quadratic certificates, stock
  Taylor/lattice bounds and margin brackets can be requested. `boundsBackend`
  selects `cpu` or optional `cuda`; `cudaDevice` defaults to 0. Coordinate-bounded
  nonlinear reference search is opt-in through `boundCoordinates`.

`options.type: european` uses the original European book and single-anchor path.
`options.type: vanilla` uses the existing European/American book and can use the
boundary-aware path. Spots map instrument names to prices; `marketPrices` follow
the ordered `positions`. Positions retain contract quantity and multiplier units.
Each snapshot explicitly supplies `valuationDate`, `horizonDate`, and rate. The
valuation date precedes `marginDate`; the horizon equals it. IV calibration,
supplied-IV status, pricing-domain failures, exercise boundaries and fixed-IV
repricing retain their existing behavior. Vanilla books expose
`suppliedVolatilities`, `volatilityBounds` and `currency`. Optional
`localPnlGradient` maps instruments to an existing supplied residual-alignment
vector for vanilla books, including nonsmooth-center use cases.

## Backtesting, identity and limitations

`BacktestRunner` adapts the application to the existing `MarginBacktester`.
History for fitting ends strictly before the evaluation date. Realized stock
P&L spans the preceding available close to the evaluation-date close, retaining
existing alignment/fill behavior and strict `realized_loss > margin` breach
classification. Existing missing-price errors propagate. This integration does
not introduce the PDF's forward-date convention or a new undetermined-day policy.

Backtest settings specify dates, a name and `confidenceLevel` (default 0.998).
Existing coverage evaluation and traffic-light classifications are reused.
Without a checkpoint store, Python replay has no file side effects. The CLI
explicitly creates a checkpoint store and writes the captured configuration,
manifest, `results.json`, and existing `breaches.csv`/`performance_metrics.csv`
schemas under the named portfolio folder. `--plots` adds `coverage.png`.
`--resume` reuses completed days only for the matching experiment identity.
Fresh runs clear only matching prior completions.

The new application uses its own configuration/experiment identity; legacy
checkpoints are not migrated or reused. Input files should remain immutable for
the lifetime of an application: its configuration identity is captured at creation.

Identity includes canonical configuration and its base directory, local input
contents, the selected native SBM binary, Python implementation sources, dependency
versions and timing version.
Online data provenance retains provider/revision semantics; cached data requires
an appropriate source revision. Driver versions are not automatically fingerprinted.

Option replay requires a separate dated realized-mark adapter and is rejected
explicitly. Existing rolling research scripts are not reinterpreted as a generic
mixed-book backtester. Stock-only Taylor bounds do not apply to option books and
return `unsupported`. Nonsmooth option references also report their applicability
failure. Numerical reference results retain their own success/certificate status;
they never replace solver candidates. The full-residual research model and its
certificates remain separate from this PDF's one-residual-direction engine.

## Module ownership and extension

`application.py` owns orchestration and composition. `market_data.py` adapts
existing acquisition. `scenarios.py` connects PCA and valuation through dispatch
and applies existing overlays. `search.py` adapts the two existing search paths;
polynomials, encoding, repair and exact valuation remain owned by their original
modules. `diagnostics.py` owns applicability and execution selection.
`configuration.py` reads YAML and reuses the public `MarginConfigParser` input
readers. `backtest.py` implements the existing backtest engine interface.
`reporting.py` owns explicit artifact writes; `cli.py` only coordinates these calls.

New valuation families belong at the valuation dispatch and existing pricer/model
interfaces. New solvers belong in the existing solver factory. New numerical
behavior requires its own model change and tests; integration adapters should
preserve the original implementations. Dependencies enter module constructors,
and callers test through the same small interfaces used by the application.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_joint_factor_margin.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_factor_stress*.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
python -m build --no-isolation
```

The focused suite compares geometry, coefficients, source energy, coordinates,
feasibility and exact P&L against existing implementations; it also exercises
future-data exclusion, batching, strict settings, immutable results, input identity,
CLI execution, resume, CSVs and plots. Plot verification requires the reports extra.
Some legacy tests reference fixtures/scripts absent from this checkout; report
those failures independently from application verification.

### Verification of this integration

Verified on Python 3.12 with CPU-only PyTorch:

* All 17 application integration tests pass. Comparison against the original
  `HEAD` option search also matched every candidate sample, candidate order,
  feasibility result, selected coordinates and P&L on the deterministic fixture.
* Complete Python discovery (`-v` with the command above): 301 tests, 290 passed,
  7 skipped, 4 errors. Errors reference absent legacy inputs:
  `tests/test_european_factor_stress.py`, `maxcut_solver_benchmark.metrics`,
  `plot_backtest_comparison`, and `config/margin.example.yaml`.
* `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSBM_BUILD_FPGA_SIM=OFF`,
  `cmake --build build -j 2`, and `ctest --test-dir build --output-on-failure`
  pass (1 native test). Full margin calculations with native SBM and Torch SBM
  both pass on CPU with factor encoding feasibility verified.
* Editable installation, sdist/wheel builds (`python -m build --no-isolation`),
  all five CLI examples, and backtest resume/CSV/PNG publication pass. An isolated
  wheel installation runs the four calculation examples outside the repository
  and imports core modules from the installed wheel. The source archive includes
  example YAMLs and prices.
* CUDA/ROCm tests and CUDA bounds were not run: this runtime has no available GPU
  and uses a CPU-only Torch wheel. FPGA simulation was not built because
  `fpga/dsb_hls.cpp` is absent from this checkout. Live yfinance testing remains
  disabled behind its existing network opt-in.

The change introduces an application YAML schema and optional ordered anchor
batching. It does not change the native ABI, model equations, penalty or repair
defaults, or input-data formats. Batch memory admission reuses the existing policy;
retained repaired candidates have the same lifetime as the original option search.

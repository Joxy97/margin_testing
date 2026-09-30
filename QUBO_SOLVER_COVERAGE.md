# Historical 20-solver benchmark coverage

This report preserves the earlier 20-entry run. Current coverage is **28 canonical
library methods**, documented in [SOLVER_LIBRARY.md](SOLVER_LIBRARY.md), with
current validation in [SOLVER_LIBRARY_VALIDATION.md](SOLVER_LIBRARY_VALIDATION.md).
The configuration files now use the consolidated registry. Counts and commands
below describe the historical run, not the current selection.

The harness now accounts for all 20 registered solver names. Fifteen existing
solver interfaces can handle the 37 catalog QUBOs unchanged: ten GPU-capable
Torch implementations and five CPU implementations. The other five names are
recorded as structurally unsupported instead of changing the published problems.

| Solver entries | Count | Device and compatibility |
|---|---:|---|
| torch_sbm, adaptive_torch_sbm, torch_svl, torch_transverse_route, torch_exchange_cascade, torch_altermagnet, torch_dynamical_geometry, torch_geometric, torch_phonon_exchange, torch_supersymmetric | 10 | CPU or CUDA; all 37 compatible |
| random, simulated_annealing, steepest_descent, tabu | 4 | Local CPU, float64; all 37 compatible |
| sbm | 1 | Native CPU C ABI, float32; requires compiled sbm_python library |
| torch_categorical, torch_categorical_trf | 2 | GPU-capable algorithms requiring complete one-hot groups; none of these 37 QUBOs qualify |
| planar_graph | 1 | Requires planar, zero-field Ising structure; all 37 rejected |
| tree_decomposition_solver, tree_decomposition_sampler | 2 | Installed backend width limit 25; all 37 rejected |

The native repository also contains a CUDA CLI backend, but the registered
`sbm` Python adapter explicitly calls the CPU C ABI. It must not be reported
as GPU execution. No solver algorithm was modified or substituted in this work.

## Prepared configurations

All three full configurations select the same 37 local instances and explicit
seeds 0 through 19, with a 60-second solve cap and 120-second setup cap.

| File | Selection | Result slots |
|---|---|---:|
| `benchmark_configs/full.json` | All 10 compatible GPU solvers, cuda:0 / float32 | 7,400 |
| `benchmark_configs/full_compatible.json` | All 15 compatible entries, with explicit CPU overrides | 11,100 |
| `benchmark_configs/full_all.json` | All 20 registered names, retaining unsupported cases | 14,800 |
| `benchmark_configs/smoke_all.json` | All 20 names on six smoke instances, CPU, seed 0, short settings | 120 |

Run from the repository root with `src` on `PYTHONPATH`:

```bash
python -m pip install -r requirements-qubo-benchmark.txt
export PYTHONPATH=src
python -m qubo_benchmark solvers
python -m qubo_benchmark validate
python -m unittest discover -s tests -p test_qubo_benchmark.py -v
python -m qubo_benchmark run --config benchmark_configs/full.json --output benchmark_results/runs/full-gpu
```

For native SBM, build the existing library before selecting it:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sbm_python --config Release -j
PYTHONPATH=src python -m qubo_benchmark run --config benchmark_configs/full_compatible.json --output benchmark_results/runs/full-compatible
```

The default Linux library path is `build/libsbm_python.so`. A solver entry can
specify `constructor: {"libraryPath": "path/to/library"}` to use another path.
The Torch entries still require a working CUDA PyTorch installation for GPU
runs. CPU wrappers never silently fall back from a GPU request; they have explicit
per-solver device and precision settings. No dataset download is part of a run.

## Validation

The command below passed **21 tests in 26.481 seconds** on the local CPU environment:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
& C:/Users/Lazar/OneDrive/Documents/.solver-validation/Scripts/python.exe -m unittest discover -s tests -p test_qubo_benchmark.py -v
```

Tests cover all ten Torch variants, all seven classical wrappers on compatible
small QUBOs, exact planar/tree objective checks, registry coverage, mixed CPU/GPU
configuration, deterministic seed handling, unsupported structures, missing native
libraries, process deadlines, all 37 imports and the four published witnesses.
The package versions actually exercised were dimod 0.12.22, dwave-samplers 1.8.0,
networkx 3.7 and CPU PyTorch 2.10.0. Native SBM's unavailable-dependency path was
tested; a native solve could not be run because the shared library, CMake and a
C++ compiler are absent locally. CUDA execution remains untested locally.

`benchmark_results/solver_structure_audit.json` records checks on every instance.
For dense cases, the exact lower bound `treewidth >= minimum graph degree`
already exceeds 25. For the remaining three 200-variable sparse cases, the
backend's actual elimination widths are 130, 95 and 116. This bound avoids
expensive ordering work for structurally impossible cases. The planar rejection
checks the true Ising field, not simply the original QUBO diagonal. These checks
do not alter the matrices, add bits or introduce constraints.

The broad local smoke command is:

```powershell
& C:/Users/Lazar/OneDrive/Documents/.solver-validation/Scripts/python.exe -m qubo_benchmark run --config benchmark_configs/smoke_all.json --output benchmark_results/smoke_all_solvers_final
```

Its authoritative per-trial records and summary are in that output directory.
The completed run recorded **120 entries: 84 completed solves, 30 expected
structural rejections, and six unavailable native-SBM entries**. Each of the
ten Torch implementations and four general-purpose classical samplers completed
all six smoke instances with independently validated binary outputs. There were
no timeouts, invalid outputs, solver errors, or below-proven-optimum alarms in
the final run. The CLI correctly returned exit code 1 because native SBM was
unavailable; the run is not described as 120 successful solves. GPU-capable
implementations were exercised on CPU, not on CUDA hardware.
The initial `smoke_all_solvers` attempt is marked `ABORTED.json` after it exposed
expensive tree-ordering preflight; use the final directory for results.

## Reporting semantics

`completed` means a real returned binary sample was independently rescored in
the original normalized objective. `unsupported` means an existing algorithm's
problem contract or width limit rejects this instance; its objective is null.
`unavailable` means a required local dependency/library/device is absent. These
statuses are separate in summaries. Unsupported cases are never counted as solves;
unavailable cases and other failures produce nonzero CLI exit status.

Planar and exact tree-solver repetitions are labeled deterministic timing
repetitions and receive no invented seed argument. Stochastic CPU samplers receive
the explicit seed through their supported interface. Adaptive SBM disables its
optional coordinate polish in the supplied GPU config. Every solver continues to
use the same independent scoring and deadline enforcement as the initial harness.
The `cpu_threads` setting controls PyTorch threads; other CPU backends retain
their own native thread policies.

Changed implementation files: `src/qubo_benchmark/adapters.py`, `runner.py`,
`__main__.py`; `tests/test_qubo_benchmark.py`; `requirements-qubo-benchmark.txt`;
the four configurations listed above; `AGENTS.md`, `QUBO_BENCHMARK.md`, this report,
and the historical report's cross-reference. Results/provenance are saved under
`benchmark_results/`. No full campaign, cloud instance, commit or push was started.

# Solver library

`src/qubo_solvers` owns every retained solver, compact problem/result types,
candidate selection, and execution helpers. There are **28 canonical solvers:
17 native tensor algorithms and 11 specialized backends**. All run on CPU;
27 also provide Torch GPU search kernels. Only `lib_planar_graph` is CPU-only.
CUDA execution is prepared but has not been tested on this CPU-only machine.

This deployment branch contains the standalone library and benchmark adapters.
The portfolio application and its integration tests remain on the earlier
branch and are not required to run the benchmark. Historically, the
application factory and benchmark adapters call this library. Historical
application module/class imports are compatibility shims with no solver kernels.
Old solver IDs are rejected with migration hints, rather than registered as
additional algorithms. Nine duplicate entries were removed from the earlier 37:
seven overlapping implementation pairs and three SBM implementations combined
into one configurable implementation.

## Installation

The core tensor API requires Torch and Python 3.10 or later. Choose dependencies
for the interface you use:

```bash
python -m pip install -e .
python -m pip install -e ".[compact]"
python -m pip install -e ".[benchmark,dev]"
```

`compact` adds NumPy/SciPy for compact QUBOs and specialized Torch backends.
`benchmark` also supplies offline validation and the planar backend's optional
dependencies. `dev` supplies
pytest and process-memory profiling. These commands are alternatives or can be
combined. The core has no direct application or NumPy dependency.

GPU execution requires a CUDA-enabled Torch installation and a compatible NVIDIA
driver on the destination host. No CMake build, native SBM shared library, custom
CUDA extension, or local CUDA compiler is required. Install the appropriate Torch
wheel for that host before running tests. An explicit unavailable CUDA device
raises an error; it does not silently select CPU.

Use an editable install or `PYTHONPATH=src` for checkout-based benchmarking.
Wheels contain the library, not the versioned benchmark datasets; retain
`benchmark_data/qubo37` in the transferred checkout for offline runs.

## Canonical inventory and migration

| Native tensor algorithms (17) | Specialized backends (11) |
|---|---|
| `lib_simulated_annealing` | `lib_simulated_bifurcation` |
| `lib_greedy_local_search` | `lib_altermagnet` |
| `lib_spin_vector_langevin` | `lib_dynamical_geometry` |
| `lib_angular_annealing` | `lib_geometric` |
| `lib_transverse_route` | `lib_phonon_exchange` |
| `lib_easy_axis_annealing` | `lib_supersymmetric` |
| `lib_spin_coherent_annealing` | `lib_categorical` |
| `lib_vector_amplitude_annealing` | `lib_categorical_trf` |
| `lib_mean_field_annealing` | `lib_planar_graph` (CPU only) |
| `lib_tap_annealing` | `lib_tree_decomposition_solver` |
| `lib_spherical_annealing` | `lib_tree_decomposition_sampler` |
| `lib_contact_annealing` | |
| `lib_replica_annealing` | |
| `lib_heat_bath_annealing` | |
| `lib_tabu_search` | |
| `lib_random_search` | |
| `lib_exchange_cascade` | |

`SOLVERS`, `NATIVE_SOLVERS` (also `LIBRARY_SOLVERS`), `BACKEND_SOLVERS`,
`GPU_SOLVERS`, `CPU_SOLVERS`, and `solver_capabilities(name)` expose the inventory
and restrictions. Related methods remain separate where their updates differ:
Metropolis versus heat bath, mean field versus TAP, and distinct continuous
annealing methods are not collapsed into one solver.

| Retired duplicate IDs | Retained implementation |
|---|---|
| `simulated_annealing` | `lib_simulated_annealing` |
| `steepest_descent` | `lib_greedy_local_search` |
| `random` | `lib_random_search` |
| `tabu` | `lib_tabu_search` |
| `torch_svl` | `lib_spin_vector_langevin` |
| `torch_transverse_route` | `lib_transverse_route` |
| `torch_exchange_cascade` | `lib_exchange_cascade` |
| `sbm`, `torch_sbm`, `adaptive_torch_sbm` | `lib_simulated_bifurcation` |

SBM selects `dynamics: standard` or `dynamics: adaptive` in solve parameters;
`standard` is the default. Migrate parameters when changing IDs: retaining the
method does not imply identical old defaults, trajectories, or postprocessing.
Other old backend IDs have corresponding canonical names in `REMOVED_SOLVERS`.

## Direct tensor interface

```python
import torch
from qubo_solvers import QUBO, create_solver

problem = QUBO(
    torch.tensor([[-1., .25], [.25, -2.]], dtype=torch.float64),
    offset=3.,
)
solver = create_solver("lib_spin_vector_langevin", max_steps=100)
result = solver.solve(
    problem, restarts=16, seed=42, batch_size=8, best_only=True,
    memory_limit_bytes=512 * 1024**2,
)
print(result.best_assignment, result.best_energy)
```

Use `problem.to(device="cuda:0")` to select CUDA explicitly. The 17 native
algorithms execute on the problem device and return device tensors, with float32
or float64 coefficients. Objectives are `x.T @ Q @ x + offset` and
`s.T @ J @ s + h.T @ s + offset`, with `s = 2*x - 1`.

`best_only=False` retains final and best assignments, energies, iteration counts,
and termination reasons for each restart. `history_interval` optionally records
iteration-indexed energies, not time-to-target measurements. Best-only mode keeps
one winner and rejects histories. Reproducibility requires matching algorithm,
seed, dtype, device, and batching settings; changing batches may change random
number consumption.

`estimate_memory` and `memory_limit_bytes` cover estimated tensor storage and
workspace, excluding allocator, BLAS, and CUDA context overhead. They are not
operating-system memory ceilings. Batching bounds trajectory workspace while
dense coefficients still require quadratic storage. Factory defaults use 1,000
steps/sweeps; `default_algorithm_parameters(name, steps=...)` resolves defaults
for native 17. Direct class constructors retain their class defaults.

`create_solver` also exposes specialized backends through a tensor wrapper
requiring `best_only=True`. It returns only the selected candidate; unavailable
iteration counts, termination reasons, and restart indices are `-1`. Histories,
supplied starts, common restart batching, and common memory limits are explicitly
rejected. Use the compact interface for backend-specific parameters, sparse
inputs, categorical groups, and resident application execution. The wrapper
performs host conversion and is not a fully resident tensor path.

## Compact QUBOs and application YAML

```python
import numpy as np
from qubo_solvers import create_bqm_solver
from qubo_solvers.backends.problem import QUBOProblem

problem = QUBOProblem(
    linear=np.array([-1., -2.]),
    quadraticHeads=np.array([0], dtype=np.uint32),
    quadraticTails=np.array([1], dtype=np.uint32),
    quadraticBiases=np.array([.5]),
    offset=3.,
)
solver = create_bqm_solver("lib_mean_field_annealing", {"device": "cpu"})
result = solver.solve(problem, {
    "steps": 100, "runs": 16, "run_batch_size": 8,
    "seed": 42, "dtype": "float64",
})
print(result.sample, result.energy)
assert result.energy == problem.energy(result.sample)
```

The same registry is used by `BQMSolverFactory` and the application's strict YAML
parser. This fragment belongs under `engine.marginCalculator`:

```yaml
type: bqm
solver:
  type: lib_mean_field_annealing
  constructorParameters:
    device: cuda:0
  solverParameters:
    steps: 1000
    runs: 16
    run_batch_size: 8
    dtype: float32
    seed: 42
```

For native algorithms, `steps` maps to SA/heat-bath sweeps or other algorithms'
`max_steps`; `runs` maps to `restarts`, and `run_batch_size` to `batch_size`.
Native algorithm options can appear in `solverParameters`. Unknown/conflicting
options fail. SA/heat-bath factory temperatures default to 2 down to 0.01.

Compact objectives use `offset + linear @ x + sum(bias*x[head]*x[tail])`.
Conversions accumulate duplicate, reversed, and diagonal entries and preserve
offsets; off-diagonal pair-once coefficients become symmetric half-bias entries.
Candidate ranking uses the original float64 coefficients. One-hot repair stays
outside native search algorithms; grouped problems reject native best-only mode
because selection needs all restart candidates. The native compact facade checks
a conservative combined host-and-tensor memory estimate before dense conversion,
which is stricter than the direct tensor estimate and still excludes context and
allocator overhead.

Resident preparation is device-based for 18 entries (native 17 plus SBM), and
uses an explicit authoritative host snapshot for seven (five physics and two
categorical backends). Planar and both tree methods have no resident preparation
hook. Consult `solver_capabilities` instead of assuming every GPU solver accepts
the same lifecycle. Multi-device execution shards independent QUBOs where
supported; one QUBO uses one device, and unsupported plans fail explicitly.

## GPU boundaries and exact tree methods

GPU capability describes search kernels, not an entirely GPU-only application.
Host setup, source-objective scoring, and optional one-hot repair remain explicit.
Physics solvers retain CPU initialization/calibration; phonon calibration uses
SciPy. Optional conditional rounding, local search, block refinement, and proof
search can also perform CPU work. Full benchmark configurations disable optional
conditional rounding, local search, and block refinement; standard SBM uses no
local polish and proof searches are off.

Both tree methods share bucket elimination: CPU graph ordering precedes Torch
factor-table reductions (`min` for optimization, `logsumexp` for Boltzmann
sampling), followed by device backtracking. `elimination_preflight` in
`qubo_solvers.backends.tree_decomposition` checks the order and table memory
before exponential allocations. Maximum supported width is 25 and the default
estimated budget is 512 MiB. Explicit orders are supported. A heuristic width
rejection is a limit of the selected order, not proof of minimum graph treewidth.
Use float64 for exact-reference comparisons; floating-point rounding still
applies. All 37 catalog instances exceed the accepted width checks.

Native best-candidate gathering avoids host scalar extraction, exchange cascade
keeps its degree scale on device, and each solve reuses immutable spin-objective
tensors between batches. Metropolis/heat-bath still update variables sequentially
and greedy stopping retains host synchronization. CUDA accuracy, latency,
allocator peaks, and multi-GPU behavior require validation on actual hardware;
CPU profiling cannot establish a GPU speedup.

## Offline benchmarks and GPU validation

All six groups and all 37 objectives are versioned in `benchmark_data/qubo37`.
No benchmark run or test downloads data. Twenty-three solvers accept every
catalog instance unchanged. The two categorical methods require complete
one-hot groups, planar requires planar zero-field structure, and both tree
methods exceed the configured width; these five receive `unsupported` records.

| Configuration | Selection | Scheduled trials |
|---|---|---:|
| `benchmark_configs/full.json` | 23 compatible GPU entries, 37 instances, 20 seeds | 17,020 |
| `benchmark_configs/full_compatible.json` | Same 23 compatible entries | 17,020 |
| `benchmark_configs/full_all.json` | All 28, including explicit unsupported records | 20,720 |
| `benchmark_configs/smoke_all.json` | All 28, six instances, CPU, one seed | 168 |

The local six-instance all-solver smoke completed 138 solves with 30 expected
unsupported records and no errors. Completion validates loading, invocation, and independent scoring; it
does not require finding an optimum. Actual local results and unresolved checks
are recorded in `SOLVER_LIBRARY_VALIDATION.md`.

On the prepared NVIDIA host:

```bash
python -m qubo_benchmark validate
python -m pytest tests/qubo_solvers tests/test_qubo_benchmark.py
python tools/profile_solver_gpu.py --device cuda:0 --solvers all --instances gka1e bqp500-1 bqp1000-1 --output benchmark_results/gpu-native-profile.json
python -m qubo_benchmark run --config benchmark_configs/full_all.json --output benchmark_results/runs/full-library-all
```

The profiling tool covers native 17 only, refuses unavailable CUDA, warms up,
synchronizes timed regions, separates resident and transfer-inclusive timings,
and independently scores outputs. Use `--profile-ops` for a separate operator
profile and `--device cpu` for CPU characterization. Output files must be new.
Use the benchmark runner for specialized backends. Full campaigns were not run
locally. Benchmark records include hashes, resolved settings, device, precision,
timing, and independent scoring; CUDA allocated/reserved peaks exclude other
processes and are not total device memory or allocation traffic.

The GPU and memory workflow used the installed `optimize-for-gpu` and
`memory-optimization` skills. Skill reference: Kassis, T., Agarwal, V., He, Y.,
Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of
Procedural Knowledge for Research Agents*,
https://doi.org/10.48550/arXiv.2609.00065.

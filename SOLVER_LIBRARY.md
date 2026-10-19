# Solver library

`src/qubo_solvers` owns every retained solver, compact problem/result types,
candidate selection, and execution helpers. There are **53 canonical solvers:
42 native tensor algorithms and 11 specialized backends**. All run on CPU;
52 also provide Torch GPU search kernels. Only `lib_planar_graph` is CPU-only.
The 23 Hamiltonian additions are listed by `HAMILTONIAN_SOLVERS`; see
[HAMILTONIAN_MODELS_IMPLEMENTATION.txt](HAMILTONIAN_MODELS_IMPLEMENTATION.txt)
for their model mapping, memory limits, numerical variants and CUDA validation.
The two added methods have passed CPU/CUDA validation on actual NVIDIA hardware;
the scope and limits of that validation are recorded below.

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

| Native tensor algorithms (19) | Specialized backends (11) |
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
| `lib_su2_chiral_isospin` | |
| `lib_gravity_tunneling` | |

`SOLVERS`, `NATIVE_SOLVERS` (also `LIBRARY_SOLVERS`), `BACKEND_SOLVERS`,
`GPU_SOLVERS`, `CPU_SOLVERS`, and `solver_capabilities(name)` expose the inventory
and restrictions. Related methods remain separate where their updates differ:
Metropolis versus heat bath, mean field versus TAP, and distinct continuous
annealing methods are not collapsed into one solver.

`SU2ChiralIsospin` evolves real unit Bloch vectors with a chiral homotopy,
rank-two skew action, and explicit geometric midpoint rotations.
`GravityTunneling` evolves a coupled population with softened configuration-space
gravity, reflected damped Verlet steps, and stochastic binary relocation.
Both retain decoded candidates using the original objective in float64. Their
state updates can use float32 or float64 through `dynamics_dtype`; returned
energies use float64 for these two methods. Compact inputs preserve original
float64 coefficients and select sparse operators through `sparse_threshold`.
Candidate scoring uses raw pair-once terms in bounded chunks before retention,
including duplicate, reversed, and self-loop records; force preparation may
aggregate those terms separately. Float64 still rounds arbitrary coefficients,
and source precision lost before calling the API cannot be recovered.
The original 28 methods and their numerical settings are preserved. The two
additions are exploratory classical heuristics with no established performance
advantage or optimality guarantee.

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

Use `problem.to(device="cuda:0")` to select CUDA explicitly. The 42 native
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
steps/sweeps for the original 17 native methods and 512 steps for the two added
methods. `default_algorithm_parameters(name, steps=...)` resolves the supplied
schedule for all 42 native methods; its omitted `steps` default remains 1,000.
Direct class constructors retain their class defaults.
The 23 Hamiltonian factory paths retain their model-specific class defaults;
their step counts and numerical choices are documented in the implementation report.

The two added classes and their factory paths default to 512 steps. Benchmark
initial settings use 16 trajectories in a single batch and candidate checks
every 16 steps. Gravity's batch width determines its interacting swarm and is
part of its fixed algorithm settings. Historical presets remain unchanged;
preflight appends the two methods from
`configs/benchmark_solvers_new_physics.json` and freezes all resolved settings.

To use float32 dynamics with original float64 source scoring on CUDA:

```python
solver = create_solver("lib_su2_chiral_isospin", dynamics_dtype="float32")
result = solver.solve(
    problem.to(device="cuda:0", dtype=torch.float64),
    restarts=16, batch_size=16, seed=42, best_only=True,
    memory_limit_bytes=512 * 1024**2,
)
```

Use `lib_gravity_tunneling` for the second method. Fixed-step schedules are the
ordinary API contract. SU(2)'s explicitly enabled wall schedule is a separate
protocol; gravity retains its fixed-step controls under such an observer.

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
because selection needs all restart candidates. The original native compact facade checks
a conservative combined host-and-tensor memory estimate before dense conversion,
which is stricter than the direct tensor estimate and still excludes context and
allocator overhead.

For the two added methods, `create_bqm_solver(...).solve(...)` uses direct compact
preparation, preserving raw source terms and avoiding dense conversion for sparse
inputs. The classes also expose `solve_compact(..., device=..., precision=...)`
and `solve_compact_resident(...)`; the resident path requires original float64
coefficient arrays already on the selected device. `estimate_compact_memory`
accounts for their preparation and population storage. In contrast, historical
and continuous benchmark workers cache a dense float64 native input for these
two methods, even when the source input is sparse. Their admission estimates
include that matrix and operator construction. Use the compact API when avoiding
quadratic sparse-input storage matters. Memory guards do not adjust gravity's
swarm width or retry with different algorithm settings.

Resident preparation is device-based for 20 entries (native 19 plus SBM), and
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
The historical benchmark inputs require no downloads. Twenty-five solvers accept every
catalog instance unchanged. The two categorical methods require complete
one-hot groups, planar requires planar zero-field structure, and both tree
methods exceed the configured width; these five receive `unsupported` records.

| Configuration | Selection | Scheduled trials |
|---|---|---:|
| `benchmark_configs/full.json` | 25 compatible GPU entries, 37 instances, 20 seeds | 18,500 |
| `benchmark_configs/full_compatible.json` | Same 25 compatible entries | 18,500 |
| `benchmark_configs/full_all.json` | All 30, including explicit unsupported records | 22,200 |
| `benchmark_configs/smoke_all.json` | All 30, six instances, CPU, one seed | 180 |

The historical 28-method six-instance all-solver smoke completed 138 solves with 30 expected
unsupported records and no errors. Completion validates loading, invocation, and independent scoring; it
does not require finding an optimum. Actual local results and unresolved checks
are recorded in `SOLVER_LIBRARY_VALIDATION.md`.

On 2026-10-14, the two added methods passed CPU/CUDA checks on an NVIDIA
RTX 4070 SUPER host with Torch 2.11.0+cu128. The fixture-enabled standalone
regression recorded 1,549 passed, 50 subtests passed, three portfolio application
cases deselected, and two warnings. All 37 offline QUBOs passed validation.
Both methods also matched independent original-source scoring on `bqp500-1`
and `bqp1000-1` with their 512-step CUDA defaults. Synthetic dense and sparse
probes checked synchronized timing, allocator peaks, and stable live allocations
across repeated solves. This evidence covers the new methods and regression
checks; it does not establish every method's performance on all GPUs, comparative
solution quality, or a completed 100-seed continuous campaign. Earlier four-GPU
runtime smoke evidence remains in [MULTIGPU_VALIDATION.md](MULTIGPU_VALIDATION.md).

On the prepared NVIDIA host:

```bash
python -m qubo_benchmark validate
python -m pytest tests/qubo_solvers tests/test_qubo_benchmark.py
python tools/profile_solver_gpu.py --device cuda:0 --solvers all --instances gka1e bqp500-1 bqp1000-1 --output benchmark_results/gpu-native-profile.json
python -m qubo_benchmark run --config benchmark_configs/full_all.json --output benchmark_results/runs/full-library-all
```

The profiling tool covers native 19 only, refuses unavailable CUDA, warms up,
synchronizes timed regions, separates resident and transfer-inclusive timings,
and independently scores outputs. Use `--profile-ops` for a separate operator
profile and `--device cpu` for CPU characterization. Output files must be new.
Use the benchmark runner for specialized backends. Full campaigns were not run
locally. Benchmark records include hashes, resolved settings, device, precision,
timing, and independent scoring; CUDA allocated/reserved peaks exclude other
processes and are not total device memory or allocation traffic.

`tools/profile_new_physics_solvers.py` provides a bounded synthetic probe for
the two additions. It separates preparation from resident search and checks
original-source scores and live allocations. Its timings are not quality or
time-to-solution results:

```bash
python tools/profile_new_physics_solvers.py --device cuda:0 --steps 512 --sizes 1000,5000 --runs 16 --repeats 2 --output benchmark_results/new-physics-probe.json
```

The GPU and memory workflow used the installed `optimize-for-gpu` and
`memory-optimization` skills. Skill reference: Kassis, T., Agarwal, V., He, Y.,
Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of
Procedural Knowledge for Research Agents*,
https://doi.org/10.48550/arXiv.2609.00065.

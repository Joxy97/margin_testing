# Simulated Bifurcation Machine

This implementation follows Algorithm 1 of Orlando et al., *High-Parallel
FPGA-Based Discrete Simulated Bifurcation for Large-Scale Optimization*
(arXiv:2510.12407v2). It implements discrete simulated bifurcation (dSB), with
the paper's optional heating term, in three forms:

- `solve_cpu`: sparse CSR matrix-vector updates, parallelized with OpenMP when
  available. Rows and trajectories are parallelized adaptively, while coupling
  reductions and time evolution use SIMD lanes. `solve_cpu_batch` additionally
  schedules independent small QUBOs across cores.
- `solve_gpu`: CUDA CSR kernels, enabled with `SBM_ENABLE_CUDA`.
- `TorchSBMBQMSolver`: block-diagonal sparse matrix-matrix updates that batch
  both independent QUBOs and randomized trajectories on CPU, one GPU, or a
  configured set of GPUs.
- `dsb_hls`: a fixed-size, fixed-point-capable Vitis HLS kernel with `P_r` row
  and `P_c` column unrolling plus `P_b` replicated row blocks. `solve_fpga_sim`
  runs the same kernel as ordinary C++ with floating-point values for verification.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/sbm_solve examples/two_variable.qubo --backend cpu
```

For CUDA, configure with `-DSBM_ENABLE_CUDA=ON`. The CUDA toolkit and a CUDA
GPU are required. The HLS source is in `fpga/dsb_hls.cpp`; define
`SBM_USE_XILINX_AP_FIXED` when compiling it with Vitis HLS. Its defaults are 256
spins, `P_c=16`, `P_r=16`, and `P_b=1`; override `SBM_FPGA_MAX_SPINS`,
`SBM_FPGA_PC`, `SBM_FPGA_PR`, and `SBM_FPGA_PB` at compile time. Biases should be scaled to fit the configured
`ap_fixed<24,8>` range before loading a physical FPGA.

## Batched Torch solver

Select the Torch implementation in application YAML without changing the
margin engine or its execution policy:

```yaml
engine:
  marginCalculator:
    type: bqm
    executionPolicy:
      type: batch
      batchSize: 105
      maxBatchBytes: 536870912
    solver:
      type: torch_sbm
      constructorParameters:
        device: auto  # CUDA when available, otherwise CPU
      solverParameters:
        steps: 1000
        runs: 16
        run_batch_size: 16
        dtype: float32
        dt: 0.1
        a0: 1.0
        c0: 0.02
        gamma: 0.0
        initial_scale: 0.1
        seed: 20260603
```

The ordinary `torch` dependency provides CPU execution. For CUDA, install the
wheel selected for the machine's CUDA version by the official PyTorch package
selector. `run_batch_size` bounds trajectory-state memory; the outer BQM batch
size bounds the block-diagonal scenario matrix. Use `float64` when validating
against the native solver and `float32` for the higher-throughput path.

To distribute every outer BQM batch across eight GPUs, replace the singular
`device` option with explicit indexed devices:

```yaml
executionPolicy:
  type: batch
  batchSize: 105  # Keep this at least as large as the GPU count.
  maxBatchBytes: 536870912
  memoryMultiplier: 3.0
solver:
  type: torch_sbm
  constructorParameters:
    devices:
      - cuda:0
      - cuda:1
      - cuda:2
      - cuda:3
      - cuda:4
      - cuda:5
      - cuda:6
      - cuda:7
  solverParameters:
    steps: 1000
    runs: 16
    run_batch_size: 16
    dtype: float32
```

The solver divides each ordered batch into contiguous, approximately equal
problem-count shards and executes one shard per GPU concurrently. Results are
restored to input order. Every `QUBOProblem` carries a stable identity-derived
seed offset, so its trajectories do not change when execution-policy batch
boundaries or device shards change. The native C ABI likewise accepts one seed
per problem while retaining batched execution. A batch with fewer
problems than devices uses only as many GPUs as it has problems. Devices must
be unique and explicitly indexed; `device` and `devices` cannot be configured
together. For a multi-device solver, `maxBatchBytes` is applied per concurrent
GPU: the execution policy scales its total batch allowance by the device count
and admits at least one problem per GPU, just as it admits one oversized
problem for a single-device solver.

AMD GPUs use PyTorch's ROCm build. PyTorch exposes HIP/ROCm devices through its
`torch.cuda` API, so the solver accepts `device: rocm`, `device: amd`, and
`device: hip` as aliases for the internal `cuda` device. `device: auto` selects
either a CUDA or ROCm accelerator whenever `torch.cuda.is_available()` is true.
The installed wheel must report a non-null `torch.version.hip`, and the GPU must
appear in AMD's ROCm compatibility matrix; the Python solver cannot add driver
or hardware support for an unsupported Radeon generation. For indexed
multi-GPU ROCm execution, use PyTorch's `cuda:0`, `cuda:1`, and so on device
names.

## Adaptive Torch solver

`AdaptiveTorchSBMBQMSolver` is registered as `adaptive_torch_sbm`. It retains
the sparse scenario/agent batching above and adds ideas used by the reference
PyTorch implementation and the bSB/dSB literature:

- selectable `discrete` and `ballistic` interaction activation;
- a pressure-slope schedule and optional heated dynamics;
- periodic per-agent Ising-energy monitoring, best-state retention, and early
  stopping after energies remain stable;
- exact float64 QUBO scoring followed by coordinate descent that preserves
  every declared one-hot group.

SB remains an unconstrained heuristic, so a low-energy terminal state is not
guaranteed to satisfy one-hot groups. Candidate selection shared by all BQM
solvers chooses the best feasible trajectory when one exists. Otherwise it
repairs every trajectory by categorical descent against the full QUBO and
rescoring; the adaptive solver can additionally polish an already feasible
answer.

```yaml
solver:
  type: adaptive_torch_sbm
  constructorParameters:
    device: auto
  solverParameters:
    steps: 1000
    runs: 16
    dtype: float32
    mode: discrete
    dt: 0.1
    pressure_slope: 0.01
    heated: false
    heat_coefficient: 0.06
    early_stopping: true
    sampling_period: 30
    convergence_threshold: 5
    track_best: true
    local_search_sweeps: 1
```

The dynamics follow Goto et al., *Science Advances* 7, eabe7953 (2021),
while the heating option follows Kanao and Goto, *Communications Physics* 5,
153 (2022). Energy-window convergence and multi-agent execution follow the
design of `bqth29/simulated-bifurcation-algorithm`.

## Spin-vector Langevin Torch solver

SVL `noise_chunk_size` (default 16) generates several time steps per trajectory
in one contiguous RNG draw. This reduces random-fill calls by approximately the
chunk size, with additional memory proportional to variables x run batch width
x noise chunk size. Zero-temperature/damping runs skip thermal RNG draws.
Changing this setting changes seeded trajectories compared with earlier releases;
for a fixed setting, problem ordering, device sharding and run batching preserve
stream identities. Exact CPU/GPU or cross-version reproducibility is not promised
by [PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).

SBM, adaptive SBM and SVL share an accumulator that scores feasible candidates
in float64 on the selected device and transfers
only the best candidates (including near ties for authoritative CPU rescoring).
If no feasible candidate exists, every candidate is still transferred and
repaired on CPU. Winners are retained incrementally across run chunks, so host
candidate retention no longer scales with the total number of runs. This avoids
changing the one-hot repair semantics to gain speed.

Categorical repair updates local fields directly from canonical CSC column
indices and values. Duplicate interactions are summed when the repair model is
built, so indexed updates visit each affected row once. This avoids constructing
a sparse submatrix and a dense full-length column for every variable flip.
The same repair code serves all BQM adapters; sweep order, tie tolerance,
feasibility preference and original float64 energy scoring remain unchanged.

`executionPolicy.prefetch: true` overlaps production of one bounded host batch
with solving the current batch. It is opt-in because it uses another host batch's
memory and calls the source iterator on a worker thread. Solver-specific working
memory estimates include integration/noise/scoring buffers in batch decisions;
these remain estimates, not a guarantee against allocator or library workspace
limits. Reduce `run_batch_size` if a single QUBO exceeds device capacity.

For GPU PCA, set `engine.riskStateGenerator.pcaGridProvider.backend` to
`{type: torch, device: cuda:0}`. The default remains `{type: numpy, device: auto}`.
PCA defaults to float32 on GPU and float64 on CPU, retaining both covariance
and observation-space paths. Set backend `dtype: float64` for validation;
`dtype: float32` also permits testing the reduced-precision path on CPU.
The surrounding risk-state API still consumes NumPy arrays, so fitted PCA outputs
return to the host once. Measure end-to-end timing before enabling this on small
windows, where transfer and decomposition synchronization can outweigh the gain.

Run the deterministic local benchmark with:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/benchmark_accelerator_pipeline.py --device cpu
```

Use `--device cuda:0` on a GPU host; timings synchronize that device. The script
checks PCA equivalence before timing and compares noise chunk sizes 1 and 16.
It is a component benchmark, not an estimate of whole-backtest speedup.

For synchronized before/after measurements of complete SBM and SVL solves:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/benchmark_torch_solvers.py --device cuda:0 \
  --variables 64 1024 8192 --steps 256 --runs 16 \
  --output /tmp/torch-solvers.json --trace-directory /tmp/torch-solver-traces
```

Run the same command on each revision, with separate output paths. JSON records
hardware/software, input identities, effective parameters, source hashes,
cold and warmed complete-solve timings after device initialization, peak allocated CUDA bytes, authoritative
energies and sample hashes. Profiling runs separately from latency measurements.
Use `--one-hot --problems 4` to include constrained selection/CPU repair,
`--run-batch-size` for bounded trajectory chunks, and
`--integrator weak_order_2 --dtype float64` to exercise SVL's second integrator.
`--device cpu` provides the portable path. These synthetic solver benchmarks
include packing, transfers and selection; they exclude risk-state generation.

SBM reuses its sparse multiplication output with `torch.addmm(..., beta=0,
out=...)`. SVL also reuses sine, cosine and force buffers for both integrators.
Scratch lifetime is one trajectory chunk; acceleration arrays remain separate
so the second SVL force evaluation cannot overwrite the first acceleration.
Arithmetic order, precision, random streams and candidate selection are preserved.
The existing conservative workspace estimate still covers these buffers.
See [GPU measurements](docs/benchmarks/torch_solver_gpu_20260909.md) for dynamics and
repair measurements and hardware limits.

`TorchSVLBQMSolver` is registered as `torch_svl`. It converts each QUBO to an
Ising problem and represents every spin by a planar rotor angle `theta_i`. For
annealing schedules `A(t)` and `B(t)`, the rotor Hamiltonian is

`H(theta,t) = -A(t) sum_i cos(theta_i) + B(t) H_Ising(sin(theta))`.

The solver evolves the corresponding underdamped Langevin system

```text
d theta_i = p_i / m dt
d p_i = -(partial H / partial theta_i + gamma p_i / m) dt
          + sqrt(2 gamma T) dW_i
```

and decodes the final rotors with `sign(sin(theta))`. It supports both the
Euler-Maruyama integrator and an explicit weak-order-2 integrator for additive
noise. Euler-Maruyama is useful as a simple reference; `weak_order_2` is the
preferred accuracy-oriented choice at a fixed step size. All returned samples
are evaluated using the original float64 QUBO energy, and the common candidate
selection/one-hot repair rules still apply.

Replicas are the stochastic trajectories called `runs` by the other Torch
solvers. They are advanced as one Torch tensor, while independent QUBOs are
packed into the same sparse block-diagonal interaction matrix. Use
`run_batch_size` to cap replica-state memory and `energy_chunk_size` to cap
exact final scoring memory.

```yaml
solver:
  type: torch_svl
  constructorParameters:
    devices: [cuda:0, cuda:1, cuda:2, cuda:3]
  solverParameters:
    steps: 4000
    runs: 64
    run_batch_size: 16
    integrator: weak_order_2
    dt: 0.01
    mass: 1.0
    damping: 0.1
    temperature: 0.001
    transverse_field_initial: 1.0
    transverse_field_final: 0.0
    problem_scale_initial: 0.0
    problem_scale_final: 1.0
    seed: 1
    dtype: float32
```

As with the SBM Torch solvers, select exactly one `device` or provide a unique
list of explicitly indexed `devices`. A multi-device solver divides each
ordered problem batch into contiguous shards, runs one shard per GPU
concurrently, and restores input order. Replica seeds include the stable QUBO
identity, so execution-policy batch boundaries, GPU sharding, and
`run_batch_size` do not change a seeded problem's trajectories. The same
`cuda:N` spelling is used for CUDA and PyTorch ROCm builds.

## Compact QUBO format

The solver consumes a sparse coordinate format. Variables are zero-based.
Repeated records are summed, and diagonal `q` records are folded into linear
biases.

```text
p qubo <number-of-variables>
o <offset>
l <variable> <linear-bias>
q <variable-u> <variable-v> <quadratic-bias>
```

It represents

`E(q) = offset + sum_i l_i q_i + sum_(i<j) Q_ij q_i q_j`.

The conversion uses `q_i=(s_i+1)/2` and the paper's convention
`H(s)=offset - 1/2 s^T J s - h^T s`. Linear fields are evaluated as couplings
to an ancillary spin fixed at `+1`, as prescribed for dSB in the paper, without
materializing that extra row in CPU/GPU memory.

## D-Wave BQM and CQM input

Install the Python bridge and export a model serialized by `BQM.to_file()` or
`CQM.to_file()`:

```bash
python -m pip install -r requirements.txt
python tools/export_dwave_qubo.py model.bqm model.qubo
./build/sbm_solve model.qubo --backend cpu --steps 10000 --runs 32
```

LP files containing a CQM are also accepted by the exporter. CQM input is
converted with `dimod.cqm_to_bqm`; `--penalty` controls its Lagrange multiplier.
The exporter writes variable labels and, for CQM input, the sample inverter to
`model.qubo.metadata.json`. D-Wave's CQM-to-BQM conversion requires linear
constraints and non-negative lower bounds for integer variables.

## Algorithm parameters

`a` increases linearly from zero to `a0`. At every step, all interactions are
computed from the same position-sign snapshot before position and momentum are
updated. `c0<=0` uses the paper's estimate based on the standard deviation of
the Ising matrix. `gamma=0` gives ordinary dSB; a positive value enables the
heating update. Multiple randomized runs are evaluated against the original
QUBO energy, and the best sample is returned.


## Execution ownership and profiling

`TorchExecution` owns device selection, locking, packing, sharding, trajectory
batching, and candidate collection for SBM and SVL. Dynamics remain in their
solver classes; adaptive SBM retains its algorithm-specific diagnostics.
`BQMResourcePlan` assigns contiguous shards once for both admission and execution.
`CandidateSelection` retains one repair model per problem across candidate chunks,
ranks feasible samples ahead of projected fallbacks, and scores source QUBO energy
in float64. This refactor does not change the integrators or their seed identities.

`maxBatchBytes` remains a per-worker workspace estimate. One oversized problem
per worker is admitted for progress. With prefetch, a CPU producer can retain the
next batch and one pending problem while the current batch solves. The optional
Python `memoryObserver` on `BatchBQMExecutionPolicy` receives immutable active,
producer, total, and peak coefficient-retention measurements. Callbacks are serial
but may run on either thread; return promptly. Counts are logical source-array
retention, not total resident memory or a host-memory limit.

Profile a complete local pipeline with:

```bash
PYTHONPATH=src python tools/profile_margin_pipeline.py path/to/gpu.yaml \
  --reference-config path/to/small-matching-cpu.yaml \
  --output /tmp/pipeline-profile.json --trace /tmp/pipeline-trace.json
```

Use matching small inputs to validate margins first, then omit the reference
when profiling representative workloads. Run each configuration in a separate
process. The JSON includes process peak RSS (Linux bytes, including imports and
any reference run), coefficient retention, CUDA peak allocated/reserved memory,
synchronized wall latency, and source margin difference. The optional Chrome
trace captures Torch operators, copies, and allocation events. Profiling and GPU
synchronization add overhead; do not compare traced timings to untraced timings.

`MarginEngineTimings.measurementVersion == 2` changes the interpretation of
`marginCalculationSeconds`: it is inclusive wall latency of the calculation,
including waiting for generation. `riskStateGenerationSeconds` measures producer
work, which can overlap solving, so stage durations are not additive. Production
measurements do not synchronize GPU devices. Backtest manifests record version 2
and new experiment fingerprints separate these measurements from old checkpoints.

The opt-in resident returns path described below is available for evaluation.
Adopting it as a default remains gated on GPU traces and end-to-end measurements.
Host-array interfaces, lazy generation, greedy comparisons, and CPU references
remain available. CPU-only measurements cannot establish a GPU speedup.

Yahoo acquisition uses per-ticker history with exception propagation. If Yahoo
reports missing prices, one wider historical request can confirm a valid empty
calendar interval; only requested dates are returned. Persistent malformed or
missing responses still raise. This avoids both retrying successful weekends
indefinitely and storing suppressed transport failures as acquired coverage.

## Native preparation and benchmark conventions

`sbm::QUBOPreparation` owns CSR topology construction and coefficient rebinding.
Both `to_ising` and the C adapter use this implementation. Its byte budget bounds
retained topology arrays; zero disables reuse. Budget changes take effect on the
next preparation. Explicit zero coefficients remain in the topology, and parallel
edges retain their individual energy contributions. Callers serialize ownership
of a preparation object when changing its budget.

The Python native solver owns a preparation context and serializes solves. New
C functions create/destroy the context and accept it at the prepared batch entry
point; existing C entry points remain exported, and Python still supports older
libraries without the context API. Rebuild the library to enable owned reuse.
Float32 dynamics and float64 scoring of the original Python QUBO are unchanged.

`sbm::maxcut::PreparedSearch` shares adjacency, gains, and exact/greedy/annealing
search across benchmarks. Its graph reference must outlive the search object.
Returned partitions are rescored against the graph. Single-instance benchmark
configuration labels exclude preparation from solve timing; batch and large
benchmark labels include it. Annealing uses a bounded flip log for best-state
updates rather than copying the partition after every improvement.

The research exporter and application share only the weighted eigensystem kernel.
The exporter retains its supplied StandardScaler normalization, sign convention,
and observation-space threshold; the application retains EW standardization and
canonical eigenvector signs. Their compatibility-penalty construction also remains
intentional: directed filtering in the exporter, symmetric nomination union in the
application. Shared numerical code does not imply interchangeable risk models.

## Opt-in resident returns execution

Recommendation 24 now has an experimental implementation. Enable it explicitly:

```yaml
engine:
  numericalExecution: {type: torch, device: cuda:0}
  # Keep the existing risk model and calculator settings. For a BQM calculator:
  marginCalculator:
    type: bqm
    solver:
      type: torch_sbm
      constructorParameters: {device: cuda:0}
```

This supports ordinary and correlated returns grids, greedy calculation, paired
comparisons, and the Torch SBM, adaptive SBM, and SVL solvers. PCA and solving must
use the same single device. `device: auto` follows the configured BQM solver, or
selects the available Torch device for greedy calculation. `device: cpu` exercises
the same tensor path without an accelerator. Omit `numericalExecution` to retain
the existing host pipeline, including multi-GPU scheduling and other risk families.

`TorchReturnsExecution` owns aligned input preparation, resident PCA at the selected precision,
conditioning, bounded correlation blocks, QUBO coefficients, and their lifetime
through solving. `TorchPCABackend.fitResident` retains fitted tensors; its existing
`fit` method explicitly materializes a host `PCAFit` for host consumers. Core
portfolio, risk-state, and QUBO domain types remain host types. The resident mode
fits once per calculation and releases its intermediates afterwards; it does not
use the host PCA-grid cache or its backend selection. The shared input preparation
preserves date alignment, EW standardization, and the historical cutoff.

The solver constructs its sparse Ising matrix directly from resident coefficients,
using the same trajectory and candidate-selection code as the host path. The
current one-hot topology is reused on device. Immutable host QUBO snapshots remain
necessary for the existing coefficient-derived seed identity, authoritative
float64 scoring, and deterministic categorical repair. Control metadata and
candidates also cross the host/device boundary. With float32 risk numerics, scoring uploads the authoritative float64 source
coefficients; widening rounded dynamics coefficients would lose source-energy ties.
This removes the fitted PCA round-trip; it does not eliminate all transfers or
synchronizations. It does not add asynchronous copy/double-buffer scheduling.

Risk numerics accept `numericalExecution.dtype: auto` (default: float32 on GPU,
float64 on CPU), `float32`, or `float64`; the solver retains its independently
configured dynamics precision. Means, scales, scenario bins, correlation blocks,
and resident QUBO dynamics coefficients follow the risk dtype. Host source QUBOs,
candidate energy scoring and repair retain float64. The effective risk `dtype` is
reported in numerical diagnostics. Model fingerprint version 4 invalidates older
backtest checkpoints. Reduced precision can change bin membership, neighbor order
for near ties, and coefficients; it is not bitwise equivalent to float64. Use
float64 for sensitive or ill-conditioned portfolios and compare the resulting
margins before adopting a precision setting. See
[float32 risk measurements](docs/benchmarks/gpu_risk_float32_20260909.md).
NumPy and Torch reductions can differ in low bits. Because seeds depend on the
resulting coefficients, heuristic BQM samples and margins can differ between
execution modes even when their risk models agree numerically. Seeds and ordering
remain independent of batch size within a mode. Compare paired greedy margins,
source energies, and benchmark outcomes before selecting a mode for an experiment.

`batchSize`, `maxBatchBytes`, and `memoryMultiplier` govern resident BQM batches.
Admission includes the fitted tensor arrays and resident coefficient estimates;
one oversized problem remains admissible for progress. One pending problem and
conditioning/correlation workspace can exist in addition to the admitted batch.
These are estimates, not a hard GPU-memory limit. CPU `prefetch` is not used in
resident mode: generation and solving execute on the same device. The returned
`MarginReport.numericalDiagnostics` records fit bytes, cumulative host source
snapshot bytes, scenario/fallback counts, peak admitted batch size, and estimated
working memory. `fittedHostMaterializationBytes` includes the small eigenvalue
vector used to build scenario axes; residuals and loading matrices stay resident.
Snapshot bytes are not measured PCIe traffic. The optional memory
observer continues to report host source-array retention; CUDA allocator peaks
and transfer traces are measured by the profiling tools.

Run a reproducible comparison matrix on the target GPU:

```bash
PYTHONPATH=src python tools/benchmark_resident_pipeline.py \
  --device cuda:0 --assets 16,256,1000 --windows 30,120 \
  --steps 256 --runs 16 --repeats 3 \
  --output-directory /tmp/resident-benchmark
```

Add `--correlated` for compatibility penalties or `--trace` for annotated Torch
traces. Every YAML, local fixture, and JSON measurement is archived. Each
measurement runs in a fresh process and includes a CPU reference; paired greedy
margins must agree within numerical tolerance. BQM margin differences are reported
because the solvers are heuristic. Timing ratios are observations for that machine
and workload, not automatic configuration recommendations. Small-window CPU runs
can favor the host path. This implementation has CPU-reference coverage and an
optional GPU regression test, but no CUDA/ROCm hardware validation or measured
GPU benefit was possible in the CPU-only development environment.

Equal-strength correlation nominations now use the lowest canonical asset index
in both NumPy and Torch, including the partitioned/top-k path above 512 assets.
Nearest-residual distance ties likewise use the earliest chronological observation.
This makes degenerate histories reproducible across these algorithms; portfolios
with tied neighbors may legitimately differ from the old unspecified tie order.
Numerical model version 3 introduced this tie rule; version 4 additionally
identifies the GPU float32 default. The fingerprint prevents resume from mixing
checkpoints made under earlier numerical policies.
Custom visitors and execution policies require the host path and are rejected by
the resident configuration rather than being silently bypassed.


## Feasibility-preserving categorical solver

`torch_categorical` is an optional alternative for complete one-hot portfolios.
It uses graph-colored categorical heat-bath annealing, not SBM/SVL equations.
The solver initializes one selected category per group and updates only mutually
noninteracting groups together. Every discrete state is feasible; the final
zero-temperature sweeps improve valid states without projecting or repairing them.

```yaml
solver:
  type: torch_categorical
  constructorParameters: {device: cuda:0}
  solverParameters:
    steps: 64
    runs: 16
    seed: 1
    dtype: float32
    run_batch_size: 16
    temperature_start: 1.0
    temperature_end: 0.01
    greedy_sweeps: 4
    noise_chunk_size: 16
    energy_chunk_size: 1000000
```

`steps` counts full color sweeps, so it is not comparable to an SBM/SVL integration
step. Temperatures are absolute objective-energy units; rescaling exposures can
require retuning them. `greedy_sweeps: 0` disables the additional zero-temperature
updates; outputs still remain feasible. Seed streams are independent of trajectory
chunks and problem batches for a fixed noise chunk size. Changing noise chunk size
can change seeded trajectories. CPU and CUDA/ROCm use the same adapter; standard
explicit multi-device configuration is inherited, but multi-GPU hardware was not
available for this validation.

Run the local synthetic example with
`PYTHONPATH=src python -m margin_engine config/categorical.example.yaml`.
It selects the categorical solver with zero one-hot penalty and uses CUDA when
available, otherwise CPU.

Declared groups must cover every variable and may be ragged/noncontiguous.
Unsupported partial coverage is rejected rather than silently changing the problem.
Diagonal terms are folded into linear costs. Within-group off-diagonal terms are
identically zero for valid candidates; common linear shifts (including the one-hot
penalty) are constant. These are removed from dynamics, while original-QUBO energy
and offsets remain authoritative for float64 final scoring. Thus `lambdaOneHot: 0`
is permissible for this solver; existing positive penalties do not enforce its
feasibility and need not be increased. Coefficient-based source seeds may still
change when the encoding penalty changes.

Trajectories execute in parallel; independent problems execute sequentially per
admitted device shard. Resident numerical execution is supported, but compiles
categorical coefficients from the existing host source snapshot and uploads them.
It does not reuse the resident Ising matrix. Memory estimates include padded ragged
groups, sparse workspace, RNG chunks and original float64 scoring. Large groups or
highly connected group graphs can limit memory efficiency and color parallelism.

Existing SBM/SVL defaults and candidate repair behavior remain unchanged. Select
this solver explicitly and compare quality/latency on the intended portfolio. See
[feasibility, penalty, and solver measurements](docs/benchmarks/one_hot_feasibility_20260909.md).

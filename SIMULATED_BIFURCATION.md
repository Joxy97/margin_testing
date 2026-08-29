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
  both independent QUBOs and randomized trajectories on CPU or CUDA.
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

AMD GPUs use PyTorch's ROCm build. PyTorch exposes HIP/ROCm devices through its
`torch.cuda` API, so the solver accepts `device: rocm`, `device: amd`, and
`device: hip` as aliases for the internal `cuda` device. `device: auto` selects
either a CUDA or ROCm accelerator whenever `torch.cuda.is_available()` is true.
The installed wheel must report a non-null `torch.version.hip`, and the GPU must
appear in AMD's ROCm compatibility matrix; the Python solver cannot add driver
or hardware support for an unsupported Radeon generation.

## Adaptive Torch solver

`AdaptiveTorchSBMBQMSolver` is registered as `adaptive_torch_sbm`. It retains
the sparse scenario/agent batching above and adds ideas used by the reference
PyTorch implementation and the bSB/dSB literature:

- selectable `discrete` and `ballistic` interaction activation;
- a pressure-slope schedule and optional heated dynamics;
- periodic per-agent Ising-energy monitoring, best-state retention, and early
  stopping after energies remain stable;
- exact float64 QUBO scoring followed by coordinate descent that repairs and
  preserves every declared one-hot group.

The last step is application-specific: SB remains an unconstrained heuristic,
so a low-energy terminal state is not guaranteed to satisfy one-hot groups.
The repair/polish step makes that constraint explicit without adding it to the
general SBM dynamics.

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

# Torch SBM/SVL GPU optimization — 9 September 2026

Follow-up: [GPU PCA/risk float32 measurements and remaining bottlenecks](gpu_risk_float32_20260909.md).

Reusable dynamics buffers improve complete unconstrained solves by 1.07–1.17×
on the tested RTX A4000. Updating repair fields directly from sparse columns
improves the tested constrained batches by 2.04–6.10×. All 16 before/after
measurement pairs returned identical samples and original QUBO energies.

The solver layer now reuses sparse multiplication outputs in SBM and reuses
sine/cosine/force storage in SVL. The shared `CandidateSelection` layer avoids
`getcol().toarray()` during categorical repair: canonical CSC indices identify
the affected fields directly. This latter improvement benefits other BQM
adapters too. It preserves duplicate aggregation, group order, tie tolerance,
feasibility preference, and scoring. No configuration, dependencies, ABI,
precision, seed policy or checkpoint identity changed.

## Measurements

Hardware: NVIDIA RTX A4000, 16,376 MiB, initially idle. Software: Python 3.12.14,
PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.5.3, Linux x86-64. Torch, OpenMP and
OpenBLAS use one CPU thread. Baseline commit:
`74995a544851e477d37fe34779fb83865b2c85a1`.

Times below are medians of five complete solves, after an initial cold solve
and two additional warmups. CUDA synchronization brackets each measurement.
The cold solve starts after device initialization; process/context startup is
excluded from these measurements.
Packing, transfers, dynamics, selection, repair and output are included.
Profiling is separate. Fixtures use seed 13 and eight sampled sparse terms per
variable before canonical aggregation; constrained fixtures add groups of four
with one-hot penalties. Sizes are variables per problem.

| Workload | Solver | Variables | Before (ms) | After (ms) | Speedup |
|---|---|---:|---:|---:|---:|
| Unconstrained | SBM | 64 | 29.486 | 25.720 | 1.146× |
| Unconstrained | SVL | 64 | 42.318 | 36.276 | 1.167× |
| Unconstrained | SBM | 1,024 | 32.101 | 28.219 | 1.138× |
| Unconstrained | SVL | 1,024 | 44.865 | 38.725 | 1.159× |
| Unconstrained | SBM | 8,192 | 57.105 | 53.328 | 1.071× |
| Unconstrained | SVL | 8,192 | 70.753 | 64.241 | 1.101× |
| One-hot batch | SBM | 64 | 241.085 | 104.170 | 2.314× |
| One-hot batch | SVL | 64 | 261.733 | 128.063 | 2.044× |
| One-hot batch | SBM | 1,024 | 5,388.572 | 883.516 | 6.099× |
| One-hot batch | SVL | 1,024 | 4,465.647 | 766.102 | 5.829× |

Unconstrained cases use one problem, float32, 256 steps and 16 runs. One-hot
cases use four problems, float32, 257 steps, five runs and run chunks of two.
SVL uses Euler–Maruyama here. Each solver retains its own original dynamics
parameters across revisions.

Float64 cases with nine runs in chunks of four, 257 steps and SVL's
`weak_order_2` integrator improved by 1.155–1.165× at 64 and 1,024 variables.
At 8,192 variables and 1,000 steps, SBM improved from 136.738 to 122.568 ms
(1.116×), and SVL from 187.285 to 168.281 ms (1.113×). These longer runs used
three measurements after the cold solve and one additional warmup.

The initial buffer-only change did little for repair-heavy batches: an
alternating-order check measured just 1.001× for SBM and 1.008× for SVL at
1,024 variables. A subsequent CPU profile attributed 8.948 of 9.094 profiled
seconds to repair, including 61,440 dense column conversions. Profiling overhead
inflates that wall time; it diagnoses the bottleneck rather than providing a
latency comparison. Direct CSC updates remove those conversions.

Separate CUDA traces at 1,024 variables/256 steps show:

| Solver | Kernels before → after | Sum of kernel durations before → after |
|---|---:|---:|
| SBM | 4,158 → 3,902 | 8.185 → 7.885 ms |
| SVL | 5,181 → 4,925 | 10.652 → 10.359 ms |

Much of the unconstrained latency benefit comes from reduced host allocation
and dispatch overhead. Kernel-duration sums exclude gaps and transfers and
must not be compared directly with complete-solve latency. Peak CUDA allocation
is recorded in the JSON; these changes do not promise a lower total peak.

## Reproduce and validate

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/benchmark_torch_solvers.py --device cuda:0 \
  --variables 64 1024 8192 --steps 256 --runs 16 --output /tmp/solvers.json

PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/benchmark_torch_solvers.py --device cuda:0 \
  --variables 64 1024 --problems 4 --one-hot --steps 257 \
  --runs 5 --run-batch-size 2 --output /tmp/solvers-onehot.json
```

Run each command on both revisions. Add `--trace-directory /tmp/solver-traces`
for a separate profiling pass. [Raw measurements](torch_solver_gpu_20260909.json)
contain all durations, effective parameters, input identities, source hashes,
sample hashes, energies, memory measurements and the other reproduction arguments.
The original profiler archive was fetched to
`/tmp/margin-gpu-baseline/margin-gpu-traces.tar.gz` locally.

Validation:

- Local full suite: `PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/margin-gpu-matplotlib .venv/bin/python -m unittest discover -s tests -p 'test_*.py'` — 218 tests, successful, four skips (three CUDA checks and opt-in network integration).
- GPU host: `PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest tests.test_candidate_selection tests.test_torch_dynamics tests.test_bqm_solver tests.test_accelerator_pipeline tests.test_device_resident_pipeline` — 68 tests, successful, three native-library skips.
- Dynamics tests compare continuous float64 states against independent dense equations on CPU and CUDA, including both SVL integrators, noise/no-noise, SBM heating, empty sparse matrices, and one/19 steps.
- Repair tests compare all 256 binary samples on each of two eight-variable fixtures against dense descent, covering noncontiguous groups, a free variable, duplicate/reversed interactions, diagonals, cancellation and ties.
- Benchmark smoke checks cover CPU execution, profiling, one variable, one step, one run and uneven run chunks. Final paired benchmark samples and energies match exactly in float32 and float64, including 1,000-step runs.

These are synthetic solver measurements on one NVIDIA GPU, not whole-backtest
speedups or guarantees for other shapes/hardware. Repair still runs on CPU.
ROCm and multiple GPUs were unavailable. Native C++/CUDA/FPGA implementations
were unchanged; native build/CTest was not rerun. Existing conservative memory
estimates remain in place.

## References

- [PyTorch `addmm`](https://docs.pytorch.org/docs/main/generated/torch.addmm.html): sparse inputs, reusable outputs and `beta=0` semantics; exercised here on the installed CPU and CUDA builds.
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065). The optimize-for-gpu skill supplied the profiling, correctness and synchronized-benchmark workflow; bibliographic details checked against the latest arXiv record.

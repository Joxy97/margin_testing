# Transverse Route integration and GPU optimization

Date: 2026-09-09.

## Implementation and provenance

`torch_transverse_route` lives in the BQM solver adapter layer and implements
the angular equations and integration schedule from
[`Joxy97/solver_testing`, revision `f71e926c2d1fee5fb3ebe6fd872161cccde738d0`](https://github.com/Joxy97/solver_testing/blob/f71e926c2d1fee5fb3ebe6fd872161cccde738d0/solvers/transverse_route.py).
The upstream solver and minimal helper modules were downloaded into a temporary
directory for comparisons; the application does not depend on that checkout.
The upstream repository did not expose a license file through GitHub's license
endpoint at retrieval time. This adapter implements the published equations in
the existing application architecture; upstream files are not vendored here.

The factory exports `TorchTransverseRouteBQMSolver`. YAML uses
`type: torch_transverse_route`, constructor `device` or `devices`, and per-call
`solverParameters`. The existing typed solver configuration and strict mapping
parser already support this composition; the resident solver allowlist was
extended. See `config/transverse_route.example.yaml` and the transverse-route
section of `SIMULATED_BIFURCATION.md` for all supported controls.

The implementation preserves Euler/Heun integration, angle wrapping, the optional
route and cubic transverse channels, monotone confinement, and initial/periodic/
final candidate checkpoints. Host float64 canonicalization aggregates reversed,
duplicate and diagonal QUBO terms. The largest absolute Ising row bound sets the
scale before converting dynamics to float32 or float64. Positive Ising `J,h` are
used here; the shared SBM converter's negative force convention is inverted.

The integration with this application deliberately uses its own stable
per-problem/per-run seeds and feasible-first candidate selection. Every distinct
infeasible candidate is repaired when no feasible candidate is available.
Original QUBO energy and deterministic tie selection remain authoritative.
Upstream dashboard snapshots are outside the `BQMOptimizationResult` interface.

## Changes supported by profiling

The original full upstream solver was profiled before the optimization was
measured. A local CPU run at 1,024 variables, 16 trajectories and 256 steps spent
about 70% of profiled self CPU time in sparse matrix multiplication. On the GPU
host, the original implementation issued 7,986 CUDA kernel launches for that
workload; host launch calls accounted for about 24% of self CPU time and
`aten::empty` for about 12%. The full upstream solve took 0.109 seconds in a
separate synchronized timing. This diagnostic uses the original upstream global
RNG, so its energy is not a matched-seed quality comparison.

See `transverse_route_upstream_profile_20260909.txt` for the original profile.
That profile records CPU dispatch/allocation events for a CUDA workload; its
tables are not a GPU kernel-time breakdown.

The retained optimizations are:

- CSR sparse matrix multiplication with all active route channels packed into
  one operation, using reusable feature and result buffers.
- Reusable trigonometric, slope and integration buffers, with in-place updates.
- CUDA graph replay of bounded integration blocks, including Heun support;
  capture cost remains part of each measured solve.
- Bounded checkpoint accumulation, reducing candidate-processing round trips.
- Exact duplicate-row removal within one-hot checkpoint batches before costly
  deterministic CPU repair. Distinct samples are retained.
- Independent scenario shards on the existing concurrent multi-device executor.

No new accelerator dependency, custom CUDA kernel, native ABI or data migration
is required. Torch remains optional until the solver is used. Sparse CPU
execution remains available.

## Measurement method

GPU host: eight NVIDIA GeForce RTX 2080 Ti GPUs, 11,264 MiB reported memory each;
Python 3.12.14, PyTorch 2.11.0+cu128, CUDA 12.8. Only the replacement host was used
for completed GPU benchmarks. The original GTX 1660 SUPER instance became
unavailable after initial hardware discovery.

Fixtures come from `tools/benchmark_torch_solvers.py`: deterministic sparse
QUBOs with eight proposed edges per variable. One-hot fixtures add disjoint
groups of four variables and their penalties. They exercise the solver workload;
they are not full portfolio backtests or evidence of margin-model calibration.

`tools/benchmark_transverse_route.py --upstream-root <checkout>` calls the pinned
upstream `_flow_rhs` and allocating Euler/Heun updates for its matched reference,
including upstream COO multiplication. Both implementations share the same
QUBO preparation, initial angles, candidate checkpoints and source-energy
selection. The reference disables deduplication and CUDA graphs; optimized
variants enable deduplication for one-hot inputs. This isolates implementation
performance without changing the discrete search or repair contract. It does
not claim byte-for-byte equivalence to the upstream full `solve` API, which uses
global RNG state and has no application-specific one-hot repair.

Timings include preparation, transfers, integration, candidate scoring, repair,
result construction and graph capture. Every participating device is synchronized
before and after each solve. Cold calls are recorded separately; reported medians
use three measured repetitions after the cold call and the configured warmups.
No concurrent timing jobs were run on the GPU host. Memory fields report Torch
peak allocated bytes, not reserved memory, CUDA context memory or whole-process
host RSS. Raw JSON records contain arguments, versions, source hashes, individual
times, energies and sample hashes.

The CPU and multi-GPU records were measured before the final one-hot-only
deduplication change. Their unconstrained path is unchanged; source hashes retain
this distinction. Local CPU and remote GPU builds also have different Torch
versions and RNG implementations. Prefer same-host reference/optimized ratios
over treating CPU/GPU ratios as a controlled hardware comparison.

## Results

Final single-GPU medians, 1,000 steps and 16 trajectories, seconds per complete
solve. All three variants returned identical samples and original-QUBO energies
at every listed size.

| Variables | Matched upstream flow | Optimized eager | Optimized graph | Reference / graph |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.2981 | 0.1776 | 0.0580 | 5.14x |
| 1,024 | 0.4322 | 0.2645 | 0.0634 | 6.82x |
| 8,192 | 0.4871 | 0.3189 | 0.1528 | 3.19x |

The final graph variant peaked at approximately 9.8, 34.5 and 59.3 MiB of
Torch-allocated device memory respectively. Full records:
`transverse_route_gpu_final_20260909.json`. The initial measurement pass is
retained in `transverse_route_gpu_20260909.json`.

Eight-problem batches, 1,000 steps, 16 trajectories per problem. Winners were
identical across the matched reference, eager, graph and eight-device variants.

| Variables per problem | One GPU, eager | One GPU, graph | Eight GPUs, graph | One / eight GPUs, graph |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.2511 s | 0.1721 s | 0.3481 s | 0.49x |
| 8,192 | 1.0489 s | 1.0763 s | 0.6720 s | 1.60x |

Thus eight GPUs improved the larger batch by about 1.6x, but made the smaller
batch about twice as slow. Graphs were slightly slower than eager execution for
the large batch on a single GPU. Records:
`transverse_route_multi_gpu_20260909.json`.

One-hot fixtures, 256 steps and 16 trajectories, including all required CPU
repair. Final deduplicated results retained identical winning samples and
energies to the matched reference.

| Variables | Matched upstream flow | Optimized eager | Optimized graph | Reference / eager |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.3163 s | 0.2027 s | 0.2179 s | 1.56x |
| 1,024 | 9.8919 s | 7.8950 s | 7.9309 s | 1.25x |

Before candidate deduplication, the 1,024-variable graph solve took 9.8021 s;
the final median is 7.9309 s. An exploratory single timing was faster, but the
repeated medians above are the reported result. This is a useful reduction in
repair work, not the 3-7x gain observed for unconstrained QUBOs. Records:
`transverse_route_one_hot_gpu_20260909.json` and
`transverse_route_one_hot_dedup_gpu_20260909.json`.

Local CPU medians for the same unconstrained sizes were 0.1281, 0.2822 and
2.1094 s with the optimized adapter, versus 0.1916, 0.9924 and 5.3776 s with the
matched reference. Records: `transverse_route_cpu_20260909.json`.

## Correctness

- Full local Python suite: 240 tests, OK, six skipped by existing availability
  and integration guards. Command:
  `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`.
- Focused remote suite: all nine tests passed, including CUDA graph/eager
  equality and all eight GPU shards. Command:
  `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /venv/main/bin/python -m unittest tests.test_torch_transverse_route -v`.
- Tests cover independently calculated flow equations, both integrators,
  optional channels, exact QUBO energy normalization, diagonal/duplicate terms,
  zero interactions, bounded run/checkpoint batches, global RNG isolation,
  immutable inputs, deterministic repair, duplicate removal, invalid parameters,
  factory construction and both host/resident YAML pipelines.
- Against upstream CPU trajectories from identical initial angles, the GPU
  comparison covers 64 steps, Euler/Heun, disabled/enabled route channels,
  gamma zero/nonzero, and float32/float64. Maximum circular angle errors observed
  were `1.10e-5` radians in float32 and `2.49e-14` in float64; acceptance bounds
  are `3e-5` and `1e-12` respectively.
- The complete synthetic ten-asset example ran on CPU and CUDA. Margins were
  `0.0004411535671456527` and `0.00044115356714565684`; paired greedy margins were
  `0.049112507090718745` and `0.04911250709071875`. These are smoke-test outputs,
  not proof that the heuristic finds the global optimum.

The native C++/CUDA build was not run because no native source or ABI changed.
Ordinary network-dependent integration tests remain guarded. Long trajectories
can amplify rounding differences, and a fixed numeric seed does not generate
the same initial random stream on CPU and CUDA. No bitwise cross-device or
global optimality guarantee is added.

## Reproduction

From the repository root, use an upstream checkout at the revision above:

```bash
PYTHONPATH=src python tools/benchmark_transverse_route.py \
  --device cuda:0 --variables 64 1024 8192 --steps 1000 --runs 16 \
  --upstream-root /path/to/solver_testing --output /tmp/transverse_gpu.json

PYTHONPATH=src python tools/benchmark_transverse_route.py \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --variables 1024 8192 --problems 8 --steps 1000 --runs 16 \
  --upstream-root /path/to/solver_testing --output /tmp/transverse_multi_gpu.json

PYTHONPATH=src python tools/benchmark_transverse_route.py \
  --device cuda:0 --variables 64 1024 --one-hot --steps 256 --runs 16 \
  --warmups 0 --upstream-root /path/to/solver_testing \
  --output /tmp/transverse_one_hot_gpu.json

PYTHONPATH=src python -m margin_engine config/transverse_route.example.yaml
```

Omit `--upstream-root` to benchmark only the installed adapter. Add
`--profile-directory /tmp/trf-traces` for Torch traces. Parameters and commands
contain no SSH credentials. The remote benchmark copy is isolated beneath
`/workspace/margin-trf-20260909`; measurement artifacts are fetched locally.

## Limits and tuning

Use one GPU for small batches. Eight-device execution shards independent QUBOs;
it does not distribute a single QUBO or its trajectories across devices. Capture,
thread scheduling, packing and CPU candidate work limit scaling. In the measured
eight-problem batch, 1,024-variable problems were faster on one GPU, whereas
8,192-variable problems benefited from eight devices.

One-hot CPU repair can dominate total time. Deduplication helps when checkpoints
repeat binary samples, but it cannot remove the work for distinct infeasible
candidates. Graph replay can also cost more than it saves on small, short solves;
`cuda_graph: false` retains the optimized eager path. Check the measured workload
instead of assuming graphs or additional GPUs always improve it.

`candidate_batch_size` controls buffered candidates and scoring/repair workspace;
`run_batch_size` controls resident trajectories. Increasing the former can remove
more duplicates but increases memory. Dense matrix selection is bounded and
explicit dense scenario batches execute sequentially within each device. The
resource policy still uses estimates and admits an oversized single problem to
make progress; allocation success is not guaranteed by an estimate.

Resident execution deliberately rebuilds normalized dynamics from authoritative
host snapshots. This preserves canonicalization and scoring semantics, but
introduces coefficient transfers and is not a fully resident optimization.

## Methodology attribution

The requested `optimize-for-gpu` skill guided profiling, bounded GPU execution,
numerical comparison and synchronized measurements. Its required attribution is
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner
(2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
Metadata was checked against the current arXiv record, revised September 2, 2026.
This credits the methodology skill, not the transverse-route algorithm, whose
source is linked above.

# GPU risk precision — 9 September 2026

Follow-up: [one-hot feasibility measurements and categorical solver](one_hot_feasibility_20260909.md).

GPU PCA and resident risk calculations now default to float32. CPU risk calculations
remain float64. Torch solver dynamics already defaulted to float32. Original-QUBO
candidate scoring and categorical repair remain float64.

On the NVIDIA RTX A4000, PCA fitting is 1.69–1.89× faster and retained PCA tensors
use exactly half the memory. Complete fresh-engine margin calculations are within
about 2% of the float64 runtime on these fixtures: this is not an established
end-to-end speedup. The largest next opportunity is CPU market-data assembly;
GPU launch overhead remains a separate target once acquisition is amortized.
With the same engine and acquisition caches reused, measured complete calculations
are 1.6–4.3% faster; this is a modest throughput improvement.

## Configuration and numerical contract

Both `engine.riskStateGenerator.pcaGridProvider.backend.dtype` and
`engine.numericalExecution.dtype` accept `auto`, `float32`, or `float64`.
`auto` selects float32 on CUDA/ROCm and float64 on CPU. NumPy PCA accepts
`auto` or `float64`. The resident block owns its own PCA backend; the host
PCA provider's precision setting does not override it.

```yaml
engine:
  numericalExecution: {type: torch, device: cuda:0, dtype: auto}
```

Set `dtype: float64` in that block to validate against the previous precision.
The solver's independent `solverParameters.dtype` remains unchanged.
Float32 propagates through PCA, scenario bins, means/scales, correlation penalties,
portfolio weighting, and resident dynamics coefficients. Input date alignment and
EW standardization remain on the host. Host source QUBOs are float64; scoring
uploads those authoritative coefficients when resident dynamics coefficients are
float32. Merely widening already-rounded dynamics coefficients could misrank near
ties. A dedicated test covers that failure.

Risk precision can change bin membership, near-tied neighbors, PCA signs for
near-tied pivots, source coefficients, and coefficient-derived random seeds.
Backtest numerical model version **4** invalidates older checkpoints. Existing
YAML remains valid; there is no new dependency or native ABI change. Numerical
diagnostics report the effective risk dtype. Memory admission remains conservative;
float64 scoring and integer indices prevent a blanket 50% reduction in total memory.

## Measurements

Hardware: NVIDIA RTX A4000, 16,376 MiB, initially idle. Python 3.12.14,
PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.5.3. Torch/OpenMP/OpenBLAS use one CPU
thread; float32 matmul precision is `highest`. No TF32 setting was enabled.

Both precision variants use the same current implementation and deterministic
input. Each has two warmups and five measurements bracketed by CUDA synchronization.
Precision order alternates for complete calculations. Complete timings include
engine construction, local CSV acquisition, risk generation, three scenarios,
packing, transfers, float32 SBM dynamics (256 steps, eight runs), float64 selection,
repair, and output. Batch size is two. The history window is 60 observations.
PCA measurements include host-to-device conversion and fitting, retaining the
result on GPU. Process startup and profiler overhead are excluded from the tables.

| Stage | Assets | Float64 ms | Float32 ms | Ratio (64 / 32) |
|---|---:|---:|---:|---:|
| PCA, two components | 64 | 2.954 | 1.746 | 1.69× |
| PCA, two components | 1,024 | 3.416 | 1.807 | 1.89× |
| Complete uncorrelated margin | 64 | 95.219 | 96.862 | 0.98× |
| Complete correlated margin | 64 | 121.329 | 123.103 | 0.99× |
| Complete uncorrelated margin | 1,024 | 624.669 | 615.441 | 1.01× |
| Complete correlated margin | 1,024 | 637.066 | 640.991 | 0.99× |

PCA fit storage falls from 34,272 to 17,136 bytes at 64 assets and from 533,472
to 266,736 bytes at 1,024 assets. For the 1,024-asset correlated complete calculation,
measured peak CUDA allocation falls from 47,650,816 to 38,036,992 bytes (20.2%).
Other complete cases show smaller reductions, recorded in the raw results.

The paired greedy margin differs from the NumPy CPU reference by at most
**0.000005211** on these complete fixtures (about 0.000006% relative).
PCA eigenspaces and eigenvalues pass explicit tolerance checks; individual factor
coordinates have maximum absolute error up to 0.000453 on the standalone fixtures.
Unit tests additionally compare every PCA output on small covariance and
observation-space fixtures.

Heuristic BQM outcomes are **not equivalent across precision**:

| Assets | Correlated | Float64 risk margin | Float32 risk margin | Change |
|---:|:---:|---:|---:|---:|
| 64 | No | 11.788052 | 11.757255 | −0.26% |
| 64 | Yes | 1.867463 | 1.618661 | −13.32% |
| 1,024 | No | 164.278519 | 164.561689 | +0.17% |
| 1,024 | Yes | 19.172444 | 19.794656 | +3.25% |

These are single-seed heuristic outcomes, not a statistical quality comparison.
Float64 source scoring preserves ranking for the candidates actually generated;
it does not make the candidate sets or risk coefficients identical. Use the
float64 override when matching existing experiment results matters, and assess
multiple seeds/margins on actual portfolios before interpreting a lower margin
as an accuracy improvement.

## Repeated calculations with one engine

A second run uses `--reuse-engine` with otherwise identical arguments, two warmups,
five alternating measurements, and cached acquisition on the same input/date.
This measures repeated calculation throughput, not a rolling-date backtest.

| Assets | Correlated | Float64 ms | Float32 ms | Ratio (64 / 32) |
|---:|:---:|---:|---:|---:|
| 64 | No | 64.138 | 63.067 | 1.017× |
| 64 | Yes | 89.260 | 87.776 | 1.017× |
| 1,024 | No | 90.044 | 88.616 | 1.016× |
| 1,024 | Yes | 106.165 | 101.811 | 1.043× |

For 1,024 correlated assets, median acquisition time is 1.69 ms in float32;
producer risk-generation time is 9.74 ms versus 11.09 ms in float64. Producer
work overlaps the inclusive margin-calculation timing and must not be added to it.
The BQM and greedy outcomes match the corresponding fresh-engine results.
[Repeated-engine raw measurements](gpu_risk_float32_cached_20260909.json) also
include per-stage timings.

## Further optimization opportunities

1. **Market-data assembly:** the separate CPU profile of the 1,024-asset correlated
   calculation attributes 0.872 seconds to two `assembleData` calls, including
   0.597 seconds in per-column `groupby(...).nunique()`. The profiled run has
   1.096 seconds of recorded total time; profiling inflates timings, so these
   are bottleneck evidence rather than an end-to-end latency estimate.
   A unique-date fast path could avoid redundant duplicate/conflict aggregation
   for ordinary close-price tables. Any implementation must preserve conflicting
   duplicate rejection, complementary missing-value merging, canonical instrument
   order, inclusive dates, and the separate derivative-quote identity rules.
2. **Torch dynamics launch overhead:** the 1,024-asset correlated trace contains
   8,836 GPU kernels totaling 30.865 ms across the full calculation. Explore
   fusing elementwise updates within the existing Torch solver or capturing
   reusable fixed-shape dynamics. Sparse operations, noise chunks, changing batch
   shapes, and host synchronization need explicit handling. Measure full solves,
   including compilation/capture amortization, before retaining such a change.
3. **Resident scoring transfers:** float32 risk encoding now needs authoritative
   float64 scoring uploads. Reusing resident integer indices and reducing staging
   allocations are candidates for measurement; dropping scoring precision would
   violate the near-tie ranking contract.

These are follow-up candidates, not claimed speedups or additional code changes.
This study covers synthetic inputs on one GPU, not complete backtests, Vanguard,
multiple GPUs, or ROCm. Ill-conditioned histories and bin-boundary portfolios
require application-specific validation. PyTorch documents the sensitivity of
linear algebra to input precision and conditioning in its
[numerical accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).

## Reproduce and validate

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/benchmark_risk_precision.py --device cuda:0 \
  --assets 64 1024 --window 60 --steps 256 --runs 8 --repeats 5 --warmups 2 \
  --trace-directory /tmp/risk-traces --output /tmp/risk-precision.json
```

Add `--reuse-engine` to retain acquisition caches between reports. This differs
from the fresh-engine table above. Add `--risk-dtype float64` to the existing
`tools/benchmark_resident_pipeline.py` for a full-precision resident reference;
that tool's solver dynamics now default to float32 and can be overridden with
`--solver-dtype float64`.

[Raw measurements](gpu_risk_float32_20260909.json) include inputs/configurations,
source hashes, durations, margins, dtype diagnostics, and peak allocations.
The original trace archive was downloaded to `/tmp/margin-float32-traces.tar.gz`.

- Local full Python suite: **222 tests, successful, four skips** (three GPU checks
  and the opt-in network integration).
- GPU: `tests.test_accelerator_pipeline tests.test_device_resident_pipeline
  tests.test_torch_dynamics tests.test_candidate_selection tests.test_bqm_solver`
  — **72 tests, successful, three native-library skips**.
- CPU float32 smoke checks passed for both precision and resident-pipeline
  benchmark tools; the canonical example YAML parses successfully.
- Native CMake/CTest was not rerun because native code and ABI were unchanged.

## Workflow reference

The optimize-for-gpu skill supplied the correctness, profiling, and synchronized
benchmark workflow. Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M.
(2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
Bibliographic details were checked against the current arXiv record.

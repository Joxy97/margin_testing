# Repair of the Group 1 factor-stress sweep

All **8,640 repaired encodings are feasible**, compared with 10 raw feasible encodings. No GPU solves were rerun. The original archive is unchanged, verified by its before/after digest.

Auxiliary reconstruction preserved the scenario for 2,575 samples, including the 10 already-valid encodings. The remaining 6,065 samples required projection onto the integer ball. Optional local improvement was then applied to every feasible scenario.

| Stage | Feasible encodings | Breaches |
|---|---:|---:|
| Raw solver output | 10 / 8,640 | 0 among the 10 valid margins; 8,630 missing |
| Projection and auxiliary reconstruction | 8,640 / 8,640 | 219 |
| Plus feasible local improvement | 8,640 / 8,640 | 0 |

Breaches are direct comparisons of realized loss against margin. Counts pool repeated solver/penalty/seed trials over the same 20 dates, 2026-08-10 through 2026-09-04. They are not 8,640 independent market observations. Radius 3 remains illustrative.

## Quality

All 8,640 samples reached a local optimum before the 1,024-move cap. Mean accepted moves: 43.33; maximum: 227. Each move respects the exact integer budget and decreases actual exponential P&L. Projection itself can reduce loss; auxiliary reconstruction alone never changes it.

Mean final margin: **0.984743%**. Mean shortfall relative to the exact-repricing continuous reference: **2.1404 basis points**; maximum: **10.7676 basis points**. The reference allows continuous coordinates, so these gaps combine discretization and local-search error. The per-trial quadratic lattice gap is also retained, but that reference optimizes a different objective (the Taylor approximation).

| Bits | Solver | Mean margin | Mean reference gap (bp) | Maximum gap (bp) | Mean repair (ms) |
|---:|---|---:|---:|---:|---:|
| 4 | torch_sbm | 0.969695% | 3.6452 | 10.7676 | 0.905 |
| 4 | torch_svl | 0.975478% | 3.0669 | 10.7676 | 1.189 |
| 4 | torch_transverse_route | 0.971270% | 3.4877 | 10.7676 | 1.224 |
| 6 | torch_sbm | 0.989933% | 1.6213 | 4.3534 | 3.513 |
| 6 | torch_svl | 0.994178% | 1.1969 | 4.3534 | 4.449 |
| 6 | torch_transverse_route | 0.992235% | 1.3912 | 4.3534 | 4.684 |
| 8 | torch_sbm | 0.984368% | 2.1778 | 6.2947 | 12.644 |
| 8 | torch_svl | 0.992718% | 1.3429 | 5.3632 | 19.413 |
| 8 | torch_transverse_route | 0.992814% | 1.3332 | 6.2947 | 19.050 |

## Timing

Additional CPU postprocessing wall time: **73.074 seconds**, including sample I/O, setup, input fingerprinting and report writes. No acquisition, PCA fitting or GPU sampling was repeated. Precomputed exponential move increments are shared per date/resolution; solutions are not cached.

| Repair work over all samples | Seconds |
|---|---:|
| decodeSeconds | 0.232 |
| projectionSeconds | 0.119 |
| initialRepricingSeconds | 0.987 |
| auxiliaryRebuildSeconds | 0.888 |
| improvementSeconds | 60.936 |
| finalValidationSeconds | 1.215 |
| totalSeconds | 64.389 |

Reusable setup took 0.238 seconds. Mean direct repair time was 7.452 ms per sample. Local improvement dominates the repair cost; projection and auxiliary reconstruction are inexpensive. Timings use a single CPU process with one BLAS thread.

The earlier GPU sweep took 266.120 seconds. Adding this separate CPU pass gives 339.194 seconds of sequential measured work across the two runs; this is not a measured integrated GPU/repair pipeline latency. Independent verification is additional.

## Artifacts

- [All repaired trials](trials.csv), [penalty/solver summaries](summary.csv), [breaches by seed](breaches_by_seed.csv).
- [Solver/resolution comparison](by_solver_resolution.csv), [timings and totals](results.json), [verification](verification.json).
- [Original sweep](../README.md), [algorithm and commands](../../../docs/benchmarks/factor_stress.md).

Verification reloads every projected and improved sample, checks product/slack consistency and the integer radius, directly reprices the portfolio, recomputes breaches, checks monotonic improvement, and independently tests every distinct claimed local optimum using direct neighboring-scenario repricing.

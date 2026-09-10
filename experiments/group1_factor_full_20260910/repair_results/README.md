# Repair of the Group 1 factor-stress sweep

All **2,646 repaired encodings are feasible**, compared with 0 raw feasible encodings. No GPU solves were rerun. The original archive is unchanged, verified by its before/after digest.

Auxiliary reconstruction preserved the scenario for 0 samples, including the 0 already-valid encodings. The remaining 2,646 samples required projection onto the integer ball. Optional local improvement was then applied to every feasible scenario.

| Stage | Feasible encodings | Breaches |
|---|---:|---:|
| Raw solver output | 0 / 2,646 | 0 among valid margins; 2,646 missing |
| Projection and auxiliary reconstruction | 2,646 / 2,646 | 0 |
| Plus feasible local improvement | 2,646 / 2,646 | 0 |

Breaches are direct comparisons of realized loss against margin. Counts pool repeated solver/penalty/seed trials over the same 294 dates, 2025-07-08 through 2026-09-04. They are not 2,646 independent market observations. Radius 3 remains illustrative.

## Quality

2,646 of 2,646 samples reached a local optimum within the 1,024-move cap. Mean accepted moves: 8.55; maximum: 48. Each move respects the exact integer budget and decreases actual exponential P&L. Projection itself can reduce loss; auxiliary reconstruction alone never changes it.

Mean final margin: **1.142348%**. Mean shortfall relative to the exact-repricing continuous reference: **0.7878 basis points**; maximum: **8.5851 basis points**. The reference allows continuous coordinates, so these gaps combine discretization and local-search error. The per-trial quadratic lattice gap is also retained, but that reference optimizes a different objective (the Taylor approximation).

| Bits | Solver | Mean margin | Mean reference gap (bp) | Maximum gap (bp) | Mean repair (ms) |
|---:|---|---:|---:|---:|---:|
| 8 | torch_sbm | 1.142177% | 0.8048 | 6.2618 | 2.140 |
| 8 | torch_svl | 1.142585% | 0.7641 | 6.2618 | 2.191 |
| 8 | torch_transverse_route | 1.142281% | 0.7944 | 8.5851 | 2.304 |

## Timing

Additional CPU postprocessing wall time: **10.546 seconds**, including sample I/O, setup, input fingerprinting and report writes. No acquisition, PCA fitting or GPU sampling was repeated. Precomputed exponential move increments are shared per date/resolution; solutions are not cached.

| Repair work over all samples | Seconds |
|---|---:|
| decodeSeconds | 0.101 |
| projectionSeconds | 0.039 |
| initialRepricingSeconds | 0.298 |
| auxiliaryRebuildSeconds | 0.380 |
| improvementSeconds | 4.520 |
| finalValidationSeconds | 0.511 |
| totalSeconds | 5.852 |

Reusable setup took 1.768 seconds. Mean direct repair time was 2.212 ms per sample. Local improvement dominates the repair cost; projection and auxiliary reconstruction are inexpensive. Timings use a single CPU process with one BLAS thread.

The earlier GPU sweep took 594.980 seconds. Adding this separate CPU pass gives 605.525 seconds of sequential measured work across the two runs; this is not a measured integrated GPU/repair pipeline latency. Independent verification is additional.

## Artifacts

- [All repaired trials](trials.csv), [penalty/solver summaries](summary.csv), [breaches by seed](breaches_by_seed.csv).
- [Solver/resolution comparison](by_solver_resolution.csv), [timings and totals](results.json), [verification](verification.json).
- [Original sweep](../README.md), [algorithm and commands](../../../docs/benchmarks/factor_stress.md).

Verification reloads every projected and improved sample, checks product/slack consistency and the integer radius, directly reprices the portfolio, recomputes breaches, checks monotonic improvement, and independently tests every distinct claimed local optimum using direct neighboring-scenario repricing.

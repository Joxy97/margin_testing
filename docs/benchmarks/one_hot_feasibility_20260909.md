# One-hot feasibility and categorical search — 9 September 2026

Measurements justify adding a solver that searches only feasible category selections.
`torch_categorical` is now registered, configurable through YAML, and usable through
the host and resident MarginEngine paths. It is graph-colored categorical heat-bath
annealing, **not a renamed SBM/SVL implementation**. Existing solver configurations
and repair behavior are preserved.

Across 120 categorical GPU measurement cases (1,920 raw candidates), every raw
candidate was feasible and production repair was never called. In the primary
64-group, λ = 2 comparison, 64 categorical sweeps were 1.94–5.89× faster than the
256/1,024-step SBM/SVL baselines, with equal or lower returned original-QUBO energy
in every matched comparison. These are synthetic single-problem measurements,
not a claim about all portfolios or large scenario batches.

## What was measured

Hardware/software: initially idle NVIDIA RTX A4000 (16,376 MiB), Python 3.12.14,
PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.5.3. One Torch/OpenMP/OpenBLAS CPU thread.
All solver dynamics use float32; candidate scoring and repair retain float64.

Fixtures use four candidate return bins per asset, signed exposures, and nonnegative
bivariate-Gaussian compatibility costs. Histories/PCA are not part of these solver
fixtures. There are 4, 64, or 63 groups; exposure scales are 1 and 10. Group edges
connect cyclic neighbors at offsets one and three. Even-sized graphs need two
colors; odd-sized graphs exercise additional colors. This is a sparse, structured
family, not a random dense graph or an exported production portfolio.

Primary penalties are 0.5, 2, 8, and a coefficient-based sufficient bound; solver
seeds are 1, 13, and 31, with 16 trajectories per solve. The primary SBM/SVL sweep
uses 256 and 1,024 integration steps. Additional 64-group runs use 2,000/4,000 steps
at λ = 2 and the sufficient bound. SVL uses its default Euler–Maruyama integrator,
`dt=0.01`, and temperature 0.01; this does not reproduce the example YAML's separately
tuned weak-order-2 schedule. SBM uses its default dynamics parameters. Effective
parameters are archived in each row.

For each configuration, one warmup precedes three complete solves timed with CUDA
synchronization. Complete latency includes coefficient preparation, transfers,
dynamics, selection/repair, and output. A separate instrumented solve captures raw
samples and measures production repair/model-building calls. Another diagnostic
applies categorical descent to **all** raw samples, including feasible ones.
These extra operations are excluded from the reported complete-solve latency.
Repair time is an instrumented CPU duration, not a precisely additive fraction of
a different uninstrumented run.

The source `seedOffset` stays fixed across penalties so λ does not incidentally
change RNG streams in this experiment. Normal application encoding derives seeds
from coefficients and therefore may change streams when λ changes. Four-group
fixtures have 256 feasible assignments and are checked against exhaustive minima.

## Penalty and feasibility results

At λ = 2, aggregated over both exposure scales and three seeds:

| Groups | Solver | Steps | Raw feasible candidates | Solves with no feasible candidate |
|---:|---|---:|---:|---:|
| 64 | SBM | 256 | 10.42% | 3 / 6 |
| 64 | SBM | 1,024 | 48.96% | 3 / 6 |
| 64 | SVL | 256 | 0.00% | 6 / 6 |
| 64 | SVL | 1,024 | 26.04% | 3 / 6 |
| 64 | SBM | 4,000 | 50.00% | 3 / 6 |
| 64 | SVL | 4,000 | 50.00% | 3 / 6 |

An infeasible candidate often leaves groups empty: the 256-step SVL rows have
87.61% empty groups on average. The λ = 2, 256-step median production repair time
is about 29.7 ms for SBM and 66.2 ms for SVL, versus median complete latencies of
56.7 and 104.2 ms respectively. Some solves incur no repair because at least one
feasible candidate exists; raw infeasibility alone does not imply repair overhead.

For the unpenalized objective

\[
F(x)=\sum_i a_i x_i+\sum_{i<j}b_{ij}x_i x_j,
\]

the benchmark uses the conservative bound

\[
\lambda > \max_i \left(|a_i|+\sum_{j\ne i}|b_{ij}|\right).
\]

Specifically it takes 1.01 times that bound plus 0.000001. A correcting bit flip
reduces the one-hot penalty by at least λ while changing F by no more than the
bound, so an infeasible binary state cannot be a local minimum under all single-bit
flips. This guarantees feasible global minima, not feasible finite-step trajectories.
An additional exhaustive six-bit check confirmed feasible global minimizers at the
bound.

Higher penalties alone were not a universal cure. At the sufficient bound,
64-group SBM still had zero feasible candidates in 9/12 primary solves. Longer
runs helped: at 4,000 steps it had feasible candidates in all six solves, but only
58.33% of raw candidates were feasible. SVL at the bound reached **100% raw feasibility
at both 2,000 and 4,000 steps**. Thus sufficiently long, strongly penalized SVL can
avoid repair on these fixtures. Its categorical counterpart still returned lower
energy in all six matched bound-penalty comparisons at each of those lengths and
was approximately 9.55×/19.03× faster (median paired ratios).

Repair also performs real optimization. In 56 of the 192 primary baseline cases,
at least one feasible raw sample existed, yet applying the diagnostic descent to
all raw samples produced a lower energy than production selection returned.
Simply deleting repair or replacing it with a final argmax would discard some of
this work without demonstrating equivalent quality.

## Categorical comparison

The new solver starts from one category per group and samples each group's
conditional categorical distribution while cooling. Graph coloring permits
simultaneous updates only for groups with no interactions between them. Four final
zero-temperature sweeps improve already-valid selections. Tests audit the one-hot
view after every update; this is not a final feasibility projection.

For λ = 2, 64 groups, both exposure scales and all three seeds:

| Solver | Work per trajectory | Median complete ms | Raw feasible candidates |
|---|---|---:|---:|
| Categorical | 32 color sweeps + 4 greedy sweeps | 18.524 | 100% |
| Categorical | 64 color sweeps + 4 greedy sweeps | 29.067 | 100% |
| SBM | 256 integration steps | 56.659 | 10.42% |
| SBM | 1,024 integration steps | 129.711 | 48.96% |
| SVL | 256 integration steps | 104.215 | 0.00% |
| SVL | 1,024 integration steps | 171.400 | 26.04% |

Sweep and integration-step counts are different units of work. Compare latency
and solution quality, not those counts. Across all 48 primary configurations per
baseline length matched against 64 categorical sweeps:

| Baseline | Categorical lower energy | Equal within 1e-8 | Higher energy |
|---|---:|---:|---:|
| SBM, 256 steps | 20 | 28 | 0 |
| SBM, 1,024 steps | 21 | 27 | 0 |
| SVL, 256 steps | 32 | 16 | 0 |
| SVL, 1,024 steps | 20 | 28 | 0 |

All small categorical cases reached the exact feasible reference within floating-point
tolerance. Larger cases have no exact-optimality claim. Comparisons use final original
QUBO energy after production selection/repair, rather than infeasible raw energy.
The 63-group additional comparisons also had no categorical quality regressions;
median paired speed ratios were 2.03–2.72× for 64 categorical sweeps against the
256/1,024-step baselines. More colors reduce the parallel-update advantage.

The new solver is not uniformly faster: across *all* primary sizes/penalties,
64 categorical sweeps versus 256-step SBM had a median paired ratio of 0.963×.
The principal benefit is guaranteed feasibility and improved quality; speed depends
on group count, graph structure, baseline tuning and whether repair dominates.

## Use and implementation limits

```bash
PYTHONPATH=src python -m margin_engine config/categorical.example.yaml
```

This runnable example uses existing ten-asset local synthetic data, nine risk states,
`lambdaOneHot: 0`, and GPU when available (CPU otherwise). The local CPU run produced
margin 0.00044115356714565526 and paired greedy margin 0.049112507090718745; they solve
different constrained/independent problems, so equality is not expected.

To change an existing configuration, replace the whole solver block:

```yaml
solver:
  type: torch_categorical
  constructorParameters: {device: cuda:0}
  solverParameters:
    steps: 64
    runs: 16
    temperature_start: 1.0
    temperature_end: 0.01
    greedy_sweeps: 4
    seed: 1
    dtype: float32
    run_batch_size: 16
```

The solver requires complete disjoint one-hot groups, supports ragged/noncontiguous
groups, and rejects uncovered variables. Within-group off-diagonal terms vanish;
diagonals are folded into linear terms; common group shifts are removed from
dynamics. Original coefficients and offsets remain authoritative for final float64
ranking. Consequently a positive one-hot penalty is unnecessary for feasibility;
λ = 0, 2, and 8 produced identical selections in an additional deterministic check.

Temperature is in objective-energy units and may need tuning when exposures or
compatibility scales change. This is still a heuristic optimizer, not an optimality
guarantee. Trajectories are batched, but problems run sequentially per device shard;
large existing SBM/SVL scenario batches may benefit from their block-diagonal
batching. Resident mode builds categorical topology from the host source snapshot
and uploads it. Padded ragged groups and many graph colors can limit efficiency.
No claim is made about real Vanguard results, dense graphs, tuned SVL schedules,
multi-GPU speedups, or rolling backtests.

The existing repair fallback remains for other solvers; it is never entered by
valid categorical trajectories. No existing experiment configuration was switched.
The new solver type participates in the ordinary configuration fingerprint, so no
additional checkpoint schema or native ABI change is needed.

## Reproduction and verification

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/benchmark_one_hot_feasibility.py --output /tmp/onehot-baseline.json

PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/benchmark_one_hot_feasibility.py --solvers torch_categorical \
  --steps 32 64 --output /tmp/onehot-categorical.json
```

Use `--groups 63 --penalties 2` for the odd-sized additional sweep and
`--groups 64 --penalties 2 --steps 2000 4000` for the longer baseline. The sufficient
bound is always included in addition to the explicit penalty list. The JSON records
raw objective/penalty energies, raw feasibility, repair calls/times, polished energies,
returned energies, exact gaps, parameters, sample hashes, and environment details.
[Combined raw results](one_hot_feasibility_20260909.json) contain all five sweeps
and source hashes.

- Local full suite: **231 tests, successful, five skips** (four GPU checks and the
  opt-in network integration).
- GPU suite: `tests.test_torch_categorical tests.test_accelerator_pipeline
  tests.test_device_resident_pipeline tests.test_torch_dynamics tests.test_bqm_solver`
  — **78 tests, successful, three native-library skips**.
- New tests cover every-update feasibility, disabled repair, exact source energy,
  canonical duplicate/reversed/diagonal semantics, within-group terms, ragged groups,
  a dense deterministic descent reference, chunk-independent raw candidates,
  invalid configurations/coverage, singleton groups, factory/YAML registration,
  and host/resident MarginEngine integration.
- Additional GPU resident float32 check with zero penalty and uneven run chunks
  passed with margin 0.1798027604818344 versus CPU reference 0.1798027685847499.
- Native CMake/CTest was not rerun: native code and ABI were unchanged. Multiple GPUs
  and ROCm hardware were unavailable. Full-size portfolios/backtests were not run;
  benchmark scope is the fixtures described above.

## References

- Gonzalez, J., Low, Y., Gretton, A., & Guestrin, C. (2011).
  [Parallel Gibbs Sampling: From Colored Fields to Thin Junction Trees](https://proceedings.mlr.press/v15/gonzalez11a.html).
  Graph coloring gives conditionally independent groups for simultaneous updates.
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
  *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
  [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
  The optimize-for-gpu skill supplied the correctness and measurement workflow;
  bibliographic details were checked against the current record earlier in this session.

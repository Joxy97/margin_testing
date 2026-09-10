# Group 1 correlation normalization and one-hot penalty sweep

Random long portfolio: 8,590 stocks, gross exposure 1, seed 20260910. Margin date: 2025-07-08. All trials use the same 105 prepared scenarios.

Hardware: 8 × Tesla V100-SXM2-32GB; PyTorch 2.7.1+cu126, CUDA 12.6. Preparation took 619.3 s and is excluded from the solve budget.

Correlation interactions fell from 1,517,938,940 to 75,459,355 across the 105 QUBOs (95.03% removed). Variable counts and one-hot groups are unchanged.

| Solver | Correlation terms | One-hot penalty | Solved | Wall s | Repair calls | Repair s¹ | Raw feasible | Margin² |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SBM | legacy | 1 | 105/105 | 164.51 | 105 | 282.42 | 0.00% | 0.00000000 |
| SVL | legacy | 1 | 105/105 | 170.74 | 105 | 322.39 | 0.00% | 0.00000000 |
| TRF | legacy | 1 | 105/105 | 235.28 | 210 | 464.01 | 0.00% | 0.01461568 |
| SBM | pruned | 1 | 105/105 | 29.29 | 105 | 117.62 | 0.00% | 0.20926884 |
| SVL | pruned | 1 | 105/105 | 25.13 | 105 | 81.48 | 0.00% | 0.19867212 |
| TRF | pruned | 1 | 105/105 | 72.90 | 210 | 304.93 | 0.00% | 0.22192963 |
| SBM | pruned | 10 | 105/105 | 31.60 | 105 | 101.67 | 0.00% | 0.20572379 |
| SVL | pruned | 10 | 105/105 | 27.40 | 105 | 84.67 | 0.00% | 0.19867212 |
| TRF | pruned | 10 | 105/105 | 73.22 | 210 | 304.28 | 0.00% | 0.22192963 |
| SBM | pruned | bound | 105/105 | 29.85 | 105 | 97.81 | 0.00% | 0.20031934 |
| SVL | pruned | bound | 105/105 | 27.37 | 105 | 85.00 | 0.00% | 0.19867212 |
| TRF | pruned | bound | 105/105 | 71.10 | 210 | 302.34 | 0.00% | 0.22192963 |

## Measured effect

- SBM: normalization/pruning reduced total trial time 5.62× and summed repair time 2.40× at λ=1. Repair calls changed from 105 to 105.
- SVL: normalization/pruning reduced total trial time 6.79× and summed repair time 3.96× at λ=1. Repair calls changed from 105 to 105.
- TRF: normalization/pruning reduced total trial time 3.23× and summed repair time 1.52× at λ=1. Repair calls changed from 210 to 210.

The sufficient per-QUBO penalties ranged from 4.673 to 484.645. At these penalties, 0 of 420 raw candidates were fully one-hot feasible.

¹ Repair seconds are summed across eight concurrently running workers, so they can exceed wall time. Sparse repair-model construction is recorded separately in the raw results.
² Margin is the greatest decoded loss among the 105 returned feasible samples, as a fraction of unit exposure. It is a heuristic result, not a certified optimum. The previous and modified correlation objectives have different meanings.

## Method and limits

- Each solver/penalty trial has a 300 s global deadline. Transfers, solver preparation, integration, candidate checking, repair, source-energy validation, and result publication fall inside that wall-clock window. Startup, tiny GPU warmups, file preload, reference-feature preparation, and scenario generation are outside it.
- Each GPU owns a fixed scenario shard; every scenario receives a share of its remaining deadline. Incomplete or late results are not counted as solved. The coordinator terminates its own workers at the global deadline.
- The sufficient penalty is max(1, nextafter(1.01 × max_i(|a_i| + sum_j |b_ij|), +∞)), computed from the nonpenalty objective. It guarantees feasible global minimizers, not feasible finite-step trajectories.
- Frozen BiqMac SAC actors control TRF and SVL. SBM uses defaults after its transferred actor worsened the pilot mean objective. Reference features use the normalized/pruned λ=1 QUBO and remain fixed across the sweep. Actor inference is timed, but no new SAC training is performed and no out-of-distribution generalization is assumed.
- The saved actor may choose algorithm parameters; execution is constrained to float32, sparse matrices, the recorded step/trajectory counts, and bounded candidate storage. TRF retains initial/final candidates. The standard deterministic categorical repair remains enabled; none of the solvers searches categories directly.
- This is one seeded portfolio and one run per setting. The early pilot overlapped CPU scenario preparation; the full sweep begins after preparation completes. Timing differences are descriptive, not confidence intervals.
- Candidate statistics are taken after the solver's existing checkpoint deduplication. The added feasibility counters and timing instrumentation remain inside the measured execution window.
- Sufficient penalties may be hundreds of times larger than individual portfolio-return coefficients. Float32 dynamics can lose small distinctions at those scales; authoritative source scoring and repair use float64.

## Artifacts

- [Plot](plots/penalty_sweep.png) · [PDF](plots/penalty_sweep.pdf)
- [Aggregate CSV](results/summary.csv) · [Per-scenario records](results/rows.json)
- [Configuration](configuration.yaml) · [Portfolio](portfolio.csv) · [Input identity](portfolio_manifest.json)
- [Prepared QUBO identities](qubos_manifest.json) · [SAC selection](sac_selection.json)

## Reproduce

From the repository root with the recorded dependencies and eight GPUs:

```bash
PYTHONPATH=src:tools python experiments/group1_penalty_sweep_20260910/prepare.py
PYTHONPATH=src:tools python experiments/group1_penalty_sweep_20260910/run_sweep.py --steps 512 --runs 1 --budget 300 --policy selected
python experiments/group1_penalty_sweep_20260910/summarize.py
```

Prepared QUBO arrays are retained on the benchmark instance under `/workspace/margin-penalty-sweep/experiments/group1_penalty_sweep_20260910/qubos/`. The local archive contains hashes, per-scenario results, and returned samples.

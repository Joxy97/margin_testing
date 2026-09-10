# Joint factor stress: 102-stock experiment

The hard-budget QUBO and the continuous/lattice references are implemented. In the remote GPU experiment, 0/27 returned heuristic solutions satisfied every product and budget constraint. No repair or reference fallback was applied to the heuristic samples.

## Requested remote run

Group 49: 102 stocks; a seeded random long-only portfolio with weights summing to 1 (seed 20260910). As-of 2026-09-08; 125 return observations from 2026-03-05 through 2026-09-07, EW decay 0.93. Two PCA factors plus one portfolio residual coordinate. Radius 3 is illustrative, with no VaR/ES or coverage claim.

Tesla V100-SXM2-32GB, cuda:0, Torch 2.7.1+cu126, float64. **runs=16, steps=10000**, three seeds per solver/resolution. The measured benchmark interval was 27.714 seconds. It includes preparation, references, solves, scoring, validation and result writes after library imports; excludes SSH and environment setup. GPU solves synchronize before and after timing. First-use initialization is included, so small timing differences should not be interpreted as steady-state performance.

| Bits/coordinate | Scenario bits | Product bits | Slack bits | Variables | Edges |
|---:|---:|---:|---:|---:|---:|
| 4 | 12 | 18 | 6 | 36 | 630 |
| 6 | 18 | 45 | 10 | 73 | 2,628 |
| 8 | 24 | 84 | 14 | 122 | 7,381 |

Edges count unique nonzero off-diagonal QUBO interactions, after aggregation. All three graphs are complete: squaring one global linearized budget couples every bit.

| Bits/coordinate | Solver | Mean solve seconds | Fully valid / trials | Scenario inside ball / trials |
|---:|---|---:|---:|---:|
| 4 | torch_sbm | 0.942 | 0/3 | 2/3 |
| 4 | torch_svl | 1.325 | 0/3 | 0/3 |
| 4 | torch_transverse_route | 0.578 | 0/3 | 0/3 |
| 6 | torch_sbm | 0.887 | 0/3 | 1/3 |
| 6 | torch_svl | 1.317 | 0/3 | 1/3 |
| 6 | torch_transverse_route | 0.642 | 0/3 | 1/3 |
| 8 | torch_sbm | 0.877 | 0/3 | 3/3 |
| 8 | torch_svl | 1.322 | 0/3 | 2/3 |
| 8 | torch_transverse_route | 0.564 | 0/3 | 2/3 |

The final column checks only the decoded scenario radius. It does not mean the auxiliary products and slack are correct. Invalid full encodings have null accepted margins in the results. These observations describe these solver settings, not a proof that no heuristic can solve the model.

## References

| Method | Exact repriced stress loss | Reference seconds |
|---|---:|---:|
| linear_analytic | 2.072684% | 0.000060 |
| quadratic_continuous | 2.072885% | 0.233201 |
| exact_repricing_continuous | 2.072886% | 0.002924 |
| Exact quadratic lattice, 4 bits/coordinate | 2.015988% | 0.008564 |
| Exact quadratic lattice, 6 bits/coordinate | 2.062463% | 0.085147 |
| Exact quadratic lattice, 8 bits/coordinate | 2.072424% | 1.433190 |

Every row is repriced with the original exponential log-return transformation. The linear, quadratic and lattice methods optimize their stated surrogate; only the exact-repricing continuous method optimizes that exponential P&L directly. The continuous quadratic and exact-repricing solutions have convex lower-bound certificates with reported gaps at floating-point precision.

The 8-bit lattice loss differs from the exact-repricing continuous loss by 0.0462 basis points of portfolio exposure. The maximum absolute Taylor error at 8,192 sampled interior/boundary points was 0.2649 basis points; this is a sampled diagnostic, not a uniform error bound.

**The worst observed loss in the fitting window was 3.4791%, above the 2.0729% radius-3 reduced-model stress loss.** Taking their maximum as a historical stress floor gives 3.4791%. That comparison shows why the illustrative radius must be calibrated and why omitted residual/sector/tail scenarios need separate validation. It is not an out-of-sample margin coverage result.

## Interpretation and artifacts

The conversion removes the 105-QUBO loop and dramatically reduces variable count, but the expanded hard penalty is dense and poorly conditioned. Maximum absolute coefficients grow from approximately 1.46e3 at 4 bits to 1.08e8 at 8 bits for this portfolio, while the source P&L is of order 1e-2. Direct integer feasibility and unexpanded energy checks are retained alongside ordinary float64 QUBO energy. The continuous reference is the useful baseline for this three-dimensional model.

- [GPU summary CSV](remote_results/summary.csv), [complete GPU results](remote_results/results.json), [sample verification](remote_results/verification.json), [GPU log](remote_run.log).
- [Initial CPU results](results/results.json): 64 runs, 2,000 steps, three seeds; 0/27 fully valid returned encodings.
- [Longer CPU check](longer_check/results.json): smallest model, 256 runs, 20,000 steps; 0/3 fully valid returned encodings.
- [Model, conversion and limitations](../../docs/benchmarks/factor_stress.md).
- `remote_verification.txt` records remote source/data hashes and idle GPU state after completion.

## Reproduce

```bash
PYTHONPATH=src python tools/benchmark_factor_stress.py \
  --group yahoo_equities_grouped/groups/US/group_49_102.csv \
  --date 2026-09-08 --bits 4 6 8 --radius 3 \
  --runs 16 --steps 10000 --repeats 3 --device cuda:0 \
  --output experiments/factor_stress_20260910/remote_results
PYTHONPATH=src python tools/report_factor_stress.py experiments/factor_stress_20260910/remote_results
```

Use a new output directory for another run. GPU benchmarking used the existing V100-compatible `/workspace/margin-sweep-venv` on the requested instance.

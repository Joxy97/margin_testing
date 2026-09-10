# Group 1: binary TRF with a bounded one-hot penalty and repair

8590 stocks, 2025-07-08, 21 x 5 = 105 QUBOs, 4 trajectories, 10000 steps.
The categorical TRF experiment is stopped. This experiment uses the original
`torch_transverse_route` solver and retains its normal repair path.

## Penalty construction

For each risk state, the experiment runner calls the standard portfolio QUBO
visitor with `lambdaOneHot=0`. It computes
`B = max_i(abs(a_i) + sum_j(abs(b_ij)))` for the non-penalty objective, including
the unchanged `lambdaCompat=0.1` terms. Diagonal terms are folded into the bound.
It then adds `lambda * (sum_group(x) - 1)^2` to every group with
`lambda = max(configured floor, nextafter(1.01 * B, +inf), 1e-6)`.
The linear, pairwise and constant penalty terms are all updated. Source seed
identity includes the new penalty. Source arrays are not mutated.

This is a conservative sufficient bound: an invalid group's correcting bit flip
decreases its penalty by at least lambda while the non-penalty objective can
increase by at most B. This excludes infeasible single-bit local minima in exact
arithmetic. It does NOT guarantee finite-time TRF feasibility or eliminate repair.
Float32 dynamics are retained for continuity with earlier runs; large penalties
can obscure small portfolio coefficients and worsen conditioning. Final scoring
and repair use the float64 source QUBO. No speed or quality improvement is assumed.

**Use `run_single_date.py`, not the plain MarginEngine YAML CLI.** The adaptive
penalty is experiment-local logic at the encoding boundary. The YAML's value 1.0
is a floor in this runner; the ordinary CLI would use it as a fixed penalty.

## Tracking

```bash
bash tools/track_group1_trf_bound.sh status
bash tools/track_group1_trf_bound.sh logs
bash tools/track_group1_trf_bound.sh fetch
```

Supervisor program: `group1_trf_bound` on `13.56.204.87:21963` (eight RTX 5090s).
Remote project: `/workspace/group1-trf-bound-20260910`.
Downloads go into this directory's `results/`.

`penalties.csv` is flushed after every encoded QUBO and records B, the actual
lambda, coefficient maxima and a float32 penalty-resolution indicator.
`progress.log` reports encoding, solving, candidate batches and repair calls.
`status.json` gives live progress. Final results are `benchmark_single_date.csv`
and `report.json`. Stage timings overlap; concurrent repair durations must not
be interpreted as wall time. Log-viewer Ctrl-C leaves the supervised job running.
Restart repeats the day; it does not resume individual QUBOs.

All changes are confined to this experiment and its tracking tool. No production
solver, parser, existing configuration or previous result is changed. No separate
unit tests are run as part of this launch; the requested margin run records its
own runtime failures and measurements.

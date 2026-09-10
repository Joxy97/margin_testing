# BiqMac MaxCut: SBM, SVL and Transverse Route

The persistent benchmark runs on `45.59.100.176:43437`, in the isolated project
`/workspace/biqmac-maxcut-20260909`, using eight RTX 2080 Ti GPUs. It is managed
by the Supervisor program `biqmac_maxcut`; it survives closing the SSH session.

## Track and fetch

Run these commands from the local repository root. SSH uses `margin_testing`
and prompts for its passphrase when needed; credentials are not stored here.

```bash
./tools/track_biqmac.sh status
./tools/track_biqmac.sh logs
./tools/track_biqmac.sh fetch
```

`logs` follows the live log; Ctrl-C closes the log viewer without stopping the
benchmark. `fetch` downloads current results into this directory's `results/`.
It can be used while the job is running or after completion. Fetch is explicit,
not a promise of a background local synchronization service.

Remote progress is in `/workspace/biqmac-maxcut-20260909/results/status.json` and
`progress.log`. The status records the active instance, solver, runs, steps,
repetition, start time, completed comparisons, percentage, and error count.
A heartbeat is printed every ten seconds; progress is at solve/repetition
boundaries, not inside an integration kernel.

From a shell on the server:

```bash
supervisorctl status biqmac_maxcut
# Stop all benchmark workers, retaining completed checkpoints:
supervisorctl stop biqmac_maxcut
# Resume the same benchmark after a stop or unexpected failure:
supervisorctl start biqmac_maxcut
```

## Scope and methodology

All 178 instances in the official [BiqMac MaxCut archive](https://biqmac.aau.at/library/tar_files/mac_all.tar.gz)
are included: 130 Rudy instances and 48 Ising instances. This is the medium-size
MaxCut collection linked from the [BiqMac library page](https://biqmac.aau.at/biqmaclib.html),
not its binary quadratic collection or a separate large-instance extension.
Published reference values are parsed from the site's
[reference tables](https://biqmac.aau.at/biqmaclib.tex); all 178 graphs have a match.
Input and source hashes, settings, versions and hardware are saved in the manifest.

The Cartesian product is `runs = 2, 4, 8, 16` and
`steps = 1000, 2000, 5000, 10000`: 16 settings and 2,848 instance/setting
comparisons. Each comparison executes all three solvers, with three timed
repetitions at fixed seed 1: 8,544 solver configurations and 25,632 timed solves.
`runs` is the number of trajectories per solve, not the timing-repeat count.
The repetitions use the same seed so timing replication does not silently triple
the independent search budget. Trials and winning bitstrings are retained in
`.checkpoints/*.json`.

Each process owns one GPU; solvers for a given instance/setting run on that same
GPU without overlapping another solver on that device. Solver order rotates
between jobs. Contexts are warmed up outside timings. CUDA is synchronized
before and after complete solver calls. Timings include preparation, transfers,
integration and source-energy scoring, but exclude input parsing, independent
cut validation and CSV publication. The other seven GPUs can be busy, so host
CPU contention is part of this parallel-throughput experiment, not isolated
single-GPU latency measurement.

All solvers use float32 dynamics, the same QUBO and trajectory count, and their
native integration/scaling defaults. SBM uses its default discrete simulated
bifurcation, SVL uses Euler-Maruyama, and TRF uses Euler with CUDA graphs and
periodic candidate checkpoints. This compares the repository implementations
as configured; equal step counts do not imply equal arithmetic or equal
candidate-checkpoint counts. Hyperparameters are not tuned against the test set.
In particular, the solvers have different internal normalization policies,
which can affect SVL on graphs with large-magnitude weights. Float64 original
QUBO energy and a direct weighted-cut calculation validate every reported result.

The GPU launch smoke test on `g05_60.0`, runs=2, steps=1000, seed=1 returned
cuts SBM=0, SVL=532, TRF=512 against a reference of 536. This exposes weak SBM
default behavior on that input; the benchmark retains the defaults and reports
the measured result instead of tuning a solver after seeing test-set outcomes.

The encoding is `QUBO(x) = -sum(w_ij * (x_i + x_j - 2*x_i*x_j))`.
Signed and repeated edges are retained; loops contribute zero. Reported cuts
are sums of crossing edge weights, not the four-times-cut Laplacian convention.

## Files and winners

All 16 CSVs are created with headers at launch and populated atomically:
`results/runs_<runs>_steps_<steps>.csv`. Each row is one graph, with each solver's
best cut, median/min/max elapsed seconds, individual trial values, published
reference gap, sample hash and errors. Cuts are maximized.

`quality_winners` lists all tied best-cut solvers. `winner` chooses the fastest
median time among those quality winners. `fastest_solver` reports speed alone.
A comparison with a solver error has no winner rather than silently favoring
successful solvers.

`summary.csv` contains per-setting standings. A quality win earns one point,
split equally among tied solvers; the setting winner has most quality-win
points, with total median time as the tie breaker. `overall.csv` applies the
same rule across settings. Standings remain provisional until
`status.json` says `completed`; `completed_with_errors` identifies an incomplete
comparison even though every job has been attempted.

Completed comparisons are checkpointed atomically. A stopped job repeats only
unfinished comparisons. Resume refuses changed inputs, settings or source hashes.
Local tests validate the signed MaxCut encoding against every cut of a small
graph and validate ties, timing medians and failure handling; a small remote run
checks the actual GPU solver path before the full sweep is launched.

The GPU measurement methodology follows the `optimize-for-gpu` skill, credited
in `docs/benchmarks/transverse_route.md`, including its Scientific Agent Skills
reference. This benchmark changes orchestration only, not solver dynamics.

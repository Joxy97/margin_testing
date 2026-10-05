# Multi-GPU runtime validation — 2026-10-05

Implemented in the local checkout and `/workspace/margin_testing` on the existing
Vast instance. No full campaign, cloud provisioning, driver replacement, commit
or push was performed. All numerical solver algorithms and frozen algorithm
settings are unchanged. `BENCHMARK_DEFINITIONS.txt` remains unchanged.

## Runtime changes

- The normal command detects every visible CUDA GPU. Explicit device selection
  and CPU fallback remain supported; `--require-gpu` forbids fallback.
- Bounded calibration compares one, two and four persistent workers per GPU,
  subject to GPU memory, aggregate host memory and container CPU quotas. A higher
  count needs more than 10% measured throughput improvement without worker errors.
  Counts are frozen for the experiment and restored on resume. Pilot seeds do
  not overlap campaign seeds; pilot outcomes are excluded from statistics.
- A shared queue interleaves solver/instance groups, preserves input affinity,
  and distributes remaining seeds to free GPU-affine workers. Different solvers
  and independent seeds run concurrently. Each trial gets fresh mutable state.
- CUDA contexts and immutable inputs are reused. Buffered worker messages are
  drained before watchdog checks, fixing false timeouts behind queued results.
  Candidate encoding uses vectorized host operations with unchanged timestamp
  and binary-validation semantics.
- Each attempt is journaled atomically, then appended to CSV. Summaries refresh
  every approximately five seconds. The result queue is bounded; historical
  captures are streamed from disk, rather than retained for the whole campaign.
- Independent scoring caches immutable weighted coefficients and overflow
  checks. Integer results remain exact, including Python bigint fallback.
- Background telemetry records GPU activity, VRAM, power and parent RSS/CPU.
  GPU activity is not measured TFLOPS. Progress includes all active trials and
  per-device counts. Every solver still has separate CSVs in one result directory.

The time budget covers RNG/state reset, algorithm preprocessing, search,
decoding, completed candidate capture and final synchronization. Startup,
immutable input preparation, independent audit, persistence and cleanup are
outside that budget. The 0.5-second watchdog grace grants no extra quality time.
Concurrency shares device and CPU capacity during each seed's wall-time budget;
these results must be distinguished from isolated latency measurements.

Main files: `runtime/{selection,cli,grid,autotune,scheduler,pool,supervisor,worker,
engine,storage,aggregate,telemetry,common}.py` under `src/qubo_benchmark`,
`src/qubo_benchmark/model.py`, `configs/benchmark_protocol.json`, focused runtime
and scorer tests, `.gitignore`, `AGENTS.md`, `README.md`, `README_BENCHMARKS.md`
and this report. No files under `src/qubo_solvers` changed.

## Hardware and test results

Actual host: four RTX 3090 GPUs, 24 GiB each; Python 3.12.14;
Torch 2.14.1 with CUDA 13.0; NVIDIA driver 590.48.01. Container CPU quota is
approximately 40.96 cores. The measured admission ceiling was 40 workers total;
that ceiling is a resource estimate, not a speed recommendation.

Focused final command:

```bash
python -m pytest tests/test_runtime_devices.py tests/test_runtime_scheduler.py tests/test_runtime_incremental.py tests/test_runtime_telemetry.py tests/test_runtime_workers.py tests/test_runtime_benchmark.py tests/test_problem_scoring.py tests/test_qubo_benchmark.py::ObjectiveTests -q
```

Result: **92 passed, 7 subtests passed, no skips, 33.12 seconds** on the GPU host.
This includes real CUDA capture/reuse checks, device discovery, admission,
calibration, exactly-once scheduling, bounded queues, cancellation, interrupted
resume, CSV repair and overflow-safe scoring.

The original four-worker greedy failure was reproduced before the fix:
one completed trial and three watchdog timeouts, exit 130. After the worker
changes, all four completed, exit 0, with solver times 0.487–0.510 seconds under
the unchanged one-second budget and 0.5-second grace. Cold command duration was
12.90 seconds versus 11.02 seconds for the aborted baseline; this establishes
correctness, not a cold-start speedup.

```bash
python run_benchmark.py 200 sparse 1 --instances representative --runs 4 --require-gpu --output-root benchmark_results/multigpu_upgrade/auto_all23
python run_benchmark.py 200 sparse 0.05 --instances representative --runs 4 --seed-workers 4 --require-gpu --output-root benchmark_results/multigpu_upgrade/deadline_50ms
```

Both used `gka1e`, every eligible solver and all four GPUs:

| Smoke | Trials | Completed with in-budget score | No in-budget candidate | Lost workers | Exit |
|---|---:|---:|---:|---:|---:|
| 1 second, automatic concurrency | 92 | 92 | 0 | 0 | 0 |
| 0.05 second, four workers/GPU | 92 | 81 | 11 | 0 | 1 |

The one-second calibration measured 9.87, 15.81 and 23.55 completed pilot trials/s
with one, two and four workers per GPU respectively. It selected four per GPU,
16 total. The final rate is 2.39 times the one-worker-per-GPU pilot rate. These
are warmed pilot-return measurements, not full-campaign or solution-quality
speedups. Calibration took 33.17 seconds. The smoke took 120.72 seconds including
calibration, before the final independent-scorer cache was added.

The 50 ms smoke used the final scorer and finished in 23.22 seconds. Its
`no_in_budget_candidate` trials were: altermagnet (1), dynamical geometry (3),
geometric (3), phonon exchange (2), supersymmetric (2). Mandatory preparation
and first completed captures can exceed 50 ms, especially with shared GPUs.
This is recorded explicitly; budgets were not extended and algorithms were not
changed. These are observed limitations on this configuration, not universal
incompatibility claims.

Telemetry mapped all four logical CUDA devices by UUID, with no telemetry
errors. Each GPU reached 99% sampled activity during the one-second smoke;
this is a peak observation, not sustained activity or peak arithmetic throughput.
The recorded maximum device memory use was approximately 1.56–1.58 GiB per GPU.

## Scoring, memory and data checks

Replayed all **97,395 captured candidates** from both smokes through the final
independent scorer. Every saved metric, score and deadline decision matched.
Separately recomputed all **173 available best scores** using dense
`offset + x.T @ Q @ x`; all matched, and all credited captures met their budgets.
The one-second smoke's 88,524 captures replayed in 6.06 seconds; its original
in-run audit total was 52.57 seconds. Different contention conditions prevent
treating that comparison as a controlled end-to-end speedup.

A controlled local CPU check on 1,000 real `gka1e` candidate vectors compared
the original scorer with the cache: 0.2373 → 0.01009 seconds, 23.5 times faster,
all scores exactly equal. Added immutable cache: 16,992 bytes for this instance.

A local storage microbenchmark (40 trials, 32 1,000-bit solutions each) measured
0.9782 → 0.2966 seconds for repeated full rewrites versus incremental appends
plus a final rebuild. Retained Python data decreased from 3,538,381 to 559,167
bytes (84.2%); traced peak decreased from 3,768,327 to 936,507 bytes. These are
fixture measurements, not GPU process memory or full-campaign estimates.

```bash
python -m qubo_benchmark validate
```

Result: **37/37 passed**, exit 0. All required data remain offline-ready:

| Variables | Sparse ready | Dense ready |
|---|---:|---:|
| 200 | 3 | 10 |
| 500 | 10 | 2 |
| 1000 | 10 | 2 |

Both completed smoke directories were resumed without adding or modifying an
attempt. The final smoke resume retained exit 1 because its 11 unavailable timed
scores remain scientific failures; successful resume does not erase them.
Aggregation of the final smoke passed: one experiment, 92 attempts, exit 0.
No benchmark or spawned worker processes remained after validation.

## Running the new version

From the repository with its environment activated:

```bash
# All matching instances, all 23 solvers, seeds 0–99, every visible GPU.
python run_benchmark.py 200 sparse 1 --require-gpu

# Optional explicit concurrency: four workers per GPU (16 on this host).
python run_benchmark.py 200 sparse 1 --require-gpu --seed-workers 4

# Optional one worker per GPU, without calibration.
python run_benchmark.py 200 sparse 1 --require-gpu --no-autotune

python monitor_benchmark.py results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY

# All 36 configurations; documented only, not executed during implementation.
python run_benchmark_grid.py --require-gpu
```

A dry run confirmed **23 × 3 × 100 = 6,900 trials** for the normal 200 sparse
configuration. Only bounded smoke tests were executed.

Output names include device and total worker counts, for example
`results/n200_sparse_t1p000s_all_g4_w16_persistent_<UTC>_<id>/`.
See the benchmark README for the full tree. Additional files include
`autotune.json`, `hardware_telemetry.csv`, `worker_capacity.json` and worker
shutdown records. Each solver has `runs.csv`, `solutions.csv`, `trace.csv`,
`summary.csv` and `errors.log`; authoritative trial records are in `attempts/`.

Schema 4 cannot resume older source/protocol identities. The earlier one-second
smoke is also intentionally historical after the scorer optimization. Current
code and final-smoke source hash:
`4a596424181363a9045d2728c75824459e4ac5ca9eba60eb16a83519db46e6ef`.

Remote evidence is under `benchmark_results/multigpu_upgrade/`: baseline and
worker-fix logs, `final_tests.log`, `offline_validation.log`, `auto_all23/`,
`deadline_50ms/`, `acceptance.json`, `final_resume.log`, `normal_dry_run.log`
and `deadline_analysis/`. Large generated results are excluded from Git.

The default concurrency ceiling is deliberately bounded at four per GPU;
`--max-auto-workers N` permits a wider powers-of-two search within admission
limits. Calibration is not a global optimum and cannot guarantee peak TFLOPS
for these small matrices. Larger instances, all six budgets and the full
production campaign have not been performance-tested by this change.

## Skills and attribution

Used the local `memory-optimization` and `optimize-for-gpu` skills. The latter's
required software attribution is retained in the benchmark README and here:
Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026),
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*,
[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).

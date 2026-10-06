# Fixed wall-clock QUBO benchmark

## New compact continuous 20-second mode

The named-argument interface selects one public input per size/category and
records nine energies during one uninterrupted run per seed. It reports only
mean signed raw gap, exact/BKS hit probability, and TTS99. Full definitions,
storage/resume semantics and timing caveats are in
[CONTINUOUS_BENCHMARK_DEFINITIONS.txt](CONTINUOUS_BENCHMARK_DEFINITIONS.txt);
all twelve references are in
[CONTINUOUS_INPUT_REFERENCES.txt](CONTINUOUS_INPUT_REFERENCES.txt).

```bash
python run_benchmark.py --variable-count 200 --sparse --dry-run --json
python run_benchmark.py --continuous --list-instances --json
# Bounded smoke: ONE seed, ONE solver, not the full campaign.
python run_benchmark.py --variable-count 200 --sparse --runs 1 \
  --config configs/benchmark_solvers_selected_20s.json \
  --solvers lib_exchange_cascade --seed-workers 1 --device cuda:0 --require-gpu --output results/smoke
python run_benchmark.py --resume results/smoke

# Use measured, quality-aware workers instead of one per GPU.
# Pilot seeds are excluded from production results; default ceiling is four.
python run_benchmark.py --variable-count 500 --dense --seed-workers auto \
  --config configs/benchmark_solvers_selected_20s.json --require-gpu --dry-run

# Post-run exports: threshold hit counts and recorded per-seed optimum times.
python tools/postprocess_continuous_results.py results/smoke --hit-percent 1
```

Default seeds are 0–99. Verified OPTIMUM hits stop immediately after completed
independent scoring; their energy is carried into later checkpoints and the
first observed time is atomically saved per seed in the progress record. BKS
matches/improvements do not trigger stopping. Configured natural solver stops
remain unchanged. Raw energy retention is now required for retrospective1% hits;
`--no-keep-raw-results` is rejected by this new CLI (historical artifacts remain).
Results use
`raw/<solver>/<problem>.npy`, `aggregated/<solver>/<problem>.csv`, and exactly
three PNGs under `plots/<problem>/`. Transfer both existing `benchmark_data/qubo37`
and new `benchmark_data/continuous12` once; inputs are not copied into results.
The historical positional interface documented below is preserved, but is a
different timing/metric/storage protocol; do not mix its experiments with this one.
The selected preset is exploratory (9 held-out mean-gap improvements, 6 ties,
3 regressions); see [CONTINUOUS_TUNING_REPORT.txt](CONTINUOUS_TUNING_REPORT.txt).
Without `--config`, the initial `benchmark_solvers_20s.json` profile is used.
`--seed-workers 1` is the default; `auto` measures1..4 workers/GPU with paired
full-budget pilot seeds and fresh holdout, or use an explicit admitted integer.
`--gpu-workers` and `--workers-per-gpu` are aliases. Automatic calibration is
per solver/input/GPU, checks host/VRAM headroom and requires >=10% throughput
gain without observed paired-incumbent/exact/1%-hit degradation. It cannot prove
unchanged quality on all future seeds. Inconclusive results retain one worker.
`--calibration-only` freezes pilot evidence without starting production seeds.
Resume restores frozen counts without recalibration; old compact schema1 cannot
resume with schema2 but remains readable for post-processing.
Percentage progress is printed after every durably committed seed.
See [EARLY_STOP_AND_WORKERS.txt](EARLY_STOP_AND_WORKERS.txt) for precise semantics,
options, memory assumptions, deployment paths and measured validation.

On the existing Vast instance, use the deployed checkout and its CUDA environment:

```bash
cd /workspace/margin_testing_continuous_release
source /workspace/margin_testing/.venv/bin/activate
# Runs all 23 solvers with 100 seeds on the selected single input, all visible GPUs.
# You start this campaign; installation/validation does not launch it.
python run_benchmark.py --variable-count 200 --sparse --seed-workers 1 \
  --config configs/benchmark_solvers_selected_20s.json --require-gpu \
  --output benchmark_results/my_200_sparse
```

Change `--seed-workers 1` to `auto` for measured calibration or to an explicit
admitted integer. Explicit sharing bypasses quality calibration; successful
four-worker smoke execution does not establish equal solution quality. Exchange
benchmark graphs reuse a worker-owned capture stream to bound BLAS workspace
caching, while graphs and optimization state remain fresh for every seed.

### Fresh-clone continuous deployment

Clone the current branch and recover the checksum-verified public inputs before
running a continuous campaign:

```bash
git clone --depth 1 --single-branch --branch solver_testing4 https://github.com/Joxy97/margin_testing.git
cd margin_testing
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark,dev]'
python -m pip check
python tools/stage_continuous_inputs.py --catalog configs/continuous_catalog.windows.json
python run_benchmark.py --variable-count 200 --sparse --seed-workers 1 \
  --catalog benchmark_data/continuous12/staged_catalog.json \
  --config configs/benchmark_solvers_selected_20s.json --require-gpu --dry-run
```

The explicit frozen catalog is required during recovery because the fingerprint
manifest binds that catalog. Always run fresh-clone campaigns with the generated
`benchmark_data/continuous12/staged_catalog.json`. The original prepared Vast
checkout instead uses its already verified normalized inputs and default Linux
catalog without restaging.

## Historical fixed-budget protocol

The remainder of this section documents the inherited positional protocol. Run
from the repository root:

```bash
python run_benchmark.py 200 sparse 0.05
```

This selects **all three** validated 200-variable sparse instances, all 23
eligible library solvers, and seeds 0 through 99 for every instance/solver:
6,900 trials, distributed across **all visible CUDA GPUs** in one invocation.
Persistent workers are reused, and a bounded pilot chooses concurrency before
the first benchmark trial. No `--instances all` flag is necessary. The other group counts
are 10, 10, 2, 10 and 2, respectively, for 200 dense, 500 sparse, 500 dense,
1000 sparse and 1000 dense. All six groups contain 37 original problems.

Allowed budgets are exactly 0.05, 0.1, 0.5, 1, 5 and 10 seconds. Equivalent
decimal spellings are accepted; other values are rejected. A solve budget is
not command duration: process startup, input validation, audit and persistence
are separately recorded overhead.

The supplied [BENCHMARK_DEFINITIONS.txt](BENCHMARK_DEFINITIONS.txt) is preserved
verbatim and copied into every experiment. Its metric definitions remain in
force. The schema-4 device scheduling and concurrency policy described here and
saved in `experiment.json` supersede the file's historical schema-3 worker
extension. The old `python -m qubo_benchmark run` interface remains available for
historical experiments; it does **not** implement this 100-seed runtime protocol.

## Historical installation and offline data

Use an environment with the NVIDIA driver and a compatible CUDA-enabled Torch
wheel already installed on the GPU host. Use the official PyTorch installation
selector for that host; no local CUDA compiler is needed by these solvers.
Do not replace a working CUDA Torch installation with the CPU-only wheel used
for the local validation. Then install this checkout's declared dependencies:

```bash
python -m pip install -e '.[benchmark,dev]'
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m qubo_benchmark download --offline
python -m qubo_benchmark validate
python run_benchmark.py --list-solvers
python run_benchmark.py --list-instances
python run_benchmark.py 200 sparse 0.05 --device cuda:0 --require-gpu --dry-run
```

The **datasets require no downloads** on the future host. Include the entire
`benchmark_data/qubo37` directory (raw sources, normalized NPZ/JSON files,
reference witnesses/evidence and manifest) and both catalog versions in the
checkout you transfer. Preflight verifies raw and normalized checksums and
compares independent original-source scores with canonical scores. Missing or
changed selected data fail preflight instead of reducing the suite. The v2
catalog has the same 37 instance definitions as the prepared original catalog.

Package installation is separate from dataset readiness. If the deployment
must also be offline for packages, prepare a wheelhouse on a machine with the
**same Linux architecture and Python version**, including the chosen CUDA
Torch wheel, the benchmark requirements, setuptools and pytest. Transfer it
with the repo, install requirements with `--no-index --find-links /path/to/wheels`,
then install this repo with `pip install --no-build-isolation --no-deps -e .`.
Windows wheels from the local development environment cannot provision Linux.
This change does not include a Linux/CUDA wheelhouse.

## Solver scope and frozen settings

The library contains 28 canonical methods. This runtime campaign intentionally
selects the **23 GPU-capable methods compatible with every catalog instance**:

```text
lib_simulated_annealing        lib_greedy_local_search
lib_spin_vector_langevin      lib_angular_annealing
lib_transverse_route           lib_easy_axis_annealing
lib_spin_coherent_annealing    lib_vector_amplitude_annealing
lib_mean_field_annealing       lib_tap_annealing
lib_spherical_annealing        lib_contact_annealing
lib_replica_annealing          lib_heat_bath_annealing
lib_tabu_search                lib_random_search
lib_exchange_cascade          lib_simulated_bifurcation
lib_altermagnet                lib_dynamical_geometry
lib_geometric                 lib_phonon_exchange
lib_supersymmetric
```

The inventory also records the five excluded methods: `lib_categorical` and
`lib_categorical_trf` require complete one-hot groups; `lib_planar_graph` is
CPU-only and requires planar zero-field structure; the tree solver and tree
sampler exceed their configured width/table limits on this catalog. Their
algorithms remain in the library. No artificial groups, padding of the supplied
problems, alternative problems, or test-only optimizers are added to the suite.
Gurobi and CPLEX are not library entry points in this checkout.

`configs/benchmark_solvers.json` contains **all resolved settings**, derived
from the existing full-compatible configuration and actual library defaults.
It freezes float32, 16 internal trajectories and the existing 1,000-step/sweep
schedules; the native batch size is eight. Other settings are method-specific
and are recorded without retuning. Optional postprocessing is disabled in this
snapshot. External seeds and wall-clock budgets are excluded from the algorithm
parameter hash and included in each logical run identity. A trajectory batch
is one trial; its duration is never divided by population size.

`configs/benchmark_protocol.json` freezes the seed protocol, representative
mapping, CPU/BLAS thread limit of one, disabled TF32, observation policy, memory
sampling and watchdog grace. Algorithm schedules evolve as originally defined;
they are not rescaled for a shorter budget. Natural early termination is kept.
Intrinsically dimension-dependent internal representation choices remain native
to the algorithm; their fixed policy and actual matrix shape/storage are saved
in `resolved_structure_json`. The source problem and scoring dimension remain
unchanged.

`--device auto` chooses every visible CUDA GPU, otherwise explicitly records CPU.
`--require-gpu` forbids CPU fallback. `--device cuda:0` selects one GPU;
`--devices cuda:0,cuda:2` selects a subset using the process's visible CUDA
indices. Every selected GPU's identity is recorded. A missing eligible
dependency/configuration is visible in the inventory and makes the campaign
incomplete; there is no silent fallback, parameter reduction or mid-campaign
OOM retuning.

## Timing, scoring and observation

By default, trials reuse persistent spawned workers assigned to specific GPUs.
An initial readiness barrier completes all worker startup before any timed
trial begins. Thereafter each ready worker independently takes its next trial;
there is no repeated global wave barrier.
Only immutable input and infrastructure survive between trials; every seed gets
a new solver object and reset RNG/state. Imports,
including lazy SciPy linear-algebra/sparse dependencies,
problem loading, canonical native-tensor conversion/transfer, context creation
and an 8x8 zero-GEMM infrastructure warmup precede the solve clock. No useful
instance optimization, target, reference witness or earlier seed is used in
warmup. Specialized compact backends retain their conversion and algorithmic
preprocessing inside the timed section; their transfer time is not separately
isolated. Native immutable device matrices are reused on a cache hit; the cache
holds only the current input per worker. Static worker-ready overhead is saved
in `warmups.csv`, with worker ID/PID, reuse and input-cache-hit fields.
`--worker-mode fresh` retains a new process per seed for comparison.

The monotonic nanosecond clock starts before RNG reset and algorithm state
initialization. Python, NumPy and Torch RNGs are seeded, including local
generators used by the original algorithms. Mutable state is never continued
between budgets or seeds. `initialization_hash`, when available, hashes the
first observed binary initialization batch, not an unobserved continuous state.

Native Torch methods report their initial candidates and existing record-step
incumbents; SBM reports decoded candidates each integration step; physics
methods report their native candidate-accumulator improvements. Identical
bitstrings are deduplicated except the final-return diagnostic. Captures clone
and synchronously copy candidates to immutable host bitstrings; timestamps are
taken **after** the copy, decoding and hash work completes. Observation and IPC
overhead stay in the timed solve. There is no inner-loop disk or console I/O.
Local iteration counters may restart between native trajectories; they are not
fabricated total evaluation counts.

Cooperative checks stop iterative work. The supervisor enforces an additional
0.5-second return grace for uninterruptible solver calls. The grace never extends the
quality deadline. A candidate captured after the deadline cannot improve the
timed result. Actual elapsed time and overshoot are never clamped. A lost GPU
worker aborts scheduling until explicit resume; no next trial is started while
that worker could still be active. Supervisor messages are drained before
checking the watchdog so a completed return queued in IPC is not mistaken for
an active solve. A separate cleanup completion message keeps state disposal
outside solver time. Run one scheduler invocation across the selected GPUs.

These mechanisms correctly account for a 50 ms budget but cannot guarantee a
candidate or a hard real-time return on every host. Mandatory matrix assembly,
factorization/calibration, first-use kernels or decoding may exceed 50 ms before
the first candidate exists. Such trials are `no_in_budget_candidate`. See
`RUNTIME_BENCHMARK_VALIDATION.md` for earlier local limitations and
[MULTIGPU_VALIDATION.md](MULTIGPU_VALIDATION.md) for the four-RTX-3090 validation:
92 focused tests passed; both bounded GPU smokes ran all 23 solvers. The 50 ms
smoke correctly recorded 11 of 92 trials without an in-budget candidate.

Every captured binary solution is independently scored in original normalized
units as `offset + x.T @ Q @ x`, with symmetric Q and exact integer arithmetic
for this catalog. Internal solver energies are diagnostics only. Source index
labels, signs, offsets and off-diagonal factors are preserved. Bitstrings are
quoted CSV text: preserve leading zeros when loading them in spreadsheets.

The signed gap is `f-R`; its percentage is `100*(f-R)/abs(R)`. At R=0 the
percentage is null and the near-target threshold is zero. The target is
one-sided, `f <= R + .01*abs(R)`. Improvements of best-known values are retained;
improvements below purported proven optima trigger quarantine/validation alarms.
References are snapshotted and never replaced by benchmark discoveries.

TTT values are **first observed completed-capture times**, upper bounds on
unobserved discovery times. No interpolation is used. Final-only output does
not manufacture an internal TTT; missing history, targets not reached, invalid
runs, and no eligible candidate have distinct reasons. Successful conditional
TTT medians include their hit counts. Standard primary success rates use the
100 planned seeds, including failures in the denominator; partial rates are
explicitly provisional. Explicit retries never replace a failed primary
attempt in the primary statistics.

CPU memory is worker-process-tree RSS sampled every 10 ms and can miss brief
peaks. GPU memory is Torch allocated/reserved baseline and allocator peak, reset
before timed work. GPU process-wide sampled memory and GPU-event elapsed time
are unavailable/null in individual trial rows. Separate device-level telemetry
records activity, total used memory and power where NVIDIA reports them; it
includes other processes and is not a per-trial allocation counter or TFLOPS
measurement. Matrix storage bytes describe the primary matrix, not all
solver memory. Preprocessing duration and total evaluation counts are not
separately instrumented; their fields are null with explanatory metadata.
Each worker's memory includes cached immutable input and its retained allocator
baseline. These counters exclude allocations made by the other seed workers.

`runs.csv` records pre-serialization end-to-end time. `finalization.csv`, joined
by attempt_id, records measured serialization time and the full trial duration
through journal/CSV persistence, including worker preparation, readiness waiting,
solve, cleanup, result-queue waiting and independent audit. It excludes initial
campaign preflight, writing that finalization record and the subsequent progress
display. Final
persistent process shutdown is separately recorded in `worker_sessions/*.json`.
Concurrent trial durations overlap; do not sum them as campaign elapsed time.
A crash after the durable trial but before
finalization leaves the overhead measurement unavailable, not guessed. Immutable
trial JSON is committed first and new CSV rows are appended after every trial;
the complete historical CSVs are not rewritten on every completion. Summaries
refresh approximately every five seconds and at completion. Startup/resume and
finalization rebuild projections from the authoritative journal. The scheduler
bounds the result backlog and keeps compact trial metadata in memory; historical
solution/trace payloads are read individually when rebuilding exports.

## Multiple GPUs, parallel seeds and worker reuse

The normal command selects every matching instance, all 23 available eligible
solvers and seeds 0-99. It discovers all visible GPUs and writes **one complete
experiment directory** across those GPUs. Separate shell invocations or manual
25-seed partitions are unnecessary.

```bash
# All visible GPUs; measured concurrency, with a default ceiling of four per GPU.
python run_benchmark.py 200 sparse 1 --require-gpu

# All visible GPUs; one persistent worker per GPU, no pilot calibration.
python run_benchmark.py 200 sparse 1 --require-gpu --no-autotune

# Two workers on each selected GPU: four concurrent workers total.
python run_benchmark.py 200 sparse 1 --devices cuda:0,cuda:2 --seed-workers 2 --require-gpu

# Inspect selection and admission without executing calibration or solvers.
python run_benchmark.py 200 sparse 1 --require-gpu --dry-run
```

`--seed-workers auto` is the default. On CUDA, the runner tests one, two and four
workers per GPU, limited by available seeds, host/VRAM admission and the default
`--max-auto-workers 4`. The pilot samples up to two selected solvers, preferring
greedy search and simulated bifurcation, on the selected instance with the most
couplings. It first warms workers, then measures completed trials per second.
Pilot budgets are bounded to 0.1-0.5 seconds. Pilot seeds are outside the campaign
schedule and pilot outcomes never enter benchmark quality statistics.

A higher worker count is accepted only if the measured throughput improves by
more than 10% without worker failures. Calibration stops at the first failed or
unhelpful count and retains the last accepted count. A failing single-worker
baseline stops preflight. This is a bounded sample, not proof of the global
optimum or performance for every solver. Candidate quality is not used to tune
concurrency. Solver parameters, internal populations and mathematical updates
are never changed. CPU fallback starts with one worker and skips GPU calibration.

`--seed-workers N` explicitly requests 1-100 workers **per selected device** and
skips calibration. On four GPUs, `--seed-workers 4` requests 16 workers total.
`--no-autotune` with auto workers keeps one per device. `--max-auto-workers N`
changes the pilot ceiling; powers of two up to that ceiling are considered.
The chosen counts are frozen before creating the experiment and restored on
resume without new pilot trials. A dry run reports initial admission only.

`worker_capacity.json` records per-device and aggregate admission estimates.
They combine the selected algorithms' working-memory estimates, free RAM/VRAM,
a 1 GiB per-process host/GPU reserve, and an 80% memory allowance. Host memory
is shared across all GPUs. CPU affinity and Linux cgroup CPU/memory limits are
checked, rather than treating the physical host's capacity as fully available.
At least one CPU worker slot is allowed per selected GPU; this can still share
a small container CPU quota. The estimate does not guarantee that allocations
fit or establish an optimal count. Actual context size and external load vary.

Workers retain their GPU assignment and cache only immutable input. A shared
queue interleaves solver/instance groups, so different solvers and independent
seeds can run at the same time. Workers prefer their current group to reuse
inputs and take remaining seeds when other groups finish. Every logical
solver/instance/seed trial is scheduled once. There is one startup readiness
barrier; afterward workers need not wait for a slower peer before taking their
next trial. Each process has independent RNGs, solver state, a deadline and a
watchdog. The result queue is bounded; workers can continue while completed
trials are independently scored and durably saved.

The time limit remains **per seed's solver wall time**: reset, algorithm setup,
search and completed candidate capture. It excludes process startup, readiness,
scoring, cleanup and writing results. Each simultaneous seed receives its own
full budget, but CPU/GPU contention within that interval counts. Increasing
concurrency may reduce the search each seed completes in that budget. Late
captures receive no timed quality credit, and actual overshoot is reported.

Execution modes distinguish these comparisons:

| Mode | Meaning |
|---|---|
| `isolated_latency` | One worker on one selected device. |
| `multi_gpu_isolated` | One worker per GPU; GPUs run different trials, with shared host resources. |
| `parallel_shared_device` | At least one GPU/device has multiple workers sharing its capacity. |

These modes describe this runner's own workers, not exclusive ownership of the
host. Do not pool their timing/quality distributions as if resource allocation
were identical. Device identities, per-device worker counts and execution mode
are recorded in resume identity and CSV metadata. The directory name includes
`_g4_w8_persistent_`, for example, for four devices and eight total workers.
Changing the selected devices or counts starts a new experiment.

Separate CUDA processes may time-share instead of overlapping GPU kernels.
[NVIDIA MPS](https://docs.nvidia.com/deploy/mps/latest/index.html) can improve
overlap on supported configurations; this runner does not configure host MPS.
Small QUBOs may remain limited by launches, synchronization or CPU work. GPU
activity is not achieved TFLOPS, and full theoretical utilization is not promised.
GPU speedup must be measured on the target hardware; CPU tests cannot establish
it. Use one scheduler invocation for the selected GPUs. Historical schema-2/3
results remain preserved artifacts, but strict source/protocol identity prevents
resuming them with this schema-4 runner.

## Run, monitor and resume

```bash
python run_benchmark.py --list-solvers
python run_benchmark.py 200 sparse 0.05 --dry-run
python run_benchmark.py 200 sparse 0.05
python run_benchmark.py 500 dense 1.0
python run_benchmark.py 1000 sparse 10.0
python run_benchmark.py 200 dense 0.1 --instances all
python run_benchmark.py 1000 dense 0.5 --instance-id p_hat1000-2__clique_QUBO
python monitor_benchmark.py results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py 200 sparse 0.05 --resume results/ACTUAL_TEST_DIRECTORY
```

Use the exact directory printed after `Output:` in place of
`ACTUAL_TEST_DIRECTORY`. Monitoring is read-only; add `--once` for one snapshot.
Progress includes solver/instance/seed position, overall completion, latest/best
gap from committed trials, and provisional hit counts. Heartbeats continue
through worker setup and solving; current-trial gaps appear after independent
post-run scoring. `status.json` also records active solver/instance/seed/device
assignments and the latest `hardware_telemetry` snapshot. The background sampler
uses `nvidia-smi` approximately every two seconds, matching device UUIDs where
available. Missing/unsupported telemetry remains null with a reason and does
not stop trials. Device activity includes other processes and does not measure
achieved TFLOPS. The monitor reads saved status without creating a CUDA context.

Resume restores the original solver/seed/configuration selection automatically:

```bash
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY --retry-failed
```

It also restores the selected GPU inventory and frozen worker counts; calibration
does not run again. Normal execution, monitoring and resume are therefore:

```bash
python run_benchmark.py 200 sparse 0.05
python monitor_benchmark.py results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY
```

Resume rejects different source content/commit, data, references, hardware,
dependencies, precision, seeds, settings and protocol. Do not edit Python source
during a campaign; changes stop scheduling. A live-process lock prevents two
writers; a stale local dead-PID lock can be recovered. Foreign-host locks fail
explicitly. A trial interrupted before terminal completion is rerun from its
original seed under a new attempt ID. Completed attempts are immutable.
`--retry-failed` adds diagnostic attempts and never erases failures. A checksum
protected atomic JSON journal is authoritative; interrupted/truncated CSV
exports are detected and rebuilt on resume. Temporary uncommitted files are
not treated as completed trials.

Exit codes: 0 completed with valid timed scores; 1 completed with recorded
failures or unavailable eligible solvers; 2 invalid preflight/resume; 130
interrupted or lost-GPU-worker stop. A reference target need not be reached for
a valid timed score.

## Output tree

```text
results/
  n200_sparse_t0p050s_all_g4_w8_persistent_<UTC>_<unique-id>/
    BENCHMARK_DEFINITIONS.txt
    experiment.json                 # frozen identity and complete plan
    experiment_metadata.csv
    environment.json
    environment.csv
    worker_capacity.json            # per-device and aggregate memory/CPU admission
    autotune.json                   # excluded pilot evidence and chosen counts, or skip reason
    hardware_telemetry.csv           # sampled device activity/memory/power, not TFLOPS
    reference_snapshot.json
    solver_configuration.json
    solver_registry.csv             # all 28, including exclusion reasons
    instances.csv
    seeds.json
    seed_schedule.csv
    execution_plan.csv
    status.json
    progress.log
    summary.csv                     # per instance/solver; refreshed about every 5 s and at completion
    warmups.csv
    finalization.csv
    attempts/<run-id>__0001.json     # durable authoritative records
    finalization/<attempt-id>.json
    worker_sessions/<session-id>.json  # persistent process exit/shutdown records
    solvers/<solver-id>/
      parameters.json
      parameters.csv
      runs.csv
      solutions.csv
      trace.csv
      summary.csv
      errors.log
```

`.lock` exists only while writing; `csv_recovery.log` appears when needed.
The example directory uses four devices and eight workers total; the actual
counts depend on selection and calibration. Runtime result schema version is 4
(solver-parameter/catalog version remains 2).
CSV nulls are empty fields, not zero. Algorithm parameters
use SHA-256 of sorted compact canonical JSON; source/data/reference/execution
identities are separately recorded. Raw and best-in-budget vectors are linked
to trace events and retain independent objectives and capture times. No polished
result is synthesized when the configured method did not produce one.

## Grid, aggregation and disconnected sessions

```bash
python run_benchmark_grid.py --instances representative --dry-run
python run_benchmark_grid.py --instances representative
python run_benchmark_grid.py --require-gpu
python run_benchmark_grid.py --require-gpu --resume results/campaign_ACTUAL_ID
python aggregate_benchmarks.py results --output analysis
```

The grid has 36 sequential configurations and calls the same public runner;
within each configuration all selected GPUs run concurrent solver/seed work.
Its default is all instances and 100 seeds: 510,600 trials with 23 solvers.
Representative mode is explicitly limited coverage. The campaign manifest is
immutable; resume requires the same grid arguments and environment. Child
directories are `campaign_<id>/configuration_00/<test-directory>` through
`configuration_35/`. Trial failures remain recorded and do not skip later grid
configurations; preflight failures or interruption stop the grid. Auto calibration
is performed separately for each new configuration, so worker counts may differ
between sizes/densities/budgets while mathematical solver parameters stay fixed.
Each resumed child restores its saved counts. The full grid
has **not** been executed during implementation.

Aggregation writes `runs.csv`, `solutions.csv`, `trace.csv`, `summary.csv`,
`quality.csv`, `timing.csv`, `memory.csv` and `aggregation.json`. It preserves
instance, hardware and parameter identities, failures, null TTTs and censoring.
Identical copied experiments are detected; conflicting copies fail. Partial
experiments remain partial. It makes no pooled scaling or group-average claim.

For an optional Linux session that survives SSH disconnection:

```bash
tmux new -s qubo-benchmark
python run_benchmark.py 200 sparse 0.05 --require-gpu
# Detach with Ctrl-b, then d; reconnect with: tmux attach -t qubo-benchmark
tar -czf qubo-results.tar.gz results
# From your own computer, copy from your existing host:
scp USER@HOST:/path/to/qubo-results.tar.gz .
```

Copy and verify the archive before separately managing the remote machine.
None of these benchmark scripts manages cloud machines, billing or credentials.

## Tests and bounded smoke

```bash
python -m pytest tests/test_runtime_benchmark.py tests/test_runtime_workers.py tests/qubo_solvers tests/test_qubo_benchmark.py -q
python run_benchmark.py 200 sparse 0.05 --instances representative --runs 1 --device cpu --output-root benchmark_results/runtime_smoke
```

For a focused worker/timing check without the full suite or full campaign:

```bash
python -m pytest tests/test_runtime_workers.py tests/test_runtime_benchmark.py tests/test_runtime_devices.py -q
python run_benchmark.py 200 sparse 0.05 --instances representative --runs 4 --solvers lib_random_search --no-autotune --require-gpu --output-root benchmark_results/multi_gpu_smoke
```

This smoke invokes only random search on one representative, using all visible
GPUs and four seeds total. Remove `--solvers lib_random_search` to check every
eligible method, or select another explicit method for a focused diagnostic.
Repeating with other size/density pairs changes the representative. It is labeled `NONSTANDARD SMOKE` and
is not a 100-seed scientific result. On NVIDIA hardware, use `--require-gpu`
and run the same tests; locally skipped CUDA tests must then run.
For a method without a 50 ms candidate, a separately labeled longer-budget
smoke can verify scoring without altering the 50 ms result.

The reproducible local acceptance helper is
`python tools/run_runtime_v2_smoke.py`; it runs exactly those six
CPU commands sequentially. `python tools/audit_runtime_v2_smoke.py`
checks the resulting 138 attempts, frozen source/configuration identities,
capture deadlines and independently reconstructs every saved best score.

## Measurement data dictionary

| Field | Meaning and inclusion |
|---|---|
| `static_setup_s` | Parent-observed preparation through ready; first use includes startup, imports, loading, native transfer and warmup. Reused cache hits skip them. |
| `actual_device`, `workers_on_device`, `seed_workers` | Device executing this trial, frozen workers on that device, and total workers across the scheduler. |
| `execution_mode`, `execution_policy_hash` | Resource-sharing label and frozen execution policy; preserve when comparing or aggregating results. |
| `transfer_s` | Native canonical tensor construction, device copy and synchronization; subset of static setup, not a pure PCIe measurement. Null for compact backends. |
| `warmups.csv:warmup_s` | Zero-GEMM warmup; subset of setup. |
| `trial_reset_init_s` | Timed prefix through global RNG reset. Algorithm-specific state initialization follows inside solve time. |
| `algorithm_preprocess_s` | Null: not separately isolated; all such work is included in solve time. |
| `actual_solve_wall_s` | Complete timed section including reset, algorithm setup, search, observation, final capture and final synchronization. Watchdog rows use supervisor elapsed time. |
| `post_budget_return_s` | Elapsed time from solve origin at an over-budget return; not an additional duration. |
| `overshoot_s` | `max(0, actual_solve_wall_s - requested_time_s)`; a derived subset, never added to solve time. |
| `cleanup_s` | Worker state cleanup outside solve time, plus join/termination if retired. Final warm-pool shutdown is in worker_sessions. |
| `audit_s` | Loading the canonical evaluator and independently scoring all captured candidates. |
| `serialization_s` | Null in the immutable attempt row; measured in joined `finalization.csv`. |
| `trial_end_to_end_s` | Inclusive overall duration; pre-serialization in runs, finalized through persistence in finalization. Do not add the constituent fields to it. |
| `observation_resolution_s` | Null: observations occur at algorithm checkpoints, without a fixed temporal sampling interval. Timestamps use `perf_counter_ns` units. |
| `iterations` | Last observed native local iteration where supplied; null when a backend does not report it. |
| `evaluations`, `completed_work_units` | Null: no common validated counter across these algorithms. |
| `replicas` | Configured replica count where defined; null when not applicable. |
| `gpu_elapsed_s`, `gpu_process_peak_sampled_bytes`, `gpu_memory_sampling_interval_s` | Null: GPU event timing and process-wide GPU sampling are not implemented; allocator counters are the separate supported measurement. |
| `postprocessed_objective` | Null: no additional benchmark polishing is performed. |
| `initial_solution_id`, `initialization_hash` | Null if the native reporting interface does not expose initial state; no invented initial vector. |

Static setup, solve, cleanup and audit are distinct phases, but their sum omits
parent coordination/row construction overhead. Use finalized end-to-end time
instead of assuming their sum is complete. Warmup, transfer, reset and overshoot
overlap the enclosing phases and must not be added again. Solution SHA-256 uses
the uninterrupted ASCII 0/1 string, without a newline. The optional initialization
hash uses UTF-8 encoding of Python `repr` of the first observed binary batch.
`measured_density` is a percentage of nonzero unordered pairs among N*(N-1)/2.

GPU observation guidance used the local optimize-for-gpu skill. Attribution:
Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026),
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*,
https://doi.org/10.48550/arXiv.2609.00065. The local memory-optimization skill also
guided bounded queues and streamed result storage. Actual GPU evidence and its
limits are recorded in [MULTIGPU_VALIDATION.md](MULTIGPU_VALIDATION.md).

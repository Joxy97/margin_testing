# Fixed wall-clock QUBO benchmark

Run from `solvers_testing/margin_testing`:

```bash
python run_benchmark.py 200 sparse 0.05
```

This selects **all three** validated 200-variable sparse instances, all 23
eligible library solvers, and seeds 0 through 99 for every instance/solver:
6,900 trials. No `--instances all` flag is necessary. The other group counts
are 10, 10, 2, 10 and 2, respectively, for 200 dense, 500 sparse, 500 dense,
1000 sparse and 1000 dense. All six groups contain 37 original problems.

Allowed budgets are exactly 0.05, 0.1, 0.5, 1, 5 and 10 seconds. Equivalent
decimal spellings are accepted; other values are rejected. A solve budget is
not command duration: process startup, input validation, audit and persistence
are separately recorded overhead.

The supplied [BENCHMARK_DEFINITIONS.txt](BENCHMARK_DEFINITIONS.txt) is preserved
verbatim and copied into every experiment. This README describes its execution
details. The old `python -m qubo_benchmark run` interface remains available for
historical experiments; it does **not** implement this 100-seed runtime protocol.

## Installation and offline data

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

`--device auto` chooses CUDA when available, otherwise explicitly records CPU.
`--device cuda:0 --require-gpu` fails if CUDA is unavailable. A missing eligible
dependency/configuration is visible in the inventory and makes the campaign
incomplete; there is no silent fallback, parameter reduction or OOM retuning.

## Timing, scoring and observation

By default, trials reuse a persistent spawned worker with a readiness handshake.
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
0.5-second cleanup grace for uninterruptible calls. The grace never extends the
quality deadline. A candidate captured after the deadline cannot improve the
timed result. Actual elapsed time and overshoot are never clamped. A lost GPU
worker aborts scheduling until explicit resume; no next trial is started while
that worker could still be active. Run only one benchmark invocation per GPU.

These mechanisms correctly account for a 50 ms budget but cannot guarantee a
candidate or a hard real-time return on every host. Mandatory matrix assembly,
factorization/calibration, first-use kernels or decoding may exceed 50 ms before
the first candidate exists. Such trials are `no_in_budget_candidate`. See
`RUNTIME_BENCHMARK_VALIDATION.md` for observed local limitations. GPU timing
remains unverified until the CUDA tests and smoke run pass on actual hardware.

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
are unavailable/null. Matrix storage bytes describe the primary matrix, not all
solver memory. Preprocessing duration and total evaluation counts are not
separately instrumented; their fields are null with explanatory metadata.
Each worker's memory includes cached immutable input and its retained allocator
baseline. These counters exclude allocations made by the other seed workers.

`runs.csv` records pre-serialization end-to-end time. `finalization.csv`, joined
by attempt_id, records measured serialization time and the full trial duration
through journal/CSV persistence, including worker preparation, readiness waiting,
solve, cleanup and independent audit. It excludes the pre-wave source check,
writing that finalization record, and the subsequent progress display. Final
persistent process shutdown is separately recorded in `worker_sessions/*.json`.
Concurrent trial durations overlap; do not sum them as campaign elapsed time.
A crash after the durable trial but before
finalization leaves the overhead measurement unavailable, not guessed.

## Parallel seeds and worker reuse

The ordinary command still selects every matching instance, all 23 available
eligible solvers, and seeds 0-99. It now reuses one warm worker by default.
To run several independent seeds at a time on the same GPU:

```bash
python run_benchmark.py 200 sparse 0.05 --device cuda:0 --require-gpu --seed-workers 4 --dry-run
python run_benchmark.py 200 sparse 0.05 --device cuda:0 --require-gpu --seed-workers 4
```

`--seed-workers` accepts 1-100. Preflight prints a memory-admitted maximum based
on the selected solvers' working-memory estimates, current free RAM/VRAM, a
1 GiB per-process host/GPU reserve and an 80% memory allowance. The saved
`worker_capacity.json` describes this estimate. It is neither a guarantee that
allocations fit nor a measurement of the fastest concurrency. Actual context
size and external GPU workloads vary. Start with a small worker count; 100 is
accepted only if the estimate admits it. No populations or solver parameters
are reduced to fit more workers.

Workers process waves of seeds for the same solver and instance. Each worker
has independent RNGs, solver state, CUDA context, deadline, and watchdog. All
workers finish input preparation before any solver clock starts in that wave.
The supervisor keeps draining other workers while each completed trial is
independently scored and atomically saved. Progress includes active seeds.

The 0.05 seconds is **per seed's solver wall time**: reset, algorithm setup,
search and completed candidate capture. It excludes Python startup, readiness
waiting, scoring and result writing. Each of four simultaneous seeds gets its
own 0.05-second budget. Device/CPU contention within that interval still counts.
Cooperative checks cannot guarantee an exact hard return at 50 ms; late
candidates receive no timed quality credit and actual overshoot is reported.

Parallel results are labeled `PARALLEL SHARED-DEVICE` / `parallel_shared_device`.
Single-worker runs are `isolated_latency`. Worker mode/count form part of result
identity, CSV metadata and resume checks; changing either starts a new result
directory. The directory name includes `_w4_persistent_`, for example. These
latency distributions must be compared separately. Resume uses the original
worker policy automatically:

```bash
python run_benchmark.py --resume results/EXACT_RESULT_DIRECTORY
python monitor_benchmark.py results/EXACT_RESULT_DIRECTORY
```

Separate CUDA processes may time-share instead of overlapping GPU kernels.
[NVIDIA MPS](https://docs.nvidia.com/deploy/mps/latest/index.html) can improve
overlap on supported configurations; this runner does not configure the host
MPS service. Multiprocess execution alone does not establish a GPU speedup.
Run only one benchmark invocation per GPU. Schema-2 result files remain
historical artifacts; strict source/protocol identity prevents resuming them
with this schema-3 runner.

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
post-run scoring.

Resume restores the original solver/seed/configuration selection automatically:

```bash
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY --retry-failed
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
  n200_sparse_t0p050s_all_w1_persistent_<UTC>_<unique-id>/
    BENCHMARK_DEFINITIONS.txt
    experiment.json                 # frozen identity and complete plan
    experiment_metadata.csv
    environment.json
    environment.csv
    worker_capacity.json            # admission estimate, not measured capacity
    reference_snapshot.json
    solver_configuration.json
    solver_registry.csv             # all 28, including exclusion reasons
    instances.csv
    seeds.json
    seed_schedule.csv
    execution_plan.csv
    status.json
    progress.log
    summary.csv                     # per instance/solver, never pooled
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
Runtime result schema version is 3 (solver-parameter/catalog version remains 2).
CSV nulls are empty fields, not zero. Algorithm parameters
use SHA-256 of sorted compact canonical JSON; source/data/reference/execution
identities are separately recorded. Raw and best-in-budget vectors are linked
to trace events and retain independent objectives and capture times. No polished
result is synthesized when the configured method did not produce one.

## Grid, aggregation and disconnected sessions

```bash
python run_benchmark_grid.py --instances representative --dry-run
python run_benchmark_grid.py --instances representative
python run_benchmark_grid.py --device cuda:0 --require-gpu
python run_benchmark_grid.py --device cuda:0 --require-gpu --resume results/campaign_ACTUAL_ID
python aggregate_benchmarks.py results --output analysis
```

The grid has 36 sequential configurations and calls the same public runner.
Its default is all instances and 100 seeds: 510,600 trials with 23 solvers.
Representative mode is explicitly limited coverage. The campaign manifest is
immutable; resume requires the same grid arguments and environment. Child
directories are `campaign_<id>/configuration_00/<test-directory>` through
`configuration_35/`. Trial failures remain recorded and do not skip later grid
configurations; preflight failures or interruption stop the grid. The full grid
has **not** been executed during implementation.

Aggregation writes `runs.csv`, `solutions.csv`, `trace.csv`, `summary.csv`,
`quality.csv`, `timing.csv`, `memory.csv` and `aggregation.json`. It preserves
instance, hardware and parameter identities, failures, null TTTs and censoring.
Identical copied experiments are detected; conflicting copies fail. Partial
experiments remain partial. It makes no pooled scaling or group-average claim.

For an optional Linux session that survives SSH disconnection:

```bash
tmux new -s qubo-benchmark
python run_benchmark.py 200 sparse 0.05 --device cuda:0 --require-gpu
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
python -m pytest tests/test_runtime_workers.py tests/test_runtime_benchmark.py -q
python run_benchmark.py 200 sparse 0.05 --instances representative --runs 4 --solvers lib_random_search --seed-workers 2 --device cuda:0 --require-gpu --output-root benchmark_results/parallel_smoke
```

Repeat that smoke command for the other five size/density pairs to invoke every
eligible method on each representative. It is labeled `NONSTANDARD SMOKE` and
is not a 100-seed scientific result. On NVIDIA hardware, use `--device cuda:0
--require-gpu` and run the same tests; locally skipped CUDA tests must then run.
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
https://doi.org/10.48550/arXiv.2609.00065. Local CPU tests establish no CUDA
speedup or timing-performance claim.

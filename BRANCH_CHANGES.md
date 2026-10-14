# solvers_testing5

Builds on the published `solver_testing4` branch at
`c7184313323b815c3539675970b3199d56938d33`.

## SU(2) and gravity solver additions

- Adds `SU2ChiralIsospin` (`lib_su2_chiral_isospin`) and `GravityTunneling`
  (`lib_gravity_tunneling`) alongside the original 28 canonical methods.
- Registers native tensor, compact and resident compact execution, benchmark
  selection, fresh per-seed state, and conservative worker memory admission.
- Uses batched Torch-native dynamics, reusable buffers, sparse CSR force paths,
  implicit rank-two chirality, and B-by-B rather than B-by-B-by-n gravity storage.
- Scores and archives original-source candidates in float64 separately from
  optional float32 dynamics, including raw duplicate/self-loop compact terms.
- Adds explicit initial presets (512 steps, 16 trajectories, cadence 16), tests,
  a bounded GPU/CPU memory/timing probe, and updated usage/clone documentation.
  Original solver mathematics and frozen presets are not retuned.
- Includes only source, required configuration, focused tests, reproducible
  tooling and documentation changes. Existing offline inputs/provenance are
  inherited; no generated results, recovered bulk data, environments or keys
  are added by this release.

Prior implementation validation: 1,549 tests and 50 subtests passed on CPU/CUDA;
three portfolio-only cases were excluded. Sparse/dense synthetic CUDA smokes
reached 10,000 variables; all 37 offline inputs validated. No full 100-seed
quality campaign was run; these exploratory classical methods do not guarantee
optimality or establish an advantage over existing solvers. Preserve the old
checkout to resume old experiments: source identity changes in this release.

## Historical solvers_testing3 release

This branch builds on `solvers_testing2` at commit
`d7dc1543962d2d151593d41d1e7510800f3cec3e`.

## Added and changed

- Automatically selects every visible CUDA GPU. `--require-gpu` forbids CPU
  fallback; `--devices cuda:0,cuda:2` selects a subset.
- Measures throughput with bounded pilot trials to select workers per GPU.
  Default candidates are 1, 2 and 4, constrained by CPU and memory admission.
  `--seed-workers N` explicitly requests N workers per selected GPU.
- Schedules different solvers, instances and independent seeds concurrently
  through persistent, GPU-affine workers that reuse immutable inputs.
- Drains queued worker messages before watchdog checks, avoiding false timeouts
  when a completed result is behind earlier captures.
- Accelerates immutable candidate encoding and exact independent scoring.
  Solver algorithms, precision and frozen mathematical parameters are unchanged.
- Commits each trial atomically and appends CSV rows incrementally. Bounded
  queues and streamed historical records reduce memory use and repeated I/O.
- Records calibration, per-device workers, GPU activity/memory/power, active
  trials and worker shutdown information.
- Adds regression tests and [GPU validation evidence](MULTIGPU_VALIDATION.md).

## Run and resume

```bash
python run_benchmark.py 200 sparse 1 --require-gpu
python monitor_benchmark.py results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY
```

The normal command selects all 23 eligible library solvers, all three matching
instances and seeds 0 through 99: 6,900 trials in one result directory across
the visible GPUs. Each seed retains its own solver-only wall-time budget.
Startup, immutable input preparation, independent scoring and persistence are
outside that budget. Shared GPU capacity still affects the work completed
within the interval.

Runtime schema is now 4. Old source/protocol identities cannot be resumed with
this version. Resume restores the recorded worker counts without recalibration.
The new release commit also changes source identity relative to pre-release
smoke artifacts. Preserve the original checkout when resuming older experiments.

## Validation and deployment contents

On four RTX 3090s: 92 automated tests and 7 subtests passed. All 37 offline
problems validated. The one-second smoke produced 92/92 valid timed results;
the 50 ms smoke produced 81/92, with 11 explicitly recorded as having no
in-budget candidate. Neither smoke lost workers. No full campaign was run.
See the validation report for commands, limitations and measured pilot throughput.

Includes the canonical 28-method library, 23-method runtime suite, configuration,
tests, documentation and all 37 offline inputs with reference provenance.
Generated results, environments, caches, credentials and unrelated market
datasets are excluded. Use the shallow single-branch clone in README.md to avoid
downloading historical Git objects. Python dependencies still require installation.

The validation report describes the earlier local/Vast implementation step;
this branch publishes that implementation with these release notes. Publishing
does not modify the existing Vast checkout or its active jobs.

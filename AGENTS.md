# Solver benchmark deployment instructions

This `solvers_testing3` branch ships the standalone canonical solver library,
benchmark runners, configuration, focused validation tests, and all 37 offline
QUBO inputs with source/reference provenance. The earlier `solvers_testing`
branch retains the broader portfolio application. Do not add market datasets,
virtual environments, generated results, caches, or credentials to this branch.

Read README.md, README_BENCHMARKS.md, BENCHMARK_DEFINITIONS.txt and
SOLVER_LIBRARY.md before changing the corresponding interfaces.

- Numerical algorithms live only in src/qubo_solvers. All benchmark adapters
  call this library. Preserve the 28 canonical IDs, including the 23 GPU-capable
  methods compatible with the complete 37-instance corpus.
- Runtime orchestration lives in src/qubo_benchmark/runtime. The normal command
  selects every matching instance and seeds 0-99. Never run the full campaign
  as an implementation test.
- Preserve mathematical updates and frozen solver settings. Time RNG reset,
  fresh algorithm state, algorithm preprocessing, search, decoding and completed
  immutable candidate capture. Exclude worker startup, immutable input setup,
  independent scoring, cleanup and persistence from the solver budget.
- Reuse infrastructure and immutable inputs only. Each seed gets fresh mutable
  state. Parallel seeds use separate spawned processes and an initial readiness
  barrier. Detect all visible GPUs by default; freeze measured concurrency before
  scheduling. Keep independent watchdogs and shared-device timing labels.
- Preserve execution-policy, source, data, environment and parameter resume
  identities. Never edit numerical source while a benchmark is active.
- Save each trial atomically; preserve failed attempts and independently score
  every captured binary solution. Never pass reference targets into solvers.
- Do not claim GPU correctness or speedup from CPU-only tests. Explicit CUDA
  requests must fail clearly if unavailable. Keep algorithms within Torch;
  no custom native CUDA build is required for current solvers.
- Use explicit validation and local imports for optional heavy dependencies.
  Match the surrounding code style. Prefer rg for searches.
- Preserve unrelated local changes. Shipping exclusions should leave unrelated
  local files intact. Clone with --depth 1 --single-branch to avoid old history.

Install from this checkout: python -m pip install -e ".[benchmark,dev]".
Focused orchestration checks:

    python -m pytest tests/test_runtime_workers.py tests/test_runtime_benchmark.py -q

Multi-GPU orchestration checks also include tests/test_runtime_devices.py,
tests/test_runtime_scheduler.py, tests/test_runtime_incremental.py and
tests/test_runtime_telemetry.py. Keep result queues and scorer caches bounded;
commit each trial before appending CSV projections. Schema-4 resume restores
the original selected devices and calibrated worker counts without retuning.

For numerical changes, use tests/qubo_solvers and tests/test_qubo_benchmark.py
on CPU and the target GPU. The standalone checkout excludes tests requiring
portfolio application shims. Use python -m qubo_benchmark validate to check
all offline inputs and available witnesses. A bounded smoke may select one
representative, a few seeds, and explicit solvers. Report actual pass/fail/skip
counts and untested GPU limitations. Never rent/manage cloud hosts or push
without user authorization.

# Runtime benchmark v2 implementation and validation

Implemented in `solvers_testing/margin_testing`. The normal entry point is:

```bash
python run_benchmark.py 200 sparse 0.05
```

This schedules all three matching instances, all 23 eligible library solvers,
and all 100 seeds: 6,900 trials. No production campaign or grid was launched.
The GPU deployment form adds `--device cuda:0 --require-gpu`. All 37 prepared
problems remain unchanged and available offline.

## Files changed for this task

- Added `run_benchmark.py`, `run_benchmark_grid.py`, `monitor_benchmark.py`,
  and `aggregate_benchmarks.py`.
- Added the `src/qubo_benchmark/runtime/` package for preflight, identities,
  isolated workers, deadlines, observation, independent scoring, durable
  persistence, resume, progress, monitoring and aggregation.
- Added `configs/benchmark_protocol.json`, `configs/benchmark_solvers.json`,
  and the supplied v2 catalog/prompt. Copied `BENCHMARK_DEFINITIONS.txt` exactly
  from the supplied file and copy it into every experiment.
- Added inert-by-default observation hooks in `src/qubo_solvers/observation.py`
  and integrated them into `solvers.py`, `backends/torch_candidates.py`,
  `torch_execution.py`, `torch_physics.py`, `simulated_bifurcation.py`,
  `altermagnet.py`, `dynamical_geometry.py`, `geometric.py`,
  `phonon_exchange.py` and `supersymmetric.py`. Mathematical updates were not
  changed. Imports precede the warm-process clock; algorithmic calibration and
  preprocessing remain timed.
- Added `tests/test_runtime_benchmark.py`, benchmark dependencies in
  `pyproject.toml` / `requirements-qubo-benchmark.txt`, and ignore rules.
- Added `README_BENCHMARKS.md`; updated `AGENTS.md` and `QUBO_BENCHMARK.md` to
  distinguish the current runtime protocol from the retained older runner.
- Saved local validation logs, XML, smoke results and two bounded smoke/audit
  helper scripts under `benchmark_results/`.

Other pre-existing working-tree changes belong to earlier solver/library work;
they were not reverted, committed or pushed.

## Detected solvers and settings

28 canonical library methods were inventoried. These 23 GPU-capable methods
are compatible with all 37 problems and runnable in the local CPU environment:

```text
lib_simulated_annealing
lib_greedy_local_search
lib_spin_vector_langevin
lib_angular_annealing
lib_transverse_route
lib_easy_axis_annealing
lib_spin_coherent_annealing
lib_vector_amplitude_annealing
lib_mean_field_annealing
lib_tap_annealing
lib_spherical_annealing
lib_contact_annealing
lib_replica_annealing
lib_heat_bath_annealing
lib_tabu_search
lib_random_search
lib_exchange_cascade
lib_simulated_bifurcation
lib_altermagnet
lib_dynamical_geometry
lib_geometric
lib_phonon_exchange
lib_supersymmetric
```

Five are deliberately excluded from this campaign, not deleted from the library:
the two categorical methods require absent one-hot groups; planar requires
planar zero-field structure and is CPU-only; the two tree methods exceed their
configured elimination-width/table limits on these problems. There are no
missing dependencies among the 23 selected methods locally. Gurobi and CPLEX
are not registered library solvers here.

Fully resolved, fixed parameters are in `configs/benchmark_solvers.json`.
They retain the existing full-compatible settings: float32, 16 trajectories,
1,000 native steps/sweeps, native batch eight, and method-specific defaults.
Parameters are not tuned by budget, seed, size or instance. External trial seeds
are separate from algorithm parameter hashes. CUDA execution remains untested
on this CPU-only Windows machine (Python 3.14.5, Torch 2.10.0+cpu).

## Actual checks and commands

The local interpreter was
`C:\Users\Lazar\OneDrive\Documents\.solver-validation\Scripts\python.exe`.
The commands below use `python` as shorthand for that interpreter.

```bash
python -m pytest tests/test_runtime_benchmark.py tests/qubo_solvers tests/test_library_bqm_solver.py tests/test_qubo_benchmark.py -q -p no:cacheprovider --junitxml=benchmark_results/runtime_v2_regression_final.xml
```

Final result: **477 passed, 225 skipped, 78 subtests passed**, in 43.45 seconds.
One informational Torch profiler warning. CUDA tests were skipped because no
CUDA device is available. Logs are in `runtime_v2_regression_final.log` and XML.
This suite had already completed before the request to avoid further lengthy
testing. No additional full test suite was launched afterward.

Earlier focused checks passed 107 tests with 55 CUDA skips. Development testing
found a subset-configuration resume bug, which was fixed and covered by the
passing interruption/resume tests. Development smoke artifacts are separate from
the final `runtime_v2_release_smoke` artifacts and are not production results.

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -m qubo_benchmark validate
python run_benchmark.py --list-solvers
python run_benchmark.py 200 sparse 0.05 --dry-run
python run_benchmark_grid.py --instances representative --dry-run
```

Editable installation succeeded; pip reported **no broken requirements**.
Offline data validation passed **37/37**. Normal preflight showed **6,900
trials**, all three 200-sparse IDs and 23 solvers. Grid dry-run printed exactly
36 sequential configurations and executed no solver.

## Data and reference readiness

| Group | Ready / required |
|---|---:|
| 200 sparse | 3 / 3 |
| 200 dense | 10 / 10 |
| 500 sparse | 10 / 10 |
| 500 dense | 2 / 2 |
| 1000 sparse | 10 / 10 |
| 1000 dense | 2 / 2 |

Every instance has a normalized definition and finite reference value. Each
passed seven independent original-source/canonical objective comparisons.
The catalog retains 15 published-proven-optimum labels and 22 best-published
reference labels. These labels are not newly established optimality proofs.

Four supplied solution witnesses were independently checked against the original
graph and normalized objective: `san200_0.9_2` (-60), `san200_0.9_3` (-44),
`p_hat1000-1` (-10), and `p_hat1000-2` (-46). The other 33 have reference values
but no supplied verified solution bitstrings. A feasible witness does not itself
prove optimality. See `benchmark_results/runtime_v2_data_validation.log`.

## Commands and output

The full directory tree and field meanings are in `README_BENCHMARKS.md`.
Each configuration creates its own directory:

```text
results/n200_sparse_t0p050s_all_<UTC>_<id>/
  experiment.json, experiment_metadata.csv, environment.json, environment.csv
  BENCHMARK_DEFINITIONS.txt, reference_snapshot.json, solver_configuration.json
  solver_registry.csv, instances.csv, seeds.json, seed_schedule.csv
  execution_plan.csv, status.json, progress.log, summary.csv, warmups.csv
  finalization.csv
  attempts/<run-id>__0001.json
  finalization/<attempt-id>.json
  solvers/<solver-id>/
    parameters.json, parameters.csv, runs.csv, solutions.csv, trace.csv
    summary.csv, errors.log
```

```bash
python run_benchmark.py 200 sparse 0.05
python monitor_benchmark.py results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY
python run_benchmark_grid.py --device cuda:0 --require-gpu
python aggregate_benchmarks.py results --output analysis
```

The actual completed 200-sparse local smoke directory is
`benchmark_results/runtime_v2_release_smoke/n200_sparse_t0p050s_representative_20260930T121124_a05ca765496a`.
Monitoring it with `--once` exited 0. Resuming it exited 0 and kept the attempt
count at **23 before / 23 after**, without running completed trials again.
Evidence is in `benchmark_results/runtime_v2_resume_check.json`.

## Final bounded smoke results

Executed:

```bash
python benchmark_results/run_runtime_v2_smoke.py
python benchmark_results/audit_runtime_v2_smoke.py
python aggregate_benchmarks.py benchmark_results/runtime_v2_release_smoke --output benchmark_results/runtime_v2_release_analysis
```

The smoke helper ran one seed, six representatives and 23 solvers at 0.05 seconds
with unchanged parameters: **138 terminal attempts, 132 valid in-budget scores,
six `no_in_budget_candidate` outcomes**. All 23 methods produced valid timed
scores on 200 sparse, 200 dense, 500 sparse, 500 dense and 1000 sparse. Their
commands exited 0. The 1000-dense command exited 1 because the following six
methods could not capture a candidate before 50 ms on this CPU:

| Method | Actual solve wall time, seconds |
|---|---:|
| lib_simulated_bifurcation | 0.1291411 |
| lib_altermagnet | 0.1291929 |
| lib_dynamical_geometry | 0.1163259 |
| lib_geometric | 0.0996037 |
| lib_phonon_exchange | 0.1432985 |
| lib_supersymmetric | 0.1203317 |

These compact backends perform required model construction/normalization,
matrix preparation and initialization before reporting candidates. Phonon also
performs its native oscillator calibration. The time limit cannot interrupt
every native operation immediately. No late candidate was awarded an in-budget
score; these failures are preserved in operational statistics. These observed
CPU limitations are not a claim that the same methods will fail on a GPU.
CUDA-specific 50 ms behavior and peak counters still require actual GPU testing.

All six experiments match the delivered numerical source hash and fixed
configuration. The bounded final artifact audit independently reconstructed
all **132 saved best objectives**, verified deadline eligibility and confirmed
138 unique primary attempts. The runner itself independently scored every
captured candidate during the smoke. Audit exited 0; see
`benchmark_results/runtime_v2_release_smoke_audit.json` and
`runtime_v2_best_score_audit.log`.

Aggregation exited 0 and produced six experiments / 138 attempts, with no
partial groups, in `benchmark_results/runtime_v2_release_analysis`.
No solver needs to reach a published optimum for these plumbing checks to pass.
The intentional six timing failures are not hidden or converted into successes.

No further solver runs or lengthy regression suites were launched after the
request to finish; the already-running bounded smoke was allowed to complete.
No CUDA timing validation, full 100-seed campaign, cloud operations, credentials,
commits or pushes were performed. A future host needs compatible installed
Python/CUDA packages; the repository already contains all required problem data.

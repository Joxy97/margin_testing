# Solver library and offline QUBO benchmarks

This branch contains the canonical 28-method solver library and the runtime
benchmark for the 23 GPU-capable methods compatible with all 37 prepared QUBOs.
This `solvers_testing3` branch ships the standalone library and benchmark. Generated
benchmark results, environments, caches and unrelated market datasets are not
part of this deployment branch.

Changes since `solvers_testing2`: automatic use of all visible GPUs, measured
worker concurrency, simultaneous solver/seed scheduling, persistent CUDA workers,
faster independent scoring, incremental result storage and hardware telemetry.
See [BRANCH_CHANGES.md](BRANCH_CHANGES.md) for the release notes and validation.

## Clone and install

```bash
git clone --depth 1 --single-branch --branch solvers_testing3 https://github.com/Joxy97/margin_testing.git
cd margin_testing
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark,dev]'
python -m pip check
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead. Use Python 3.10
or newer with compatible numerical-package wheels. For NVIDIA execution, install
a CUDA-enabled PyTorch wheel compatible with the host driver in this environment
before the editable install. If using a GPU image's preinstalled Torch, create
the environment with `python -m venv --system-site-packages .venv` instead.
Check the active interpreter, not a different system installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python run_benchmark.py --list-solvers
python run_benchmark.py 200 sparse 0.05 --device cuda:0 --require-gpu --dry-run
```

All raw source data, normalized matrices, reference witnesses and provenance are
included under `benchmark_data/qubo37`. The runner never downloads problem data.
Python packages still need installation; for completely disconnected deployment,
prepare a wheelhouse matching the target OS/Python/CUDA environment beforehand.
An editable installation from this checkout is required because benchmark data
and configurations remain beside the source rather than inside a standalone wheel.

## Run

```bash
# Small optional GPU smoke: one representative, one seed, all 23 methods.
python run_benchmark.py 200 sparse 0.05 --instances representative --runs 1 --device cuda:0 --require-gpu --output-root results/gpu_smoke

# Normal command: all matching instances, 100 seeds each, all 23 methods,
# and every visible GPU. Concurrency is selected by a bounded pilot.
python run_benchmark.py 200 sparse 0.05

# Use all visible GPUs, one worker per GPU, without calibration.
python run_benchmark.py 200 sparse 1 --require-gpu --no-autotune

# Explicitly use two GPUs and two workers per GPU: four workers total.
python run_benchmark.py 200 sparse 1 --devices cuda:0,cuda:1 --seed-workers 2 --require-gpu
```

The normal command uses **all visible CUDA GPUs**, with CPU fallback if none
are available. `--require-gpu` forbids CPU fallback; `--device cuda:0` limits the
run to one GPU. A single invocation produces one complete result directory,
including every seed and every selected GPU. Do not also launch separate
benchmark processes on those GPUs.

Persistent workers keep their assigned GPU and reuse immutable inputs. A shared
queue schedules both different solvers and independent seeds concurrently.
The default `--seed-workers auto` tries bounded pilots with one, two and four
workers per GPU, subject to CPU and memory admission. It increases concurrency
only for a measured throughput gain greater than 10%, then freezes the counts.
Pilot trials are separate from benchmark statistics. `--seed-workers N` skips
calibration and requests N workers **per GPU**; `--no-autotune` keeps one per GPU.

Each trial retains its full solver wall-time budget. Startup, independent
scoring and result writing are outside it. Multiple workers on a GPU share
capacity during that budget, so their quality and timing distributions must be
distinguished from one-worker-per-GPU results. More workers and higher GPU
activity do not guarantee higher throughput or theoretical TFLOPS utilization.

Each configuration creates `results/<configuration_timestamp_id>/`, containing
separate `solvers/<id>/runs.csv`, `solutions.csv`, `trace.csv`, and `summary.csv`,
plus frozen JSON configuration, calibration evidence, hardware telemetry, logs
and durable resume records. CSV rows and the authoritative JSON journal are
saved after every trial; aggregate summaries refresh approximately every five
seconds and at completion.

```bash
python monitor_benchmark.py results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY
python aggregate_benchmarks.py results --output analysis
```

See [README_BENCHMARKS.md](README_BENCHMARKS.md) for all commands, protocol and
output fields, and [SOLVER_LIBRARY.md](SOLVER_LIBRARY.md) for the library API.
The historical portfolio application, unrelated tests/tools, market fixtures,
local virtual environments and generated results are not shipped here. Use the
shallow, single-branch clone above to avoid downloading historical Git objects.
Resume restores the original selected devices and worker counts without
recalibration. Runtime schema 4 includes the new execution policy and cannot
resume older source/protocol identities. See the benchmark README for admission,
timing, output fields and bounded tests. Four-RTX-3090 smoke results, measured
pilot throughput and short-budget limitations are in
[MULTIGPU_VALIDATION.md](MULTIGPU_VALIDATION.md). Validate measured throughput on
each target system; generated result artifacts are not bundled with the source.

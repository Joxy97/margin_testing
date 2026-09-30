# Solver library and offline QUBO benchmarks

This branch contains the canonical 28-method solver library and the runtime
benchmark for the 23 GPU-capable methods compatible with all 37 prepared QUBOs.
The original application source and library adapters are retained. Generated
benchmark results, environments, caches and unrelated market datasets are not
part of this deployment branch.

## Clone and install

```bash
git clone --depth 1 --single-branch --branch solvers_testing https://github.com/Joxy97/margin_testing.git
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

# Normal command: all matching instances, 100 seeds each, all 23 methods.
python run_benchmark.py 200 sparse 0.05
```

The normal command automatically uses `cuda:0` when available, otherwise CPU.
Add `--device cuda:0 --require-gpu` to forbid CPU fallback. Run only one campaign
per GPU. The first example's 50 ms limit applies to each solve, not process
startup or complete command duration. Failure to produce a candidate before the
deadline is saved explicitly; reaching the reference optimum is not required.

Each configuration creates `results/<configuration_timestamp_id>/`, containing
separate `solvers/<id>/runs.csv`, `solutions.csv`, `trace.csv`, and `summary.csv`,
plus frozen JSON configuration, metadata, logs and durable resume records.

```bash
python monitor_benchmark.py results/ACTUAL_TEST_DIRECTORY
python run_benchmark.py --resume results/ACTUAL_TEST_DIRECTORY
python aggregate_benchmarks.py results --output analysis
```

See [README_BENCHMARKS.md](README_BENCHMARKS.md) for all commands, protocol and
output fields, and [SOLVER_LIBRARY.md](SOLVER_LIBRARY.md) for the library API.
The historical application needs `pip install -e '.[application]'` in addition
to benchmark dependencies. Its unrelated market fixtures are not shipped here.
Local CPU validation passed; GPU correctness/timing still needs a smoke test on
the target hardware. Results from earlier local validation are not bundled.

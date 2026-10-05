# Offline QUBO benchmark

For the current fixed wall-clock, 100-seed campaign use
[README_BENCHMARKS.md](README_BENCHMARKS.md) and `python run_benchmark.py`.
The commands below document the retained earlier benchmark interface and data
pipeline; its 20-seed configurations are not the runtime v2 protocol.

The harness uses all 37 published instances selected by the supplied catalog.
Source files, normalized matrices and references are prepared in the repository:
**the future GPU machine needs no dataset downloads**. Run, validate and tests
are offline. Solver execution uses only the canonical `qubo_solvers` library.

## Setup and commands

Run from `solvers_testing/margin_testing`. On the NVIDIA host first install a
compatible CUDA-enabled Torch build, then the repository dependencies:

```bash
python -m pip install -e '.[benchmark,dev]'
export PYTHONPATH=src
python -m qubo_benchmark list
python -m qubo_benchmark solvers
python -m qubo_benchmark download --offline
python -m qubo_benchmark validate
python -m pytest tests/qubo_solvers tests/test_qubo_benchmark.py
python -m qubo_benchmark run --config benchmark_configs/smoke_all.json --output benchmark_results/runs/local-smoke
python -m qubo_benchmark summarize benchmark_results/runs/local-smoke
```

PowerShell uses `$env:PYTHONPATH='src'`. Global `--data PATH` precedes the
subcommand; the default is relative to the repository, not the working directory.
Output directories must be new/empty. No native build/shared library is required.

Only for an explicitly requested refresh on a machine with downloads enabled:

```bash
python -m qubo_benchmark download
python -m qubo_benchmark normalize
python -m qubo_benchmark validate
```

Download reuses hash-verified cache entries, uses bounded retries/timeouts, rejects
HTML/truncated bodies, and validates source formats. Modified cached files are
not silently replaced. Normalize/validate continue across independent instances
and exit nonzero on missing/invalid data. Tests fail if catalog data are missing.

## Later full run (not launched locally)

```bash
python -m qubo_benchmark run --config benchmark_configs/full_all.json --output benchmark_results/runs/full-library-all
python -m qubo_benchmark summarize benchmark_results/runs/full-library-all
```

| Configuration | Methods | Instances / seeds | Result slots |
|---|---:|---|---:|
| `full.json` / `full_compatible.json` | 23 compatible CPU/CUDA methods | 37 / 20 | 17,020 |
| `full_all.json` | All 28 canonical methods | 37 / 20 | 20,720 |
| `smoke_all.json` | All 28, local CPU | Six / one | 168 |

Full configurations request `cuda:0`, float32, 1,000 steps/sweeps and 16
trajectories where supported, seeds 0–19, a 60-second solve deadline and separate
120-second setup deadline. Planar has explicit CPU/float64 overrides. These are
caps, not promises that a method consumes its entire budget. CUDA requests fail
explicitly when unavailable. Standard/adaptive SBM are settings on one method.

Five methods cannot accept these catalog objectives unchanged: two categorical
methods need complete one-hot groups, planar needs planar zero-field Ising
structure, and both tree methods exceed their width limit. Full-all runs retain
explicit `unsupported` records without an invented objective. Errors, timeouts
and unavailable dependencies/devices are counted separately and yield nonzero
CLI exit status. `full.json` and `full_compatible.json` select the same 23
compatible methods. No objective or structure is modified to force eligibility.

Tree factor arithmetic and backtracking now use Torch CPU/CUDA. CPU setup checks
width <= 25 and the configured memory budget before exponential allocation.
The lower bound `treewidth >= minimum degree` cheaply rejects dense graphs;
remaining cases use deterministic min-fill ordering. Rejection of a heuristic
order does not prove no better order exists. All 37 still fail these limits.

The six smoke instances cover every size/density group. Native tensor methods
use two steps/sweeps; specialized dynamics use 16, one trajectory, seed zero.
The smoke tests loading, invocation and scoring, not optimum attainment.

## Data and provenance

| Variables | Sparse ready | Dense ready |
|---:|---:|---:|
| 200 | 3/3 | 10/10 |
| 500 | 10/10 | 2/2 |
| 1000 | 10/10 | 2/2 |

`benchmark_data/qubo37/catalog.json` is the unchanged supplied manifest, and
`IMPLEMENTATION_REQUIREMENTS.md` preserves the supplied requirements.
`raw/` holds unchanged downloaded source bytes and URL-addressed metadata;
`normalized/` holds upper-triangle NPZ arrays and readable per-instance JSON;
`references/` holds the four original witness files and validated bitstrings;
`evidence/` holds format/reference documentation. Each source records its URL,
retrieval time and SHA-256. Each normalized matrix records the parser version,
transformation, source-label mapping, measured density, original catalog entry,
and normalized checksum. SHA-256 is a local integrity fingerprint, not an
independent authenticity certificate.

The Glasgow DIMACS endpoint repeatedly returned truncated graph bodies or timed
out, including bounded byte-range attempts. The **same four named DIMACS
instances** were downloaded from the [Jožef Stefan Institute research mirror](https://e6.ijs.si/~matjaz/maxclique/DIMACS/DIMACS_subset/).
`source_overrides.json` explicitly records this retrieval change; it does not
alter any problem or catalog value. Each complete mirror file starts with the
entire corresponding partial Glasgow response. Full vertex/edge counts and
all four Glasgow witnesses validate. The original partial responses remain
preserved as provenance and are never used as solver input.

The unique OR-Library URL is cached once for all ten bqp1000 members. All
selection uses the manifest's one-based bundle positions. The parser preserves
declared dimensions, including zero rows; rejects conflicting/duplicate entries;
and handles a single triangle or matching full symmetry without summing copies.

## Objective and independent validation

Every problem minimizes `offset + x.T @ Q @ x`, with symmetric Q. Biq Mac data
already use minimization. OR-Library bqp1000 matrices are negated exactly once;
catalog reference values are already normalized and are never negated again.
For clique data, diagonal entries are -1 and each nonedge has symmetric entries
+1, giving `-selected_count + 2*selected_nonedge_pairs`. There are no auxiliary
bits or portfolio/one-hot constraints. Infeasible selections are never reported
as valid cliques.

Canonical scoring uses int64 only when a Python-integer absolute-value bound
proves it safe; otherwise it uses Python integers. Floating helper objectives
use float64 (tests use rtol 1e-12, atol 1e-9 when needed). Catalog inputs are
integral and validation requires exact equality. Each of the 37 imports is
checked on seven reproducible vectors against a separate source-format scorer,
its dense matrix, and its sparse matrix. Unit tests also cover serialization,
adapter conversion, nonzero offsets, invalid samples, and exhaustive tiny graphs.

The four supplied witnesses validate distinct labels, cardinality, every
selected adjacency, and normalized objective:

| Instance | Cardinality | Objective |
|---|---:|---:|
| san200_0.9_2 | 60 | -60 |
| san200_0.9_3 | 44 | -44 |
| p_hat1000-1 | 10 | -10 |
| p_hat1000-2 | 46 | -46 |

Label mapping was checked in the publisher's `MaxClique.java` reader/printer:
the reader subtracts one and the printer emits `i+1`. `BBMC.java` restores
original vertex indices before printing. Downloaded code was read as text only,
never executed; its hashes/inspection notes are retained without redistributing
the Java bodies. No label base is inferred from an absence of label zero.

All 33 native published values match checked table entries (Biq Mac's published
LaTeX tables for 23, and the catalog-cited metadata table for ten bqp1000 values).
Published statuses remain exactly as supplied. Evidence checking, availability
of vectors, and independent vector evaluation are separate fields. The 33 null
vector URLs remain unavailable. Witness feasibility is not an independent proof
of global optimality; no new proof is claimed.

## Solvers and timing

All adapters use `qubo_solvers.create_bqm_solver` and library-owned compact
problem/result types. Application factories and executable tools use the same
registry. The 17 native tensor and 11 specialized methods are documented in
[SOLVER_LIBRARY.md](SOLVER_LIBRARY.md). Nine overlapping entries were removed
from the former 37-entry inventory. Old IDs are migration errors, not extra
registered methods; historical records retain their original identities.

All 27 Torch methods support float32/float64; planar receives float64. Tree
exactness is subject to precision, so use float64 for demanding reference work.
Tree tests compare exhaustive minima, partition functions and sample frequencies.
Planar and exact-tree repetitions are labeled deterministic timing repetitions.
No commercial backend is added or substituted.

GPU-capable describes search execution. CPU topology/preparation, authoritative
source scoring, optional repair/refinement and categorical/physics host snapshots
are explicit boundaries. Inventory and result metadata expose resident modes.
Full configs disable exposed physics rounding/block refinement/local search;
standard SBM local search and geometric/SUSY proof search also default to zero.

The adapter passes Q's diagonal as linear terms and `2*Q[i,j]` for each pair
once. It uses identity variable order, no coefficient scaling, and no rounding.
Existing solvers own their internal Ising conversion, normalization and binary
candidate selection. The runner strictly rejects nonbinary/nonfinite output and
rescores in original normalized units. Published targets and witnesses never
enter the worker: it receives only the objective NPZ and solver configuration.

The GPU configuration disables exposed local-search, conditional-rounding and
block-polishing switches where available, with standard SBM's default
`local_search_sweeps=0`. No harness repair/polish is added.
Fixed internal steps remain part of the original solver implementation. If you
enable polish, assign a distinct variant `id`/`output_variant`; its cost belongs
to that variant's solve time. Trials are labeled as independent seeded starts,
not deterministic timing repetitions, and the harness adds no artificial noise.

Each run uses a fresh spawned process. Setup and solving have separate deadlines;
a nonreturning worker is terminated. GPU calls are synchronized before/after
solving; device initialization belongs to setup. There is no solver warmup, so
first-call kernel/JIT cost is included. Existing interfaces combine matrix
preparation/transfers with solving; `transfer_seconds=null` explicitly records
that these costs cannot be isolated without changing those interfaces.
The result records parent loading/validation, worker loading/setup, synchronized
solve, and end-to-end times. Parent load time is amortizable across trials and
shown explicitly. Solver-reported energy is diagnostic only.

These interfaces expose no timed incumbent callback. Therefore history and
time-to-target remain null; a final objective is never converted into a fake
time-to-target. A killed call has no recoverable incumbent. Results retain the
hardware/device, precision, explicit and resolved parameters, seed, source and
matrix hashes, solver source hash, Git HEAD, and a hash including uncommitted
benchmark/optimization code. Configurations and per-run JSON are saved before
summarization. Signed deltas are retained: scores below a published proven
optimum trigger a validation alarm; scores below a best-known target are
candidate improvements needing review. Summaries separate all six groups,
structural families, and solver variants, and retain every per-instance result.

## Git deployment

Include `src/qubo_solvers`, benchmark code/configs/tests and
the approximately 20 MB `benchmark_data/qubo37` bundle in the future commit.
These paths are not ignored. `.gitattributes` preserves source bytes across
Windows/Linux checkouts; no Git LFS dataset fetch is required. The library wheel
alone does not include benchmark data; run from a prepared checkout.

Current validation is in [SOLVER_LIBRARY_VALIDATION.md](SOLVER_LIBRARY_VALIDATION.md)
and `benchmark_results/smoke_library_consolidated/`. Earlier validation/coverage
reports and smoke folders preserve historical inventories. Local tests used CPU
Torch: actual CUDA execution, GPU speed/memory and the full campaign remain
untested. No cloud service, credentials, commit or push was used.

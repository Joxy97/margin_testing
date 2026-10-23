ARCHIVED LM23, SM23 AND SM20 QUBO RESULTS

Code branch: https://github.com/Joxy97/margin_testing/tree/solvers_testing
Results branch: https://github.com/Joxy97/margin_testing/tree/solvers_testing_results

There are 11 original campaigns (23 solvers each, 25,300 trials), plus two
SM23 campaigns (23 solvers each, 4,600 trials), and one dense-n200 SM20
campaign (20 solvers, 2,000 trials). All 31,900 attempts are complete, with
zero recorded failures. No 10000-dense campaign is present.

Folder layout, identical for sparse and dense:
  <density>_<n>/manifest.json, state.json, completion.json
  <density>_<n>/raw/<solver>/<problem>.npy
  <density>_<n>/aggregated/<solver>/<problem>.csv
  <density>_<n>/progress/<solver>/<problem>.json
  <density>_<n>/plots/, logs/, solver_settings/
For n200, each problem directory contains the following experiment folders:
  <density>_200/LM23/ (original23; canonical layout above)
  <density>_200/SM23/ (new23; same canonical layout)

The redundant sparse margin_testing_codev4/benchmark_results wrappers were
removed. Original log bytes and setup/completion markers are retained in
each problem's logs/. Original transport archives remain at the top level.
postprocessed/ contains additional exports where present in the original
data, and for both new campaigns. Original result bytes were not rewritten.

SM23 is a separate experiment within each n200 problem folder because it
has a distinct immutable manifest, configuration, source and hardware
identity. This avoids forging a combined resumable experiment. Read both
LM23/raw/ and SM23/raw/ for all 46 solvers on either n200 problem.
The old 23 and SM23 solver IDs are disjoint; each group has 100 seeds.

PLots/ retains all 33 original figures unchanged and adds six *_SM23.png
figures (exact hits, mean gap percent, 1%-threshold hits; sparse and dense).
Each new figure has 6 panels: 4+4+4+4+4+3 solvers. The new figures use
the original Windows System.Drawing renderer: matching Segoe UI fonts,
panel borders, legend placement, colors, axes and the 1% gap guide.
See METRIC_DEFINITIONS.txt for QUBO, metrics, timing and missing values.

Each campaign's solver_settings/experiment.json exports the full recorded
manifest, including parameters, invocation, seeds, timing, worker plan,
hardware/environment, input and source hashes. solver_parameters.json is a
configuration-format parameter export. For new trials use the original
frozen production_config.json and campaign wrapper for faithful replay.
reproducibility/all_solver_settings.csv provides a searchable parameter
table across all campaigns. No trajectories or defaults were inferred.

See reproducibility/RETESTING.txt for source archives and replay commands.
SM23/reproducibility/campaign/ contains tuning raw trials, selected and fully
resolved settings, guards, readiness/environment logs and launch scripts.
Some historical instructions describe the pre-launch state; those files
are preserved evidence, not the current completion status. No captured
production terminal transcript was found; per-seed progress and completion
records, campaign_progress.json and all available tuning/setup logs are kept.

Validate all stored raw data without launching trials:
  python -m pip install numpy
  python scripts/build_publication.py --validate-only
Regenerate derived settings and the six SM23 figures:
  python -m pip install numpy
  python scripts/build_publication.py
Figure generation requires Windows PowerShell and System.Drawing. Validation
only remains cross-platform. scripts/plot_sm23.ps1 can also be run directly.
These commands do not alter original raw data, manifests or original plots.

reproducibility/validation_report.json records the full data validation.
original_file_preservation.json maps all 909 original files to their
current paths and SHA256. SHA256SUMS.json covers this published tree except
itself and caches. Git treats result payloads as binary to preserve bytes.

BRANCH RENAME, 2026-10-19:
solvers_testing5 was renamed to solvers_testing. Active code README and
instructions use the new name. Historical logs, immutable manifests and
source archives retain their original names and hashes; any old branch
reference in them denotes historical provenance. Use solvers_testing for
current code, and the included exact source archives for faithful retests.

SM20 ADDITION, 2026-10-23
dense_200/SM20/ contains the twenty open-system models, all 100 seeds each,
using the same exact gka5e input as SM23. Dense n200 now has 66 distinct
solvers across LM23, SM23 and SM20; sparse n200 retains its 46 solvers.
The new SM20 folder uses the same raw, aggregated, progress, postprocessed,
plots, solver_settings and reproducibility structure as SM23.

PLots/ adds three *_SM20.png figures in the same SM23 style: exact hits,
mean gap percent and hits within 1%. The six-panel canvas has five panels
with four solvers each; the sixth notes that all 20 solvers are shown.
SM20/plots/gka5e/ also contains the standard gap, hit probability and TTS99
plots. All previous result payloads and plot images remain byte-identical.

Exact parameters, 16 internal trajectories, seeds, finite step schedules,
worker plan and software/hardware identities are in SM20/solver_settings/.
SM20 used four RTX 5060 Ti GPUs with one seed worker per GPU. These are
initial candidate defaults, without broad tuning. See SM20/reproducibility/
REPLAY.txt and the SM20 source/input archives for a fresh reproducible test.
No production terminal transcript was captured; the actual setup/test logs,
per-seed progress and final completion records are retained without invention.

Audit this addition and its derived exports without launching trials:
  python scripts/build_sm20_publication.py --validate-only
Refresh its quality summary and searchable settings/publication indexes:
  python scripts/build_sm20_publication.py
Regenerate its three styled figures on Windows:
  powershell -File scripts/plot_sm20.ps1 -VariableCount 200 -DensityCategory dense
The existing build_publication.py --validate-only also includes SM20.
See SM20/postprocessed/final_quality_summary.csv for the final per-solver
best/mean/worst gaps, exact hits and 1% hits.

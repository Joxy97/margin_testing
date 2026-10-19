ARCHIVED ORIGINAL23 AND SM23 QUBO RESULTS

Code branch: https://github.com/Joxy97/margin_testing/tree/solvers_testing
Results branch: https://github.com/Joxy97/margin_testing/tree/solvers_testing_results

There are 11 original campaigns (23 solvers each, 25,300 trials), plus two
new SM23 campaigns (23 solvers each, 4,600 trials). All 29,900 attempts are
complete, with zero recorded failures. No 10000-dense campaign is present.

Folder layout, identical for sparse and dense:
  codev4_<n>_<density>/manifest.json, state.json, completion.json
  codev4_<n>_<density>/raw/<solver>/<problem>.npy
  codev4_<n>_<density>/aggregated/<solver>/<problem>.csv
  codev4_<n>_<density>/progress/<solver>/<problem>.json
  codev4_<n>_<density>/plots/, logs/, solver_settings/
For n200, the problem directory contains only these two experiment folders:
  codev4_200_<density>/LM23/ (original23; canonical layout above)
  codev4_200_<density>/SM23/ (new23; same canonical layout)

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

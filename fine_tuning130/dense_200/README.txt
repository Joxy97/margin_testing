fine_tuning130 — dense-200 post-tuning production campaign

This is the latest production rerun of 130 tuned solvers on gka5e, not the earlier parameter-search trials. Every solver used seeds 0-99, a 20-second cap, and checkpoints at 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, and 20 seconds. All 13,000 runs succeeded.

raw/, aggregated/, progress/, postprocessed/, manifest.json, state.json, and completion.json form the compact plot-ready result set. solver_settings/ records the exact production parameters. plots/gka5e/ contains nine runtime figures and nine publication-style figures.

downloads/archives/ contains all seven checksum-verified source archives, including every per-seed production JSON. downloads/hosts/ keeps each host's run log and identity/completion records without duplicating the 26,000 extracted trial/claim files. download_manifest.json binds the archives and host completion records to the 13,000 successful runs.

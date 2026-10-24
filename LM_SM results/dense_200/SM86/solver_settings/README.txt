SM86 SOLVER SETTINGS - gka5e, n=200, dense

This is a DERIVED, NON-RESUMABLE collection of four original campaigns.
The collection contains 86 library entries: 78 ran 100 seeds each (7,800
trials), and eight structured decoder models are recorded in unsupported.json.
Unsupported models have no invented trial rows, parameter presets or failures.

experiment.json copies the derived collection manifest and adds its canonical
manifest_sha256. Its partitions preserve every original environment, invocation,
worker plan, source/input/configuration identity and complete solver parameter
mapping. solver_parameters.json copies actual recorded values, never defaults
resolved from newer source. solver_origin maps every flat solver to its original
host/manifest; original progress files retain that original manifest identity.

All original campaign/results and setup evidence is preserved in
../reproducibility/campaign/partition_N/. The flat raw, progress and aggregated
files are byte-identical convenience copies. Never resume into this SM86 root.
Faithful retesting uses an original partition campaign and the exact archived
source/input files, with NEW output directories. See ../reproducibility/REPLAY.txt.

Each solver has benchmark seeds 0 through 99. The parameter runs=16 denotes
16 internal trajectories within each seeded trial; run_batch_size=16 and
best_only=true. These are initial candidate settings, without broad tuning.
The complete physical/numerical parameters are in solver_parameters.json.

The cap is 20 seconds. Checkpoints are 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10,
and 20 seconds. wall_schedule=false: model-specific finite max_steps is a
separate work limit. A natural return retains its incumbent through 20 seconds
without restarting; a completed independently certified optimum stops early
and is carried forward. Reaching a 1% gap does not trigger the optimum stop.
An empty early checkpoint remains unavailable; later candidates are not
backdated. A 20-second sample does not imply 20 seconds of active search.

Immutable input/scoring-operator preparation precedes the solve clock.
RNG reset, fresh solver state, preprocessing, search, decoding, independent
candidate scoring, GPU completion, scalar copies and validation after that
origin are INCLUDED. One seed worker runs per GPU; each host uses its four
recorded GPUs, one CPU thread per worker, TF32 disabled and no autotuning.
Exact hardware/package/driver versions are recorded per partition; this is
not a single-hardware timing comparison. Dynamics are FP32, authoritative
source coefficients retain FP64. The configured 536870912-byte tensor memory
guard is not a measurement of total process/CUDA memory consumption.

The minimization reference is -35154. Relative gap is
100*(E - (-35154))/35154. A 1% hit requires E <= -34802.46, hence integer
E <= -34803. Missing/failed trials remain misses in the denominator of 100.
The original aggregate gap_to_solution uses absolute objective units;
postprocessed/final_quality_summary.csv reports relative percentages.

Derived canonical manifest SHA-256: 00dd24ed5ec73b34110a8a1e520a6a09702c17cff8fc639edd8f9e4f9f93340c
Actual shared numerical/runtime snapshot: ecdeb6896f4cc85ecf8e3b47abd7836f39ecf15c9c73dfa99609320018f61ede
Exact normalized input SHA-256: 3a2d6521caefa62a867a7b96dae169388b14088b1ba309dbcdef989b9460d375
Each original manifest canonical and file-byte SHA is listed separately.
No solver reruns or numerical changes were made while exporting these results.

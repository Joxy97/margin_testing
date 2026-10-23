SM20 SOLVER SETTINGS - gka5e, n=200, dense

Source of truth
experiment.json copies the original campaign manifest, adding its canonical
JSON SHA-256 as manifest_sha256, matching the SM23 settings schema. The canonical
digest and original file byte hash are recorded separately below.
solver_parameters.json copies the exact recorded per-solver parameter mapping.
Neither file resolves defaults from a later version of the code. Historical
manifest.json, completion.json, raw data and their identities remain unchanged.

Trials and trajectories
There are 20 new open-system solver models, each with 100 independent trial
seeds (0 through 99): 2,000 trials in total. The per-solver parameter named
"runs" is 16 internal trajectories inside one seeded trial, not the number
of independent benchmark seeds. run_batch_size=16 and best_only=true for all
20 solvers. Each model's remaining physical and numerical parameters are
listed without omission in solver_parameters.json.

Schedule and timing
Each trial has a 20-second wall-clock budget. The nine checkpoints are
0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10 and 20 seconds. These are reporting times;
max_steps is a separate, model-specific bound on integration/sampling work.
The launcher uses a finite fixed-step schedule (wall_schedule=false).
A naturally returned incumbent is retained at later checkpoints through 20s;
the solver is not restarted. A verified optimum also stops the trial early,
and its independently rescored incumbent is carried to later checkpoints.
The published optimum for this minimization problem is -35154. Only a
verified optimum hit triggers the optimum stop rule, not reaching a 1% gap.
Each checkpoint uses the last completed independently scored incumbent at or
before that boundary. A reported 20s value does not imply 20s of active work.

Immutable input/scoring-operator preparation occurs before the solve clock.
Independent candidate scoring, GPU completion, scalar copies and validation
performed after the solve origin are INCLUDED in that clock. The archived
reproducibility/campaign/OPEN_SYSTEM_DENSE_CAMPAIGN.txt contains an older claim
that scoring is excluded; that statement is inaccurate. The archived file is
preserved unchanged as historical evidence. This clarification follows the
actual continuous_worker.py implementation and the checkpoint eligibility rule.

Execution
One seed worker per GPU, four GPUs, with homogeneous solver groups. The exact
devices, worker plan, package versions, CUDA/driver, platform and hardware
identity are copied in experiment.json. Dynamics use float32; source objective
scoring is performed independently with the runtime's FP64 source coefficients.
The configured per-trial memory limit is 536870912 bytes (512 MiB); it is an
admission/allocation guard, not a claim that total GPU process usage is 512 MiB.
TF32 is disabled. Autotuning is disabled. The settings are the initial candidate
defaults validated during implementation; no broad parameter tuning was done.

Immutable identifiers
Canonical manifest SHA-256 (experiment.manifest_sha256): 551d9d92fb607132ddd773fb14e3e2ad3c91daadb5888ed8cc2c9a2ce39ce442
Original manifest file byte SHA-256: 04b0b3b8aad98d9b0660f11b71f193aeafd8208cd30fb14fa43c20b901390a53
Numerical source snapshot: 29624a8421d59da66f358ff6b1a4efb491553fe4ab523abdc78d893ab579bc8c
Configuration SHA-256: 0aa8b5680ff55390f5d5e23e8b90e187b0956a92ccc846949061d06fb6682abd
Input gka5e SHA-256: 3a2d6521caefa62a867a7b96dae169388b14088b1ba309dbcdef989b9460d375
Launcher SHA-256: 5723dae3d12ed3c4096f06de3a4b650eced69365421cc4f8a9e057dd05232d84
Campaign identity and all frozen campaign/input file hashes are preserved in
experiment.json -> execution -> open_system_campaign_identity and the campaign
bundle under ../reproducibility. See ../reproducibility/REPLAY.txt for replay.

Comparison limits
These settings describe this completed SM20 experiment, not tuned SM23/LM23
presets. Internal trajectory counts, finite step limits and polishing may
differ between model families. Benchmark seeds are stochastic repetitions,
and a success probability estimate is based on these 100 trials only.

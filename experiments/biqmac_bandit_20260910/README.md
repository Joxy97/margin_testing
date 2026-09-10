# BiqMac model-free parameter-learning pilot

Runs on the isolated eight-RTX-5090 host `13.56.204.87:21963`, Supervisor program
`biqmac_bandit`, project `/workspace/biqmac-bandit-20260910`. The other host's
`biqmac_maxcut_long` experiment is not modified or stopped.

```bash
bash tools/track_biqmac_bandit.sh status
bash tools/track_biqmac_bandit.sh logs
bash tools/track_biqmac_bandit.sh fetch
```

`Ctrl-C` stops following logs, not the experiment. Fetches are safe while running;
the last JSONL line can be incomplete during a live copy. Final artifacts are
published atomically. The process intentionally refuses an existing experiment
directory: automatic crash resume is not implemented, rather than silently
repeating training updates. Supervisor does not automatically restart it.

## Protocol

- All 178 prepared BiqMac instances, original unmodified weighted MaxCut QUBOs.
- Fixed 32 trajectories per call; SBM/SVL 5,000 steps, binary TRF 10,000 steps.
- One persistent worker per GPU, eight concurrent episodes, shared CPU learners.
- 200 training episodes per solver; checkpoints at 50, 100 and 200 episodes.
- Approximately 60/20/20 split by graph groups, stratified by archive family.
  Rudy numbered realizations and Ising seed variants stay together. Exact groups
  and counts are captured in `experiment.json`, not presumed to be exact ratios.
- Each training graph is visited before the shuffled schedule repeats.
- Validation: each checkpoint on every validation graph, one common fresh seed;
  select checkpoint per solver by mean relative gap.
- Freeze policies, then evaluate every instance with three fresh seeds and three
  methods: learned stochastic policy, repository defaults, uniform random arms.
  Only rows labeled `test` measure held-out generalization. Training/validation
  evaluation rows are explicitly in-sample. No test rewards update a policy.

## Learner and parameters

This is a contextual, delayed-feedback engineering adaptation of model-free
[EXP3-IX (Neu, 2015)](https://proceedings.neurips.cc/paper/2015/file/e5a4d6bf330f23a8707bb0d6001dfbe8-Paper.pdf),
not a claim of the universally best RL method or its original regret guarantee.
The coordinator updates shared log weights after every training solver call,
using its recorded selection probability and implicit exploration denominator.
Learning rate is 0.05 and implicit exploration is 0.025. Reward is the on-time
increase in the episode's best cut divided by the reference magnitude (at least
one); improvements telescope to final quality. A late/failed call earns zero.
The initial incumbent is the valid all-zero assignment with cut zero.

Contexts use graph size, at most 1,024 sampled absolute edge weights, and whether
this is the episode's first call or a restart. Density is recorded. There is no
spectral inspection, local probing search, landscape model or reward predictor.
Reference cuts are training reward normalizers and post-solve evaluation labels,
never policy input features. Unseen context buckets fall back to uniform arms.

Each solver gets 32 reproducible joint arms, including the unchanged benchmark
default. Algorithmic dimensions covered:

- SBM: `dt`, `a0`, `c0`, `gamma`, `initial_scale`.
- SVL: `dt`, `mass`, `damping`, `temperature`, both transverse-field endpoints,
  both problem-scale endpoints, and integrator.
- TRF: `time_step`, `mobility`, `route_strength`, `gamma`, both kappa endpoints,
  schedule exponent, integrator, candidate interval and candidate batch size.
- Execution dimensions: precision, trajectory batch size and energy chunk size;
  SVL noise chunks; TRF matrix format/threshold, CUDA graphs and graph block size.

The learner chooses joint configurations, not independent continuous optima.
This finite pilot does not exhaust the continuous parameter space. Runs/steps,
seeds, memory safety ceilings and irrelevant aliases/deduplication switches are
controlled rather than learned. TRF collects intermediate candidates, so equal
trajectory counts do not imply equal numbers of scored samples across solvers.
Invalid/numerically unstable arms are recorded as failed calls, never repaired.

## The 20-second contract

The window starts with an already parsed, resident host QUBO and initialized CUDA
context. It includes lightweight features, coordinator selection latency, solver
preparation/transfers/graph capture, dynamics, float64 cut scoring and host result
materialization. Disk loading and one tiny CUDA warmup per solver are excluded.
Every actual benchmark call retains its fixed steps/runs.

Only results fully available within 20 seconds improve the incumbent. Restarts
stop when remaining time is below 1.1 times the episode's median completed call
duration. The valid zero-cut incumbent exists even if no call finishes in time.
This is a best-by-deadline quality experiment, **not hard real-time GPU kernel
preemption**: an in-flight full solver call may finish late during cleanup; its
result is rejected and `cleanup_overrun_s` records the extra occupied wall time.
`best_time_s` is time until the best accepted cut, not an optimality certificate.

The runner validates binary samples and original-QUBO energy against weighted
cut scoring on every call, records exceptions, and checks input file digests.
Existing solver implementations are unchanged. No separate unit-test suite was
requested or run for this orchestration addition.

## Outputs and expected time

`status.json` reports active GPU/solver/episode/arm and phase progress;
`progress.log` prints starts, every completed call and episode results.
`trials.jsonl` contains actual parameters, probabilities, rewards, timings and
errors. Per-episode JSON stores the best assignment and anytime improvement
curve. `training.csv`, `validation.csv`, `evaluation.csv`, `summary.csv`, policy
checkpoints and the input/action/split manifest support later comparisons.
Summary quality wins compare parameter-selection methods within each solver and
count ties; evaluation CSV also permits cross-solver quality/time comparisons.

Ideal time is `(600 + 9 * validation_instances + 27 * 178) * 20 / 8` seconds,
roughly four hours. Allow approximately **5-6 hours** for startup, scheduling,
captures, uneven runtimes and cleanup overruns; this is not a measured guarantee.
Large repeated overruns can make it longer and are visible in the live CSV/log.

GPU experiment design follows the ScientificAgentSkills `optimize-for-gpu`
guidance: keep the existing Torch backend, isolate devices, synchronize complete
solves and retain original objective scoring rather than alter solver math.

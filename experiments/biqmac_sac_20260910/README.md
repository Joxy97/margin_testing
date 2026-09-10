# BiqMac discrete SAC pilot

Independent experiment on `45.59.100.176:43437`, using all eight RTX 2080 Ti GPUs.
Supervisor program: `biqmac_sac`. Remote project: `/workspace/biqmac-sac-20260910`.
The stopped non-RL sweep and the RTX 5090 server's EXP3-IX experiment are untouched.

```bash
bash tools/track_biqmac_sac.sh status
bash tools/track_biqmac_sac.sh logs
bash tools/track_biqmac_sac.sh fetch
```

`Ctrl-C` leaves the log viewer without stopping the job. Fetch copies partial or
complete results into this directory's `results/`; a live JSONL copy can end with
an incomplete line. Supervisor does not restart failed jobs automatically. The
runner refuses an existing experiment directory; checkpoint restoration and
automatic resume are not implemented.

## Fixed experimental budget

- SBM, SVL and binary TRF: **32 trajectories and 10,000 steps per solver call**.
- Each episode scores the best valid cut available within **20 seconds**.
- 200 training episodes per solver; checkpoints after 50, 100 and 200 episodes.
- Same grouped split and action bank as the earlier EXP3 pilot: 110 training,
  39 validation, 29 held-out test graphs. Related realizations stay together.
- Select a checkpoint separately for each solver using validation mean gap,
  then freeze policies. Validation uses one fresh common seed per graph.
- Evaluate all 178 graphs, three fresh seeds, SAC/default/uniform-random methods.
  Only the 29 held-out graphs support generalization claims. Train/validation
  evaluations are explicitly labeled in-sample; no evaluation policy updates.
- Eight persistent GPU workers, one CPU learner per solver. No GPU sharing
  between solver workers; each keeps its own cached parsed QUBOs.

## What SAC learns

The runner implements discrete Soft Actor-Critic: a stochastic actor, twin Q
critics, target critics, and replay of previous transitions. Each solver has its
own 32-arm configuration bank, actor, critics and replay buffer. Joint arms vary
the same algorithmic and bounded execution controls as EXP3; arm zero preserves
the benchmark defaults. This is **not continuous parameter optimization** and
does not exhaust all parameter combinations. Seeds and safety ceilings remain
controlled; run batching still totals 32 trajectories. TRF scores intermediate
candidates as before, so equal trajectories do not mean equal sample counts.

Small CPU networks use two 64-unit hidden layers. Adam learning rate is 0.0003,
batch size 64, replay capacity 100,000 and two updates per completed transition
after a 64-transition uniform warmup. Target interpolation is 0.005, gradients are
clipped at 10, fixed entropy coefficient is 0.002. Episodes are finite and the
discount is 1.0. The exploration bonus regularizes the quality objective; these
are pilot hyperparameters, not claimed optimal settings.

At most 1,024 edge weights are sampled by the existing worker. Observations use
graph size/density, coarse weight scale, first/restart flag, elapsed fraction,
call count, last call duration and whether it improved the incumbent. There is
no eigendecomposition, probing search, transition model or QUBO landscape model.
The critics predict returns from experience, not individual QUBO energies.
Reference values are not policy features.

Reward is the on-time incumbent improvement divided by the reference magnitude
(at least one). Improvements telescope to final cut quality relative to the
valid initial zero-cut assignment. The next action request supplies the actual
next observation; episode completion marks the final transition terminal.
Training replay updates occur before selecting the next action or at episode
completion. Evaluation actors are frozen and do not perform replay updates.

## Timing, output and limitations

The existing worker starts the 20-second clock after parsing/caching the QUBO and
warming CUDA. Feature extraction, policy selection/queue latency, solver input
preparation, transfers, graph capture, complete dynamics, float64 cut scoring and
host assignment materialization are inside the window. Late results earn no
credit; `cleanup_overrun_s` records time spent finishing an in-flight call. This
does **not** preempt GPU kernels or guarantee a full 10,000-step call completes
within 20 seconds. If none does, the valid zero-cut fallback remains available.
Restart admission uses the existing median-duration heuristic. Learning between
calls consumes training time; frozen evaluation includes inference but no training.

Every call retains input hashes, binary/energy consistency checks, and original
MaxCut scoring. `progress.log` prints every call and episode, including deadline
and error flags and learner loss/update statistics. `trials.jsonl` records actual
parameters, rewards, transitions and learning timings. Episode JSONs contain best
assignments and improvement curves. CSVs cover training, validation, evaluation
and final split-level summaries, including mean/median/p90 gaps and both
within-solver method wins and cross-solver quality wins (ties counted).
Torch checkpoints include networks, optimizers, replay and RNG state.

5,757 windows imply roughly **four hours** if every window occupies exactly 20
seconds with ideal eight-way scheduling. On the RTX 2080 Ti host, plan provisionally
for **6-8 hours**, potentially longer if fixed-length calls repeatedly overrun.
This is an estimate, not a measured completion guarantee. Large gaps or late-call
rates are meaningful outcomes, not reasons to silently reduce steps or runs.

SAC, defaults and uniform random are matched on this server. Direct differences
against the older EXP3 experiment cannot be attributed solely to the learner:
that experiment uses RTX 5090 GPUs and 5,000 steps for SBM/SVL. A causal algorithm
comparison would require a matched EXP3 rerun, which is not launched here.

No production solver math or original EXP3 runner was changed. The new file is an
experiment orchestration/learning adapter. No separate unit-test suite was
requested or run; the requested experiment itself performs runtime scoring checks.
See `../../docs/benchmarks/biqmac_sac_20260910.md` for primary-source research and
algorithm limitations. GPU scheduling follows ScientificAgentSkills
`optimize-for-gpu`: retain Torch, isolate devices and score synchronized solves.

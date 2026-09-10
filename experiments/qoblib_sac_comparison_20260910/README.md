# QOBLIB: discrete SAC followed by Hybrid SAC

The idle first host, 45.59.100.176:43437, supplies eight independent RTX 2080 Ti
GPUs. No previous benchmark or margin result is changed. Input artifacts remain
under `/workspace/qoblib-eligibility-20260910/results`.

## Schedule and comparison

There are two sequential six-hour wall-clock blocks: discrete SAC first, Hybrid
SAC second. Each block reserves four hours for training, one for validation,
and one for held-out evaluation. CUDA startup, input loading, policy inference,
learner updates, scoring, persistence, failed calls, and worker replacement all
consume the budget. A few seconds are reserved for phase cleanup; OS process
termination is not a real-time guarantee. Size-weighted slots are planned before
any solving, with equal aggregate slot seconds for SBM, SVL, and TRF. The schedule
stops launching work before each boundary.

Both methods use 32 trajectories and 10,000 steps per complete solver call,
variable-duration problem slots, float64 dynamics, trajectory batches of eight, and
8,192-edge scoring chunks. SVL noise chunks are 16. TRF uses sparse matrices,
128-candidate batches and 25-step graph blocks. These memory controls are fixed,
not learned. Solver estimates are checked against 80% of 11 GiB before calls.
OOMs and timeouts are outcomes, not silently removed cases. This runner does not
claim that the earlier 845 estimated-admissible cases have been GPU-validated.

A timed-out GPU process is terminated and replaced; independent pipes avoid
corrupting another worker's shared queue. Only completed, scored, persisted
improvements inside the window count. The binary zero vector is an explicitly
scored fallback, not a claim of feasibility. Missing results remain missing.
An interrupted trajectory does not provide an intermediate solution.

The prior SAC implementation is **discrete SAC over 32 fixed configurations**,
not continuous SAC. Hybrid SAC learns a categorical branch distribution and
a conditional tanh-Gaussian parameter distribution. Twin critics, target critics,
replay, pathwise continuous gradients and exact categorical expectations are
used. Discrete and continuous entropy coefficients are each 0.002. Inactive
continuous parameters are masked from both critics and entropy. Runs and steps
are not learned. Continuous ranges and categorical branches are exported in
the manifest and defined in `tools/qoblib_hybrid_sac.py`.

Continuous controls cover all numeric dynamics controls exposed in the existing
action bank. Discrete choices include zero/active controls, SVL/TRF integrators,
and TRF candidate intervals. Hardware execution knobs remain fixed for memory
safety and fair allocation. The baseline bank has the same safety overrides;
its finite action support is intentionally different from the hybrid policy.

## Size-based time allocation

The planner uses `sqrt(binary_variables + 2 * quadratic_interactions)` as a cheap
work proxy, not a measured runtime prediction. Budgets increase with this proxy,
bounded by five and 300 seconds, and are normalized before any GPU solving. An
instance has the same window for all solvers and both learning methods. Steps
and runs remain fixed; larger windows permit longer completed calls/restarts.

Each phase prepays at most 85% of its eight-GPU capacity, divided equally among
solvers. The remainder covers startup, worker replacement, learner updates,
checkpointing and scheduling. Training repetitions are selected around a
20-second reference average; validation schedules both saved checkpoints, and
evaluation requests two repetitions when the minimum windows fit. Every case
in each partition is scheduled. Eight statically balanced queues are shuffled
within each GPU to mix sizes while preserving their prepaid durations.

`time_budget.json` records each case/solver/checkpoint/repetition window, GPU
assignment, aggregate solver seconds and reserved time. Logs print the allotted
window at each start. CLI controls are `--min-window`, `--max-window`,
`--target-window`, and `--utilization`. An infeasible plan is rejected rather
than silently dropping cases. The phase deadlines still take priority if actual
overhead exhausts the reserve; remaining slots get `not_started_budget` rows.

## Splits and scoring

The inventory deduplicates alternative formulations, choosing the smallest
estimated footprint. Splits are seeded and grouped by source identity within
each family, approximately 60/20/20 in groups rather than necessarily cases.
Portfolio parameter/perturbation variants, sports Addition variants, Birkhoff
sizes, topology sizes and Steiner point-set identifiers stay together. LABS
uses length buckets. This is source-name grouping, not a proof that no hidden
generator relationship or graph isomorphism exists. All available admitted sizes
are included; this is not a newly imposed minimum-variable filter.

Training checkpoints at two and four hours are compared on paired validation
case/repetition results. Family-averaged quality chooses a checkpoint separately
for each solver. The final checkpoint wins ties or absent evidence. Evaluation
freezes policies and does not update replay or optimizers. Both methods share
splits, slot order, source identities and solver seeds, but timeouts/restarts can
produce different coverage. Final comparisons must use matched completed cases
and report coverage rather than treating missing results as wins.

For converted LPs, integer constraint residuals are checked with no feasibility
tolerance. Feasible solutions rank above infeasible solutions; feasible original
objectives are reported with the original objective sense. The normalized reward
potential is `0.5 - 0.5*tanh(E / max(1, objective_range_bound))` for feasible
solutions and `-0.5*v/(1+v)` for mean squared constraint violation `v` otherwise.
Feasible penalty terms vanish; infeasible objectives are not reported through
numerically dangerous penalty subtraction. For native QS models, rewards use
bounded original-QUBO energy improvement. Original-domain feasibility for native
QS is unknown, not inferred from a low energy. No reference optimum is used.
Rewards are improvements in incumbent potential during each finite window.

Loading and coefficient aggregation happen once per episode. The policy uses
eight cheap features including size, density, at most 1,024 sampled coefficient
signs, constraint presence, elapsed fraction and incumbent/call information.
There is no eigensolver, landscape predictor, local search, or model-based RL.

## Commands and artifacts

```bash
PYTHONPATH=src /venv/main/bin/python -u tools/benchmark_qoblib_sac_comparison.py \
  --inputs /workspace/qoblib-eligibility-20260910/results \
  --output /workspace/qoblib-sac-comparison-20260910/results
bash tools/track_qoblib_sac_comparison.sh status
bash tools/track_qoblib_sac_comparison.sh logs
bash tools/track_qoblib_sac_comparison.sh fetch
```

The manifest captures the split, action support and timing contract. Each method
has training/validation/evaluation CSVs, half/final Torch checkpoints containing
networks, optimizers and replay, selected-policy metadata, and packed best binary
assignments. `trials.jsonl` records each completed action and reward;
`status.json` and the Supervisor progress log report live progress. Do not load
Torch checkpoints from untrusted sources. Outputs are not resumable by this
runner; an existing manifest prevents accidental overwriting.

No separate unit tests, syntax checks, GPU smoke tests, or numerical-equivalence
suite were requested or run during implementation. Runtime binary/energy and
residual checks are part of the experiment, not a substitute for such tests.

## References

- Delalleau et al., *Discrete and Continuous Action Representation for Practical
  RL in Video Games*: https://arxiv.org/abs/1912.11077.
- Christodoulou, *Soft Actor-Critic for Discrete Action Settings*:
  https://arxiv.org/abs/1910.07207.
- QOBLIB source and data attribution: https://github.com/ZIB-AOPT/QOBLIB.

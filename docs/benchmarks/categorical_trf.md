# Categorical TRF: constrained mirror-flow variant

`torch_categorical_trf` is an experimental TRF-inspired solver, not an equivalent
implementation of the upstream binary angular TRF. It shares schedule, Euler/Heun
integration, seeded trajectory batching, sparse execution, CUDA graph capture,
checkpoint scoring and multi-device scheduling infrastructure with binary TRF.
Its constrained potential and state representation are different. No claim of
superior quality or speed is made before measurements are available.

## Model and equations

Every variable must belong to exactly one nonempty, disjoint one-hot group.
Ragged and noncontiguous groups are supported. The existing categorical compiler
folds diagonal terms into the linear objective, aggregates duplicate/reversed
edges, removes within-group off-diagonal terms, and subtracts a common linear
shift per group. These operations preserve all feasible source energies up to
a constant. The compiler also computes its usual group coloring, although this
flow does not use colors.

Let `l` be the resulting linear objective and `A` its symmetric cross-group
adjacency. Both are divided by `max_i(abs(l_i) + sum_j(abs(A_ij)))`, or 1 for a
zero objective, separately for each problem. All transformations precede the
float32 cast. Source coefficients and source-energy scoring remain float64.

Logits `z` define `p = softmax(z)` independently within each group. Define
`q = p * (1-p)` and the time-varying potential:

```text
F(p, kappa) = l.T p + 0.5 p.T A p
             - 0.5 route_strength q.T A q - 0.5 kappa p.T p

g = l + A p - route_strength (1-2p) * (A q) - kappa p
dz_i/dt = -mobility * (g_i - sum_{j in group(i)} p_j g_j)
```

This induces replicator/mirror descent on each probability simplex in the
continuous-time limit. The route term vanishes at categorical vertices. Negative
initial kappa discourages early concentration; positive final kappa encourages
categorical decisions. Finite Euler/Heun steps are heuristic, not guaranteed to
decrease the source objective. This is a newly defined categorical analogue,
not the original cosine/sine TRF differential equation.

Logits start independently uniform in `[-0.1, 0.1]` using stable problem/run seed
streams. After each update, group maxima are subtracted and logits floored at
-80 (float32) or -600 (float64) to limit saturation. Category indices are updated
at every integration step by group argmax, with ties choosing the first variable
in the declared group order. Checkpoint bitstrings are materialized directly
from these category indices. Thus every discrete candidate is one-hot; there is
no binary thresholding followed by projection or categorical descent. Continuous
probabilities are a simplex relaxation, not binary samples.

Initial, periodic and final candidates are ranked by the original float64 QUBO.
The shared accumulator remains the source-energy boundary. No binary gamma
channel is defined for this variant, so nonzero `gamma` is rejected. No greedy
polishing sweeps are added. Defaults otherwise match `torch_transverse_route`.

## Integration and limits

The existing generic `BQMSolverConfig` and strict YAML composition boundary accept
the newly registered name without schema changes. Use constructor `device` or
`devices` and per-call `solverParameters` as with binary TRF. Resident execution
uses the authoritative host snapshot to compile the categorical objective, then
solves on the configured single device. Dense/sparse selection and safety limits
follow binary TRF. Memory admission additionally accounts for padded group tables;
very unequal group sizes can require substantial padding.

The Group 1 configuration is
`experiments/group1_categorical_trf_20260910/categorical_trf.yaml`:
8590 stocks, 2025-07-08, 105 PCA risk scenarios, runs=4, steps=10000, float32,
eight RTX 5090 GPUs. The earlier source one-hot penalty of 1.0 is retained for
comparability, although it vanishes for feasible samples and is removed from
dynamics. No existing experiment YAML or completed result is overwritten.

```bash
bash tools/track_group1_categorical_trf.sh status
bash tools/track_group1_categorical_trf.sh logs
bash tools/track_group1_categorical_trf.sh fetch
```

The supervised remote process is `group1_categorical_trf`, under
`/workspace/group1-categorical-trf-20260910`. Logs include scenario generation,
per-device solving and checkpoint scoring. The runner aborts if candidate repair
is invoked. Results include `benchmark_single_date.csv`, `report.json`, the
configuration snapshot and live `status.json`. Ctrl-C on the log viewer does not
stop the calculation. A stopped calculation restarts the whole day, not a partial
scenario checkpoint. The BiqMac sweep on the other server is unaffected.

The requested full-day calculation is the first application measurement. Unit,
finite-difference, CPU/GPU parity and graph/eager equivalence tests have not been
run for this new variant; they remain necessary before treating it as validated
research software. A completed margin run alone does not establish optimality
or numerical equivalence to either binary TRF or categorical heat-bath annealing.
GPU measurement methodology follows the `optimize-for-gpu` guidance credited in
`docs/benchmarks/transverse_route.md`.

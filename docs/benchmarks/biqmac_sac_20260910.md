# Discrete SAC for BiqMac solver configuration selection

Date: 2026-09-10. Status: proposed experiment; no measurements reported.

Recommend **discrete Soft Actor-Critic (SAC)** as a model-free alternative to the existing EXP3-IX configuration selector. SAC combines a stochastic actor, learned value critics, and entropy regularization with off-policy learning. The original evidence concerns continuous-control benchmarks, not BiqMac or these solvers. [Haarnoja et al., 2018](https://proceedings.mlr.press/v80/haarnoja18b.html)

Use the discrete-action formulation for a categorical policy over a finite configuration bank. Christodoulou derives this extension and evaluates it on Atari; that evidence motivates a candidate method, not a performance prediction here. [Christodoulou, 2019](https://arxiv.org/abs/1910.07207)

## Fixed experimental protocol

| Component | Required design |
| --- | --- |
| Solvers | Separate selectors for SBM, SVL, and TRF. |
| Actions | One fixed bank of 32 joint parameter configurations per solver. Each action selects a complete parameter tuple. Include the solver default and archive all tuples before training. |
| Solver calls | Every configuration and control uses `runs=32`, `steps=10000`. The 32 trajectories are distinct from the 32 bank actions. |
| Episode | One instance, one 20-second best-by-deadline window; repeatedly select configurations and retain the best eligible result. |
| Training | 200 training episodes per solver, sampled only from the training partition. Record selector and solver seeds and actual compute usage. |
| Split | Grouped 110/39/29 train/validation/test split across 178 instances. Keep related-instance groups intact; archive membership before training. |
| Selection | Use the 39 validation instances for checkpoint/hyperparameter selection; keep the 29 test instances untouched until final evaluation. |
| Evaluation | Freeze policies and preprocessing, then evaluate all 178 instances with three paired evaluation seeds per solver and method. Disable learning and replay updates. |
| Controls | Repeated default configuration and uniform random selection from the identical 32-action bank, using the same call settings, hardware, and deadline accounting. |

The finite bank makes this **discrete configuration selection, not continuous parameter optimization**. Freeze bank construction using predefined ranges or training data only; validation/test outcomes must not shape the bank.

## Learning and measurement

Proposed observations are inexpensive instance summaries, remaining time, incumbent energy, and prior-call outcomes. Use a terminal reward based on negative normalized best energy, with normalization fixed from training data. SAC learns action-value critics for expected return; it learns **no QUBO landscape or transition model**. Its off-policy replay permits reuse of collected experience, but guarantees neither superiority over EXP3-IX nor sample efficiency within 200 episodes on these tasks. [Haarnoja et al., 2018](https://proceedings.mlr.press/v80/haarnoja18b.html)

Count policy inference, transfers, solver execution, synchronization, and scoring within each evaluation window. Admit only results available by the deadline; exclude late completions and record windows without an eligible result. Log training updates and setup costs separately, and apply identical warm-up rules to controls.

Report original-QUBO best energy, reference gaps where available, valid-result counts, completed calls, and paired differences against both controls. Summarize three-seed variability and each split separately: evaluation on all 178 is descriptive, while the untouched 29-instance test partition supports the generalization claim. Archive hardware, software, banks, split, seeds, and timing rules.

## Interpretation limits

Critic error, reward scaling, limited training coverage, and configuration-bank quality can limit performance. Treat SAC as an empirical alternative, not an established improvement. GPU hardware and budgets differ from the older EXP3 experiment, so historical differences do not support a causal direct comparison. A claim that SAC improves on EXP3-IX requires rerunning EXP3-IX with matched hardware, banks, splits, budgets, seeds, and deadline accounting.

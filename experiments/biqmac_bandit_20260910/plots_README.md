# EXP3-IX plots and solver comparison

Generate the dashboard, seven PNG charts, PDF and analysis CSVs from downloaded
results without rerunning any solvers:

```bash
MPLCONFIGDIR=/tmp/margin-matplotlib .venv/bin/python tools/plot_biqmac_bandit.py
```

Open `plots/index.html` or `plots/bandit_plots.pdf`. The dashboard leads with the
29 held-out test graphs and compares all three solvers under learned, default
and uniform-random parameter selection. Interactive per-graph tables also expose
validation and training partitions, labeled as in-sample.

Charts separate mean/median/p90 reference gaps, tie-split solver win credit,
paired learning effects, anytime mean gap, observed time to a 1% reference gap,
per-graph quality, training/validation history and deadline overruns.

The anytime and target curves come from saved episode improvement histories,
not inference from final solution times. Target success rates retain unsuccessful
episodes in their denominators. Median target times in CSVs/tables condition on
success and must be read alongside success rates. These are not statistical
restart time-to-solution estimates. All-zero incumbents and late-call rejection
follow the original experiment's scoring contract.

Comparisons pair graph and seed; quality ties split credit within each parameter
selection method. Positive paired gap reduction means the learned method is
better. Per-graph table highlights compare seed-mean cuts, a different aggregation
from episode-level win credit. No significance claim is made. Training rolling
means use changing graph subsets and are not fixed-problem learning curves.

This experiment uses 32 trajectories, SBM/SVL 5,000 steps and TRF 10,000 steps on
RTX 5090 GPUs. It is not the SAC experiment with all solvers at 10,000 steps.
The downloaded experiment is complete: 4,806 evaluation episodes, plus 600
training and 351 validation episodes. Plotting adds no solver changes. No separate
unit tests or visual-validation pass were requested or run.

# SAC solver-comparison plots

```bash
MPLCONFIGDIR=/tmp/margin-matplotlib .venv/bin/python tools/plot_biqmac_bandit.py --learner sac --results experiments/biqmac_sac_20260910/results
```

Open `plots/index.html` for the interactive report or `plots/sac_plots.pdf` for
all seven charts. Tables and CSVs retain method identifiers `sac`, `default`,
and `random`; the original results are not modified.

The main comparison uses 29 held-out graphs and three evaluation seeds. Training
and validation partitions remain separately labeled. All three solvers use
32 runs, 10,000 steps and 20-second best-by-deadline windows on eight RTX 2080 Ti
GPUs. The complete experiment contains 4,806 evaluation episodes.

Charts include quality/win comparisons, paired improvements over controls,
anytime quality, observed time to a 1% reference gap, per-problem gaps, training
and checkpoint history, and deadline diagnostics. Target timing is reconstructed
from saved improvement curves; unsuccessful episodes remain in success-rate
denominators. Conditional median hit times are not standalone speed rankings.
Late results receive no quality credit. No statistical significance is claimed.

No solver jobs are started or stopped by plotting. No separate tests or visual
validation are run. The proposed correlation-scaled margin change was canceled
before implementation or execution.

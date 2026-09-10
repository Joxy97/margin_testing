# BiqMac extended-step sweep

Stopped at the user's request; the partial remote results have been fetched into
`results/`. The independent RL experiment on the RTX 5090 host was left untouched.

Generate the partial-result dashboard, PNGs, PDF and coverage statistics with:

```bash
.venv/bin/python tools/plot_biqmac_results.py --results experiments/biqmac_maxcut_long_20260910/results --expected-instances 178
```

Open `plots/index.html` for the dashboard or `plots/biqmac_plots.pdf` for all charts.
Grey cells mean no completed comparison, not a solver loss. The coverage CSV
reports completed versus planned graphs per setting. Aggregates use available
complete comparisons; different settings can have different graph subsets.

SBM, SVL and Transverse Route on all 178 previously prepared BiqMac MaxCut graphs.
No reinforcement learning or parameter tuning is used.

- Runs: 2, 4, 8, 16, 32.
- Steps: 10000, 20000, 40000.
- Full Cartesian product: 15 settings, 2670 graph/setting comparisons.
- Three fixed-seed timing repetitions per solver: 24030 timed solves; seed 1.
- Eight RTX 2080 Ti GPUs on `45.59.100.176:43437`.
- The original solver defaults, float32 dynamics, scoring and winner rules are unchanged.

## Progress and results

From the repository root:

```bash
bash tools/track_biqmac_long.sh status
bash tools/track_biqmac_long.sh logs
bash tools/track_biqmac_long.sh fetch
```

`logs` follows live output. Ctrl-C closes the viewer without stopping the job.
`fetch` synchronizes CSVs, checkpoints, metadata and logs into this directory's
`results/`. Downloads are explicit, not continuous background synchronization.
SSH prompts for the existing private key passphrase when needed.

Remote project: `/workspace/biqmac-maxcut-20260909`.
New output directory: `/workspace/biqmac-maxcut-20260909/results_long_20260910`.
Supervisor program: `biqmac_maxcut_long`. It survives disconnecting SSH.
The original sweep and its results are not changed.

From a shell on the server:

```bash
supervisorctl status biqmac_maxcut_long
supervisorctl stop biqmac_maxcut_long
supervisorctl start biqmac_maxcut_long
```

Restart resumes completed checkpoints, provided inputs/settings/source hashes match.
The job creates `runs_<runs>_steps_<steps>.csv` for each setting plus `summary.csv`
and `overall.csv`. Header-only files indicate settings not yet reached, not missing
graphs. A local snapshot can lag behind the server until the next fetch.

Cuts are maximized. Quality ties are retained; `winner` breaks quality ties by
median complete-solve time, while `fastest_solver` measures speed alone. Timings
are not time-to-first-target. See `../biqmac_maxcut_20260909/README.md` for the
shared encoding, reference data, timing and quality methodology.

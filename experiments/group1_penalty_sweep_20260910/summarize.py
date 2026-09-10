"""Render completed sweep measurements without treating missing solves as results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE=Path(__file__).resolve().parent


def main():
    summaries=json.loads((HERE/'results/summary.json').read_text())
    rows=json.loads((HERE/'results/rows.json').read_text())
    preparation=json.loads((HERE/'qubos_manifest.json').read_text())
    manifest=json.loads((HERE/'portfolio_manifest.json').read_text())
    scenario_bounds={r['scenario']:r['greedy_margin'] for r in preparation['scenarios']}
    for summary in summaries:
        matched=[r for r in rows if r['trial']==summary['trial'] and r['status']=='solved']
        if len(matched)!=summary['solved'] or len({r['scenario'] for r in matched})!=len(matched):
            raise ValueError('Trial counts or scenario identities do not match')
        if summary['complete'] and {r['scenario'] for r in matched}!=set(range(105)):
            raise ValueError('A complete trial must cover all 105 scenarios')
        for row in matched:
            if not 0.<=row['loss']<=scenario_bounds[row['scenario']]+1e-10:
                raise ValueError('Decoded loss exceeds the independent per-asset bound')
            if row['penalty_setting']=='bound' and row['lambdaOneHot']<=row['sufficient_bound']:
                raise ValueError('The strong penalty must strictly exceed its bound')
    out=HERE/'plots';out.mkdir(exist_ok=True)
    solvers=['SBM','SVL','TRF']
    variants=[('legacy','1'),('pruned','1'),('pruned','10'),('pruned','bound')]
    labels=['Previous terms\nλ = 1','Normalized + pruned\nλ = 1','Normalized + pruned\nλ = 10','Normalized + pruned\nλ > row bound']
    colors=['#4263eb','#0b8f87','#d97425']
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    fields=[('wall_seconds','Complete trial wall time (s)'),
            ('repair_seconds','Repair time summed over workers (s)'),
            ('repair_calls','Number of repair calls'),
            ('margin','Worst decoded portfolio loss')]
    for ax,(field,title) in zip(axes.flat,fields):
        for i,(solver,color) in enumerate(zip(solvers,colors)):
            selected=[next((r for r in summaries if (r['solver'],r['variant'],r['penalty_setting'])==(solver,*variant)),None) for variant in variants]
            values=[]
            for r in selected:
                if r is None or (field=='margin' and not r['complete']): values.append(np.nan)
                else: values.append(r[field])
            bars=ax.bar(np.arange(4)+(i-1)*.24,values,width=.23,label=solver,color=color)
            for bar,r in zip(bars,selected):
                if r is not None and not r['complete']: bar.set_hatch('//')
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].axhline(300,color='#a22',linestyle='--',linewidth=1,label='300 s cap')
    axes[0,0].legend()
    fig.suptitle('Group 1: 8,590 stocks · 105 scenarios · binary solvers',fontsize=16)
    fig.savefig(out/'penalty_sweep.png',dpi=180)
    fig.savefig(out/'penalty_sweep.pdf')
    raw=sum(r['raw_correlation_edges'] for r in preparation['scenarios'])
    retained=sum(r['retained_correlation_edges'] for r in preparation['scenarios'])
    report=[
        '# Group 1 correlation normalization and one-hot penalty sweep',
        '',
        f"Random long portfolio: {manifest['assets']:,} stocks, gross exposure 1, seed {manifest['seed']}. "
        f"Margin date: {manifest['marginDate']}. All trials use the same 105 prepared scenarios.",
        '',
        f"Hardware: 8 × {preparation['hardware'][0]}; PyTorch {preparation['torch']}, CUDA {preparation['cuda']}. "
        f"Preparation took {preparation['seconds']:.1f} s and is excluded from the solve budget.",
        '',
        f"Correlation interactions fell from {raw:,} to {retained:,} across the 105 QUBOs "
        f"({100*(1-retained/raw):.2f}% removed). Variable counts and one-hot groups are unchanged.",
        '',
        '| Solver | Correlation terms | One-hot penalty | Solved | Wall s | Repair calls | Repair s¹ | Raw feasible | Margin² |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summaries:
        fraction=100*r['raw_feasible']/max(1,r['raw_candidates'])
        margin=f"{r['margin']:.8f}" if r['complete'] else 'incomplete'
        report.append(f"| {r['solver']} | {r['variant']} | {r['penalty_setting']} | {r['solved']}/{r['requested']} | "
            f"{r['wall_seconds']:.2f} | {r['repair_calls']} | {r['repair_seconds']:.2f} | {fraction:.2f}% | {margin} |")
    report += ['', '## Measured effect', '']
    for solver in solvers:
        selected={ (r['variant'],r['penalty_setting']):r for r in summaries if r['solver']==solver }
        if ('legacy','1') in selected and ('pruned','1') in selected:
            old,new=selected['legacy','1'],selected['pruned','1']
            report.append(f"- {solver}: normalization/pruning reduced total trial time "
                f"{old['wall_seconds']/new['wall_seconds']:.2f}× and summed repair time "
                f"{old['repair_seconds']/new['repair_seconds']:.2f}× at λ=1. "
                f"Repair calls changed from {old['repair_calls']} to {new['repair_calls']}.")
    strong=[r for r in rows if r['penalty_setting']=='bound' and r['status']=='solved']
    if strong:
        report += ['',f"The sufficient per-QUBO penalties ranged from {min(r['lambdaOneHot'] for r in strong):.3f} "
            f"to {max(r['lambdaOneHot'] for r in strong):.3f}. "
            f"At these penalties, {sum(r['raw_feasible'] for r in strong)} of "
            f"{sum(r['raw_candidates'] for r in strong)} raw candidates were fully one-hot feasible."]
    report += ['',
        '¹ Repair seconds are summed across eight concurrently running workers, so they can exceed wall time. '
        'Sparse repair-model construction is recorded separately in the raw results.',
        '² Margin is the greatest decoded loss among the 105 returned feasible samples, as a fraction of unit exposure. '
        'It is a heuristic result, not a certified optimum. The previous and modified correlation objectives have different meanings.',
        '',
        '## Method and limits',
        '',
        '- Each solver/penalty trial has a 300 s global deadline. Transfers, solver preparation, integration, candidate checking, '
        'repair, source-energy validation, and result publication fall inside that wall-clock window. '
        'Startup, tiny GPU warmups, file preload, reference-feature preparation, and scenario generation are outside it.',
        '- Each GPU owns a fixed scenario shard; every scenario receives a share of its remaining deadline. '
        'Incomplete or late results are not counted as solved. The coordinator terminates its own workers at the global deadline.',
        '- The sufficient penalty is max(1, nextafter(1.01 × max_i(|a_i| + sum_j |b_ij|), +∞)), '
        'computed from the nonpenalty objective. It guarantees feasible global minimizers, not feasible finite-step trajectories.',
        '- Frozen BiqMac SAC actors control TRF and SVL. SBM uses defaults after its transferred actor worsened the pilot mean objective. '
        'Reference features use the normalized/pruned λ=1 QUBO and remain fixed across the sweep. '
        'Actor inference is timed, but no new SAC training is performed and no out-of-distribution generalization is assumed.',
        '- The saved actor may choose algorithm parameters; execution is constrained to float32, sparse matrices, '
        'the recorded step/trajectory counts, and bounded candidate storage. TRF retains initial/final candidates. '
        'The standard deterministic categorical repair remains enabled; none of the solvers searches categories directly.',
        '- This is one seeded portfolio and one run per setting. The early pilot overlapped CPU scenario preparation; '
        'the full sweep begins after preparation completes. Timing differences are descriptive, not confidence intervals.',
        '- Candidate statistics are taken after the solver\'s existing checkpoint deduplication. '
        'The added feasibility counters and timing instrumentation remain inside the measured execution window.',
        '- Sufficient penalties may be hundreds of times larger than individual portfolio-return coefficients. '
        'Float32 dynamics can lose small distinctions at those scales; authoritative source scoring and repair use float64.',
        '',
        '## Artifacts',
        '',
        '- [Plot](plots/penalty_sweep.png) · [PDF](plots/penalty_sweep.pdf)',
        '- [Aggregate CSV](results/summary.csv) · [Per-scenario records](results/rows.json)',
        '- [Configuration](configuration.yaml) · [Portfolio](portfolio.csv) · [Input identity](portfolio_manifest.json)',
        '- [Prepared QUBO identities](qubos_manifest.json) · [SAC selection](sac_selection.json)',
        '',
        '## Reproduce',
        '',
        'From the repository root with the recorded dependencies and eight GPUs:',
        '',
        '```bash',
        'PYTHONPATH=src:tools python experiments/group1_penalty_sweep_20260910/prepare.py',
        'PYTHONPATH=src:tools python experiments/group1_penalty_sweep_20260910/run_sweep.py --steps 512 --runs 1 --budget 300 --policy selected',
        'python experiments/group1_penalty_sweep_20260910/summarize.py',
        '```',
        '',
        'Prepared QUBO arrays are retained on the benchmark instance under '
        '`/workspace/margin-penalty-sweep/experiments/group1_penalty_sweep_20260910/qubos/`. '
        'The local archive contains hashes, per-scenario results, and returned samples.',
        '',
    ]
    (HERE/'README.md').write_text('\n'.join(report))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Group 1 penalty sweep</title>'
        '<style>body{font-family:system-ui;max-width:1400px;margin:32px auto;padding:0 24px}img{width:100%}</style>'
        '<h1>Group 1 penalty sweep</h1><p>8,590 stocks · 105 scenarios · 300-second budget per solver setting.</p>'
        '<img src="penalty_sweep.png" alt="Solver timings, repair time, repair counts and decoded loss">'
        '<p><a href="../results/summary.csv">Aggregate CSV</a> · <a href="penalty_sweep.pdf">PDF</a> · <a href="../README.md">Method and results</a></p>')


if __name__=='__main__':main()

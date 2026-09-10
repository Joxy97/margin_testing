#!/usr/bin/env python3
"""Plot frozen bandit evaluation, paired controls and observed anytime curves."""

import argparse
import base64
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm
import numpy as np

from plot_biqmac_results import COLORS, SOLVERS, heading, style


METHODS = ("learned", "default", "random")
LABELS = {"learned": "EXP3-IX", "default": "Defaults", "random": "Uniform random"}
LEARNED_METHOD = "learned"
PLOT_PDF = "bandit_plots.pdf"
SPLITS = ("test", "validation", "train")
FLOATS = ("best_cut", "reference_cut", "relative_gap", "best_time_s",
          "search_elapsed_s", "cleanup_overrun_s", "budget_s")
INTS = ("seed", "checkpoint", "calls", "accepted_calls", "failed_calls", "runs", "steps")


def read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in FLOATS:
            row[key] = float(row[key])
        for key in INTS:
            row[key] = int(row[key])
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    grouped = defaultdict(list)
    contests = defaultdict(dict)
    for row in rows:
        grouped[row["split"], row["solver"], row["method"]].append(row)
        contests[row["split"], row["instance"], row["seed"], row["method"]][row["solver"]] = row["best_cut"]
    summaries = []
    for (split, solver, method), subset in sorted(grouped.items()):
        points, matched = 0., 0
        for row in subset:
            contest = contests[split, row["instance"], row["seed"], method]
            if len(contest) != len(SOLVERS):
                continue
            matched += 1
            best = max(contest.values())
            winners = [s for s, cut in contest.items() if abs(cut - best) <= 1e-8]
            if solver in winners:
                points += 1. / len(winners)
        gaps = [100 * r["relative_gap"] for r in subset]
        accepted = [r["best_time_s"] for r in subset if r["accepted_calls"] > 0]
        item = dict(split=split, solver=solver, method=method, episodes=len(subset),
                    instances=len({r["instance"] for r in subset}),
                    mean_gap_percent=float(np.mean(gaps)), median_gap_percent=float(np.median(gaps)),
                    p90_gap_percent=float(np.percentile(gaps, 90)),
                    matched_solver_contests=matched, quality_win_credit_percent=100 * points / matched if matched else None,
                    median_best_time_s=float(np.median(accepted)) if accepted else None,
                    mean_occupied_time_s=float(np.mean([r["search_elapsed_s"] + r["cleanup_overrun_s"] for r in subset])),
                    median_accepted_calls=float(np.median([r["accepted_calls"] for r in subset])),
                    zero_accepted_episodes=sum(r["accepted_calls"] == 0 for r in subset),
                    failed_calls=sum(r["failed_calls"] for r in subset),
                    late_episode_percent=100 * sum(r["cleanup_overrun_s"] > 0 for r in subset) / len(subset),
                    mean_cleanup_overrun_s=float(np.mean([r["cleanup_overrun_s"] for r in subset])),
                    max_cleanup_overrun_s=max(r["cleanup_overrun_s"] for r in subset))
        for target in (0, 1, 5):
            times = [r[f"target_{target}_s"] for r in subset if r[f"target_{target}_s"] is not None]
            item[f"target_{target}_hit_percent"] = 100 * len(times) / len(subset)
            item[f"target_{target}_median_hit_s"] = float(np.median(times)) if times else None
        summaries.append(item)
    return summaries


def paired_effect(rows):
    index = {(r["split"], r["instance"], r["seed"], r["solver"], r["method"]): r for r in rows}
    effects = []
    for split in SPLITS:
        for solver in SOLVERS:
            for baseline in ("default", "random"):
                delta = []
                for row in rows:
                    if row["split"] != split or row["solver"] != solver or row["method"] != LEARNED_METHOD:
                        continue
                    other = index.get((split, row["instance"], row["seed"], solver, baseline))
                    if other is not None:
                        delta.append(100 * (other["relative_gap"] - row["relative_gap"]))
                if delta:
                    effects.append(dict(split=split, solver=solver, baseline=baseline, matched_episodes=len(delta),
                        mean_gap_reduction_pp=float(np.mean(delta)), median_gap_reduction_pp=float(np.median(delta)),
                        learned_wins=sum(d > 1e-8 for d in delta), ties=sum(abs(d) <= 1e-8 for d in delta),
                        learned_losses=sum(d < -1e-8 for d in delta)))
    return effects


def quality_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(top=.84, bottom=.09, hspace=.43, wspace=.25)
    heading(fig, "Held-out test: quality and wins", "29 unseen graphs x 3 evaluation seeds. Quality ties split one point; lower gaps are better.")
    index = {(r["solver"], r["method"]): r for r in summary if r["split"] == "test"}
    for ax, key, title in zip(axes.flat,
        ("mean_gap_percent", "median_gap_percent", "p90_gap_percent", "quality_win_credit_percent"),
        ("Mean reference gap (%)", "Median reference gap (%)", "90th-percentile reference gap (%)", "Best-cut credit within each method (%)")):
        x = np.arange(3)
        for j, method in enumerate(METHODS):
            bars = ax.bar(x + (j - 1) * .25, [index[s, method][key] for s in SOLVERS], width=.24,
                          color=[COLORS[s] for s in SOLVERS], alpha=(1., .55, .28)[j],
                          hatch=(None, "//", "..")[j], label=LABELS[method])
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xticks(x, SOLVERS)
        ax.set_title(title, loc="left", fontsize=12)
        ax.margins(y=.22)
        ax.grid(axis="y", alpha=.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
    return fig


def effect_figure(effects):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=.77, bottom=.18, wspace=.3)
    heading(fig, "Did learning beat the controls?", f"Paired by held-out graph and seed. Positive = {LABELS[LEARNED_METHOD]} improves quality; negative = the control is better.")
    for ax, baseline in zip(axes, ("default", "random")):
        subset = {r["solver"]: r for r in effects if r["split"] == "test" and r["baseline"] == baseline}
        values = [subset[s]["mean_gap_reduction_pp"] for s in SOLVERS]
        bars = ax.barh(SOLVERS, values, color=[COLORS[s] for s in SOLVERS])
        ax.bar_label(bars, fmt="%+.3f", padding=5, fontsize=10)
        ax.axvline(0, color="#23333c", linewidth=1)
        ax.margins(x=.25)
        ax.set_title(f"{LABELS[LEARNED_METHOD]} versus {LABELS[baseline]}", loc="left")
        ax.set_xlabel("Mean reference-gap reduction (percentage points)")
        ax.grid(axis="x", alpha=.3)
    fig.text(.07, .05, "These are descriptive paired differences, not confidence intervals or proof of statistical significance.", fontsize=9)
    return fig


def anytime_figure(rows, grid, target=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(top=.77, bottom=.19, wspace=.25)
    heading(fig, "How soon is a good solution available?" if target else "Quality as the 20-second window unfolds",
            "Held-out graphs only. Target: reference gap <= 1%. Unsuccessful episodes remain in the denominator." if target else
            "Held-out graphs only. Incumbents reconstructed from saved on-time improvement curves; lower is better.")
    for ax, method in zip(axes, METHODS):
        for solver in SOLVERS:
            subset = [r for r in rows if r["split"] == "test" and r["method"] == method and r["solver"] == solver]
            if target:
                y = 100 * np.mean([[r["target_1_s"] is not None and r["target_1_s"] <= t for t in grid] for r in subset], axis=0)
            else:
                y = np.mean([r["anytime_gap"] for r in subset], axis=0)
            ax.step(grid, y, where="post", color=COLORS[solver], label=solver, linewidth=2)
        ax.set_title(LABELS[method], loc="left")
        ax.set_xlabel("Seconds from episode start")
        ax.set_xlim(0, grid[-1])
        ax.set_ylabel("Episodes reaching target (%)" if target else "Mean reference gap (%)")
        if target:
            ax.set_ylim(0, 101)
        else:
            ax.set_yscale("symlog", linthresh=.01)
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
    fig.text(.07, .05, "Observed first target hit, not statistical restart TTS. Late completions are excluded." if target else
             "Gap axis is linear below 0.01%, logarithmic above. Zero-cut fallback is available at time zero.", fontsize=9)
    return fig


def instance_figure(rows):
    subset = [r for r in rows if r["split"] == "test"]
    names = sorted({r["instance"] for r in subset})
    fig, ax = plt.subplots(figsize=(14, max(9, len(names) * .29 + 3)))
    fig.subplots_adjust(top=.84, left=.22, bottom=.13, right=.91)
    heading(fig, "Every held-out problem", "Mean reference gap (%) across three seeds. Smaller values are better; each graph has equal weight.")
    columns = [(m, s) for m in METHODS for s in SOLVERS]
    matrix = np.array([[100 * np.mean([r["relative_gap"] for r in subset if
        r["instance"] == name and r["method"] == m and r["solver"] == s]) for m, s in columns] for name in names])
    norm = SymLogNorm(linthresh=.1, vmin=0, vmax=max(1., float(matrix.max())))
    plot = ax.imshow(matrix, norm=norm, cmap="YlOrRd", aspect="auto")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if norm(value) > .7 else "#23333c")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xticks(range(len(columns)), [f"{LABELS[m]}\n{s}" for m, s in columns], fontsize=9)
    ax.axvline(2.5, color="white", linewidth=3)
    ax.axvline(5.5, color="white", linewidth=3)
    fig.colorbar(plot, ax=ax, fraction=.035, pad=.025, label="Gap %; nonlinear color scale")
    return fig


def training_figure(training, validation):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(top=.84, bottom=.12, hspace=.43, wspace=.3)
    heading(fig, "Training history and checkpoint selection", "Top: rolling 20-episode training gap, with changing graph mix. Bottom: fixed validation set, one seed.")
    for j, solver in enumerate(SOLVERS):
        subset = sorted((r for r in training if r["solver"] == solver), key=lambda r: r["seed"])
        values = np.array([100 * r["relative_gap"] for r in subset])
        rolling = [float(np.mean(values[max(0, i - 19):i + 1])) for i in range(len(values))]
        axes[0, j].plot(np.arange(1, len(values) + 1), rolling, color=COLORS[solver])
        axes[0, j].set_title(solver, loc="left", color=COLORS[solver])
        axes[0, j].set_xlabel("Training episode")
        checkpoints = sorted({r["checkpoint"] for r in validation if r["solver"] == solver})
        gaps = [100 * np.mean([r["relative_gap"] for r in validation if r["solver"] == solver and r["checkpoint"] == cp]) for cp in checkpoints]
        axes[1, j].plot(checkpoints, gaps, "o-", color=COLORS[solver])
        chosen = int(np.argmin(gaps))
        axes[1, j].scatter([checkpoints[chosen]], [gaps[chosen]], s=140, facecolors="none", edgecolors="#23333c", linewidths=2)
        axes[1, j].set_title(f"Selected checkpoint: {checkpoints[chosen]}", loc="left", fontsize=11)
        axes[1, j].set_xticks(checkpoints)
        axes[1, j].set_xlabel("Episodes per solver")
        for ax in axes[:, j]:
            ax.set_ylabel("Mean reference gap (%)")
            ax.grid(alpha=.3)
    fig.text(.07, .03, "The training trace is not a fixed-problem learning curve. Only held-out evaluation measures generalization.", fontsize=9)
    return fig


def reliability_figure(summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(top=.77, bottom=.19, wspace=.28)
    heading(fig, "Deadline and throughput diagnostics", "Held-out episodes. Faster calls can permit more restarts; the quality objective remains best cut by 20 seconds.")
    index = {(r["solver"], r["method"]): r for r in summary if r["split"] == "test"}
    for ax, key, title in zip(axes, ("late_episode_percent", "mean_cleanup_overrun_s", "median_accepted_calls"),
        ("Episodes with cleanup overruns (%)", "Mean extra occupied time (s)", "Median accepted calls per episode")):
        for j, method in enumerate(METHODS):
            ax.bar(np.arange(3) + (j - 1) * .25, [index[s, method][key] for s in SOLVERS], width=.24,
                   color=[COLORS[s] for s in SOLVERS], alpha=(1., .55, .28)[j], hatch=(None, "//", "..")[j], label=LABELS[method])
        ax.set_xticks(range(3), SOLVERS)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    fig.text(.07, .05, "An in-flight call may finish after 20 seconds, but its result receives no quality credit. No GPU kernel preemption.", fontsize=9)
    return fig


def dashboard(output, summary, effects, rows, figures, metadata):
    per_graph = defaultdict(list)
    for row in rows:
        per_graph[row["split"], row["instance"], row["method"], row["solver"]].append(row)
    data = []
    for (split, name, method, solver), subset in per_graph.items():
        hits = [r["target_1_s"] for r in subset if r["target_1_s"] is not None]
        data.append(dict(split=split, instance=name, method=method, solver=solver, seeds=len(subset),
                         gap=100 * float(np.mean([r["relative_gap"] for r in subset])),
                         cut=float(np.mean([r["best_cut"] for r in subset])),
                         reference=subset[0]["reference_cut"], hits=len(hits),
                         target_s=float(np.median(hits)) if hits else None))
    cards = []
    for solver in SOLVERS:
        row = next(s for s in summary if s["split"] == "test" and s["method"] == LEARNED_METHOD and s["solver"] == solver)
        effect = next(s for s in effects if s["split"] == "test" and s["solver"] == solver and s["baseline"] == "default")
        cards.append(f'<article style="--solver:{COLORS[solver]}"><h2>{solver}</h2><strong>{row["mean_gap_percent"]:.3f}%</strong>'
                     f'<p>mean test gap with {LABELS[LEARNED_METHOD]}<br>{row["quality_win_credit_percent"]:.1f}% best-cut credit<br>'
                     f'{row["target_1_hit_percent"]:.1f}% reach a 1% gap<br>{effect["mean_gap_reduction_pp"]:+.3f} pp improvement over defaults</p></article>')
    gallery = []
    for filename, title in figures:
        encoded = base64.b64encode((output / filename).read_bytes()).decode("ascii")
        gallery.append(f'<details {"open" if filename == "test_quality.png" else ""}><summary>{html.escape(title)}</summary>'
                       f'<a href="{filename}">Full-size PNG</a><img loading="lazy" alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"></details>')
    table_rows = "".join(f'<tr><td>{r["split"]}</td><td>{r["solver"]}</td><td>{LABELS[r["method"]]}</td>'
        f'<td>{r["episodes"]}</td><td>{r["mean_gap_percent"]:.3f}%</td><td>{r["median_gap_percent"]:.3f}%</td>'
        f'<td>{r["p90_gap_percent"]:.3f}%</td><td>{r["quality_win_credit_percent"]:.1f}%</td>'
        f'<td>{r["target_1_hit_percent"]:.1f}%</td></tr>' for split in SPLITS for r in summary if r["split"] == split)
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BiqMac EXP3-IX | quality, speed and solver comparison</title>
<style>
:root{--paper:#f8f5ef;--ink:#23333c;--rule:#d6d4ca}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top right,#dcebe2,transparent 50%),var(--paper);color:var(--ink);font:17px Georgia,serif}main{max-width:1450px;margin:auto;padding:40px 28px}h1{font-size:clamp(36px,5vw,68px);font-weight:normal;letter-spacing:-2px;margin:16px 0}h2{font-weight:normal}.eyebrow,select,input,th,td{font-family:ui-monospace,monospace}.eyebrow{font-size:12px;letter-spacing:3px}.intro{max-width:1000px;line-height:1.65}.notice{border-left:4px solid #d97532;padding:14px 20px;background:#ffffff88;line-height:1.6;margin:16px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin:30px 0}article{background:#ffffff88;border-top:5px solid var(--solver);padding:15px 20px}article h2{color:var(--solver);margin-top:3px}article strong{font-size:44px;font-weight:normal}article p{font-size:15px;line-height:1.8}.controls{display:flex;flex-wrap:wrap;gap:18px;margin:22px 0}label{display:grid;gap:7px}select,input{padding:10px;border:1px solid var(--rule);background:white}.table-wrap{overflow:auto;max-height:650px;border:1px solid var(--rule)}table{border-collapse:collapse;width:100%;font-size:12px;background:#ffffff88}td,th{padding:12px;border-bottom:1px solid var(--rule);white-space:nowrap;text-align:left}th{position:sticky;top:0;background:#e6ebe4}td b,td small{display:block;line-height:1.7}.best{background:#dcebdc}a{color:#237d91}section{margin-top:38px}summary{cursor:pointer;font-size:25px;margin:16px 0}details{border-top:1px solid var(--rule);padding:10px 0}img{display:block;width:100%;height:auto;margin-top:15px}footer{font-size:14px;line-height:1.7;margin-top:30px;border-top:1px solid var(--rule);padding-top:18px}@media(max-width:650px){main{padding:24px 14px}.cards{grid-template-columns:1fr}}
</style><main><div class="eyebrow">BIQMAC / EXP3-IX / FROZEN EVALUATION</div><h1>Learning to choose.<br>Solvers compared.</h1>
<p class="intro">__EPISODES__ evaluation episodes on eight RTX 5090 GPUs. Each 20-second window can try multiple solver calls. The headline comparison uses only the 29 held-out test graphs, three seeds each; training and validation results are shown separately.</p>
<div class="notice"><b>Budget:</b> 32 trajectories per call; SBM and SVL use 5,000 steps, TRF uses 10,000. This is the EXP3-IX experiment, not the newer SAC run. <b>Timing:</b> quality is credited only when available within 20 seconds. Time-to-target curves use observed first attainment of a 1% reference gap, not an estimated statistical restart TTS.</div>
<div class="cards">__CARDS__</div><div class="notice">Learning must beat <b>uniform random configuration selection</b>, not only defaults, to demonstrate benefit beyond parameter diversity. Positive paired gap reduction means improvement. No statistical significance claim is made from these descriptive summaries.</div>
<section><h2>Compare every problem</h2><div class="controls"><label>Partition<select id="split"><option value="test">Held-out test</option><option value="validation">Validation (used for selection)</option><option value="train">Training (in-sample)</option></select></label><label>Parameter selection<select id="method"><option value="learned">EXP3-IX</option><option value="default">Defaults</option><option value="random">Uniform random</option></select></label><label>Problem<input id="query" placeholder="Filter instance name"></label></div><p id="count"></p><div class="table-wrap"><table><thead><tr><th>Instance</th><th>Reference</th><th>SBM</th><th>SVL</th><th>TRF</th></tr></thead><tbody id="body"></tbody></table></div><p>Cells show seed-mean cut and gap, target-hit count and median first-hit time among successful seeds. Green marks the best seed-mean cut; this differs from episode-level tie-split win credit.</p></section>
<section><h2>Plots and data</h2><p><a href="bandit_plots.pdf">All charts (PDF)</a> &middot; <a href="solver_method_summary.csv">Solver/method statistics</a> &middot; <a href="paired_effects.csv">Paired learning effects</a> &middot; <a href="episode_metrics.csv">Episode timings and target hits</a> &middot; <a href="coverage.csv">Coverage</a></p>__GALLERY__</section>
<details><summary>All partitions: descriptive summaries</summary><div class="table-wrap"><table><thead><tr><th>Split</th><th>Solver</th><th>Method</th><th>Episodes</th><th>Mean gap</th><th>Median gap</th><th>P90 gap</th><th>Win credit</th><th>1% target hits</th></tr></thead><tbody>__SUMMARY__</tbody></table></div></details>
<footer>Reference gap = 100 x (reference - cut) / |reference|. Lower is better. Means weight graph/seed episodes equally, not absolute graph cut magnitudes. Quality ties split a point within each method. Target success rates include failures; conditional hit times alone must not be used to rank solvers. Zero-cut fallbacks, failed calls and cleanup overruns remain visible in the CSVs. No solvers were rerun to generate these plots. Training wall time and source metadata: <a href="plot_metadata.json">metadata</a>.</footer></main>
<script>const data=__DATA__;const solvers=['SBM','SVL','TRF'];const $=id=>document.getElementById(id);const fmt=(n,d=3)=>Number(n).toLocaleString(undefined,{maximumFractionDigits:d});function render(){const selected=data.filter(r=>r.split===$('split').value&&r.method===$('method').value&&r.instance.toLowerCase().includes($('query').value.toLowerCase()));const names=[...new Set(selected.map(r=>r.instance))].sort();$('count').textContent=`${names.length} graphs shown`;const body=$('body');body.replaceChildren();for(const name of names){const rs=selected.filter(r=>r.instance===name),best=Math.max(...rs.map(r=>r.cut)),tr=document.createElement('tr');function cell(text){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);return td;}cell(name);cell(fmt(rs[0].reference));for(const solver of solvers){const r=rs.find(r=>r.solver===solver);if(!r){cell('No data');continue;}const td=cell(''),b=document.createElement('b'),small=document.createElement('small');b.textContent=`gap ${fmt(r.gap)}% | cut ${fmt(r.cut)}`;small.textContent=`1% target: ${r.hits}/${r.seeds} | ${r.target_s===null?'not reached':fmt(r.target_s)+' s median hit'}`;td.append(b,small);if(Math.abs(r.cut-best)<=1e-8)td.classList.add('best');}body.appendChild(tr);}}for(const id of ['split','method','query'])$(id).addEventListener('input',render);render();</script></html>'''
    if LEARNED_METHOD == "sac":
        document = document.replace("This is the EXP3-IX experiment, not the newer SAC run.",
                                    "This is the discrete SAC experiment.")
        document = document.replace("EXP3-IX", "SAC").replace('value="learned"', 'value="sac"')
        document = document.replace("eight RTX 5090 GPUs", "eight RTX 2080 Ti GPUs")
        document = document.replace("SBM and SVL use 5,000 steps, TRF uses 10,000.",
                                    "SBM, SVL and TRF each use 10,000 steps.")
    document = document.replace("bandit_plots.pdf", PLOT_PDF)
    for token, value in {"__EPISODES__": f"{len(rows):,}", "__CARDS__": "".join(cards),
        "__GALLERY__": "".join(gallery), "__SUMMARY__": table_rows,
        "__DATA__": json.dumps(data, allow_nan=False).replace("<", "\\u003c")}.items():
        document = document.replace(token, value)
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    global LEARNED_METHOD, METHODS, PLOT_PDF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/biqmac_bandit_20260910/results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learner", choices=("exp3", "sac"), default="exp3")
    args = parser.parse_args()
    LEARNED_METHOD = "sac" if args.learner == "sac" else "learned"
    METHODS = (LEARNED_METHOD, "default", "random")
    LABELS[LEARNED_METHOD] = "SAC" if args.learner == "sac" else "EXP3-IX"
    PLOT_PDF = "sac_plots.pdf" if args.learner == "sac" else "bandit_plots.pdf"
    output = args.output or args.results.parent / "plots"
    output.mkdir(parents=True, exist_ok=True)
    evaluation = read_rows(args.results / "evaluation.csv")
    training = read_rows(args.results / "training.csv")
    validation = read_rows(args.results / "validation.csv")
    manifest = json.loads((args.results / "experiment.json").read_text())
    grid = np.linspace(0., float(manifest["arguments"]["budget"]), 201)
    episode_fields = list(evaluation[0])
    for row in evaluation:
        episode = json.loads((args.results / "episodes" / (row["id"] + ".json")).read_text())
        curve = sorted((p for p in episode["curve"] if p["time_s"] <= row["budget_s"]), key=lambda p: p["time_s"])
        times = np.array([0.] + [p["time_s"] for p in curve])
        cuts = np.array([0.] + [p["cut"] for p in curve])
        reference = row["reference_cut"]
        denominator = max(1., abs(reference))
        row["anytime_gap"] = 100 * (reference - cuts[np.searchsorted(times, grid, side="right") - 1]) / denominator
        for target in (0, 1, 5):
            hit = np.flatnonzero(100 * (reference - cuts) / denominator <= target + 1e-8)
            row[f"target_{target}_s"] = float(times[hit[0]]) if len(hit) else None
    summary = aggregate(evaluation)
    effects = paired_effect(evaluation)
    coverage = [dict(split=split, solver=solver, method=method,
                    completed=sum(r["split"] == split and r["solver"] == solver and r["method"] == method for r in evaluation),
                    expected=3 * manifest["split_counts"][split])
                for split in SPLITS for solver in SOLVERS for method in METHODS]
    write_csv(output / "coverage.csv", coverage)
    write_csv(output / "solver_method_summary.csv", summary)
    write_csv(output / "paired_effects.csv", effects)
    write_csv(output / "episode_metrics.csv", [{k: r[k] for k in episode_fields + [f"target_{t}_s" for t in (0, 1, 5)]} for r in evaluation])
    style()
    figures = [("test_quality.png", "Held-out solver quality and wins", lambda: quality_figure(summary)),
               ("learning_effect.png", "Learning versus defaults and uniform random", lambda: effect_figure(effects)),
               ("anytime_quality.png", "Quality throughout the 20-second window", lambda: anytime_figure(evaluation, grid)),
               ("time_to_target.png", "Observed time to a 1% reference gap", lambda: anytime_figure(evaluation, grid, True)),
               ("test_instances.png", "Every held-out graph: all solvers and methods", lambda: instance_figure(evaluation)),
               ("training_validation.png", "Training history and validation checkpoints", lambda: training_figure(training, validation)),
               ("deadline_diagnostics.png", "Deadline overruns and accepted-call throughput", lambda: reliability_figure(summary))]
    with PdfPages(output / PLOT_PDF) as pdf:
        for filename, title, create in figures:
            print(f"Plotting: {title}", flush=True)
            fig = create()
            fig.savefig(output / filename, dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    dashboard(output, summary, effects, evaluation, [(name, title) for name, title, _ in figures],
              {"results": str(args.results), "split_counts": manifest["split_counts"],
               "steps": manifest["steps"], "runs": manifest["runs"], "evaluation_episodes": len(evaluation),
               "coverage": coverage, "target_gap_percent": 1,
               "timing": "Observed first target hit from saved incumbent histories; no statistical restart TTS"})
    print(json.dumps({"report": str(output / "index.html"), "test_learned": [r for r in summary if
        r["split"] == "test" and r["method"] == LEARNED_METHOD],
        "test_paired_effects": [r for r in effects if r["split"] == "test"]}, indent=2))


if __name__ == "__main__":
    main()

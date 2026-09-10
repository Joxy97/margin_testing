"""Plot measured BiqMac solve latency and quality without inferring target-hit TTS."""

import argparse
import base64
import csv
import html
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


SOLVERS = ("SBM", "SVL", "TRF")
COLORS = {"SBM": "#237d91", "SVL": "#d97532", "TRF": "#438653"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments/biqmac_maxcut_20260909/results"
BACKGROUND = "#f8f5ef"
INK = "#23333c"


def load_results(directory):
    rows = []
    seen = set()
    settings = set()
    skipped = 0
    for path in sorted(directory.glob("runs_*_steps_*.csv")):
        parts = path.stem.split("_")
        settings.add((int(parts[1]), int(parts[3])))
        with path.open(newline="") as stream:
            for source in csv.DictReader(stream):
                row = dict(source)
                for key in ("runs", "steps", "vertices"):
                    row[key] = int(row[key])
                identity = row["instance"], row["runs"], row["steps"]
                if identity in seen:
                    raise ValueError(f"Duplicate comparison: {identity}")
                seen.add(identity)
                if row["status"] != "ok":
                    skipped += 1
                    continue
                row["reference_cut"] = float(row["reference_cut"])
                for solver in SOLVERS:
                    prefix = solver.lower()
                    for suffix in ("best_cut", "median_seconds", "gap_percent"):
                        key = f"{prefix}_{suffix}"
                        row[key] = float(row[key])
                        if not np.isfinite(row[key]):
                            raise ValueError(f"Non-finite {key} in {identity}")
                    if row[f"{prefix}_median_seconds"] <= 0:
                        raise ValueError(f"Non-positive solve time in {identity}")
                rows.append(row)
    if not rows:
        raise ValueError(f"No completed comparison rows in {directory}")
    settings = sorted(settings)
    instances = sorted({r["instance"] for r in rows})
    if skipped:
        print(f"Excluded {skipped} incomplete/error comparisons; these are not solver losses.", flush=True)
    return rows, settings, instances


def winners(row):
    return row["quality_winners"].split("|")


def summarize(rows, **labels):
    summaries = []
    for solver in SOLVERS:
        prefix = solver.lower()
        times = np.array([r[f"{prefix}_median_seconds"] for r in rows])
        gaps = np.array([r[f"{prefix}_gap_percent"] for r in rows])
        points = sum(1 / len(winners(r)) for r in rows if solver in winners(r))
        summaries.append(dict(labels, solver=solver, comparisons=len(rows),
            quality_win_points=points, quality_win_percent=100 * points / len(rows),
            fastest_wins=sum(r["fastest_solver"] == solver for r in rows),
            quality_then_time_wins=sum(r["winner"] == solver for r in rows),
            reference_matches=sum(abs(r[f"{prefix}_best_cut"] - r["reference_cut"]) <= 1e-8 for r in rows),
            mean_gap_percent=float(gaps.mean()), median_gap_percent=float(np.median(gaps)),
            p90_gap_percent=float(np.percentile(gaps, 90)),
            median_solve_ms=float(1000 * np.median(times)),
            p10_solve_ms=float(1000 * np.percentile(times, 10)),
            p90_solve_ms=float(1000 * np.percentile(times, 90))))
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbc8c0", "grid.color": "#dfdcd5",
        "axes.titleweight": "bold", "savefig.facecolor": BACKGROUND})


def heading(fig, title, subtitle):
    fig.suptitle(title, x=.07, y=.975, ha="left", fontsize=21, fontweight="bold")
    fig.text(.07, .925, subtitle, ha="left", fontsize=10, color="#586873")


def overview(rows, summaries):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(top=.84, bottom=.09, hspace=.44, wspace=.25)
    heading(fig, "Who wins? Quality and speed are different contests",
            f"{len(rows):,} graph/budget comparisons; equal weight per comparison. Quality ties split one point.")
    metrics = [("quality_win_percent", "Share of best-cut credit", "% of comparisons"),
               ("fastest_wins", "Fastest complete solve", "comparisons"),
               ("reference_matches", "Published reference matched", "comparisons"),
               ("mean_gap_percent", "Mean shortfall from reference", "% gap; lower is better")]
    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        values = [s[key] for s in summaries]
        bars = ax.bar(SOLVERS, values, color=[COLORS[s] for s in SOLVERS], width=.58)
        ax.bar_label(bars, labels=[f"{v:,.1f}" if "percent" in key else f"{v:,.0f}" for v in values], padding=5)
        ax.set_title(title, loc="left", pad=15)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2 or 1)
        ax.yaxis.grid(True, alpha=.6)
        ax.set_axisbelow(True)
    fig.text(.07, .025, "Native solver defaults, not tuned per instance. Mean gaps expose failures that median gaps can hide.", fontsize=9)
    return fig


def budget_heatmaps(summaries, runs, steps):
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.subplots_adjust(top=.86, bottom=.09, left=.09, right=.88, hspace=.6, wspace=.26)
    heading(fig, "The budget map", "Available complete comparisons only; n = graphs per setting. Grey = no completed results. Cohorts can differ.")
    index = {(s["runs"], s["steps"], s["solver"]): s for s in summaries}
    metrics = [("quality_win_percent", "Best-cut credit (%)", "YlGnBu"),
               ("mean_gap_percent", "Mean reference gap (%)", "YlOrRd"),
               ("median_solve_ms", "Median complete solve (ms)", "YlOrRd")]
    for row_number, (key, label, cmap) in enumerate(metrics):
        maximum = max(s[key] for s in summaries)
        minimum = min(0, min(s[key] for s in summaries))
        for col, solver in enumerate(SOLVERS):
            ax = axes[row_number, col]
            matrix = np.array([[index.get((r, t, solver), {}).get(key, np.nan) for t in steps] for r in runs])
            palette = plt.get_cmap(cmap).copy()
            palette.set_bad("#d9d7d0")
            plot = ax.imshow(np.ma.masked_invalid(matrix), cmap=palette, vmin=minimum, vmax=maximum or 1, aspect="auto")
            for (i, j), value in np.ndenumerate(matrix):
                if not np.isfinite(value):
                    ax.text(j, i, "No data", ha="center", va="center", fontsize=9, color=INK)
                    continue
                label_value = f"{value:.1f}" if key != "median_solve_ms" else f"{value:.0f}"
                label_value += f"\nn={index[runs[i], steps[j], solver]['comparisons']}"
                ax.text(j, i, label_value, ha="center", va="center", fontsize=10,
                        color="white" if (value - minimum) / (maximum - minimum or 1) > .65 else INK)
            ax.set_xticks(range(len(steps)), [f"{s:,}" for s in steps])
            ax.set_yticks(range(len(runs)), runs)
            ax.set_xlabel("steps")
            if col == 0:
                ax.set_ylabel("runs")
            ax.set_title(f"{solver} | {label}", color=COLORS[solver], fontsize=10, pad=10)
        fig.colorbar(plot, ax=list(axes[row_number]), fraction=.025, pad=.025)
    fig.text(.09, .025, "More blue = more quality wins. More red = larger error / longer runtime. Each row uses one shared color scale.", fontsize=9)
    return fig


def speed_quality(summaries, runs):
    panel_rows = (len(runs) + 1) // 2
    fig, axes = plt.subplots(panel_rows, 2, figsize=(14, 5 * panel_rows), squeeze=False)
    fig.subplots_adjust(top=.84, bottom=.11, hspace=.36, wspace=.22)
    heading(fig, "Speed versus solution quality", "Each point aggregates available graphs at one budget; cohorts may differ. Lower-left is better; labels show steps.")
    for ax, run in zip(axes.flat, runs):
        for solver in SOLVERS:
            subset = sorted((s for s in summaries if s["runs"] == run and s["solver"] == solver), key=lambda s: s["steps"])
            x = [s["median_solve_ms"] for s in subset]
            y = [s["mean_gap_percent"] for s in subset]
            ax.plot(x, y, "o-", color=COLORS[solver], label=solver, linewidth=2, markersize=6)
            for item, a, b in zip(subset, x, y):
                ax.annotate(f"{item['steps'] // 1000}k", (a, b), xytext=(5, 5), textcoords="offset points", fontsize=8, color=COLORS[solver])
        ax.set_xscale("log")
        ax.set_xlabel("median complete solve (ms; log scale)")
        ax.set_ylabel("mean reference gap (%)")
        ax.set_title(f"{run} trajectories", loc="left")
        ax.grid(True, alpha=.6)
        ax.legend(frameon=False)
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)
    fig.text(.07, .025, "Solve latency is NOT time-to-first-target. No trajectory hit times or independent-seed success probabilities were recorded.", fontsize=9)
    return fig


def instance_map(rows, settings, instances):
    masks = {name: 1 << i for i, name in enumerate(SOLVERS)}
    palette = ["#d9d7d0", COLORS["SBM"], COLORS["SVL"], "#b49d65",
               COLORS["TRF"], "#63aaa1", "#a5a446", "#6c7378"]
    index = {(r["instance"], r["runs"], r["steps"]): r for r in rows}
    metadata = {r["instance"]: r for r in rows}
    names = sorted(instances, key=lambda name: (metadata[name]["family"], name))
    fig, axes = plt.subplots(1, 3, figsize=(19, max(11, len(names) * .17 + 3.5)), sharey=True)
    fig.subplots_adjust(top=.92, bottom=.05, left=.16, right=.98, wspace=.15)
    fig.suptitle("Exactly where each solver wins", x=.16, y=.984, ha="left", fontsize=23, fontweight="bold")
    fig.text(.16, .966, "One row per graph; one column per runs/steps setting. Zoom into the PNG or PDF for graph labels.", fontsize=10)
    for ax, key, title in zip(axes, ("quality_winners", "fastest_solver", "winner"),
                             ("Best cut (ties retained)", "Shortest solve time", "Best cut, then shortest time")):
        values = np.array([[sum(masks[s] for s in index[name, r, t][key].split("|"))
                            if (name, r, t) in index else 0 for r, t in settings] for name in names])
        ax.imshow(values, cmap=ListedColormap(palette), norm=BoundaryNorm(np.arange(-.5, 8.5), 8), aspect="auto", interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_xticks(range(len(settings)), [f"{r}/{t // 1000}k" for r, t in settings], rotation=90, fontsize=8)
        ax.set_xlabel("runs / steps")
        ax.set_yticks(range(len(names)), [f"{name}  [{metadata[name]['vertices']}]" for name in names], fontsize=7)
        for col in range(1, len(settings)):
            if settings[col][0] != settings[col - 1][0]:
                ax.axvline(col - .5, color=BACKGROUND, linewidth=2)
        for row in range(1, len(names)):
            if metadata[names[row]]["family"] != metadata[names[row - 1]]["family"]:
                ax.axhline(row - .5, color=INK, linewidth=1.5)
    legend = [Patch(color=palette[0], label="No completed comparison")]
    for length in (1, 2, 3):
        for combination in itertools.combinations(SOLVERS, length):
            value = sum(masks[s] for s in combination)
            legend.append(Patch(color=palette[value], label=" + ".join(combination)))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.56, .952), ncol=4, frameon=False)
    return fig


def distributions(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7))
    fig.subplots_adjust(top=.78, bottom=.18, left=.08, right=.97, wspace=.25)
    heading(fig, "Look beyond the averages", "Empirical distributions over all graph/budget comparisons. A curve farther left is better.")
    for solver in SOLVERS:
        prefix = solver.lower()
        for ax, key, scale in ((axes[0], "median_seconds", 1000), (axes[1], "gap_percent", 1)):
            values = np.sort([r[f"{prefix}_{key}"] * scale for r in rows])
            percent = 100 * np.arange(1, len(values) + 1) / len(values)
            ax.step(values, percent, where="post", color=COLORS[solver], label=solver, linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("complete solve (ms; log scale)")
    axes[1].set_xlabel("reference gap (%)")
    for ax in axes:
        ax.set_ylabel("comparisons at or below threshold (%)")
        ax.set_ylim(0, 101)
        ax.grid(True, alpha=.6)
        ax.legend(frameon=False)
    fig.text(.08, .055, "Timings: median of three same-seed trials per solve. These distributions mix graph families and sizes, not confidence intervals.", fontsize=9)
    return fig


def report_html(output, rows, settings, summaries, figures, expected_instances):
    data = []
    for row in rows:
        item = {key: row[key] for key in ("instance", "family", "vertices", "runs", "steps", "reference_cut", "quality_winners", "fastest_solver", "winner")}
        item["solvers"] = {s: {"cut": row[f"{s.lower()}_best_cut"],
                              "gap": row[f"{s.lower()}_gap_percent"],
                              "ms": row[f"{s.lower()}_median_seconds"] * 1000} for s in SOLVERS}
        data.append(item)
    cards = "".join(f'<article style="--solver:{COLORS[s["solver"]]}"><h2>{s["solver"]}</h2>'
                    f'<strong>{s["quality_win_percent"]:.1f}%</strong> best-cut credit'
                    f'<p>{s["median_solve_ms"]:.1f} ms median solve<br>{s["mean_gap_percent"]:.2f}% mean reference gap</p></article>' for s in summaries)
    gallery = []
    for filename, title in figures:
        encoded = base64.b64encode((output / filename).read_bytes()).decode("ascii")
        gallery.append(f'<details {"open" if filename == "overview.png" else ""}><summary>{html.escape(title)}</summary>'
                       f'<a href="{filename}">Open full-size PNG</a><img loading="lazy" alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"></details>')
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BiqMac | solver comparison</title>
<style>
:root{--ink:#23333c;--paper:#f8f5ef;--rule:#d6d4ca}*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:radial-gradient(ellipse at top right,#e0ece4,transparent 45%),var(--paper);font:16px Georgia,serif}
main{max-width:1450px;margin:auto;padding:40px 28px}h1{font-size:clamp(35px,5vw,68px);font-weight:normal;margin:12px 0 18px;letter-spacing:-2px}
.eyebrow,small,select,input,button,th,td{font-family:ui-monospace,monospace}.eyebrow{letter-spacing:3px;font-size:12px}
.intro{max-width:930px;font-size:19px;line-height:1.6}.notice{border-left:4px solid #d97532;padding:12px 20px;background:#ffffff80;line-height:1.6}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin:30px 0}article{border-top:5px solid var(--solver);padding:12px 18px;background:#ffffff88}
article h2{color:var(--solver);font-size:24px}article strong{display:block;font-size:45px;font-weight:normal}article p{line-height:1.6}
section{margin:40px 0}h2{font-size:30px;font-weight:normal}.controls{display:flex;gap:15px;flex-wrap:wrap;align-items:end;margin:18px 0}
label{display:grid;gap:7px}select,input{padding:10px;background:white;border:1px solid var(--rule);border-radius:4px}
.table-wrap{overflow:auto;max-height:650px;border:1px solid var(--rule)}table{width:100%;border-collapse:collapse;background:#fff9;font-size:12px}
th{position:sticky;top:0;background:#eaece6;z-index:1;text-align:left}td,th{padding:12px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
td b{display:block;font-size:13px}td small{line-height:1.7}.best{background:#dfeddf}.quick{box-shadow:inset 0 -3px #438653}
details{margin:18px 0;border-top:1px solid var(--rule);padding:18px 0}summary{font-size:25px;cursor:pointer;margin-bottom:15px}
img{width:100%;height:auto;display:block;margin-top:16px}a{color:#237d91}footer{font-size:14px;line-height:1.6;border-top:1px solid var(--rule);padding-top:22px}
@media(max-width:650px){main{padding:24px 14px}.cards{grid-template-columns:1fr;gap:10px}article strong{font-size:35px}}
</style><main><div class="eyebrow">BIQMAC / MAXCUT / MEASURED RESULTS</div>
<h1>Where does each solver win?</h1>
<p class="intro">SBM, SVL and Transverse Route across __COUNT__ graph/budget comparisons on eight RTX 2080 Ti GPUs. The charts separate best-cut quality from raw speed: a quick answer is not necessarily a good answer.</p>
<div class="notice"><b>Timing definition:</b> complete solve latency, including preparation, transfers, integration and energy scoring. It is not time-to-first-target or statistical time-to-solution. Each number is the median of three same-seed trials; no independent-seed success probability was measured.</div>
<div class="notice"><b>Coverage:</b> __COVERAGE__ Complete comparisons only. Missing/error comparisons are not solver losses. Settings may contain different graph subsets, so aggregate cross-budget differences can reflect coverage rather than improvement. Grey cells indicate missing results; graphs with no completed comparisons anywhere are not shown. <a href="coverage.csv">Coverage CSV</a></div>
<div class="cards">__CARDS__</div>
<section><h2>Explore individual problems</h2><p>Green cells tie for the best cut. An underline marks the fastest solver. Gap is percentage shortfall from the published reference; lower is better.</p>
<div class="controls"><label>Runs<select id="runs">__RUNS__</select></label><label>Steps<select id="steps">__STEPS__</select></label>
<label>Problem filter<input id="query" placeholder="e.g. g05 or t2g"></label>
<label>Quality winner<select id="solver"><option value="">All solvers</option><option>SBM</option><option>SVL</option><option>TRF</option></select></label>
<label>Sort<select id="sort"><option value="instance">Instance name</option><option value="disagreement">Quality disagreement (largest first)</option><option value="time">Slowest comparison first</option></select></label></div>
<p id="count"></p><div class="table-wrap"><table><thead><tr><th>Instance / vertices</th><th>Reference</th><th>SBM</th><th>SVL</th><th>TRF</th><th>Best cut, then time</th></tr></thead><tbody id="body"></tbody></table></div></section>
<section><h2>Charts</h2><p><a href="biqmac_plots.pdf">Download all charts as PDF</a> &middot; <a href="budget_summary.csv">Budget statistics CSV</a></p>__GALLERY__</section>
<footer>Quality-win credit splits one point across tied best-cut solvers. Final winner breaks quality ties by median latency. Each graph/budget pair receives equal weight; absolute cuts are not averaged across graphs. Native solver defaults were retained, and equal steps do not imply equal computational work. Concurrent workers can introduce host contention. No benchmark was rerun to create these plots.</footer></main>
<script>
const rows=__DATA__;
const names=['SBM','SVL','TRF'];
const $=id=>document.getElementById(id);
function format(v,n=2){return Number(v).toLocaleString(undefined,{maximumFractionDigits:n});}
function render(){
 let selected=rows.filter(r=>r.runs===+$('runs').value&&r.steps===+$('steps').value&&r.instance.toLowerCase().includes($('query').value.toLowerCase())&&(!$('solver').value||r.quality_winners.split('|').includes($('solver').value)));
 const spread=r=>Math.max(...names.map(s=>r.solvers[s].gap))-Math.min(...names.map(s=>r.solvers[s].gap));
 const slow=r=>Math.max(...names.map(s=>r.solvers[s].ms));
 selected.sort((a,b)=>$('sort').value==='disagreement'?spread(b)-spread(a):$('sort').value==='time'?slow(b)-slow(a):a.instance.localeCompare(b.instance));
 $('count').textContent=`${selected.length} instances shown`;
 const body=$('body');body.replaceChildren();
 for(const r of selected){const tr=document.createElement('tr');
  function cell(text){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);return td;}
  cell(`${r.instance} [${r.vertices}]`);cell(format(r.reference_cut));
  for(const s of names){const v=r.solvers[s],td=cell(''),b=document.createElement('b'),small=document.createElement('small');b.textContent=`cut ${format(v.cut)}`;small.textContent=`gap ${format(v.gap,3)}% | ${format(v.ms)} ms`;td.append(b,small);if(r.quality_winners.split('|').includes(s))td.classList.add('best');if(r.fastest_solver===s)td.classList.add('quick');}
  cell(r.winner);body.appendChild(tr);
 }
}
for(const id of ['runs','steps','query','solver','sort'])$(id).addEventListener('input',render);
render();
</script></html>"""
    replacements = {"__COUNT__": f"{len(rows):,}", "__CARDS__": cards,
        "__COVERAGE__": f"{len(rows):,} / {expected_instances * len(settings):,} planned graph/budget comparisons; {expected_instances} graphs, {len(settings)} settings.",
        "__RUNS__": "".join(f'<option {"selected" if r == max(s[0] for s in settings) else ""}>{r}</option>' for r in sorted({s[0] for s in settings})),
        "__STEPS__": "".join(f'<option {"selected" if t == max(s[1] for s in settings) else ""}>{t}</option>' for t in sorted({s[1] for s in settings})),
        "__GALLERY__": "".join(gallery), "__DATA__": json.dumps(data).replace("<", "\\u003c")}
    for key, value in replacements.items():
        document = document.replace(key, value)
    (output / "index.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-instances", type=int,
                        help="Planned graph count, including graphs with no completed comparisons")
    args = parser.parse_args()
    output = args.output or args.results.parent / "plots"
    rows, settings, instances = load_results(args.results)
    runs, steps = sorted({r for r, _ in settings}), sorted({t for _, t in settings})
    expected_instances = args.expected_instances or len(instances)
    if expected_instances < len(instances):
        raise ValueError("Expected instance count is smaller than the observed instance count")
    output.mkdir(parents=True, exist_ok=True)
    style()
    overall = summarize(rows)
    budget = []
    coverage = []
    for r, t in settings:
        subset = [row for row in rows if row["runs"] == r and row["steps"] == t]
        coverage.append(dict(runs=r, steps=t, completed=len(subset), expected=expected_instances,
                             missing=expected_instances - len(subset)))
        if subset:
            budget.extend(summarize(subset, runs=r, steps=t))
    write_csv(output / "coverage.csv", coverage)
    write_csv(output / "budget_summary.csv", budget)
    write_csv(output / "overall_plot_summary.csv", overall)
    figures = [("overview.png", "Overall winners", lambda: overview(rows, overall)),
               ("budget_heatmaps.png", "Quality and runtime across all budgets", lambda: budget_heatmaps(budget, runs, steps)),
               ("speed_quality.png", "Speed-quality trade-off", lambda: speed_quality(budget, runs)),
               ("distributions.png", "Runtime and quality distributions", lambda: distributions(rows)),
               ("instance_winners.png", "Winner map for every instance", lambda: instance_map(rows, settings, instances))]
    with PdfPages(output / "biqmac_plots.pdf") as pdf:
        for filename, title, create in figures:
            print(f"Plotting: {title}", flush=True)
            fig = create()
            fig.savefig(output / filename, dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    report_html(output, rows, settings, overall, [(name, title) for name, title, _ in figures], expected_instances)
    print(json.dumps({"report": str(output / "index.html"), "instances": len(instances),
                      "settings": len(settings), "comparisons": len(rows), "coverage": coverage,
                      "overall": overall}, indent=2))


if __name__ == "__main__":
    main()

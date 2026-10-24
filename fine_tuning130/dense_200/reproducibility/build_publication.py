"""Validate and publish fine_tuning130 rankings and quality-grouped figures."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve()
BATCH = HERE.parents[1]
DATA = BATCH / "plot_ready"
RESULTS_ROOT = BATCH.parents[1]
GLOBAL_PLOTS = RESULTS_ROOT / "PLots"
CHECKPOINTS = np.asarray((0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
REFERENCE = -35154.0
PROBLEM = "gka5e"
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7",
          "#e69f00", "#56b4e9", "#000000", "#6f5aa5")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def same(actual: float, expected: float) -> bool:
    if math.isnan(expected):
        return math.isnan(actual)
    if math.isinf(expected):
        return actual == expected
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-7)


def tts99(seconds: float, probability: float) -> float:
    if probability == 0:
        return math.inf
    if probability == 1:
        return 0.0
    return seconds * math.log(0.01) / math.log1p(-probability)


def load_and_validate():
    manifest = read_json(DATA / "manifest.json")
    if manifest.get("partial_campaign") is not False:
        raise ValueError("Expected the complete seven-host production campaign")
    solvers = sorted(manifest["solvers"])
    if len(solvers) != 130 or manifest["seeds"] != list(range(100)):
        raise ValueError("Expected 130 solvers and seeds 0 through 99")
    if not np.array_equal(np.asarray(manifest["checkpoints_seconds"]), CHECKPOINTS):
        raise ValueError("Unexpected checkpoint schedule")
    problems = manifest["problems"]
    if len(problems) != 1:
        raise ValueError("Expected one dense-200 problem")
    problem = problems[0]
    if (problem["id"], problem["n"], problem["density"], float(problem["reference"]),
            problem["reference_type"]) != (PROBLEM, 200, "dense", REFERENCE, "OPTIMUM"):
        raise ValueError("Unexpected problem or optimum")
    state = read_json(DATA / "state.json")["payload"]
    if not state["complete"] or state["planned"] != 13000 or state["terminal"] != 13000 or state["failures"]:
        raise ValueError("Compact campaign is not complete and failure-free")

    curves = {}
    quality = {}
    threshold = REFERENCE + 0.01 * abs(REFERENCE)
    for solver in solvers:
        matrix = np.load(DATA / "raw" / solver / f"{PROBLEM}.npy", allow_pickle=False)
        if (matrix.dtype != np.float64 or matrix.shape != (100, 9)
                or np.isinf(matrix).any() or not np.isfinite(matrix[:, -1]).all()):
            raise ValueError(f"Invalid raw matrix: {solver}")
        if np.any(matrix[np.isfinite(matrix)] < REFERENCE):
            raise ValueError(f"Invalid incumbent history: {solver}")
        for row in matrix:
            present = row[np.isfinite(row)]
            if np.any(np.diff(present) > 0):
                raise ValueError(f"Incumbent worsens within a seed: {solver}")
        present = np.isfinite(matrix)
        exact = (present & (matrix == REFERENCE)).sum(axis=0).astype(int)
        near = (present & (matrix <= threshold)).sum(axis=0).astype(int)
        gap = np.asarray([
            float(matrix[:, index].mean() - REFERENCE) if present[:, index].all() else math.nan
            for index in range(matrix.shape[1])
        ])
        gap_percent = 100.0 * gap / abs(REFERENCE)
        probability = exact.astype(float) / 100.0
        tts = np.asarray([tts99(float(t), float(p)) for t, p in zip(CHECKPOINTS, probability)])
        with (DATA / "aggregated" / solver / f"{PROBLEM}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 9:
            raise ValueError(f"Invalid aggregate checkpoint count: {solver}")
        for index, row in enumerate(rows):
            observed = (float(row["checkpoint_seconds"]), float(row["gap_to_solution"]),
                        float(row["hit_probability"]), float(row["TTS99"]))
            expected = (float(CHECKPOINTS[index]), float(gap[index]),
                        float(probability[index]), float(tts[index]))
            if not all(same(a, e) for a, e in zip(observed, expected)):
                raise ValueError(f"Aggregate differs from raw matrix: {solver}, checkpoint {index}")
        curves[solver] = {
            "gap_to_solution": gap,
            "hit_probability": probability,
            "TTS99": tts,
            "exact_hits": exact,
            "hits_within_1pct": near,
            "gap_to_reference_percent": gap_percent,
        }
        quality[solver] = {
            "within": int(near[-1]),
            "exact": int(exact[-1]),
            "mean_gap": float(gap_percent[-1]),
        }
    order = sorted(solvers, key=lambda solver: (
        -quality[solver]["within"], -quality[solver]["exact"],
        quality[solver]["mean_gap"], solver,
    ))
    return manifest, problem, curves, quality, order


def page_groups(order):
    counts = ((7, 7, 7, 7, 7, 7), (7, 7, 7, 7, 7, 7), (8, 8, 8, 8, 7, 7))
    groups = []
    cursor = 0
    for page_counts in counts:
        page = []
        for count in page_counts:
            page.append(order[cursor:cursor + count])
            cursor += count
        groups.append(page)
    if cursor != len(order) or any(not panel for page in groups for panel in page):
        raise ValueError("Invalid panel allocation")
    return groups


def rank_span(panel, order):
    first = order.index(panel[0]) + 1
    return first, first + len(panel) - 1


def runtime_figures(problem, curves, order, groups):
    labels = {
        "gap_to_solution": "Mean signed gap (raw objective units)",
        "hit_probability": "Hit probability",
        "TTS99": "TTS99 (seconds)",
    }
    output = BATCH / "plots" / PROBLEM
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for part, panels in enumerate(groups, 1):
        for metric, ylabel in labels.items():
            figure, axes = plt.subplots(2, 3, figsize=(21.6, 9.4), dpi=110)
            figure.suptitle(
                f"{PROBLEM}: n=200, dense, OPTIMUM, seeds=100 | fine_tuning130, part {part} of 3",
                fontsize=14,
            )
            figure.text(
                .5, .017,
                "130 tuned solvers ordered by final solution quality; all seven production partitions complete.",
                ha="center", fontsize=9,
            )
            for axes_item, selected in zip(axes.ravel(), panels):
                first, last = rank_span(selected, order)
                for index, solver in enumerate(selected):
                    y = np.asarray(curves[solver][metric], dtype=float).copy()
                    y[~np.isfinite(y)] = np.nan
                    axes_item.plot(CHECKPOINTS, y, marker=".", linewidth=1.1,
                                   color=COLORS[index], label=solver)
                axes_item.set_xscale("log")
                axes_item.set_xlim(CHECKPOINTS[0], CHECKPOINTS[-1])
                axes_item.set_xticks(CHECKPOINTS)
                axes_item.set_xticklabels([f"{value:g}" for value in CHECKPOINTS])
                axes_item.set_xlabel("Checkpoint (seconds)")
                axes_item.set_ylabel(ylabel)
                axes_item.set_title(f"Quality ranks {first}-{last}", fontsize=9)
                if metric == "hit_probability":
                    axes_item.set_ylim(-.025, 1.025)
                elif metric == "TTS99":
                    axes_item.set_yscale("symlog", linthresh=CHECKPOINTS[0])
                    axes_item.set_ylim(bottom=0)
                    if not any(np.isfinite(curves[solver][metric]).any() for solver in selected):
                        axes_item.text(.5, .5, "No finite TTS99: zero observed hits",
                                       transform=axes_item.transAxes, ha="center", va="center", fontsize=9)
                axes_item.grid(True, alpha=.25)
                axes_item.legend(fontsize=5.8 if len(selected) == 8 else 6, loc="best")
            figure.tight_layout(rect=(0, .04, 1, .96))
            path = output / f"{metric}_fine_tuning130_part{part}.png"
            figure.savefig(path, dpi=110, facecolor="white")
            plt.close(figure)
            paths.append(path)
    return paths


def summary_figure(metric, title, unit, part, panels, curves, order):
    figure = plt.figure(figsize=(22.5, 16.4), dpi=100, facecolor="white")
    ink = "#252f3d"
    muted = "#5f6975"
    grid = "#e0e5eb"
    border = "#d1d9e1"
    figure.text(.024, .975, f"{title} | fine_tuning130, part {part} of 3",
                ha="left", va="top", fontsize=21, fontweight="bold", color=ink)
    figure.text(.024, .948,
                f"{PROBLEM} | 200 variables | dense QUBO | published optimum -35154 | "
                "100 seeds per solver | 130 tuned solvers, quality-ranked",
                ha="left", va="top", fontsize=11, color=muted)
    figure.text(.024, .019, "Checkpoint time (seconds; logarithmic x-axis)",
                ha="left", va="bottom", fontsize=11, color=muted)
    figure.text(.024, .006,
                "Ranking: within-1% hits descending, exact hits descending, mean final gap ascending.",
                ha="left", va="bottom", fontsize=9, color=muted)

    for panel_index, selected in enumerate(panels):
        column = panel_index % 3
        row = panel_index // 3
        px = 45 + 735 * column
        py = 112 + 725 * row
        panel_left, panel_bottom = px / 2250, 1 - (py + 700) / 1640
        panel_width, panel_height = 705 / 2250, 700 / 1640
        figure.add_artist(Rectangle((panel_left, panel_bottom), panel_width, panel_height,
                                    transform=figure.transFigure, fill=False,
                                    edgecolor=border, linewidth=1))
        first, last = rank_span(selected, order)
        figure.text((px + 23) / 2250, 1 - (py + 15) / 1640,
                    f"Quality ranks {first}-{last}", ha="left", va="top",
                    fontsize=13, fontweight="bold", color=ink)
        figure.text((px + 23) / 2250, 1 - (py + 47) / 1640,
                    unit, ha="left", va="top", fontsize=9, color=muted)
        axes = figure.add_axes(((px + 70) / 2250, 1 - (py + 78 + 390) / 1640,
                               605 / 2250, 390 / 1640))
        values = np.concatenate([np.asarray(curves[solver][metric], dtype=float) for solver in selected])
        finite = values[np.isfinite(values)]
        if metric == "gap_to_reference_percent":
            lower = min(0.0, math.floor(float(finite.min()) * 2 * 1.08) / 2) if finite.size else 0.0
            upper = max(2.0, math.ceil(float(finite.max()) * 2 * 1.08) / 2) if finite.size else 2.0
            axes.set_ylim(lower, upper)
            axes.axhline(1.0, color="#545b65", linestyle="--", linewidth=1.0)
        else:
            axes.set_ylim(0, 100)
        handles, legend_labels = [], []
        for index, solver in enumerate(selected):
            y = np.asarray(curves[solver][metric], dtype=float).copy()
            y[~np.isfinite(y)] = np.nan
            color = COLORS[index]
            axes.plot(CHECKPOINTS, y, color=color, marker="o", markersize=3.5, linewidth=2.0)
            handles.append(Line2D([0], [0], color=color, marker="o", markersize=3.5, linewidth=2.0))
            legend_labels.append(solver.removeprefix("lib_"))
        axes.set_xscale("log")
        axes.set_xlim(CHECKPOINTS[0], CHECKPOINTS[-1])
        axes.set_xticks(CHECKPOINTS)
        axes.set_xticklabels([f"{value:g}" for value in CHECKPOINTS], fontsize=8, color=muted)
        axes.tick_params(axis="y", labelsize=8, colors=muted)
        axes.grid(True, color=grid, linewidth=.8)
        for spine in axes.spines.values():
            spine.set_visible(False)
        figure.legend(handles, legend_labels, loc="upper left", frameon=False,
                      bbox_to_anchor=((px + 23) / 2250, 1 - (py + 490) / 1640),
                      borderaxespad=0, fontsize=8.2, handlelength=2.0,
                      labelspacing=.55, handletextpad=.7)
    GLOBAL_PLOTS.mkdir(parents=True, exist_ok=True)
    path = GLOBAL_PLOTS / f"200_dense_{metric}_fine_tuning130_part{part}.png"
    figure.savefig(path, dpi=100, facecolor="white")
    plt.close(figure)
    internal = BATCH / "plots" / PROBLEM / path.name
    internal.parent.mkdir(parents=True, exist_ok=True)
    internal.write_bytes(path.read_bytes())
    return path, internal


def summary_figures(curves, order, groups):
    specs = (
        ("exact_hits", "Exact optimum hits vs time", "Reference hits out of 100 planned seeds"),
        ("hits_within_1pct", "Seeds reaching at least 99% of published optimum quality vs time",
         "Qualifying seeds out of 100; exact hits included"),
        ("gap_to_reference_percent", "Mean percentage gap to optimum vs time",
         "0% = optimum; dashed line = 1% gap; lower is better"),
    )
    paths = []
    for part, panels in enumerate(groups, 1):
        for metric, title, unit in specs:
            paths.extend(summary_figure(metric, title, unit, part, panels, curves, order))
    return paths


def ranking_text(quality, order):
    heading = [
        "FINE_TUNING130: DENSE 200 SOLVERS, BEST FIRST",
        "",
        "Problem: gka5e, 200 variables, dense QUBO; minimize x.T @ Q @ x.",
        "Published proven optimum: -35154. Results: final 20-second checkpoint.",
        "130 tuned solvers, each with all 100 completed seeds (0-99): 13,000 trials, zero failures.",
        "Early verified optima carried to the 20-second checkpoint are included.",
        "",
        "Ranking: seeds within 1% descending; exact optimum seeds descending;",
        "then mean percentage gap ascending. Exact ties use the recorded solver ID.",
        "Alphabetical ordering of exact ties does not imply a quality difference.",
        "Ordering uses unrounded values; displayed mean gaps have six decimal places.",
        "",
        "Relative gap (%) = 100 * (E - E_optimum) / abs(E_optimum).",
        "Within 1% means this percentage value is <= 1.",
        "For this integer QUBO, qualifying energies are <= -34803.",
        "Exact optimum: E = -35154. Mean gap is the arithmetic mean across all 100 seeds.",
        "Exact optimum hits are included in the within-1% counts.",
        "",
    ]
    columns = ("Rank", "Solver (recorded ID)", "Batch", "Seeds within 1%", "Exact optimum seeds", "Mean gap")
    rows = []
    for rank, solver in enumerate(order, 1):
        item = quality[solver]
        rows.append((str(rank), solver, "fine_tuning130", f"{item['within']}/100",
                     f"{item['exact']}/100", f"{item['mean_gap']:.6f}%"))
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    def line(values):
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"
    table = [line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    table.extend(line(row) for row in rows)
    footer = [
        "",
        "SOURCE AND VALIDATION",
        "",
        "Source: fine_tuning130/plot_ready raw matrices and checked aggregate curves.",
        f"Manifest SHA-256: {sha256(DATA / 'manifest.json')}",
        f"Completion SHA-256: {sha256(DATA / 'completion.json')}",
        f"Quality summary SHA-256: {sha256(DATA / 'postprocessed' / 'final_quality_summary.csv')}",
        "All 130 raw matrices were recomputed and compared with the saved aggregates before plotting.",
    ]
    return "\n".join(heading + table + footer) + "\n"


def main():
    manifest, problem, curves, quality, order = load_and_validate()
    groups = page_groups(order)
    group_record = {
        "schema_version": 1,
        "description": "Contiguous final-quality ranking groups used in every fine_tuning130 figure.",
        "ranking": ["hits_within_1pct descending", "exact_hits descending",
                    "mean_gap_percent ascending", "solver_id ascending for exact ties"],
        "parts": [{
            "part": part,
            "panels": [{
                "panel": panel_index,
                "first_rank": rank_span(panel, order)[0],
                "last_rank": rank_span(panel, order)[1],
                "solvers": panel,
            } for panel_index, panel in enumerate(panels, 1)],
        } for part, panels in enumerate(groups, 1)],
    }
    write_json(BATCH / "reproducibility" / "solver_quality_groups.json", group_record)
    ranking_path = BATCH / "FINE_TUNING130_DENSE_200_SOLVER_RANKING.txt"
    ranking_path.write_text(ranking_text(quality, order), encoding="utf-8")
    runtime_paths = runtime_figures(problem, curves, order, groups)
    summary_paths = summary_figures(curves, order, groups)
    artifacts = [ranking_path, *runtime_paths, *summary_paths]
    report = {
        "schema_version": 1,
        "status": "passed",
        "campaign": manifest["campaign"],
        "solvers": len(order),
        "trials": len(order) * 100,
        "failures": 0,
        "runtime_figures": len(runtime_paths),
        "summary_figures_and_internal_copies": len(summary_paths),
        "global_summary_figures": len(summary_paths) // 2,
        "exact_hits_at_20s": sum(item["exact"] for item in quality.values()),
        "within_1pct_at_20s": sum(item["within"] for item in quality.values()),
        "artifacts": {str(path.relative_to(RESULTS_ROOT)).replace("\\", "/"): sha256(path)
                      for path in artifacts},
    }
    write_json(BATCH / "reproducibility" / "publication_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

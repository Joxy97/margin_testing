#!/usr/bin/env python3
"""Plot per-size Torch SVL runs-by-steps quality and runtime lines."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

PLOT_CACHE = Path(tempfile.gettempdir()) / "margin_testing_matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot


def _read(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = [
            {
                "vertices": int(row["vertices"]),
                "runs": int(row["runs"]),
                "steps": int(row["steps"]),
                "quality": float(row["quality"]),
                "seconds": float(row["solve_seconds"]),
            }
            for row in csv.DictReader(source)
        ]
    if not rows:
        raise ValueError(f"{path} has no completed results")
    return rows


def _summaries(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    groups: dict[tuple[int, int, int], list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["vertices"]), int(row["runs"]), int(row["steps"]))].append(row)
    return [
        {
            "vertices": vertices,
            "runs": runs,
            "steps": steps,
            "observations": len(group),
            "quality_median": statistics.median(float(row["quality"]) for row in group),
            "seconds_median": statistics.median(float(row["seconds"]) for row in group),
        }
        for (vertices, runs, steps), group in sorted(groups.items())
    ]


def _plot(output: Path, vertices: int, rows: list[dict[str, float | int]]) -> None:
    runs = sorted({int(row["runs"]) for row in rows})
    steps = sorted({int(row["steps"]) for row in rows})
    figure, axes = pyplot.subplots(1, 2, figsize=(15, 6))
    for run in runs:
        series = sorted(
            (row for row in rows if int(row["runs"]) == run),
            key=lambda row: int(row["steps"]),
        )
        x = [int(row["steps"]) for row in series]
        axes[0].plot(
            x,
            [float(row["quality_median"]) for row in series],
            marker="o",
            linewidth=1.8,
            label=f"{run} runs",
        )
        axes[1].plot(
            x,
            [float(row["seconds_median"]) for row in series],
            marker="o",
            linewidth=1.8,
            label=f"{run} runs",
        )
    for axis in axes:
        axis.set_xticks(steps)
        axis.set_xlabel("Steps")
        axis.grid(alpha=0.25)
        axis.legend(title="Trajectories")
    axes[0].set_title("Solution quality")
    axes[0].set_ylabel("Cut / reference cut (trial median)")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set_title("Solver runtime")
    axes[1].set_ylabel("Seconds (trial median, logarithmic)")
    axes[1].set_yscale("log")
    figure.suptitle(f"Torch SVL runs × steps: n={vertices}")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_dir or arguments.results.parent / "runs_steps_plots"
    output.mkdir(parents=True, exist_ok=True)
    summaries = _summaries(_read(arguments.results))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for vertices in sorted({int(row["vertices"]) for row in summaries}):
        _plot(
            output / f"vertices_{vertices:07d}.png",
            vertices,
            [row for row in summaries if int(row["vertices"]) == vertices],
        )
    print(f"wrote plots for {len({row['vertices'] for row in summaries})} sizes to {output}")


if __name__ == "__main__":
    main()

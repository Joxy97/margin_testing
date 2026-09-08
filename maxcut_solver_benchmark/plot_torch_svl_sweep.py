#!/usr/bin/env python3
"""Plot Torch SVL quality and runtime against every swept parameter."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def _read(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = []
        for raw in csv.DictReader(source):
            rows.append(
                {
                    "parameter": raw["parameter"],
                    "parameter_value": json.loads(raw["parameter_value"]),
                    "vertices": int(raw["vertices"]),
                    "quality": float(raw["quality"]),
                    "seconds": float(raw["solve_seconds"]),
                }
            )
    if not rows:
        raise ValueError(f"{path} has no sweep results")
    return rows


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    values: dict[tuple[str, str], object] = {}
    for row in rows:
        parameter = str(row["parameter"])
        encoded = json.dumps(row["parameter_value"], sort_keys=True)
        key = (parameter, encoded, int(row["vertices"]))
        groups[key].append(row)
        values[(parameter, encoded)] = row["parameter_value"]
    result = []
    for (parameter, encoded, vertices), group in sorted(groups.items()):
        quality = [float(row["quality"]) for row in group]
        seconds = [float(row["seconds"]) for row in group]
        result.append(
            {
                "parameter": parameter,
                "parameter_value": json.dumps(values[(parameter, encoded)]),
                "vertices": vertices,
                "observations": len(group),
                "quality_median": statistics.median(quality),
                "quality_min": min(quality),
                "quality_max": max(quality),
                "seconds_median": statistics.median(seconds),
                "seconds_min": min(seconds),
                "seconds_max": max(seconds),
            }
        )
    return result


def _sort_values(values: set[object]) -> list[object]:
    if all(isinstance(value, (int, float)) for value in values):
        return sorted(values, key=float)
    return sorted(values, key=str)


def _plot_parameter(
    output: Path,
    parameter: str,
    rows: list[dict[str, object]],
    solver_label: str = "Torch SVL",
) -> None:
    values = _sort_values({row["parameter_value"] for row in rows})
    positions = {json.dumps(value): index for index, value in enumerate(values)}
    figure, (quality_axis, time_axis) = pyplot.subplots(1, 2, figsize=(13, 5.5))
    for vertices in sorted({int(row["vertices"]) for row in rows}):
        series = [row for row in rows if int(row["vertices"]) == vertices]
        series.sort(key=lambda row: positions[json.dumps(row["parameter_value"])])
        x = [positions[json.dumps(row["parameter_value"])] for row in series]
        quality_axis.plot(
            x, [float(row["quality_median"]) for row in series],
            marker="o", label=f"n={vertices}", linewidth=1,
        )
        time_axis.plot(
            x, [float(row["seconds_median"]) for row in series],
            marker="o", label=f"n={vertices}", linewidth=1,
        )
    labels = [str(value) for value in values]
    for axis in (quality_axis, time_axis):
        axis.set_xticks(range(len(values)), labels, rotation=25, ha="right")
        axis.set_xlabel(parameter)
        axis.grid(alpha=0.25)
    quality_axis.set_title("Solution quality")
    quality_axis.set_ylabel("Cut / reference cut (trial median)")
    quality_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    time_axis.set_title("Solver runtime")
    time_axis.set_ylabel("Seconds (trial median, logarithmic)")
    time_axis.set_yscale("log")
    time_axis.legend(fontsize=7, ncol=2, loc="best")
    figure.suptitle(f"{solver_label} sensitivity: {parameter}")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_dir or arguments.results.parent / "parameter_plots"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((arguments.results.parent / "manifest.json").read_text())
    solver_label = {
        "torch_sbm": "Torch SBM",
        "torch_svl": "Torch SVL",
    }.get(manifest.get("solver"), "Torch SVL")
    summaries = _summaries(_read(arguments.results))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    parameters = sorted({str(row["parameter"]) for row in summaries if row["parameter"] != "baseline"})
    for parameter in parameters:
        parameter_rows = [row for row in summaries if row["parameter"] == parameter]
        # Add baseline observations as that parameter's default point.
        baseline_rows = [row for row in summaries if row["parameter"] == "baseline"]
        if baseline_rows:
            default = manifest["baseline"][parameter]
            for row in baseline_rows:
                copied = dict(row)
                copied["parameter"] = parameter
                copied["parameter_value"] = default
                parameter_rows.append(copied)
        for row in parameter_rows:
            if isinstance(row["parameter_value"], str):
                try:
                    row["parameter_value"] = json.loads(row["parameter_value"])
                except json.JSONDecodeError:
                    pass
        _plot_parameter(
            output / f"{parameter}.png", parameter, parameter_rows, solver_label
        )
    print(f"wrote {len(parameters)} parameter plots and {output / 'summary.csv'}")


if __name__ == "__main__":
    main()

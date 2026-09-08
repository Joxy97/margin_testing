#!/usr/bin/env python3
"""Plot aggregate line and 3-D views of the BiqMac Torch SVL sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def plot(results_path: Path) -> None:
    manifest_path = results_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    trial_count = int(manifest.get("trials", 1))
    rows_by_key: dict[tuple[str, float, int, int], dict[str, str]] = {}
    with results_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if int(row["trial"]) >= trial_count:
                continue
            key = (row["instance"], float(row["dt"]), int(row["steps"]), int(row["trial"]))
            rows_by_key[key] = row
    if not rows_by_key:
        raise ValueError(f"no results found in {results_path}")

    grouped: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)
    for (_, dt, steps, _), row in rows_by_key.items():
        grouped[(dt, steps)].append(row)
    dts = sorted({key[0] for key in grouped})
    steps_values = sorted({key[1] for key in grouped})
    output = results_path.parent

    summary_path = output / "aggregate_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as destination:
        fields = ("dt", "steps", "records", "instances", "quality_median", "quality_q25",
                  "quality_q75", "solve_seconds_median", "solve_seconds_q25", "solve_seconds_q75")
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for dt in dts:
            for steps in steps_values:
                rows = grouped.get((dt, steps), [])
                if not rows:
                    continue
                quality = np.asarray([float(row["quality"]) for row in rows])
                seconds = np.asarray([float(row["solve_seconds"]) for row in rows])
                writer.writerow({
                    "dt": dt, "steps": steps, "records": len(rows),
                    "instances": len({row["instance"] for row in rows}),
                    "quality_median": np.median(quality), "quality_q25": np.quantile(quality, .25),
                    "quality_q75": np.quantile(quality, .75), "solve_seconds_median": np.median(seconds),
                    "solve_seconds_q25": np.quantile(seconds, .25),
                    "solve_seconds_q75": np.quantile(seconds, .75),
                })

    colors = plt.cm.viridis(np.linspace(0, 1, len(dts)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for dt, color in zip(dts, colors):
        present = [steps for steps in steps_values if (dt, steps) in grouped]
        quality = [_median([float(row["quality"]) for row in grouped[(dt, steps)]]) for steps in present]
        seconds = [_median([float(row["solve_seconds"]) for row in grouped[(dt, steps)]]) for steps in present]
        label = f"dt={dt:g}"
        axes[0].plot(present, quality, marker="o", ms=3, lw=1.4, color=color, label=label)
        axes[1].plot(present, seconds, marker="o", ms=3, lw=1.4, color=color, label=label)
    axes[0].set(title="Torch SVL solution quality", xlabel="Integration steps", ylabel="Cut / published reference")
    axes[1].set(title="Torch SVL solver runtime", xlabel="Integration steps", ylabel="Solve time (seconds)")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8, ncols=2)
    fig.suptitle("Median across selected BiqMac instances (8 runs per grid point)")
    fig.savefig(output / "quality_runtime_lines.png", dpi=180)
    plt.close(fig)

    instances = sorted({key[0] for key in rows_by_key})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, constrained_layout=True)
    for axis, instance in zip(axes.flat, instances):
        for dt, color in zip(dts, colors):
            points = sorted(
                (steps, float(row["quality"]))
                for (name, row_dt, steps, _), row in rows_by_key.items()
                if name == instance and row_dt == dt
            )
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points],
                          marker="o", ms=2.5, lw=1.1, color=color, label=f"dt={dt:g}")
        axis.set_title(instance)
        axis.set_xlabel("Integration steps")
        axis.set_ylabel("Cut / published reference")
        axis.grid(alpha=.25)
    axes.flat[-1].legend(fontsize=7, ncols=2)
    fig.suptitle("Torch SVL quality by BiqMac instance (8 runs per grid point)")
    fig.savefig(output / "quality_lines_by_instance.png", dpi=180)
    plt.close(fig)

    dt_grid, steps_grid = np.meshgrid(dts, steps_values)
    for field, title, zlabel, filename in (
        ("quality", "Torch SVL solution-quality surface", "Cut / published reference", "quality_surface_3d.png"),
        ("solve_seconds", "Torch SVL runtime surface", "Solve time (seconds)", "runtime_surface_3d.png"),
    ):
        z = np.full(dt_grid.shape, np.nan)
        for row_index, steps in enumerate(steps_values):
            for column_index, dt in enumerate(dts):
                rows = grouped.get((dt, steps), [])
                if rows:
                    z[row_index, column_index] = _median([float(row[field]) for row in rows])
        fig = plt.figure(figsize=(10, 7), constrained_layout=True)
        axis = fig.add_subplot(111, projection="3d")
        surface = axis.plot_surface(dt_grid, steps_grid, np.ma.masked_invalid(z), cmap="viridis",
                                    edgecolor="none", antialiased=True)
        axis.set(title=title, xlabel="dt", ylabel="Integration steps", zlabel=zlabel)
        axis.view_init(elev=28, azim=-130)
        fig.colorbar(surface, ax=axis, shrink=.65, pad=.1, label=zlabel)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    print(f"Plotted {len(rows_by_key)} records into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    plot(args.results.resolve())


if __name__ == "__main__":
    main()

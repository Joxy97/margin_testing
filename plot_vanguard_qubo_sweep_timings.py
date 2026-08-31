#!/usr/bin/env python3
"""Plot solve-time dependence for a stored-QUBO parameter sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy
import pandas


REQUIRED_COLUMNS = {"steps", "runs", "margin", "total_seconds"}


def load_summary(path: Path) -> pandas.DataFrame:
    data = pandas.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("summary contains no measurements")
    if data.duplicated(["steps", "runs"]).any():
        raise ValueError("summary contains duplicate steps/runs measurements")
    if (data[["steps", "runs", "total_seconds"]] <= 0).any().any():
        raise ValueError("steps, runs, and total_seconds must be positive")
    result = data.copy()
    result["solve_minutes"] = result["total_seconds"] / 60.0
    return result.sort_values(["steps", "runs"])


def plot_time_vs_steps(data: pandas.DataFrame, output: Path) -> None:
    figure, axis = pyplot.subplots(figsize=(10, 6.2), constrained_layout=True)
    runs = sorted(data["runs"].unique())
    colors = cm.viridis(numpy.linspace(0.08, 0.92, len(runs)))
    for color, run_count in zip(colors, runs):
        subset = data[data["runs"] == run_count]
        axis.plot(
            subset["steps"],
            subset["solve_minutes"],
            marker="o",
            linewidth=2,
            markersize=4.5,
            color=color,
            label=f"runs = {run_count}",
        )
    axis.set_title("Vanguard Torch SBM solve time vs. number of steps")
    axis.set_xlabel("SBM steps")
    axis.set_ylabel("Solve time for 105 QUBOs (minutes)")
    axis.grid(True, alpha=0.28)
    axis.legend(title="Trajectories", ncol=2, frameon=True)
    figure.savefig(output, dpi=180)
    pyplot.close(figure)


def plot_time_vs_runs(data: pandas.DataFrame, output: Path) -> None:
    figure, axis = pyplot.subplots(figsize=(11, 6.5), constrained_layout=True)
    steps = sorted(data["steps"].unique())
    colors = cm.plasma(numpy.linspace(0.05, 0.95, len(steps)))
    for color, step_count in zip(colors, steps):
        subset = data[data["steps"] == step_count]
        axis.plot(
            subset["runs"],
            subset["solve_minutes"],
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=color,
            label=f"steps = {step_count}",
        )
    run_ticks = sorted(data["runs"].unique())
    axis.set_xscale("log", base=2)
    axis.set_xticks(run_ticks, labels=[str(value) for value in run_ticks])
    axis.set_title("Vanguard Torch SBM solve time vs. number of runs")
    axis.set_xlabel("SBM runs")
    axis.set_ylabel("Solve time for 105 QUBOs (minutes)")
    axis.grid(True, alpha=0.28)
    axis.legend(
        title="Step count",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=8,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def plot_time_surface(data: pandas.DataFrame, output: Path) -> None:
    pivot = data.pivot(index="runs", columns="steps", values="solve_minutes")
    if pivot.isna().any().any():
        raise ValueError("3D surface requires a complete steps/runs grid")
    step_grid, run_grid = numpy.meshgrid(
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
    )
    time_grid = pivot.to_numpy(dtype=float)

    figure = pyplot.figure(figsize=(11, 7.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    color_norm = Normalize(
        vmin=float(numpy.min(time_grid)),
        vmax=float(numpy.max(time_grid)),
    )
    surface = axis.plot_surface(
        step_grid,
        run_grid,
        time_grid,
        cmap="viridis",
        norm=color_norm,
        edgecolor="black",
        linewidth=0.25,
        antialiased=True,
        alpha=0.94,
    )
    axis.scatter(
        step_grid,
        run_grid,
        time_grid,
        color="black",
        s=8,
        alpha=0.65,
    )
    axis.set_title("Vanguard Torch SBM solve-time surface", pad=18)
    axis.set_xlabel("SBM steps", labelpad=10)
    axis.set_ylabel("SBM runs", labelpad=10)
    axis.set_zlabel("Solve time for 105 QUBOs (minutes)", labelpad=10)
    axis.set_yticks(pivot.index.to_numpy())
    axis.view_init(elev=27, azim=-132)
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.68, pad=0.09)
    colorbar.set_label("Solve time (minutes)")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def plot_quality_vs_steps(data: pandas.DataFrame, output: Path) -> None:
    figure, axis = pyplot.subplots(figsize=(10, 6.2), constrained_layout=True)
    runs = sorted(data["runs"].unique())
    colors = cm.viridis(numpy.linspace(0.08, 0.92, len(runs)))
    for color, run_count in zip(colors, runs):
        subset = data[data["runs"] == run_count]
        axis.plot(
            subset["steps"],
            subset["margin"],
            marker="o",
            linewidth=2,
            markersize=4.5,
            color=color,
            label=f"runs = {run_count}",
        )
    axis.set_title("Vanguard Torch SBM solution quality vs. number of steps")
    axis.set_xlabel("SBM steps")
    axis.set_ylabel("Decoded margin (higher is better)")
    axis.grid(True, alpha=0.28)
    axis.legend(title="Trajectories", ncol=2, frameon=True)
    figure.savefig(output, dpi=180)
    pyplot.close(figure)


def plot_quality_vs_runs(data: pandas.DataFrame, output: Path) -> None:
    figure, axis = pyplot.subplots(figsize=(11, 6.5), constrained_layout=True)
    steps = sorted(data["steps"].unique())
    colors = cm.plasma(numpy.linspace(0.05, 0.95, len(steps)))
    for color, step_count in zip(colors, steps):
        subset = data[data["steps"] == step_count]
        axis.plot(
            subset["runs"],
            subset["margin"],
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=color,
            label=f"steps = {step_count}",
        )
    run_ticks = sorted(data["runs"].unique())
    axis.set_xscale("log", base=2)
    axis.set_xticks(run_ticks, labels=[str(value) for value in run_ticks])
    axis.set_title("Vanguard Torch SBM solution quality vs. number of runs")
    axis.set_xlabel("SBM runs")
    axis.set_ylabel("Decoded margin (higher is better)")
    axis.grid(True, alpha=0.28)
    axis.legend(
        title="Step count",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=8,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def plot_quality_surface(data: pandas.DataFrame, output: Path) -> None:
    pivot = data.pivot(index="runs", columns="steps", values="margin")
    if pivot.isna().any().any():
        raise ValueError("3D surface requires a complete steps/runs grid")
    step_grid, run_grid = numpy.meshgrid(
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
    )
    quality_grid = pivot.to_numpy(dtype=float)

    figure = pyplot.figure(figsize=(11, 7.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    color_norm = Normalize(
        vmin=float(numpy.min(quality_grid)),
        vmax=float(numpy.max(quality_grid)),
    )
    surface = axis.plot_surface(
        step_grid,
        run_grid,
        quality_grid,
        cmap="plasma",
        norm=color_norm,
        edgecolor="black",
        linewidth=0.25,
        antialiased=True,
        alpha=0.94,
    )
    axis.scatter(
        step_grid,
        run_grid,
        quality_grid,
        color="black",
        s=8,
        alpha=0.65,
    )
    axis.set_title("Vanguard Torch SBM solution-quality surface", pad=18)
    axis.set_xlabel("SBM steps", labelpad=10)
    axis.set_ylabel("SBM runs", labelpad=10)
    axis.set_zlabel("Decoded margin (higher is better)", labelpad=10)
    axis.set_yticks(pivot.index.to_numpy())
    axis.view_init(elev=27, azim=-132)
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.68, pad=0.09)
    colorbar.set_label("Decoded margin")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()

    summary = arguments.summary.expanduser().resolve()
    output_directory = (
        arguments.output_directory.expanduser().resolve()
        if arguments.output_directory
        else summary.parent
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    data = load_summary(summary)

    outputs = (
        output_directory / "solve_time_vs_steps.png",
        output_directory / "solve_time_vs_runs.png",
        output_directory / "solve_time_surface_3d.png",
        output_directory / "solution_quality_vs_steps.png",
        output_directory / "solution_quality_vs_runs.png",
        output_directory / "solution_quality_surface_3d.png",
    )
    plot_time_vs_steps(data, outputs[0])
    plot_time_vs_runs(data, outputs[1])
    plot_time_surface(data, outputs[2])
    plot_quality_vs_steps(data, outputs[3])
    plot_quality_vs_runs(data, outputs[4])
    plot_quality_surface(data, outputs[5])
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

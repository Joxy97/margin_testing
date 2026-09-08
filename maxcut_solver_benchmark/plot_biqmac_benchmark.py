#!/usr/bin/env python3
"""Summarize and plot partial or completed Biq Mac MaxCut benchmark results."""

from __future__ import annotations

import argparse
import csv
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
import numpy


SOLVER_LABELS = {
    "torch_svl": "Torch SVL", "ortools_cpsat": "OR-Tools CP-SAT",
    "scip": "SCIP", "highs": "HiGHS", "cplex": "CPLEX",
}


def _read(path: Path) -> list[dict[str, object]]:
    latest = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            latest[(row["instance"], row["solver"], int(row["trial"]))] = row
    rows = []
    best_observed: dict[str, float] = defaultdict(lambda: -math.inf)
    for row in latest.values():
        try:
            cut = float(row["cut"])
        except ValueError:
            cut = math.nan
        if math.isfinite(cut):
            best_observed[row["instance"]] = max(best_observed[row["instance"]], cut)
    for row in latest.values():
        try:
            cut = float(row["cut"])
        except ValueError:
            cut = math.nan
        published = float(row["reference_cut"]) if row["reference_cut"] else None
        reference = published if published is not None else best_observed[row["instance"]]
        rows.append({
            "instance": row["instance"], "solver": row["solver"],
            "trial": int(row["trial"]), "status": row["status"],
            "seconds": float(row["solve_seconds"]), "cut": cut,
            "reference": reference,
            "reference_kind": "published_optimum" if published is not None else "best_observed",
            "quality": cut / reference if math.isfinite(cut) and reference else math.nan,
            "success": math.isfinite(cut) and cut + 1e-7 >= reference,
        })
    return rows


def _tts(seconds: float, successes: int, attempts: int, confidence: float = 0.99) -> float:
    probability = successes / attempts
    if probability <= 0: return math.inf
    repetitions = 1 if probability >= 1 else math.ceil(math.log1p(-confidence) / math.log1p(-probability))
    return seconds * repetitions


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows: groups[(str(row["instance"]), str(row["solver"]))].append(row)
    result = []
    for (instance, solver), group in sorted(groups.items()):
        finite_quality = [float(row["quality"]) for row in group if math.isfinite(float(row["quality"]))]
        successes = sum(bool(row["success"]) for row in group)
        mean_seconds = statistics.fmean(float(row["seconds"]) for row in group)
        result.append({
            "instance": instance, "solver": solver, "attempts": len(group),
            "status": ";".join(sorted({str(row["status"]) for row in group})),
            "reference_cut": group[0]["reference"], "reference_kind": group[0]["reference_kind"],
            "quality_median": statistics.median(finite_quality) if finite_quality else math.nan,
            "quality_best": max(finite_quality) if finite_quality else math.nan,
            "solve_seconds_mean": mean_seconds, "successes": successes,
            "success_probability": successes / len(group),
            "tts_99_seconds": _tts(mean_seconds, successes, len(group)),
        })
    return result


def _plot(path: Path, summaries: list[dict[str, object]], field: str, ylabel: str, log: bool = False) -> None:
    instances = sorted({str(row["instance"]) for row in summaries})
    solvers = [name for name in SOLVER_LABELS if any(row["solver"] == name for row in summaries)]
    x = numpy.arange(len(instances)); width = 0.8 / len(solvers)
    figure, axis = pyplot.subplots(figsize=(15, 6))
    for index, solver in enumerate(solvers):
        lookup = {str(row["instance"]): float(row[field]) for row in summaries if row["solver"] == solver}
        values = [lookup.get(instance, math.nan) for instance in instances]
        finite = [value if math.isfinite(value) else numpy.nan for value in values]
        axis.bar(x + (index - (len(solvers) - 1) / 2) * width, finite, width, label=SOLVER_LABELS[solver])
    axis.set_xticks(x, instances, rotation=45, ha="right")
    axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.25); axis.legend(ncol=3)
    if log: axis.set_yscale("log")
    figure.tight_layout(); figure.savefig(path, dpi=180); pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path); parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(); output = args.output_dir or args.results.parent / "plots"; output.mkdir(parents=True, exist_ok=True)
    summaries = _summaries(_read(args.results))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    _plot(output / "solution_quality.png", summaries, "quality_median", "Cut / reference cut")
    _plot(output / "solve_time.png", summaries, "solve_seconds_mean", "Mean solve seconds", True)
    _plot(output / "tts_99.png", summaries, "tts_99_seconds", "Empirical 99% TTS (seconds)", True)
    print(f"wrote {len(summaries)} summaries and three plots to {output}")


if __name__ == "__main__": main()

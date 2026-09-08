#!/usr/bin/env python3
"""Plot quality and empirical time-to-solution for a partial MaxCut sweep."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
import statistics
import tempfile

PLOT_CACHE = Path(tempfile.gettempdir()) / "margin_testing_matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot


SOLVER_LABELS = {
    "torch_sbm": "Torch SBM",
    "adaptive_torch_sbm": "Adaptive Torch SBM",
    "torch_svl": "Torch SVL",
}
COLORS = {
    "torch_sbm": "tab:blue",
    "adaptive_torch_sbm": "tab:orange",
    "torch_svl": "tab:green",
}
REQUIRED_COLUMNS = {
    "instance_id",
    "vertices",
    "solver",
    "amortized_seconds",
    "cut",
    "reference_cut",
    "success",
}


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid success value {value!r}")


def _read(path: Path) -> list[dict[str, object]]:
    """Read valid completed rows; tolerate a truncated final CSV row."""
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, raw in enumerate(reader, start=2):
            try:
                vertices = int(raw["vertices"])
                amortized_seconds = float(raw["amortized_seconds"])
                cut = float(raw["cut"])
                reference_cut = float(raw["reference_cut"])
                success = _boolean(raw["success"])
                solver = raw["solver"].strip()
                if (
                    vertices <= 0
                    or amortized_seconds <= 0.0
                    or not solver
                    or not all(
                        math.isfinite(x)
                        for x in (amortized_seconds, cut, reference_cut)
                    )
                    or reference_cut <= 0.0
                ):
                    raise ValueError("invalid numeric value")
            except (TypeError, ValueError):
                # An interrupted append can leave only the last record incomplete.
                if line_number == 2 or any(value not in (None, "") for value in raw.values()):
                    print(f"warning: skipping malformed row {line_number} in {path}")
                continue
            rows.append(
                {
                    "vertices": vertices,
                    "instance_id": raw["instance_id"].strip(),
                    "solver": solver,
                    "amortized_seconds": amortized_seconds,
                    "quality": cut / reference_cut,
                    "success": success,
                }
            )
    if not rows:
        raise ValueError(f"{path} has no valid completed rows")
    return rows


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _tts(seconds: float, successes: int, attempts: int, confidence: float) -> float:
    """Estimate time for at least one success at the requested confidence."""
    probability = successes / attempts
    if probability <= 0.0:
        return math.inf
    repetitions = 1 if probability >= 1.0 else math.ceil(
        math.log1p(-confidence) / math.log1p(-probability)
    )
    return seconds * repetitions


def _summaries(
    rows: list[dict[str, object]], confidence: float
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["vertices"]), str(row["solver"]))].append(row)
    result = []
    for (vertices, solver), group in sorted(groups.items()):
        quality = [float(row["quality"]) for row in group]
        successes = sum(bool(row["success"]) for row in group)
        instances: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            instances[str(row["instance_id"])].append(row)
        instance_tts = []
        for trials in instances.values():
            trial_successes = sum(bool(row["success"]) for row in trials)
            mean_seconds = statistics.fmean(
                float(row["amortized_seconds"]) for row in trials
            )
            instance_tts.append(
                _tts(mean_seconds, trial_successes, len(trials), confidence)
            )
        finite_tts = [value for value in instance_tts if math.isfinite(value)]
        median_tts = (
            statistics.median(instance_tts) if instance_tts else math.nan
        )
        result.append(
            {
                "vertices": vertices,
                "solver": solver,
                "instances": len(instances),
                "observations": len(group),
                "quality_median": statistics.median(quality),
                "quality_q25": _quantile(quality, 0.25),
                "quality_q75": _quantile(quality, 0.75),
                "successes": successes,
                "success_probability": successes / len(group),
                "finite_tts_instances": len(finite_tts),
                "tts_seconds_median": median_tts,
                "tts_seconds_q25": _quantile(instance_tts, 0.25),
                "tts_seconds_q75": _quantile(instance_tts, 0.75),
            }
        )
    return result


def _write_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    columns = list(summaries[0])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)


def _series(
    summaries: list[dict[str, object]], solver: str
) -> list[dict[str, object]]:
    return sorted(
        (row for row in summaries if row["solver"] == solver),
        key=lambda row: int(row["vertices"]),
    )


def _plot_quality(output: Path, summaries: list[dict[str, object]]) -> None:
    figure, axis = pyplot.subplots(figsize=(10, 6))
    for solver in sorted({str(row["solver"]) for row in summaries}):
        rows = _series(summaries, solver)
        x = [int(row["vertices"]) for row in rows]
        median = [float(row["quality_median"]) for row in rows]
        lower = [float(row["quality_q25"]) for row in rows]
        upper = [float(row["quality_q75"]) for row in rows]
        color = COLORS.get(solver)
        axis.plot(
            x,
            median,
            marker="o",
            label=SOLVER_LABELS.get(solver, solver),
            color=color,
        )
        axis.fill_between(x, lower, upper, color=color, alpha=0.16)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    axis.set_title("MaxCut solution quality")
    axis.set_xlabel("Vertices")
    axis.set_ylabel("Cut / reference cut (median; IQR band)")
    axis.set_xscale("log", base=2)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    pyplot.close(figure)


def _plot_tts(
    output: Path, summaries: list[dict[str, object]], confidence: float
) -> None:
    figure, axis = pyplot.subplots(figsize=(10, 6))
    omitted = []
    for solver in sorted({str(row["solver"]) for row in summaries}):
        rows = _series(summaries, solver)
        plottable = [
            row
            for row in rows
            if math.isfinite(float(row["tts_seconds_median"]))
        ]
        omitted.extend((solver, int(row["vertices"])) for row in rows if row not in plottable)
        if not plottable:
            continue
        axis.plot(
            [int(row["vertices"]) for row in plottable],
            [float(row["tts_seconds_median"]) for row in plottable],
            marker="o",
            label=SOLVER_LABELS.get(solver, solver),
            color=COLORS.get(solver),
        )
    axis.set_title(f"MaxCut empirical time to solution ({confidence:.1%} confidence)")
    axis.set_xlabel("Vertices")
    axis.set_ylabel("Estimated seconds")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend()
    if omitted:
        print(f"warning: omitted {len(omitted)} zero-success solver/size groups from TTS")
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, help="raw_results.csv from run_benchmark.py"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.99)
    arguments = parser.parse_args()
    if not 0.0 < arguments.confidence < 1.0:
        parser.error("--confidence must be between zero and one")
    output_dir = arguments.output_dir or arguments.results.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = _summaries(_read(arguments.results), arguments.confidence)
    outputs = [
        output_dir / "summary.csv",
        output_dir / "solution_quality.png",
        output_dir / "time_to_solution.png",
    ]
    _write_summary(outputs[0], summaries)
    _plot_quality(outputs[1], summaries)
    _plot_tts(outputs[2], summaries, arguments.confidence)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a resumable Torch SVL steps-by-dt sweep on selected Biq Mac graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from margin_calculator.optimization.optimization_solver.bqm_solver import TorchSVLBQMSolver
from run_biqmac_benchmark import _cut, _graph, _qubo


CSV_FIELDS = (
    "timestamp_utc", "instance", "vertices", "edges", "density", "trial",
    "solver_seed", "runs", "steps", "dt", "solve_seconds", "cut",
    "reference_cut", "quality", "success",
)


def _completed(path: Path) -> set[tuple[str, int, str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as source:
        return {
            (row["instance"], int(row["steps"]), row["dt"], int(row["trial"]))
            for row in csv.DictReader(source)
        }


def run(config_path: Path, resume: bool) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["runs"]) != 8:
        raise ValueError("the steps/dt sweep requires exactly 8 runs")
    steps_values = [int(value) for value in config["steps_values"]]
    if steps_values != list(range(1000, 10001, 1000)):
        raise ValueError("steps_values must be 1000 through 10000 in increments of 1000")
    dt_values = [float(value) for value in config["dt_values"]]
    if len(dt_values) < 3 or any(value <= 0 for value in dt_values):
        raise ValueError("dt_values must contain at least three positive values")
    data = (config_path.parent / config["data_dir"]).resolve()
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "raw_results.csv"
    if results_path.exists() and not resume:
        raise FileExistsError(f"results exist in {output}; pass --resume")
    completed = _completed(results_path) if resume else set()
    graphs = [_graph(data / relative) for relative in config["instances"]]
    manifest = dict(config)
    manifest.update(
        design="cartesian_steps_dt", solver="torch_svl",
        configuration_count=len(steps_values) * len(dt_values),
        expected_records=len(graphs) * len(steps_values) * len(dt_values) * int(config["trials"]),
        started_utc=datetime.now(timezone.utc).isoformat(), csv_schema=list(CSV_FIELDS),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    solver = TorchSVLBQMSolver("cpu")
    # One untimed warm-up keeps initialization out of measured solver time.
    solver.solve(
        _qubo(graphs[0]),
        {"runs": 1, "steps": 10, "dt": dt_values[0], "seed": int(config["seed"]),
         "dtype": "float32", "run_batch_size": 1, **config["fixed_parameters"]},
    )
    for graph in graphs:
        problem = _qubo(graph)
        reference = float(config["references"][graph.name])
        density = 2.0 * len(graph.heads) / (graph.vertices * (graph.vertices - 1))
        for dt in dt_values:
            dt_key = format(dt, ".10g")
            for steps in steps_values:
                for trial in range(int(config["trials"])):
                    key = (graph.name, steps, dt_key, trial)
                    if key in completed:
                        continue
                    seed = int(config["seed"]) + trial * 104_729
                    parameters = {
                        "runs": 8, "steps": steps, "dt": dt, "seed": seed,
                        "dtype": "float32", "run_batch_size": 8,
                        **config["fixed_parameters"],
                    }
                    started = time.perf_counter()
                    result = solver.solve(problem, parameters)
                    elapsed = time.perf_counter() - started
                    cut = _cut(graph, result.sample)
                    if not math.isclose(float(result.energy), -cut, abs_tol=1e-5):
                        raise RuntimeError("Torch SVL returned inconsistent MaxCut energy")
                    row = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "instance": graph.name,
                        "vertices": graph.vertices, "edges": len(graph.heads), "density": density,
                        "trial": trial, "solver_seed": seed, "runs": 8, "steps": steps,
                        "dt": dt_key, "solve_seconds": elapsed, "cut": cut,
                        "reference_cut": reference, "quality": cut / reference,
                        "success": int(cut + 1e-7 >= reference),
                    }
                    header = not results_path.exists() or results_path.stat().st_size == 0
                    with results_path.open("a", newline="", encoding="utf-8") as destination:
                        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
                        if header: writer.writeheader()
                        writer.writerow(row); destination.flush(); os.fsync(destination.fileno())
                    completed.add(key)
                    print(
                        f"{graph.name} dt={dt_key} steps={steps} trial={trial} "
                        f"quality={row['quality']:.6f} time={elapsed:.3f}s",
                        flush=True,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); run(args.config.resolve(), args.resume)


if __name__ == "__main__": main()

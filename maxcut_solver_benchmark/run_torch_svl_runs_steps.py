#!/usr/bin/env python3
"""Run a resumable Cartesian Torch SVL runs-by-steps MaxCut sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from margin_calculator.optimization.optimization_solver.bqm_solver import TorchSVLBQMSolver
from run_benchmark import _cut, _estimated_bytes, _qubo, _synchronize
from run_torch_svl_sweep import _instances


FIXED_PARAMETERS = (
    "dt", "mass", "damping", "temperature", "integrator",
    "transverse_field_initial", "transverse_field_final",
    "problem_scale_initial", "problem_scale_final",
)
CSV_FIELDS = (
    "timestamp_utc", "configuration_id", "instance_id", "vertices", "edges",
    "trial", "solver_seed", "runs", "steps", *FIXED_PARAMETERS,
    "solve_seconds", "energy", "cut", "reference_cut", "quality", "success",
    "estimated_bytes",
)


def _config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    required = {
        "source_results_dir", "output_dir", "sizes", "trials", "graph_seed",
        "device", "dtype", "run_batch_size", "memory_limit_gb", "runs_values",
        "steps_values", "fixed_parameters",
    }
    if set(value) != required:
        raise ValueError(
            f"config keys must be exactly {sorted(required)}; got {sorted(value)}"
        )
    if set(value["fixed_parameters"]) != set(FIXED_PARAMETERS):
        raise ValueError("fixed_parameters must specify every non-runs/non-steps parameter")
    value["sizes"] = [int(item) for item in value["sizes"]]
    value["runs_values"] = [int(item) for item in value["runs_values"]]
    value["steps_values"] = [int(item) for item in value["steps_values"]]
    value["trials"] = int(value["trials"])
    value["graph_seed"] = int(value["graph_seed"])
    value["run_batch_size"] = int(value["run_batch_size"])
    value["memory_limit_gb"] = float(value["memory_limit_gb"])
    if any(item <= 0 for item in value["runs_values"] + value["steps_values"]):
        raise ValueError("runs_values and steps_values must be positive")
    if value["trials"] <= 0 or value["memory_limit_gb"] <= 0:
        raise ValueError("trials and memory_limit_gb must be positive")
    return value


def _completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as source:
        return {
            (row["configuration_id"], row["instance_id"], int(row["trial"]))
            for row in csv.DictReader(source)
        }


def _append(path: Path, row: dict[str, Any]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        destination.flush()
        os.fsync(destination.fileno())


def run(config_path: Path, resume: bool) -> None:
    config = _config(config_path)
    source = (config_path.parent / config["source_results_dir"]).resolve()
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "raw_results.csv"
    if results_path.exists() and not resume:
        raise FileExistsError(f"results already exist in {output}; pass --resume")
    completed = _completed(results_path) if resume else set()
    instances = _instances(source, config["sizes"])
    manifest = dict(config)
    manifest.update(
        design="cartesian_runs_steps",
        configuration_count=len(config["runs_values"]) * len(config["steps_values"]),
        expected_records=(
            len(config["runs_values"]) * len(config["steps_values"])
            * len(instances) * config["trials"]
        ),
        started_utc=datetime.now(timezone.utc).isoformat(),
        csv_schema=list(CSV_FIELDS),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    solver = TorchSVLBQMSolver(config["device"])
    warmup_parameters = dict(config["fixed_parameters"])
    warmup_parameters.update(
        runs=1, steps=10, seed=config["graph_seed"], dtype=config["dtype"],
        run_batch_size=config["run_batch_size"],
    )
    solver.solve(_qubo(instances[0]), warmup_parameters)
    _synchronize([config["device"]])
    memory_limit = int(config["memory_limit_gb"] * 1024 ** 3)

    for runs in config["runs_values"]:
        for steps in config["steps_values"]:
            configuration_id = f"runs_{runs:02d}_steps_{steps:05d}"
            for instance in instances:
                problem = _qubo(instance)
                estimate = _estimated_bytes(
                    instance.vertices, len(instance.heads), runs,
                    config["dtype"], "torch_svl", 1,
                )
                if estimate > memory_limit:
                    raise MemoryError(
                        f"n={instance.vertices} estimate {estimate:,} exceeds {memory_limit:,}"
                    )
                for trial in range(config["trials"]):
                    key = (configuration_id, instance.instanceId, trial)
                    if key in completed:
                        continue
                    seed = config["graph_seed"] + trial * 104_729
                    parameters = dict(config["fixed_parameters"])
                    parameters.update(
                        runs=runs, steps=steps, seed=seed, dtype=config["dtype"],
                        run_batch_size=config["run_batch_size"],
                    )
                    _synchronize([config["device"]])
                    started = time.perf_counter()
                    result = solver.solve(problem, parameters)
                    _synchronize([config["device"]])
                    elapsed = time.perf_counter() - started
                    cut = _cut(instance.heads, instance.tails, instance.weights, result.sample)
                    if not math.isclose(float(result.energy), -cut, abs_tol=1.0e-7):
                        raise RuntimeError("Torch SVL returned inconsistent MaxCut energy")
                    row = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "configuration_id": configuration_id,
                        "instance_id": instance.instanceId,
                        "vertices": instance.vertices,
                        "edges": len(instance.heads),
                        "trial": trial,
                        "solver_seed": seed,
                        "runs": runs,
                        "steps": steps,
                        **config["fixed_parameters"],
                        "solve_seconds": elapsed,
                        "energy": float(result.energy),
                        "cut": cut,
                        "reference_cut": instance.referenceCut,
                        "quality": cut / instance.referenceCut,
                        "success": int(cut + 1.0e-9 >= instance.referenceCut),
                        "estimated_bytes": estimate,
                    }
                    _append(results_path, row)
                    completed.add(key)
                    print(
                        f"runs={runs} steps={steps} n={instance.vertices} trial={trial} "
                        f"quality={row['quality']:.6f} time={elapsed:.3f}s",
                        flush=True,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    try:
        run(arguments.config.resolve(), arguments.resume)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()

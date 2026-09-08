#!/usr/bin/env python3
"""Run a resumable one-parameter-at-a-time Torch SBM MaxCut sweep."""

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

from margin_calculator.optimization.optimization_solver.bqm_solver import TorchSBMBQMSolver
from run_benchmark import _cut, _estimated_bytes, _qubo, _synchronize
from run_torch_svl_sweep import _instances


DYNAMIC_PARAMETERS = ("dt", "a0", "c0", "gamma", "initial_scale")
CSV_FIELDS = (
    "timestamp_utc", "configuration_id", "parameter", "parameter_value",
    "instance_id", "vertices", "edges", "trial", "solver_seed", "runs",
    "steps", *DYNAMIC_PARAMETERS, "solve_seconds", "energy", "cut",
    "reference_cut", "quality", "success", "estimated_bytes",
)


def _config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    required = {
        "source_results_dir", "output_dir", "sizes", "trials", "graph_seed",
        "device", "dtype", "run_batch_size", "memory_limit_gb", "steps",
        "runs", "baseline", "sweep",
    }
    if set(value) != required:
        raise ValueError(f"config keys must be exactly {sorted(required)}")
    if set(value["baseline"]) != set(DYNAMIC_PARAMETERS) or set(value["sweep"]) != set(DYNAMIC_PARAMETERS):
        raise ValueError("baseline and sweep must specify every Torch SBM dynamics parameter")
    for name in DYNAMIC_PARAMETERS:
        values = value["sweep"][name]
        if not isinstance(values, list) or not values or value["baseline"][name] not in values:
            raise ValueError(f"sweep.{name} must be nonempty and include its baseline")
    value["sizes"] = [int(item) for item in value["sizes"]]
    for name in ("trials", "graph_seed", "run_batch_size", "steps", "runs"):
        value[name] = int(value[name])
    value["memory_limit_gb"] = float(value["memory_limit_gb"])
    if value["steps"] != 1000 or value["runs"] != 4:
        raise ValueError("this sweep requires steps=1000 and runs=4")
    return value


def _configurations(config: dict[str, Any]) -> list[tuple[str, str, Any, dict[str, Any]]]:
    baseline = dict(config["baseline"])
    result = [("baseline", "baseline", "baseline", baseline)]
    for name in DYNAMIC_PARAMETERS:
        for index, value in enumerate(config["sweep"][name]):
            if value == baseline[name]:
                continue
            parameters = dict(baseline)
            parameters[name] = value
            result.append((f"{name}_{index:02d}", name, value, parameters))
    return result


def _completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as source:
        return {(row["configuration_id"], row["instance_id"], int(row["trial"])) for row in csv.DictReader(source)}


def _append(path: Path, row: dict[str, Any]) -> None:
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        if header:
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
    configurations = _configurations(config)
    manifest = dict(config)
    manifest.update(
        design="one_parameter_at_a_time", solver="torch_sbm",
        configuration_count=len(configurations),
        expected_records=len(configurations) * len(instances) * config["trials"],
        started_utc=datetime.now(timezone.utc).isoformat(), csv_schema=list(CSV_FIELDS),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    solver = TorchSBMBQMSolver(config["device"])
    warmup = dict(config["baseline"])
    warmup.update(steps=10, runs=1, seed=config["graph_seed"], dtype=config["dtype"], run_batch_size=1)
    solver.solve(_qubo(instances[0]), warmup)
    memory_limit = int(config["memory_limit_gb"] * 1024 ** 3)
    for configuration_id, parameter, parameter_value, dynamics in configurations:
        for instance in instances:
            problem = _qubo(instance)
            estimate = _estimated_bytes(instance.vertices, len(instance.heads), config["runs"], config["dtype"], "torch_sbm", 1)
            if estimate > memory_limit:
                raise MemoryError(f"n={instance.vertices} estimate exceeds configured memory limit")
            for trial in range(config["trials"]):
                key = (configuration_id, instance.instanceId, trial)
                if key in completed:
                    continue
                seed = config["graph_seed"] + trial * 104_729
                parameters = dict(dynamics)
                parameters.update(steps=config["steps"], runs=config["runs"], seed=seed, dtype=config["dtype"], run_batch_size=config["run_batch_size"])
                _synchronize([config["device"]])
                started = time.perf_counter()
                result = solver.solve(problem, parameters)
                _synchronize([config["device"]])
                elapsed = time.perf_counter() - started
                cut = _cut(instance.heads, instance.tails, instance.weights, result.sample)
                if not math.isclose(float(result.energy), -cut, abs_tol=1e-7):
                    raise RuntimeError("Torch SBM returned inconsistent MaxCut energy")
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "configuration_id": configuration_id,
                    "parameter": parameter, "parameter_value": json.dumps(parameter_value), "instance_id": instance.instanceId,
                    "vertices": instance.vertices, "edges": len(instance.heads), "trial": trial, "solver_seed": seed,
                    "runs": config["runs"], "steps": config["steps"], **dynamics, "solve_seconds": elapsed,
                    "energy": float(result.energy), "cut": cut, "reference_cut": instance.referenceCut,
                    "quality": cut / instance.referenceCut, "success": int(cut + 1e-9 >= instance.referenceCut),
                    "estimated_bytes": estimate,
                }
                _append(results_path, row)
                completed.add(key)
                print(f"{configuration_id} n={instance.vertices} trial={trial} quality={row['quality']:.6f} time={elapsed:.3f}s", flush=True)


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

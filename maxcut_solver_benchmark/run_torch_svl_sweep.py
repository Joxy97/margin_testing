#!/usr/bin/env python3
"""Run a resumable one-parameter-at-a-time Torch SVL MaxCut sweep."""

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

import numpy

from margin_calculator.optimization.optimization_solver.bqm_solver import (
    TorchSVLBQMSolver,
)
from run_benchmark import (
    MaxCutInstance,
    _cut,
    _estimated_bytes,
    _qubo,
    _synchronize,
)


DYNAMIC_PARAMETERS = (
    "steps",
    "runs",
    "dt",
    "mass",
    "damping",
    "temperature",
    "integrator",
    "transverse_field_initial",
    "transverse_field_final",
    "problem_scale_initial",
    "problem_scale_final",
)
CSV_FIELDS = (
    "timestamp_utc",
    "configuration_id",
    "parameter",
    "parameter_value",
    "instance_id",
    "vertices",
    "edges",
    "trial",
    "solver_seed",
    *DYNAMIC_PARAMETERS,
    "solve_seconds",
    "energy",
    "cut",
    "reference_cut",
    "quality",
    "success",
    "estimated_bytes",
)


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    required = {
        "source_results_dir", "output_dir", "sizes", "trials", "graph_seed",
        "device", "dtype", "run_batch_size", "memory_limit_gb", "baseline",
        "sweep",
    }
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing or unknown:
        raise ValueError(
            f"invalid sweep config keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    baseline = dict(value["baseline"])
    sweep = dict(value["sweep"])
    if set(baseline) != set(DYNAMIC_PARAMETERS):
        raise ValueError("baseline must specify every Torch SVL dynamics parameter")
    if set(sweep) != set(DYNAMIC_PARAMETERS):
        raise ValueError("sweep must specify every Torch SVL dynamics parameter")
    if any(not isinstance(values, list) or not values for values in sweep.values()):
        raise ValueError("every sweep parameter must have a nonempty value list")
    for name, values in sweep.items():
        if baseline[name] not in values:
            raise ValueError(f"sweep.{name} must include its baseline value")
    if int(value["trials"]) <= 0 or float(value["memory_limit_gb"]) <= 0:
        raise ValueError("trials and memory_limit_gb must be positive")
    value["sizes"] = [int(size) for size in value["sizes"]]
    value["trials"] = int(value["trials"])
    value["graph_seed"] = int(value["graph_seed"])
    value["run_batch_size"] = int(value["run_batch_size"])
    value["memory_limit_gb"] = float(value["memory_limit_gb"])
    value["baseline"] = baseline
    value["sweep"] = sweep
    return value


def _configurations(config: dict[str, Any]) -> list[tuple[str, str, Any, dict[str, Any]]]:
    """Return unique OFAT configurations, including the baseline once."""
    baseline = config["baseline"]
    result = [("baseline", "baseline", "baseline", dict(baseline))]
    for name in DYNAMIC_PARAMETERS:
        for index, value in enumerate(config["sweep"][name]):
            if value == baseline[name]:
                continue
            parameters = dict(baseline)
            parameters[name] = value
            result.append((f"{name}_{index:02d}", name, value, parameters))
    return result


def _load_instance(path: Path) -> MaxCutInstance:
    with numpy.load(path) as data:
        return MaxCutInstance(
            path.stem,
            int(data["vertices"]),
            data["heads"],
            data["tails"],
            data["weights"],
            int(data["graph_seed"]),
            float(data["reference_cut"]),
            str(data["reference_kind"]),
        )


def _instances(source: Path, sizes: list[int]) -> list[MaxCutInstance]:
    result = []
    for vertices in sizes:
        matches = sorted((source / "instances").glob(f"er_n{vertices:07d}_*.npz"))
        if not matches:
            raise FileNotFoundError(f"no saved MaxCut instance for n={vertices} in {source}")
        result.extend(_load_instance(path) for path in matches)
    return result


def _completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as source:
        return {
            (row["configuration_id"], row["instance_id"], int(row["trial"]))
            for row in csv.DictReader(source)
        }


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        destination.flush()
        os.fsync(destination.fileno())


def run(config_path: Path, resume: bool) -> None:
    config = _read_config(config_path)
    source = (config_path.parent / config["source_results_dir"]).resolve()
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "raw_results.csv"
    if csv_path.exists() and not resume:
        raise FileExistsError(f"results already exist in {output}; pass --resume")
    completed = _completed(csv_path) if resume else set()
    instances = _instances(source, config["sizes"])
    configurations = _configurations(config)
    manifest = dict(config)
    manifest.update(
        design="one_parameter_at_a_time",
        configuration_count=len(configurations),
        started_utc=datetime.now(timezone.utc).isoformat(),
        csv_schema=list(CSV_FIELDS),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    solver = TorchSVLBQMSolver(config["device"])
    warmup = _qubo(instances[0])
    warmup_parameters = dict(config["baseline"])
    warmup_parameters.update(
        seed=config["graph_seed"], dtype=config["dtype"],
        run_batch_size=config["run_batch_size"],
    )
    solver.solve(warmup, warmup_parameters)
    _synchronize([config["device"]])
    memory_limit = int(config["memory_limit_gb"] * 1024 ** 3)

    for configuration_id, parameter, parameter_value, dynamics in configurations:
        for instance in instances:
            problem = _qubo(instance)
            estimate = _estimated_bytes(
                instance.vertices, len(instance.heads), int(dynamics["runs"]),
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
                parameters = dict(dynamics)
                parameters.update(
                    seed=seed, dtype=config["dtype"],
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
                    "parameter": parameter,
                    "parameter_value": json.dumps(parameter_value),
                    "instance_id": instance.instanceId,
                    "vertices": instance.vertices,
                    "edges": len(instance.heads),
                    "trial": trial,
                    "solver_seed": seed,
                    **dynamics,
                    "solve_seconds": elapsed,
                    "energy": float(result.energy),
                    "cut": cut,
                    "reference_cut": instance.referenceCut,
                    "quality": cut / instance.referenceCut,
                    "success": int(cut + 1.0e-9 >= instance.referenceCut),
                    "estimated_bytes": estimate,
                }
                _append(csv_path, [row])
                completed.add(key)
                print(
                    f"{configuration_id} n={instance.vertices} trial={trial} "
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

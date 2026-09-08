#!/usr/bin/env python3
"""Resumable MaxCut benchmark for the three Torch QUBO solvers.

Run this file from the repository root with ``PYTHONPATH=src``.  Instances are
generated once, stored as compressed NumPy files, and reused by every solver.
Only settings shared by all three solvers are supplied by this benchmark.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import heapq
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy

from margin_calculator.optimization.optimization_problem.qubo_problem import (
    QUBOProblem,
)
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    AdaptiveTorchSBMBQMSolver,
    TorchSBMBQMSolver,
    TorchSVLBQMSolver,
)


SOLVERS = {
    "torch_sbm": TorchSBMBQMSolver,
    "adaptive_torch_sbm": AdaptiveTorchSBMBQMSolver,
    "torch_svl": TorchSVLBQMSolver,
}
CSV_FIELDS = (
    "timestamp_utc",
    "instance_id",
    "vertices",
    "edges",
    "density",
    "graph_seed",
    "solver",
    "trial",
    "solver_seed",
    "runs",
    "steps",
    "device_count",
    "batch_size",
    "solve_seconds",
    "amortized_seconds",
    "energy",
    "cut",
    "reference_cut",
    "reference_kind",
    "success",
    "estimated_bytes",
)

_REFERENCE_ADJACENCY: Any | None = None


@dataclass(frozen=True)
class MaxCutInstance:
    instanceId: str
    vertices: int
    heads: numpy.ndarray
    tails: numpy.ndarray
    weights: numpy.ndarray
    graphSeed: int
    referenceCut: float
    referenceKind: str


def _config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("benchmark config must contain a JSON object")
    allowed = {
        "sizes", "instances_per_size", "trials", "edge_probability",
        "graph_seed", "runs", "steps", "dtype", "devices", "output_dir",
        "memory_fraction", "exact_reference_max_vertices",
        "reference_restarts", "reference_workers", "batch_size",
        "run_batch_size", "solvers", "memory_limit_gb",
    }
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown benchmark config keys: {sorted(unknown)}")
    result = {
        "sizes": value.get("sizes", [64, 128, 256, 512, 1024, 2048, 4096]),
        "instances_per_size": int(value.get("instances_per_size", 3)),
        "trials": int(value.get("trials", 5)),
        "edge_probability": float(value.get("edge_probability", 0.1)),
        "graph_seed": int(value.get("graph_seed", 20250831)),
        "runs": int(value.get("runs", 4)),
        "steps": int(value.get("steps", 1000)),
        "dtype": str(value.get("dtype", "float32")),
        "devices": value.get("devices", ["auto"]),
        "output_dir": str(value.get("output_dir", "results")),
        "memory_fraction": float(value.get("memory_fraction", 0.75)),
        "memory_limit_gb": (
            None
            if value.get("memory_limit_gb") is None
            else float(value["memory_limit_gb"])
        ),
        "exact_reference_max_vertices": int(
            value.get("exact_reference_max_vertices", 22)
        ),
        "reference_restarts": int(value.get("reference_restarts", 64)),
        "reference_workers": int(value.get("reference_workers", min(4, os.cpu_count() or 1))),
        "batch_size": (
            0 if value.get("batch_size") is None else int(value["batch_size"])
        ),
        "run_batch_size": int(value.get("run_batch_size", 4)),
        "solvers": value.get("solvers", list(SOLVERS)),
    }
    result["sizes"] = [int(size) for size in result["sizes"]]
    if not result["sizes"] or any(size < 2 for size in result["sizes"]):
        raise ValueError("sizes must contain integers of at least two")
    if result["sizes"] != sorted(set(result["sizes"])):
        raise ValueError("sizes must be unique and increasing")
    if result["runs"] != 4:
        raise ValueError("runs must be exactly 4 for this comparison")
    if result["steps"] < 1000:
        raise ValueError("steps must be at least 1000")
    if result["dtype"] not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    if not 0.0 < result["edge_probability"] <= 1.0:
        raise ValueError("edge_probability must be in (0, 1]")
    if not 0.0 < result["memory_fraction"] <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if result["memory_limit_gb"] is not None and result["memory_limit_gb"] <= 0.0:
        raise ValueError("memory_limit_gb must be positive when specified")
    for name in (
        "instances_per_size", "trials", "reference_restarts",
        "reference_workers",
    ):
        if result[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if result["batch_size"] < 0 or result["run_batch_size"] <= 0:
        raise ValueError("batch_size must be nonnegative and run_batch_size positive")
    if (
        not isinstance(result["devices"], list)
        or not result["devices"]
        or any(not isinstance(device, str) for device in result["devices"])
    ):
        raise ValueError("devices must be a nonempty list of device names")
    if (
        not isinstance(result["solvers"], list)
        or not result["solvers"]
        or any(name not in SOLVERS for name in result["solvers"])
    ):
        raise ValueError(f"solvers must be selected from {sorted(SOLVERS)}")
    return result


def _generate_graph(vertices: int, probability: float, seed: int) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Generate an unweighted Erdos-Renyi graph without an n-by-n array."""
    rng = numpy.random.default_rng(seed)
    head_parts: list[numpy.ndarray] = []
    tail_parts: list[numpy.ndarray] = []
    for head in range(vertices - 1):
        chosen = numpy.flatnonzero(rng.random(vertices - head - 1) < probability)
        if len(chosen):
            head_parts.append(numpy.full(len(chosen), head, dtype=numpy.uint32))
            tail_parts.append((chosen + head + 1).astype(numpy.uint32))
    if not head_parts:
        # A nonempty graph makes quality ratios and timing interpretation sane.
        return (
            numpy.asarray([0], dtype=numpy.uint32),
            numpy.asarray([1], dtype=numpy.uint32),
            numpy.asarray([1.0], dtype=numpy.float64),
        )
    heads = numpy.concatenate(head_parts)
    tails = numpy.concatenate(tail_parts)
    return heads, tails, numpy.ones(len(heads), dtype=numpy.float64)


def _cut(heads: numpy.ndarray, tails: numpy.ndarray, weights: numpy.ndarray, sample: Sequence[int]) -> float:
    values = numpy.asarray(sample, dtype=numpy.uint8)
    return float(weights[values[heads] != values[tails]].sum())


def _exact_reference(vertices: int, heads: numpy.ndarray, tails: numpy.ndarray, weights: numpy.ndarray) -> float:
    best = 0.0
    # Fix vertex zero to zero, removing the complement symmetry.  Chunking
    # bounds temporary RAM even at the upper exact-reference limit.
    count = 1 << (vertices - 1)
    bit_positions = numpy.arange(vertices - 1, dtype=numpy.uint64)
    for start in range(0, count, 65_536):
        integers = numpy.arange(start, min(start + 65_536, count), dtype=numpy.uint64)
        samples = numpy.zeros((len(integers), vertices), dtype=numpy.uint8)
        samples[:, 1:] = ((integers[:, None] >> bit_positions) & 1).astype(numpy.uint8)
        cuts = numpy.zeros(len(integers), dtype=numpy.float64)
        # Edge chunks avoid an assignments-by-all-edges temporary.
        for edge_start in range(0, len(heads), 4096):
            edge_stop = min(edge_start + 4096, len(heads))
            different = samples[:, heads[edge_start:edge_stop]] != samples[:, tails[edge_start:edge_stop]]
            cuts += different @ weights[edge_start:edge_stop]
        best = max(best, float(cuts.max(initial=0.0)))
    return best


def _local_search_reference(adjacency: Any, seed: int) -> float:
    vertices = adjacency.shape[0]
    rng = numpy.random.default_rng(seed)
    sample = rng.integers(0, 2, size=vertices, dtype=numpy.uint8)
    spins = 1.0 - 2.0 * sample.astype(numpy.float64)
    gains = spins * adjacency.dot(spins)
    versions = numpy.zeros(vertices, dtype=numpy.uint32)
    heap = [(-float(gain), vertex, 0) for vertex, gain in enumerate(gains)]
    heapq.heapify(heap)
    while heap:
        negative_gain, vertex, version = heapq.heappop(heap)
        if version != int(versions[vertex]):
            continue
        if -negative_gain <= 1.0e-12:
            break
        old_spin = spins[vertex]
        sample[vertex] ^= 1
        spins[vertex] = -old_spin
        gains[vertex] = -gains[vertex]
        versions[vertex] += 1
        heapq.heappush(heap, (-float(gains[vertex]), vertex, int(versions[vertex])))
        edge_start = adjacency.indptr[vertex]
        edge_stop = adjacency.indptr[vertex + 1]
        neighbors = adjacency.indices[edge_start:edge_stop]
        neighbor_weights = adjacency.data[edge_start:edge_stop]
        gains[neighbors] -= 2.0 * neighbor_weights * spins[neighbors] * old_spin
        for neighbor in neighbors:
            versions[neighbor] += 1
            heapq.heappush(
                heap,
                (-float(gains[neighbor]), int(neighbor), int(versions[neighbor])),
            )
    return float(-0.25 * spins @ adjacency.dot(spins) + 0.25 * adjacency.data.sum())


def _forked_reference(seed: int) -> float:
    if _REFERENCE_ADJACENCY is None:
        raise RuntimeError("reference worker has no adjacency matrix")
    return _local_search_reference(_REFERENCE_ADJACENCY, seed)


def _heuristic_reference(
    vertices: int,
    heads: numpy.ndarray,
    tails: numpy.ndarray,
    weights: numpy.ndarray,
    seed: int,
    restarts: int,
    workers: int = 1,
) -> float:
    from scipy.sparse import csr_matrix

    rows = numpy.concatenate((heads, tails))
    columns = numpy.concatenate((tails, heads))
    adjacency = csr_matrix(
        (numpy.concatenate((weights, weights)), (rows, columns)),
        shape=(vertices, vertices),
    )
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    restart_seeds = numpy.random.SeedSequence(seed).generate_state(
        restarts, dtype=numpy.uint64
    )
    worker_count = min(workers, restarts)
    if worker_count == 1 or "fork" not in multiprocessing.get_all_start_methods():
        return max(
            _local_search_reference(adjacency, int(restart_seed))
            for restart_seed in restart_seeds
        )
    global _REFERENCE_ADJACENCY
    _REFERENCE_ADJACENCY = adjacency
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            return max(executor.map(_forked_reference, map(int, restart_seeds)))
    finally:
        _REFERENCE_ADJACENCY = None


def _qubo(instance: MaxCutInstance) -> QUBOProblem:
    linear = numpy.zeros(instance.vertices, dtype=numpy.float64)
    numpy.add.at(linear, instance.heads, -instance.weights)
    numpy.add.at(linear, instance.tails, -instance.weights)
    return QUBOProblem(
        linear,
        instance.heads,
        instance.tails,
        2.0 * instance.weights,
    )


def _available_host_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(pages * os.sysconf("SC_PAGE_SIZE"))


def _estimated_bytes(vertices: int, edges: int, runs: int, dtype: str, solver: str, problems: int) -> int:
    scalar = 4 if dtype == "float32" else 8
    # QUBO + symmetric CSR, candidate copies, and solver working tensors.
    state_arrays = {"torch_sbm": 7, "adaptive_torch_sbm": 14, "torch_svl": 10}[solver]
    per_problem = 80 * edges + 32 * vertices + state_arrays * vertices * runs * scalar
    return int(per_problem * problems * 1.25 + 32 * 1024 * 1024)


def _devices(configured: list[str]) -> list[str]:
    import torch

    if configured in (["auto"], ["all"]):
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        return ["cpu"]
    return configured


def _synchronize(devices: Sequence[str]) -> None:
    import torch

    for name in devices:
        device = torch.device(name)
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def _device_budget(devices: Sequence[str], fraction: float) -> int:
    """Return the smallest currently free per-device memory allowance."""
    import torch

    free = []
    for name in devices:
        device = torch.device(name)
        if device.type == "cuda":
            free.append(int(torch.cuda.mem_get_info(device)[0] * fraction))
    return min(free, default=sys.maxsize)


def _solver(name: str, devices: Sequence[str]) -> Any:
    solver_class = SOLVERS[name]
    if len(devices) == 1:
        return solver_class(device=devices[0])
    return solver_class(devices=list(devices))


def _warm_up(solver: Any, device_count: int, parameters: dict[str, Any], devices: Sequence[str]) -> None:
    """Initialize every configured device before collecting timings."""
    warmup = QUBOProblem(
        numpy.asarray([-1.0, -1.0]),
        numpy.asarray([0], dtype=numpy.uint32),
        numpy.asarray([1], dtype=numpy.uint32),
        numpy.asarray([2.0]),
    )
    _synchronize(devices)
    solver.solveMany([warmup] * device_count, parameters)
    _synchronize(devices)


def _load_completed(path: Path) -> set[tuple[str, str, int, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (
                row["instance_id"], row["solver"], int(row["trial"]),
                int(row["runs"]), int(row["steps"]),
            )
            for row in csv.DictReader(stream)
        }


def _write_records(csv_path: Path, jsonl_path: Path, records: Iterable[dict[str, Any]]) -> None:
    records = list(records)
    if not records:
        return
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(records)
        stream.flush()
        os.fsync(stream.fileno())
    with jsonl_path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _instance(path: Path, vertices: int, probability: float, seed: int, exact_limit: int, restarts: int, reference_workers: int) -> MaxCutInstance:
    if path.exists():
        with numpy.load(path) as data:
            return MaxCutInstance(
                path.stem, int(data["vertices"]), data["heads"], data["tails"],
                data["weights"], int(data["graph_seed"]),
                float(data["reference_cut"]), str(data["reference_kind"]),
            )
    heads, tails, weights = _generate_graph(vertices, probability, seed)
    if vertices <= exact_limit:
        reference = _exact_reference(vertices, heads, tails, weights)
        kind = "exact"
    else:
        reference = _heuristic_reference(
            vertices, heads, tails, weights, seed ^ 0x5DEECE66D,
            restarts, reference_workers,
        )
        kind = "multistart_local_search"
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(
        path, vertices=vertices, heads=heads, tails=tails, weights=weights,
        graph_seed=seed, reference_cut=reference, reference_kind=kind,
    )
    return MaxCutInstance(path.stem, vertices, heads, tails, weights, seed, reference, kind)


def run(config_path: Path, resume: bool) -> None:
    config = _config(config_path)
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "raw_results.csv"
    jsonl_path = output / "raw_results.jsonl"
    if not resume and (csv_path.exists() or jsonl_path.exists()):
        raise FileExistsError(f"results already exist in {output}; pass --resume or choose another output_dir")
    completed = _load_completed(csv_path) if resume else set()
    resolved_devices = _devices(config["devices"])
    manifest = dict(config)
    manifest.update(
        resolved_devices=resolved_devices,
        started_utc=datetime.now(timezone.utc).isoformat(),
        csv_schema=list(CSV_FIELDS),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    available_host_budget = int(
        _available_host_bytes() * config["memory_fraction"]
    )
    configured_host_budget = (
        sys.maxsize
        if config["memory_limit_gb"] is None
        else int(config["memory_limit_gb"] * 1024 ** 3)
    )
    host_budget = min(available_host_budget, configured_host_budget)
    batch_size = config["batch_size"] or config["instances_per_size"]
    shared_parameters = {
        "runs": config["runs"], "steps": config["steps"],
        "seed": config["graph_seed"], "dtype": config["dtype"],
        "run_batch_size": config["run_batch_size"],
    }
    solvers = {
        name: _solver(name, resolved_devices) for name in config["solvers"]
    }
    for name, solver in solvers.items():
        _warm_up(solver, len(resolved_devices), shared_parameters, resolved_devices)
        print(f"warmed up {name} on {len(resolved_devices)} device(s)", flush=True)

    for vertices in config["sizes"]:
        expected_edges = max(1, math.ceil(config["edge_probability"] * vertices * (vertices - 1) / 2))
        active_batch = min(batch_size, config["instances_per_size"])
        host_estimate = max(
            _estimated_bytes(vertices, expected_edges, config["runs"], config["dtype"], name, config["instances_per_size"])
            for name in config["solvers"]
        )
        per_device_problems = math.ceil(active_batch / len(resolved_devices))
        device_estimate = max(
            _estimated_bytes(vertices, expected_edges, config["runs"], config["dtype"], name, per_device_problems)
            for name in config["solvers"]
        )
        device_budget = _device_budget(resolved_devices, config["memory_fraction"])
        if host_estimate > host_budget or device_estimate > device_budget:
            print(
                f"stopping before n={vertices}: host estimate {host_estimate:,} / budget {host_budget:,}; "
                f"per-device estimate {device_estimate:,} / budget {device_budget:,}",
                flush=True,
            )
            break
        instances = []
        for index in range(config["instances_per_size"]):
            seed = config["graph_seed"] + vertices * 1_000_003 + index
            instance_id = f"er_n{vertices:07d}_p{config['edge_probability']:.6f}_i{index:03d}_s{seed}"
            instances.append(
                _instance(
                    output / "instances" / f"{instance_id}.npz", vertices,
                    config["edge_probability"], seed,
                    config["exact_reference_max_vertices"], config["reference_restarts"],
                    config["reference_workers"],
                )
            )
        problems = [_qubo(instance) for instance in instances]
        for solver_name in config["solvers"]:
            solver = solvers[solver_name]
            for trial in range(config["trials"]):
                solver_seed = config["graph_seed"] + trial * 104_729
                parameters = {
                    "runs": config["runs"], "steps": config["steps"],
                    "seed": solver_seed, "dtype": config["dtype"],
                    "run_batch_size": config["run_batch_size"],
                }
                for start in range(0, len(problems), batch_size):
                    stop = min(start + batch_size, len(problems))
                    pending = [
                        index for index in range(start, stop)
                        if (instances[index].instanceId, solver_name, trial, config["runs"], config["steps"]) not in completed
                    ]
                    if not pending:
                        continue
                    batch_problems = [problems[index] for index in pending]
                    total_edges = sum(
                        problems[index].interactionCount for index in pending
                    )
                    estimated = _estimated_bytes(
                        vertices,
                        math.ceil(total_edges / len(pending)),
                        config["runs"],
                        config["dtype"],
                        solver_name,
                        len(pending),
                    )
                    _synchronize(resolved_devices)
                    started = time.perf_counter()
                    results = solver.solveMany(batch_problems, parameters)
                    _synchronize(resolved_devices)
                    elapsed = time.perf_counter() - started
                    now = datetime.now(timezone.utc).isoformat()
                    records = []
                    for index, result in zip(pending, results):
                        instance = instances[index]
                        cut = _cut(instance.heads, instance.tails, instance.weights, result.sample)
                        if not math.isclose(float(result.energy), -cut, rel_tol=1.0e-7, abs_tol=1.0e-7):
                            raise RuntimeError(f"{solver_name} returned inconsistent energy for {instance.instanceId}")
                        records.append({
                            "timestamp_utc": now, "instance_id": instance.instanceId,
                            "vertices": vertices, "edges": len(instance.heads),
                            "density": 2.0 * len(instance.heads) / (vertices * (vertices - 1)),
                            "graph_seed": instance.graphSeed, "solver": solver_name,
                            "trial": trial, "solver_seed": solver_seed,
                            "runs": config["runs"], "steps": config["steps"],
                            "device_count": len(resolved_devices), "batch_size": len(pending),
                            "solve_seconds": elapsed, "amortized_seconds": elapsed / len(pending),
                            "energy": float(result.energy), "cut": cut,
                            "reference_cut": instance.referenceCut,
                            "reference_kind": instance.referenceKind,
                            "success": int(cut + 1.0e-9 >= instance.referenceCut),
                            "estimated_bytes": estimated,
                        })
                    _write_records(csv_path, jsonl_path, records)
                    completed.update(
                        (record["instance_id"], record["solver"], record["trial"], record["runs"], record["steps"])
                        for record in records
                    )
                    print(f"n={vertices} {solver_name} trial={trial} batch={len(pending)} time={elapsed:.3f}s", flush=True)


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

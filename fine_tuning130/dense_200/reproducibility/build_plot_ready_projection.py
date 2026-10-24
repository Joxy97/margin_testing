"""Build the standard compact result layout from downloaded production trials.

Only host result trees whose completion record says ``complete`` are included.
The downloaded per-seed JSON files remain the authoritative production evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve()
BATCH_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
RUNTIME_ROOT = REPO_ROOT / "fine_tuning_publish_20261024"
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from qubo_benchmark.runtime.common import atomic, digest, filehash  # noqa: E402
from qubo_benchmark.runtime.compact_metrics import aggregate_energies  # noqa: E402
from qubo_benchmark.runtime.compact_storage import _row_hash, _write_array, _write_csv  # noqa: E402
from tools.postprocess_continuous_results import export_results  # noqa: E402
from tools.validate_continuous_results import validate  # noqa: E402


REFERENCE = -35154.0
PROBLEM = "gka5e"
CHECKPOINTS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
SEEDS = tuple(range(100))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def collect_completed_hosts(hosts_root: Path):
    trials = {}
    hosts = []
    parameters = {}
    identities = {}
    for host_root in sorted(p for p in hosts_root.iterdir() if p.is_dir()):
        result_root = host_root / "results"
        completion_path = result_root / "completion.json"
        if not completion_path.is_file():
            continue
        completion = load_json(completion_path)
        if completion.get("status") != "complete":
            continue
        if completion.get("failed") != 0 or completion.get("successful") != completion.get("planned"):
            raise ValueError(f"Completed host has failed or missing trials: {host_root.name}")
        host_solvers = set(completion["solvers"])
        if len(host_solvers) != completion["solver_count"]:
            raise ValueError(f"Solver count mismatch on {host_root.name}")
        if set(trials).intersection(host_solvers):
            raise ValueError(f"Overlapping solver assignment on {host_root.name}")
        for solver in sorted(host_solvers):
            seed_files = sorted((result_root / "trials" / solver).glob("*.json"), key=lambda p: int(p.stem))
            if [int(p.stem) for p in seed_files] != list(SEEDS):
                raise ValueError(f"Incomplete seed schedule: {host_root.name}/{solver}")
            records = [load_json(path) for path in seed_files]
            for seed, record in zip(SEEDS, records):
                if record.get("solver") != solver or record.get("seed") != seed:
                    raise ValueError(f"Trial identity mismatch: {host_root.name}/{solver}/{seed}")
                if record.get("status") != "complete":
                    raise ValueError(f"Non-success trial: {host_root.name}/{solver}/{seed}")
                energies = record.get("energies")
                if not isinstance(energies, list) or len(energies) != len(CHECKPOINTS):
                    raise ValueError(f"Checkpoint mismatch: {host_root.name}/{solver}/{seed}")
                finite = [float(x) for x in energies if x is not None]
                if not finite or any(not math.isfinite(x) for x in finite):
                    raise ValueError(f"Missing/nonfinite energy: {host_root.name}/{solver}/{seed}")
                if any(b > a for a, b in zip(finite, finite[1:])):
                    raise ValueError(f"Incumbent worsens: {host_root.name}/{solver}/{seed}")
            first_parameters = records[0]["parameters"]
            if any(record["parameters"] != first_parameters for record in records[1:]):
                raise ValueError(f"Parameters differ between seeds: {host_root.name}/{solver}")
            trials[solver] = records
            parameters[solver] = first_parameters
        identities[host_root.name] = completion["identity"]
        hosts.append(host_root.name)
    if not trials:
        raise ValueError("No completed production hosts were found")
    return hosts, trials, parameters, identities


def build(output: Path) -> dict:
    hosts_root = BATCH_ROOT / "downloads" / "hosts"
    plan_path = next(hosts_root.glob("*/campaign_plan.json"))
    plan = load_json(plan_path)
    hosts, trials, parameters, identities = collect_completed_hosts(hosts_root)
    solvers = sorted(trials)
    expected = set().union(*(set(plan["assignments"][host]) for host in hosts))
    if set(solvers) != expected:
        raise ValueError("Downloaded solver set differs from campaign assignments")
    if len(solvers) != len(plan["solver_ids"]):
        raise ValueError(f"Expected all {len(plan['solver_ids'])} solvers, found {len(solvers)}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output: {output}")

    manifest = {
        "campaign": plan["campaign"],
        "scope": "Latest post-tuning 100-seed production run; tuning-search trials are excluded.",
        "partial_campaign": False,
        "completed_hosts": hosts,
        "planned_host_count": 7,
        "planned_solver_count": len(plan["solver_ids"]),
        "downloaded_solver_count": len(solvers),
        "git_commit": plan["git_commit"],
        "source_sha256": plan["source_sha256"],
        "selected_settings_sha256": plan["selected_settings_sha256"],
        "campaign_plan_sha256": sha256(plan_path),
        "host_production_identities": identities,
        "configuration": plan["protocol_sha256"],
        "inputs": plan["input"],
        "execution": {
            "budget_seconds": plan["budget_seconds"],
            "cpu_threads": plan["cpu_threads"],
            "tf32": plan["tf32"],
            "workers_per_gpu": plan["workers_per_gpu"],
            "gpu_counts": {host: plan["gpu_counts"][host] for host in hosts},
            "stop_on_optimum": True,
            "checkpoint_policy": "last independently scored incumbent at or before each boundary; optimum carried forward",
        },
        "solvers": {solver: parameters[solver] for solver in solvers},
        "problems": [{
            "id": PROBLEM,
            "n": 200,
            "density": "dense",
            "reference": REFERENCE,
            "reference_type": "OPTIMUM",
            "integer_objective": True,
        }],
        "metrics": ["gap_to_solution", "hit_probability", "TTS99"],
        "tts99_convention": "0 at p=1; infinity at p=0",
    }
    manifest.update(
        schema_version=2,
        seeds=list(SEEDS),
        checkpoints_seconds=list(CHECKPOINTS),
        keep_raw_results=True,
        tolerance=1e-9,
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / "manifest.json", manifest)
    identity = digest(manifest)
    aggregates = {}
    for solver in solvers:
        matrix = np.empty((len(SEEDS), len(CHECKPOINTS)), dtype=np.float64)
        statuses = []
        timings = []
        for seed, record in zip(SEEDS, trials[solver]):
            matrix[seed] = [np.nan if value is None else float(value) for value in record["energies"]]
            statuses.append(record["status"])
            timings.append(record.get("time_to_optimum_s"))
        raw_path = output / "raw" / solver / f"{PROBLEM}.npy"
        _write_array(raw_path, matrix)
        aggregate_path = output / "aggregated" / solver / f"{PROBLEM}.csv"
        rows = aggregate_energies(matrix, CHECKPOINTS, REFERENCE, "OPTIMUM", tolerance=0.0)
        _write_csv(aggregate_path, rows)
        aggregate_sha = filehash(aggregate_path)
        aggregates[solver] = {PROBLEM: aggregate_sha}
        group = {
            "status": statuses,
            "row_sha256": [_row_hash(row) for row in matrix],
            "time_to_optimum_s": timings,
            "aggregate_sha256": aggregate_sha,
            "raw_discarded": False,
            "manifest_sha256": identity,
            "solver_id": solver,
            "problem_id": PROBLEM,
        }
        atomic(output / "progress" / solver / f"{PROBLEM}.json", {
            "sha256": digest(group), "payload": group
        })
    state = {
        "schema_version": 2,
        "manifest_sha256": identity,
        "complete": True,
        "planned": len(solvers) * len(SEEDS),
        "terminal": len(solvers) * len(SEEDS),
        "failures": 0,
    }
    atomic(output / "state.json", {"sha256": digest(state), "payload": state})
    completion = {"schema_version": 2, "manifest_sha256": identity, "aggregates": aggregates}
    atomic(output / "completion.json", {"sha256": digest(completion), "payload": completion})
    export_results(output, output / "postprocessed", hit_percent=1.0)

    quality = []
    for solver in solvers:
        matrix = np.load(output / "raw" / solver / f"{PROBLEM}.npy", allow_pickle=False)
        final = matrix[:, -1]
        gap = 100.0 * (final - REFERENCE) / abs(REFERENCE)
        quality.append({
            "solver": solver,
            "problem": PROBLEM,
            "checkpoint_seconds": 20,
            "planned_seeds": len(SEEDS),
            "valid_seeds": int(np.isfinite(final).sum()),
            "exact_hits": int((final == REFERENCE).sum()),
            "hits_within_1pct": int((final <= REFERENCE + 0.01 * abs(REFERENCE)).sum()),
            "best_energy": float(final.min()),
            "mean_energy": float(final.mean()),
            "worst_energy": float(final.max()),
            "best_gap_percent": float(gap.min()),
            "mean_gap_percent": float(gap.mean()),
            "worst_gap_percent": float(gap.max()),
        })
    write_csv(output / "postprocessed" / "final_quality_summary.csv", quality[0].keys(), quality)

    manifest_written = load_json(output / "manifest.json")
    write_json(output / "solver_settings" / "experiment.json", {
        "manifest_sha256": digest(manifest_written), **manifest_written
    })
    write_json(output / "solver_settings" / "solver_parameters.json", {
        "campaign": plan["campaign"],
        "partial_campaign": False,
        "completed_hosts": hosts,
        "solvers": parameters,
    })
    result = validate(output)
    result.update({
        "completed_hosts": len(hosts),
        "solvers": len(solvers),
        "trials": len(solvers) * len(SEEDS),
        "exact_hits_at_20s": sum(row["exact_hits"] for row in quality),
        "within_1pct_at_20s": sum(row["hits_within_1pct"] for row in quality),
        "output": str(output),
    })
    write_json(output / "projection_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BATCH_ROOT / "plot_ready")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

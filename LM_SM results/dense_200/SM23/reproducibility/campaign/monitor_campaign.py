"""Read-only progress and ETA for retained-raw CompactStore campaigns.

Uses only Python's standard library. Per-solver checked progress is authoritative;
the state.json display counters can legitimately lag a just-committed seed.
This tool verifies metadata checksums, not the raw numerical result arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

SUCCESS = frozenset(("complete", "completed"))
TERMINAL = SUCCESS | frozenset(("failed", "unavailable", "no_in_budget_candidate",
    "error", "watchdog_timeout", "invalid_output", "validation_error", "oom"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def checked(path):
    wrapper = json.loads(Path(path).read_bytes())
    payload = wrapper.get("payload")
    if not isinstance(payload, dict) or wrapper.get("sha256") != digest(payload):
        raise ValueError(f"Metadata checksum mismatch: {path}")
    return payload


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe manifest identifier: {value!r}")
    return value


def discover(root):
    root = Path(root).resolve()
    if not root.exists():
        return []
    paths = []
    for count, (directory, folders, files) in enumerate(os.walk(root, followlinks=False)):
        if count > 2000:
            raise ValueError("Progress discovery exceeded 2,000 directories")
        directory = Path(directory)
        if "manifest.json" in files:
            path = directory / "manifest.json"
            value = json.loads(path.read_bytes())
            if value.get("schema_version") == 2 and "seeds" in value and "solvers" in value:
                paths.append(path)
                folders[:] = []
                continue
        if len(directory.relative_to(root).parts) >= 6:
            folders[:] = []
        else:
            folders[:] = sorted(folder for folder in folders if not folder.startswith(".") and
                folder.lower() not in {"raw", "progress", "plots", "aggregated", "repo", "__pycache__"})
    return paths


def load_progress(path):
    path = Path(path)
    manifest = json.loads(path.read_bytes())
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Unsupported compact schema: {path}")
    identity = digest(manifest)
    seeds = manifest["seeds"]
    if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError(f"Invalid immutable seed schedule: {path}")
    solvers = [identifier(solver) for solver in manifest["solvers"]]
    if not solvers:
        raise ValueError("Manifest has no solvers")
    execution = manifest.get("execution", {})
    budget = float(execution.get("budget_seconds", 20.))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Manifest has an invalid per-seed budget")
    gpu_count = len(set(device for device in execution.get("devices", []) if device.startswith("cuda:")))
    state_path = path.parent / "state.json"
    state = checked(state_path) if state_path.exists() else None
    if state and (state.get("manifest_sha256") != identity or state.get("schema_version") != 2):
        raise ValueError("State/manifest identity mismatch")
    groups, attempts = [], set()
    for problem in manifest["problems"]:
        name = identifier(problem["id"])
        rows = []
        for solver in solvers:
            for seed in seeds:
                key = (name, solver, seed)
                if key in attempts:
                    raise ValueError("Duplicate problem in manifest")
                attempts.add(key)
            progress_path = path.parent / "progress" / solver / f"{name}.json"
            if not progress_path.exists():
                if state:
                    raise ValueError(f"Missing committed progress file: {progress_path}")
                statuses = [None] * len(seeds)
            else:
                progress = checked(progress_path)
                if (progress.get("manifest_sha256") != identity or progress.get("solver_id") != solver
                        or progress.get("problem_id") != name):
                    raise ValueError(f"Progress identity mismatch: {progress_path}")
                statuses = progress["status"]
                hashes = progress["row_sha256"]
                if len(statuses) != len(seeds) or len(hashes) != len(seeds):
                    raise ValueError("Progress length differs from immutable seeds")
                for status, checksum in zip(statuses, hashes):
                    if status is None:
                        if checksum is not None:
                            raise ValueError("Pending seed has committed row checksum")
                    elif status not in TERMINAL or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                        raise ValueError("Invalid terminal status or missing committed row checksum")
            completed = sum(status is not None for status in statuses)
            failures = sum(status is not None and status not in SUCCESS for status in statuses)
            rows.append(dict(solver=solver, completed=completed, total=len(seeds), failures=failures,
                             percent=100. * completed / len(seeds)))
        total = len(seeds) * len(solvers)
        completed = sum(row["completed"] for row in rows)
        groups.append(dict(problem=name, n=problem["n"], density=problem["density"],
            manifest=str(path.resolve()), manifest_sha256=identity,
            completed=completed, total=total, failures=sum(row["failures"] for row in rows),
            percent=100. * completed / total, gpu_count=gpu_count,
            budget_seconds=budget, solvers=rows,
            fallback_eta_seconds=budget * (total - completed) / max(1, gpu_count)))
    # State counters are intentionally not used: progress commits precede them.
    return groups, attempts


class Rates:
    def __init__(self):
        self.origins = {}

    def apply(self, key, completed, total, fallback, now):
        origin = self.origins.get(key)
        if origin is None or completed < origin[1]:
            origin = self.origins[key] = (now, completed)
        elapsed, observed = now - origin[0], completed - origin[1]
        if completed == total:
            return 0., "complete"
        if elapsed > 0 and observed > 0:
            return (total - completed) * elapsed / observed, "measured since monitor start"
        return fallback, "budget-based estimate; setup overhead excluded"


def snapshot(roots, rates=None, now=None):
    rates = rates or Rates()
    now = time.monotonic() if now is None else now
    groups, planned = [], set()
    paths = sorted({path.resolve() for root in roots for path in discover(root)})
    for path in paths:
        current, attempts = load_progress(path)
        if planned & attempts:
            raise ValueError("Duplicate solver/problem/seeds across output roots; do not monitor mirrored copies together")
        planned |= attempts
        groups.extend(current)
    for group in groups:
        key = (group["manifest_sha256"], group["problem"])
        eta, policy = rates.apply(key, group["completed"], group["total"],
                                  group["fallback_eta_seconds"], now)
        group.update(eta_seconds=eta, eta_policy=policy)
        for solver in group["solvers"]:
            fallback = group["budget_seconds"] * (solver["total"] - solver["completed"]) / max(1, group["gpu_count"])
            eta, policy = rates.apply(key + (solver["solver"],), solver["completed"],
                                      solver["total"], fallback, now)
            solver.update(eta_seconds=eta, eta_policy=policy)
    completed, total = sum(g["completed"] for g in groups), sum(g["total"] for g in groups)
    # Distinct result roots may run on separate hosts; without a shared host
    # identity, summing group ETAs is the explicit conservative assumption.
    eta, policy = rates.apply("overall", completed, total,
        sum(group["fallback_eta_seconds"] for group in groups), now)
    return dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),
        completed=completed, total=total, failures=sum(g["failures"] for g in groups),
        percent=100. * completed / total if total else 0., eta_seconds=eta if total else None,
        eta_policy=policy, groups=groups,
        validation="checked metadata and immutable plan; raw arrays are not rescored by this monitor")


def duration(seconds):
    if seconds is None:
        return "unknown"
    minutes = math.ceil(max(0., seconds) / 60.)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def display(report):
    print(f"{report['updated_at_utc']}  {report['completed']}/{report['total']} "
          f"({report['percent']:.2f}%)  failures={report['failures']}  ETA~{duration(report['eta_seconds'])}")
    if not report["groups"]:
        print("No compact campaign manifests yet; production may not have started.")
    for group in report["groups"]:
        print(f"  {group['problem']} {group['n']}_{group['density']}: "
              f"{group['completed']}/{group['total']} ({group['percent']:.2f}%) "
              f"failures={group['failures']} ETA~{duration(group['eta_seconds'])}")
        for solver in group["solvers"]:
            print(f"    {solver['solver']:<31} {solver['completed']:>3}/{solver['total']:<3} "
                  f"{solver['percent']:6.2f}%  failures={solver['failures']} "
                  f"remaining work~{duration(solver['eta_seconds'])}")
    print(f"  ETA: {report['eta_policy']}; completion includes recorded failures.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, action="append", required=True,
                        help="Result root; may be repeated for nonoverlapping campaigns")
    parser.add_argument("--watch", action="store_true", help="Poll read-only until complete or Ctrl+C")
    parser.add_argument("--interval", type=float, default=10., help="Watch interval in seconds (default10)")
    parser.add_argument("--json", action="store_true", help="Print one JSON snapshot per poll")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be finite and at least1 second")
    rates = Rates()
    try:
        while True:
            report = snapshot(args.output, rates)
            if args.json:
                print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
            else:
                display(report)
            if not args.watch or (report["total"] and report["completed"] == report["total"]):
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Progress validation failed: {error}\n")


if __name__ == "__main__":
    main()

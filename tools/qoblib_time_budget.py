"""Prepaid size-weighted GPU slots for fixed-wall-time experiments."""
from __future__ import annotations

import math
import random


def plan_phase(entries, phase, seconds, solvers, seed, target_window=20.,
               minimum=5., maximum=300., utilization=.85):
    """Give each solver equal time, then statically balance eight GPU queues."""
    subset = [entry for entry in entries if entry["split"] == phase]
    if not subset:
        raise ValueError(f"Empty {phase} split")
    if not 0. < utilization < 1. or not 0. < minimum <= target_window <= maximum:
        raise ValueError("Require 0 < utilization < 1 and min <= target <= max window")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Phase duration must be finite and positive")
    capacity = 8. * seconds * utilization
    checkpoints = ("half", "final") if phase == "validation" else ("final",)
    base_count = len(subset) * len(solvers) * len(checkpoints)
    desired = (max(1, int(capacity / (base_count * target_window)))
               if phase == "training" else 1 if phase == "validation" else 2)
    affordable = int(capacity / (base_count * minimum))
    if affordable < 1:
        raise ValueError(f"{phase} cannot cover all cases at the minimum window")
    repeats = min(desired, affordable)
    multiplicity = repeats * len(checkpoints)
    target = capacity / len(solvers)
    weights = {e["case_key"]: math.sqrt(max(1, int(e["variables"]) + 2 * int(e["interactions"]))) for e in subset}

    def allocation(scale):
        return {key: min(maximum, max(minimum, scale * weight)) for key, weight in weights.items()}

    low, high = 0., maximum / min(weights.values())
    for _ in range(64):
        middle = (low + high) / 2.
        if sum(allocation(middle).values()) * multiplicity <= target:
            low = middle
        else:
            high = middle
    windows = {key: math.floor(value * 1000.) / 1000. for key, value in allocation(low).items()}
    jobs = [dict(entry=entry, solver=solver, repeat=repeat, checkpoint=checkpoint,
                 budget_s=windows[entry["case_key"]])
            for repeat in range(repeats) for entry in subset for solver in solvers for checkpoint in checkpoints]
    loads, queues = [0.] * 8, [[] for _ in range(8)]
    for job in sorted(jobs, key=lambda j: (-j["budget_s"], j["entry"]["case_key"], j["repeat"], j["checkpoint"], j["solver"])):
        gpu = min(range(8), key=lambda i: (loads[i], i))
        queues[gpu].append(job)
        loads[gpu] += job["budget_s"]
    if max(loads) > seconds - 10.:
        raise ValueError(f"{phase} assignment leaves insufficient cleanup room; reduce utilization/max-window")
    for gpu, queue in enumerate(queues):
        random.Random(f"{seed}/{phase}/{gpu}").shuffle(queue)
    paid = sum(windows.values()) * multiplicity
    summary = dict(phase=phase, wall_seconds=seconds, repeats=repeats, unique_cases=len(subset),
                   planned_jobs=len(jobs), seconds_per_solver={s: paid for s in solvers},
                   assigned_gpu_seconds=loads, reserved_wall_seconds=seconds - max(loads),
                   min_window_s=min(windows.values()), max_window_s=max(windows.values()),
                   size_proxy="sqrt(binary_variables + 2 * quadratic_interactions)", utilization=utilization,
                   slots=[dict(gpu=gpu, case_key=j["entry"]["case_key"], solver=j["solver"], repeat=j["repeat"],
                               checkpoint=j["checkpoint"], budget_s=j["budget_s"])
                          for gpu, queue in enumerate(queues) for j in queue])
    return queues, summary

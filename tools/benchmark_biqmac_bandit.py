#!/usr/bin/env python3
"""Model-free, deadline-scored BiqMac parameter selection on independent GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import time
import traceback

import numpy as np

from benchmark_biqmac import SOLVERS, atomicCsv, atomicJson, parameters, readGraph, stamp


STEPS = {"SBM": 5000, "SVL": 5000, "TRF": 10000}
FIELDS = ["id", "phase", "split", "solver", "method", "checkpoint", "instance",
          "seed", "gpu", "best_cut", "reference_cut", "relative_gap", "best_time_s",
          "search_elapsed_s", "cleanup_overrun_s", "calls", "accepted_calls",
          "failed_calls", "runs", "steps", "budget_s"]


def stableSeed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8],
                          "little") % (2**63 - 1)


def actionBank(seed, count=32):
    """Joint arms span algorithmic controls and bounded execution choices."""
    rng = random.Random(seed)
    bank = {solver: [{}] for solver in SOLVERS}
    log = lambda low, high: math.exp(rng.uniform(math.log(low), math.log(high)))
    for _ in range(count - 1):
        common = {"dtype": rng.choice(["float32", "float32", "float64"]),
                  "run_batch_size": rng.choice([8, 16, 32]),
                  "energy_chunk_size": rng.choice([8192, 65536])}
        bank["SBM"].append(dict(common, dt=log(.05, 1.2), a0=log(.3, 3.),
                                c0=rng.choice([0., log(1e-8, .1)]),
                                gamma=rng.choice([0., log(1e-5, .03)]),
                                initial_scale=log(.005, .3)))
        bank["SVL"].append(dict(common, dt=log(.001, .05), mass=log(.2, 5.),
                                damping=log(.02, 3.),
                                temperature=rng.choice([0., log(1e-5, .3)]),
                                transverse_field_initial=log(.2, 4.),
                                transverse_field_final=rng.choice([0., log(.001, .2)]),
                                problem_scale_initial=rng.choice([0., log(1e-9, .001)]),
                                problem_scale_final=log(1e-7, 2.),
                                integrator=rng.choice(["euler_maruyama", "weak_order_2"]),
                                noise_chunk_size=rng.choice([16, 64, 128])))
        bank["TRF"].append(dict(common, time_step=log(.005, .2), mobility=log(.3, 3.),
                                route_strength=rng.choice([0., log(.05, 5.)]),
                                gamma=rng.choice([0., log(.001, 1.)]),
                                kappa_initial=rng.uniform(-3., -.1),
                                kappa_final=log(.3, 5.), schedule_exponent=log(.3, 3.),
                                integrator=rng.choice(["euler", "heun"]),
                                candidate_interval=rng.choice([10, 25, 100, 250]),
                                candidate_batch_size=rng.choice([32, 128, 256]),
                                matrix_format=rng.choice(["sparse", "dense", "auto"]),
                                sparse_threshold=rng.choice([.05, .15, .4]),
                                cuda_graph=rng.choice([True, True, False]),
                                graph_steps=rng.choice([10, 25, 50])))
    return bank


def groupedSplit(entries, seed):
    """Keep numbered realizations of a graph family/size in the same partition."""
    rng = random.Random(seed)
    result = {}
    for family in sorted({e["family"] for e in entries}):
        groups = {}
        for entry in entries:
            if entry["family"] != family:
                continue
            name = entry["name"]
            group = name.split(".")[0] if family == "rudy" else name.rsplit("_", 1)[0]
            groups.setdefault(group, []).append(entry)
        keys = sorted(groups)
        rng.shuffle(keys)
        ntrain = min(len(keys) - 2, max(1, round(.6 * len(keys))))
        nvalid = min(len(keys) - ntrain - 1, max(1, round(.2 * len(keys))))
        if ntrain < 1 or nvalid < 1:
            raise ValueError(f"Not enough groups for a leakage-free split in {family}")
        for index, group in enumerate(keys):
            split = "train" if index < ntrain else "validation" if index < ntrain + nvalid else "test"
            for entry in groups[group]:
                result[entry["path"]] = {"split": split, "group": f"{family}/{group}"}
    return result


def contextFor(problem, weights):
    # No eigensolver, local search, reference value or landscape prediction.
    sample = weights[::max(1, len(weights) // 1024)][:1024]
    scale = float(np.median(np.abs(sample))) if len(sample) else 0.
    density = 2 * len(weights) / max(1, problem.variableCount * (problem.variableCount - 1))
    return f"{'small' if problem.variableCount <= 150 else 'large'}:{'heavy' if scale > 10 else 'light'}", {
        "vertices": problem.variableCount, "density": density,
        "sampled_median_abs_weight": scale, "sample_count": len(sample)}


def gpuWorker(gpu, inbox, events, inputs, bank, budget):
    """One resident process owns one GPU; the coordinator owns all learner state."""
    try:
        import torch
        from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
        from margin_calculator.optimization.optimization_solver.bqm_solver.bqm_solver_factory import BQMSolverFactory
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(gpu)
        solvers = {name: BQMSolverFactory.create(kind, {"device": f"cuda:{gpu}"})
                   for name, kind in SOLVERS.items()}
        tiny = QUBOProblem(np.array([-1., -1.]), np.array([0], dtype=np.uint32),
                           np.array([1], dtype=np.uint32), np.array([2.]))
        for name, solver in solvers.items():
            solver.solve(tiny, parameters(name, 2, 16, 1))
        torch.cuda.synchronize()
        events.put({"type": "ready", "gpu": gpu, "hardware": torch.cuda.get_device_name(gpu),
                    "torch": torch.__version__, "numpy": np.__version__})
        cache = {}
        while True:
            job = inbox.get()
            if job is None:
                return
            entry = job["entry"]
            if entry["path"] not in cache:
                path = Path(inputs) / entry["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Input digest mismatch: {path}")
                cache[entry["path"]] = readGraph(path)
            problem, heads, tails, weights = cache[entry["path"]]
            torch.cuda.synchronize()
            start = time.perf_counter()
            context, features = contextFor(problem, weights)
            best = 0.
            best_time = time.perf_counter() - start
            best_bits = "0" * problem.variableCount
            calls = accepted = failed = 0
            durations = []
            curve = [{"time_s": best_time, "cut": best}]
            while time.perf_counter() - start < budget:
                elapsed = time.perf_counter() - start
                if durations and budget - elapsed < 1.1 * float(np.median(durations)):
                    break
                state = context + (":first" if calls == 0 else ":restart")
                events.put({"type": "action", "gpu": gpu, "id": job["id"],
                            "context": state, "call": calls, "elapsed_s": elapsed})
                action = inbox.get()
                if time.perf_counter() - start >= budget:
                    break
                supplied = parameters(job["solver"], 32, STEPS[job["solver"]],
                                      stableSeed(job["solver"], entry["path"], job["seed"], calls))
                supplied.update(bank[job["solver"]][action["arm"]])
                before = time.perf_counter()
                error = None
                cut = None
                previous = best
                try:
                    result = solvers[job["solver"]].solve(problem, supplied)
                    torch.cuda.synchronize()
                    bits = np.asarray(result.sample)
                    if bits.shape != (problem.variableCount,) or not np.all((bits == 0) | (bits == 1)):
                        raise ValueError("Solver returned an invalid binary sample")
                    cut = float(weights[bits[heads] != bits[tails]].sum(dtype=np.float64))
                    if not math.isfinite(cut) or not math.isfinite(float(result.energy)):
                        raise ValueError("Nonfinite cut/energy")
                    if not np.isclose(float(result.energy), -cut, rtol=1e-9, atol=1e-5):
                        raise ValueError("Original QUBO energy disagrees with weighted cut")
                    if cut > float(entry["reference_cut"]) + 1e-5:
                        raise ValueError("Cut exceeds the supplied BiqMac reference; investigate input/scoring")
                    # Materialization/validation is part of the deadline, not just the kernel.
                    candidate_bits = "".join(map(str, bits.astype(np.int8).tolist()))
                    finished = time.perf_counter() - start
                    if finished <= budget:
                        accepted += 1
                        if cut > best:
                            best, best_bits, best_time = cut, candidate_bits, finished
                            curve.append({"time_s": best_time, "cut": best})
                except Exception:
                    error = traceback.format_exc()
                    failed += 1
                    torch.cuda.synchronize()
                    finished = time.perf_counter() - start
                duration = time.perf_counter() - before
                durations.append(duration)
                calls += 1
                events.put({"type": "trial", "gpu": gpu, "id": job["id"],
                            "solver": job["solver"], "context": state, "arm": action["arm"],
                            "probability": action["probability"], "call": calls,
                            "parameters": supplied, "cut": cut, "best_cut": best,
                            "gain": best - previous, "duration_s": duration,
                            "finished_s": finished, "on_time": finished <= budget,
                            "error": error})
                if error:
                    torch.cuda.empty_cache()
            elapsed = time.perf_counter() - start
            reference = float(entry["reference_cut"])
            row = {key: job[key] for key in ("id", "phase", "split", "solver", "method", "checkpoint", "seed")}
            row.update(instance=entry["name"], gpu=gpu, best_cut=best, reference_cut=reference,
                       relative_gap=(reference - best) / max(abs(reference), 1.), best_time_s=best_time,
                       search_elapsed_s=min(elapsed, budget), cleanup_overrun_s=max(0., elapsed - budget),
                       calls=calls, accepted_calls=accepted, failed_calls=failed,
                       runs=32, steps=STEPS[job["solver"]], budget_s=budget)
            events.put({"type": "result", "gpu": gpu, "row": row,
                        "best_bits": best_bits, "curve": curve, "features": features})
    except BaseException:
        events.put({"type": "fatal", "gpu": gpu, "error": traceback.format_exc()})


class Experiment:
    def __init__(self, args, entries, split, bank):
        self.args, self.entries, self.split, self.bank = args, entries, split, bank
        self.output = Path(args.output)
        self.models = {}
        self.snapshots = {}
        self.rows = []
        self.started = time.time()
        self.rng = np.random.default_rng(args.seed)
        self.ctx = mp.get_context("spawn")
        self.events = self.ctx.Queue()
        self.inboxes = [self.ctx.Queue() for _ in range(args.gpus)]
        self.processes = []
        self.ready = {}
        self.journal = (self.output / "trials.jsonl").open("a", buffering=1)

    def start(self):
        for gpu, inbox in enumerate(self.inboxes):
            process = self.ctx.Process(target=gpuWorker,
                                       args=(gpu, inbox, self.events, self.args.inputs,
                                             self.bank, self.args.budget), name=f"biqmac-gpu-{gpu}")
            process.start()
            self.processes.append(process)

    def close(self):
        for inbox in self.inboxes:
            inbox.put(None)
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        self.journal.close()

    def job(self, phase, entry, solver, method, seed, checkpoint=0):
        identifier = f"{phase}_{solver}_{method}_{checkpoint}_{entry['name']}_{seed}"
        return {"id": identifier, "phase": phase, "entry": entry, "solver": solver,
                "method": method, "seed": seed, "checkpoint": checkpoint,
                "split": self.split[entry["path"]]["split"]}

    def status(self, phase, active, completed, total):
        atomicJson(self.output / "status.json", {
            "updated": stamp(), "phase": phase, "phase_completed": completed,
            "phase_total": total, "total_completed": len(self.rows),
            "wall_seconds": time.time() - self.started, "active": active,
            "devices": self.ready, "pid": os.getpid()})

    def runPhase(self, phase, jobs, frozen=None):
        pending = iter(jobs)
        active = {}
        completed = 0

        def dispatch(gpu):
            job = next(pending, None)
            if job is not None:
                active[gpu] = dict(job, last_event=stamp())
                self.inboxes[gpu].put(job)
                print(f"{stamp()} START gpu={gpu} {job['id']}", flush=True)

        for gpu in self.ready:
            dispatch(gpu)
        self.status(phase, active, completed, len(jobs))
        last_status = time.monotonic()
        while completed < len(jobs):
            try:
                event = self.events.get(timeout=2)
            except queue.Empty:
                if any(not process.is_alive() for process in self.processes):
                    raise RuntimeError("GPU worker exited unexpectedly; see progress.log")
                self.status(phase, active, completed, len(jobs))
                continue
            gpu = event["gpu"]
            if event["type"] == "fatal":
                raise RuntimeError(event["error"])
            if event["type"] == "ready":
                self.ready[gpu] = {k: v for k, v in event.items() if k not in ("type", "gpu")}
                print(f"{stamp()} READY gpu={gpu} {self.ready[gpu]}", flush=True)
                dispatch(gpu)
                continue
            job = active[gpu]
            job["last_event"] = stamp()
            key = job["solver"] + "/" + event.get("context", "")
            if event["type"] == "action":
                count = len(self.bank[job["solver"]])
                if job["method"] == "default":
                    arm, probability = 0, 1.
                elif job["method"] == "random":
                    rng = np.random.default_rng(stableSeed(job["id"], event["call"], self.args.seed))
                    arm, probability = int(rng.integers(count)), 1. / count
                else:
                    model = self.models if job["phase"] == "training" else frozen
                    log_weights = np.array(model.get(key, [0.] * count), dtype=np.float64)
                    probabilities = np.exp(log_weights - np.max(log_weights))
                    probabilities /= probabilities.sum()
                    rng = self.rng if job["phase"] == "training" else np.random.default_rng(
                        stableSeed(job["id"], event["call"], self.args.seed))
                    arm = int(rng.choice(count, p=probabilities))
                    probability = float(probabilities[arm])
                self.inboxes[gpu].put({"arm": arm, "probability": probability})
                job.update(call=event["call"], arm=arm, context=event["context"], elapsed_s=event["elapsed_s"])
            elif event["type"] == "trial":
                reward = min(1., max(0., event["gain"] / max(1., abs(float(job["entry"]["reference_cut"])))))
                if job["phase"] == "training":
                    values = self.models.setdefault(key, [0.] * len(self.bank[job["solver"]]))
                    values[event["arm"]] -= .05 * (1. - reward) / (event["probability"] + .025)
                    maximum = max(values)
                    self.models[key] = [v - maximum for v in values]
                event.update(phase=job["phase"], method=job["method"], reward=reward,
                             timestamp=stamp(), instance=job["entry"]["name"])
                self.journal.write(json.dumps(event, allow_nan=False) + "\n")
                print(f"{stamp()} CALL gpu={gpu} {job['id']} arm={event['arm']} "
                      f"call={event['call']} seconds={event['duration_s']:.3f} "
                      f"best={event['best_cut']:.6g} on_time={event['on_time']} "
                      f"error={bool(event['error'])}", flush=True)
                if event["error"]:
                    print(event["error"], flush=True)
            elif event["type"] == "result":
                row = event["row"]
                atomicJson(self.output / "episodes" / (row["id"] + ".json"), event)
                self.rows.append(row)
                completed += 1
                atomicCsv(self.output / f"{row['phase']}.csv",
                          [r for r in self.rows if r["phase"] == row["phase"]], FIELDS)
                print(f"{stamp()} END {phase} {completed}/{len(jobs)} gpu={gpu} "
                      f"{row['id']} cut={row['best_cut']:.6g} gap={row['relative_gap']:.6%} "
                      f"calls={row['accepted_calls']}/{row['calls']} "
                      f"overrun={row['cleanup_overrun_s']:.3f}s", flush=True)
                del active[gpu]
                dispatch(gpu)
            if time.monotonic() - last_status >= 1 or event["type"] == "result":
                self.status(phase, active, completed, len(jobs))
                last_status = time.monotonic()

    def run(self):
        training = [e for e in self.entries if self.split[e["path"]]["split"] == "train"]
        validation = [e for e in self.entries if self.split[e["path"]]["split"] == "validation"]
        schedule = {}
        for solver in SOLVERS:
            rng = random.Random(stableSeed(self.args.seed, solver))
            schedule[solver] = []
            while len(schedule[solver]) < 200:
                cycle = list(training)
                rng.shuffle(cycle)
                schedule[solver].extend(cycle)
        previous = 0
        for checkpoint in (50, 100, 200):
            jobs = [self.job("training", schedule[solver][i], solver, "learned", i, checkpoint)
                    for i in range(previous, checkpoint) for solver in SOLVERS]
            self.runPhase(f"training_to_{checkpoint}", jobs)
            self.snapshots[checkpoint] = json.loads(json.dumps(self.models))
            atomicJson(self.output / f"policy_{checkpoint}.json", {
                "log_weights": self.models, "rng_state": self.rng.bit_generator.state,
                "episodes_per_solver": checkpoint, "eta": .05, "implicit_exploration": .025})
            previous = checkpoint
        for checkpoint in (50, 100, 200):
            jobs = [self.job("validation", entry, solver, "learned", 10001, checkpoint)
                    for entry in validation for solver in SOLVERS]
            self.runPhase(f"validation_{checkpoint}", jobs, self.snapshots[checkpoint])
        selected = {}
        frozen = {}
        for solver in SOLVERS:
            scores = {checkpoint: float(np.mean([r["relative_gap"] for r in self.rows
                      if r["phase"] == "validation" and r["solver"] == solver
                      and r["checkpoint"] == checkpoint])) for checkpoint in (50, 100, 200)}
            chosen = min(scores, key=scores.get)
            selected[solver] = {"checkpoint": chosen, "validation_mean_gaps": scores}
            frozen.update({k: v for k, v in self.snapshots[chosen].items() if k.startswith(solver + "/")})
        atomicJson(self.output / "selected_policy.json", {"selection": selected, "log_weights": frozen})
        jobs = [self.job("evaluation", entry, solver, method, seed,
                         selected[solver]["checkpoint"] if method == "learned" else 0)
                for entry in self.entries for seed in (20001, 20002, 20003)
                for solver in SOLVERS for method in ("learned", "default", "random")]
        random.Random(self.args.seed + 1).shuffle(jobs)
        self.runPhase("evaluation", jobs, frozen)
        summary = []
        evaluated = [r for r in self.rows if r["phase"] == "evaluation"]
        for split in ("train", "validation", "test"):
            for solver in SOLVERS:
                for method in ("learned", "default", "random"):
                    rows = [r for r in evaluated if r["split"] == split and
                            r["solver"] == solver and r["method"] == method]
                    wins = sum(r["best_cut"] >= max(other["best_cut"] for other in evaluated
                               if other["instance"] == r["instance"] and other["seed"] == r["seed"]
                               and other["solver"] == solver) - 1e-8 for r in rows)
                    summary.append({"split": split, "solver": solver, "method": method,
                                    "episodes": len(rows), "quality_wins_including_ties": wins,
                                    "mean_relative_gap": float(np.mean([r["relative_gap"] for r in rows])),
                                    "mean_best_time_s": float(np.mean([r["best_time_s"] for r in rows])),
                                    "zero_accepted_episodes": sum(r["accepted_calls"] == 0 for r in rows),
                                    "late_episodes": sum(r["cleanup_overrun_s"] > 0 for r in rows)})
        atomicCsv(self.output / "summary.csv", summary, list(summary[0]))
        self.status("complete", {}, len(jobs), len(jobs))
        print(f"{stamp()} COMPLETE wall_hours={(time.time() - self.started) / 3600:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--budget", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.gpus < 1 or args.budget <= 0 or not math.isfinite(args.budget):
        parser.error("gpus and budget must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists():
        raise RuntimeError("Output already contains an experiment; use a new directory (no automatic resume)")
    manifest = json.loads((Path(args.inputs) / "manifest.json").read_text())
    entries = manifest["instances"]
    split = groupedSplit(entries, args.seed)
    bank = actionBank(args.seed)
    counts = {part: sum(s["split"] == part for s in split.values()) for part in ("train", "validation", "test")}
    windows = 600 + counts["validation"] * 9 + len(entries) * 27
    atomicJson(output / "experiment.json", {
        "created": stamp(), "arguments": vars(args), "steps": STEPS, "runs": 32,
        "split_counts": counts, "split": split, "input_manifest": manifest,
        "action_bank": bank, "ideal_budget_hours": windows * args.budget / args.gpus / 3600,
        "algorithm": "context-bucket EXP3-IX adaptation, eta=.05, gamma=.025",
        "reward": "on-time incumbent gain / max(1, abs(reference_cut)); train only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": "resident parsed QUBO; feature extraction through validated host result; late calls excluded",
        "reference": "https://proceedings.neurips.cc/paper/2015/file/e5a4d6bf330f23a8707bb0d6001dfbe8-Paper.pdf"})
    print(f"{stamp()} START instances={len(entries)} split={counts} windows={windows} "
          f"ideal_hours={windows * args.budget / args.gpus / 3600:.2f} gpus={args.gpus}", flush=True)
    experiment = Experiment(args, entries, split, bank)
    try:
        experiment.start()
        experiment.run()
    except BaseException:
        atomicJson(output / "failure.json", {"timestamp": stamp(), "error": traceback.format_exc()})
        raise
    finally:
        experiment.close()


if __name__ == "__main__":
    main()

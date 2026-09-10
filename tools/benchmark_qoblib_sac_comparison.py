#!/usr/bin/env python3
"""Sequential six-hour discrete SAC / Hybrid SAC QOBLIB comparison.

This is a research runner, not a production solver adapter. Each GPU has an
isolated process/pipe; the CPU coordinator owns all three learners and replay.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import re
import time
import traceback

import numpy as np
import torch

from benchmark_biqmac import SOLVERS, atomicCsv, atomicJson, stamp
from benchmark_biqmac_bandit import actionBank, stableSeed
from qoblib_hybrid_sac import DiscreteSAC, HybridSAC, RANGES, branches, decode, frozen_actor, frozen_choice
from qoblib_time_budget import plan_phase


FIELDS = ["method", "phase", "solver", "case_key", "family", "repeat", "checkpoint",
          "gpu", "status", "calls", "accepted_calls", "quality", "energy", "feasible",
          "residual_squared", "objective_estimate", "best_time_s", "wall_s", "runs", "steps", "budget_s"]


def save_torch(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def group_key(entry):
    family, name = entry["case_key"].split("/", 1)
    if family == "01-marketsplit":
        group = name.rsplit("_", 1)[0]
    elif family == "02-labs":
        group = "length_bucket_" + str(int(re.search(r"\d+", name)[0]) // 10)
    elif family == "03-birkhoff":
        group = "size_" + name.split("-")[1]
    elif family == "04-steiner":
        group = re.search(r"s\d+", name)[0] + "_" + name.rsplit("_rs", 1)[-1]
    elif family == "05-sports":
        match = re.match(r"Addition_\d+", name)
        group = match[0] if match else re.sub(r"_(Tiny|Small|Medium|Large|NoSoft)$", "", name)
    elif family == "06-portfolio":
        group = re.search(r"a\d+_t\d+", name)[0]
    elif family == "07-independentset":
        group = re.sub(r"(_\d+|\.clq)$", "", name) if name.startswith(("R_", "frb")) else name
    elif family == "10-topology":
        group = name.rsplit("_", 1)[0]
    else:
        # Routing's numbered files are separate source cases, not repetitions
        # of an identical graph. Keep their full source identity.
        group = name
    return family + "/" + group


def inventory(root, seed):
    with (root / "inventory.csv").open(newline="") as handle:
        candidates = [r for r in csv.DictReader(handle) if r["status"] == "eligible_estimated"]
    unique = {}
    for row in sorted(candidates, key=lambda r: (max(json.loads(r["memory_estimates"]).values()), r["id"])):
        unique.setdefault(row["case_key"], row)
    entries = list(unique.values())
    for entry in entries:
        entry["group"] = group_key(entry)
    for family in sorted({e["family"] for e in entries}):
        groups = sorted({e["group"] for e in entries if e["family"] == family})
        random.Random(stableSeed(seed, family)).shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f"Cannot create three grouped partitions for {family}")
        train = min(len(groups) - 2, max(1, round(.6 * len(groups))))
        valid = min(len(groups) - train - 1, max(1, round(.2 * len(groups))))
        mapping = {g: "training" if i < train else "validation" if i < train + valid else "evaluation"
                   for i, g in enumerate(groups)}
        for entry in entries:
            if entry["family"] == family:
                entry["split"] = mapping[entry["group"]]
    return sorted(entries, key=lambda e: e["case_key"])


def supplied_parameters(solver, chosen, seed):
    # Bound hardware choices to the preparation admission envelope. These are
    # deliberately not actions; neither learner can buy more VRAM or runs.
    result = dict(chosen)
    result.update(steps=10000, runs=32, seed=seed, dtype="float64",
                  run_batch_size=8, energy_chunk_size=8192)
    if solver == "SVL":
        result["noise_chunk_size"] = 16
    if solver == "TRF":
        result.update(matrix_format="sparse", candidate_batch_size=128,
                      max_variables=2**31 - 1, cuda_graph=True, graph_steps=25)
    return result


class LoadedQubo:
    def __init__(self, root, entry):
        from scipy import sparse
        from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
        path = root / entry["qubo"]
        with np.load(path, allow_pickle=False) as arrays:
            self.problem = QUBOProblem(arrays["linear"], arrays["heads"], arrays["tails"],
                                       arrays["biases"], float(arrays["offset"]))
        self.meta = json.loads((path.parent / "result.json").read_text())
        self.constraint = None
        if not self.meta["native_qs"]:
            self.constraint = sparse.load_npz(path.parent / "residual_matrix.npz")
            self.target = np.load(path.parent / "residual_target.npy", allow_pickle=False)
        p = self.problem
        self.scale = max(1., float(np.abs(p.linear).sum() + np.abs(p.quadraticBiases).sum()))
        self.features = [math.log1p(p.variableCount) / math.log1p(3e6),
                         2. * p.interactionCount / max(1, p.variableCount * (p.variableCount - 1))]
        sample = p.quadraticBiases[::max(1, p.interactionCount // 1024)][:1024]
        self.features += [float(np.mean(sample < 0)) if len(sample) else 0.,
                          float(self.constraint is not None)]

    def score(self, bits):
        p = self.problem
        if bits.shape != (p.variableCount,) or not np.all((bits == 0) | (bits == 1)):
            raise ValueError("Nonbinary or incorrectly shaped sample")
        energy = p.offset + float(p.linear @ bits)
        for start in range(0, p.interactionCount, 8192):
            stop = start + 8192
            energy += float(p.quadraticBiases[start:stop] @ (
                bits[p.quadraticHeads[start:stop]] * bits[p.quadraticTails[start:stop]]))
        if not math.isfinite(energy):
            raise FloatingPointError("Nonfinite original-QUBO energy")
        residual_squared = None
        feasible = None
        objective = None
        if self.constraint is None:
            quality = -math.tanh((energy - p.offset) / self.scale)
        else:
            residual = self.constraint @ bits - self.target
            residual_squared = float(residual @ residual)
            # Integer residuals: no epsilon-based relaxation of feasibility.
            feasible = bool(np.all(residual == 0.))
            if feasible:
                objective = energy * self.meta["objective_sense"]
                quality = .5 - .5 * math.tanh(energy / max(1., self.meta["objective_range_bound"]))
            else:
                violation = residual_squared / max(1, len(residual))
                quality = -.5 * violation / (1. + violation)
        return dict(quality=quality, energy=energy, feasible=feasible,
                    residual_squared=residual_squared, objective_estimate=objective)

    def state(self, start, deadline, call, quality, duration):
        budget = deadline - start
        return np.array(self.features + [min(1., max(0., (time.monotonic() - start) / budget)),
                        float(quality), min(1., math.log1p(call) / math.log1p(64)),
                        min(1., duration / budget)], dtype=np.float32)


def gpu_worker(gpu, connection, inputs):
    try:
        from margin_calculator.optimization.optimization_solver.bqm_solver.bqm_solver_factory import BQMSolverFactory
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(gpu)
        torch.cuda.init()
        solvers = {name: BQMSolverFactory.create(kind, {"device": f"cuda:{gpu}"}) for name, kind in SOLVERS.items()}
        connection.send(dict(type="ready", hardware=torch.cuda.get_device_name(gpu)))
        while True:
            job = connection.recv()
            if job is None:
                return
            start, deadline = job["start"], job["deadline"]
            loaded = LoadedQubo(Path(inputs), job["entry"])
            best = loaded.score(np.zeros(loaded.problem.variableCount, dtype=np.uint8))
            best.update(best_time_s=time.monotonic() - start, calls=0, accepted_calls=0)
            connection.send(dict(type="initial", best=best))
            call, duration = 0, 0.
            while time.monotonic() < deadline:
                if duration and deadline - time.monotonic() < 1.1 * duration:
                    break
                state = loaded.state(start, deadline, call, best["quality"], duration)
                connection.send(dict(type="action", state=state, call=call))
                reply = connection.recv()
                before = time.monotonic()
                if before >= deadline:
                    break
                parameters = supplied_parameters(job["solver"], reply["parameters"],
                    stableSeed(job["solver"], job["entry"]["case_key"], job["repeat"], call))
                old_quality = best["quality"]
                error = None
                accepted = False
                try:
                    solver = solvers[job["solver"]]
                    if solver.estimatedWorkingMemoryBytes(loaded.problem, parameters) > int(11 * 1024**3 * .8):
                        raise MemoryError("Solver working-memory estimate exceeds admission limit")
                    result = solver.solve(loaded.problem, parameters)
                    torch.cuda.synchronize(gpu)
                    bits = np.asarray(result.sample, dtype=np.float64)
                    candidate = loaded.score(bits)
                    if not np.isclose(candidate["energy"], float(result.energy), rtol=1e-9, atol=1e-5):
                        raise ValueError("Independent scoring disagrees with returned QUBO energy")
                    finished = time.monotonic()
                    accepted = finished < deadline
                    if accepted and candidate["quality"] > best["quality"]:
                        temporary = Path(job["assignment"]).with_suffix(".tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, packed_bits=np.packbits(bits.astype(np.uint8)),
                                                variables=loaded.problem.variableCount)
                        # Assignment persistence is part of the scoring window.
                        if time.monotonic() < deadline:
                            temporary.replace(job["assignment"])
                            best.update(candidate, best_time_s=time.monotonic() - start)
                except Exception:
                    error = traceback.format_exc()
                    torch.cuda.empty_cache()
                call += 1
                duration = time.monotonic() - before
                best["calls"] = call
                best["accepted_calls"] += int(accepted)
                following = loaded.state(start, deadline, call, best["quality"], duration)
                terminal = bool(error) or deadline - time.monotonic() < 1.1 * duration
                connection.send(dict(type="trial", state=state, next_state=following,
                    reward=best["quality"] - old_quality, terminal=terminal, best=dict(best),
                    duration_s=duration, parameters=parameters, error=error,
                    finished=time.monotonic(), on_time=time.monotonic() <= deadline))
                if terminal:
                    break
            connection.send(dict(type="done", best=best))
            del loaded
            torch.cuda.empty_cache()
    except (EOFError, BrokenPipeError):
        return
    except BaseException:
        try:
            connection.send(dict(type="fatal", error=traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass


class Comparison:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "manifest.json").exists():
            raise ValueError("Output already has a manifest; choose a new directory")
        self.entries = inventory(Path(args.inputs), args.seed)
        self.plans, self.plan_summaries = {}, {}
        for phase, fraction in (("training", 4/6), ("validation", 1/6), ("evaluation", 1/6)):
            self.plans[phase], self.plan_summaries[phase] = plan_phase(
                self.entries, phase, args.hours * 3600. * fraction, tuple(SOLVERS), args.seed,
                target_window=args.target_window, minimum=args.min_window,
                maximum=args.max_window, utilization=args.utilization)
        self.bank = actionBank(args.seed, 32)
        # Baseline retains its finite configuration bank, with the same memory
        # controls as Hybrid SAC. It cannot select an unsafe dense matrix.
        self.context = mp.get_context("spawn")
        self.rows, self.workers, self.snapshots = [], {}, {}
        self.journal = (self.root / "trials.jsonl").open("a", buffering=1)
        self.start = time.monotonic()
        self.origin = time.time()
        atomicJson(self.root / "manifest.json", dict(created=stamp(), inputs=args.inputs,
            seed=args.seed, eligible_cases=len(self.entries), split=self.entries,
            methods=["sac", "hybrid_sac"], baseline="discrete SAC, 32 fixed configurations",
            hours_per_method=args.hours, phases={"training": 4/6, "validation": 1/6, "evaluation": 1/6},
            runs=32, steps=10000, window_policy="prepaid size-weighted slots",
            min_window_seconds=args.min_window, max_window_seconds=args.max_window,
            target_window_seconds=args.target_window, utilization=args.utilization, gpus=8, dtype="float64",
            run_batch_size=8, memory_fraction=.8, bank=self.bank,
            hybrid_ranges=RANGES, hybrid_branches={s: branches(s) for s in SOLVERS},
            split_policy="source-name grouped; parameter variants grouped; not graph-isomorphism detection",
            quality_policy="feasible-first for converted LP; bounded QUBO energy for native QS",
            setup_inside_budget=True, external_test_validation_run=False))
        atomicCsv(self.root / "split.csv", self.entries,
                  sorted({key for entry in self.entries for key in entry}))
        atomicJson(self.root / "time_budget.json", self.plan_summaries)
        for phase, summary in self.plan_summaries.items():
            print(f"{stamp()} PLAN {phase} jobs={summary['planned_jobs']} "
                  f"window={summary['min_window_s']:.3f}..{summary['max_window_s']:.3f}s "
                  f"solver_gpu_seconds={summary['seconds_per_solver']} "
                  f"reserved_wall={summary['reserved_wall_seconds']:.1f}s", flush=True)

    def boot(self, gpu):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=gpu_worker, args=(gpu, child, self.args.inputs))
        process.start()
        child.close()
        self.workers[gpu] = dict(process=process, pipe=parent, ready=False, active=None,
                                 boot=time.monotonic())

    def stop(self, gpu):
        worker = self.workers.pop(gpu)
        worker["pipe"].close()
        if worker["process"].is_alive():
            worker["process"].terminate()
        worker["process"].join(timeout=.2)
        if worker["process"].is_alive():
            worker["process"].kill()
            worker["process"].join(timeout=.2)

    def checkpoint(self, name):
        self.snapshots[name] = {s: frozen_actor(m) for s, m in self.models.items()}
        save_torch(self.directory / (name + ".pt"),
                   {s: m.checkpoint() for s, m in self.models.items()})
        print(f"{stamp()} CHECKPOINT {self.method}/{name}", flush=True)

    def schedule(self, phase):
        return {gpu: collections.deque(dict(job) for job in jobs)
                for gpu, jobs in enumerate(self.plans[phase])}

    def finish(self, gpu, status):
        worker = self.workers[gpu]
        job = worker["active"]
        if job is None:
            return
        best = job.get("best", {})
        if self.phase == "training" and job.get("pending") is not None:
            state, action = job["pending"]
            following = state.copy()
            following[4] = 1.
            self.models[job["solver"]].remember(state, action, 0., following, True)
        row = dict(method=self.method, phase=self.phase, solver=job["solver"],
                   case_key=job["entry"]["case_key"], family=job["entry"]["family"],
                   repeat=job["repeat"], checkpoint=job["checkpoint"], gpu=gpu, status=status,
                   wall_s=time.monotonic() - job["start"], runs=32, steps=10000,
                   budget_s=job["budget_s"])
        row.update(best)
        self.rows.append(row)
        atomicCsv(self.directory / (self.phase + ".csv"),
                  [r for r in self.rows if r["method"] == self.method and r["phase"] == self.phase], FIELDS)
        print(f"{stamp()} END {self.method}/{self.phase} gpu={gpu} {job['solver']} "
              f"{row['case_key']} status={status} quality={row.get('quality')} "
              f"feasible={row.get('feasible')} calls={row.get('calls', 0)}", flush=True)
        worker["active"] = None

    def run_phase(self, phase, end, half=None):
        self.phase = phase
        for gpu in range(8):
            self.boot(gpu)
        jobs = self.schedule(phase)
        last_status = 0.
        try:
            while time.monotonic() < end - 5.:
                now = time.monotonic()
                if half is not None and now >= half and "half" not in self.snapshots:
                    self.checkpoint("half")
                for gpu, worker in list(self.workers.items()):
                    job = worker["active"]
                    if job and now >= job["deadline"]:
                        status = "complete" if job.get("done") else "timeout"
                        self.finish(gpu, status)
                        if status == "timeout":
                            self.stop(gpu)
                            self.boot(gpu)
                        continue
                    if not worker["process"].is_alive() or (not worker["ready"] and now - worker["boot"] > 90):
                        if job:
                            self.finish(gpu, "worker_failure")
                        self.stop(gpu)
                        raise RuntimeError(f"GPU {gpu} failed; stop instead of silently reducing GPU count")
                    if worker["ready"] and job is None and jobs[gpu] and now + jobs[gpu][0]["budget_s"] < end - 5.:
                        job = jobs[gpu].popleft()
                        if phase == "evaluation":
                            job["checkpoint"] = self.selected[job["solver"]]
                        identifier = hashlib.sha256(json.dumps([self.method, phase, job["solver"],
                            job["entry"]["case_key"], job["repeat"], job["checkpoint"]]).encode()).hexdigest()[:24]
                        job.update(start=time.monotonic(), assignment=str(self.directory / "assignments" / (identifier + ".npz")))
                        job["deadline"] = job["start"] + job["budget_s"]
                        worker["active"] = job
                        worker["pipe"].send(job)
                        print(f"{stamp()} START {self.method}/{phase} gpu={gpu} {job['solver']} "
                              f"{job['entry']['case_key']} repeat={job['repeat']} checkpoint={job['checkpoint']} "
                              f"budget={job['budget_s']:.3f}s", flush=True)
                connections = {w["pipe"]: gpu for gpu, w in self.workers.items()}
                for connection in wait(list(connections), timeout=.05):
                    gpu = connections[connection]
                    worker = self.workers[gpu]
                    try:
                        event = connection.recv()
                    except EOFError as error:
                        raise RuntimeError(f"GPU {gpu} disconnected") from error
                    kind = event["type"]
                    if kind == "ready":
                        worker["ready"] = True
                        print(f"{stamp()} READY gpu={gpu} {event['hardware']}", flush=True)
                        continue
                    if kind == "fatal":
                        raise RuntimeError(event["error"])
                    job = worker["active"]
                    if job is None:
                        raise RuntimeError("Unexpected worker event without an active job")
                    if kind == "action":
                        state = event["state"]
                        if phase == "training":
                            action = self.models[job["solver"]].choose(state)
                            if self.method == "sac":
                                action = action[0]
                        else:
                            action = frozen_choice(self.snapshots[job["checkpoint"]][job["solver"]],
                                                   state, self.method == "hybrid_sac")
                        parameters = (decode(job["solver"], action) if self.method == "hybrid_sac"
                                      else self.bank[job["solver"]][action])
                        job["pending"] = state, action
                        connection.send(dict(parameters=parameters))
                    elif kind == "trial":
                        pending = job.pop("pending", None)
                        if pending is None:
                            raise RuntimeError("Trial has no associated policy action")
                        if phase == "training":
                            self.models[job["solver"]].remember(event["state"], pending[1], event["reward"],
                                                               event["next_state"], event["terminal"])
                        job["best"] = event["best"]
                        record = dict(event, method=self.method, phase=phase, solver=job["solver"],
                                      case_key=job["entry"]["case_key"], gpu=gpu,
                                      action=pending[1], timestamp=stamp())
                        for key in ("state", "next_state"):
                            record[key] = record[key].tolist()
                        if self.method == "hybrid_sac":
                            record["action"] = [pending[1][0], pending[1][1].tolist()]
                        self.journal.write(json.dumps(record, allow_nan=False) + "\n")
                        print(f"{stamp()} CALL gpu={gpu} {job['solver']} duration={event['duration_s']:.3f}s "
                              f"reward={event['reward']:.6g} error={bool(event['error'])}", flush=True)
                        if event["error"]:
                            print(event["error"], flush=True)
                    elif kind == "initial":
                        job["best"] = event["best"]
                    elif kind == "done":
                        job["best"], job["done"] = event["best"], True
                if now - last_status > 5.:
                    atomicJson(self.root / "status.json", dict(updated=stamp(), method=self.method,
                        phase=phase, elapsed_s=now - self.start,
                        scheduled_finish_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.origin + self.args.hours * 7200)),
                        phase_remaining_s=max(0., end - now),
                        completed=sum(r["status"] != "not_started_budget" for r in self.rows),
                        queued_jobs=sum(len(queue) for queue in jobs.values()),
                        active={g: {"case": w["active"]["entry"]["case_key"], "solver": w["active"]["solver"],
                                    "budget_s": w["active"]["budget_s"]}
                                for g, w in self.workers.items() if w["active"]},
                        learner={s: m.lastMetrics for s, m in self.models.items()}))
                    last_status = now
        finally:
            for gpu in list(self.workers):
                if self.workers[gpu]["active"]:
                    self.finish(gpu, "phase_boundary")
                self.stop(gpu)
            for gpu, queue in jobs.items():
                for job in queue:
                    self.rows.append(dict(method=self.method, phase=phase, solver=job["solver"],
                        case_key=job["entry"]["case_key"], family=job["entry"]["family"],
                        repeat=job["repeat"], checkpoint=job["checkpoint"], gpu=gpu,
                        status="not_started_budget", calls=0, accepted_calls=0, wall_s=0.,
                        runs=32, steps=10000, budget_s=job["budget_s"]))
            atomicCsv(self.directory / (phase + ".csv"),
                      [r for r in self.rows if r["method"] == self.method and r["phase"] == phase], FIELDS)

    def select(self):
        selected, evidence = {}, {}
        for solver in SOLVERS:
            rows = [r for r in self.rows if r["method"] == self.method and
                    r["phase"] == "validation" and r["solver"] == solver]
            paired = collections.defaultdict(dict)
            for row in rows:
                paired[(row["case_key"], row["repeat"])][row["checkpoint"]] = row
            family = {name: collections.defaultdict(list) for name in ("half", "final")}
            for pair in paired.values():
                if set(pair) != {"half", "final"}:
                    continue
                for name, row in pair.items():
                    family[name][row["family"]].append(row.get("quality", -1.))
            means = {name: float(np.mean([np.mean(v) for v in values.values()])) if values else -1.
                     for name, values in family.items()}
            selected[solver] = "half" if means["half"] > means["final"] else "final"
            evidence[solver] = dict(mean_family_quality=means, selected=selected[solver],
                                    paired_cases=sum(len(v) for v in family["final"].values()))
        atomicJson(self.directory / "selected_policy.json", evidence)
        return selected

    def run(self):
        block = self.args.hours * 3600.
        try:
            for index, method in enumerate(("sac", "hybrid_sac")):
                self.method = method
                self.directory = self.root / method
                (self.directory / "assignments").mkdir(parents=True)
                self.snapshots, self.selected = {}, {}
                self.models = {s: (DiscreteSAC(32, stableSeed(self.args.seed, s)) if method == "sac"
                                   else HybridSAC(s, stableSeed(self.args.seed, s))) for s in SOLVERS}
                begin = self.start + index * block
                self.run_phase("training", begin + block * 4/6, half=begin + block * 2/6)
                if "half" not in self.snapshots:
                    self.checkpoint("half")
                self.checkpoint("final")
                self.run_phase("validation", begin + block * 5/6)
                self.selected = self.select()
                self.run_phase("evaluation", begin + block)
                while True:
                    remaining = begin + block - time.monotonic()
                    if remaining <= 0.:
                        break
                    time.sleep(min(.2, remaining))
            atomicJson(self.root / "status.json", dict(phase="complete", updated=stamp(),
                       elapsed_s=time.monotonic() - self.start, episodes=len(self.rows)))
        except BaseException:
            atomicJson(self.root / "status.json", dict(phase="failed", updated=stamp(),
                       error=traceback.format_exc(), elapsed_s=time.monotonic() - self.start))
            raise
        finally:
            self.journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--hours", type=float, default=6.)
    parser.add_argument("--target-window", type=float, default=20., help="Reference window for training repetition count")
    parser.add_argument("--min-window", type=float, default=5.)
    parser.add_argument("--max-window", type=float, default=300.)
    parser.add_argument("--utilization", type=float, default=.85)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (
            args.hours, args.target_window, args.min_window, args.max_window, args.utilization)):
        parser.error("Durations and utilization must be finite and positive")
    if not args.min_window <= args.target_window <= args.max_window or args.utilization >= 1:
        parser.error("Require min-window <= target-window <= max-window and utilization < 1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    Comparison(args).run()


if __name__ == "__main__":
    main()

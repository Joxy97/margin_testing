#!/usr/bin/env python3
"""Discrete SAC configuration learning for deadline-scored BiqMac MaxCut."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import benchmark_biqmac_bandit as bandit
from benchmark_biqmac import SOLVERS, atomicCsv, atomicJson, stamp


FEATURES = 8
HYPERPARAMETERS = {"hidden_width": 64, "learning_rate": .0003, "batch_size": 64,
                   "replay_capacity": 100000, "updates_per_transition": 2,
                   "discount": 1., "entropy_coefficient": .002,
                   "target_tau": .005, "gradient_clip": 10.}


def network(arms):
    return nn.Sequential(nn.Linear(FEATURES, 64), nn.ReLU(), nn.Linear(64, 64),
                         nn.ReLU(), nn.Linear(64, arms))


class DiscreteSAC:
    """CPU actor, twin critics and replay; graph solving stays on dedicated GPUs."""

    def __init__(self, arms, seed):
        torch.manual_seed(seed)
        self.arms = arms
        self.actor = network(arms)
        self.q1, self.q2 = network(arms), network(arms)
        self.target1, self.target2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=.0003)
        self.qParameters = [*self.q1.parameters(), *self.q2.parameters()]
        self.criticOptimizer = torch.optim.Adam(self.qParameters, lr=.0003)
        self.rng = np.random.default_rng(seed)
        self.replay = []
        self.cursor = 0
        self.updates = 0
        self.lastMetrics = {}

    def choose(self, state):
        if len(self.replay) < 64:
            return int(self.rng.integers(self.arms)), 1. / self.arms
        with torch.no_grad():
            probabilities = self.actor(torch.as_tensor(state)).softmax(-1).numpy().astype(np.float64)
        probabilities /= probabilities.sum()
        arm = int(self.rng.choice(self.arms, p=probabilities))
        return arm, float(probabilities[arm])

    def remember(self, state, arm, reward, next_state, terminal):
        item = (state.copy(), arm, reward, next_state.copy(), float(terminal))
        if len(self.replay) < 100000:
            self.replay.append(item)
        else:
            self.replay[self.cursor] = item
        self.cursor = (self.cursor + 1) % 100000
        if len(self.replay) < 64:
            return
        for _ in range(2):
            self.update()

    def update(self):
        batch = [self.replay[int(i)] for i in self.rng.choice(len(self.replay), 64, replace=False)]
        states = torch.as_tensor(np.stack([x[0] for x in batch]))
        actions = torch.tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([x[3] for x in batch]))
        terminal = torch.tensor([x[4] for x in batch], dtype=torch.float32)
        with torch.no_grad():
            next_log = self.actor(next_states).log_softmax(-1)
            target_q = torch.minimum(self.target1(next_states), self.target2(next_states))
            next_value = (next_log.exp() * (target_q - .002 * next_log)).sum(-1)
            # Finite episodes, gamma=1: no additional preference for early gains.
            target = rewards + (1. - terminal) * next_value
        critic_loss = F.smooth_l1_loss(self.q1(states).gather(1, actions).squeeze(1), target)
        critic_loss = critic_loss + F.smooth_l1_loss(self.q2(states).gather(1, actions).squeeze(1), target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite SAC critic loss")
        self.criticOptimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.qParameters, 10.)
        self.criticOptimizer.step()
        log_probabilities = self.actor(states).log_softmax(-1)
        with torch.no_grad():
            q = torch.minimum(self.q1(states), self.q2(states))
        actor_loss = (log_probabilities.exp() * (.002 * log_probabilities - q)).sum(-1).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Nonfinite SAC actor loss")
        self.actorOptimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.actorOptimizer.step()
        with torch.no_grad():
            for target_model, model in ((self.target1, self.q1), (self.target2, self.q2)):
                for target_parameter, parameter in zip(target_model.parameters(), model.parameters()):
                    target_parameter.lerp_(parameter, .005)
        self.updates += 1
        self.lastMetrics = {"critic_loss": float(critic_loss.detach()),
                            "actor_loss": float(actor_loss.detach()),
                            "entropy": float(-(log_probabilities.exp() * log_probabilities).sum(-1).mean().detach()),
                            "updates": self.updates, "replay_size": len(self.replay)}

    def checkpoint(self):
        return {"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "target1": self.target1.state_dict(), "target2": self.target2.state_dict(),
                "actor_optimizer": self.actorOptimizer.state_dict(),
                "critic_optimizer": self.criticOptimizer.state_dict(), "replay": self.replay,
                "cursor": self.cursor, "updates": self.updates, "rng": self.rng.bit_generator.state}


def gpuWorker10000(*args):
    # This process-local adapter leaves the original EXP3 runner and solvers unchanged.
    bandit.STEPS.update({solver: 10000 for solver in SOLVERS})
    bandit.gpuWorker(*args)


def observation(job, event, budget):
    entry = job["entry"]
    vertices = entry["vertices"]
    density = 2. * entry["edges"] / max(1, vertices * (vertices - 1))
    return np.array([
        math.log1p(vertices) / math.log1p(10000), min(1., density),
        float(":heavy" in event["context"]), float(event["call"] == 0),
        min(1., max(0., event["elapsed_s"] / budget)),
        min(1., math.log1p(event["call"]) / math.log1p(64)),
        min(1., job.get("last_duration_s", 0.) / budget),
        float(job.get("last_improved", False))], dtype=np.float32)


def saveTorch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class SACExperiment(bandit.Experiment):
    """Independent orchestration; reuses only the existing execution/data contract."""

    def __init__(self, args, entries, split, bank):
        super().__init__(args, entries, split, bank)
        self.learners = {solver: DiscreteSAC(len(bank[solver]), bandit.stableSeed(args.seed, solver))
                         for solver in SOLVERS}
        self.transitions = {}

    def start(self):
        for gpu, inbox in enumerate(self.inboxes):
            process = self.ctx.Process(target=gpuWorker10000,
                                       args=(gpu, inbox, self.events, self.args.inputs, self.bank, self.args.budget),
                                       name=f"biqmac-sac-gpu-{gpu}")
            process.start()
            self.processes.append(process)

    def finishTransition(self, gpu, job, next_state, terminal):
        transition = self.transitions.pop(gpu, None)
        if transition is None or "reward" not in transition:
            return
        learner = self.learners[job["solver"]]
        before = time.perf_counter()
        learner.remember(transition["state"], transition["arm"], transition["reward"], next_state, terminal)
        self.journal.write(json.dumps({"type": "learning_update", "timestamp": stamp(),
            "id": job["id"], "solver": job["solver"], "gpu": gpu,
            "state": transition["state"].tolist(), "arm": transition["arm"],
            "reward": transition["reward"], "next_state": next_state.tolist(),
            "terminal": terminal, "learner_seconds": time.perf_counter() - before,
            **learner.lastMetrics}, allow_nan=False) + "\n")

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
                if any(not p.is_alive() for p in self.processes):
                    raise RuntimeError("SAC experiment GPU worker exited; see progress.log")
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
            training = job["phase"] == "training"
            if event["type"] == "action":
                state = observation(job, event, self.args.budget)
                count = len(self.bank[job["solver"]])
                if training:
                    self.finishTransition(gpu, job, state, False)
                    arm, probability = self.learners[job["solver"]].choose(state)
                    self.transitions[gpu] = {"state": state, "arm": arm}
                elif job["method"] == "default":
                    arm, probability = 0, 1.
                else:
                    rng = np.random.default_rng(bandit.stableSeed(job["id"], event["call"], self.args.seed))
                    if job["method"] == "random":
                        arm, probability = int(rng.integers(count)), 1. / count
                    else:
                        with torch.no_grad():
                            probabilities = frozen[job["solver"]](torch.as_tensor(state)).softmax(-1).numpy().astype(np.float64)
                        probabilities /= probabilities.sum()
                        arm = int(rng.choice(count, p=probabilities))
                        probability = float(probabilities[arm])
                self.inboxes[gpu].put({"arm": arm, "probability": probability})
                job.update(call=event["call"], arm=arm, context=event["context"], elapsed_s=event["elapsed_s"])
            elif event["type"] == "trial":
                reward = min(1., max(0., event["gain"] / max(1., abs(float(job["entry"]["reference_cut"])))))
                if training:
                    self.transitions[gpu]["reward"] = reward
                job.update(last_duration_s=event["duration_s"], last_improved=event["gain"] > 0,
                           best_cut=event["best_cut"])
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
                if training:
                    self.finishTransition(gpu, job, np.zeros(FEATURES, dtype=np.float32), True)
                row = event["row"]
                atomicJson(self.output / "episodes" / (row["id"] + ".json"), event)
                self.rows.append(row)
                completed += 1
                atomicCsv(self.output / f"{row['phase']}.csv",
                          [r for r in self.rows if r["phase"] == row["phase"]], bandit.FIELDS)
                metrics = self.learners[job["solver"]].lastMetrics
                print(f"{stamp()} END {phase} {completed}/{len(jobs)} gpu={gpu} {row['id']} "
                      f"cut={row['best_cut']:.6g} gap={row['relative_gap']:.6%} "
                      f"calls={row['accepted_calls']}/{row['calls']} overrun={row['cleanup_overrun_s']:.3f}s "
                      f"learner={json.dumps(metrics)}", flush=True)
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
            rng = random.Random(bandit.stableSeed(self.args.seed, solver))
            schedule[solver] = []
            while len(schedule[solver]) < 200:
                cycle = list(training)
                rng.shuffle(cycle)
                schedule[solver].extend(cycle)
        previous = 0
        for checkpoint in (50, 100, 200):
            jobs = [self.job("training", schedule[solver][i], solver, "sac", i, checkpoint)
                    for i in range(previous, checkpoint) for solver in SOLVERS]
            self.runPhase(f"training_to_{checkpoint}", jobs)
            self.snapshots[checkpoint] = {solver: copy.deepcopy(learner.actor).eval()
                                          for solver, learner in self.learners.items()}
            saveTorch(self.output / f"checkpoint_{checkpoint}.pt", {
                "episodes_per_solver": checkpoint, "hyperparameters": HYPERPARAMETERS,
                "solvers": {solver: learner.checkpoint() for solver, learner in self.learners.items()}})
            previous = checkpoint
        for checkpoint in (50, 100, 200):
            jobs = [self.job("validation", entry, solver, "sac", 10001, checkpoint)
                    for entry in validation for solver in SOLVERS]
            self.runPhase(f"validation_{checkpoint}", jobs, self.snapshots[checkpoint])
        selected, frozen = {}, {}
        for solver in SOLVERS:
            scores = {checkpoint: float(np.mean([r["relative_gap"] for r in self.rows
                      if r["phase"] == "validation" and r["solver"] == solver
                      and r["checkpoint"] == checkpoint])) for checkpoint in (50, 100, 200)}
            chosen = min(scores, key=scores.get)
            selected[solver] = {"checkpoint": chosen, "validation_mean_gaps": scores}
            frozen[solver] = self.snapshots[chosen][solver]
        atomicJson(self.output / "selected_policy.json", selected)
        saveTorch(self.output / "selected_actors.pt", {solver: model.state_dict() for solver, model in frozen.items()})
        jobs = [self.job("evaluation", entry, solver, method, seed,
                         selected[solver]["checkpoint"] if method == "sac" else 0)
                for entry in self.entries for seed in (20001, 20002, 20003)
                for solver in SOLVERS for method in ("sac", "default", "random")]
        random.Random(self.args.seed + 1).shuffle(jobs)
        self.runPhase("evaluation", jobs, frozen)
        self.summarize()
        self.status("complete", {}, len(jobs), len(jobs))
        print(f"{stamp()} COMPLETE wall_hours={(time.time() - self.started) / 3600:.2f}", flush=True)

    def summarize(self):
        rows = [r for r in self.rows if r["phase"] == "evaluation"]
        within_solver = {}
        across_solvers = {}
        for row in rows:
            within_key = row["instance"], row["seed"], row["solver"]
            across_key = row["instance"], row["seed"], row["method"]
            within_solver[within_key] = max(within_solver.get(within_key, 0.), row["best_cut"])
            across_solvers[across_key] = max(across_solvers.get(across_key, 0.), row["best_cut"])
        summary = []
        for split in ("train", "validation", "test"):
            for solver in SOLVERS:
                for method in ("sac", "default", "random"):
                    subset = [r for r in rows if r["split"] == split and r["solver"] == solver and r["method"] == method]
                    summary.append({"split": split, "solver": solver, "method": method,
                        "episodes": len(subset),
                        "method_quality_wins_including_ties": sum(r["best_cut"] >= within_solver[r["instance"], r["seed"], solver] - 1e-8 for r in subset),
                        "solver_quality_wins_including_ties": sum(r["best_cut"] >= across_solvers[r["instance"], r["seed"], method] - 1e-8 for r in subset),
                        "mean_relative_gap": float(np.mean([r["relative_gap"] for r in subset])),
                        "median_relative_gap": float(np.median([r["relative_gap"] for r in subset])),
                        "p90_relative_gap": float(np.percentile([r["relative_gap"] for r in subset], 90)),
                        "mean_best_time_s": float(np.mean([r["best_time_s"] for r in subset])),
                        "zero_accepted_episodes": sum(r["accepted_calls"] == 0 for r in subset),
                        "late_episodes": sum(r["cleanup_overrun_s"] > 0 for r in subset),
                        "failed_calls": sum(r["failed_calls"] for r in subset)})
        atomicCsv(self.output / "summary.csv", summary, list(summary[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--budget", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.gpus < 1 or not math.isfinite(args.budget) or args.budget <= 0:
        parser.error("gpus and budget must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists():
        raise RuntimeError("Output already contains an experiment; use a new directory (no automatic resume)")
    manifest = json.loads((Path(args.inputs) / "manifest.json").read_text())
    entries = manifest["instances"]
    split = bandit.groupedSplit(entries, args.seed)
    bank = bandit.actionBank(args.seed)
    counts = {part: sum(s["split"] == part for s in split.values()) for part in ("train", "validation", "test")}
    windows = 600 + counts["validation"] * 9 + len(entries) * 27
    sources = [Path(__file__), Path(bandit.__file__), Path(__file__).with_name("benchmark_biqmac.py")]
    atomicJson(output / "experiment.json", {
        "created": stamp(), "algorithm": "discrete soft actor-critic", "arguments": vars(args),
        "steps": {solver: 10000 for solver in SOLVERS}, "runs": 32,
        "hyperparameters": HYPERPARAMETERS, "split_counts": counts, "split": split,
        "input_manifest": manifest, "action_bank": bank,
        "ideal_budget_hours": windows * args.budget / args.gpus / 3600,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "features": ["log_vertices", "density", "sampled_weight_heavy", "first_call",
                     "elapsed_fraction", "log_call_count", "last_duration_fraction", "last_call_improved"],
        "reward": "on-time incumbent gain / max(1, abs(reference_cut)); training only; no reference policy features",
        "timing": "resident parsed QUBO; feature extraction through validated host result; late calls excluded",
        "references": ["https://proceedings.mlr.press/v80/haarnoja18b.html", "https://arxiv.org/abs/1910.07207"]})
    print(f"{stamp()} START SAC instances={len(entries)} split={counts} windows={windows} "
          f"ideal_hours={windows * args.budget / args.gpus / 3600:.2f} gpus={args.gpus} runs=32 steps=10000", flush=True)
    experiment = SACExperiment(args, entries, split, bank)
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

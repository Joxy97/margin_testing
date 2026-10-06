"""Bounded tuning of one uninterrupted 20-second solve per seed.

Screen three immutable configurations on two paired instance/seed blocks, then
evaluate the preselected winner and baseline on fresh paired blocks: at most 180
methods. This exploratory design is deliberately not a 100-seed campaign.
References enter parent-side metrics only. Trial files contain nine energies,
configuration identity and operational status, never vectors or iteration logs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CHECKPOINTS = (.05, .1, .2, .5, 1., 2., 5., 10., 20.)
PROTECTED = ("lib_simulated_annealing", "lib_simulated_bifurcation",
             "lib_greedy_local_search", "lib_tabu_search", "lib_tap_annealing")
NATIVE = ("lib_simulated_annealing", "lib_greedy_local_search",
          "lib_spin_vector_langevin", "lib_angular_annealing",
          "lib_transverse_route", "lib_easy_axis_annealing",
          "lib_spin_coherent_annealing", "lib_vector_amplitude_annealing",
          "lib_mean_field_annealing", "lib_tap_annealing",
          "lib_spherical_annealing", "lib_contact_annealing",
          "lib_replica_annealing", "lib_heat_bath_annealing",
          "lib_tabu_search", "lib_random_search", "lib_exchange_cascade")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    from qubo_benchmark.runtime.common import atomic
    atomic(path, value)


def source_identity():
    files = [p for folder in ("src/qubo_solvers", "src/qubo_benchmark")
             for p in sorted((ROOT / folder).rglob("*.py"))]
    files.append(Path(__file__))
    files.extend(ROOT / "configs" / name for name in ("benchmark_solvers.json",
                 "benchmark_solvers_tuned.json", "benchmark_solvers_20s.json", "continuous_tuning_plan.json"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": commit, "source_sha256": digest(hashes), "files": hashes}


def assert_protected(settings, baseline=None, *, n_variables=None, protected_memory_guard_override=False):
    baseline = baseline or read_json(ROOT / "configs/benchmark_solvers.json")["solvers"]
    for solver in PROTECTED:
        expected = copy.deepcopy(baseline[solver])
        # Explicit user authorization is narrowly limited to this storage guard
        # for large inputs. It does not change trajectories, steps or dynamics.
        if protected_memory_guard_override and n_variables is not None and n_variables >= 5000:
            if "memory_limit_bytes" in expected:
                expected["memory_limit_bytes"] = 8 * 1024**3
        if settings.get(solver) != expected:
            raise ValueError(f"Protected parameters changed: {solver}")


def make_settings():
    """An explicit initial preset; size profiles are resolved before scheduling."""
    baseline = read_json(ROOT / "configs/benchmark_solvers.json")["solvers"]
    tuned = read_json(ROOT / "configs/benchmark_solvers_tuned.json")["solvers"]
    settings = copy.deepcopy(tuned)
    for solver in PROTECTED:
        settings[solver] = copy.deepcopy(baseline[solver])
    for solver, parameters in settings.items():
        if solver in PROTECTED:
            continue
        step_key = "sweeps" if "sweeps" in parameters else "max_steps" if "max_steps" in parameters else "steps"
        parameters[step_key] = 10000000
        parameters["run_batch_size"] = parameters["runs"]
        if "memory_limit_bytes" in parameters and not parameters.get("graph_block"):
            # Five dense matrices alone need about 2 GB at n=10,000/float32.
            parameters["memory_limit_bytes"] = 8 * 1024**3
    assert_protected(settings, baseline)
    return dict(schema_version=2,
                description="Initial 20-second presets; no global optimum or performance claim. "
                "Protected five match the original baseline exactly. Resolve immutable size "
                "profiles and opt-in wall-clock schedules before each campaign; finite protected searches may return early. "
                "User-authorized exception: for n>=5000 only, raise the existing protected native memory guard "
                "to 8 GiB; all numerical parameters and SBM remain unchanged.",
                protected_memory_guard_override=True,
                solvers=settings)


def make_plan():
    settings = make_settings()
    return dict(
        schema_version=1, budget_s=20., checkpoints_s=list(CHECKPOINTS),
        protected=list(PROTECTED), protected_memory_guard_override=True,
        protected_memory_guard_authorization="User authorized raising ONLY existing native protected "
            "memory_limit_bytes to 8 GiB for n>=5000 (SA, greedy, tabu, TAP). "
            "All other protected parameters, n<=1000 guards and SBM remain original.",
        base_configuration="configs/benchmark_solvers_20s.json",
        previous_configuration_sha256=digest(read_json(ROOT / "configs/benchmark_solvers_tuned.json")),
        protected_configuration_sha256=digest({s: settings["solvers"][s] for s in PROTECTED}),
        training_instances=["gka1e", "be200.8.1"],
        holdout_instances=["bqp500-2", "be200.8.2"],
        training_seeds=[101, 102], holdout_seeds=[201, 202],
        block_policy="paired: representative 0/seed 0, representative 1/seed 1; not a cross-product",
        candidates=[dict(id="profile", population_multiplier=1., step_multiplier=1.),
                    dict(id="population64", population_multiplier=2., step_multiplier=1.),
                    dict(id="late_freeze", population_multiplier=1., step_multiplier=1.,
                         exchange_freeze_fraction=.95)],
        size_profiles=[dict(max_variables=1000, steps=10000000, population=32),
                       dict(max_variables=5000, steps=10000000, population=32, max_dense_variables=16384),
                       dict(max_variables=10000, steps=10000000, population=32, max_dense_variables=16384)],
        schedule_policy="wall-clock annealing controls over one stateful 20-second search for the 18 "
                        "tunable methods only; protected five retain their legacy schedules",
        calibration=dict(overprovision=1.25, maximum_steps=100000000,
                         input_format="profiles[solver][n_variables].steps_per_second", 
                         timing_only=True, excluded_from_quality=True),
        selection_metrics=["mean_gap_raw", "exact_bks_hit_probability", "tts99_s"],
        tts99_definition="t*ln(0.01)/ln(1-p); p=0 unavailable; p=1 literal limit 0",
        selection_policy="20-second mean signed raw objective gap (energy minus reference), "
                         "descending exact/BKS hit probability, "
                         "ascending TTS99; candidate ID breaks ties. Select before holdout.",
        holdout_policy="Preselected winner and profile baseline, two fresh paired seeds on disjoint "
                       "instances; never reselect on holdout. Skip identical duplicate baseline.",
        execution=dict(workers_per_device=1, cpu_threads=1, setup_timeout_s=120.,
                       return_grace_s=5., watchdog_s=150., result_queue_limit=8,
                       memory_admission_fraction=.8, context_reserve_bytes=1024**3,
                       dense_matrix_copies=12),
        limitations=["At most 180 exploratory trials for 18 tunable methods, not a 100-seed campaign.",
                     "Paired blocks do not estimate independent instance and seed effects.",
                     "TTS99 uses checkpoint hit probability, no interpolation or TTT.",
                     "Step ceilings are immutable guards, not measured throughput or a new start at a checkpoint.",
                     "Native dense storage remains quadratic even for sparse source objectives.",
                     "Only the explicitly authorized existing native protected memory guard becomes "
                     "8 GiB at n>=5000; all other protected parameters and SBM remain original.",
                     "Carry incumbents after a natural return; finite schedules may produce flat curves."])


def validate_plan(plan, settings):
    assert_protected(settings["solvers"])
    if plan["budget_s"] != 20. or tuple(plan["checkpoints_s"]) != CHECKPOINTS:
        raise ValueError("Continuous tuning requires the declared nine checkpoints through 20 seconds")
    if set(plan["protected"]) != set(PROTECTED):
        raise ValueError("All five protected solver IDs must remain protected")
    if plan.get("protected_memory_guard_override") is not True or settings.get("protected_memory_guard_override") is not True:
        raise ValueError("Large-N protected memory override must record its explicit authorization")
    if plan["protected_configuration_sha256"] != digest({s: settings["solvers"][s] for s in PROTECTED}):
        raise ValueError("Protected configuration hash changed")
    if plan["previous_configuration_sha256"] != digest(read_json(ROOT / "configs/benchmark_solvers_tuned.json")):
        raise ValueError("Previous tuning configuration changed; regenerate a separate plan")
    candidates = plan["candidates"]
    if len(candidates) != 3 or len({c["id"] for c in candidates}) != 3 or candidates[0]["id"] != "profile":
        raise ValueError("Predeclare exactly three candidates, with profile baseline first")
    if plan["execution"]["workers_per_device"] != 1:
        raise ValueError("Continuous tuning uses one worker per device")
    train, holdout = plan["training_seeds"], plan["holdout_seeds"]
    if len(train) != 2 or len(holdout) != 2 or len(set(train + holdout)) != 4:
        raise ValueError("Predeclare two training and two fresh holdout seeds")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 or s >= 2**63-1 for s in train + holdout):
        raise ValueError("Seeds must be valid nonnegative integer RNG seeds")


def resolve_parameters(solver, parameters, candidate, n_variables, plan, calibration=None):
    """Return one frozen configuration, independent of objectives and references."""
    result = copy.deepcopy(parameters)
    if solver in PROTECTED:
        assert_protected({s: result if s == solver else
                          read_json(ROOT / "configs/benchmark_solvers.json")["solvers"][s]
                          for s in PROTECTED})
        if plan.get("protected_memory_guard_override") and n_variables >= 5000:
            if "memory_limit_bytes" in result:
                result["memory_limit_bytes"] = 8 * 1024**3
        return result
    profiles = plan["size_profiles"]
    profile = next((p for p in profiles if n_variables <= p["max_variables"]), None)
    if profile is None:
        raise ValueError("No immutable size profile above 10,000 variables")
    if n_variables >= 5000 and "max_dense_variables" in result:
        # Native padding can exceed 10,000 even when the source input has exactly
        # 10,000 variables. Admit the same padded dense operator instead of
        # forcing a nearly dense CSR operator with three times the storage.
        result["max_dense_variables"] = 16384
    step_key = "sweeps" if "sweeps" in result else "max_steps" if "max_steps" in result else "steps"
    steps = max(result[step_key], profile["steps"])
    measured = (calibration or {}).get("profiles", {}).get(solver, {}).get(str(n_variables))
    if measured:
        rate = measured["steps_per_second"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Calibration throughput must be finite and positive")
        steps = max(profile["steps"], result[step_key], math.ceil(rate * plan["budget_s"] *
                                              plan["calibration"]["overprovision"]))
    result[step_key] = min(plan["calibration"]["maximum_steps"],
                           math.ceil(steps * candidate["step_multiplier"]))
    result["runs"] = math.ceil(result["runs"] * candidate["population_multiplier"])
    # All initial native trajectories share one stateful search; a later batch
    # would otherwise create a fresh start after an early finite search.
    result["run_batch_size"] = result["runs"]
    if solver == "lib_exchange_cascade" and "exchange_freeze_fraction" in candidate:
        result["freeze_fraction"] = candidate["exchange_freeze_fraction"]
    elif candidate["id"] == "late_freeze":
        # A modest integration-time ablation changes a third configuration while
        # retaining every force equation. Exchange uses its explicit late freeze.
        for key in ("time_step",):
            if key in result:
                result[key] *= .5
        if solver in ("lib_random_search", "lib_heat_bath_annealing"):
            result["runs"] *= 3
            result["run_batch_size"] = result["runs"]
    return result


def memory_estimate(solver, parameters, n_variables, plan):
    """Conservative admission estimate, separate from unchanged solver guards."""
    width = 8 if parameters["dtype"] == "float64" else 4
    batch = min(parameters["runs"], parameters.get("run_batch_size") or parameters["runs"])
    copies = plan["execution"]["dense_matrix_copies"]
    replica = parameters.get("replicas", 1)
    storage = copies * n_variables * n_variables * width
    workspace = 128 * batch * n_variables * width * max(1, replica)
    return storage + workspace + plan["execution"]["context_reserve_bytes"]


def load_inputs(path):
    """Read a standalone objective manifest; never pass its reference to workers."""
    path = Path(path).resolve()
    value = read_json(path)
    entries = value["instances"] if isinstance(value, dict) else value
    normalized = []
    for entry in entries:
        item = dict(entry)
        item["n_variables"] = item.get("n_variables", item.get("binary_variables", item.get("n")))
        item["reference_objective"] = item.get("reference_objective", item.get("normalized_reference_objective_min"))
        npz = item.get("npz", item.get("npz_path", item.get("objective_path")))
        if npz is None:
            from qubo_benchmark.pipeline import paths
            npz = paths(ROOT / "benchmark_data/qubo37", item["instance_id"])[0]
        npz = Path(npz)
        item["npz"] = str(npz.resolve() if npz.is_absolute() else (path.parent / npz).resolve())
        if not isinstance(item["n_variables"], int) or item["n_variables"] <= 0:
            raise ValueError("Each input needs a positive n_variables")
        reference = item["reference_objective"]
        if isinstance(reference, bool) or not isinstance(reference, (float, int)) or not math.isfinite(reference):
            raise ValueError("Each input needs a finite post-solve reference_objective")
        if not Path(item["npz"]).is_file():
            raise ValueError(f"Missing objective: {item['npz']}")
        item["npz_sha256"] = hashlib.sha256(Path(item["npz"]).read_bytes()).hexdigest()
        normalized.append(item)
    if len({e["instance_id"] for e in normalized}) != len(normalized):
        raise ValueError("Duplicate input identity")
    return normalized


def representatives(inputs, plan, requested=None, phase="screen"):
    if requested:
        by_id = {e["instance_id"]: e for e in inputs}
        if len(requested) != 2 or len(set(requested)) != 2:
            raise ValueError("Exactly two distinct representative IDs are required")
        return [by_id[name] for name in requested]
    ids = plan["training_instances" if phase == "screen" else "holdout_instances"]
    by_id = {e["instance_id"]: e for e in inputs}
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"Missing tuning representatives: {sorted(missing)}")
    return [by_id[name] for name in ids]


def tasks_for(plan, settings, inputs, phase="screen", selected=None, calibration=None, solvers=None):
    if phase not in ("screen", "holdout") or len(inputs) != 2:
        raise ValueError("Use two paired input blocks in screen or holdout")
    seeds = plan["training_seeds"] if phase == "screen" else plan["holdout_seeds"]
    tasks = []
    for solver, parameters in settings["solvers"].items():
        if solver in PROTECTED or (solvers and solver not in solvers):
            continue
        candidates = plan["candidates"]
        if phase == "holdout":
            if not selected or solver not in selected:
                raise ValueError(f"No frozen screening selection for {solver}")
            ids = {"profile", selected[solver]}
            candidates = [c for c in candidates if c["id"] in ids]
            if selected[solver] not in {c["id"] for c in candidates}:
                raise ValueError("Unknown selected candidate")
        for candidate in candidates:
            for entry, seed in zip(inputs, seeds):
                task = dict(phase=phase, solver=solver, candidate=candidate["id"],
                            instance=entry["instance_id"], n_variables=entry["n_variables"],
                            npz=entry["npz"], npz_sha256=entry["npz_sha256"], seed=seed,
                            parameters=resolve_parameters(solver, parameters, candidate,
                                entry["n_variables"], plan, calibration), budget=plan["budget_s"],
                            checkpoints=list(plan["checkpoints_s"]))
                task["id"] = digest(task)[:24]
                tasks.append(task)
    return tasks


def checkpoint_metrics(rows, references, checkpoints=CHECKPOINTS):
    """Mean signed raw gap (energy minus reference), exact/BKS hits and TTS99."""
    if not rows:
        raise ValueError("Metrics need planned trial rows")
    result = []
    for index, elapsed in enumerate(checkpoints):
        gaps = []
        hits = alarms = 0
        for row in rows:
            reference = references[row["instance"]]
            value = row["energies"][index]
            if value is None:
                continue
            target = reference["reference_objective"]
            proven = reference.get("reference_status") in ("published_proven_optimum", "proven_optimum")
            alarm = proven and value < target
            alarms += int(alarm)
            if row["status"] not in ("ok", "complete", "completed", "no_in_budget_candidate") or alarm:
                continue
            gaps.append(value - target)
            hits += int(value == target if proven else value <= target)
        probability = hits / len(rows)
        tts = None if probability == 0 else 0. if probability == 1 else (
            elapsed * math.log(.01) / math.log1p(-probability))
        result.append(dict(checkpoint_s=elapsed,
                           mean_gap_raw=sum(gaps) / len(gaps) if len(gaps) == len(rows) else None,
                           scored_trials=len(gaps), planned_trials=len(rows),
                           exact_bks_hit_probability=probability, hits=hits, tts99_s=tts,
                           validation_alarms=alarms))
    return result


def select_winners(rows, references, plan):
    groups = {}
    for row in rows:
        if row["phase"] == "screen":
            groups.setdefault((row["solver"], row["candidate"]), []).append(row)
    ranked = {}
    summaries = []
    expected = len(plan["training_seeds"])
    for (solver, candidate), trials in sorted(groups.items()):
        if len(trials) != expected:
            raise ValueError(f"Incomplete screening block: {solver}/{candidate}")
        metrics = checkpoint_metrics(trials, references, plan["checkpoints_s"])
        final = metrics[-1]
        summaries.append(dict(solver=solver, candidate=candidate, metrics=metrics))
        rank = (final["mean_gap_raw"] if final["mean_gap_raw"] is not None else math.inf,
                -final["exact_bks_hit_probability"],
                final["tts99_s"] if final["tts99_s"] is not None else math.inf, candidate)
        ranked.setdefault(solver, []).append((rank, candidate))
    for solver, candidates in ranked.items():
        if len(candidates) != len(plan["candidates"]):
            raise ValueError(f"Incomplete candidate set: {solver}")
        if all(not math.isfinite(rank[0]) for rank, _ in candidates):
            raise ValueError(f"No eligible screening candidate: {solver}")
    return {solver: min(candidates)[1] for solver, candidates in ranked.items()}, summaries


def normalize_result(result, checkpoints):
    if not isinstance(result, dict):
        raise ValueError("Continuous trial must return a compact result dictionary")
    energies = result.get("energies", result.get("checkpoint_energies"))
    if not isinstance(energies, (list, tuple)) or len(energies) != len(checkpoints):
        raise ValueError("Continuous trial must return nine checkpoint energies")
    best = None
    for value in energies:
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                if isinstance(value, float) and math.isnan(value) and best is None:
                    continue
                raise ValueError("Checkpoint energy must be finite or null")
            if best is not None and value > best:
                raise ValueError("Checkpoint incumbent energies must be nonincreasing")
            best = value
        elif best is not None:
            raise ValueError("An incumbent cannot disappear after capture")
    return dict(energies=[None if isinstance(v, float) and math.isnan(v) else v for v in energies],
                status=result.get("status", "ok"), error=result.get("error"))


def _worker(device, incoming, outgoing, plan):
    try:
        from qubo_benchmark.runtime.worker import WarmWorker
        from qubo_benchmark.runtime.continuous_worker import run_continuous_trial
        warm = WarmWorker(dict(device=device, cpu_threads=1))
        outgoing.put(dict(kind="ready", device=device))
        while True:
            task = incoming.get()
            if task is None:
                return
            outgoing.put(dict(kind="preparing", device=device, task=task["id"]))
            estimate = memory_estimate(task["solver"], task["parameters"], task["n_variables"], plan)
            if device.startswith("cuda"):
                free, _ = warm.torch.cuda.mem_get_info(device)
                if estimate > free * plan["execution"]["memory_admission_fraction"]:
                    row = dict(energies=[None] * len(task["checkpoints"]), status="memory_admission_rejected",
                               error=f"Conservative estimate {estimate} exceeds device admission allowance")
                    outgoing.put(dict(kind="result", device=device, task=task["id"], result=row))
                    continue
            # This is the complete solver-facing job. No manifest/reference entry
            # is present in this process, including worker warmup and preparation.
            job = {k: task[k] for k in ("solver", "parameters", "seed", "npz", "budget", "checkpoints")}
            job.update(device=device, cpu_threads=1, wall_schedule=task["solver"] not in PROTECTED)
            try:
                if hashlib.sha256(Path(job["npz"]).read_bytes()).hexdigest() != task["npz_sha256"]:
                    raise ValueError("Objective input changed after preflight")
                warm.prepare(job)
                outgoing.put(dict(kind="started", device=device, task=task["id"]))
                row = normalize_result(run_continuous_trial(warm, job), task["checkpoints"])
            except Exception as exc:
                row = dict(energies=[None] * len(task["checkpoints"]), status="error",
                           error=f"{type(exc).__name__}: {exc}")
            outgoing.put(dict(kind="result", device=device, task=task["id"], result=row))
    except BaseException:
        outgoing.put(dict(kind="fatal", device=device, error=traceback.format_exc()))


def run_tasks(tasks, output, devices, plan, identity):
    """One process/device, bounded queues and setup/solve watchdogs."""
    output = Path(output)
    pending = []
    for task in tasks:
        path = output / "trials" / f"{task['id']}.json"
        if path.exists():
            existing = read_json(path)
            if existing["task_sha256"] != digest(task) or existing["identity_sha256"] != digest(identity):
                raise ValueError("Committed trial identity does not match resume")
        else:
            pending.append(task)
    if not pending:
        return
    context = mp.get_context("spawn")
    outgoing = context.Queue(maxsize=plan["execution"]["result_queue_limit"])
    workers = {}
    completed = len(tasks) - len(pending)
    todo = iter(pending)
    atomic_json(output / "status.json", dict(state="startup", completed=completed, total=len(tasks),
                phase=tasks[0]["phase"] if tasks else None, devices=devices))
    try:
        for device in devices:
            incoming = context.Queue(maxsize=1)
            process = context.Process(target=_worker, args=(device, incoming, outgoing, plan))
            process.start()
            workers[device] = dict(process=process, incoming=incoming, active=None,
                                   phase="startup", since=time.monotonic())
        ready = set()
        while len(ready) < len(workers):
            try:
                message = outgoing.get(timeout=1)
            except queue.Empty:
                for worker in workers.values():
                    if not worker["process"].is_alive() or time.monotonic() - worker["since"] > plan["execution"]["setup_timeout_s"]:
                        raise RuntimeError("Worker startup exceeded deadline or process exited")
                atomic_json(output / "status.json", dict(state="startup", completed=completed,
                            total=len(tasks), ready_devices=sorted(ready)))
                continue
            if message["kind"] == "fatal":
                raise RuntimeError(message["error"])
            if message["kind"] == "ready":
                ready.add(message["device"])
        # Initial infrastructure barrier: no solve before every GPU is ready.
        if source_identity() != identity["source"]:
            raise RuntimeError("Source or HEAD changed before dispatch")
        for worker in workers.values():
            task = next(todo, None)
            worker["incoming"].put(task)
            worker.update(active=task, phase="setup" if task else "idle", since=time.monotonic())
        while completed < len(tasks):
            message = None
            try:
                message = outgoing.get(timeout=.5)
            except queue.Empty:
                pass
            if message:
                device = message["device"]
                worker = workers[device]
                if message["kind"] == "fatal":
                    raise RuntimeError(message["error"])
                if message["kind"] == "started":
                    worker.update(phase="solve", since=time.monotonic())
                elif message["kind"] == "result":
                    task = worker["active"]
                    if task is None or message["task"] != task["id"]:
                        raise RuntimeError("Unexpected worker trial identity")
                    if source_identity() != identity["source"]:
                        raise RuntimeError("Source or HEAD changed while jobs were active")
                    row = dict(task, device=device, **message["result"],
                               task_sha256=digest(task), identity_sha256=digest(identity))
                    # Persist configuration once in frozen tasks; trial rows remain compact.
                    for key in ("parameters", "npz", "npz_sha256", "checkpoints", "budget", "n_variables"):
                        row.pop(key, None)
                    atomic_json(output / "trials" / f"{task['id']}.json", row)
                    completed += 1
                    next_task = next(todo, None)
                    worker["incoming"].put(next_task)
                    worker.update(active=next_task, phase="setup" if next_task else "idle", since=time.monotonic())
                    print(f"{completed}/{len(tasks)} {task['solver']} {task['candidate']} {row['status']}", flush=True)
            now = time.monotonic()
            for device, worker in workers.items():
                task = worker["active"]
                if task is None:
                    continue
                limit = (plan["execution"]["setup_timeout_s"] if worker["phase"] == "setup"
                         else task["budget"] + plan["execution"]["return_grace_s"])
                if not worker["process"].is_alive() or now - worker["since"] > limit:
                    # Terminate and abort scheduling; never reuse a device while
                    # the timed-out process could still be executing.
                    worker["process"].terminate()
                    worker["process"].join(timeout=5)
                    row = dict(id=task["id"], phase=task["phase"], solver=task["solver"],
                               candidate=task["candidate"], instance=task["instance"], seed=task["seed"],
                               device=device, energies=[None] * len(task["checkpoints"]),
                               status="watchdog_timeout",
                               error=f"{worker['phase']} watchdog", task_sha256=digest(task),
                               identity_sha256=digest(identity))
                    atomic_json(output / "trials" / f"{task['id']}.json", row)
                    raise RuntimeError(f"{device} worker failed; remaining trials pending for resume")
            atomic_json(output / "status.json", dict(state="running", completed=completed, total=len(tasks),
                        active={d: w["active"]["id"] for d, w in workers.items() if w["active"]}))
    except BaseException as exc:
        atomic_json(output / "status.json", dict(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    completed=completed, total=len(tasks), error=str(exc)))
        raise
    finally:
        for worker in workers.values():
            process = worker["process"]
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for worker in workers.values():
            worker["incoming"].close()
        outgoing.close()
    atomic_json(output / "status.json", dict(state="complete", completed=len(tasks), total=len(tasks)))


def freeze_json(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen resume artifact changed: {path}")
    else:
        atomic_json(path, value)


def run(plan, settings, inputs, output, devices, calibration=None, solvers=None, requested=None,
        holdout_requested=None):
    from qubo_benchmark.runtime.common import DirectoryLock
    from qubo_benchmark.runtime.selection import environment
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_plan(plan, settings)
    selected_inputs = representatives(inputs, plan, requested)
    holdout_inputs = representatives(inputs, plan, holdout_requested, phase="holdout")
    if {e["instance_id"] for e in selected_inputs} & {e["instance_id"] for e in holdout_inputs}:
        raise ValueError("Training and holdout objectives must be disjoint")
    env = environment(",".join(devices), require_gpu=devices != ["cpu"])
    identity = dict(source=source_identity(), plan_sha256=digest(plan), settings_sha256=digest(settings),
                    inputs_sha256=digest(inputs), devices=devices, workers_per_device=1,
                    environment=env, calibration_sha256=digest(calibration), solvers=solvers,
                    representative_ids=[e["instance_id"] for e in selected_inputs],
                    holdout_ids=[e["instance_id"] for e in holdout_inputs])
    with DirectoryLock(output):
        freeze_json(output / "manifest.json", identity)
        freeze_json(output / "plan.json", plan)
        freeze_json(output / "initial_configuration.json", settings)
        freeze_json(output / "reference_snapshot.json", inputs)
        screen = tasks_for(plan, settings, selected_inputs, calibration=calibration, solvers=solvers)
        freeze_json(output / "screen_tasks.json", screen)
        run_tasks(screen, output, devices, plan, identity)
        rows = [read_json(output / "trials" / f"{task['id']}.json") for task in screen]
        refs = {e["instance_id"]: e for e in inputs}
        selected, summary = select_winners(rows, refs, plan)
        freeze_json(output / "selected.json", selected)
        atomic_json(output / "screen_summary.json", summary)
        holdout = tasks_for(plan, settings, holdout_inputs, phase="holdout", selected=selected,
                            calibration=calibration, solvers=solvers)
        freeze_json(output / "holdout_tasks.json", holdout)
        run_tasks(holdout, output, devices, plan, identity)
        validation = [read_json(output / "trials" / f"{task['id']}.json") for task in holdout]
        atomic_json(output / "holdout_summary.json", [dict(solver=solver, candidate=candidate,
                    preselected=candidate == selected[solver],
                    metrics=checkpoint_metrics([r for r in validation if r["solver"] == solver
                                                and r["candidate"] == candidate], refs,
                                               plan["checkpoints_s"]))
                    for solver, candidate in sorted({(r["solver"], r["candidate"]) for r in validation})])
        frozen = dict(schema_version=2, solvers=copy.deepcopy(settings["solvers"]),
                      description="Preselected bounded-screen winners; held-out metrics reported without reselection. "
                                  "Exploratory evidence only; no universal or global optimum claim.",
                      selected_candidates=selected, size_profiles=plan["size_profiles"],
                      calibration=calibration, protected=list(PROTECTED),
                      protected_memory_guard_override=True,
                      default_profile_variables=1000, size_configurations={})
        for size in (100, 200, 500, 1000, 5000, 10000):
            frozen["size_configurations"][str(size)] = copy.deepcopy(settings["solvers"])
            for solver in settings["solvers"]:
                if solver in PROTECTED:
                    continue
                candidate_id = selected.get(solver, "profile")
                candidate = next(c for c in plan["candidates"] if c["id"] == candidate_id)
                frozen["size_configurations"][str(size)][solver] = resolve_parameters(
                    solver, settings["solvers"][solver], candidate, size, plan, calibration)
            for solver in PROTECTED:
                frozen["size_configurations"][str(size)][solver] = resolve_parameters(
                    solver, settings["solvers"][solver], plan["candidates"][0], size, plan, calibration)
            assert_protected(frozen["size_configurations"][str(size)], n_variables=size,
                             protected_memory_guard_override=True)
        frozen["solvers"] = copy.deepcopy(frozen["size_configurations"]["1000"])
        freeze_json(output / "benchmark_solvers_selected_20s.json", frozen)
        atomic_json(output / "status.json", dict(state="complete", screen_trials=len(screen),
                    holdout_trials=len(holdout), total_trials=len(screen)+len(holdout),
                    selected_sha256=digest(selected), caveats=plan["limitations"]))
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "dry-run", "run"))
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/continuous_tuning_plan.json")
    parser.add_argument("--configuration", type=Path, default=ROOT / "configs/benchmark_solvers_20s.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "configs/qubo_benchmark_catalog_200_500_1000_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results/continuous_tuning")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--instances", help="Two representative IDs, comma-separated")
    parser.add_argument("--holdout-instances", help="Two disjoint holdout objective IDs, comma-separated")
    parser.add_argument("--solvers", help="Explicit bounded subset of the 18 tunable methods")
    parser.add_argument("--calibration", type=Path, help="Pre-measured timing profiles; no targets or energies")
    args = parser.parse_args()
    if args.command == "plan":
        atomic_json(args.plan, make_plan())
        atomic_json(args.configuration, make_settings())
        return
    plan, settings = read_json(args.plan), read_json(args.configuration)
    validate_plan(plan, settings)
    inputs = load_inputs(args.inputs)
    calibration = read_json(args.calibration) if args.calibration else None
    solvers = args.solvers.split(",") if args.solvers else None
    if solvers and (set(solvers) - (set(settings["solvers"]) - set(PROTECTED))):
        parser.error("Select only canonical tunable IDs; protected methods are unchanged comparisons")
    requested = args.instances.split(",") if args.instances else None
    selected_inputs = representatives(inputs, plan, requested)
    holdout_requested = args.holdout_instances.split(",") if args.holdout_instances else None
    holdout_inputs = representatives(inputs, plan, holdout_requested, phase="holdout")
    if args.command == "dry-run":
        tasks = tasks_for(plan, settings, selected_inputs, calibration=calibration, solvers=solvers)
        print(json.dumps(dict(screen_trials=len(tasks), maximum_holdout_trials=len(tasks)*2//3,
                             representatives=[e["instance_id"] for e in selected_inputs],
                             holdout_instances=[e["instance_id"] for e in holdout_inputs],
                             checkpoints_s=plan["checkpoints_s"], protected=list(PROTECTED),
                             gpu_memory_estimate_max=max(memory_estimate(t["solver"], t["parameters"],
                                                       t["n_variables"], plan) for t in tasks),
                             limitations=plan["limitations"]), indent=2))
        return
    from qubo_benchmark.runtime.selection import device_selection
    devices = device_selection(args.devices)
    if args.devices in ("auto", "all") and devices == ["cpu"]:
        parser.error("GPU tuning requires available CUDA; use --devices cpu explicitly for a CPU diagnostic")
    run(plan, settings, inputs, args.output, devices, calibration, solvers, requested, holdout_requested)


if __name__ == "__main__":
    main()

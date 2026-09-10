"""One-day raw Transverse Route diagnostic; never repair or decode invalid samples."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

import numpy as np
import torch

from margin_engine.yaml_application import MarginApplicationConfig
from margin_calculator.calculation_outcome import CalculationOutcome
from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_transverse_route_bqm_solver import TorchTransverseRouteBQMSolver


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_no_repair"
LOCK = threading.Lock()
STOP = threading.Event()
START = time.perf_counter()
STATUS = {"state": "initializing", "generated": 0, "completed": 0,
          "feasible_scenarios": 0, "active": {}}
PROBLEMS = {}
ROWS = []


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def snapshot():
    with LOCK:
        payload = dict(STATUS, elapsed_seconds=time.perf_counter() - START)
        temp = OUT / "status.json.tmp"
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        temp.replace(OUT / "status.json")


def heartbeat():
    while not STOP.wait(15):
        snapshot()
        with LOCK:
            log(f"HEARTBEAT elapsed={time.perf_counter() - START:.1f}s "
                f"generated={STATUS['generated']} completed={STATUS['completed']} "
                f"feasible={STATUS['feasible_scenarios']} active={STATUS['active']}")


def forbid_repair(*args, **kwargs):
    raise RuntimeError("Repair was invoked in a no-repair diagnostic")


def raw_add(self, samples):
    """Prefer an unmodified feasible sample; otherwise retain the raw energy winner."""
    started = time.perf_counter()
    info = PROBLEMS[id(self.problem)]
    if self.groups:
        selected = samples[:, self.groupVariables].to(torch.int64)
        prefix = torch.cat((torch.zeros((len(samples), 1), device=samples.device,
                                       dtype=torch.int64), selected.cumsum(dim=1)), dim=1)
        counts = prefix[:, self.groupOffsets[1:]] - prefix[:, self.groupOffsets[:-1]]
        violations = (counts != 1).sum(dim=1)
    else:
        violations = torch.zeros(len(samples), device=samples.device, dtype=torch.int64)
    valid = violations == 0
    feasible_count = int(valid.sum().item())
    min_violations = int(violations.min().item())
    values = samples.to(torch.float64)
    energies = values @ self.linear + self.problem.offset
    for start in range(0, self.problem.interactionCount, self.chunkSize):
        stop = start + self.chunkSize
        products = values[:, self.heads[start:stop]] * values[:, self.tails[start:stop]]
        energies += products @ self.biases[start:stop]
    if not bool(torch.isfinite(energies).all()):
        raise ValueError("Non-finite raw candidate energies")

    def best_of(mask):
        minimum = energies[mask].min()
        near = mask & (energies <= minimum + self.tolerance)
        host = samples[near].to(torch.uint8).cpu().numpy()
        return min(((tuple(int(v) for v in sample), self.problem.energy(sample))
                    for sample in host), key=lambda item: (item[1], item[0]))

    raw_best = best_of(torch.ones(len(samples), device=samples.device, dtype=torch.bool))
    if feasible_count:
        result = best_of(valid)
        if not self.feasible:
            self.best = None
        self.feasible = True
    else:
        result = raw_best
    if feasible_count or not self.feasible:
        if self.best is None or (result[1], result[0]) < (self.best[1], self.best[0]):
            self.best = result
    with LOCK:
        info["candidates_checked"] += len(samples)
        info["feasible_candidates"] += feasible_count
        info["min_violated_groups"] = min(info["min_violated_groups"], min_violations)
        info["best_raw_energy"] = min(info["best_raw_energy"], raw_best[1])
        info["scoring_seconds"] += time.perf_counter() - started
    log(f"RAW_CANDIDATES scenario={info['scenario']} device={samples.device} "
        f"count={len(samples)} feasible={feasible_count} min_violated_groups={min_violations} "
        f"raw_energy={raw_best[1]:.9g} scoring_seconds={time.perf_counter() - started:.3f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        config = HERE / "transverse_route.yaml"
        shutil.copy2(config, OUT / "configuration.yaml")
        app = MarginApplicationConfig.fromYaml(config)
        engine = app.createEngine()
        calculator = engine.marginCalculator
        params = calculator.solverParameters
        if params["steps"] != 1000 or params["runs"] != 4:
            raise ValueError("This diagnostic requires steps=1000 and runs=4")
        devices = calculator.bqmSolver.devices
        original_solve = TorchTransverseRouteBQMSolver._solveBatch

        def timed_solve(solver, problems, parameters):
            device = str(solver.device)
            for problem in problems:
                PROBLEMS[id(problem)]["device"] = device
            with LOCK:
                STATUS["active"][device] = [PROBLEMS[id(p)]["scenario"] for p in problems]
            log(f"SOLVE_START device={device} scenarios={STATUS['active'][device]}")
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = original_solve(solver, problems, parameters)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            for problem in problems:
                PROBLEMS[id(problem)]["solve_batch_seconds"] = elapsed
            with LOCK:
                STATUS["active"].pop(device, None)
            log(f"SOLVE_END device={device} problems={len(problems)} seconds={elapsed:.3f}")
            return result

        def diagnostic_outcome(risk_states, portfolio):
            contexts = {}
            maximum = 0.0

            def encoded():
                for index, risk_state in enumerate(risk_states, 1):
                    context, problem = calculator._encodeRiskState(risk_state, portfolio)
                    info = {"scenario": index, "variables": problem.variableCount,
                            "edges": problem.interactionCount, "groups": len(tuple(problem.iterOneHotGroups())),
                            "candidates_checked": 0, "feasible_candidates": 0,
                            "min_violated_groups": problem.variableCount,
                            "best_raw_energy": float("inf"), "scoring_seconds": 0.0}
                    with LOCK:
                        PROBLEMS[id(problem)] = info
                        STATUS["generated"] = index
                    contexts[id(context)] = (problem, info)
                    log(f"SCENARIO_READY index={index} variables={problem.variableCount}")
                    yield context, problem

            execution = calculator.executionPolicy.execute(calculator.bqmSolver, encoded(), params)
            fields = ["scenario", "device", "variables", "edges", "groups", "candidates_checked",
                      "feasible_candidates", "min_violated_groups", "best_raw_energy",
                      "scoring_seconds", "solve_batch_seconds", "selected_energy",
                      "selected_violated_groups", "margin"]
            try:
                with (OUT / "scenarios.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    stream.flush()
                    for context, result in execution:
                        problem, info = contexts.pop(id(context))
                        sample = np.asarray(result.sample, dtype=np.int8)
                        if sample.shape != (problem.variableCount,) or not np.isin(sample, [0, 1]).all():
                            raise ValueError("Solver returned an invalid binary sample")
                        violated = sum(int(sample[list(group)].sum()) != 1
                                       for group in problem.iterOneHotGroups())
                        margin = None
                        if violated == 0:
                            margin = max(0.0, calculator.bqmVisitor.decodeMargin(context, portfolio, result))
                            maximum = max(maximum, margin)
                        row = dict(info, selected_energy=problem.energy(sample),
                                   selected_violated_groups=violated, margin=margin)
                        writer.writerow(row)
                        stream.flush()
                        ROWS.append(row)
                        with LOCK:
                            STATUS["completed"] += 1
                            STATUS["feasible_scenarios"] += int(violated == 0)
                            PROBLEMS.pop(id(problem))
                        log(f"SCENARIO_END index={info['scenario']} violations={violated} margin={margin}")
            finally:
                close = getattr(execution, "close", None)
                if close is not None:
                    close()
            # Internal carrier only; final publication requires complete feasibility.
            return CalculationOutcome(maximum)

        CandidateSelection._repairCandidate = staticmethod(forbid_repair)
        TorchCandidateAccumulator.add = raw_add
        TorchTransverseRouteBQMSolver._solveBatch = timed_solve
        calculator.calculateOutcome = diagnostic_outcome
        with LOCK:
            STATUS["state"] = "running"
        hardware = {str(device): torch.cuda.get_device_name(device) for device in devices}
        log(f"START NO_REPAIR date={app.marginDate} steps=1000 runs=4 hardware={hardware}")
        for device in devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        report = engine.generateReport(app.portfolio, app.marginDate)
        for device in devices:
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        feasible = sum(row["selected_violated_groups"] == 0 for row in ROWS)
        complete = bool(ROWS) and feasible == len(ROWS)
        result = {"date": str(app.marginDate), "group": "US_group_1", "assets": 8590,
                  "solver": "torch_transverse_route", "repair": False, "steps": 1000,
                  "runs": 4, "seed": params.get("seed"), "dtype": params.get("dtype"),
                  "devices": len(devices), "scenarios": len(ROWS), "feasible_scenarios": feasible,
                  "candidates_checked": sum(row["candidates_checked"] for row in ROWS),
                  "feasible_candidates": sum(row["feasible_candidates"] for row in ROWS),
                  "margin": report.margin if complete else None,
                  "best_feasible_scenario_margin": report.margin if feasible else None,
                  "margin_status": "available" if complete else "unavailable_incomplete_feasibility",
                  "wall_seconds": wall, **asdict(report.timings)}
        with (OUT / "benchmark_single_date.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result))
            writer.writeheader()
            writer.writerow(result)
        metadata = dict(result, hardware=hardware, torch_version=torch.__version__,
                        cuda_version=torch.version.cuda,
                        notes=["No repair, projection, or infeasible-sample decoding is performed.",
                               "Candidate counts are after solver deduplication within candidate chunks.",
                               "Stage spans overlap; concurrent solve times must not be summed as wall time.",
                               "Wall time includes acquisition, generation, encoding, solving and scoring.",
                               "No full margin is published unless every scenario has a feasible candidate."])
        (OUT / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with LOCK:
            STATUS.update(state="completed", wall_seconds=wall, margin=result["margin"])
        log("FINISHED " + json.dumps(result))
    except BaseException:
        with LOCK:
            STATUS.update(state="failed", error=traceback.format_exc())
        raise
    finally:
        STOP.set()
        thread.join()
        snapshot()


if __name__ == "__main__":
    main()

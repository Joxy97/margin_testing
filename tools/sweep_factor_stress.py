"""Rolling factor-stress penalty sweep with persistent workers on every GPU."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

import numpy as np
import pandas as pd

from benchmark_factor_stress import prepareApplication, writeJson, normalizedProblem, referenceRow
from risk_state_generator import ReturnsPCAGrid, ReturnsPCAKey
from risk_state_generator.factor_stress_model import FactorStressModel
from margin_calculator.optimization.factor_stress import (
    FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective, solveLattice,
)
from margin_calculator.optimization.factor_stress_reference import (
    solveLinearReference, solveQuadraticReference, solveRepricedReference,
)
from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem


SOLVERS = ("torch_sbm", "torch_svl", "torch_transverse_route")


def measured(timings, name, function):
    started = time.perf_counter()
    value = function()
    timings[name] = time.perf_counter()-started
    return value


def loadModel(path):
    with np.load(path, allow_pickle=False) as values:
        model = FactorStressModel(tuple(values["instruments"]), values["exposures"],
                                 values["center"], values["directions"])
        objective = QuadraticStressObjective(float(values["constant"]), values["gradient"], values["hessian"])
    return model, objective


def prepare(args):
    """One shared fit per date; realized close is excluded from every PCA fit."""
    output = args.output
    started = time.perf_counter()
    app = prepareApplication(args, output)
    date_column = pd.read_csv(args.group, nrows=0).columns[0]
    dates = pd.to_datetime(pd.read_csv(args.group, usecols=[date_column])[date_column])
    if dates.duplicated().any():
        raise ValueError("group dates must be unique")
    dates = sorted(d.date() for d in dates if args.date is None or d.date() <= args.date)
    selected = dates[-args.days:]
    if len(selected) != args.days or dates.index(selected[0]) < args.window+1:
        raise ValueError("not enough observations for the requested rolling window")
    if len(app.portfolio.instruments) != args.expected_stocks:
        raise ValueError("group does not contain the expected number of stocks")
    engine = app.createEngine()
    metadata_seconds = time.perf_counter()-started
    before = time.perf_counter()
    engine.prepareBacktest(app.portfolio, selected)
    prefetch_seconds = time.perf_counter()-before
    (output/"models").mkdir()
    days = []
    for index, day in enumerate(selected):
        begin = time.perf_counter()
        timings = {}
        data = measured(timings, "data_acquisition_seconds", lambda: engine.getPortfolioMarketData(app.portfolio, day))
        key = ReturnsPCAKey(app.portfolio.instruments, args.window, day, args.ew_lambda, 2)
        grid = measured(timings, "pca_fit_seconds", lambda: ReturnsPCAGrid.construct(key, data))
        if not grid.calibrationEndDate < day:
            raise AssertionError("calibration must precede realized close")
        model = measured(timings, "factor_model_seconds", lambda: FactorStressModel.fromPCAGrid(grid, app.portfolio))
        objective = measured(timings, "objective_seconds", lambda: QuadraticStressObjective(*model.quadraticCoefficients()))
        before = time.perf_counter()
        prices = data.copy()
        if "date" in prices:
            prices = prices.set_index("date")
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().loc[:, list(app.portfolio.instruments)].ffill().dropna()
        current = prices.loc[pd.Timestamp(day)].to_numpy(dtype=float)
        previous = prices.loc[prices.index < pd.Timestamp(day)].iloc[-1].to_numpy(dtype=float)
        if not np.isfinite(current).all() or not np.isfinite(previous).all() or np.any(previous <= 0):
            raise ValueError("realized prices must be finite and previous prices positive")
        realized = float((current/previous-1) @ model.exposures)
        previous_date = prices.index[prices.index < pd.Timestamp(day)][-1].date()
        timings["realized_pnl_seconds"] = time.perf_counter()-before
        before = time.perf_counter()
        history = grid.logReturnMean + grid.logReturnScale*(grid.pcaMean + grid.factors@grid.loadings + grid.residuals)
        historical_margin = max(0., -float((np.expm1(history) @ model.exposures).min()))
        timings["historical_stress_seconds"] = time.perf_counter()-before
        references = []
        for name, solve in (("linear_analytic", lambda: solveLinearReference(objective, args.radius)),
                            ("quadratic_continuous", lambda: solveQuadraticReference(objective, args.radius)),
                            ("exact_repricing_continuous", lambda: solveRepricedReference(model, args.radius))):
            before = time.perf_counter()
            result = solve()
            elapsed = time.perf_counter()-before
            timings[name+"_seconds"] = elapsed
            row = referenceRow(name, result, model, objective, elapsed)
            row.update(breach=bool(-realized > row["margin"]))
            references.append(row)
        lattices = {}
        for bits in args.bits:
            before = time.perf_counter()
            encoded = FactorStressQUBO.build(objective, FactorStressQUBOConfig(bits, args.radius))
            integers, value, count = solveLattice(encoded)
            coordinates = encoded.coordinates(encoded.encodeIntegers(integers))
            margin = max(0., -float(model.pnl(coordinates)))
            elapsed = time.perf_counter()-before
            timings[f"lattice_{bits}_seconds"] = elapsed
            lattices[str(bits)] = dict(objective=value, margin=margin, coordinates=coordinates.tolist(),
                                      integer_coordinates=integers.tolist(), seconds=elapsed, evaluations=count,
                                      breach=bool(-realized > margin))
        before = time.perf_counter()
        np.savez(output/"models"/f"{index:03d}.npz", instruments=np.array(model.instruments),
                 exposures=model.exposures, center=model.center, directions=model.directions,
                 constant=objective.constant, gradient=objective.gradient, hessian=objective.hessian)
        timings["model_write_seconds"] = time.perf_counter()-before
        timings["preparation_day_wall_seconds"] = time.perf_counter()-begin
        row = dict(index=index, date=str(day), previous_close_date=str(previous_date),
                   calibration_start=str(grid.calibrationStartDate), calibration_end=str(grid.calibrationEndDate),
                   stocks=len(model.instruments), realized_pnl=realized, realized_loss=-realized,
                   historical_margin=historical_margin, references=references, lattice=lattices,
                   pca_explained=grid.explained.tolist(), timings=timings)
        days.append(row)
        writeJson(output/"days.json", {"days": days})
        writeJson(output/"status.json", dict(stage="preparing", prepared=len(days), total=args.days,
                                            date=str(day), stocks=len(model.instruments)))
        print(json.dumps(dict(stage="prepared_day", date=str(day), seconds=timings["preparation_day_wall_seconds"])), flush=True)
    return days, dict(metadata_seconds=metadata_seconds, prefetch_seconds=prefetch_seconds,
                      preparation_wall_seconds=time.perf_counter()-started)


def buildTrial(objective, bits, multiplier, radius, seed_offset):
    encoded = FactorStressQUBO.build(objective, FactorStressQUBOConfig(bits, radius, 1.1, multiplier))
    normalized, scale = normalizedProblem(encoded.problem)
    # Paired initial conditions across penalties, independent of GPU and batching.
    problem = QUBOProblem(normalized.linear, normalized.quadraticHeads, normalized.quadraticTails,
                         normalized.quadraticBiases, offset=normalized.offset, seedOffset=seed_offset)
    return encoded, problem, scale


def batches(args):
    groups = {name: [] for name in SOLVERS}
    for day in range(args.days):
        for bits in args.bits:
            for repeat in range(args.repeats):
                for penalty_index, multiplier in enumerate(args.multipliers):
                    seed_offset = day*100000 + bits*1000 + repeat
                    for solver in SOLVERS:
                        trial = dict(id=f"d{day:03d}_b{bits}_p{penalty_index:02d}_r{repeat}_{solver}",
                                     day=day, bits=bits, repeat=repeat, penalty_index=penalty_index,
                                     multiplier=multiplier, solver=solver, seed_offset=seed_offset)
                        groups[solver].append(trial)
    result = []
    # Interleave solvers so the first eight workers immediately run all three.
    length = len(groups[SOLVERS[0]])
    for start in range(0, length, args.batch_size):
        for solver in SOLVERS:
            result.append(dict(id=len(result), solver=solver, trials=groups[solver][start:start+args.batch_size]))
    return result


def worker(gpu, inbox, events, config):
    try:
        import torch
        from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(gpu)
        output = Path(config["output"])
        days = json.loads((output/"days.json").read_text())["days"]
        models = {d["index"]: loadModel(output/"models"/f"{d['index']:03d}.npz") for d in days}
        repairs = {}
        if config.get("repair_samples", False):
            from margin_calculator.optimization.factor_stress_repair import FactorStressRepair
            (output/"repaired_samples").mkdir(exist_ok=True)
        solvers = {name: BQMSolverFactory.create(name, {"device": f"cuda:{gpu}"}) for name in SOLVERS}
        params = dict(steps=config["steps"], runs=config["runs"], run_batch_size=config["runs"],
                      dtype="float64", seed=config["seed"], energy_chunk_size=8192)
        tiny = QUBOProblem([-1., -.5], [0], [1], [1.])
        before = time.perf_counter()
        for name, solver in solvers.items():
            warm = {**params, "steps": 16, "runs": 1, "run_batch_size": 1}
            if name == "torch_transverse_route":
                warm.update(matrix_format="sparse", candidate_interval=16, cuda_graph=True)
            solver.solve(tiny, warm)
        torch.cuda.synchronize()
        events.put(dict(type="ready", gpu=gpu, warmup_seconds=time.perf_counter()-before,
                        name=torch.cuda.get_device_name(gpu), torch=torch.__version__))
        while True:
            batch = inbox.get()
            if batch is None:
                break
            wall_start = time.perf_counter()
            events.put(dict(type="started", batch=batch["id"], gpu=gpu, solver=batch["solver"],
                            jobs=len(batch["trials"]), started=time.time()))
            encoded_models, problems, rows = [], [], []
            solver = solvers[batch["solver"]]
            supplied = dict(params)
            if batch["solver"] == "torch_transverse_route":
                supplied.update(matrix_format="sparse", candidate_interval=max(1, config["steps"]//8),
                                cuda_graph=True, graph_steps=25)
            torch.cuda.reset_peak_memory_stats(gpu)
            for trial in batch["trials"]:
                model, objective = models[trial["day"]]
                started = time.perf_counter()
                encoded, problem, scale = buildTrial(objective, trial["bits"], trial["multiplier"],
                                                    trial.get("radius", config["radius"]), trial["seed_offset"])
                build_seconds = time.perf_counter()-started
                encoded_models.append(encoded)
                problems.append(problem)
                rows.append(dict(**trial, gpu=gpu, batch=batch["id"], date=days[trial["day"]]["date"],
                                 variables=problem.variableCount, edges=problem.interactionCount,
                                 scenario_bits=encoded.scenarioBits, product_bits=len(encoded.productPairs),
                                 slack_bits=len(encoded.slackWeights), penalty=encoded.penalty,
                                 sufficient_penalty=encoded.penalty > encoded.objectiveRangeBound,
                                 global_solver_scale=scale, qubo_build_seconds=build_seconds))
            estimate = sum(solver.estimatedWorkingMemoryBytes(p, supplied) for p in problems)
            free, _ = torch.cuda.mem_get_info(gpu)
            if estimate > .7*free:
                raise MemoryError(f"batch estimate {estimate} exceeds 70% free memory on GPU {gpu}")
            torch.cuda.synchronize()
            begin = time.perf_counter()
            results = solver.solveMany(problems, supplied)
            torch.cuda.synchronize()
            solve_seconds = time.perf_counter()-begin
            if len(results) != len(rows):
                raise AssertionError("solver returned wrong number of results")
            for row, encoded, result in zip(rows, encoded_models, results):
                before = time.perf_counter()
                sample = np.array(result.sample, dtype=np.uint8)
                info = encoded.diagnostics(sample)
                coordinates = encoded.coordinates(sample)
                expanded = encoded.problem.energy(sample)
                validation_seconds = time.perf_counter()-before
                model = models[row["day"]][0]
                before = time.perf_counter()
                pnl = float(model.pnl(coordinates))
                repricing_seconds = time.perf_counter()-before
                day = days[row["day"]]
                accepted = max(0., -pnl) if info["encoding_feasible"] else None
                row.update(**info, coordinates=coordinates.tolist(), raw_repriced_pnl=pnl,
                           accepted_margin=accepted,
                           breach=None if accepted is None else bool(day["realized_loss"] > accepted),
                           realized_pnl=day["realized_pnl"], realized_loss=day["realized_loss"],
                           quadratic_gap_to_lattice=None if accepted is None or str(row["bits"]) not in day["lattice"] else info["objective"]-day["lattice"][str(row["bits"])]["objective"],
                           expanded_energy=expanded, energy_cancellation_error=expanded-info["decomposed_energy"],
                           validation_seconds=validation_seconds, repricing_seconds=repricing_seconds,
                           amortized_solve_seconds=solve_seconds/len(rows), batch_solve_seconds=solve_seconds,
                           batch_jobs=len(rows), parameters=supplied)
                if config.get("repair_samples", False):
                    key = row["day"], row["bits"], encoded.config.radius
                    if key not in repairs:
                        repairs[key] = FactorStressRepair(model, encoded)
                    fixed = repairs[key].repair(sample)
                    repaired_margin = max(0., -fixed.pnl)
                    row.update(repaired_margin=repaired_margin, repaired_pnl=fixed.pnl,
                               repaired_breach=bool(day["realized_loss"] > repaired_margin),
                               projected_margin=max(0., -fixed.projectedPnL),
                               repair_converged=fixed.converged, repair_steps=fixed.steps,
                               repaired_integers=fixed.integers.tolist(),
                               repair_seconds=fixed.totalSeconds,
                               repair_stages={name: value for name, value in vars(fixed).items()
                                              if name.endswith("Seconds")})
                    np.save(output/"repaired_samples"/(row["id"]+".npy"), fixed.sample)
                before = time.perf_counter()
                np.save(output/"samples"/(row["id"]+".npy"), sample)
                row["sample_write_seconds"] = time.perf_counter()-before
            batch_record = dict(batch=batch["id"], gpu=gpu, solver=batch["solver"], jobs=len(rows),
                                solve_seconds=solve_seconds, worker_wall_seconds=time.perf_counter()-wall_start,
                                estimated_memory_bytes=estimate, peak_allocated_bytes=torch.cuda.max_memory_allocated(gpu),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(gpu), parameters=supplied)
            writeJson(output/"batches"/f"{batch['id']:04d}.json", dict(batch=batch_record, rows=rows))
            events.put(dict(type="finished", **batch_record))
    except BaseException:
        events.put(dict(type="error", gpu=gpu, traceback=traceback.format_exc()))
        raise


def execute(args, work):
    context = mp.get_context("spawn")
    events, inbox = context.Queue(), context.Queue()
    for batch in work:
        inbox.put(batch)
    for _ in args.devices:
        inbox.put(None)
    config = dict(output=str(args.output), steps=args.steps, runs=args.runs, seed=args.seed, radius=args.radius,
                  repair_samples=getattr(args, "repair_samples", False))
    before = time.perf_counter()
    processes = [context.Process(target=worker, args=(gpu, inbox, events, config)) for gpu in args.devices]
    for process in processes:
        process.start()
    active, ready, complete, count = {}, [], [], 0
    peak_jobs = 0
    try:
        with (args.output/"events.jsonl").open("w", buffering=1) as stream:
            while len(complete) < len(work):
                try:
                    event = events.get(timeout=5)
                except queue.Empty:
                    if any(p.exitcode not in (None, 0) for p in processes) or all(not p.is_alive() for p in processes):
                        raise RuntimeError("GPU workers exited before all results were collected")
                    continue
                stream.write(json.dumps(event)+"\n")
                if event["type"] == "error":
                    raise RuntimeError(event["traceback"])
                if event["type"] == "ready":
                    ready.append(event)
                elif event["type"] == "started":
                    active[event["gpu"]] = event
                    peak_jobs = max(peak_jobs, sum(item["jobs"] for item in active.values()))
                elif event["type"] == "finished":
                    complete.append(event)
                    active.pop(event["gpu"], None)
                    count += event["jobs"]
                status = dict(stage="sweeping", completed_jobs=count, total_jobs=sum(len(b["trials"]) for b in work),
                              completed_batches=len(complete), total_batches=len(work), active=list(active.values()),
                              peak_parallel_jobs=peak_jobs, elapsed_seconds=time.perf_counter()-before)
                writeJson(args.output/"status.json", status)
                if event["type"] == "finished":
                    print(json.dumps(status), flush=True)
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError("worker did not finish cleanly")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    return dict(worker_phase_wall_seconds=time.perf_counter()-before,
                summed_batch_solve_seconds=sum(b["solve_seconds"] for b in complete),
                summed_worker_batch_seconds=sum(b["worker_wall_seconds"] for b in complete),
                peak_parallel_jobs=peak_jobs, workers=ready)


def summarize(output):
    days = json.loads((output/"days.json").read_text())["days"]
    rows = []
    for path in sorted((output/"batches").glob("*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate trial IDs")
    fields = ["id", "date", "bits", "multiplier", "solver", "repeat", "gpu", "batch", "variables", "edges", "penalty",
              "sufficient_penalty", "encoding_feasible", "scenario_feasible", "product_violations", "budget_equation_residual",
              "accepted_margin", "breach", "realized_pnl", "realized_loss",
              "quadratic_gap_to_lattice", "qubo_build_seconds", "amortized_solve_seconds", "validation_seconds",
              "repricing_seconds", "sample_write_seconds", "energy_cancellation_error"]
    with (output/"trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["bits"], row["multiplier"], row["solver"]), []).append(row)
    summary = []
    for (bits, multiplier, solver), group in sorted(grouped.items()):
        valid = [r for r in group if r["encoding_feasible"]]
        summary.append(dict(bits=bits, multiplier=multiplier, solver=solver, trials=len(group), valid=len(valid),
                            missing_margins=len(group)-len(valid), breaches=sum(r["breach"] for r in valid),
                            breach_rate=None if not valid else sum(r["breach"] for r in valid)/len(valid),
                            mean_margin=None if not valid else float(np.mean([r["accepted_margin"] for r in valid])),
                            mean_amortized_solve_seconds=float(np.mean([r["amortized_solve_seconds"] for r in group])),
                            min_edges=min(r["edges"] for r in group), max_edges=max(r["edges"] for r in group),
                            variables=group[0]["variables"]))
    with (output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    coverage = []
    for name in ("linear_analytic", "quadratic_continuous", "exact_repricing_continuous"):
        reference = [next(r for r in day["references"] if r["method"] == name) for day in days]
        coverage.append(dict(method=name, dates=len(days), breaches=sum(r["breach"] for r in reference),
                             mean_margin=float(np.mean([r["margin"] for r in reference]))))
    writeJson(output/"summary.json", dict(groups=summary, references=coverage))
    return dict(trials=len(rows), valid=sum(r["encoding_feasible"] for r in rows),
                summed_qubo_build_seconds=sum(r["qubo_build_seconds"] for r in rows),
                summed_validation_seconds=sum(r["validation_seconds"] for r in rows),
                summed_repricing_seconds=sum(r["repricing_seconds"] for r in rows),
                summed_sample_write_seconds=sum(r["sample_write_seconds"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--expected-stocks", type=int, default=8590)
    parser.add_argument("--window", type=int, default=125)
    parser.add_argument("--ew-lambda", type=float, default=.93)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--radius", type=float, default=3.)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0.]+[10.**i for i in range(-12, 3)])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--devices", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.group = args.group.resolve()
    if any(v <= 0 for v in (args.days, args.window, args.repeats, args.runs, args.steps, args.batch_size)):
        parser.error("counts must be positive")
    for bits in args.bits:
        for multiplier in args.multipliers:
            FactorStressQUBOConfig(bits, args.radius, 1.1, multiplier)
    if len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("devices must be distinct nonnegative GPU indices")
    if not 0 < args.ew_lambda <= 1 or args.seed < 0:
        parser.error("invalid EW decay or seed")
    if (args.output/"timings.json").exists():
        raise FileExistsError("completed experiment already exists")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.reuse_prepared:
        prior = json.loads((args.output/"settings.json").read_text())
        for key in ("window", "ew_lambda", "seed", "radius", "days", "expected_stocks", "bits"):
            if prior[key] != getattr(args, key):
                raise ValueError(f"prepared {key} differs from requested setting")
        if prior["group_sha256"] != hashlib.sha256(args.group.read_bytes()).hexdigest():
            raise ValueError("prepared dataset differs from requested dataset")
        days = json.loads((args.output/"days.json").read_text())["days"]
        timing = json.loads((args.output/"preparation_timing.json").read_text())
        if len(days) != args.days:
            raise ValueError("prepared date count differs from requested count")
    else:
        days, timing = prepare(args)
        writeJson(args.output/"preparation_timing.json", timing)
    settings = {k: str(v) if isinstance(v, (Path, date)) else v for k, v in vars(args).items()}
    settings["group_sha256"] = hashlib.sha256(args.group.read_bytes()).hexdigest()
    settings["dates"] = [d["date"] for d in days]
    settings["penalty_definition"] = "multiplier * 1.1 * objective coefficient absolute-sum bound"
    settings["breach_definition"] = "previous close to evaluation-date close loss > accepted margin; fit excludes evaluation date"
    settings["timing_definition"] = "shared preparation measured once; batch solve time amortized across jobs, not individual latency"
    writeJson(args.output/"settings.json", settings)
    if args.prepare_only:
        return
    for name in ("samples", "batches"):
        (args.output/name).mkdir(exist_ok=True)
    work = batches(args)
    writeJson(args.output/"schedule.json", {"batches": work})
    timing.update(execute(args, work))
    before = time.perf_counter()
    timing.update(summarize(args.output))
    timing["reporting_seconds"] = time.perf_counter()-before
    timing["invocation_wall_seconds"] = time.perf_counter()-started
    timing["total_wall_seconds"] = timing["invocation_wall_seconds"] + (timing["preparation_wall_seconds"] if args.reuse_prepared else 0)
    writeJson(args.output/"timings.json", timing)
    writeJson(args.output/"status.json", dict(stage="complete", **timing))
    print(json.dumps(dict(stage="complete", **timing)), flush=True)


if __name__ == "__main__":
    main()

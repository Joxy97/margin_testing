"""Compare one joint hard-budget QUBO with continuous and exact lattice references.

Run from the repository root with PYTHONPATH=src. The application owns acquisition;
this research CLI uses its public PCA types without generating the old scenario grid.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from margin_engine import MarginApplicationConfig
from risk_state_generator import ReturnsPCAGrid, ReturnsPCAKey
from risk_state_generator.factor_stress_model import FactorStressModel
from margin_calculator.optimization.factor_stress import (
    FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective, solveLattice,
)
from margin_calculator.optimization.factor_stress_reference import (
    solveLinearReference, solveQuadraticReference, solveRepricedReference,
)
from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory


ROOT = Path(__file__).resolve().parents[1]


def writeJson(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def emit(output: Path, value: dict) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)
    writeJson(output/"status.json", value)


def prepareApplication(args, output: Path) -> MarginApplicationConfig:
    source = args.group.resolve()
    # Header/date metadata determines the portfolio and as-of date. Actual price
    # acquisition below goes through MarginEngine, with canonical instrument order.
    header = pd.read_csv(source, nrows=0).columns.tolist()
    tickers = sorted(header[1:])
    dates = pd.to_datetime(pd.read_csv(source, usecols=[header[0]])[header[0]])
    as_of = args.date or (dates.max().date()+timedelta(days=1))
    rng = np.random.default_rng(args.seed)
    weights = rng.uniform(.1, 1., len(tickers))
    weights /= weights.sum()
    portfolio = output/"portfolio.csv"
    with portfolio.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("client_id", "ticker", "weight"))
        writer.writerows(("factor_stress_random", t, format(w, ".17g")) for t, w in zip(tickers, weights))
    document = {
        "marginDate": str(as_of),
        "portfolio": {"csv": "portfolio.csv", "clientId": "factor_stress_random", "cash": "0"},
        "engine": {
            "dataManager": {"cacheType": "lru", "memorySize": 4, "maxMemoryBytes": 268435456},
            "downloadManager": {
                "downloadAlgorithm": "single_request", "downloadParameters": {},
                "providerSelection": "local_first", "providers": {"local_csv": {"type": "local_csv"}},
                "requestParameters": {"locations": [str(source)]}},
            "riskStateGenerator": {"type": "returns_vola_grid", "components": 2,
                "ew_window": args.window, "ew_lambda": args.ew_lambda,
                "scenariosPerComponents": [1, 1], "nZBins": 3},
            "marginCalculator": {"type": "state_aware_greedy"}}}
    (output/"acquisition.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    return MarginApplicationConfig.fromYaml(output/"acquisition.yaml")


def referenceRow(name, result, model, objective, seconds):
    z = result.coordinates
    return dict(method=name, seconds=seconds, coordinates=z.tolist(),
                objective=float(result.objective), quadratic_pnl=float(objective.value(z)),
                repriced_pnl=float(model.pnl(z)), margin=max(0., -float(model.pnl(z))),
                radius=float(np.linalg.norm(z)), lower_bound=result.lowerBound,
                optimality_gap=result.optimalityGap, success=result.success, message=result.message)


def normalizedProblem(problem: QUBOProblem) -> tuple[QUBOProblem, float]:
    """Positive global scaling for solver dynamics, with source scoring retained."""
    magnitude = max(float(np.abs(problem.linear).max(initial=0)),
                    float(np.abs(problem.quadraticBiases).max(initial=0)))
    scale = magnitude or 1.
    return QUBOProblem(problem.linear/scale, problem.quadraticHeads, problem.quadraticTails,
                       problem.quadraticBiases/scale, offset=problem.offset/scale,
                       seedOffset=problem.seedOffset), scale


def run(args) -> None:
    import torch

    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"results.json").exists():
        raise FileExistsError(f"completed results already exist in {output}; choose another --output")
    started = perf_counter()
    app = prepareApplication(args, output)
    data = app.createEngine().getPortfolioMarketData(app.portfolio, app.marginDate)
    key = ReturnsPCAKey(app.portfolio.instruments, args.window, app.marginDate, args.ew_lambda, 2)
    grid = ReturnsPCAGrid.construct(key, data)
    model = FactorStressModel.fromPCAGrid(grid, app.portfolio)
    objective = QuadraticStressObjective(*model.quadraticCoefficients())
    settings = dict(group=str(args.group.resolve()), group_sha256=hashlib.sha256(args.group.read_bytes()).hexdigest(),
                    margin_date=str(app.marginDate), calibration_start=str(grid.calibrationStartDate),
                    calibration_end=str(grid.calibrationEndDate), stocks=len(model.instruments),
                    observations=args.window, ew_lambda=args.ew_lambda, seed=args.seed,
                    radius=args.radius, radius_calibration="illustrative; no target coverage claimed",
                    horizon="one observation interval (daily closes)", bits=args.bits,
                    pca_explained=grid.explained.tolist(), residual_gradient=float(objective.gradient[-1]),
                    device=args.device, torch=torch.__version__, numpy=np.__version__,
                    steps=args.steps, runs=args.runs, repeats=args.repeats,
                    preparation_seconds=perf_counter()-started)
    writeJson(output/"settings.json", settings)
    np.savez(output/"fitted_model.npz", instruments=np.array(model.instruments), exposures=model.exposures,
             center=model.center, directions=model.directions, gradient=objective.gradient,
             hessian=objective.hessian, constant=objective.constant)
    emit(output, dict(stage="prepared", stocks=len(model.instruments), coordinates=model.dimension,
                      calibration_end=str(grid.calibrationEndDate), pca_explained=grid.explained.tolist()))
    references = []
    for name, solve in (("linear_analytic", lambda: solveLinearReference(objective, args.radius)),
                        ("quadratic_continuous", lambda: solveQuadraticReference(objective, args.radius)),
                        ("exact_repricing_continuous", lambda: solveRepricedReference(model, args.radius))):
        before = perf_counter()
        result = solve()
        row = referenceRow(name, result, model, objective, perf_counter()-before)
        references.append(row)
        emit(output, dict(stage="reference", **row))
    # Historical portfolio stresses are repriced separately from the reduced
    # model. This is an in-sample stress check, not a coverage backtest.
    historical_log = (grid.logReturnMean + grid.logReturnScale
                      * (grid.pcaMean + grid.factors@grid.loadings + grid.residuals))
    historical_pnl = np.expm1(historical_log) @ model.exposures
    historical_margin = max(0., -float(historical_pnl.min()))
    rng = np.random.default_rng(args.seed+1)
    directions = rng.normal(size=(4096, model.dimension))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    validation = np.vstack((args.radius*directions,
                            args.radius*directions*rng.random((len(directions), 1))**(1/model.dimension)))
    approximation_error = np.abs(model.pnl(validation)-objective.value(validation))
    validation_summary = dict(points=len(validation), max_absolute_taylor_error=float(approximation_error.max()),
                              mean_absolute_taylor_error=float(approximation_error.mean()),
                              historical_observed_margin=historical_margin)
    rows, sizes, lattice_rows = [], [], []
    for k in args.bits:
        before = perf_counter()
        encoded = FactorStressQUBO.build(objective, FactorStressQUBOConfig(k, args.radius))
        q = encoded.problem
        solver_q, scale = normalizedProblem(q)
        size = dict(bits_per_coordinate=k, scenario_bits=encoded.scenarioBits,
                    product_bits=len(encoded.productPairs), slack_bits=len(encoded.slackWeights),
                    variables=q.variableCount, edges=q.interactionCount,
                    edge_density=q.interactionCount/(q.variableCount*(q.variableCount-1)/2),
                    penalty=encoded.penalty, objective_range_bound=encoded.objectiveRangeBound,
                    global_solver_scale=scale, build_seconds=perf_counter()-before)
        sizes.append(size)
        emit(output, dict(stage="qubo_built", **size))
        before = perf_counter()
        integer_point, lattice_value, evaluations = solveLattice(encoded)
        lattice_seconds = perf_counter()-before
        sample = encoded.encodeIntegers(integer_point)
        info = encoded.diagnostics(sample)
        if not info["encoding_feasible"]:
            raise AssertionError("lattice reference must satisfy every encoded constraint")
        z = encoded.coordinates(sample)
        lattice = dict(bits_per_coordinate=k, coordinates=z.tolist(), integer_coordinates=integer_point.tolist(),
                       quadratic_pnl=lattice_value, repriced_pnl=float(model.pnl(z)),
                       margin=max(0., -float(model.pnl(z))), seconds=lattice_seconds, evaluations=evaluations,
                       expanded_energy=q.energy(sample), stable_energy=info["decomposed_energy"],
                       energy_cancellation_error=q.energy(sample)-info["decomposed_energy"],
                       quadratic_discretization_gap=lattice_value-references[1]["objective"])
        lattice_rows.append(lattice)
        np.save(output/f"lattice_{k}_sample.npy", sample)
        emit(output, dict(stage="lattice_reference", **lattice))
        for solver_name in args.solvers:
            solver = BQMSolverFactory.create(solver_name, {"device": args.device})
            for repeat in range(args.repeats):
                parameters = dict(steps=args.steps, runs=args.runs, run_batch_size=args.runs,
                                  dtype="float64", seed=args.seed+repeat)
                if solver_name == "torch_transverse_route":
                    parameters.update(candidate_interval=max(1, args.steps//8), matrix_format="dense")
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                before = perf_counter()
                result = solver.solve(solver_q, parameters)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = perf_counter()-before
                raw_sample = np.array(result.sample, dtype=np.uint8)
                info = encoded.diagnostics(raw_sample)
                z = encoded.coordinates(raw_sample)
                pnl = float(model.pnl(z))
                row = dict(bits_per_coordinate=k, solver=solver_name, repeat=repeat,
                           seconds=seconds, variables=q.variableCount, edges=q.interactionCount,
                           parameters=parameters, **info, coordinates=z.tolist(),
                           raw_repriced_pnl=pnl,
                           accepted_margin=max(0., -pnl) if info["encoding_feasible"] else None,
                           quadratic_gap_to_lattice=info["objective"]-lattice_value
                               if info["encoding_feasible"] else None,
                           expanded_energy=q.energy(raw_sample),
                           energy_cancellation_error=q.energy(raw_sample)-info["decomposed_energy"])
                rows.append(row)
                np.save(output/f"{solver_name}_{k}_{repeat}_sample.npy", raw_sample)
                emit(output, dict(stage="solver_completed", **row))
                writeJson(output/"partial_results.json", dict(settings=settings, sizes=sizes,
                          references=references, lattice=lattice_rows, solvers=rows,
                          validation=validation_summary))
    results = dict(settings=settings, sizes=sizes, references=references, lattice=lattice_rows,
                   solvers=rows, validation=validation_summary, total_seconds=perf_counter()-started)
    writeJson(output/"results.json", results)
    fields = ["bits_per_coordinate", "solver", "repeat", "variables", "edges", "seconds",
              "encoding_feasible", "scenario_feasible", "product_violations", "budget_equation_residual",
              "accepted_margin", "quadratic_gap_to_lattice", "energy_cancellation_error"]
    with (output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    emit(output, dict(stage="complete", stocks=len(model.instruments), sizes=sizes,
                      accepted_solver_results=sum(r["encoding_feasible"] for r in rows),
                      solver_results=len(rows), seconds=results["total_seconds"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, default=ROOT/"yahoo_equities_grouped/groups/US/group_49_102.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/factor_stress_20260910/results")
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--window", type=int, default=125)
    parser.add_argument("--ew-lambda", type=float, default=.93)
    parser.add_argument("--radius", type=float, default=3.)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--solvers", nargs="+", choices=["torch_sbm", "torch_svl", "torch_transverse_route"],
                        default=["torch_sbm", "torch_svl", "torch_transverse_route"])
    args = parser.parse_args()
    if args.steps < 1 or args.runs < 1 or args.repeats < 1 or args.window < 2:
        parser.error("steps, runs and repeats must be positive; window must be at least two")
    if not 0 < args.ew_lambda <= 1 or args.seed < 0:
        parser.error("ew-lambda must be in (0,1] and seed must be nonnegative")
    for k in args.bits:
        FactorStressQUBOConfig(k, args.radius)
    run(args)


if __name__ == "__main__":
    main()

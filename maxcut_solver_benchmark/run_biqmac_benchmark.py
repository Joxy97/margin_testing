#!/usr/bin/env python3
"""Benchmark Torch SVL and mathematical-programming solvers on Biq Mac MaxCut."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_solver.bqm_solver import TorchSVLBQMSolver


CSV_FIELDS = (
    "timestamp_utc", "instance", "vertices", "edges", "solver", "trial",
    "status", "solve_seconds", "cut", "reference_cut", "reference_kind",
    "quality", "success", "message",
)


@dataclass(frozen=True)
class Graph:
    name: str
    vertices: int
    heads: numpy.ndarray
    tails: numpy.ndarray
    weights: numpy.ndarray


def _graph(path: Path) -> Graph:
    with path.open(encoding="utf-8") as source:
        header = source.readline().split()
        if len(header) != 2:
            raise ValueError(f"invalid Biq Mac header in {path}")
        vertices, edge_count = map(int, header)
        rows = [line.split() for line in source if line.strip()]
    if len(rows) != edge_count:
        raise ValueError(f"{path} declares {edge_count} edges but contains {len(rows)}")
    heads = numpy.asarray([int(row[0]) - 1 for row in rows], dtype=numpy.uint32)
    tails = numpy.asarray([int(row[1]) - 1 for row in rows], dtype=numpy.uint32)
    weights = numpy.asarray([float(row[2]) for row in rows], dtype=numpy.float64)
    if numpy.any(heads >= vertices) or numpy.any(tails >= vertices):
        raise ValueError(f"edge endpoint outside [1,{vertices}] in {path}")
    return Graph(path.name, vertices, heads, tails, weights)


def _cut(graph: Graph, sample: Any) -> float:
    values = numpy.asarray(sample, dtype=numpy.uint8)
    return float(graph.weights[values[graph.heads] != values[graph.tails]].sum())


def _qubo(graph: Graph) -> QUBOProblem:
    linear = numpy.zeros(graph.vertices, dtype=numpy.float64)
    numpy.add.at(linear, graph.heads, -graph.weights)
    numpy.add.at(linear, graph.tails, -graph.weights)
    return QUBOProblem(linear, graph.heads, graph.tails, 2.0 * graph.weights)


def _torch_svl_call(
    solver: TorchSVLBQMSolver,
    problem: QUBOProblem,
    graph: Graph,
    seed: int,
    runs: int,
    steps: int,
) -> tuple[float, float]:
    started = time.perf_counter()
    result = solver.solve(
        problem,
        {
            "runs": runs, "steps": steps, "dt": 0.02, "mass": 0.5,
            "damping": 0.2, "temperature": 0.01,
            "integrator": "weak_order_2", "transverse_field_initial": 0.5,
            "transverse_field_final": 0.0, "problem_scale_initial": 0.1,
            "problem_scale_final": 2.0, "seed": seed, "dtype": "float32",
            "run_batch_size": runs,
        },
    )
    return _cut(graph, result.sample), time.perf_counter() - started


def _torch_svl(graph: Graph, limit: float, seed: int) -> tuple[str, float, str]:
    """Calibrate runs and steps per graph, then spend most of the time budget."""
    solver = TorchSVLBQMSolver("cpu")
    problem = _qubo(graph)
    pilot_runs, pilot_steps = 8, 5000
    pilot_cut, pilot_seconds = _torch_svl_call(
        solver, problem, graph, seed, pilot_runs, pilot_steps
    )
    usable_seconds = max(limit - pilot_seconds - 1.0, pilot_seconds)
    workload_scale = max(2.0, 1.3 * usable_seconds / max(pilot_seconds, 1e-6))
    desired_runs = pilot_runs * math.sqrt(workload_scale)
    runs = max(
        16,
        max(value for value in (8, 16, 32, 64) if value <= min(desired_runs, 64)),
    )
    steps = max(
        pilot_steps,
        min(50_000, int(pilot_steps * workload_scale * pilot_runs / runs)),
    )
    final_cut, final_seconds = _torch_svl_call(
        solver, problem, graph, seed, runs, steps
    )
    message = (
        f"pilot={pilot_runs}x{pilot_steps}/{pilot_seconds:.3f}s; "
        f"calibrated={runs}x{steps}/{final_seconds:.3f}s; "
        f"budget={limit:.3f}s"
    )
    return "feasible", max(pilot_cut, final_cut), message


def _ortools(graph: Graph, limit: float, seed: int) -> tuple[str, float, str]:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x_{i}") for i in range(graph.vertices)]
    y = [model.new_bool_var(f"y_{i}") for i in range(len(graph.heads))]
    for index, (head, tail) in enumerate(zip(graph.heads, graph.tails)):
        difference = model.new_int_var(-1, 1, f"d_{index}")
        model.add(difference == x[int(head)] - x[int(tail)])
        model.add_abs_equality(y[index], difference)
    integer_weights = numpy.rint(graph.weights).astype(numpy.int64)
    if not numpy.allclose(integer_weights, graph.weights):
        raise ValueError("CP-SAT adapter requires integer Biq Mac edge weights")
    model.maximize(sum(int(weight) * variable for weight, variable in zip(integer_weights, y)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed % (2**31 - 1)
    solver.parameters.max_memory_in_mb = 6000
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "no_solution", math.nan, solver.status_name(status)
    sample = [solver.value(variable) for variable in x]
    return ("optimal" if status == cp_model.OPTIMAL else "feasible"), _cut(graph, sample), solver.status_name(status)


def _scip(graph: Graph, limit: float, seed: int) -> tuple[str, float, str]:
    from pyscipopt import Model, quicksum

    model = Model("maxcut")
    model.hideOutput(True)
    model.setRealParam("limits/time", limit)
    model.setRealParam("limits/memory", 6000.0)
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", seed % 2_000_000_000)
    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(graph.vertices)]
    y = [model.addVar(vtype="B", name=f"y_{i}") for i in range(len(graph.heads))]
    for index, (head, tail) in enumerate(zip(graph.heads, graph.tails)):
        left, right = x[int(head)], x[int(tail)]
        model.addCons(y[index] >= left - right)
        model.addCons(y[index] >= right - left)
        model.addCons(y[index] <= left + right)
        model.addCons(y[index] <= 2 - left - right)
    model.setObjective(quicksum(float(w) * v for w, v in zip(graph.weights, y)), "maximize")
    model.optimize()
    if model.getNSols() == 0:
        return "no_solution", math.nan, str(model.getStatus())
    solution = model.getBestSol()
    sample = [round(model.getSolVal(solution, variable)) for variable in x]
    status = str(model.getStatus())
    return ("optimal" if status == "optimal" else "feasible"), _cut(graph, sample), status


def _linear_constraints(graph: Graph) -> tuple[Any, numpy.ndarray, numpy.ndarray]:
    from scipy.sparse import coo_matrix

    edge_count = len(graph.heads)
    row, column, data, lower, upper = [], [], [], [], []
    for edge, (head, tail) in enumerate(zip(graph.heads, graph.tails)):
        y = graph.vertices + edge
        terms = (
            (((y, 1), (int(head), -1), (int(tail), 1)), 0, numpy.inf),
            (((y, 1), (int(head), 1), (int(tail), -1)), 0, numpy.inf),
            (((y, 1), (int(head), -1), (int(tail), -1)), -numpy.inf, 0),
            (((y, 1), (int(head), 1), (int(tail), 1)), -numpy.inf, 2),
        )
        for coefficients, *bounds in terms:
            constraint = len(lower)
            for variable, coefficient in coefficients:
                row.append(constraint); column.append(variable); data.append(coefficient)
            lower.append(bounds[0]); upper.append(bounds[1])
    matrix = coo_matrix((data, (row, column)), shape=(len(lower), graph.vertices + edge_count)).tocsr()
    return matrix, numpy.asarray(lower), numpy.asarray(upper)


def _highs(graph: Graph, limit: float, seed: int) -> tuple[str, float, str]:
    del seed
    from scipy.optimize import Bounds, LinearConstraint, milp

    count = graph.vertices + len(graph.heads)
    objective = numpy.concatenate((numpy.zeros(graph.vertices), -graph.weights))
    matrix, lower, upper = _linear_constraints(graph)
    result = milp(
        objective, integrality=numpy.ones(count), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": limit, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.x is None:
        return "no_solution", math.nan, result.message
    status = "optimal" if result.success else "feasible"
    return status, _cut(graph, numpy.rint(result.x[:graph.vertices])), result.message


def _cplex(graph: Graph, limit: float, seed: int) -> tuple[str, float, str]:
    import cplex

    model = cplex.Cplex()
    model.set_log_stream(None); model.set_error_stream(None); model.set_warning_stream(None); model.set_results_stream(None)
    model.parameters.timelimit.set(limit)
    model.parameters.threads.set(1)
    model.parameters.randomseed.set(seed % 2_100_000_000)
    model.parameters.workmem.set(5500)
    count = graph.vertices + len(graph.heads)
    objective = [0.0] * graph.vertices + graph.weights.tolist()
    model.variables.add(obj=objective, lb=[0.0] * count, ub=[1.0] * count, types="B" * count)
    expressions, senses, rhs = [], [], []
    for edge, (head, tail) in enumerate(zip(graph.heads, graph.tails)):
        y = graph.vertices + edge; head = int(head); tail = int(tail)
        for indices, values, sense, bound in (
            ([y, head, tail], [1, -1, 1], "G", 0), ([y, head, tail], [1, 1, -1], "G", 0),
            ([y, head, tail], [1, -1, -1], "L", 0), ([y, head, tail], [1, 1, 1], "L", 2),
        ):
            expressions.append(cplex.SparsePair(ind=indices, val=values)); senses.append(sense); rhs.append(bound)
    model.linear_constraints.add(lin_expr=expressions, senses="".join(senses), rhs=rhs)
    model.objective.set_sense(model.objective.sense.maximize)
    try:
        model.solve()
    except cplex.exceptions.CplexSolverError as error:
        return "unavailable", math.nan, str(error)
    if not model.solution.is_primal_feasible():
        return "no_solution", math.nan, model.solution.get_status_string()
    sample = model.solution.get_values(0, graph.vertices - 1)
    status_text = model.solution.get_status_string()
    return ("optimal" if "optimal" in status_text.lower() else "feasible"), _cut(graph, numpy.rint(sample)), status_text


SOLVERS: dict[str, Callable[[Graph, float, int], tuple[str, float, str]]] = {
    "torch_svl": _torch_svl, "ortools_cpsat": _ortools, "scip": _scip,
    "highs": _highs, "cplex": _cplex,
}


def run(config_path: Path, resume: bool) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = (config_path.parent / config["data_dir"]).resolve()
    output = (config_path.parent / config["output_dir"]).resolve(); output.mkdir(parents=True, exist_ok=True)
    results_path = output / "raw_results.csv"
    completed = set()
    if results_path.exists():
        if not resume: raise FileExistsError(f"results exist in {output}; pass --resume")
        with results_path.open(newline="", encoding="utf-8") as source:
            completed = {
                (row["instance"], row["solver"], int(row["trial"]))
                for row in csv.DictReader(source)
                if row["status"] != "error"
            }
    elif resume:
        completed = set()
    graphs = [_graph(data / relative) for relative in config["instances"]]
    max_vertices = int(config.get("max_vertices", 100))
    if max(graph.vertices for graph in graphs) > max_vertices:
        raise ValueError(
            f"configured Biq Mac subset exceeds the {max_vertices}-vertex safety cap"
        )
    manifest = dict(config)
    manifest.update(started_utc=datetime.now(timezone.utc).isoformat(), osqp_status="installed_but_not_applicable_to_binary_nonconvex_maxcut", csv_schema=list(CSV_FIELDS))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for graph in graphs:
        reference = config["references"].get(graph.name)
        for solver_name in config["solvers"]:
            trials = config["torch_trials"] if solver_name == "torch_svl" else 1
            for trial in range(trials):
                key = (graph.name, solver_name, trial)
                if key in completed: continue
                started = time.perf_counter()
                try:
                    status, cut, message = SOLVERS[solver_name](graph, float(config["time_limit_seconds"]), int(config["seed"]) + trial * 104729)
                except Exception as error:
                    status, cut, message = "error", math.nan, f"{type(error).__name__}: {error}"
                elapsed = time.perf_counter() - started
                quality = cut / reference if reference is not None and math.isfinite(cut) and reference != 0 else math.nan
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "instance": graph.name,
                    "vertices": graph.vertices, "edges": len(graph.heads), "solver": solver_name, "trial": trial,
                    "status": status, "solve_seconds": elapsed, "cut": cut, "reference_cut": reference or "",
                    "reference_kind": "published_optimum" if reference is not None else "best_observed",
                    "quality": quality, "success": int(reference is not None and math.isfinite(cut) and cut + 1e-7 >= reference),
                    "message": message.replace("\n", " ")[:500],
                }
                header = not results_path.exists() or results_path.stat().st_size == 0
                with results_path.open("a", newline="", encoding="utf-8") as destination:
                    writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
                    if header: writer.writeheader()
                    writer.writerow(row); destination.flush(); os.fsync(destination.fileno())
                completed.add(key)
                print(f"{graph.name} {solver_name} trial={trial} status={status} cut={cut} quality={quality:.6f} time={elapsed:.3f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); run(args.config.resolve(), args.resume)


if __name__ == "__main__": main()

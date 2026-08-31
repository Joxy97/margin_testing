#!/usr/bin/env python3
"""Build Vanguard QUBOs once and benchmark Torch SBM parameter sweeps.

The timed region is deliberately limited to calls to ``BQMSolver.solveMany``.
Market-data acquisition, risk-state generation, QUBO encoding, artifact I/O,
and result decoding are excluded from the reported timings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
from time import perf_counter, strftime
from typing import Any, Iterable, Sequence

import numpy
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from margin_calculator import BQMMarginCalculator  # noqa: E402
from margin_calculator.optimization.optimization_problem.qubo_problem import (  # noqa: E402
    QUBOProblem,
)
from margin_calculator.optimization.optimization_solver.bqm_solver import (  # noqa: E402
    BQMSolverFactory,
)
from margin_engine import MarginApplicationConfig  # noqa: E402
from risk_state_generator import RiskStateGenerationContext  # noqa: E402


DEFAULT_STEPS = (1000, 1500, 2000, 3000, 4000, 5000, 6000, 7500, 9000, 10000)
DEFAULT_RUNS = (16, 24, 32, 40, 48, 64, 80, 96, 128)
ARTIFACT_FORMAT = 1
PROBLEM_SEED_STRIDE = 0x0D1B54A32D192ED03
MAX_TORCH_SEED = (1 << 63) - 1


def log(message: str) -> None:
    """Print one timestamped progress message immediately."""
    print(f"[{strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def atomic_yaml(path: Path, document: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    os.replace(temporary, path)


def atomic_json(path: Path, document: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def config_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_values(risk_state: Any, portfolio: Any) -> numpy.ndarray:
    """Return per-variable portfolio P&L in QUBO variable order."""
    dense_grid = risk_state.returnsVolaGrid
    weights = numpy.fromiter(
        (
            float(portfolio.weights.get(instrument, 0))
            for instrument in dense_grid.instruments
        ),
        dtype=numpy.float64,
        count=len(dense_grid.instruments),
    )
    weighted_returns = weights[:, None] * dense_grid.gridValues[:, :, 0]
    return numpy.ascontiguousarray(
        weighted_returns[dense_grid.validStateMask],
        dtype=numpy.float64,
    )


def save_artifact(
    path: Path,
    problem: QUBOProblem,
    variable_returns: numpy.ndarray,
) -> None:
    """Store one QUBO and the minimal metadata required for margin decoding."""
    if variable_returns.shape != (problem.variableCount,):
        raise ValueError("decode values do not match the QUBO variable count")
    group_sizes = numpy.fromiter(
        (len(group) for group in problem.oneHotGroups),
        dtype=numpy.int32,
        count=len(problem.oneHotGroups),
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        numpy.savez(
            stream,
            linear=problem.linear,
            quadratic_heads=problem.quadraticHeads,
            quadratic_tails=problem.quadraticTails,
            quadratic_biases=problem.quadraticBiases,
            offset=numpy.asarray(problem.offset, dtype=numpy.float64),
            one_hot_group_sizes=group_sizes,
            variable_returns=variable_returns,
        )
    os.replace(temporary, path)


def load_artifact(path: Path) -> tuple[QUBOProblem, numpy.ndarray]:
    """Load one trusted artifact produced by :func:`save_artifact`."""
    with numpy.load(path, allow_pickle=False) as stored:
        group_sizes = numpy.asarray(
            stored["one_hot_group_sizes"], dtype=numpy.int64
        )
        offsets = numpy.empty(len(group_sizes) + 1, dtype=numpy.int64)
        offsets[0] = 0
        numpy.cumsum(group_sizes, out=offsets[1:])
        groups = tuple(
            tuple(range(int(offsets[index]), int(offsets[index + 1])))
            for index in range(len(group_sizes))
        )
        problem = QUBOProblem(
            linear=stored["linear"],
            quadraticHeads=stored["quadratic_heads"],
            quadraticTails=stored["quadratic_tails"],
            quadraticBiases=stored["quadratic_biases"],
            offset=float(stored["offset"]),
            oneHotGroups=groups,
        )
        returns = numpy.ascontiguousarray(
            stored["variable_returns"], dtype=numpy.float64
        )
    if returns.shape != (problem.variableCount,):
        raise ValueError(f"Invalid decode metadata in {path}")
    return problem, returns


def decode_margin(
    problem: QUBOProblem,
    variable_returns: numpy.ndarray,
    sample: Sequence[int],
) -> float:
    """Apply the engine's deterministic invalid-one-hot decoding rule."""
    values = numpy.asarray(sample)
    if values.shape != (problem.variableCount,):
        raise ValueError("solver sample length does not match the QUBO")
    if not numpy.all((values == 0) | (values == 1)):
        raise ValueError("solver returned a non-binary sample")

    portfolio_return = 0.0
    for group in problem.oneHotGroups:
        indices = numpy.fromiter(group, dtype=numpy.int64, count=len(group))
        selected = indices[values[indices] == 1]
        candidates = selected if len(selected) else indices
        chosen = int(candidates[numpy.argmin(variable_returns[candidates])])
        portfolio_return += float(variable_returns[chosen])
    return -portfolio_return


def build_qubos(
    application: MarginApplicationConfig,
    qubo_directory: Path,
    expected_count: int,
    base_config: Path,
) -> dict[str, Any]:
    """Generate and persist the risk-state QUBOs exactly once."""
    qubo_directory.mkdir(parents=True, exist_ok=True)
    existing = list(qubo_directory.iterdir())
    if existing:
        raise FileExistsError(
            f"QUBO directory is not empty and has no valid manifest: {qubo_directory}"
        )

    engine = application.createEngine()
    calculator = engine.marginCalculator
    if not isinstance(calculator, BQMMarginCalculator):
        raise TypeError("configuration must select a BQM margin calculator")

    log("Acquiring market data before QUBO generation (not timed as solve time).")
    request = engine.riskStateGenerator.createDataRequest(
        application.portfolio, application.marginDate
    ).withProviderParameters(engine.configs.downloadManager.requestParameters)
    market_data = engine.getPortfolioMarketData(
        application.portfolio, application.marginDate
    )
    context = RiskStateGenerationContext(
        marketData=market_data,
        dataRequest=request,
        marginDate=application.marginDate,
    )

    artifacts: list[dict[str, Any]] = []
    build_started = perf_counter()
    risk_states = engine.riskStateGenerator.getRiskStates(context)
    for index, risk_state in enumerate(risk_states):
        if index >= expected_count:
            raise RuntimeError(
                f"risk generator produced more than {expected_count} QUBOs"
            )
        problem = calculator.bqmVisitor.createBQM(
            risk_state,
            application.portfolio,
            calculator.modelParameters,
        )
        returns = decode_values(risk_state, application.portfolio)
        artifact_name = f"qubo_{index:03d}.npz"
        artifact_path = qubo_directory / artifact_name
        save_artifact(artifact_path, problem, returns)
        artifact_bytes = artifact_path.stat().st_size
        artifacts.append(
            {
                "index": index,
                "file": artifact_name,
                "variables": problem.variableCount,
                "interactions": problem.interactionCount,
                "numericBytes": problem.numericMemoryBytes,
                "artifactBytes": artifact_bytes,
            }
        )
        log(
            f"QUBO READY [{index + 1}/{expected_count}] "
            f"variables={problem.variableCount:,}, "
            f"interactions={problem.interactionCount:,}, "
            f"file={artifact_bytes / (1024 ** 2):.1f} MiB."
        )

    if len(artifacts) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} QUBOs, generated {len(artifacts)}"
        )
    build_seconds = perf_counter() - build_started
    manifest = {
        "formatVersion": ARTIFACT_FORMAT,
        "baseConfig": str(base_config),
        "baseConfigSha256": config_digest(base_config),
        "marginDate": application.marginDate.isoformat(),
        "quboCount": len(artifacts),
        "buildSeconds": build_seconds,
        "totalNumericBytes": sum(item["numericBytes"] for item in artifacts),
        "totalArtifactBytes": sum(item["artifactBytes"] for item in artifacts),
        "artifacts": artifacts,
    }
    atomic_json(qubo_directory / "manifest.json", manifest)
    log(
        f"All {len(artifacts)} QUBOs stored in {qubo_directory} "
        f"after {build_seconds:.3f}s (excluded from sweep timings)."
    )
    return manifest


def load_or_build_manifest(
    application: MarginApplicationConfig,
    qubo_directory: Path,
    expected_count: int,
    base_config: Path,
) -> dict[str, Any]:
    manifest_path = qubo_directory / "manifest.json"
    if not manifest_path.exists():
        return build_qubos(
            application, qubo_directory, expected_count, base_config
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("formatVersion") != ARTIFACT_FORMAT:
        raise ValueError("stored QUBO artifact format is incompatible")
    if manifest.get("baseConfigSha256") != config_digest(base_config):
        raise ValueError("stored QUBOs were built from a different configuration")
    if manifest.get("marginDate") != application.marginDate.isoformat():
        raise ValueError("stored QUBOs use a different margin date")
    if manifest.get("quboCount") != expected_count:
        raise ValueError(
            f"stored QUBO count is {manifest.get('quboCount')}, "
            f"expected {expected_count}"
        )
    missing = [
        item["file"]
        for item in manifest["artifacts"]
        if not (qubo_directory / item["file"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"stored QUBO artifacts are missing: {missing}")
    log(f"Reusing {expected_count} stored QUBOs from {qubo_directory}.")
    return manifest


def absolute_config_document(base_config: Path) -> dict[str, Any]:
    """Make copied configuration paths independent of the output folder."""
    with base_config.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    base_directory = base_config.parent

    def resolve(value: str) -> str:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (base_directory / path).resolve())

    portfolio = document.get("portfolio", {})
    if "csv" in portfolio:
        portfolio["csv"] = resolve(str(portfolio["csv"]))
    request_parameters = (
        document.get("engine", {})
        .get("downloadManager", {})
        .get("requestParameters", {})
    )
    if "locations" in request_parameters:
        request_parameters["locations"] = [
            resolve(str(location)) for location in request_parameters["locations"]
        ]
    document.pop("backtest", None)
    return document


def write_combination_config(
    base_document: dict[str, Any],
    output_path: Path,
    margin_date: str,
    steps: int,
    runs: int,
    devices: Sequence[str],
    run_batch_size: int | None,
) -> None:
    # YAML round-tripping makes an isolated primitive deep copy.
    document = yaml.safe_load(yaml.safe_dump(base_document))
    document["marginDate"] = margin_date
    solver = document["engine"]["marginCalculator"]["solver"]
    solver["type"] = "torch_sbm"
    solver["constructorParameters"] = {"devices": list(devices)}
    solver_parameters = solver.setdefault("solverParameters", {})
    solver_parameters["steps"] = steps
    solver_parameters["runs"] = runs
    solver_parameters["run_batch_size"] = run_batch_size
    atomic_yaml(output_path, document)


def heartbeat(stop: Event, label: str, interval: int, started: float) -> None:
    while not stop.wait(interval):
        log(f"{label} RUNNING; solve call elapsed={perf_counter() - started:.1f}s.")


def batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_combination(
    solver: Any,
    solver_parameters: dict[str, Any],
    artifacts: Sequence[dict[str, Any]],
    qubo_directory: Path,
    qubo_batch_size: int,
    status_interval: int,
    label: str,
) -> tuple[float, float]:
    """Return greatest margin and solveMany-only wall time."""
    maximum_margin = 0.0
    solve_seconds = 0.0
    artifact_batches = list(batches(artifacts, qubo_batch_size))
    global_problem_start = 0
    solver.beginSeries()
    try:
        for batch_number, artifact_batch in enumerate(artifact_batches, start=1):
            loaded = [
                load_artifact(qubo_directory / item["file"])
                for item in artifact_batch
            ]
            problems = [item[0] for item in loaded]
            variable_returns = [item[1] for item in loaded]
            batch_label = (
                f"{label} batch {batch_number}/{len(artifact_batches)}"
            )
            log(f"{batch_label} SOLVE START ({len(problems)} QUBOs).")
            solve_started = perf_counter()
            stop = Event()
            reporter = Thread(
                target=heartbeat,
                args=(stop, batch_label, status_interval, solve_started),
                daemon=True,
            )
            reporter.start()
            try:
                batch_parameters = dict(solver_parameters)
                batch_parameters["seed"] = (
                    int(solver_parameters.get("seed", 1))
                    + PROBLEM_SEED_STRIDE * global_problem_start
                ) % MAX_TORCH_SEED
                results = solver.solveMany(problems, batch_parameters)
            finally:
                solve_finished = perf_counter()
                stop.set()
                reporter.join()
            batch_solve_seconds = solve_finished - solve_started
            solve_seconds += batch_solve_seconds
            if len(results) != len(problems):
                raise RuntimeError("Torch SBM returned an incomplete QUBO batch")
            for problem, returns, result in zip(
                problems, variable_returns, results
            ):
                maximum_margin = max(
                    maximum_margin,
                    decode_margin(problem, returns, result.sample),
                )
            log(
                f"{batch_label} SOLVE COMPLETE in "
                f"{batch_solve_seconds:.3f}s."
            )
            global_problem_start += len(problems)
    finally:
        solver.endSeries()
    return maximum_margin, solve_seconds


def write_summary(output_root: Path) -> None:
    rows = []
    for result_path in sorted((output_root / "results").glob("*.result.yaml")):
        with result_path.open("r", encoding="utf-8") as stream:
            result = yaml.safe_load(stream)
        timings = result["timings"]
        stem = result_path.name.removesuffix(".result.yaml")
        rows.append(
            {
                "steps": result["steps"],
                "runs": result["runs"],
                "margin_date": result["marginDate"],
                "margin": result["margin"],
                "data_acquisition_seconds": timings[
                    "dataAcquisitionSeconds"
                ],
                "risk_state_generation_seconds": timings[
                    "riskStateGenerationSeconds"
                ],
                "margin_calculation_seconds": timings[
                    "marginCalculationSeconds"
                ],
                "total_seconds": timings["totalSeconds"],
                "config": f"config/{stem}.yaml",
                "result": f"results/{result_path.name}",
            }
        )
    if not rows:
        raise RuntimeError("no sweep results were produced")
    summary_path = output_root / "summary.csv"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, summary_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one date's QUBOs once, persist them, and time only "
            "Torch SBM solveMany calls across a steps/runs sweep."
        )
    )
    parser.add_argument("config", type=Path, help="base application YAML")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=positive_integer,
        default=list(DEFAULT_STEPS),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=positive_integer,
        default=list(DEFAULT_RUNS),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Torch devices; defaults to constructorParameters in the YAML",
    )
    parser.add_argument(
        "--qubo-batch-size",
        type=positive_integer,
        default=8,
        help="number of stored QUBOs loaded per solveMany call",
    )
    parser.add_argument(
        "--run-batch-size",
        type=positive_integer,
        default=16,
        help="Torch trajectories per internal run batch",
    )
    parser.add_argument(
        "--expected-qubos",
        type=positive_integer,
        default=105,
    )
    parser.add_argument(
        "--status-interval",
        type=positive_integer,
        default=60,
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="store the QUBOs but do not execute the parameter sweep",
    )
    parser.add_argument(
        "--limit-combinations",
        type=nonnegative_integer,
        default=0,
        help="testing aid: run at most this many combinations; zero means all",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    base_config = arguments.config.expanduser().resolve()
    output_root = arguments.output.expanduser().resolve()
    if not base_config.is_file():
        raise FileNotFoundError(f"configuration not found: {base_config}")
    output_root.mkdir(parents=True, exist_ok=True)
    config_directory = output_root / "config"
    result_directory = output_root / "results"
    qubo_directory = output_root / "qubos"
    config_directory.mkdir(exist_ok=True)
    result_directory.mkdir(exist_ok=True)

    application = MarginApplicationConfig.fromYaml(base_config)
    calculator_config = application.engine.marginCalculator
    solver_config = getattr(calculator_config, "solver", None)
    if solver_config is None or solver_config.solverType != "torch_sbm":
        raise ValueError("base configuration must select solver type torch_sbm")
    constructor_parameters = dict(solver_config.constructorParameters)
    if arguments.devices:
        devices = tuple(arguments.devices)
        constructor_parameters = {"devices": devices}
    elif "devices" in constructor_parameters:
        devices = tuple(str(item) for item in constructor_parameters["devices"])
    else:
        devices = (str(constructor_parameters.get("device", "auto")),)

    manifest = load_or_build_manifest(
        application,
        qubo_directory,
        arguments.expected_qubos,
        base_config,
    )
    if arguments.build_only:
        log("BUILD-ONLY COMPLETE; no solver calls were made.")
        return 0

    solver = BQMSolverFactory.createBQMSolver(
        "torch_sbm", constructor_parameters
    )
    # Resolve devices now so setup/import time remains outside every result.
    resolved_devices = tuple(solver.devices)
    log(f"Torch SBM devices ready: {', '.join(resolved_devices)}.")
    base_document = absolute_config_document(base_config)
    base_solver_parameters = dict(solver_config.solverParameters)
    artifacts = manifest["artifacts"]
    combinations = [
        (steps, runs)
        for steps in arguments.steps
        for runs in arguments.runs
    ]
    if arguments.limit_combinations:
        combinations = combinations[: arguments.limit_combinations]
    log(
        f"Starting {len(combinations)} combinations over "
        f"{len(artifacts)} stored QUBOs."
    )

    for number, (steps, runs) in enumerate(combinations, start=1):
        stem = f"vanguard_steps_{steps:06d}_runs_{runs:04d}"
        result_path = result_directory / f"{stem}.result.yaml"
        config_path = config_directory / f"{stem}.yaml"
        if result_path.exists():
            log(f"[{number}/{len(combinations)}] SKIP existing {result_path.name}.")
            continue
        write_combination_config(
            base_document,
            config_path,
            application.marginDate.isoformat(),
            steps,
            runs,
            resolved_devices,
            arguments.run_batch_size,
        )
        parameters = dict(base_solver_parameters)
        parameters.update(
            {
                "steps": steps,
                "runs": runs,
                "run_batch_size": min(arguments.run_batch_size, runs),
            }
        )
        label = f"[{number}/{len(combinations)}] steps={steps}, runs={runs}"
        log(f"{label} START.")
        margin, solve_seconds = run_combination(
            solver,
            parameters,
            artifacts,
            qubo_directory,
            arguments.qubo_batch_size,
            arguments.status_interval,
            label,
        )
        result = {
            "steps": steps,
            "runs": runs,
            "marginDate": application.marginDate.isoformat(),
            "margin": float(margin),
            "timings": {
                "dataAcquisitionSeconds": 0.0,
                "riskStateGenerationSeconds": 0.0,
                "marginCalculationSeconds": float(solve_seconds),
                "totalSeconds": float(solve_seconds),
            },
        }
        atomic_yaml(result_path, result)
        write_summary(output_root)
        log(
            f"{label} COMPLETE margin={margin}, "
            f"QUBO solve time={solve_seconds:.3f}s."
        )

    write_summary(output_root)
    log(f"SWEEP COMPLETE. Summary: {output_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; stored QUBOs and completed results were preserved.")
        raise SystemExit(130)

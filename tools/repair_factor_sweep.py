"""Apply deterministic repair to an archived sweep without rerunning solvers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark_factor_stress import writeJson
from sweep_factor_stress import loadModel
from report_factor_sweep import writeCsv
from margin_calculator.optimization.factor_stress import FactorStressQUBO, FactorStressQUBOConfig
from margin_calculator.optimization.factor_stress_repair import FactorStressRepair, FactorStressRepairConfig


def inputDigest(source):
    """Fingerprint all raw numeric results and fits before and after processing."""
    digest = hashlib.sha256()
    for directory, pattern in (("models", "*.npz"), ("samples", "*.npy"), ("batches", "*.json")):
        for path in sorted((source/directory).glob(pattern)):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    for name in ("days.json", "settings.json", "timings.json", "schedule.json"):
        digest.update((source/name).read_bytes())
    return digest.hexdigest()


def summaryRows(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["bits"], row["multiplier"], row["solver"]), []).append(row)
    result = []
    for (bits, multiplier, solver), group in sorted(groups.items()):
        result.append(dict(bits=bits, multiplier=multiplier, solver=solver, trials=len(group),
            raw_feasible=sum(r["raw_encoding_feasible"] for r in group), repaired_feasible=len(group),
            projected_count=sum(r["projection_required"] for r in group),
            projection_only_breaches=sum(r["projected_breach"] for r in group),
            repaired_breaches=sum(r["repaired_breach"] for r in group),
            mean_projected_margin=float(np.mean([r["projected_margin"] for r in group])),
            mean_repaired_margin=float(np.mean([r["repaired_margin"] for r in group])),
            mean_reference_gap=float(np.mean([r["reference_margin_gap"] for r in group])),
            max_reference_gap=max(r["reference_margin_gap"] for r in group),
            mean_repair_seconds=float(np.mean([r["totalSeconds"] for r in group])),
            max_steps=max(r["steps"] for r in group), converged=sum(r["converged"] for r in group)))
    return result


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("repair output must be separate from the raw results directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty repair output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("projected_samples", "samples", "batches"):
        (output/name).mkdir()
    started = perf_counter()
    before = inputDigest(source)
    settings = json.loads((source/"settings.json").read_text())
    days = json.loads((source/"days.json").read_text())["days"]
    raw_timings = json.loads((source/"timings.json").read_text())
    config = FactorStressRepairConfig(args.max_steps, args.tolerance)
    writeJson(output/"settings.json", dict(source=str(source), config=asdict(config),
        scoring="exact exponential P&L; 26-neighbor local descent", input_sha256=before,
        execution="CPU, sequential; precomputed move increments shared per date/resolution; no solution cache"))
    models, repairs, setup_seconds = {}, {}, 0.
    rows = []
    for path in sorted((source/"batches").glob("*.json")):
        batch = json.loads(path.read_text())
        repaired_rows = []
        for raw in batch["rows"]:
            key = raw["day"], raw["bits"]
            if key not in repairs:
                begin = perf_counter()
                if raw["day"] not in models:
                    models[raw["day"]] = loadModel(source/"models"/f"{raw['day']:03d}.npz")
                model, objective = models[raw["day"]]
                # Product/slack layout and radius are independent of penalty.
                # Both repaired states have zero constraint penalty at any P.
                encoding = FactorStressQUBO.build(objective, FactorStressQUBOConfig(raw["bits"], settings["radius"]))
                repairs[key] = FactorStressRepair(model, encoding, config)
                setup_seconds += perf_counter()-begin
            repair = repairs[key]
            sample = np.load(source/"samples"/(raw["id"]+".npy"), allow_pickle=False)
            result = repair.repair(sample)
            np.testing.assert_allclose(result.rawPnL, raw["raw_repriced_pnl"], rtol=0, atol=1e-12)
            projected = not np.array_equal(result.originalIntegers, result.projectedIntegers)
            if projected == raw["scenario_feasible"]:
                raise AssertionError("projection decision disagrees with raw radius validation")
            if not projected and result.rawPnL != result.projectedPnL:
                raise AssertionError("auxiliary-only repair changed P&L")
            encoding = repair.encoding
            model, objective = models[raw["day"]]
            day = days[raw["day"]]
            reference = next(r for r in day["references"] if r["method"] == "exact_repricing_continuous")
            projected_margin, repaired_margin = max(0., -result.projectedPnL), max(0., -result.pnl)
            row = {name: raw[name] for name in ("id", "day", "date", "bits", "multiplier", "solver", "repeat", "variables", "edges")}
            for name, value in vars(result).items():
                if not isinstance(value, np.ndarray):
                    row[name] = value
            row.update(raw_encoding_feasible=raw["encoding_feasible"], raw_scenario_feasible=raw["scenario_feasible"],
                projection_required=projected, raw_integers=result.originalIntegers.tolist(),
                projected_integers=result.projectedIntegers.tolist(), repaired_integers=result.integers.tolist(),
                projection_distance=float(np.linalg.norm(result.projectedIntegers-result.originalIntegers)*repair.scale),
                total_scenario_distance=float(np.linalg.norm(result.integers-result.originalIntegers)*repair.scale),
                projected_margin=projected_margin, repaired_margin=repaired_margin,
                projected_breach=bool(day["realized_loss"] > projected_margin),
                repaired_breach=bool(day["realized_loss"] > repaired_margin), realized_loss=day["realized_loss"],
                reference_margin=reference["margin"], reference_margin_gap=reference["margin"]-repaired_margin,
                quadratic_gap_to_lattice=float(objective.value(result.integers*repair.scale))-day["lattice"][str(raw["bits"])]["objective"],
                improvement_in_margin=repaired_margin-projected_margin,
                raw_amortized_solve_seconds=raw["amortized_solve_seconds"])
            np.save(output/"projected_samples"/(raw["id"]+".npy"), result.projectedSample)
            np.save(output/"samples"/(raw["id"]+".npy"), result.sample)
            repaired_rows.append(row)
        rows.extend(repaired_rows)
        writeJson(output/"batches"/path.name, {"rows": repaired_rows})
        status = dict(stage="repairing", completed=len(rows), total=raw_timings["trials"],
                      elapsed_seconds=perf_counter()-started,
                      converged=sum(r["converged"] for r in rows))
        writeJson(output/"status.json", status)
        if len(rows) % 320 == 0:
            print(json.dumps(status), flush=True)
    if len(rows) != raw_timings["trials"] or len({r["id"] for r in rows}) != len(rows):
        raise AssertionError("repair coverage differs from original sweep")
    if before != inputDigest(source):
        raise AssertionError("raw input archive was modified")
    csv_rows = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
    writeCsv(output/"trials.csv", csv_rows)
    summary = summaryRows(rows)
    writeCsv(output/"summary.csv", summary)
    groups = {}
    for row in rows:
        groups.setdefault((row["bits"], row["multiplier"], row["solver"], row["repeat"]), []).append(row)
    writeCsv(output/"breaches_by_seed.csv", [dict(bits=k[0], multiplier=k[1], solver=k[2], repeat=k[3],
        dates=len(group), projected_breaches=sum(r["projected_breach"] for r in group),
        repaired_breaches=sum(r["repaired_breach"] for r in group)) for k, group in sorted(groups.items())])
    stages = {name: sum(r[name] for r in rows) for name in (
        "decodeSeconds", "projectionSeconds", "initialRepricingSeconds", "auxiliaryRebuildSeconds",
        "improvementSeconds", "finalValidationSeconds", "totalSeconds")}
    result = dict(trials=len(rows), raw_feasible=sum(r["raw_encoding_feasible"] for r in rows),
        repaired_feasible=len(rows), auxiliary_only_scenarios=sum(not r["projection_required"] for r in rows),
        projected_scenarios=sum(r["projection_required"] for r in rows),
        converged=sum(r["converged"] for r in rows),
        projected_breaches=sum(r["projected_breach"] for r in rows),
        repaired_breaches=sum(r["repaired_breach"] for r in rows),
        mean_margin=float(np.mean([r["repaired_margin"] for r in rows])),
        mean_reference_gap=float(np.mean([r["reference_margin_gap"] for r in rows])),
        max_reference_gap=max(r["reference_margin_gap"] for r in rows),
        mean_steps=float(np.mean([r["steps"] for r in rows])), max_steps=max(r["steps"] for r in rows),
        setup_seconds=setup_seconds, stage_seconds=stages, repair_wall_seconds=perf_counter()-started,
        input_sha256=before, raw_inputs_unchanged=True,
        raw_sweep_wall_seconds=raw_timings["total_wall_seconds"])
    writeJson(output/"results.json", result)
    writeJson(output/"status.json", dict(stage="complete", **result))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    run(parser.parse_args())

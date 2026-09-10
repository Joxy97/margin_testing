"""Independently verify repaired samples and summarize their quality and cost."""

import argparse
import json
from pathlib import Path

import numpy as np

from sweep_factor_stress import loadModel
from repair_factor_sweep import inputDigest
from report_factor_sweep import writeCsv
from margin_calculator.optimization.factor_stress import FactorStressQUBO, FactorStressQUBOConfig


def report(output):
    settings = json.loads((output/"settings.json").read_text())
    source = Path(settings["source"])
    raw_settings = json.loads((source/"settings.json").read_text())
    days = json.loads((source/"days.json").read_text())["days"]
    result = json.loads((output/"results.json").read_text())
    raw_summary = json.loads((source/"summary.json").read_text())
    raw_breaches = sum(row["breaches"] for row in raw_summary["groups"])
    if inputDigest(source) != settings["input_sha256"]:
        raise ValueError("raw archive differs from the repair input")
    models, encodings, rows = {}, {}, []
    optimality_checked = set()
    for path in sorted((output/"batches").glob("*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if row["day"] not in models:
                models[row["day"]] = loadModel(source/"models"/f"{row['day']:03d}.npz")
            model, objective = models[row["day"]]
            key = row["day"], row["bits"]
            if key not in encodings:
                encodings[key] = FactorStressQUBO.build(objective, FactorStressQUBOConfig(row["bits"], raw_settings["radius"]))
            encoding = encodings[key]
            raw = np.load(source/"samples"/(row["id"]+".npy"), allow_pickle=False)
            raw_t = encoding.integerCoordinates(raw)
            np.testing.assert_array_equal(raw_t, row["raw_integers"])
            for directory, label, pnl_name, margin_name, breach_name in (
                ("projected_samples", "projected_integers", "projectedPnL", "projected_margin", "projected_breach"),
                ("samples", "repaired_integers", "pnl", "repaired_margin", "repaired_breach")):
                sample = np.load(output/directory/(row["id"]+".npy"), allow_pickle=False)
                if not encoding.diagnostics(sample)["encoding_feasible"]:
                    raise ValueError("saved repaired sample violates constraints")
                coordinates = encoding.coordinates(sample)
                np.testing.assert_array_equal(encoding.integerCoordinates(sample), row[label])
                pnl = float(model.pnl(coordinates))
                np.testing.assert_allclose(pnl, row[pnl_name], rtol=0, atol=1e-12)
                margin = max(0., -pnl)
                np.testing.assert_allclose(margin, row[margin_name], rtol=0, atol=1e-12)
                if (days[row["day"]]["realized_loss"] > margin) != row[breach_name]:
                    raise ValueError("saved breach differs from direct comparison")
            if row["pnl"] > row["projectedPnL"]:
                raise ValueError("improvement worsened P&L")
            if not row["projection_required"]:
                np.testing.assert_array_equal(raw_t, row["projected_integers"])
                if row["rawPnL"] != row["projectedPnL"]:
                    raise ValueError("auxiliary repair changed scenario P&L")
            point = np.array(row["repaired_integers"])
            optimum_key = (*key, *point)
            if row["converged"] and optimum_key not in optimality_checked:
                from itertools import product
                candidates = point+np.array(list(product((-1, 0, 1), repeat=model.dimension)))
                candidates = candidates[np.sum(candidates*candidates, axis=1) <= encoding.latticeRadius**2]
                values = model.pnl(candidates*(encoding.config.radius/encoding.latticeRadius))
                if float(values.min()) < row["pnl"]-settings["config"]["improvementTolerance"]-1e-14:
                    raise ValueError("claimed local optimum has an improving feasible neighbor")
                optimality_checked.add(optimum_key)
            rows.append(row)
    if len(rows) != result["trials"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("repair archive is incomplete or contains duplicate samples")
    grouped = []
    for bits in raw_settings["bits"]:
        for solver in ("torch_sbm", "torch_svl", "torch_transverse_route"):
            group = [r for r in rows if r["bits"] == bits and r["solver"] == solver]
            grouped.append(dict(bits=bits, solver=solver, trials=len(group),
                projected_breaches=sum(r["projected_breach"] for r in group),
                repaired_breaches=sum(r["repaired_breach"] for r in group),
                mean_margin=float(np.mean([r["repaired_margin"] for r in group])),
                mean_gap_bps=10000*float(np.mean([r["reference_margin_gap"] for r in group])),
                max_gap_bps=10000*max(r["reference_margin_gap"] for r in group),
                mean_repair_ms=1000*float(np.mean([r["totalSeconds"] for r in group]))))
    writeCsv(output/"by_solver_resolution.csv", grouped)
    verification = dict(status="passed", projected_samples=len(rows), improved_samples=len(rows),
        local_optima_checked=len(optimality_checked), raw_archive_unchanged=True,
        pnl_absolute_tolerance=1e-12, breaches_recomputed=True)
    (output/"verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    lines = ["# Repair of the Group 1 factor-stress sweep", "",
        f"All **{result['trials']:,} repaired encodings are feasible**, compared with "
        f"{result['raw_feasible']} raw feasible encodings. No GPU solves were rerun. "
        "The original archive is unchanged, verified by its before/after digest.", "",
        f"Auxiliary reconstruction preserved the scenario for {result['auxiliary_only_scenarios']:,} "
        f"samples, including the {result['raw_feasible']:,} already-valid encodings. The remaining "
        f"{result['projected_scenarios']:,} samples required projection onto the integer ball. "
        "Optional local improvement was then applied to every feasible scenario.", "",
        "| Stage | Feasible encodings | Breaches |", "|---|---:|---:|",
        f"| Raw solver output | {result['raw_feasible']:,} / {result['trials']:,} | "
        f"{raw_breaches} among valid margins; {result['trials']-result['raw_feasible']:,} missing |",
        f"| Projection and auxiliary reconstruction | {result['trials']:,} / {result['trials']:,} | {result['projected_breaches']} |",
        f"| Plus feasible local improvement | {result['trials']:,} / {result['trials']:,} | {result['repaired_breaches']} |", "",
        "Breaches are direct comparisons of realized loss against margin. Counts pool repeated "
        f"solver/penalty/seed trials over the same {len(days)} dates, {days[0]['date']} through {days[-1]['date']}. "
        f"They are not {result['trials']:,} independent market observations. Radius 3 remains illustrative.", "",
        "## Quality", "",
        f"{result['converged']:,} of {result['trials']:,} samples reached a local optimum within "
        f"the {settings['config']['maxSteps']:,}-move cap. "
        f"Mean accepted moves: {result['mean_steps']:.2f}; maximum: {result['max_steps']}. "
        "Each move respects the exact integer budget and decreases actual exponential P&L. "
        "Projection itself can reduce loss; auxiliary reconstruction alone never changes it.", "",
        f"Mean final margin: **{100*result['mean_margin']:.6f}%**. "
        f"Mean shortfall relative to the exact-repricing continuous reference: "
        f"**{10000*result['mean_reference_gap']:.4f} basis points**; "
        f"maximum: **{10000*result['max_reference_gap']:.4f} basis points**. "
        "The reference allows continuous coordinates, so these gaps combine discretization "
        "and local-search error. The per-trial quadratic lattice gap is also retained, but "
        "that reference optimizes a different objective (the Taylor approximation).", "",
        "| Bits | Solver | Mean margin | Mean reference gap (bp) | Maximum gap (bp) | Mean repair (ms) |",
        "|---:|---|---:|---:|---:|---:|"]
    for row in grouped:
        lines.append(f"| {row['bits']} | {row['solver']} | {100*row['mean_margin']:.6f}% | "
                     f"{row['mean_gap_bps']:.4f} | {row['max_gap_bps']:.4f} | {row['mean_repair_ms']:.3f} |")
    lines += ["", "## Timing", "",
        f"Additional CPU postprocessing wall time: **{result['repair_wall_seconds']:.3f} seconds**, "
        "including sample I/O, setup, input fingerprinting and report writes. No acquisition, "
        "PCA fitting or GPU sampling was repeated. Precomputed exponential move increments "
        "are shared per date/resolution; solutions are not cached.", "",
        "| Repair work over all samples | Seconds |", "|---|---:|"]
    for name, seconds in result["stage_seconds"].items():
        lines.append(f"| {name} | {seconds:.3f} |")
    lines += ["", f"Reusable setup took {result['setup_seconds']:.3f} seconds. Mean direct repair "
        f"time was {1000*result['stage_seconds']['totalSeconds']/result['trials']:.3f} ms per sample. "
        "Local improvement dominates the repair cost; projection and auxiliary reconstruction "
        "are inexpensive. Timings use a single CPU process with one BLAS thread.", "",
        f"The earlier GPU sweep took {result['raw_sweep_wall_seconds']:.3f} seconds. Adding this "
        f"separate CPU pass gives {result['raw_sweep_wall_seconds']+result['repair_wall_seconds']:.3f} "
        "seconds of sequential measured work across the two runs; this is not a measured "
        "integrated GPU/repair pipeline latency. Independent verification is additional.", "",
        "## Artifacts", "",
        "- [All repaired trials](trials.csv), [penalty/solver summaries](summary.csv), [breaches by seed](breaches_by_seed.csv).",
        "- [Solver/resolution comparison](by_solver_resolution.csv), [timings and totals](results.json), [verification](verification.json).",
        "- [Original sweep](../README.md), [algorithm and commands](../../../docs/benchmarks/factor_stress.md).",
        "", "Verification reloads every projected and improved sample, checks product/slack "
        "consistency and the integer radius, directly reprices the portfolio, recomputes "
        "breaches, checks monotonic improvement, and independently tests every distinct "
        "claimed local optimum using direct neighboring-scenario repricing.", ""]
    (output/"README.md").write_text("\n".join(lines))
    print(json.dumps(verification))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)

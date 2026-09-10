"""Verify a complete rolling sweep and report timings, graph sizes and breaches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sweep_factor_stress import buildTrial, loadModel


def writeCsv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graphSizeRows(rows, bits_values):
    result = []
    for bits in bits_values:
        group = [r for r in rows if r["bits"] == bits]
        first = group[0]
        counts = {}
        for label, selected in (("positive", [r for r in group if r["multiplier"] > 0]),
                                ("zero", [r for r in group if r["multiplier"] == 0])):
            counts[label+"_min_edges"] = min((r["edges"] for r in selected), default=None)
            counts[label+"_max_edges"] = max((r["edges"] for r in selected), default=None)
        result.append(dict(bits=bits, scenario_bits=first["scenario_bits"], product_bits=first["product_bits"],
                           slack_bits=first["slack_bits"], variables=first["variables"], **counts))
    return result


def generate(output):
    settings = json.loads((output/"settings.json").read_text())
    days = json.loads((output/"days.json").read_text())["days"]
    timing = json.loads((output/"timings.json").read_text())
    schedule = json.loads((output/"schedule.json").read_text())["batches"]
    expected = {trial["id"]: trial for batch in schedule for trial in batch["trials"]}
    models = {day["index"]: loadModel(output/"models"/f"{day['index']:03d}.npz") for day in days}
    cache, rows, batches = {}, [], []
    for path in sorted((output/"batches").glob("*.json")):
        result = json.loads(path.read_text())
        batches.append(result["batch"])
        for row in result["rows"]:
            trial = expected.pop(row["id"])
            for field in ("day", "bits", "repeat", "multiplier", "solver", "seed_offset"):
                if trial[field] != row[field]:
                    raise ValueError(f"schedule mismatch: {field}")
            key = row["day"], row["bits"], row["multiplier"]
            model, objective = models[row["day"]]
            if key not in cache:
                cache[key] = buildTrial(objective, row["bits"], row["multiplier"], settings["radius"], 0)[0]
            encoded = cache[key]
            sample = np.load(output/"samples"/(row["id"]+".npy"), allow_pickle=False)
            info = encoded.diagnostics(sample)
            for field in ("encoding_feasible", "scenario_feasible", "product_violations", "budget_equation_residual"):
                if info[field] != row[field]:
                    raise ValueError(f"sample validation mismatch: {field}")
            if encoded.problem.variableCount != row["variables"] or encoded.problem.interactionCount != row["edges"]:
                raise ValueError("graph size mismatch")
            pnl = float(model.pnl(encoded.coordinates(sample)))
            np.testing.assert_allclose(pnl, row["raw_repriced_pnl"], rtol=0, atol=1e-12)
            accepted = max(0., -pnl) if info["encoding_feasible"] else None
            day = days[row["day"]]
            if not day["calibration_end"] < row["date"]:
                raise ValueError("calibration includes evaluation date")
            breach = None if accepted is None else day["realized_loss"] > accepted
            if breach != row["breach"] or (accepted is None) != (row["accepted_margin"] is None):
                raise ValueError("margin or direct breach classification mismatch")
            if accepted is not None:
                np.testing.assert_allclose(accepted, row["accepted_margin"], rtol=0, atol=1e-12)
            for parameter in ("runs", "steps"):
                if row["parameters"][parameter] != settings[parameter]:
                    raise ValueError("trial resource settings differ from requested settings")
            rows.append(row)
    if expected:
        raise ValueError(f"missing {len(expected)} scheduled trials")
    stage_names = sorted({key for day in days for key in day["timings"]})
    stages = [dict(stage=name, total_seconds=sum(day["timings"].get(name, 0) for day in days),
                   mean_per_date_seconds=sum(day["timings"].get(name, 0) for day in days)/len(days))
              for name in stage_names]
    writeCsv(output/"preparation_stages.csv", stages)
    writeCsv(output/"daily_references.csv", [dict(date=day["date"], realized_pnl=day["realized_pnl"],
              realized_loss=day["realized_loss"], method=reference["method"], margin=reference["margin"],
              breach=reference["breach"], reference_seconds=reference["seconds"])
              for day in days for reference in day["references"]])
    grouped = {}
    for row in rows:
        key = row["bits"], row["multiplier"], row["solver"], row["repeat"]
        grouped.setdefault(key, []).append(row)
    series = []
    for (bits, multiplier, solver, repeat), group in sorted(grouped.items()):
        valid = [r for r in group if r["encoding_feasible"]]
        series.append(dict(bits=bits, multiplier=multiplier, solver=solver, repeat=repeat, dates=len(group),
                           valid_dates=len(valid), missing_margins=len(group)-len(valid),
                           breaches=sum(r["breach"] for r in valid),
                           breach_rate=None if not valid else sum(r["breach"] for r in valid)/len(valid),
                           mean_margin=None if not valid else float(np.mean([r["accepted_margin"] for r in valid]))))
    writeCsv(output/"breaches_by_seed.csv", series)
    graph_sizes = graphSizeRows(rows, settings["bits"])
    writeCsv(output/"graph_sizes.csv", graph_sizes)
    verified = dict(status="passed", samples=len(rows), batches=len(batches),
                    valid=sum(r["encoding_feasible"] for r in rows),
                    breaches=sum(r["breach"] is True for r in rows),
                    missing_margins=sum(not r["encoding_feasible"] for r in rows),
                    gpu_indices=sorted({r["gpu"] for r in rows}),
                    complete_coverage_series=sum(s["valid_dates"] == len(days) for s in series))
    (output/"verification.json").write_text(json.dumps(verified, indent=2)+"\n")
    lines = ["# Group 1 factor-stress Cartesian sweep", "",
             f"Completed {len(rows):,} trials on {len(verified['gpu_indices'])} GPUs for all "
             f"{settings['expected_stocks']:,} stocks. Every solve uses {settings['runs']} trajectories "
             f"and {settings['steps']:,} steps. The sweep combines bits {settings['bits']}, "
             f"{len(settings['multipliers'])} penalty multipliers, three solvers, {settings['repeats']} "
             f"seeds and {len(days)} evaluation dates ({days[0]['date']} to {days[-1]['date']}).", "",
             "The fixed random long-only portfolio uses seed 20260910 and total exposure 1. "
             "Each date fits two PCA factors and one portfolio residual direction using 125 "
             "strictly prior return observations with EW decay 0.93. Radius 3 remains illustrative.", "",
             "## Penalties and graph sizes", "",
             f"Tested penalty multipliers: {settings['multipliers']}. Both "
             "constraint penalties use `P = multiplier * 1.1 * objectiveRangeBound`. P&L "
             "coefficients remain fixed. The conservative global-optimum feasibility guarantee "
             "applies only when P exceeds the objective range bound. Smaller values are diagnostic.", "",
             "| Bits/coordinate | Scenario | Product | Slack | Variables | Nonzero-penalty edges | Zero-penalty edges |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in graph_sizes:
        positive = "—" if row["positive_min_edges"] is None else str(row["positive_min_edges"]) if row["positive_min_edges"] == row["positive_max_edges"] else f"{row['positive_min_edges']}–{row['positive_max_edges']}"
        zero = "—" if row["zero_min_edges"] is None else str(row["zero_min_edges"]) if row["zero_min_edges"] == row["zero_max_edges"] else f"{row['zero_min_edges']}–{row['zero_max_edges']}"
        lines.append(f"| {row['bits']} | {row['scenario_bits']} | {row['product_bits']} | {row['slack_bits']} | "
                     f"{row['variables']} | {positive} | {zero} |")
    lines += ["", "Counts are measured after coefficient aggregation. At zero penalty, auxiliary "
              "and slack bits remain declared but are disconnected.", "", "## Feasibility and breaches", "",
              f"**{verified['valid']:,}/{len(rows):,} returned encodings were fully feasible.** "
              f"There were {verified['breaches']} direct breaches among valid margins and "
              f"{verified['missing_margins']:,} missing margins due to invalid encodings. "
              f"{verified['complete_coverage_series']} solver/penalty/resolution/seed series had "
              "a valid margin for every evaluation date.", "",
              "A breach is exactly `realized_loss > margin`, using the preceding available close "
              "to the evaluation-date close. Gains have negative realized loss. No historical "
              "floor, repair, reference fallback or warm-start margin replaces a solver result. "
              "Empty breach fields mean missing margins, not successful coverage. Each row in "
              f"`breaches_by_seed.csv` is one {len(days)}-date series; pooled repeated-date trials are not "
              "independent observations and do not establish tail calibration.", "",
              "| Reference | Dates | Breaches | Mean margin |",
              "|---|---:|---:|---:|"]
    summary = json.loads((output/"summary.json").read_text())
    for row in summary["references"]:
        lines.append(f"| {row['method']} | {row['dates']} | {row['breaches']} | {100*row['mean_margin']:.6f}% |")
    lines += ["", "## Timing", "",
              f"Measured total benchmark wall time: **{timing['total_wall_seconds']:.3f} s**. "
              f"Shared preparation: **{timing['preparation_wall_seconds']:.3f} s**. "
              f"Worker phase, including worker startup and warmup: **{timing['worker_phase_wall_seconds']:.3f} s**. "
              f"Reporting: **{timing['reporting_seconds']:.3f} s**. The shell timer in `run.log` "
              "also includes interpreter/import startup. SSH, transfer and installation are excluded.", "",
              f"Peak concurrency was **{timing['peak_parallel_jobs']} QUBOs**. The sum of "
              f"synchronized batch solve times was {timing['summed_batch_solve_seconds']:.3f} s "
              "across GPUs. Parallel work sums are not wall time. Trial solve times are batch "
              "time divided by 32, explicitly amortized throughput costs rather than individual "
              "request latency. Candidate scoring is included in the solver call. Constraint "
              "validation and exact exponential repricing are measured separately after solving.", "",
              "| Shared stage, all dates | Seconds |", "|---|---:|",
              f"| Metadata and configuration | {timing['metadata_seconds']:.3f} |",
              f"| Prefetch acquisition | {timing['prefetch_seconds']:.3f} |"]
    for row in stages:
        if row["stage"] != "preparation_day_wall_seconds":
            lines.append(f"| {row['stage']} | {row['total_seconds']:.3f} |")
    lines += ["", "| Trial work, summed across workers | Seconds |", "|---|---:|"]
    for key in ("summed_qubo_build_seconds", "summed_validation_seconds", "summed_repricing_seconds", "summed_sample_write_seconds"):
        lines.append(f"| {key} | {timing[key]:.3f} |")
    lines += ["", "Exact lattice and continuous reference timings above are benchmark validation "
              "overhead, not required work for a heuristic-only margin calculation. Per-date "
              "stage details are in `preparation_stages.csv` and `days.json`.", "",
              "## Files", "",
              "- [Grouped sweep results](results/summary.csv) and [all trials](results/trials.csv).",
              "- [Breaches by seed](results/breaches_by_seed.csv), [daily reference margins](results/daily_references.csv).",
              "- [Graph sizes](results/graph_sizes.csv), [timings](results/timings.json), [verification](results/verification.json).",
              "- [Model and encoding guide](../../docs/benchmarks/factor_stress.md).",
              "", "The archive includes fitted models, every binary sample, the fixed portfolio, "
              "configuration, complete schedule and batch records. Verification reconstructs "
              "every QUBO, checks every saved sample, reprices it, and recomputes breaches.", ""]
    (output.parent/"README.md").write_text("\n".join(lines))
    print(json.dumps(verified))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)

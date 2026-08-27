#!/usr/bin/env python3
"""Stream Ocean BQMs through the native margin solver benchmark."""

import argparse
import csv
import re
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import dimod
import numpy as np


LABEL = re.compile(r"x_(\d+)_(\d+)$")
METHOD_NAMES = {
    "greedy_local_search": "Greedy local search",
    "simulated_annealing": "Simulated annealing",
    "simulated_bifurcation": "dSB, 15 steps",
}


def scenario_number(path: Path) -> int:
    return int(path.stem.removeprefix("scenario_"))


def compact_model(source: Path, destination: Path, lambda_one_hot: float):
    prepare_started = time.perf_counter()
    load_started = time.perf_counter()
    with source.open("rb") as handle:
        bqm = dimod.BinaryQuadraticModel.from_file(handle)
    ocean_load_ms = (time.perf_counter() - load_started) * 1000.0
    if bqm.vartype is not dimod.BINARY:
        raise ValueError(f"{source} is not a binary BQM")

    parsed = []
    for label in bqm.variables:
        match = LABEL.fullmatch(str(label))
        if match is None:
            raise ValueError(f"unsupported variable label {label!r} in {source}")
        parsed.append((int(match.group(1)), int(match.group(2)), label))
    parsed.sort()
    variable_order = [item[2] for item in parsed]
    assets = [item[0] for item in parsed]
    if assets != sorted(assets):
        raise ValueError(f"non-contiguous asset labels in {source}")
    counts = np.bincount(assets)
    if len(counts) == 0 or np.any(counts == 0):
        raise ValueError(f"missing one-hot asset group in {source}")
    group_offsets = np.empty(len(counts) + 1, dtype=np.uint64)
    group_offsets[0] = 0
    np.cumsum(counts, out=group_offsets[1:])

    linear, quadratic, offset = bqm.to_numpy_vectors(
        variable_order=variable_order, sort_indices=False
    )
    heads, tails, biases = quadratic
    linear = np.asarray(linear, dtype="<f8")
    portfolio_linear = np.asarray(linear + lambda_one_hot, dtype="<f8")
    heads = np.asarray(heads, dtype="<u4")
    tails = np.asarray(tails, dtype="<u4")
    biases = np.asarray(biases, dtype="<f8")
    with destination.open("wb") as output:
        output.write(b"SBMBQM1\0")
        output.write(struct.pack("<QQQd", len(linear), len(biases), len(counts), offset))
        linear.tofile(output)
        portfolio_linear.tofile(output)
        heads.tofile(output)
        tails.tofile(output)
        biases.tofile(output)
        group_offsets.tofile(output)
    prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
    return ocean_load_ms, prepare_ms, len(linear), len(biases), len(counts)


def write_summary(
    rows, output_dir: Path, dataset_name: str, steps: int,
    greedy_sweeps: int, sa_sweeps: int,
):
    methods = sorted({row["method"] for row in rows})
    summary_rows = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        feasible = [row for row in selected if int(row["one_hot_violations"]) == 0]
        summary_rows.append(
            {
                "method": method,
                "scenarios": len(selected),
                "median_solve_ms": statistics.median(float(row["solve_ms"]) for row in selected),
                "median_total_run_ms": statistics.median(float(row["total_run_ms"]) for row in selected),
                "total_run_time_ms": sum(float(row["total_run_ms"]) for row in selected),
                "mean_relative_gap": statistics.mean(float(row["relative_gap"]) for row in selected),
                "wins": sum(float(row["relative_gap"]) <= 1e-12 for row in selected),
                "raw_feasible_scenarios": sum(int(row["raw_one_hot_violations"]) == 0 for row in selected),
                "decoded_feasible_scenarios": len(feasible),
                "margin_requirement": max((float(row["margin"]) for row in feasible), default=float("nan")),
                "median_repair_ms": statistics.median(float(row["repair_ms"]) for row in selected),
                "maximum_peak_rss_mib": max(float(row["peak_rss_mib"]) for row in selected),
            }
        )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    by_method = {row["method"]: row for row in summary_rows}
    asset_groups = int(rows[0]["assets"])
    preparation_by_scenario = {
        int(row["scenario"]): float(row["prepare_ms"]) for row in rows
    }
    lines = [
        f"# {dataset_name} margin QUBO benchmark",
        "",
        f"All {len(rows) // 3} scenario BQMs were streamed one at a time. dSB used exactly {steps} steps; "
        f"greedy used at most {greedy_sweeps} sweeps and simulated annealing used {sa_sweeps} sweeps. ",
        f"Median Ocean load time was {statistics.median(float(row['ocean_load_ms']) for row in rows):,.2f} ms and median total compact preparation time was "
        f"{statistics.median(preparation_by_scenario.values()):,.2f} ms per scenario. Shared preparation took "
        f"{sum(preparation_by_scenario.values()) / 1000.0:,.2f} s overall and is excluded from solver run time.",
        "Every raw sample is passed through the same deterministic one-hot decoder. Margin is the negative portfolio return of that feasible decoded sample.",
        "",
        "| Method | Median solve ms | Median decode ms | Median total ms | Total all scenarios s | Mean relative objective gap | Best-energy wins | Raw feasible | Margin requirement | Peak RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["greedy_local_search", "simulated_annealing", "simulated_bifurcation"]:
        row = by_method[method]
        lines.append(
            f"| {METHOD_NAMES[method]} | {float(row['median_solve_ms']):,.2f} | {float(row['median_repair_ms']):,.2f} | "
            f"{float(row['median_total_run_ms']):,.2f} | {float(row['total_run_time_ms']) / 1000.0:,.2f} | "
            f"{float(row['mean_relative_gap']):.6%} | {row['wins']}/{row['scenarios']} | "
            f"{row['raw_feasible_scenarios']}/{row['scenarios']} | {float(row['margin_requirement']):.6%} | "
            f"{float(row['maximum_peak_rss_mib']):,.1f} |"
        )
    lines += [
        "",
        "Relative objective gap is measured after one-hot decoding against the lowest energy observed for the same scenario; it is not a certified optimality gap.",
        "The reported margin requirement is the maximum feasible scenario loss for each solver.",
        f"Median raw one-hot violations were "
        f"{statistics.median(int(row['raw_one_hot_violations']) for row in rows if row['method'] == 'greedy_local_search'):.0f} for greedy, "
        f"{statistics.median(int(row['raw_one_hot_violations']) for row in rows if row['method'] == 'simulated_annealing'):.0f} for SA, and "
        f"{statistics.median(int(row['raw_one_hot_violations']) for row in rows if row['method'] == 'simulated_bifurcation'):.0f} for dSB out of {asset_groups:,} groups; "
        "therefore the 15-step dSB result is strongly dependent on the decoder.",
        "Raw per-scenario measurements are in `results.csv`; aggregates are in `summary.csv`.",
        "",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("build/bqm_margin_benchmark"))
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--greedy-sweeps", type=int, default=5)
    parser.add_argument("--sa-sweeps", type=int, default=8)
    parser.add_argument("--lambda-one-hot", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.steps != 15:
        raise ValueError("this experiment requires dSB steps fixed to 15")

    paths = sorted(args.input_dir.glob("scenario_*.bqm"), key=scenario_number)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"no scenario BQMs found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "scenario", "variables", "interactions", "assets", "ocean_load_ms", "prepare_ms", "method",
        "solve_ms", "energy", "reference_energy", "relative_gap", "portfolio_return",
        "margin", "raw_one_hot_violations", "repair_ms", "total_run_ms", "one_hot_violations",
        "selected_variables", "peak_rss_mib", "configuration",
    ]
    rows = []
    results_path = args.output_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for position, path in enumerate(paths, 1):
            scenario = scenario_number(path)
            with tempfile.NamedTemporaryFile(suffix=".sbmbqm") as temporary:
                ocean_load_ms, prepare_ms, variables, interactions, assets = compact_model(
                    path, Path(temporary.name), args.lambda_one_hot
                )
                completed = subprocess.run(
                    [
                        str(args.executable.resolve()), temporary.name, str(args.steps),
                        str(args.greedy_sweeps), str(args.sa_sweeps), str(20260827 + scenario),
                    ],
                    check=True, text=True, capture_output=True,
                )
            parsed = []
            for line in completed.stdout.splitlines():
                method, solve_ms, energy, portfolio_return, margin, raw_violations, repair_ms, selected, rss, configuration = line.split("\t")
                parsed.append(
                    {
                        "scenario": scenario, "variables": variables,
                        "interactions": interactions, "assets": assets,
                        "ocean_load_ms": ocean_load_ms, "prepare_ms": prepare_ms,
                        "method": method, "solve_ms": float(solve_ms), "energy": float(energy),
                        "portfolio_return": float(portfolio_return), "margin": float(margin),
                        "raw_one_hot_violations": int(raw_violations), "repair_ms": float(repair_ms),
                        "total_run_ms": float(solve_ms) + float(repair_ms),
                        "one_hot_violations": 0, "selected_variables": int(selected),
                        "peak_rss_mib": float(rss), "configuration": configuration,
                    }
                )
            reference = min(row["energy"] for row in parsed)
            denominator = max(abs(reference), 1.0)
            for row in parsed:
                row["reference_energy"] = reference
                row["relative_gap"] = max(0.0, (row["energy"] - reference) / denominator)
                writer.writerow(row)
                rows.append(row)
            output.flush()
            print(
                f"[{position}/{len(paths)}] scenario {scenario}: n={variables}, m={interactions}, "
                + ", ".join(
                    f"{row['method']}={row['solve_ms']:.1f}ms/E={row['energy']:.6g}/raw_viol={row['raw_one_hot_violations']}"
                    for row in parsed
                ),
                flush=True,
            )
    write_summary(
        rows, args.output_dir, args.input_dir.parent.name, args.steps,
        args.greedy_sweeps, args.sa_sweeps,
    )


if __name__ == "__main__":
    main()

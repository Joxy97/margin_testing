"""Verify archived samples and render a factor-stress benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from risk_state_generator.factor_stress_model import FactorStressModel
from margin_calculator.optimization.factor_stress import (
    FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective,
)


def verify(directory: Path) -> dict:
    result = json.loads((directory/"results.json").read_text())
    with np.load(directory/"fitted_model.npz", allow_pickle=False) as data:
        model = FactorStressModel(tuple(data["instruments"]), data["exposures"], data["center"], data["directions"])
        objective = QuadraticStressObjective(float(data["constant"]), data["gradient"], data["hessian"])
    problems = {k: FactorStressQUBO.build(objective, FactorStressQUBOConfig(k, result["settings"]["radius"]))
                for k in result["settings"]["bits"]}
    checked = 0
    for row in result["solvers"]:
        encoded = problems[row["bits_per_coordinate"]]
        sample = np.load(directory/f"{row['solver']}_{row['bits_per_coordinate']}_{row['repeat']}_sample.npy",
                         allow_pickle=False)
        diagnostics = encoded.diagnostics(sample)
        for key in ("product_violations", "budget_equation_residual", "integer_squared_radius",
                    "integer_budget", "scenario_feasible", "encoding_feasible"):
            if diagnostics[key] != row[key]:
                raise ValueError(f"archived {key} does not match sample")
        np.testing.assert_allclose(encoded.coordinates(sample), row["coordinates"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(float(model.pnl(encoded.coordinates(sample))), row["raw_repriced_pnl"],
                                   rtol=0, atol=1e-13)
        if encoded.problem.variableCount != row["variables"] or encoded.problem.interactionCount != row["edges"]:
            raise ValueError("archived variable/edge counts do not match rebuilt QUBO")
        if not diagnostics["encoding_feasible"] and row["accepted_margin"] is not None:
            raise ValueError("an invalid encoding must not publish an accepted margin")
        for parameter in ("steps", "runs"):
            if row["parameters"][parameter] != result["settings"][parameter]:
                raise ValueError(f"trial {parameter} differs from requested settings")
        checked += 1
    for row in result["lattice"]:
        encoded = problems[row["bits_per_coordinate"]]
        sample = np.load(directory/f"lattice_{row['bits_per_coordinate']}_sample.npy", allow_pickle=False)
        if not encoded.diagnostics(sample)["encoding_feasible"]:
            raise ValueError("archived lattice solution is infeasible")
        np.testing.assert_allclose(float(objective.value(encoded.coordinates(sample))), row["quadratic_pnl"],
                                   rtol=0, atol=1e-13)
        np.testing.assert_allclose(float(model.pnl(encoded.coordinates(sample))), row["repriced_pnl"],
                                   rtol=0, atol=1e-13)
        checked += 1
    record = dict(verified_samples=checked, status="passed",
                  reference_margin_with_historical_floor=max(result["references"][2]["margin"],
                                                             result["validation"]["historical_observed_margin"]))
    (directory/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    return result


def report(directory: Path, result: dict) -> None:
    settings = result["settings"]
    exact = next(r for r in result["references"] if r["method"] == "exact_repricing_continuous")
    historical = result["validation"]["historical_observed_margin"]
    valid = sum(r["encoding_feasible"] for r in result["solvers"])
    lines = ["# Joint factor stress: 102-stock experiment", "",
             "The hard-budget QUBO and the continuous/lattice references are implemented. "
             f"In the remote GPU experiment, {valid}/{len(result['solvers'])} returned heuristic "
             "solutions satisfied every product and budget constraint. No repair or reference "
             "fallback was applied to the heuristic samples.", "",
             "## Requested remote run", "",
             f"Group 49: {settings['stocks']} stocks; a seeded random long-only portfolio "
             f"with weights summing to 1 (seed {settings['seed']}). As-of {settings['margin_date']}; "
             f"125 return observations from {settings['calibration_start']} through {settings['calibration_end']}, "
             "EW decay 0.93. Two PCA factors plus one portfolio residual coordinate. "
             "Radius 3 is illustrative, with no VaR/ES or coverage claim.", "",
             f"Tesla V100-SXM2-32GB, cuda:0, Torch {settings['torch']}, float64. "
             f"**runs={settings['runs']}, steps={settings['steps']}**, three seeds per solver/resolution. "
             f"The measured benchmark interval was {result['total_seconds']:.3f} seconds. "
             "It includes preparation, references, solves, scoring, validation and result writes "
             "after library imports; excludes SSH and environment setup. GPU solves synchronize "
             "before and after timing. First-use initialization is included, so small timing "
             "differences should not be interpreted as steady-state performance.", "",
             "| Bits/coordinate | Scenario bits | Product bits | Slack bits | Variables | Edges |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in result["sizes"]:
        lines.append(f"| {row['bits_per_coordinate']} | {row['scenario_bits']} | {row['product_bits']} | "
                     f"{row['slack_bits']} | {row['variables']} | {row['edges']:,} |")
    lines += ["", "Edges count unique nonzero off-diagonal QUBO interactions, after aggregation. "
              "All three graphs are complete: squaring one global linearized budget couples every bit.", "",
              "| Bits/coordinate | Solver | Mean solve seconds | Fully valid / trials | Scenario inside ball / trials |",
              "|---:|---|---:|---:|---:|"]
    for k in settings["bits"]:
        for name in ("torch_sbm", "torch_svl", "torch_transverse_route"):
            rows = [r for r in result["solvers"] if r["bits_per_coordinate"] == k and r["solver"] == name]
            if rows:
                lines.append(f"| {k} | {name} | {np.mean([r['seconds'] for r in rows]):.3f} | "
                             f"{sum(r['encoding_feasible'] for r in rows)}/{len(rows)} | "
                             f"{sum(r['scenario_feasible'] for r in rows)}/{len(rows)} |")
    lines += ["", "The final column checks only the decoded scenario radius. It does not mean the "
              "auxiliary products and slack are correct. Invalid full encodings have null accepted "
              "margins in the results. These observations describe these solver settings, not a proof "
              "that no heuristic can solve the model.", "", "## References", "",
              "| Method | Exact repriced stress loss | Reference seconds |",
              "|---|---:|---:|"]
    for row in result["references"]:
        lines.append(f"| {row['method']} | {100*row['margin']:.6f}% | {row['seconds']:.6f} |")
    for row in result["lattice"]:
        lines.append(f"| Exact quadratic lattice, {row['bits_per_coordinate']} bits/coordinate | "
                     f"{100*row['margin']:.6f}% | {row['seconds']:.6f} |")
    lines += ["", "Every row is repriced with the original exponential log-return transformation. "
              "The linear, quadratic and lattice methods optimize their stated surrogate; only the "
              "exact-repricing continuous method optimizes that exponential P&L directly. "
              "The continuous quadratic and exact-repricing solutions have convex lower-bound "
              "certificates with reported gaps at floating-point precision.", "",
              f"The 8-bit lattice loss differs from the exact-repricing continuous loss by "
              f"{10000*abs(exact['margin']-result['lattice'][-1]['margin']):.4f} basis points "
              "of portfolio exposure. The maximum absolute Taylor error at 8,192 sampled "
              f"interior/boundary points was {10000*result['validation']['max_absolute_taylor_error']:.4f} "
              "basis points; this is a sampled diagnostic, not a uniform error bound.", "",
              f"**The worst observed loss in the fitting window was {100*historical:.4f}%, "
              f"above the {100*exact['margin']:.4f}% radius-3 reduced-model stress loss.** "
              f"Taking their maximum as a historical stress floor gives {100*max(historical,exact['margin']):.4f}%. "
              "That comparison shows why the illustrative radius must be calibrated and why "
              "omitted residual/sector/tail scenarios need separate validation. It is not an "
              "out-of-sample margin coverage result.", "",
              "## Interpretation and artifacts", "",
              "The conversion removes the 105-QUBO loop and dramatically reduces variable count, "
              "but the expanded hard penalty is dense and poorly conditioned. Maximum absolute "
              "coefficients grow from approximately 1.46e3 at 4 bits to 1.08e8 at 8 bits for "
              "this portfolio, while the source P&L is of order 1e-2. Direct integer feasibility "
              "and unexpanded energy checks are retained alongside ordinary float64 QUBO energy. "
              "The continuous reference is the useful baseline for this three-dimensional model.", "",
              "- [GPU summary CSV](remote_results/summary.csv), [complete GPU results](remote_results/results.json), "
              "[sample verification](remote_results/verification.json), [GPU log](remote_run.log).",
              "- [Initial CPU results](results/results.json): 64 runs, 2,000 steps, three seeds; "
              "0/27 fully valid returned encodings.",
              "- [Longer CPU check](longer_check/results.json): smallest model, 256 runs, 20,000 steps; "
              "0/3 fully valid returned encodings.",
              "- [Model, conversion and limitations](../../docs/benchmarks/factor_stress.md).",
              "- `remote_verification.txt` records remote source/data hashes and idle GPU state after completion.",
              "", "## Reproduce", "", "```bash",
              "PYTHONPATH=src python tools/benchmark_factor_stress.py \\",
              "  --group yahoo_equities_grouped/groups/US/group_49_102.csv \\",
              "  --date 2026-09-08 --bits 4 6 8 --radius 3 \\",
              "  --runs 16 --steps 10000 --repeats 3 --device cuda:0 \\",
              "  --output experiments/factor_stress_20260910/remote_results",
              "PYTHONPATH=src python tools/report_factor_stress.py experiments/factor_stress_20260910/remote_results",
              "```", "", "Use a new output directory for another run. GPU benchmarking used the existing "
              "V100-compatible `/workspace/margin-sweep-venv` on the requested instance.", ""]
    (directory.parent/"README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    report(args.directory, result)
    print(json.dumps(dict(verified=True, trials=len(result["solvers"]), report=str(args.directory.parent/"README.md"))))


if __name__ == "__main__":
    main()

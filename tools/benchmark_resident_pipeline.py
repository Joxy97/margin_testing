"""Compare host and resident execution over deterministic portfolio/window sizes.

Each measurement runs in a fresh process so peak RSS is comparable. GPU traces
are optional; small CPU/greedy reference checks precede interpretation of speed.
"""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

import numpy
import pandas
import yaml


def positive_sizes(value):
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from error
    if not result or min(result) < 2:
        raise argparse.ArgumentTypeError("assets and windows must be at least two")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--assets", type=positive_sizes, default=[16, 64])
    parser.add_argument("--windows", type=positive_sizes, default=[16, 60])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--correlated", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.runs, args.repeats, args.threads) < 1:
        parser.error("steps, runs, repeats and threads must be positive")
    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(repository / "src"),
                       OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                       MKL_NUM_THREADS=str(args.threads))
    results = []
    for assets in args.assets:
        for window in args.windows:
            case = root / f"assets_{assets}_window_{window}"
            case.mkdir(exist_ok=True)
            rng = numpy.random.default_rng(2400 + assets + window)
            names = [f"A{index:06d}" for index in range(assets)]
            dates = pandas.date_range("2024-01-01", periods=window + 2)
            prices = pandas.DataFrame(100 * numpy.exp(numpy.cumsum(
                rng.normal(0, .015, (window + 2, assets)), axis=0)), columns=names)
            prices.insert(0, "date", dates)
            prices.to_csv(case / "prices.csv", index=False)
            config = {
                "marginDate": str(dates[-1].date()),
                "portfolio": {"weights": {name: -7 if index % 2 else 10 for index, name in enumerate(names)}},
                "engine": {
                    "downloadManager": {"providers": {"local": "local_csv"},
                                        "requestParameters": {"location": "prices.csv"}},
                    "riskStateGenerator": {"type": "correlated_returns_vola_grid" if args.correlated else "returns_vola_grid",
                        "ew_window": window, "components": 1, "scenariosPerComponents": [3],
                        "nZBins": 3, "allowEmptyBinFallback": True},
                    "marginCalculator": {"type": "bqm", "comparison": {"type": "state_aware_greedy"},
                        "solver": {"type": "torch_sbm", "constructorParameters": {"device": "cpu"},
                            "solverParameters": {"steps": args.steps, "runs": args.runs, "seed": 24,
                                                 "dtype": "float64", "run_batch_size": min(8, args.runs)}},
                        "executionPolicy": {"type": "batch", "batchSize": 2, "maxBatchBytes": 256 * 1024 * 1024}},
                },
            }
            measurements = {}
            # CPU reference, conventional pipeline on the selected device, then
            # resident pipeline on that same device. All configurations are archived.
            for mode in ("cpu_reference", "host", "resident"):
                config["engine"].pop("numericalExecution", None)
                device = "cpu" if mode == "cpu_reference" else args.device
                config["engine"]["marginCalculator"]["solver"]["constructorParameters"]["device"] = device
                if mode == "resident":
                    config["engine"]["numericalExecution"] = {"type": "torch", "device": device}
                path = case / f"{mode}.yaml"
                path.write_text(yaml.safe_dump(config))
                values = []
                for repeat in range(1 if mode == "cpu_reference" else args.repeats):
                    output = case / f"{mode}_{repeat}.json"
                    command = [sys.executable, str(repository / "tools/profile_margin_pipeline.py"),
                               str(path), "--output", str(output)]
                    if args.trace:
                        command += ["--trace", str(case / f"{mode}_{repeat}.trace.json")]
                    subprocess.run(command, cwd=repository, env=environment, check=True)
                    values.append(json.loads(output.read_text()))
                measurements[mode] = values
            expected = measurements["cpu_reference"][0]["comparisonMargins"]["greedy"]
            for mode, values in measurements.items():
                for value in values:
                    if not math.isclose(value["comparisonMargins"]["greedy"], expected, rel_tol=1e-9, abs_tol=1e-10):
                        raise RuntimeError(f"Greedy reference mismatch for {case.name}, {mode}")
            host_time = median(value["synchronizedWallSeconds"] for value in measurements["host"])
            resident_time = median(value["synchronizedWallSeconds"] for value in measurements["resident"])
            results.append({"assets": assets, "window": window, "device": args.device,
                "correlated": args.correlated, "traced": args.trace,
                "hostSeconds": host_time, "residentSeconds": resident_time,
                "hostOverResidentRatio": host_time / resident_time,
                "greedyReferenceMargin": expected,
                "cpuReferenceMargin": measurements["cpu_reference"][0]["margin"],
                "residentMargin": measurements["resident"][0]["margin"],
                "measurements": measurements})
            (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(root / "summary.json")


if __name__ == "__main__":
    main()

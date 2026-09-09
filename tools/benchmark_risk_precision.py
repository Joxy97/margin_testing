"""Compare float64/float32 PCA and complete resident margin calculations.

PYTHONPATH=src python tools/benchmark_risk_precision.py --device cuda:0 --output /tmp/risk.json
"""

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import median
import tempfile
from time import perf_counter

import numpy
import pandas
import torch
import yaml

from margin_engine import MarginApplicationConfig, MarginEngine
from risk_state_generator import NumpyPCABackend, TorchPCABackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assets", nargs="+", type=int, default=[64, 1024])
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--reuse-engine", action="store_true", help="retain acquisition caches between calculations")
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.assets, args.window) < 3 or min(args.steps, args.runs, args.repeats, args.warmups) < 1:
        parser.error("assets/window must be >= 3; steps/runs/repeats/warmups must be positive")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(call):
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        result = call()
        synchronize()
        return result, perf_counter() - start, (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)

    output = {"arguments": vars(args) | {"output": str(args.output), "trace_directory": str(args.trace_directory)},
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": numpy.__version__, "matmulPrecision": torch.get_float32_matmul_precision(),
        "sourceHashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("src").rglob("*.py"))},
        "pca": [], "pipeline": []}
    for assets in args.assets:
        rng = numpy.random.default_rng(2400 + assets + args.window)
        values = rng.normal(size=(args.window, assets))
        weights = .94 ** numpy.arange(args.window - 1, -1, -1)
        weights /= weights.sum()
        expected = NumpyPCABackend().fit(values, weights, 2)
        for dtype in ("float64", "float32"):
            backend = TorchPCABackend(str(device), dtype)
            for _ in range(args.warmups):
                backend.fitResident(values, weights, 2)
            durations, peaks = [], []
            for _ in range(args.repeats):
                fit, elapsed, peak = measure(lambda: backend.fitResident(values, weights, 2))
                durations.append(elapsed)
                peaks.append(peak)
                host = fit.toHost()
                fit_bytes = sum(v.numel() * v.element_size() for v in vars(fit).values())
                del fit
            errors = {name: float(numpy.max(numpy.abs(getattr(host, name) - getattr(expected, name)))) for name in vars(expected)}
            # Compare the subspace, since equivalent eigenvector signs can change.
            numpy.testing.assert_allclose(host.loadings.T @ host.loadings,
                expected.loadings.T @ expected.loadings, rtol=2e-4, atol=2e-5)
            numpy.testing.assert_allclose(host.lambdas, expected.lambdas, rtol=2e-5, atol=2e-5)
            output["pca"].append({"assets": assets, "dtype": dtype, "seconds": durations,
                "medianSeconds": median(durations), "peakAllocatedBytes": peaks, "fitBytes": fit_bytes,
                "maxAbsoluteErrors": errors})

        with tempfile.TemporaryDirectory(prefix="margin-risk-precision-") as directory:
            names = [f"A{i:06d}" for i in range(assets)]
            dates = pandas.date_range("2024-01-01", periods=args.window + 2)
            prices = pandas.DataFrame(100 * numpy.exp(rng.normal(0, .015, (len(dates), assets)).cumsum(axis=0)), columns=names)
            prices.insert(0, "date", dates)
            prices.to_csv(Path(directory) / "prices.csv", index=False)
            for correlated in (False, True):
                config = {"marginDate": str(dates[-1].date()),
                    "portfolio": {"weights": {name: -7 if i % 2 else 10 for i, name in enumerate(names)}},
                    "engine": {"downloadManager": {"providers": {"local": "local_csv"},
                        "requestParameters": {"location": "prices.csv"}},
                        "riskStateGenerator": {"type": "correlated_returns_vola_grid" if correlated else "returns_vola_grid",
                            "ew_window": args.window, "components": 1, "scenariosPerComponents": [3],
                            "nZBins": 3, "allowEmptyBinFallback": True},
                        "marginCalculator": {"type": "state_aware_greedy"}}}
                reference = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport().margin
                config["engine"]["marginCalculator"] = {"type": "bqm", "comparison": {"type": "state_aware_greedy"},
                    "solver": {"type": "torch_sbm", "constructorParameters": {"device": str(device)},
                        "solverParameters": {"steps": args.steps, "runs": args.runs, "seed": 24,
                            "dtype": "float32", "run_batch_size": min(8, args.runs)}},
                    "executionPolicy": {"type": "batch", "batchSize": 2, "maxBatchBytes": 256 * 1024 * 1024}}
                applications, measurements = {}, {}
                for dtype in ("float64", "float32"):
                    config["engine"]["numericalExecution"] = {"type": "torch", "device": str(device), "dtype": dtype}
                    application = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory)
                    if args.reuse_engine:
                        engine = MarginEngine(application.engine)
                        applications[dtype] = lambda engine=engine, application=application: engine.generateReport(
                            application.portfolio, application.marginDate)
                    else:
                        applications[dtype] = application.generateReport
                    measurements[dtype] = {"assets": assets, "correlated": correlated, "dtype": dtype,
                        "configuration": json.loads(json.dumps(config)), "cpuGreedyMargin": reference, "runs": []}
                    for _ in range(args.warmups):
                        applications[dtype]()
                # Alternate precision order to reduce clock/temperature bias.
                for repeat in range(args.repeats):
                    for dtype in (("float64", "float32") if repeat % 2 == 0 else ("float32", "float64")):
                        report, elapsed, peak = measure(applications[dtype])
                        greedy = report.comparisonMargins["greedy"]
                        if not math.isclose(greedy, reference, rel_tol=2e-5, abs_tol=2e-5):
                            raise AssertionError(f"Greedy mismatch: {assets}, {correlated}, {dtype}: {greedy} vs {reference}")
                        if not 0 <= report.margin <= greedy + 2e-5 * max(1, abs(greedy)):
                            raise AssertionError("BQM margin outside independent greedy bound")
                        measurements[dtype]["runs"].append({"seconds": elapsed, "margin": report.margin,
                            "greedyMargin": greedy, "peakAllocatedBytes": peak,
                            "diagnostics": dict(report.numericalDiagnostics), "timings": asdict(report.timings)})
                for dtype, result in measurements.items():
                    result["medianSeconds"] = median(r["seconds"] for r in result["runs"])
                    output["pipeline"].append(result)
                if args.trace_directory:
                    args.trace_directory.mkdir(parents=True, exist_ok=True)
                    name = args.trace_directory / f"assets-{assets}-correlated-{correlated}"
                    cpu = cProfile.Profile()
                    cpu.runcall(applications["float32"])
                    cpu.dump_stats(str(name) + ".prof")
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=True) as profile:
                        applications["float32"]()
                        synchronize()
                    profile.export_chrome_trace(str(name) + ".json")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Completed {assets} assets", flush=True)


if __name__ == "__main__":
    main()

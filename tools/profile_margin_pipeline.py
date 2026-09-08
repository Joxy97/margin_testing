"""Profile a complete configured pipeline, with optional Torch transfer traces.

Run one process per configuration for comparable peak RSS. Use a small matching
CPU YAML as --reference-config before profiling representative GPU workloads.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
from time import perf_counter

from margin_engine import MarginApplicationConfig
from margin_calculator import BQMMarginCalculatorConfig, BatchBQMExecutionPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="optional Chrome/Torch operator trace")
    parser.add_argument("--reference-config", type=Path)
    args = parser.parse_args()
    application = MarginApplicationConfig.fromYaml(args.config)
    reference = None
    if args.reference_config:
        reference = MarginApplicationConfig.fromYaml(args.reference_config).generateReport()
    latest_memory = None

    def observe(memory):
        nonlocal latest_memory
        latest_memory = memory

    calculator = application.engine.marginCalculator
    if isinstance(calculator, BQMMarginCalculatorConfig) and isinstance(calculator.executionPolicy, BatchBQMExecutionPolicy):
        application = replace(application, engine=replace(application.engine, marginCalculator=replace(
            calculator, executionPolicy=replace(calculator.executionPolicy, memoryObserver=observe))))
    torch = None
    try:
        import torch
    except ImportError:
        if args.trace:
            parser.error("--trace requires Torch")
    devices = list(range(torch.cuda.device_count())) if torch is not None else []
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    profile = nullcontext()
    if args.trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if devices:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile = torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=True)
    start = perf_counter()
    with profile as trace:
        report = application.generateReport()
        for device in devices:
            torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace.export_chrome_trace(str(args.trace))
    result = {
        "config": str(args.config.resolve()),
        "margin": report.margin,
        "comparisonMargins": dict(report.comparisonMargins),
        "numericalDiagnostics": dict(report.numericalDiagnostics),
        "timings": asdict(report.timings),
        "synchronizedWallSeconds": elapsed,
        "processPeakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "coefficientMemory": None if latest_memory is None else asdict(latest_memory),
        "cudaPeakAllocatedBytes": {str(device): torch.cuda.max_memory_allocated(device) for device in devices},
        "cudaPeakReservedBytes": {str(device): torch.cuda.max_memory_reserved(device) for device in devices},
        "referenceMargin": None if reference is None else reference.margin,
        "referenceMarginDifference": None if reference is None else report.margin - reference.margin,
        "torchVersion": None if torch is None else torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

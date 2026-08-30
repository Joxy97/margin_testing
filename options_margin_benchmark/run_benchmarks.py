#!/usr/bin/env python3
"""Run every generated option-margin benchmark YAML file."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from margin_engine import MarginApplicationConfig


GENERATED = ROOT / "generated"
MANIFEST = GENERATED / "manifest.csv"
RESULTS = GENERATED / "results.csv"


def main() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        benchmarks = list(csv.DictReader(stream))

    results = []
    for index, benchmark in enumerate(benchmarks, start=1):
        started = perf_counter()
        try:
            application = MarginApplicationConfig.fromYaml(
                GENERATED / benchmark["config"]
            )
            margin = application.generateReport().margin
            status, error = "ok", ""
        except Exception as exception:
            margin = ""
            status, error = "error", f"{type(exception).__name__}: {exception}"
        results.append({
            **benchmark,
            "margin": margin,
            "elapsed_seconds": perf_counter() - started,
            "status": status,
            "error": error,
        })
        if index % 10 == 0 or index == len(benchmarks):
            print(f"completed {index}/{len(benchmarks)}")

    with RESULTS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    failures = sum(item["status"] != "ok" for item in results)
    print(f"wrote {RESULTS}: {len(results) - failures} passed, {failures} failed")


if __name__ == "__main__":
    main()

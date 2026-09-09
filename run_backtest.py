#!/usr/bin/env python3
"""Run YAML-configured margin backtests and write CSV reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtesting.experiment import BacktestExperiment, experimentFingerprint as _experimentFingerprint
from margin_engine import MarginApplicationConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a margin backtest configured by a YAML file."
    )
    parser.add_argument("config", type=Path, help="backtest YAML file")
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="override backtest.outputDirectory from the YAML file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume matching completed days from the output checkpoint",
    )
    arguments = parser.parse_args()

    try:
        experiment = BacktestExperiment.fromYaml(arguments.config, arguments.output_directory)
    except ValueError as error:
        parser.error(str(error))
    def print_progress(name, backtest_date, index, total):
        print(
            f"[{name}] Processing date {index}/{total}: {backtest_date.isoformat()}",
            flush=True,
        )

    files = experiment.run(
        resume=arguments.resume,
        onDayStarted=print_progress,
    ).reportFiles
    for name, report_files in files.items():
        print(f"{name}:")
        print(f"  breaches: {report_files.breaches}")
        print(f"  performance: {report_files.performanceMetrics}")


if __name__ == "__main__":
    main()

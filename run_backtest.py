#!/usr/bin/env python3
"""Run YAML-configured margin backtests and write CSV reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtesting import BacktestCSVReporter
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
    arguments = parser.parse_args()

    application = MarginApplicationConfig.fromYaml(arguments.config)
    output_directory = (
        arguments.output_directory or application.backtestOutputDirectory
    )
    if output_directory is None:
        parser.error(
            "set backtest.outputDirectory in YAML or pass --output-directory"
        )

    results = application.generateBacktest()
    files = BacktestCSVReporter().write(results, output_directory)
    for name, report_files in files.items():
        print(f"{name}:")
        print(f"  breaches: {report_files.breaches}")
        print(f"  performance: {report_files.performanceMetrics}")


if __name__ == "__main__":
    main()

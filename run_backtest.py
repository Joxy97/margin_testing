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

from backtesting import BacktestCheckpointStore, BacktestCSVReporter
from margin_engine import MarginApplicationConfig


def _experimentFingerprint(
    configBytes: bytes,
    application: MarginApplicationConfig,
) -> str:
    """Fingerprint canonical requests plus every configured local data file."""
    digest = hashlib.sha256(configBytes)
    for name, request in sorted(application.backtestRequests.items()):
        digest.update(str(name).encode("utf-8"))
        for instrument in request.portfolio.instruments:
            digest.update(str(instrument).encode("utf-8"))
            digest.update(
                str(request.portfolio.weights[instrument]).encode("utf-8")
            )
        digest.update(str(request.portfolio.cash).encode("utf-8"))
        for backtest_date in request.dates:
            digest.update(backtest_date.isoformat().encode("ascii"))

    request_parameters = application.engine.downloadManager.requestParameters
    configured_paths = []
    if request_parameters.get("location") is not None:
        configured_paths.append(request_parameters["location"])
    configured_paths.extend(request_parameters.get("locations", ()))
    for value in sorted(map(str, configured_paths)):
        path = Path(value).expanduser().resolve()
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


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

    application = MarginApplicationConfig.fromYaml(arguments.config)
    output_directory = (
        arguments.output_directory or application.backtestOutputDirectory
    )
    if output_directory is None:
        parser.error(
            "set backtest.outputDirectory in YAML or pass --output-directory"
        )

    config_path = arguments.config.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config_fingerprint = hashlib.sha256(config_bytes).hexdigest()
    fingerprint = _experimentFingerprint(config_bytes, application)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "experiment_config.yaml").write_bytes(config_bytes)
    (output_directory / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "configPath": str(config_path),
                "configSha256": config_fingerprint,
                "experimentSha256": fingerprint,
                "checkpointSchema": 1,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoints = BacktestCheckpointStore(
        output_directory / ".checkpoints",
        fingerprint,
    )
    results = application.generateBacktest(
        checkpointStore=checkpoints,
        resume=arguments.resume,
    )
    files = BacktestCSVReporter().write(results, output_directory)
    for name, report_files in files.items():
        print(f"{name}:")
        print(f"  breaches: {report_files.breaches}")
        print(f"  performance: {report_files.performanceMetrics}")


if __name__ == "__main__":
    main()

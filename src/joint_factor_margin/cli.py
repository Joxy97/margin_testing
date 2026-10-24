"""Command-line adapters for calculation, replay and artifact publication."""

import argparse
import json
from pathlib import Path
import sys
import yaml

from .configuration import ApplicationConfig
from .backtest import BacktestRunner
from .reporting import writeJson, writeBacktest
from .application import createApplication


def main(argv=None):
    parser = argparse.ArgumentParser(prog="factor-margin", description="Existing joint factor-stress model application")
    commands = parser.add_subparsers(dest="command", required=True)
    calculate = commands.add_parser("calculate", help="Calculate one dated portfolio margin")
    calculate.add_argument("config", type=Path)
    calculate.add_argument("--output", type=Path, help="Write JSON instead of printing it")
    backtest = commands.add_parser("backtest", help="Replay existing stock-exposure backtesting")
    backtest.add_argument("config", type=Path)
    backtest.add_argument("--output", required=True, type=Path)
    backtest.add_argument("--resume", action="store_true")
    backtest.add_argument("--plots", action="store_true", help="Requires the reports extra")
    args = parser.parse_args(argv)
    try:
        config = ApplicationConfig.fromYaml(args.config)
        if args.command == "calculate":
            result = config.createApplication().calculate(config.request)
            if args.output:
                writeJson(result, args.output)
            else:
                print(json.dumps(result.toMapping(), indent=2, sort_keys=True, allow_nan=False))
        else:
            if config.backtest is None:
                raise ValueError("configuration has no backtest section")
            if config.request.options is not None:
                raise ValueError("option backtesting requires a dated realized-mark adapter")
            from backtesting import BacktestCheckpointStore
            identity = config.identity()
            output = args.output.expanduser().resolve()
            store = BacktestCheckpointStore(output/".checkpoints", identity)
            writeJson({"schemaVersion": 1, "experimentIdentity": identity,
                       "timingVersion": 2, "configPath": str(args.config.resolve()),
                       "inputFiles": config.inputFiles}, output/"manifest.json")
            if config.sourceText is not None:
                (output/"configuration.yaml").write_text(config.sourceText)
            results = BacktestRunner().run(createApplication(config.engine, identity), config.request.portfolio,
                config.backtest, checkpointStore=store, resume=args.resume)
            writeBacktest(results, output, plots=args.plots)
            print(f"Backtest reports: {output}")
    except (ValueError, TypeError, ImportError, OSError, RuntimeError, yaml.YAMLError) as error:
        print(f"factor-margin: {error}", file=sys.stderr)
        return 2
    return 0

"""Run a margin calculation from a YAML application configuration."""

import argparse

import yaml

from .yaml_application import MarginApplicationConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a margin report from YAML configuration."
    )
    parser.add_argument("config", help="path to the YAML application file")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="run the YAML backtest block instead of one margin report",
    )
    arguments = parser.parse_args()
    application = MarginApplicationConfig.fromYaml(arguments.config)
    if arguments.backtest:
        batch = application.generateBacktest()
        output = {
            name: {
                "violations": result.violations,
                "baselProbability": result.baselProbability,
                "baselColor": result.baselColor.value,
                "confidenceLevel": result.confidenceLevel,
                "dailyResults": [
                    {
                        "date": daily.date.isoformat(),
                        "margin": daily.margin,
                        "realizedPnL": daily.realizedPnL,
                        "grossExposure": daily.grossExposure,
                        "marginPercent": daily.marginPercent,
                        "breach": daily.breach,
                    }
                    for daily in result.dailyResults
                ],
            }
            for name, result in batch.results.items()
        }
        print(yaml.safe_dump({"backtest": output}, sort_keys=False), end="")
        return
    report = application.generateReport()
    print(
        yaml.safe_dump(
            {
                "margin": report.margin,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()

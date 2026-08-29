"""Run a margin calculation from a YAML application configuration."""

import argparse

import yaml

from .yaml_application import MarginApplicationConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a margin report from YAML configuration."
    )
    parser.add_argument("config", help="path to the YAML application file")
    arguments = parser.parse_args()
    report = MarginApplicationConfig.fromYaml(arguments.config).generateReport()
    print(yaml.safe_dump({"margin": report.margin}, sort_keys=False), end="")


if __name__ == "__main__":
    main()

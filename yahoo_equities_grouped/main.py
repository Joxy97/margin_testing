"""CLI entry point for grouping substantial Yahoo equity data."""

import argparse
from pathlib import Path

from .exporter import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent.parent / "yahoo_equities")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()

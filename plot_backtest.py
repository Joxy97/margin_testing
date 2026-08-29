#!/usr/bin/env python3
"""Plot realized loss and margin as percentages of gross exposure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as pyplot
import numpy
import pandas


REQUIRED_COLUMNS = {
    "portfolio",
    "date",
    "realized_pnl",
    "gross_exposure",
    "margin_percent",
}


def plotBacktest(csvPath: str | Path, outputPath: str | Path | None = None) -> Path:
    """Create a realized-loss/margin chart from a breaches CSV file."""
    csv_path = Path(csvPath).expanduser().resolve()
    output_path = (
        Path(outputPath).expanduser().resolve()
        if outputPath is not None
        else csv_path.with_name("pnl_margin_plot.png")
    )
    data = pandas.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing columns: {sorted(missing_columns)}"
        )
    if data.empty:
        raise ValueError(f"{csv_path} contains no backtest rows")

    data = data.copy()
    data["date"] = pandas.to_datetime(data["date"], errors="raise")
    for column in ("realized_pnl", "gross_exposure", "margin_percent"):
        data[column] = pandas.to_numeric(data[column], errors="raise")
    numeric_values = data[
        ["realized_pnl", "gross_exposure", "margin_percent"]
    ].to_numpy(dtype=float)
    if not numpy.isfinite(numeric_values).all():
        raise ValueError("plot values must be finite")
    if numpy.any(data["gross_exposure"].to_numpy(dtype=float) == 0.0):
        raise ValueError("gross exposure must be nonzero")

    data = data.sort_values("date")
    data["realized_loss_percent"] = (
        100.0
        * numpy.maximum(0.0, -data["realized_pnl"])
        / data["gross_exposure"]
    )
    portfolios = tuple(data["portfolio"].astype(str).unique())
    portfolio_label = ", ".join(portfolios)

    figure, axis = pyplot.subplots(figsize=(12, 6))
    axis.plot(
        data["date"],
        data["realized_loss_percent"],
        label="Realized loss",
        color="tab:blue",
        linewidth=1.1,
    )
    axis.plot(
        data["date"],
        data["margin_percent"],
        label="Margin",
        color="tab:orange",
        linewidth=1.8,
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(f"Realized Loss and Margin — {portfolio_label}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Percent of gross exposure (%)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    locator = mdates.AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    pyplot.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot realized loss percent and margin percent from breaches.csv"
        )
    )
    parser.add_argument("csv", type=Path, help="path to breaches.csv")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path; defaults beside the input CSV",
    )
    arguments = parser.parse_args()
    print(plotBacktest(arguments.csv, arguments.output))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare fetched Torch SBM and greedy backtest margin results."""

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
KEY_COLUMNS = ["portfolio", "date"]


def _loadBacktest(csvPath: str | Path, label: str) -> pandas.DataFrame:
    csv_path = Path(csvPath).expanduser().resolve()
    data = pandas.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"{csv_path} contains no backtest rows")

    data = data.copy()
    data["portfolio"] = data["portfolio"].astype(str)
    data["date"] = pandas.to_datetime(data["date"], errors="raise")
    for column in ("realized_pnl", "gross_exposure", "margin_percent"):
        data[column] = pandas.to_numeric(data[column], errors="raise")
    values = data[
        ["realized_pnl", "gross_exposure", "margin_percent"]
    ].to_numpy(dtype=float)
    if not numpy.isfinite(values).all():
        raise ValueError(f"{label} plot values must be finite")
    if numpy.any(data["gross_exposure"].to_numpy(dtype=float) == 0.0):
        raise ValueError(f"{label} gross exposure must be nonzero")
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{label} contains duplicate portfolio/date rows")
    return data.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _alignedResults(
    fetched: pandas.DataFrame,
    greedy: pandas.DataFrame,
) -> pandas.DataFrame:
    fetched_keys = pandas.MultiIndex.from_frame(fetched[KEY_COLUMNS])
    greedy_keys = pandas.MultiIndex.from_frame(greedy[KEY_COLUMNS])
    if not fetched_keys.equals(greedy_keys):
        missing_from_greedy = len(fetched_keys.difference(greedy_keys))
        missing_from_fetched = len(greedy_keys.difference(fetched_keys))
        raise ValueError(
            "fetched and greedy results must have identical portfolio/date "
            f"rows; missing from greedy: {missing_from_greedy}, missing from "
            f"fetched: {missing_from_fetched}"
        )

    aligned = fetched[KEY_COLUMNS].copy()
    aligned["realized_pnl"] = fetched["realized_pnl"]
    aligned["gross_exposure"] = fetched["gross_exposure"]
    aligned["fetched_margin_percent"] = fetched["margin_percent"]
    aligned["greedy_margin_percent"] = greedy["margin_percent"]
    if not numpy.allclose(
        fetched["realized_pnl"],
        greedy["realized_pnl"],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("fetched and greedy realized PnL values differ")
    if not numpy.allclose(
        fetched["gross_exposure"],
        greedy["gross_exposure"],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("fetched and greedy gross exposures differ")
    aligned["realized_loss_percent"] = (
        100.0
        * numpy.maximum(0.0, -aligned["realized_pnl"])
        / aligned["gross_exposure"]
    )
    aligned["margin_difference"] = (
        aligned["fetched_margin_percent"]
        - aligned["greedy_margin_percent"]
    )
    return aligned


def plotBacktestComparison(
    fetchedCsvPath: str | Path,
    greedyCsvPath: str | Path,
    outputPath: str | Path | None = None,
) -> Path:
    """Plot aligned Torch SBM and greedy margin time series."""
    fetched_path = Path(fetchedCsvPath).expanduser().resolve()
    output_path = (
        Path(outputPath).expanduser().resolve()
        if outputPath is not None
        else fetched_path.with_name("fetched_vs_greedy_plot.png")
    )
    data = _alignedResults(
        _loadBacktest(fetched_path, "fetched"),
        _loadBacktest(greedyCsvPath, "greedy"),
    )
    portfolios = tuple(data["portfolio"].unique())
    portfolio_label = ", ".join(portfolios)
    differences = data["margin_difference"].to_numpy(dtype=float)
    mean_absolute_difference = float(numpy.mean(numpy.abs(differences)))
    maximum_absolute_difference = float(numpy.max(numpy.abs(differences)))

    figure, (margin_axis, difference_axis) = pyplot.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    margin_axis.plot(
        data["date"],
        data["realized_loss_percent"],
        label="Realized loss",
        color="tab:blue",
        linewidth=1.0,
        alpha=0.75,
    )
    margin_axis.plot(
        data["date"],
        data["fetched_margin_percent"],
        label="Torch SBM (fetched)",
        color="tab:orange",
        linewidth=1.8,
    )
    margin_axis.plot(
        data["date"],
        data["greedy_margin_percent"],
        label="Greedy",
        color="tab:green",
        linewidth=1.5,
        linestyle="--",
    )
    margin_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    margin_axis.set_title(f"Torch SBM vs Greedy Margin — {portfolio_label}")
    margin_axis.set_ylabel("Percent of gross exposure (%)")
    margin_axis.grid(True, alpha=0.25)
    margin_axis.legend()

    difference_axis.plot(
        data["date"],
        data["margin_difference"],
        color="tab:purple",
        linewidth=1.2,
    )
    difference_axis.fill_between(
        data["date"],
        0.0,
        data["margin_difference"],
        where=data["margin_difference"] >= 0.0,
        color="tab:red",
        alpha=0.25,
        label="Torch SBM higher",
    )
    difference_axis.fill_between(
        data["date"],
        0.0,
        data["margin_difference"],
        where=data["margin_difference"] < 0.0,
        color="tab:green",
        alpha=0.25,
        label="Torch SBM lower",
    )
    difference_axis.axhline(0.0, color="black", linewidth=0.8)
    difference_axis.set_ylabel("Difference\n(percentage points)")
    difference_axis.set_xlabel("Date")
    difference_axis.grid(True, alpha=0.25)
    difference_axis.legend(loc="upper right", ncol=2, fontsize="small")
    difference_axis.text(
        0.01,
        0.95,
        (
            f"Mean absolute difference: {mean_absolute_difference:.4f} pp\n"
            f"Maximum absolute difference: {maximum_absolute_difference:.4f} pp"
        ),
        transform=difference_axis.transAxes,
        va="top",
        fontsize="small",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    locator = mdates.AutoDateLocator()
    difference_axis.xaxis.set_major_locator(locator)
    difference_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    pyplot.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fetched Torch SBM and greedy breaches CSV files"
    )
    parser.add_argument("fetched_csv", type=Path)
    parser.add_argument("greedy_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path; defaults beside the fetched CSV",
    )
    arguments = parser.parse_args()
    print(
        plotBacktestComparison(
            arguments.fetched_csv,
            arguments.greedy_csv,
            arguments.output,
        )
    )


if __name__ == "__main__":
    main()

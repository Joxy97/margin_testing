#!/usr/bin/env python3
"""Create matplotlib charts for generated option-margin benchmark results."""

from __future__ import annotations

import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import tempfile
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
PLOT_CACHE = Path(tempfile.gettempdir()) / "margin_testing_matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot
import numpy
import pandas


GENERATED = ROOT / "generated"
RESULTS = GENERATED / "results.csv"
PLOTS = GENERATED / "plots"
WORKBOOK = ROOT / "ES_NQ_threeway comparison results.xlsx"

XML_NAMESPACE = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office": (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ),
}


def _workbookResults() -> pandas.DataFrame:
    """Read the three reference margin series without an Excel dependency."""
    records = []
    with ZipFile(WORKBOOK) as workbook:
        shared_root = ElementTree.fromstring(
            workbook.read("xl/sharedStrings.xml")
        )
        text_tag = f"{{{XML_NAMESPACE['main']}}}t"
        shared = [
            "".join(item.text or "" for item in cell.iter(text_tag))
            for cell in shared_root
        ]
        relations = ElementTree.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relations}
        workbook_root = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        sheets = workbook_root.find("main:sheets", XML_NAMESPACE)
        for sheet in sheets:
            name = sheet.attrib["name"]
            symbol, category = name.split(maxsplit=1)
            relation = sheet.attrib[f"{{{XML_NAMESPACE['office']}}}id"]
            target = str(PurePosixPath("xl") / targets[relation]).replace(
                "xl/../", ""
            )
            root = ElementTree.fromstring(workbook.read(target))
            rows = root.findall(".//main:sheetData/main:row", XML_NAMESPACE)
            for row in rows[1:]:
                values = {}
                for cell in row.findall("main:c", XML_NAMESPACE):
                    column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                    value = cell.find("main:v", XML_NAMESPACE)
                    if value is None:
                        continue
                    item = value.text
                    if cell.attrib.get("t") == "s":
                        item = shared[int(item)]
                    values[column] = item
                if not values.get("C", "").lower().startswith("portfolio"):
                    continue
                for column, method in (
                    ("D", "CME"),
                    ("E", "CoH"),
                    ("F", "CoH new parameters"),
                ):
                    if column in values or column == "D":
                        records.append({
                            "symbol": symbol,
                            "category": category,
                            "method": method,
                            "margin": abs(float(values.get(column, 0.0))),
                        })
    return pandas.DataFrame.from_records(records)


def _plotComparison(
    symbol: str,
    category: str,
    margins: pandas.Series,
    reference: pandas.DataFrame,
) -> Path:
    figure, axis = pyplot.subplots(figsize=(12, 7))
    series = {
        method: group["margin"].to_numpy(dtype=float)
        for method, group in reference.groupby("method")
    }
    series["Our state-aware greedy"] = margins.to_numpy(dtype=float)
    for method, values in series.items():
        ordered = numpy.sort(values)
        percentile = numpy.linspace(0.0, 100.0, len(ordered))
        axis.plot(
            percentile,
            ordered,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=f"{method} (n={len(ordered)})",
        )
    axis.set_title(f"{symbol} {category.title()} — margin comparison")
    axis.set_xlabel("Portfolio percentile within each source")
    axis.set_ylabel("Margin (USD)")
    axis.set_yscale("symlog", linthresh=10_000)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = PLOTS / f"{symbol}_{category}_comparison.png"
    figure.savefig(output, dpi=160, bbox_inches="tight")
    pyplot.close(figure)
    return output


def main() -> None:
    results = pandas.read_csv(RESULTS)
    failures = results.loc[results["status"] != "ok"]
    if not failures.empty:
        raise RuntimeError(f"Cannot plot {len(failures)} failed benchmarks")
    results["margin"] = pandas.to_numeric(results["margin"], errors="raise")
    results = results.sort_values(["symbol", "category", "date", "portfolio"])
    reference = _workbookResults()
    PLOTS.mkdir(parents=True, exist_ok=True)

    for (symbol, category), group in results.groupby(["symbol", "category"]):
        labels = [
            f"{date}\n{portfolio}"
            for date, portfolio in zip(group["date"], group["portfolio"])
        ]
        width = max(14.0, 0.24 * len(group))
        figure, axis = pyplot.subplots(figsize=(width, 7))
        axis.plot(
            range(len(group)),
            group["margin"],
            color="tab:blue",
            marker="o",
            markersize=4,
            linewidth=1.2,
        )
        axis.set_title(f"{symbol} {category.title()} — State-aware greedy margin")
        axis.set_xlabel("Benchmark portfolio")
        axis.set_ylabel("Margin (USD)")
        axis.set_yscale("log")
        axis.set_xticks(range(len(group)), labels, rotation=90, fontsize=6)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        output = PLOTS / f"{symbol}_{category}.png"
        figure.savefig(output, dpi=160, bbox_inches="tight")
        pyplot.close(figure)
        print(output)
        comparison = _plotComparison(
            symbol,
            category,
            group["margin"],
            reference.loc[
                (reference["symbol"] == symbol)
                & (reference["category"] == category)
            ],
        )
        print(comparison)

    summary = results.groupby(["symbol", "category"], as_index=False).agg(
        median_margin=("margin", "median"),
        maximum_margin=("margin", "max"),
    )
    labels = summary["symbol"] + " " + summary["category"]
    figure, axis = pyplot.subplots(figsize=(11, 6))
    positions = list(range(len(summary)))
    axis.plot(
        positions,
        summary["maximum_margin"],
        label="Maximum",
        marker="^",
        markersize=8,
    )
    axis.plot(
        positions,
        summary["median_margin"],
        label="Median",
        marker="o",
        markersize=7,
    )
    axis.set_title("Option-margin benchmark summary")
    axis.set_ylabel("Margin (USD)")
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = PLOTS / "summary.png"
    figure.savefig(output, dpi=160, bbox_inches="tight")
    pyplot.close(figure)
    print(output)


if __name__ == "__main__":
    main()

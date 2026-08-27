#!/usr/bin/env python3
"""Aggregate the ten-QUBO RAM-capped benchmark."""

import argparse
import csv
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    groups = defaultdict(list)
    with open(args.input, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            groups[(int(row["vertices"]), row["method"])].append(row)

    columns = [
        "vertices",
        "method",
        "qubos",
        "median_solve_ms",
        "mean_quality_ratio",
        "minimum_quality_ratio",
        "wins",
        "maximum_peak_rss_mib",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for (vertices, method), rows in sorted(groups.items()):
            ratios = [float(row["quality_ratio"]) for row in rows]
            writer.writerow(
                {
                    "vertices": vertices,
                    "method": method,
                    "qubos": len(rows),
                    "median_solve_ms": statistics.median(
                        float(row["solve_ms"]) for row in rows
                    ),
                    "mean_quality_ratio": statistics.mean(ratios),
                    "minimum_quality_ratio": min(ratios),
                    "wins": sum(ratio >= 1.0 - 1e-12 for ratio in ratios),
                    "maximum_peak_rss_mib": max(
                        float(row["peak_rss_mib"]) for row in rows
                    ),
                }
            )


if __name__ == "__main__":
    main()

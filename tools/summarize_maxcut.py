#!/usr/bin/env python3
"""Aggregate raw MaxCut benchmark rows by graph size and method."""

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

    with open(args.output, "w", newline="", encoding="utf-8") as destination:
        columns = [
            "vertices",
            "method",
            "instances",
            "median_solve_ms",
            "mean_quality_ratio",
            "minimum_quality_ratio",
            "best_observed_wins",
            "maximum_peak_rss_mib",
        ]
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for (vertices, method), rows in sorted(groups.items()):
            ratios = [float(row["quality_ratio"]) for row in rows]
            writer.writerow(
                {
                    "vertices": vertices,
                    "method": method,
                    "instances": len(rows),
                    "median_solve_ms": statistics.median(
                        float(row["solve_ms"]) for row in rows
                    ),
                    "mean_quality_ratio": statistics.mean(ratios),
                    "minimum_quality_ratio": min(ratios),
                    "best_observed_wins": sum(ratio >= 1.0 - 1e-12 for ratio in ratios),
                    "maximum_peak_rss_mib": max(
                        float(row["peak_rss_mib"]) for row in rows
                    ),
                }
            )


if __name__ == "__main__":
    main()

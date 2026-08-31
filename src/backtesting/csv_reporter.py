"""CSV reporting for named portfolio margin backtests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from .backtest_results import BacktestBatchResults, BacktestResults


@dataclass(frozen=True)
class BacktestReportFiles:
    """Paths written for one named portfolio backtest."""

    breaches: Path
    performanceMetrics: Path


class BacktestCSVReporter:
    """Write daily breach data and aggregate timing metrics as CSV files."""

    _timingFields = (
        ("data_acquisition", "dataAcquisitionSeconds"),
        ("risk_state_generation", "riskStateGenerationSeconds"),
        ("margin_calculation", "marginCalculationSeconds"),
        ("realized_data_acquisition", "realizedDataAcquisitionSeconds"),
        ("realized_pnl_calculation", "realizedPnLCalculationSeconds"),
        ("total", "totalSeconds"),
    )

    def write(
        self,
        results: BacktestBatchResults,
        outputDirectory: str | Path,
    ) -> Mapping[str, BacktestReportFiles]:
        """Write a separate two-file report for every named portfolio."""
        if not isinstance(results, BacktestBatchResults):
            raise TypeError("results must be BacktestBatchResults")
        output_directory = Path(outputDirectory).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, BacktestReportFiles] = {}
        used_directories: set[str] = set()
        for name, portfolio_results in results.results.items():
            directory_name = self._safeDirectoryName(name)
            if directory_name in used_directories:
                raise ValueError(
                    "Portfolio names must have distinct filesystem-safe names"
                )
            used_directories.add(directory_name)
            portfolio_directory = output_directory / directory_name
            portfolio_directory.mkdir(parents=True, exist_ok=True)
            files = BacktestReportFiles(
                breaches=portfolio_directory / "breaches.csv",
                performanceMetrics=(
                    portfolio_directory / "performance_metrics.csv"
                ),
            )
            self._writeBreaches(files.breaches, name, portfolio_results)
            self._writePerformance(
                files.performanceMetrics,
                name,
                portfolio_results,
            )
            written[name] = files
        return MappingProxyType(written)

    @staticmethod
    def _safeDirectoryName(name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
        return normalized or "portfolio"

    @staticmethod
    def _writeBreaches(
        path: Path,
        portfolioName: str,
        results: BacktestResults,
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            comparison_names = sorted(
                {
                    name
                    for daily in results.dailyResults
                    for name in daily.comparisonMargins
                }
            )
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "portfolio",
                    "date",
                    "margin",
                    "realized_pnl",
                    "realized_loss",
                    "gross_exposure",
                    "margin_percent",
                    "margin_error",
                    "shortfall",
                    "breach",
                    "covered",
                    *(
                        field
                        for name in comparison_names
                        for field in (
                            f"{name}_margin",
                            f"margin_minus_{name}",
                            f"{name}_margin_error",
                            f"{name}_shortfall",
                            f"{name}_breach",
                        )
                    ),
                ),
            )
            writer.writeheader()
            for daily in results.dailyResults:
                row = {
                    "portfolio": portfolioName,
                    "date": daily.date.isoformat(),
                    "margin": daily.margin,
                    "realized_pnl": daily.realizedPnL,
                    "realized_loss": daily.realizedLoss,
                    "gross_exposure": daily.grossExposure,
                    "margin_percent": daily.marginPercent,
                    "margin_error": daily.marginError,
                    "shortfall": daily.shortfall,
                    "breach": daily.breach,
                    "covered": daily.covered,
                }
                for name in comparison_names:
                    comparison = daily.comparisonMargins.get(name)
                    row[f"{name}_margin"] = comparison
                    row[f"margin_minus_{name}"] = (
                        None if comparison is None else daily.margin - comparison
                    )
                    comparison_error = (
                        None
                        if comparison is None
                        else comparison - daily.realizedLoss
                    )
                    row[f"{name}_margin_error"] = comparison_error
                    row[f"{name}_shortfall"] = (
                        None
                        if comparison_error is None
                        else max(0.0, -comparison_error)
                    )
                    row[f"{name}_breach"] = (
                        None
                        if comparison is None
                        else daily.realizedLoss > comparison
                    )
                writer.writerow(row)

    def _writePerformance(
        self,
        path: Path,
        portfolioName: str,
        results: BacktestResults,
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            comparison_names = sorted(
                {
                    name
                    for daily in results.dailyResults
                    for name in daily.comparisonMargins
                }
            )
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "portfolio",
                    "days",
                    "violations",
                    "breach_rate",
                    "confidence_level",
                    "basel_probability",
                    "coverage_p_value",
                    "basel_color",
                    "stage",
                    "total_seconds",
                    "average_seconds_per_day",
                    "average_margin",
                    *(
                        field
                        for name in comparison_names
                        for field in (
                            f"{name}_violations",
                            f"{name}_breach_rate",
                            f"{name}_average_margin",
                            f"average_margin_minus_{name}",
                        )
                    ),
                ),
            )
            writer.writeheader()
            common = self._performanceSummary(results, comparison_names)
            for stage, field_name in self._timingFields:
                total = sum(
                    getattr(daily.timings, field_name)
                    for daily in results.dailyResults
                )
                writer.writerow(
                    common
                    | {
                        "portfolio": portfolioName,
                        "days": results.days,
                        "violations": results.violations,
                        "breach_rate": results.violations / results.days,
                        "confidence_level": results.confidenceLevel,
                        "basel_probability": results.baselProbability,
                        "coverage_p_value": results.coveragePValue,
                        "basel_color": results.baselColor.value,
                        "stage": stage,
                        "total_seconds": total,
                        "average_seconds_per_day": total / results.days,
                    }
                )
            writer.writerow(
                common
                | {
                    "portfolio": portfolioName,
                    "days": results.days,
                    "violations": results.violations,
                    "breach_rate": results.violations / results.days,
                    "confidence_level": results.confidenceLevel,
                    "basel_probability": results.baselProbability,
                    "coverage_p_value": results.coveragePValue,
                    "basel_color": results.baselColor.value,
                    "stage": "preparation",
                    "total_seconds": results.preparationSeconds,
                    "average_seconds_per_day": (
                        results.preparationSeconds / results.days
                    ),
                }
            )

    @staticmethod
    def _performanceSummary(
        results: BacktestResults,
        comparisonNames: list[str],
    ) -> dict[str, float | int]:
        primary_average = sum(
            daily.margin for daily in results.dailyResults
        ) / results.days
        summary: dict[str, float | int] = {
            "average_margin": primary_average,
        }
        for name in comparisonNames:
            margins = [
                daily.comparisonMargins[name]
                for daily in results.dailyResults
                if name in daily.comparisonMargins
            ]
            violations = sum(
                daily.realizedLoss > daily.comparisonMargins[name]
                for daily in results.dailyResults
                if name in daily.comparisonMargins
            )
            average = sum(margins) / len(margins)
            summary[f"{name}_violations"] = violations
            summary[f"{name}_breach_rate"] = violations / len(margins)
            summary[f"{name}_average_margin"] = average
            summary[f"average_margin_minus_{name}"] = (
                primary_average - average
            )
        return summary

"""Tests for fetched-versus-greedy backtest plotting."""

import tempfile
import unittest
from pathlib import Path

import pandas

from plot_backtest_comparison import plotBacktestComparison


class PlotBacktestComparisonTest(unittest.TestCase):
    @staticmethod
    def _writeCsv(path: Path, margins: list[float]) -> None:
        pandas.DataFrame(
            {
                "portfolio": ["default", "default"],
                "date": ["2025-01-01", "2025-01-02"],
                "realized_pnl": [-0.01, 0.02],
                "gross_exposure": [1.0, 1.0],
                "margin_percent": margins,
            }
        ).to_csv(path, index=False)

    def test_creates_comparison_beside_fetched_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fetched = root / "fetched.csv"
            greedy = root / "greedy.csv"
            self._writeCsv(fetched, [6.0, 7.0])
            self._writeCsv(greedy, [5.5, 6.5])

            output = plotBacktestComparison(fetched, greedy)

            self.assertEqual(output, root / "fetched_vs_greedy_plot.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_rejects_misaligned_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fetched = root / "fetched.csv"
            greedy = root / "greedy.csv"
            self._writeCsv(fetched, [6.0, 7.0])
            self._writeCsv(greedy, [5.5, 6.5])
            greedy_data = pandas.read_csv(greedy)
            greedy_data.loc[1, "date"] = "2025-01-03"
            greedy_data.to_csv(greedy, index=False)

            with self.assertRaisesRegex(ValueError, "identical"):
                plotBacktestComparison(fetched, greedy)


if __name__ == "__main__":
    unittest.main()

"""Complete experiment lifecycle through its public run interface."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml


class BacktestExperimentTest(unittest.TestCase):
    def test_completed_experiment_resumes_and_publishes_reports(self):
        from backtesting.experiment import BacktestExperiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prices.csv").write_text("date,A\n2024-01-01,100\n2024-01-02,101\n"
                                             "2024-01-03,99\n2024-01-04,102\n")
            path = root / "experiment.yaml"
            path.write_text(yaml.safe_dump({
                "marginDate": "2024-01-04", "portfolio": {"weights": {"A": 1}},
                "engine": {"downloadManager": {"providers": {"local": "local_csv"},
                    "requestParameters": {"location": "prices.csv"}},
                    "riskStateGenerator": {"ew_window": 2, "scenariosPerComponents": [1],
                        "nZBins": 1, "allowEmptyBinFallback": True},
                    "marginCalculator": {"type": "greedy"}},
                "backtest": {"dates": ["2024-01-04"], "outputDirectory": "output"},
            }))
            original = path.read_bytes()
            experiment = BacktestExperiment.fromYaml(path)
            path.write_bytes(original + b"\n# edited after loading\n")
            initial = experiment.run()
            self.assertEqual((root / "output/experiment_config.yaml").read_bytes(), original)
            path.write_bytes(original)
            resumed = BacktestExperiment.fromYaml(path).run(resume=True)
            self.assertEqual(initial.results.results["default"].dailyResults,
                             resumed.results.results["default"].dailyResults)
            self.assertEqual(resumed.results.results["default"].preparationSeconds, 0.)
            self.assertTrue(resumed.reportFiles["default"].breaches.is_file())
            self.assertEqual(json.loads((root / "output/experiment_manifest.json").read_text())["checkpointSchema"], 2)
            self.assertEqual(json.loads((root / "output/experiment_manifest.json").read_text())["numericalModelVersion"], 3)

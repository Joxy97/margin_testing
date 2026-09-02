"""Tests for complete YAML-driven margin application configuration."""

import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import yaml

from data_manager import PartitionedPickleDataStore
from download_unit import LocalCSVDataProvider, ProductChunker
from margin_calculator import (
    BQMMarginCalculatorConfig,
    BatchBQMExecutionPolicy,
    GreedyMarginCalculatorConfig,
)
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    TorchSBMBQMSolver,
)
from margin_engine import MarginApplicationConfig, MarginReport
from risk_state_generator import (
    CorrelatedReturnsVolaGridRiskStateGeneratorConfig,
    ReturnsVolaGridRiskStateGeneratorConfig,
)


class YamlConfigurationTest(unittest.TestCase):
    def test_loads_a_wide_portfolio_and_capitalized_csv_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "portfolio.csv").write_text(
                "client_id,AAPL,MSFT\nclient,0.6,0.4\n",
                encoding="utf-8",
            )
            (root / "prices.csv").write_text(
                "Date,AAPL,MSFT\n2025-01-02,10,20\n2025-01-03,11,21\n",
                encoding="utf-8",
            )
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2025-01-02",
                        "portfolio": {
                            "csv": "portfolio.csv",
                            "clientId": "client",
                        },
                        "backtest": {
                            "datesFromCsv": {"path": "prices.csv"},
                        },
                        "engine": {"marginCalculator": {"type": "greedy"}},
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(config_path)

        self.assertEqual(application.portfolio.weights["AAPL"], Decimal("0.6"))
        self.assertEqual(
            application.backtestRequests["default"].dates,
            (date(2025, 1, 2), date(2025, 1, 3)),
        )

    def test_long_portfolio_csv_sums_duplicate_instrument_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "portfolio.csv").write_text(
                "client_id,ticker,weight\nclient,AAPL,1.25\nclient,AAPL,-0.25\n",
                encoding="utf-8",
            )
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2025-01-02",
                        "portfolio": {
                            "csv": "portfolio.csv",
                            "clientId": "client",
                        },
                        "engine": {"marginCalculator": {"type": "greedy"}},
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(config_path)

        self.assertEqual(application.portfolio.weights["AAPL"], Decimal("1.00"))

    def test_loads_and_runs_a_complete_local_greedy_application(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = root / "prices.csv"
            rows = ["date,AAPL"]
            for day in range(11):
                current = date(2024, 1, 1) + timedelta(days=day)
                rows.append(f"{current.isoformat()},{100 + day}")
            prices.write_text("\n".join(rows), encoding="utf-8")
            config_path = root / "margin.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {
                            "weights": {"AAPL": "10"},
                            "cash": "5",
                        },
                        "engine": {
                            "downloadManager": {
                                "providers": {"local": "local_csv"},
                                "requestParameters": {
                                    "location": "prices.csv"
                                },
                            },
                            "dataManager": {
                                "memorySize": 2,
                                "backingStore": {
                                    "type": "partitioned_pickle",
                                    "directory": "cache",
                                },
                            },
                            "riskStateGenerator": {
                                "type": "returns_vola_grid",
                                "ew_window": 5,
                                "components": 1,
                                "scenariosPerComponents": [1],
                                "nZBins": 3,
                            },
                            "marginCalculator": {"type": "greedy"},
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(config_path)
            report = application.generateReport()

            self.assertEqual(application.marginDate, date(2024, 1, 11))
            self.assertEqual(application.portfolio.cash, 5)
            self.assertIsInstance(
                application.engine.marginCalculator,
                GreedyMarginCalculatorConfig,
            )
            self.assertIsInstance(
                application.engine.riskStateGenerator,
                ReturnsVolaGridRiskStateGeneratorConfig,
            )
            self.assertIsInstance(
                next(iter(application.engine.downloadManager.providers.values())),
                LocalCSVDataProvider,
            )
            self.assertIsInstance(
                application.engine.dataManager.backingStore,
                PartitionedPickleDataStore,
            )
            self.assertEqual(
                application.engine.downloadManager.requestParameters["location"],
                str(prices),
            )
            self.assertIsInstance(report, MarginReport)

    def test_loads_the_complete_example_bqm_configuration(self) -> None:
        application = MarginApplicationConfig.fromYaml(
            "config/margin.example.yaml"
        )

        calculator = application.engine.marginCalculator
        generator = application.engine.riskStateGenerator
        self.assertIsInstance(calculator, BQMMarginCalculatorConfig)
        self.assertEqual(calculator.solver.solverType, "torch_sbm")
        self.assertEqual(
            calculator.solver.constructorParameters,
            {"device": "auto"},
        )
        self.assertEqual(calculator.solver.solverParameters["runs"], 16)
        self.assertEqual(calculator.comparisonPnlAnchor, "market")
        self.assertEqual(
            calculator.solver.solverParameters["dtype"],
            "float32",
        )
        self.assertIsInstance(calculator.executionPolicy, BatchBQMExecutionPolicy)
        self.assertEqual(calculator.executionPolicy.batchSize, 105)
        self.assertEqual(calculator.executionPolicy.maxBatchBytes, 536870912)
        self.assertIsInstance(
            generator,
            CorrelatedReturnsVolaGridRiskStateGeneratorConfig,
        )
        self.assertEqual(generator.components, 2)
        self.assertEqual(generator.scenariosPerComponents, (21, 5))
        self.assertEqual(generator.nZBins, 21)
        self.assertEqual(
            application.backtestOutputDirectory,
            Path("backtest_results/example").resolve(),
        )
        self.assertEqual(
            application.backtestRequests["default"].dates,
            (
                date(2024, 1, 29),
                date(2024, 1, 30),
                date(2024, 1, 31),
            ),
        )

    def test_builds_nested_download_chunkers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {
                            "downloadManager": {
                                "providers": {"yahoo": "yfinance"},
                                "downloadAlgorithm": "exponential_backoff",
                                "downloadParameters": {
                                    "time": 0,
                                    "chunker": {
                                        "type": "product",
                                        "first": {
                                            "type": "instrument",
                                            "batchSize": 5,
                                        },
                                        "second": {
                                            "type": "date",
                                            "batchSize": 30,
                                        },
                                    },
                                },
                            },
                            "marginCalculator": {"type": "greedy"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        self.assertIsInstance(
            application.engine.downloadManager.downloadParameters["chunker"],
            ProductChunker,
        )

    def test_constructs_torch_sbm_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {
                            "marginCalculator": {
                                "type": "bqm",
                                "solver": {
                                    "type": "torch_sbm",
                                    "constructorParameters": {"device": "cpu"},
                                    "solverParameters": {
                                        "steps": 25,
                                        "runs": 4,
                                        "dtype": "float64",
                                    },
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        calculator = application.engine.marginCalculator
        self.assertIsInstance(calculator, BQMMarginCalculatorConfig)
        solver = calculator.solver.createBQMSolver()
        self.assertIsInstance(solver, TorchSBMBQMSolver)
        self.assertEqual(solver.device, "cpu")
        self.assertEqual(calculator.solver.solverParameters["runs"], 4)

    def test_constructs_multi_device_torch_sbm_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch_multi_gpu.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {
                            "marginCalculator": {
                                "type": "bqm",
                                "solver": {
                                    "type": "torch_sbm",
                                    "constructorParameters": {
                                        "devices": ["cuda:0", "cuda:1"]
                                    },
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        calculator = application.engine.marginCalculator
        self.assertIsInstance(calculator, BQMMarginCalculatorConfig)
        solver = calculator.solver.createBQMSolver()
        self.assertIsInstance(solver, TorchSBMBQMSolver)
        self.assertEqual(solver.requestedDevices, ("cuda:0", "cuda:1"))

    def test_rejects_unknown_yaml_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-11",
                        "portfolio": {"weights": {"AAPL": 1}},
                        "engine": {"unknown": True},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown YAML keys"):
                MarginApplicationConfig.fromYaml(path)


if __name__ == "__main__":
    unittest.main()

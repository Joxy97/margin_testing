"""Tests for PCA grids, keys, providers, and scenario generators."""

import unittest
from datetime import date
from unittest.mock import Mock, call

from scenario_generator import (
    PCAGrid,
    PCAGridFactory,
    PCAGridProvider,
    PCAKey,
    PCAScenario,
    ReturnsPCAGrid,
    ReturnsPCAKey,
    ReturnsVolaGridRiskState,
    ReturnsVolaGridPCAScenario,
    ReturnsVolaGridScenarioGenerator,
    ScenarioGenerator,
)


class PCAGridTest(unittest.TestCase):
    def setUp(self) -> None:
        PCAGridProvider().purgePCAGrids()

    def test_returns_pca_grid_constructor_and_calculated_fields(self) -> None:
        dates = (date(2024, 1, 1), date(2024, 1, 31))

        grid = ReturnsPCAGrid(["AAPL"], dates, 0.94, 3)

        self.assertIsInstance(grid, PCAGrid)
        self.assertEqual(grid.dates, dates)
        self.assertEqual(grid.ew_lambda, 0.94)
        self.assertEqual(grid.components, 3)
        self.assertIsNone(grid.lambdas)
        self.assertIsNone(grid.loadings)
        self.assertIsNone(grid.factors)

    def test_returns_generator_is_concrete_and_stores_its_provider(self) -> None:
        provider = PCAGridProvider()

        generator = ReturnsVolaGridScenarioGenerator(provider)

        self.assertIsInstance(generator, ScenarioGenerator)
        self.assertIs(
            generator._ReturnsVolaGridScenarioGenerator__pcaGridProvider,
            provider,
        )

    def test_returns_vola_grid_pca_scenario_fields(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            1,
        )

        scenario = ReturnsVolaGridPCAScenario(key, (1.0, -0.5))

        self.assertIsInstance(scenario, PCAScenario)
        self.assertIs(scenario.pcaKey, key)
        self.assertEqual(scenario.point, (1.0, -0.5))

    def test_returns_generator_uses_an_existing_pca_grid(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            1,
        )
        grid = ReturnsPCAGrid(["AAPL"], key.dates, 0.94, 1)
        provider = Mock(spec=PCAGridProvider)
        provider.getPCAGrid.return_value = grid
        generator = ReturnsVolaGridScenarioGenerator(provider)
        generator.getPCAKey = Mock(return_value=key)
        data = object()

        result = generator.getRiskStates(data)

        generator.getPCAKey.assert_called_once_with(data)
        provider.getPCAGrid.assert_called_once_with(key)
        provider.createPCAGrid.assert_not_called()
        self.assertEqual(result, [])

    def test_returns_generator_creates_a_missing_pca_grid(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            1,
        )
        provider = Mock(spec=PCAGridProvider)
        provider.getPCAGrid.return_value = None
        generator = ReturnsVolaGridScenarioGenerator(provider)
        generator.getPCAKey = Mock(return_value=key)
        data = object()

        result = generator.getRiskStates(data)

        provider.createPCAGrid.assert_called_once_with(key, data)
        self.assertEqual(result, [])

    def test_returns_generator_converts_every_generated_scenario(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            1,
        )
        scenarios = [
            ReturnsVolaGridPCAScenario(key, (0.0,)),
            ReturnsVolaGridPCAScenario(key, (1.0,)),
        ]
        risk_states = [
            ReturnsVolaGridRiskState({}),
            ReturnsVolaGridRiskState({}),
        ]
        provider = Mock(spec=PCAGridProvider)
        provider.getPCAGrid.return_value = ReturnsPCAGrid(
            ["AAPL"], key.dates, key.ew_lambda, key.components
        )
        generator = ReturnsVolaGridScenarioGenerator(provider)
        generator.getPCAKey = Mock(return_value=key)
        generator.generateScenarios = Mock(return_value=scenarios)
        generator.getRiskState = Mock(side_effect=risk_states)

        result = generator.getRiskStates(object())

        generator.generateScenarios.assert_called_once_with(key)
        generator.getRiskState.assert_has_calls(
            [call(scenarios[0]), call(scenarios[1])]
        )
        self.assertEqual(result, risk_states)

    def test_returns_generator_conversion_hooks_are_not_implemented(self) -> None:
        generator = ReturnsVolaGridScenarioGenerator()
        key = ReturnsPCAKey(
            ["AAPL"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            1,
        )

        with self.assertRaises(NotImplementedError):
            generator.getPCAKey(object())
        with self.assertRaises(NotImplementedError):
            generator.getRiskState(PCAScenario(key))

    def test_pca_grid_provider_is_a_singleton_with_shared_grid_dict(self) -> None:
        first = PCAGridProvider()
        second = PCAGridProvider()

        self.assertIs(first, second)
        self.assertEqual(first.pcaGrids, {})

    def test_returns_pca_keys_compare_and_hash_by_their_fields(self) -> None:
        dates = (date(2024, 1, 1), date(2024, 1, 31))
        first = ReturnsPCAKey(["AAPL", "MSFT"], dates, 0.94, 2)
        second = ReturnsPCAKey(iter(["AAPL", "MSFT"]), dates, 0.94, 2)
        different = ReturnsPCAKey(["AAPL"], dates, 0.94, 2)
        different_components = ReturnsPCAKey(
            ["AAPL", "MSFT"], dates, 0.94, 1
        )

        self.assertIsInstance(first, PCAKey)
        self.assertTrue(first.equals(second))
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertNotEqual(first, different)
        self.assertNotEqual(first, different_components)

    def test_returns_pca_key_can_identify_and_cache_a_grid(self) -> None:
        dates = (date(2024, 1, 1), date(2024, 1, 31))
        key = ReturnsPCAKey(["AAPL"], dates, 0.94, 2)
        grid = ReturnsPCAGrid(["AAPL"], dates, 0.94, 2)
        provider = PCAGridProvider()

        provider.pcaGrids[key] = grid

        self.assertTrue(key.equals(grid))
        self.assertIs(provider.getPCAGrid(key), grid)

    def test_provider_returns_none_for_an_unknown_key(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            1,
        )

        self.assertIsNone(PCAGridProvider().getPCAGrid(key))

    def test_provider_removes_one_or_multiple_grids(self) -> None:
        dates = (date(2024, 1, 1), date(2024, 1, 31))
        first_key = ReturnsPCAKey(["AAPL"], dates, 0.94, 2)
        second_key = ReturnsPCAKey(["MSFT"], dates, 0.94, 2)
        third_key = ReturnsPCAKey(["NVDA"], dates, 0.94, 2)
        provider = PCAGridProvider()
        provider.pcaGrids = {
            first_key: ReturnsPCAGrid(["AAPL"], dates, 0.94, 2),
            second_key: ReturnsPCAGrid(["MSFT"], dates, 0.94, 2),
            third_key: ReturnsPCAGrid(["NVDA"], dates, 0.94, 2),
        }

        provider.removePCAGrid(first_key)
        provider.removePCAGrid(first_key)
        provider.removePCAGrids([second_key])

        self.assertEqual(set(provider.pcaGrids), {third_key})

    def test_provider_purges_all_grids(self) -> None:
        dates = (date(2024, 1, 1), date(2024, 1, 31))
        key = ReturnsPCAKey(["AAPL"], dates, 0.94, 2)
        provider = PCAGridProvider()
        provider.pcaGrids[key] = ReturnsPCAGrid(
            ["AAPL"], dates, 0.94, 2
        )

        provider.purgePCAGrids()

        self.assertEqual(provider.pcaGrids, {})

    def test_grid_factory_creates_a_grid_for_a_returns_key(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL", "MSFT"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            3,
        )

        grid = PCAGridFactory.createPCAGrid(key, object())

        self.assertIsInstance(grid, ReturnsPCAGrid)
        self.assertEqual(grid.components, 3)

    def test_provider_creates_and_caches_a_grid_through_the_factory(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            (date(2024, 1, 1), date(2024, 1, 31)),
            0.94,
            1,
        )
        provider = PCAGridProvider()

        grid = provider.createGridPCA(key, object())

        self.assertIsInstance(grid, ReturnsPCAGrid)
        self.assertIs(provider.getPCAGrid(key), grid)


if __name__ == "__main__":
    unittest.main()

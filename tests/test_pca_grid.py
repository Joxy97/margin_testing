"""Tests for PCA grids, keys, providers, and risk-state generators."""

import unittest
from datetime import date
from unittest.mock import Mock, call

import numpy
import pandas
from sklearn.preprocessing import StandardScaler

from download_unit import DataRequest
from portfolio import Portfolio
from risk_state_generator import (
    CacheFactory,
    CorrelatedReturnsVolaGridRiskStateGenerator,
    CorrelatedReturnsVolaGridRiskState,
    CorrelationFactors,
    PCAGrid,
    PCAGridFactory,
    PCAGridProvider,
    PCAKey,
    PCAScenario,
    ReturnsPCAGrid,
    ReturnsPCAKey,
    ReturnsVolaGridRiskState,
    ReturnsVolaGridPCAScenario,
    ReturnsVolaGridRiskStateGenerator,
    RiskStateGenerator,
    RiskStateGenerationContext,
)


class PCAGridTest(unittest.TestCase):
    def setUp(self) -> None:
        provider = PCAGridProvider()
        provider.setCache(CacheFactory.createCache("lru"))

    @staticmethod
    def _price_data() -> pandas.DataFrame:
        observations = numpy.arange(35, dtype=float)
        return pandas.DataFrame(
            {
                "date": pandas.date_range("2023-12-31", periods=35),
                "AAPL": 100.0 + observations + observations % 3,
                "MSFT": 80.0 + 0.5 * observations + observations % 5,
                "NVDA": 60.0 + 1.5 * observations + observations % 2,
            }
        )

    @staticmethod
    def _generation_context(
        key: ReturnsPCAKey,
        data: object,
    ) -> RiskStateGenerationContext:
        request = DataRequest(
            instruments=key.instruments,
            start_date=key.start_date,
            end_date=key.start_date,
            data_type="closePrices",
        )
        return RiskStateGenerationContext(
            marketData=data,  # type: ignore[arg-type]
            dataRequest=request,
            marginDate=key.start_date,
        )

    def test_returns_pca_grid_constructor_and_calculated_fields(self) -> None:
        start_date = date(2024, 1, 1)

        grid = ReturnsPCAGrid(["AAPL"], 30, start_date, 0.94, 3)

        self.assertIsInstance(grid, PCAGrid)
        self.assertEqual(grid.ew_window, 30)
        self.assertEqual(grid.current_date, start_date)
        self.assertEqual(grid.ew_lambda, 0.94)
        self.assertEqual(grid.components, 3)
        self.assertIsNone(grid.lambdas)
        self.assertIsNone(grid.explained)
        self.assertIsNone(grid.loadings)
        self.assertIsNone(grid.factors)
        self.assertIsNone(grid.pcaMean)
        self.assertIsNone(grid.residuals)
        self.assertIsNone(grid.maxAbsoluteZ)
        self.assertIsNone(grid.logReturnMean)
        self.assertIsNone(grid.logReturnScale)

    def test_returns_generator_is_concrete_and_stores_its_provider(self) -> None:
        provider = PCAGridProvider()

        generator = ReturnsVolaGridRiskStateGenerator(provider)

        self.assertIsInstance(generator, RiskStateGenerator)
        self.assertIs(
            generator._ReturnsVolaGridRiskStateGenerator__pcaGridProvider,
            provider,
        )
        self.assertEqual(generator.scenariosPerComponents, ())

    def test_returns_vola_grid_pca_scenario_fields(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            30,
            date(2024, 1, 1),
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
            30,
            date(2024, 1, 1),
            0.94,
            1,
        )
        grid = ReturnsPCAGrid(
            ["AAPL"], key.ew_window, key.start_date, 0.94, 1
        )
        provider = Mock(spec=PCAGridProvider)
        provider.getPCAGrid.return_value = grid
        generator = ReturnsVolaGridRiskStateGenerator(
            provider,
            ew_window=key.ew_window,
            ew_lambda=key.ew_lambda,
            components=key.components,
        )
        generator._generatePCAScenarios = Mock(return_value=[])
        context = self._generation_context(key, object())

        result = list(generator.getRiskStates(context))

        provider.getPCAGrid.assert_called_once_with(key)
        provider.createPCAGrid.assert_not_called()
        self.assertEqual(result, [])

    def test_returns_generator_creates_a_missing_pca_grid(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            30,
            date(2024, 1, 1),
            0.94,
            1,
        )
        provider = Mock(spec=PCAGridProvider)
        provider.getPCAGrid.return_value = None
        grid = ReturnsPCAGrid(
            ["AAPL"], key.ew_window, key.start_date, 0.94, 1
        )
        provider.createPCAGrid.return_value = grid
        generator = ReturnsVolaGridRiskStateGenerator(
            provider,
            ew_window=key.ew_window,
            ew_lambda=key.ew_lambda,
            components=key.components,
        )
        generator._generatePCAScenarios = Mock(return_value=[])
        data = object()
        context = self._generation_context(key, data)

        result = list(generator.getRiskStates(context))

        provider.createPCAGrid.assert_called_once_with(key, data)
        self.assertEqual(result, [])

    def test_returns_generator_converts_every_generated_scenario(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            30,
            date(2024, 1, 1),
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
            ["AAPL"],
            key.ew_window,
            key.start_date,
            key.ew_lambda,
            key.components,
        )
        generator = ReturnsVolaGridRiskStateGenerator(
            provider,
            ew_window=key.ew_window,
            ew_lambda=key.ew_lambda,
            components=key.components,
            scenariosPerComponents=(1,),
        )
        generate_scenarios = Mock(return_value=scenarios)
        generator._generatePCAScenarios = generate_scenarios
        generator.getRiskState = Mock(side_effect=risk_states)
        context = self._generation_context(key, object())

        result = list(generator.getRiskStates(context))

        generate_scenarios.assert_called_once_with(
            key,
            provider.getPCAGrid.return_value,
            (1,),
            1.0,
        )
        generator.getRiskState.assert_has_calls(
            [
                call(
                    scenarios[0],
                    provider.getPCAGrid.return_value,
                    context,
                ),
                call(
                    scenarios[1],
                    provider.getPCAGrid.return_value,
                    context,
                ),
            ]
        )
        self.assertEqual(result, risk_states)

    def test_returns_generator_stores_its_pca_configuration(self) -> None:
        generator = ReturnsVolaGridRiskStateGenerator(
            ew_window=30,
            ew_lambda=0.94,
            components=2,
        )

        self.assertEqual(generator.ew_window, 30)
        self.assertEqual(generator.ew_lambda, 0.94)
        self.assertEqual(generator.components, 2)
        self.assertFalse(hasattr(generator, "pcaKey"))

    def test_returns_generator_builds_one_conditioned_risk_state(self) -> None:
        data = self._price_data()
        key = ReturnsPCAKey(
            ["AAPL", "MSFT"], 30, date(2024, 1, 31), 0.94, 1
        )
        grid = ReturnsPCAGrid.construct(key, data)
        scenario = ReturnsVolaGridPCAScenario(key, (0.0,))
        generator = ReturnsVolaGridRiskStateGenerator(
            scenariosPerComponents=(3,),
            nZBins=5,
            nNearest=10,
        )

        risk_state = generator.getRiskState(
            scenario,
            grid,
            self._generation_context(key, data),
        )

        self.assertIsInstance(risk_state, ReturnsVolaGridRiskState)
        self.assertEqual(
            set(risk_state.returnsVolaGrid), {"AAPL", "MSFT"}
        )
        for instrument_grid in risk_state.returnsVolaGrid.values():
            self.assertEqual(instrument_grid.shape[1], 2)
            self.assertGreaterEqual(len(instrument_grid), 1)
            self.assertLessEqual(len(instrument_grid), 5)
            self.assertTrue(numpy.isfinite(instrument_grid).all())
            self.assertTrue((instrument_grid[:, 1] > 0.0).all())
        self.assertIsNotNone(risk_state.returnBounds)
        expected_bounds = numpy.array(
            [
                [grid[:, 0].min(), grid[:, 0].max()]
                for grid in risk_state.returnsVolaGrid.values()
            ]
        )
        numpy.testing.assert_allclose(risk_state.returnBounds, expected_bounds)

    def test_correlated_generator_adds_state_pair_coefficients(self) -> None:
        data = self._price_data()
        key = ReturnsPCAKey(
            ["AAPL", "MSFT", "NVDA"],
            30,
            date(2024, 1, 31),
            0.94,
            1,
        )
        grid = ReturnsPCAGrid.construct(key, data)
        scenario = ReturnsVolaGridPCAScenario(key, (0.0,))
        generator = CorrelatedReturnsVolaGridRiskStateGenerator(
            scenariosPerComponents=(3,),
            nZBins=5,
            nNearest=10,
            topKNeighbors=2,
        )

        risk_state = generator.getRiskState(
            scenario,
            grid,
            {"data": object()},
        )

        self.assertIsInstance(
            risk_state,
            CorrelatedReturnsVolaGridRiskState,
        )
        self.assertEqual(
            set(risk_state.returnsVolaGrid),
            {"AAPL", "MSFT", "NVDA"},
        )
        self.assertTrue(risk_state.correlations)
        self.assertIsInstance(risk_state.correlations, CorrelationFactors)
        self.assertTrue(numpy.all(risk_state.correlations.coefficients > 0.0))
        self.assertTrue(numpy.all(risk_state.correlations.firstAssets >= 0))
        self.assertTrue(numpy.all(risk_state.correlations.secondAssets >= 0))

    def test_correlated_generator_uses_marginlab_compatibility_formula(
        self,
    ) -> None:
        generator = CorrelatedReturnsVolaGridRiskStateGenerator()
        correlations = generator._getStatePairCoefficients(
            standardizedGrids=[
                numpy.array([-1.0, 1.0]),
                numpy.array([-2.0, 2.0]),
            ],
            scenarioCenter=numpy.array([0.0, 0.0]),
            conditionalStd=numpy.array([2.0, 3.0]),
            neighborIndices=numpy.array([[1], [0]]),
            neighborCorrelations=numpy.array([[0.5], [0.5]]),
        )

        expected_b = 0.5 * 3.0 / 2.0 * -1.0
        denominator = 3.0**2 * (1.0 - 0.5**2)
        expected_coefficient = (-2.0 - expected_b) ** 2 / denominator
        self.assertAlmostEqual(
            correlations.coefficients[
                (correlations.firstAssets == 0)
                & (correlations.firstStates == 0)
                & (correlations.secondAssets == 1)
                & (correlations.secondStates == 0)
            ][0],
            expected_coefficient,
        )

    def test_blockwise_neighbors_match_dense_correlations(self) -> None:
        rng = numpy.random.default_rng(9)
        residuals = rng.normal(size=(20, 8))
        generator = CorrelatedReturnsVolaGridRiskStateGenerator(
            topKNeighbors=3,
            correlationBlockBytes=8 * 8 * 2,
        )

        standard_deviation, neighbors, correlations = (
            generator._getConditionalNeighbors(
                residuals,
                numpy.ones(8),
            )
        )
        dense = numpy.clip(numpy.corrcoef(residuals.T), -0.999, 0.999)
        absolute = numpy.abs(dense)
        numpy.fill_diagonal(absolute, -numpy.inf)
        expected_neighbors = numpy.argsort(absolute, axis=1)[:, -3:]

        numpy.testing.assert_allclose(
            standard_deviation,
            numpy.std(residuals, axis=0, ddof=1),
        )
        numpy.testing.assert_array_equal(neighbors, expected_neighbors)
        numpy.testing.assert_allclose(
            correlations,
            numpy.take_along_axis(dense, expected_neighbors, axis=1),
        )

    def test_returns_generator_builds_the_factor_scenario_grid(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL", "MSFT"], 30, date(2024, 1, 1), 0.94, 2
        )
        grid = ReturnsPCAGrid(
            ["AAPL", "MSFT"], 30, date(2024, 1, 1), 0.94, 2
        )
        grid.lambdas = numpy.array([4.0, 9.0])
        generator = ReturnsVolaGridRiskStateGenerator(
            ew_window=key.ew_window,
            ew_lambda=key.ew_lambda,
            components=key.components,
            scenariosPerComponents=(3, 5)
        )

        scenarios = list(
            generator._generatePCAScenarios(
                key,
                grid,
                generator.scenariosPerComponents,
                generator.tailDensityGamma,
            )
        )
        points = numpy.asarray([scenario.point for scenario in scenarios])

        self.assertFalse(hasattr(generator, "generateScenarios"))
        self.assertFalse(hasattr(generator, "generatePCAScenarios"))
        self.assertEqual(points.shape, (15, 2))
        numpy.testing.assert_allclose(numpy.unique(points[:, 0]), [-2, 0, 2])
        numpy.testing.assert_allclose(
            numpy.unique(points[:, 1]), [-6, -3, 0, 3, 6]
        )
        self.assertTrue(all(scenario.pcaKey is key for scenario in scenarios))

    def test_returns_generator_validates_scenarios_per_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integers"):
            ReturnsVolaGridRiskStateGenerator(scenariosPerComponents=(3, 0))

        key = ReturnsPCAKey(["AAPL"], 30, date(2024, 1, 1), 0.94, 1)
        grid = ReturnsPCAGrid(["AAPL"], 30, date(2024, 1, 1), 0.94, 1)
        grid.lambdas = numpy.array([1.0])
        generator = ReturnsVolaGridRiskStateGenerator(
            ew_window=key.ew_window,
            ew_lambda=key.ew_lambda,
            components=key.components,
            scenariosPerComponents=(2,)
        )

        with self.assertRaisesRegex(ValueError, "positive odd integers"):
            generator._generatePCAScenarios(
                key,
                grid,
                generator.scenariosPerComponents,
                generator.tailDensityGamma,
            )

        generator.scenariosPerComponents = (1, 3)
        with self.assertRaisesRegex(ValueError, "length must match"):
            generator._generatePCAScenarios(
                key,
                grid,
                generator.scenariosPerComponents,
                generator.tailDensityGamma,
            )

    def test_returns_generator_applies_tail_density_warping(self) -> None:
        key = ReturnsPCAKey(["AAPL"], 30, date(2024, 1, 1), 0.94, 1)
        grid = ReturnsPCAGrid(["AAPL"], 30, date(2024, 1, 1), 0.94, 1)
        grid.lambdas = numpy.array([4.0])
        generator = ReturnsVolaGridRiskStateGenerator(
            ew_window=key.ew_window,
            ew_lambda=key.ew_lambda,
            components=key.components,
            scenariosPerComponents=(5,),
            tailDensityGamma=2.0,
        )

        scenarios = generator._generatePCAScenarios(
            key,
            grid,
            generator.scenariosPerComponents,
            generator.tailDensityGamma,
        )

        numpy.testing.assert_allclose(
            [scenario.point[0] for scenario in scenarios],
            [-4.0, -1.0, 0.0, 1.0, 4.0],
        )

    def test_pca_grid_providers_have_independent_caches(self) -> None:
        first = PCAGridProvider()
        second = PCAGridProvider()

        self.assertIsNot(first, second)
        self.assertIsNot(first.cache, second.cache)
        self.assertEqual(first.cache.memory, {})

    def test_returns_pca_keys_compare_and_hash_by_their_fields(self) -> None:
        start_date = date(2024, 1, 1)
        first = ReturnsPCAKey(["AAPL", "MSFT"], 30, start_date, 0.94, 2)
        second = ReturnsPCAKey(
            iter(["AAPL", "MSFT"]), 30, start_date, 0.94, 2
        )
        different = ReturnsPCAKey(["AAPL"], 30, start_date, 0.94, 2)
        different_components = ReturnsPCAKey(
            ["AAPL", "MSFT"], 30, start_date, 0.94, 1
        )
        different_window = ReturnsPCAKey(
            ["AAPL", "MSFT"], 60, start_date, 0.94, 2
        )

        self.assertIsInstance(first, PCAKey)
        self.assertTrue(first.equals(second))
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertNotEqual(first, different)
        self.assertNotEqual(first, different_components)
        self.assertNotEqual(first, different_window)

    def test_returns_pca_key_can_cache_a_grid(self) -> None:
        start_date = date(2024, 1, 1)
        key = ReturnsPCAKey(["AAPL"], 30, start_date, 0.94, 2)
        grid = ReturnsPCAGrid(["AAPL"], 30, start_date, 0.94, 2)
        provider = PCAGridProvider()

        provider.cache.insert(key, grid)

        self.assertFalse(key.equals(grid))
        self.assertIs(provider.getPCAGrid(key), grid)

    def test_provider_returns_none_for_an_unknown_key(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            30,
            date(2024, 1, 1),
            0.94,
            1,
        )

        self.assertIsNone(PCAGridProvider().getPCAGrid(key))

    def test_provider_allows_its_cache_to_be_replaced(self) -> None:
        provider = PCAGridProvider()
        replacement = CacheFactory.createCache("lru", 2)

        provider.setCache(replacement)

        self.assertIs(provider.cache, replacement)

    def test_grid_factory_creates_a_grid_for_a_returns_key(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL", "MSFT"],
            30,
            date(2024, 1, 31),
            0.94,
            2,
        )

        grid = PCAGridFactory.createPCAGrid(key, self._price_data())

        self.assertIsInstance(grid, ReturnsPCAGrid)
        self.assertEqual(grid.ew_window, 30)
        self.assertEqual(grid.current_date, date(2024, 1, 31))
        self.assertEqual(grid.components, 2)
        self.assertEqual(grid.loadings.shape, (2, 2))
        self.assertEqual(grid.factors.shape, (30, 2))

    def test_provider_creates_and_caches_a_grid_through_the_factory(self) -> None:
        key = ReturnsPCAKey(
            ["AAPL"],
            30,
            date(2024, 1, 31),
            0.94,
            1,
        )
        provider = PCAGridProvider()

        grid = provider.createPCAGrid(key, self._price_data())

        self.assertIsInstance(grid, ReturnsPCAGrid)
        self.assertIs(provider.getPCAGrid(key), grid)

    def test_provider_evicts_the_least_recently_used_grid(self) -> None:
        start_date = date(2024, 1, 31)
        first_key = ReturnsPCAKey(["AAPL"], 30, start_date, 0.94, 1)
        second_key = ReturnsPCAKey(["MSFT"], 30, start_date, 0.94, 1)
        third_key = ReturnsPCAKey(["NVDA"], 30, start_date, 0.94, 1)
        provider = PCAGridProvider()
        provider.setCache(CacheFactory.createCache("lru", 2))

        data = self._price_data()
        first_grid = provider.createPCAGrid(first_key, data)
        provider.createPCAGrid(second_key, data)
        self.assertIs(provider.getPCAGrid(first_key), first_grid)
        provider.createPCAGrid(third_key, data)

        self.assertIsNone(provider.getPCAGrid(second_key))
        self.assertIs(provider.getPCAGrid(first_key), first_grid)
        self.assertIsNotNone(provider.getPCAGrid(third_key))

    def test_returns_pca_grid_constructs_weighted_centered_pca(self) -> None:
        data = pandas.DataFrame(
            {
                "date": pandas.date_range("2024-01-01", periods=6),
                "AAPL": [100.0, 104.0, 102.0, 108.0, 105.0, 120.0],
                "MSFT": [50.0, 49.0, 53.0, 51.0, 56.0, 60.0],
            }
        )
        key = ReturnsPCAKey(
            ["AAPL", "MSFT"],
            3,
            date(2024, 1, 5),
            0.5,
            2,
        )

        grid = ReturnsPCAGrid.construct(key, data)

        prices = data.loc[0:3, ["AAPL", "MSFT"]]
        log_returns = numpy.log(prices / prices.shift(1)).iloc[1:]
        standardized = StandardScaler().fit_transform(log_returns)
        weights = numpy.array([0.25, 0.5, 1.0])
        weights /= weights.sum()
        expected_mean = numpy.sum(weights[:, None] * standardized, axis=0)
        centered = standardized - expected_mean
        covariance = centered.T @ (weights[:, None] * centered)
        expected_eigenvalues, eigenvectors = numpy.linalg.eigh(covariance)
        order = numpy.argsort(expected_eigenvalues)[::-1]
        expected_eigenvalues = expected_eigenvalues[order]
        expected_loadings = eigenvectors[:, order].T
        expected_factors = centered @ expected_loadings.T

        self.assertEqual(grid.current_date, date(2024, 1, 5))
        numpy.testing.assert_allclose(grid.lambdas, expected_eigenvalues)
        numpy.testing.assert_allclose(
            grid.explained,
            expected_eigenvalues / expected_eigenvalues.sum(),
        )
        numpy.testing.assert_allclose(grid.loadings, expected_loadings)
        numpy.testing.assert_allclose(grid.factors, expected_factors)
        numpy.testing.assert_allclose(grid.pcaMean, expected_mean)
        expected_residuals = standardized - (
            expected_mean + expected_factors @ expected_loadings
        )
        numpy.testing.assert_allclose(grid.residuals, expected_residuals)
        numpy.testing.assert_allclose(
            grid.maxAbsoluteZ,
            numpy.max(numpy.abs(standardized), axis=0),
        )
        numpy.testing.assert_allclose(
            grid.logReturnMean,
            StandardScaler().fit(log_returns).mean_,
        )
        numpy.testing.assert_allclose(
            grid.logReturnScale,
            StandardScaler().fit(log_returns).scale_,
        )

    def test_returns_pca_grid_aligns_missing_prices_like_marginlab(self) -> None:
        data = pandas.DataFrame(
            {
                "date": pandas.date_range("2024-01-01", periods=7),
                "AAPL": [
                    100.0,
                    101.0,
                    numpy.nan,
                    103.0,
                    104.0,
                    105.0,
                    999.0,
                ],
                "MSFT": [
                    numpy.nan,
                    50.0,
                    51.0,
                    52.0,
                    53.0,
                    54.0,
                    999.0,
                ],
            }
        )
        key = ReturnsPCAKey(
            ["AAPL", "MSFT"],
            3,
            date(2024, 1, 7),
            0.94,
            2,
        )

        grid = ReturnsPCAGrid.construct(key, data)

        self.assertEqual(grid.factors.shape, (3, 2))
        self.assertTrue(numpy.isfinite(grid.factors).all())

    def test_wide_returns_pca_matches_dense_covariance(self) -> None:
        rng = numpy.random.default_rng(7)
        instruments = tuple(f"asset_{index}" for index in range(12))
        returns = rng.normal(0.0, 0.01, size=(6, len(instruments)))
        prices = 100.0 * numpy.exp(
            numpy.vstack((numpy.zeros(12), returns)).cumsum(axis=0)
        )
        data = pandas.DataFrame(prices, columns=instruments)
        data.insert(0, "date", pandas.date_range("2024-01-01", periods=7))
        key = ReturnsPCAKey(
            instruments,
            6,
            date(2024, 1, 8),
            0.94,
            3,
        )

        grid = ReturnsPCAGrid.construct(key, data)
        standardized = StandardScaler().fit_transform(returns)
        weights = key.ew_lambda ** numpy.arange(5, -1, -1, dtype=float)
        weights /= weights.sum()
        mean = numpy.sum(weights[:, None] * standardized, axis=0)
        centered = standardized - mean
        covariance = centered.T @ (weights[:, None] * centered)
        expected = numpy.linalg.eigvalsh(covariance)[::-1][:3]

        numpy.testing.assert_allclose(grid.lambdas, expected, atol=1e-12)
        numpy.testing.assert_allclose(
            grid.loadings @ grid.loadings.T,
            numpy.eye(3),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()

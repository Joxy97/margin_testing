"""Tests for deterministic filtered-historical EVT risk-state generation."""

import math
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

import numpy
import pandas
import yaml

from download_unit import DataRequest
from margin_calculator import StateAwareGreedyRiskStateVisitor
from margin_engine import MarginApplicationConfig
from portfolio import Portfolio
from risk_state_generator import (
    ConditionalFilter,
    DeterministicRule,
    EVTMarginalFitter,
    FHSEVTRiskState,
    FHSEVTRiskStateGenerator,
    FHSEVTRiskStateGeneratorConfig,
    FHSScenario,
    GeneralizedPareto,
    PortfolioRiskStateBQMVisitor,
    ScenarioReducer,
    ScenarioType,
    buildJointTailStresses,
    constructDependenceCells,
)
from risk_state_generator.risk_state_generation_context import (
    RiskStateGenerationContext,
)


class FHSEVTComponentsTest(unittest.TestCase):
    def test_gjr_variance_recursion_uses_the_previous_shock(self) -> None:
        innovations = numpy.array([1.0, -2.0, 3.0])

        variances = ConditionalFilter._variancePath(
            innovations,
            0,
            omega=0.1,
            alpha=0.2,
            gamma=0.3,
            beta=0.4,
        )

        initial = numpy.var(innovations, ddof=1)
        expected_second = 0.1 + 0.2 * 1.0 + 0.4 * initial
        expected_third = 0.1 + 0.2 * 4.0 + 0.3 * 4.0 + 0.4 * expected_second
        numpy.testing.assert_allclose(
            variances,
            [initial, expected_second, expected_third],
        )

    def test_zero_asymmetry_is_the_garch_recursion(self) -> None:
        innovations = numpy.array([-1.0, 2.0, -3.0, 1.0])

        first = ConditionalFilter._variancePath(
            innovations, 0, 0.1, 0.2, 0.0, 0.7
        )
        second = ConditionalFilter._variancePath(
            innovations, 0, 0.1, 0.2, 0.0, 0.7
        )

        numpy.testing.assert_array_equal(first, second)

    def test_primary_gjr_filter_fits_a_deterministic_gjr_process(self) -> None:
        generator = numpy.random.default_rng(3)
        observations = 600
        innovations = numpy.zeros(observations)
        variances = numpy.full(observations, 0.0001)
        normals = generator.standard_normal(observations)
        innovations[0] = math.sqrt(variances[0]) * normals[0]
        for index in range(1, observations):
            previous = innovations[index - 1]
            variances[index] = (
                2e-6
                + 0.06 * previous**2
                + 0.08 * float(previous < 0.0) * previous**2
                + 0.88 * variances[index - 1]
            )
            innovations[index] = math.sqrt(variances[index]) * normals[index]

        fitted = ConditionalFilter(
            meanModel="zero",
            varianceModel="gjr_garch",
            burnIn=40,
            optimizerMaxIterations=500,
        ).fit(innovations)

        self.assertEqual(fitted.modelFamily, "gjr_garch")
        self.assertFalse(fitted.usedFallback)
        self.assertLess(fitted.parameters["persistence"], 0.995 + 1e-12)
        self.assertGreater(fitted.varianceForecast, 0.0)

    def test_frozen_filter_parameters_reproduce_the_calibration_forecast(self) -> None:
        index = numpy.arange(120, dtype=float)
        values = 0.01 * numpy.sin(index / 4.0) + 0.002 * numpy.cos(index)
        conditional_filter = ConditionalFilter(
            meanModel="ar",
            arOrder=3,
            varianceModel="ewma",
            burnIn=10,
        )

        calibrated = conditional_filter.fit(values)
        reapplied = conditional_filter.applyCalibrated(values, calibrated)

        numpy.testing.assert_allclose(reapplied.mean, calibrated.mean)
        numpy.testing.assert_allclose(reapplied.variances, calibrated.variances)
        self.assertAlmostEqual(reapplied.meanForecast, calibrated.meanForecast)
        self.assertAlmostEqual(
            reapplied.volatilityForecast,
            calibrated.volatilityForecast,
        )

    def test_gpd_quantile_inverts_the_cdf_for_all_shape_signs(self) -> None:
        probabilities = numpy.linspace(0.01, 0.99, 40)
        for shape in (-0.2, 0.0, 0.2):
            distribution = GeneralizedPareto(shape, 1.3)
            values = distribution.quantile(probabilities)

            numpy.testing.assert_allclose(
                distribution.cdf(values), probabilities, atol=1e-12
            )

    def test_evt_marginal_is_monotone_and_moment_normalized(self) -> None:
        probabilities = (numpy.arange(400, dtype=float) + 0.5) / 400.0
        residuals = numpy.log(probabilities / (1.0 - probabilities))
        weights = numpy.full(400, 1.0 / 400.0)
        marginal = EVTMarginalFitter(
            tailMassCandidates=(0.05, 0.10),
            minimumTailObservations=15,
            quadraturePoints=256,
        ).fit(residuals, weights)

        grid = numpy.linspace(1e-5, 1.0 - 1e-5, 2000)
        quantiles = marginal.quantile(grid)
        nodes, node_weights = numpy.polynomial.legendre.leggauss(256)
        integration_grid = 0.5 * (nodes + 1.0)
        integration_weights = 0.5 * node_weights
        integrated = marginal.quantile(integration_grid)

        self.assertTrue(numpy.all(numpy.diff(quantiles) >= 0.0))
        self.assertAlmostEqual(float(integration_weights @ integrated), 0.0, places=10)
        self.assertAlmostEqual(
            float(integration_weights @ integrated**2), 1.0, places=10
        )

    def test_weighted_dependence_cells_partition_every_factor(self) -> None:
        residuals = numpy.array(
            [[3.0, -1.0], [1.0, 2.0], [2.0, 0.0]], dtype=float
        )
        weights = numpy.array([0.2, 0.5, 0.3])

        cells = constructDependenceCells(
            residuals,
            weights,
            (date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)),
            ("r1", "r2", "r3"),
        )

        numpy.testing.assert_allclose(
            cells.upper - cells.lower,
            numpy.repeat(weights[:, None], 2, axis=1),
        )
        for factor in range(2):
            intervals = sorted(zip(cells.lower[:, factor], cells.upper[:, factor]))
            self.assertAlmostEqual(intervals[0][0], 0.0)
            self.assertAlmostEqual(intervals[-1][1], 1.0)
            for previous, current in zip(intervals, intervals[1:]):
                self.assertAlmostEqual(previous[1], current[0])

    def test_fixed_rule_is_reproducible_and_avoids_endpoints(self) -> None:
        first = DeterministicRule.build(4, 7)
        second = DeterministicRule.build(4, 7)

        numpy.testing.assert_array_equal(first.nodes, second.nodes)
        numpy.testing.assert_array_equal(first.weights, second.weights)
        self.assertTrue(numpy.all((first.nodes > 0.0) & (first.nodes < 1.0)))
        numpy.testing.assert_allclose(
            first.weights @ first.nodes,
            numpy.full(4, 0.5),
            atol=1e-15,
        )

    def test_joint_stresses_preserve_observed_crisis_sign_directions(self) -> None:
        residuals = numpy.linspace(-3.0, 3.0, 200)
        weights = numpy.full(len(residuals), 1.0 / len(residuals))
        marginals = tuple(
            EVTMarginalFitter(
                tailMassCandidates=(0.10,),
                minimumTailObservations=10,
                quadraturePoints=128,
            ).fit(residuals + shift, weights)
            for shift in (0.0, 0.2, -0.1)
        )
        directions = numpy.array(
            [[-1.5, -0.7, 0.4], [1.1, -0.3, 1.8]],
            dtype=float,
        )

        stresses = buildJointTailStresses(
            marginals,
            numpy.zeros(3),
            numpy.full(3, 0.01),
            directions,
            ("lower-owner", "upper-owner"),
            sigmaLevels=(3,),
        )

        self.assertEqual(len(stresses), 2)
        numpy.testing.assert_array_equal(
            numpy.sign(stresses[0].innovations), numpy.sign(directions[0])
        )
        numpy.testing.assert_array_equal(
            numpy.sign(stresses[1].innovations), numpy.sign(directions[1])
        )
        self.assertEqual(stresses[0].ownerRowId, "lower-owner")
        self.assertEqual(stresses[1].ownerRowId, "upper-owner")

    def test_reducer_preserves_stresses_and_probability_mass(self) -> None:
        probability = tuple(
            FHSScenario(
                scenarioId=f"PROB:{index:03d}",
                scenarioType=ScenarioType.PROBABILITY,
                probability=1.0 / 20.0,
                innovations=numpy.array([float(index), float(index % 3)]),
                factorChanges=numpy.array([float(index), float(index % 3)]),
            )
            for index in range(20)
        )
        stresses = tuple(
            FHSScenario(
                scenarioId=f"STRESS:{index}",
                scenarioType=ScenarioType.STRESS,
                probability=0.0,
                innovations=numpy.array([-10.0 - index, -10.0]),
                factorChanges=numpy.array([-1.0 - index, -1.0]),
                protected=True,
                protectionReason="test stress",
            )
            for index in range(2)
        )

        reduced = ScenarioReducer(targetScenarios=8, localSwapPasses=1).reduce(
            probability, stresses
        )

        self.assertEqual(len(reduced), 8)
        self.assertEqual(
            {
                item.scenarioId
                for item in reduced
                if item.scenarioType is ScenarioType.STRESS
            },
            {"STRESS:0", "STRESS:1"},
        )
        self.assertAlmostEqual(
            math.fsum(
                item.probability
                for item in reduced
                if item.scenarioType is ScenarioType.PROBABILITY
            ),
            1.0,
        )
        self.assertTrue(
            all(
                item.probability == 0.0
                for item in reduced
                if item.scenarioType is ScenarioType.STRESS
            )
        )


class FHSEVTRiskStateGeneratorTest(unittest.TestCase):
    @staticmethod
    def _marketData() -> tuple[pandas.DataFrame, date]:
        observations = 170
        dates = pandas.bdate_range("2023-01-02", periods=observations + 1)
        index = numpy.arange(observations, dtype=float)
        returns = numpy.column_stack(
            (
                0.0003 + 0.012 * numpy.sin(index / 5.0),
                0.0001 + 0.009 * numpy.sin(index / 5.0 + 0.7),
            )
        )
        prices = 100.0 * numpy.exp(
            numpy.vstack((numpy.zeros(2), returns)).cumsum(axis=0)
        )
        data = pandas.DataFrame(prices, columns=("AAPL", "MSFT"))
        data.insert(0, "date", dates)
        return data, dates[-1].date()

    @staticmethod
    def _generator(targetScenarios: int = 15) -> FHSEVTRiskStateGenerator:
        return FHSEVTRiskStateGenerator(
            historyDays=800,
            minimumObservations=100,
            varianceModel="ewma",
            burnIn=5,
            tailMassCandidates=(0.10,),
            minimumTailObservations=5,
            evtQuadraturePoints=128,
            targetScenarios=targetScenarios,
            localSwapPasses=0,
        )

    def test_generator_builds_a_history_request_in_stable_portfolio_order(self) -> None:
        generator = self._generator()
        portfolio = Portfolio(
            weights={"MSFT": Decimal("2"), "AAPL": Decimal("1")}
        )
        margin_date = date(2024, 1, 31)

        request = generator.createDataRequest(portfolio, margin_date)

        self.assertEqual(request.instruments, ("AAPL", "MSFT"))
        self.assertEqual(request.start_date, margin_date - timedelta(days=800))
        self.assertEqual(request.end_date, margin_date)
        self.assertEqual(request.data_type, "closePrices")

    def test_generator_rejects_nonboolean_reduction_switch(self) -> None:
        with self.assertRaisesRegex(TypeError, "reduceScenarios must be a bool"):
            FHSEVTRiskStateGenerator(reduceScenarios=1)

    def test_generator_emits_exact_joint_scenarios_deterministically(self) -> None:
        data, margin_date = self._marketData()
        request = DataRequest(
            instruments=("AAPL", "MSFT"),
            start_date=data["date"].iloc[0].date(),
            end_date=margin_date,
            data_type="closePrices",
        )
        context = RiskStateGenerationContext(data, request, margin_date)
        first_generator = self._generator()
        second_generator = self._generator()

        first = tuple(first_generator.getRiskStates(context))
        second = tuple(second_generator.getRiskStates(context))

        self.assertEqual(len(first), 15)
        self.assertTrue(all(isinstance(item, FHSEVTRiskState) for item in first))
        self.assertEqual(
            [item.scenarioId for item in first],
            [item.scenarioId for item in second],
        )
        numpy.testing.assert_array_equal(
            numpy.vstack([item.factorChanges for item in first]),
            numpy.vstack([item.factorChanges for item in second]),
        )
        self.assertEqual(first_generator.lastValidationReport.status, "PASS")
        self.assertEqual(
            sum(item.scenarioType is ScenarioType.STRESS for item in first), 6
        )
        self.assertAlmostEqual(
            math.fsum(
                item.probability
                for item in first
                if item.scenarioType is ScenarioType.PROBABILITY
            ),
            1.0,
        )
        self.assertTrue(
            all(item.returnsVolaGrid.stateCounts.tolist() == [1, 1] for item in first)
        )

        portfolio = Portfolio(
            weights={"AAPL": Decimal("10"), "MSFT": Decimal("-3")}
        )
        state = first[0]
        expected = sum(
            float(portfolio.weights[instrument])
            * float(state.returnsVolaGrid[instrument][0, 0])
            for instrument in state.returnsVolaGrid.instruments
        )
        self.assertAlmostEqual(
            StateAwareGreedyRiskStateVisitor().portfolioPnl(state, portfolio),
            expected,
        )
        problem = PortfolioRiskStateBQMVisitor().createBQM(
            state,
            portfolio,
            {"lambdaOneHot": 2.0},
        )
        self.assertEqual(problem.variableCount, 2)
        self.assertEqual(problem.oneHotGroups, ((0,), (1,)))

    def test_generator_supports_the_production_105_scenario_contract(self) -> None:
        data, margin_date = self._marketData()
        request = DataRequest(
            instruments=("AAPL", "MSFT"),
            start_date=data["date"].iloc[0].date(),
            end_date=margin_date,
            data_type="closePrices",
        )
        generator = self._generator(targetScenarios=105)

        states = tuple(
            generator.getRiskStates(
                RiskStateGenerationContext(data, request, margin_date)
            )
        )

        self.assertEqual(len(states), 105)
        self.assertEqual(generator.lastValidationReport.scenarioCount, 105)

    def test_generator_can_publish_every_parent_without_reduction(self) -> None:
        data, margin_date = self._marketData()
        request = DataRequest(
            instruments=("AAPL", "MSFT"),
            start_date=data["date"].iloc[0].date(),
            end_date=margin_date,
            data_type="closePrices",
        )
        generator = FHSEVTRiskStateGenerator(
            historyDays=800,
            minimumObservations=100,
            varianceModel="ewma",
            burnIn=5,
            tailMassCandidates=(0.10,),
            minimumTailObservations=5,
            evtQuadraturePoints=128,
            stressSigmaLevels=(3,),
            targetScenarios=1,
            reduceScenarios=False,
            localSwapPasses=0,
        )

        states = tuple(
            generator.getRiskStates(
                RiskStateGenerationContext(data, request, margin_date)
            )
        )

        expected = len(generator.lastDependenceCells.weights) + 2
        self.assertEqual(len(states), expected)
        self.assertEqual(generator.lastValidationReport.scenarioCount, expected)
        self.assertEqual(generator.lastValidationReport.probabilityCount, expected - 2)
        self.assertAlmostEqual(
            math.fsum(
                state.probability
                for state in states
                if state.scenarioType is ScenarioType.PROBABILITY
            ),
            1.0,
        )

    def test_generator_reuses_structural_calibration_between_daily_updates(self) -> None:
        data, final_date = self._marketData()
        first_date = data["date"].iloc[-2].date()
        request = DataRequest(
            instruments=("AAPL", "MSFT"),
            start_date=data["date"].iloc[0].date(),
            end_date=final_date,
            data_type="closePrices",
        )
        generator = FHSEVTRiskStateGenerator(
            historyDays=800,
            minimumObservations=100,
            meanModel="ar",
            arOrder=3,
            varianceModel="ewma",
            burnIn=10,
            tailMassCandidates=(0.10,),
            minimumTailObservations=5,
            evtQuadraturePoints=128,
            targetScenarios=15,
            localSwapPasses=0,
            recalibrationIntervalDays=366,
        )

        tuple(
            generator.getRiskStates(
                RiskStateGenerationContext(data, request, first_date)
            )
        )
        calibration_date = generator._calibrationDate
        first_forecasts = tuple(
            item.volatilityForecast for item in generator.lastFilterResults
        )
        tuple(
            generator.getRiskStates(
                RiskStateGenerationContext(data, request, final_date)
            )
        )

        self.assertEqual(generator._calibrationDate, calibration_date)
        self.assertNotEqual(
            tuple(item.volatilityForecast for item in generator.lastFilterResults),
            first_forecasts,
        )

    def test_missing_quote_does_not_create_a_zero_return(self) -> None:
        data, margin_date = self._marketData()
        data.loc[20, "AAPL"] = numpy.nan
        generator = self._generator()

        changes, dates, _ = generator._prepareMarketData(
            data, ("AAPL", "MSFT"), margin_date
        )

        self.assertEqual(len(changes), len(data) - 4)
        self.assertNotIn(data.loc[20, "date"].date(), dates)
        self.assertNotIn(data.loc[21, "date"].date(), dates)

    def test_yaml_registers_fhs_evt_and_converts_sequence_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "margin.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "marginDate": "2024-01-31",
                        "portfolio": {"weights": {"AAPL": "1"}},
                        "engine": {
                            "riskStateGenerator": {
                                "type": "fhs_evt",
                                "varianceModel": "ewma",
                                "tailMassCandidates": [0.05, 0.10],
                                "stressSigmaLevels": [3, 4],
                                "reduceScenarios": False,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            application = MarginApplicationConfig.fromYaml(path)

        config = application.engine.riskStateGenerator
        self.assertIsInstance(config, FHSEVTRiskStateGeneratorConfig)
        self.assertEqual(config.tailMassCandidates, (0.05, 0.10))
        self.assertEqual(config.stressSigmaLevels, (3, 4))
        self.assertFalse(config.reduceScenarios)


if __name__ == "__main__":
    unittest.main()

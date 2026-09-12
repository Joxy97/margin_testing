"""Independent price references and mixed-option calibration/search invariants."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from decimal import Decimal
from itertools import product
import unittest

import numpy as np

from option_pricing import (
    AmericanEquityBinomialPricingModel, EquityBlackScholesPricingModel,
    JuZhongPricingModel, NonsmoothOptionError, OptionBook, OptionPricingError,
    VanillaPriceContext,
)
from portfolio import DerivativePosition, EquityOptionContract, FuturesOptionContract, Portfolio
from risk_state_generator import FactorStressModel, OptionFactorStressModel
from margin_calculator.optimization import (
    FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective, solveOptionFactorStress,
)
from margin_calculator.optimization.factor_stress_repair import FactorStressRepair, FactorStressRepairConfig
from margin_calculator.optimization.optimization_result import BQMOptimizationResult


TODAY = date(2025, 3, 3)


def make_book(styles=("A", "E"), types=("P", "C"), quantities=("-2", "1"),
              spots=(100.,), expiry_days=180, horizon_days=1, strike="100", supplied=None):
    positions = tuple(DerivativePosition(EquityOptionContract("A", TODAY+timedelta(days=expiry_days),
        Decimal(strike), kind, style, Decimal("100"), dividendYield=.03), Decimal(quantity))
        for style, kind, quantity in zip(styles, types, quantities))
    marks = [float(VanillaPriceContext(float(strike), expiry_days/365., .04, .03, .27,
        p.contract.optionType, p.contract.exerciseStyle).evaluate(spots[0])) for p in positions]
    return OptionBook(("A",), spots, positions, marks, TODAY, TODAY+timedelta(days=horizon_days), .04,
                      suppliedVolatilities=supplied)


class CenterSolver:
    """Deliberately poor solver: boundary seeds must independently find risk."""

    def solve(self, problem, parameters):
        sample = np.ones(problem.variableCount, dtype=np.uint8)
        return BQMOptimizationResult(sample, problem.energy(sample))


class JuZhongTest(unittest.TestCase):
    def test_document_prices_and_boundary_regression(self):
        call = VanillaPriceContext(100., .5, .03, .07, .2, "C", "A")
        put = VanillaPriceContext(100., 3., .08, 0., .2, "P", "A")
        self.assertAlmostEqual(float(call.evaluate(100.)), 4.76818024, places=7)
        self.assertAlmostEqual(float(put.evaluate(100.)), 6.95561233, places=7)
        self.assertAlmostEqual(put.boundary, 82.2759889300, places=8)
        self.assertAlmostEqual(put.premium, 8.3733727010, places=8)
        self.assertLess(abs(put.boundaryResidual), 1e-9)

    def test_independent_refining_crr_reference(self):
        for kind, time, rate, yield_ in (("C", .5, .03, .07), ("P", 3., .08, 0.)):
            ju = JuZhongPricingModel().price(100., 100., time, rate, .2, kind, yield_)
            prices = [AmericanEquityBinomialPricingModel(steps).price(100., 100., time, rate, .2, kind, yield_)
                      for steps in (800, 1600)]
            self.assertLess(abs(prices[1]-prices[0]), .01)
            self.assertLess(abs(ju-prices[1]), .04)
            self.assertGreater(abs(ju-prices[1]), .001)  # Approximation is visible.

    def test_zero_rate_limits_and_no_exercise_shortcuts(self):
        for kind, rate, yield_ in (("C", .04, 0.), ("P", 0., .04), ("P", 0., 0.)):
            c = VanillaPriceContext(100., 1., rate, yield_, .2, kind, "A")
            self.assertIsNone(c.boundary)
            for spot in (60., 100., 160.):
                expected = EquityBlackScholesPricingModel().price(spot, 100., 1., rate, .2, kind, yield_)
                self.assertAlmostEqual(float(c.evaluate(spot)), expected, places=12)
        zero = VanillaPriceContext(100., 1., 0., .04, .2, "C", "A")
        near = replace(zero, riskFreeRate=1e-10)
        np.testing.assert_allclose(zero.evaluate([90., 100., 130.]), near.evaluate([90., 100., 130.]), atol=1e-8)

    def test_log_derivatives_on_both_branches_and_gamma_sign(self):
        for kind in ("C", "P"):
            context = VanillaPriceContext(100., 1., .04, .03, .27, kind, "A")
            for spot in (context.boundary*.8, context.boundary*1.2, 100.):
                value, first, second = context.evaluate(spot, derivatives=True)
                bump = 1e-5
                low = context.evaluate(spot*np.exp(-bump), derivatives=True)
                high = context.evaluate(spot*np.exp(bump), derivatives=True)
                self.assertAlmostEqual(float(first), float((high[0]-low[0])/(2*bump)), delta=2e-6)
                self.assertAlmostEqual(float(second), float((high[1]-low[1])/(2*bump)), delta=2e-5)
            _, d, g = replace(context, exerciseStyle="E").evaluate(100., derivatives=True)
            self.assertGreater(float(g-d), 0.)

    def test_boundary_value_matching_slope_jump_and_nonsmooth_rejection(self):
        c = VanillaPriceContext(100., 3., .08, 0., .2, "P", "A")
        self.assertAlmostEqual(float(c.evaluate(c.boundary)), 100-c.boundary)
        with self.assertRaises(NonsmoothOptionError):
            c.evaluate(c.boundary, derivatives=True)
        spot = c.boundary*np.exp(1e-8)
        delta = float(c.evaluate(spot, derivatives=True)[1])/spot
        self.assertAlmostEqual(delta+1, c.c*c.premium/c.boundary, delta=1e-7)
        self.assertAlmostEqual(c.c*c.premium/c.boundary, .03133987185, places=9)

    def test_domain_screening_expiry_and_invalid_parameters(self):
        c = VanillaPriceContext(100., 1., .04, .03, .27, "C", "A")
        item = c.screenInterval(60., 180.)
        self.assertGreater(item["denominator_minimum"], 0.)
        expiry = replace(c, timeToExpiry=0.)
        np.testing.assert_array_equal(expiry.evaluate([80., 100., 120.]), [0., 0., 20.])
        with self.assertRaises(NonsmoothOptionError):
            expiry.evaluate(100., derivatives=True)
        for kwargs in ({"riskFreeRate": -.01}, {"dividendYield": -.01}, {"volatility": 0.},
                       {"timeToExpiry": -1.}, {"strike": float("inf")}):
            with self.assertRaises(ValueError):
                replace(c, **kwargs)
        for spots in ([0.], [float("nan")], [-1.]):
            with self.assertRaises(ValueError):
                c.evaluate(spots)
        with self.assertRaises(FrozenInstanceError):
            c.boundary = 0.


class OptionBookTest(unittest.TestCase):
    def test_mixed_style_calibration_mark_anchor_and_immutable_diagnostics(self):
        options = make_book(horizon_days=0)
        np.testing.assert_allclose(options.impliedVolatilities, .27, atol=1e-10)
        self.assertAlmostEqual(float(options.pnl([0.])), 0., places=7)
        self.assertEqual([x.status for x in options.calibrations], ["bracketed_root", "calibrated"])
        self.assertTrue(all(abs(x.residual) < 1e-9 for x in options.calibrations))
        for arr in (options.spotPrices, options.marketPrices, options.impliedVolatilities):
            with self.assertRaises(ValueError):
                arr.setflags(write=True)
        self.assertEqual(options.pnl(np.zeros((2, 3, 1))).shape, (2, 3))
        self.assertEqual(options.pnl(np.zeros((0, 1))).shape, (0,))

    def test_american_plateau_requires_explicit_iv_and_preserves_mark(self):
        options = make_book(styles=("A",), types=("P",), quantities=("-1",), spots=(20.,), supplied=(.27,))
        self.assertEqual(options.marketPrices[0], 80.)
        with self.assertRaisesRegex(ValueError, "unidentifiable"):
            replace(options, suppliedVolatilities=None)
        shifted = replace(options, marketPrices=[80.1])
        self.assertEqual(shifted.calibrations[0].status, "supplied")
        self.assertAlmostEqual(shifted.calibrations[0].residual, -.1)
        self.assertAlmostEqual(float(shifted.pnl([0.])), 10.)

    def test_bounds_currencies_styles_and_horizon_validation(self):
        options = make_book()
        for changes in ({"marketPrices": [101., 1.]}, {"volatilityBounds": (.3, .4)},
                        {"currency": "EUR"}, {"suppliedVolatilities": (.2,)},
                        {"suppliedVolatilities": (float("nan"), .2)},
                        {"suppliedVolatilities": (.27, .27), "marketPrices": [101., 1.]},
                        {"horizonDate": TODAY+timedelta(days=181)}, {"riskFreeRate": -.01}):
            with self.assertRaises(ValueError):
                replace(options, **changes)
        future = FuturesOptionContract("A", TODAY+timedelta(days=180), Decimal(100), "C")
        with self.assertRaisesRegex(TypeError, "equity option"):
            replace(options, positions=(DerivativePosition(future, Decimal(1)),), marketPrices=[5.])

    def test_maturity_roll_multiplier_decomposition_and_mixed_hessian(self):
        options = make_book(horizon_days=3)
        model = OptionFactorStressModel(FactorStressModel(("A",), [10000.], [.003], [[.1, -.06]]), options)
        point = np.array([.2, -.1])
        for axis in np.eye(2)*1e-5:
            numerical = (model.pnl(point+axis)-model.pnl(point-axis))/2e-5
            self.assertAlmostEqual(float(numerical), float(model.pnlGradient(point)@(axis/1e-5)), delta=1e-5)
            np.testing.assert_allclose((model.pnlGradient(point+axis)-model.pnlGradient(point-axis))/2e-5,
                                      model.pnlHessian(point)@(axis/1e-5), rtol=1e-7, atol=1e-5)
        parts = options.pnlByStyle(model._logReturns(point))
        self.assertAlmostEqual(float(parts["E"]+parts["A"]+model.equityModel.pnl(point)), float(model.pnl(point)))
        self.assertFalse(model.isConvex)
        self.assertEqual(options.contexts[0].timeToExpiry, 177/365.)

    def test_mixed_pca_residual_and_explicit_nonsmooth_alignment(self):
        from tests.test_european_factor_stress import book as european_book
        from risk_state_generator import ReturnsPCAGrid, ReturnsPCAKey
        import pandas as pd
        rng = np.random.default_rng(4)
        prices = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0, .02, (42, 3)), axis=0)),
            index=pd.date_range("2025-01-01", periods=42), columns=list("ABC"))
        grid = ReturnsPCAGrid.construct(ReturnsPCAKey(tuple("ABC"), 30, date(2025, 2, 10), .93, 2), prices)
        old = european_book(instruments=tuple("ABC"), spots=[100., 100., 100.])
        options = OptionBook(old.instruments, old.spotPrices, old.positions, old.marketPrices,
                             old.valuationDate, old.horizonDate, old.riskFreeRate)
        portfolio = Portfolio({s: Decimal(0) for s in "ABC"})
        model = OptionFactorStressModel.fromPCAGrid(grid, portfolio, options)
        local, _ = options.logDerivatives(model.equityModel.center)
        residuals = grid.residuals*grid.logReturnScale
        weights = .93**np.arange(29, -1, -1); weights /= weights.sum()
        residuals -= weights@residuals
        self.assertAlmostEqual((local@model.equityModel.directions[:, -1])**2,
                               float(weights@(residuals@local)**2), places=7)
        explicit = OptionFactorStressModel.fromPCAGrid(grid, portfolio, options, localPnlGradient=[1., 0., 0.])
        self.assertEqual(explicit.residualAlignment, "supplied")


class OptionSearchTest(unittest.TestCase):
    def test_expiry_plateau_is_escaped_and_all_candidates_are_feasible(self):
        options = make_book(styles=("A",), types=("C",), quantities=("-1",), strike="110", horizon_days=180)
        model = OptionFactorStressModel(FactorStressModel(("A",), [0.], [0.], [[.2]]), options)
        self.assertEqual(float(model.pnlGradient([0.])[0]), 0.)
        result = solveOptionFactorStress(model, CenterSolver(), config=FactorStressQUBOConfig(3, 1.))
        self.assertGreater(result.margin, 0.)
        np.testing.assert_allclose(result.coordinates, [1.])
        self.assertEqual(result.boundaryCoverage, ((0, True, True),))
        self.assertTrue(all(np.sum(c.integers*c.integers) <= 9 for c in result.candidates))
        self.assertAlmostEqual(result.pnl, result.stockPnL+result.europeanPnL+result.americanPnL)
        self.assertFalse(all(result.solverFeasibility))

    def test_nonsmooth_center_uses_other_anchors_and_fixed_geometry(self):
        options = make_book(styles=("A",), types=("C",), quantities=("-1",), horizon_days=180)
        model = OptionFactorStressModel(FactorStressModel(("A",), [0.], [0.], [[.2]]), options)
        with self.assertRaises(NonsmoothOptionError):
            model.quadraticCoefficients()
        result = solveOptionFactorStress(model, CenterSolver(), config=FactorStressQUBOConfig(3, 1.))
        self.assertGreater(result.nonsmoothAnchors, 0)
        self.assertGreater(result.smoothAnchors, 0)
        self.assertEqual(float(model.equityModel.directions[0, 0]), .2)
        self.assertAlmostEqual(result.pnl, float(model.pnl(result.coordinates)))

    def test_zero_loading_at_boundary_reports_missing_sides(self):
        options = make_book(styles=("A",), types=("C",), quantities=("-1",), horizon_days=180)
        model = OptionFactorStressModel(FactorStressModel(("A",), [0.], [0.], [[0.]]), options)
        result = solveOptionFactorStress(model, CenterSolver(), config=FactorStressQUBOConfig(2, 1.))
        self.assertEqual(result.smoothAnchors, 0)
        self.assertEqual(result.boundaryCoverage, ((0, False, False),))

    def test_qubo_energy_and_option_count_invariance(self):
        options = make_book()
        model = OptionFactorStressModel(FactorStressModel(("A",), [1000.], [.001], [[.1]]), options)
        encoding = FactorStressQUBO.build(QuadraticStressObjective(*model.quadraticCoefficients()), FactorStressQUBOConfig(2, 1.))
        for sample in product((0, 1), repeat=encoding.problem.variableCount):
            self.assertAlmostEqual(encoding.problem.energy(sample), encoding.diagnostics(sample)["decomposed_energy"], delta=1e-7)
        doubled = replace(model, options=replace(options, positions=options.positions*2, marketPrices=np.tile(options.marketPrices, 2)))
        second = FactorStressQUBO.build(QuadraticStressObjective(*doubled.quadraticCoefficients()), encoding.config)
        self.assertEqual(encoding.problem.variableCount, second.problem.variableCount)

    def test_generic_repair_prices_only_feasible_neighbors(self):
        class BoundedModel:
            dimension = 1
            def pnl(self, z):
                z = np.asarray(z)
                if np.any(abs(z) > 1):
                    raise ValueError("outside price domain")
                return -z[..., 0]
        encoding = FactorStressQUBO.build(QuadraticStressObjective(0., [-1.], [[0.]]), FactorStressQUBOConfig(2, 1.))
        result = FactorStressRepair(BoundedModel(), encoding).repair(encoding.encodeIntegers(np.array([1])))
        self.assertEqual(result.pnl, -1.)


if __name__ == "__main__":
    unittest.main()

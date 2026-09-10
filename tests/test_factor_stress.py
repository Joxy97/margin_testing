"""Hard-budget quadratization and independent factor-margin references."""

from datetime import date
from decimal import Decimal
from itertools import product
import unittest

import numpy as np
import pandas as pd

from portfolio import Portfolio
from risk_state_generator import ReturnsPCAGrid, ReturnsPCAKey
from risk_state_generator.factor_stress_model import FactorStressModel
from margin_calculator.optimization.factor_stress import (
    FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective, solveLattice,
)
from margin_calculator.optimization.factor_stress_reference import (
    solveLinearReference, solveQuadraticReference, solveRepricedReference,
)


class FactorStressQUBOTest(unittest.TestCase):
    def test_penalty_sweep_scales_only_constraints(self):
        objective = QuadraticStressObjective(.3, [-2., .7], [[.5, -.2], [-.2, .4]])
        reference = FactorStressQUBO.build(objective, FactorStressQUBOConfig(2, 2.))
        for multiplier in (0., 1e-6, .1, 10.):
            model = FactorStressQUBO.build(objective, FactorStressQUBOConfig(2, 2., 1.1, multiplier))
            self.assertAlmostEqual(model.penalty, multiplier*reference.penalty)
            for bits in product((0, 1), repeat=model.problem.variableCount):
                before = reference.diagnostics(bits)
                expected = before["objective"] + multiplier*(before["decomposed_energy"]-before["objective"])
                self.assertAlmostEqual(model.problem.energy(bits), expected, places=9)
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                FactorStressQUBOConfig(penaltyMultiplier=value)
        with self.assertRaises(TypeError):
            FactorStressQUBOConfig(penaltyMultiplier=True)

    def test_exhaustive_qubo_matches_constraints_and_global_optimum(self):
        objective = QuadraticStressObjective(.3, [-2., .7], [[.5, -.2], [-.2, .4]])
        model = FactorStressQUBO.build(objective, FactorStressQUBOConfig(2, 2.))
        self.assertEqual(model.problem.variableCount, 7)  # 4 scenario, 2 product, 1 slack
        self.assertEqual(model.problem.oneHotGroups, ())
        minimum = float("inf")
        minimizer = None
        for bits in product((0, 1), repeat=model.problem.variableCount):
            info = model.diagnostics(bits)
            energy = model.problem.energy(bits)
            self.assertAlmostEqual(energy, info["decomposed_energy"], places=10)
            if energy < minimum:
                minimum, minimizer = energy, bits
        point, expected, _ = solveLattice(model)
        self.assertTrue(model.diagnostics(minimizer)["encoding_feasible"])
        self.assertAlmostEqual(minimum, expected)
        self.assertAlmostEqual(model.problem.energy(model.encodeIntegers(point)), expected)

    def test_feasible_encoding_and_canonical_edges(self):
        objective = QuadraticStressObjective(0., [.1, -.2, .3], np.eye(3)*.1)
        for bits in (2, 4, 6, 8):
            model = FactorStressQUBO.build(objective, FactorStressQUBOConfig(bits, 3.))
            m = model.latticeRadius
            for point in ([0, 0, 0], [-m, 0, 0], [0, m, 0], [0, 0, -m]):
                sample = model.encodeIntegers(np.array(point))
                np.testing.assert_array_equal(model.integerCoordinates(sample), point)
                self.assertTrue(model.diagnostics(sample)["encoding_feasible"])
                # Expanded hard penalties cancel large terms in float64. Check
                # energy at its arithmetic scale; integer feasibility above is
                # exact and decomposed scoring has no such cancellation.
                q = model.problem
                active = sample.astype(bool)
                mass = (abs(q.offset) + np.abs(q.linear[active]).sum()
                        + np.abs(q.quadraticBiases[active[q.quadraticHeads]
                                                  & active[q.quadraticTails]]).sum())
                self.assertAlmostEqual(q.energy(sample), float(objective.value(model.coordinates(sample))),
                                       delta=max(1e-12, 16*np.finfo(float).eps*mass))
            q = model.problem
            self.assertTrue(np.all(q.quadraticHeads < q.quadraticTails))
            self.assertTrue(np.all(q.quadraticBiases != 0))
            self.assertEqual(len(set(zip(q.quadraticHeads, q.quadraticTails))), q.interactionCount)

    def test_product_and_budget_violations_are_checked_separately(self):
        model = FactorStressQUBO.build(QuadraticStressObjective(0, [1.], [[0.]]),
                                      FactorStressQUBOConfig(3))
        sample = model.encodeIntegers(np.array([0]))
        sample[model.scenarioBits] ^= 1
        self.assertEqual(model.diagnostics(sample)["product_violations"], 1)
        self.assertFalse(model.diagnostics(sample)["encoding_feasible"])
        sample = model.encodeIntegers(np.array([0]))
        sample[-1] ^= 1
        info = model.diagnostics(sample)
        self.assertEqual(info["product_violations"], 0)
        self.assertTrue(info["scenario_feasible"])
        self.assertFalse(info["encoding_feasible"])

    def test_lattice_reference_matches_full_enumeration_including_indefinite_case(self):
        rng = np.random.default_rng(12)
        for dimension in (1, 2, 3):
            for sign in (-1, 1):
                a = rng.normal(size=(dimension, dimension))
                objective = QuadraticStressObjective(.2, rng.normal(size=dimension), sign*a.T@a)
                model = FactorStressQUBO.build(objective, FactorStressQUBOConfig(3, 2.))
                points = np.array([p for p in product(range(-3, 4), repeat=dimension)
                                   if sum(v*v for v in p) <= 9])
                expected = objective.value(points*2/3).min()
                point, actual, _ = solveLattice(model)
                self.assertAlmostEqual(actual, expected)
                self.assertTrue(model.diagnostics(model.encodeIntegers(point))["encoding_feasible"])

    def test_invalid_inputs_and_immutable_coefficients(self):
        for kwargs in ({"bitsPerCoordinate": 1}, {"bitsPerCoordinate": 9}, {"radius": 0},
                       {"radius": float("nan")}, {"penaltySafety": 1}):
            with self.assertRaises(ValueError):
                FactorStressQUBOConfig(**kwargs)
        with self.assertRaises(TypeError):
            FactorStressQUBOConfig(bitsPerCoordinate=True)
        for g, h in (([1, 2], [[1]]), ([1], [[float("nan")]]), ([1, 2], [[1, 2], [0, 1]])):
            with self.assertRaises(ValueError):
                QuadraticStressObjective(0, g, h)
        gradient = np.array([1.])
        obj = QuadraticStressObjective(0, gradient, [[0.]])
        gradient[0] = 3
        self.assertEqual(obj.gradient[0], 1)
        with self.assertRaises(ValueError):
            obj.gradient.setflags(write=True)
        model = FactorStressQUBO.build(obj)
        with self.assertRaises(ValueError):
            model.encodeIntegers(np.array([model.latticeRadius+1]))
        with self.assertRaises(ValueError):
            model.encodeIntegers(np.array([.5]))
        for sample in ([1], np.full(model.problem.variableCount, .5)):
            with self.assertRaises(ValueError):
                model.diagnostics(sample)


class FactorStressModelTest(unittest.TestCase):
    def test_pca_residual_direction_preserves_portfolio_local_variance_and_no_lookahead(self):
        rng = np.random.default_rng(4)
        returns = rng.normal(0, .02, (41, 5))
        dates = pd.date_range("2025-01-01", periods=42)
        prices = pd.DataFrame(100*np.exp(np.vstack((np.zeros(5), np.cumsum(returns, axis=0)))),
                              index=dates, columns=list("ABCDE"))
        key = ReturnsPCAKey(tuple(prices.columns), 30, date(2025, 2, 10), .93, 2)
        grid = ReturnsPCAGrid.construct(key, prices)
        portfolio = Portfolio({i: Decimal(".2") for i in prices.columns})
        model = FactorStressModel.fromPCAGrid(grid, portfolio)
        original = model.directions.copy()
        changed = prices.copy()
        changed.loc[changed.index >= pd.Timestamp(key.start_date)] *= 100
        later = FactorStressModel.fromPCAGrid(ReturnsPCAGrid.construct(key, changed), portfolio)
        np.testing.assert_array_equal(later.directions, original)
        self.assertLess(grid.calibrationEndDate, key.start_date)
        weights = .93**np.arange(29, -1, -1)
        weights /= weights.sum()
        residuals = grid.residuals*grid.logReturnScale
        residuals -= weights@residuals
        local = model.exposures*np.exp(model.center)
        variance = weights@(residuals@local)**2
        self.assertAlmostEqual((local@model.directions[:, -1])**2, variance)
        p0, g, h = model.quadraticCoefficients()
        np.testing.assert_allclose(g, model.pnlGradient(np.zeros(3)))
        self.assertAlmostEqual(p0, float(model.pnl(np.zeros(3))))
        eps = 1e-5
        for i in range(3):
            delta = np.eye(3)[i]*eps
            np.testing.assert_allclose((model.pnlGradient(delta)-model.pnlGradient(-delta))/(2*eps),
                                       h[:, i], rtol=1e-7, atol=1e-12)
        reversed_portfolio = Portfolio({"X": Decimal("1")})
        with self.assertRaises(ValueError):
            FactorStressModel.fromPCAGrid(grid, reversed_portfolio)

    def test_continuous_references_and_convex_certificates(self):
        objective = QuadraticStressObjective(.01, [.03, -.04], np.zeros((2, 2)))
        linear = solveLinearReference(objective, 3)
        self.assertAlmostEqual(linear.objective, -.14)
        quadratic = solveQuadraticReference(objective, 3)
        np.testing.assert_allclose(quadratic.coordinates, linear.coordinates, atol=1e-6)
        self.assertLess(quadratic.optimalityGap, 1e-10)
        model = FactorStressModel(("A",), [1.], [.01], [[.05]])
        exact = solveRepricedReference(model, 3.)
        self.assertAlmostEqual(exact.objective, np.expm1(.01-.15))
        self.assertLess(exact.optimalityGap, 1e-10)
        self.assertLessEqual(np.linalg.norm(exact.coordinates), 3.)
        signed = FactorStressModel(("A",), [-1.], [.01], [[.05]])
        self.assertIsNone(solveRepricedReference(signed, 3.).lowerBound)
        zero = solveLinearReference(QuadraticStressObjective(.1, [0.], [[0.]]), 2.)
        np.testing.assert_array_equal(zero.coordinates, [0.])
        with self.assertRaises(ValueError):
            solveLinearReference(objective, 0)


if __name__ == "__main__":
    unittest.main()

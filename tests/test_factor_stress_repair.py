"""Exact feasibility, deterministic movement and loss-preserving repair."""

from itertools import product
import unittest

import numpy as np

from risk_state_generator.factor_stress_model import FactorStressModel
from margin_calculator.optimization.factor_stress import (
    FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective,
)
from margin_calculator.optimization.factor_stress_repair import (
    FactorStressRepair, FactorStressRepairConfig, projectIntegerBall,
)


def fixture(dimension=3, bits=3):
    model = FactorStressModel(tuple("ABC"[:dimension]), np.ones(dimension)/dimension,
                             np.zeros(dimension), np.eye(dimension)*.03)
    encoding = FactorStressQUBO.build(QuadraticStressObjective(*model.quadraticCoefficients()),
                                     FactorStressQUBOConfig(bits, 3.))
    return model, encoding


class FactorStressRepairTest(unittest.TestCase):
    def test_eleven_coordinate_pairwise_repair_and_original_qubo(self):
        rng = np.random.default_rng(39)
        model = FactorStressModel(tuple(f'A{i}' for i in range(15)), rng.normal(size=15),
                                  np.zeros(15), rng.normal(0, .01, (15, 11)))
        encoded = FactorStressQUBO.build(QuadraticStressObjective(*model.quadraticCoefficients()),
                                         FactorStressQUBOConfig(8, 5.756441175254773))
        self.assertEqual(encoded.problem.variableCount, 410)
        with self.assertRaises(ValueError):
            FactorStressRepair(model, encoded)
        repair = FactorStressRepair(model, encoded, FactorStressRepairConfig(neighborhood='pairwise'))
        self.assertEqual(repair.moves.shape, (242, 11))
        self.assertEqual(len(set(map(tuple, repair.moves))), 242)
        self.assertTrue(np.all(np.count_nonzero(repair.moves, axis=1) <= 2))
        sample = rng.integers(0, 2, encoded.problem.variableCount, dtype=np.uint8)
        result = repair.repair(sample)
        self.assertTrue(result.converged)
        self.assertTrue(encoded.diagnostics(result.sample)['encoding_feasible'])
        self.assertLessEqual(result.pnl, result.projectedPnL)
        np.testing.assert_array_equal(result.sample, repair.repair(sample).sample)
        neighbors = result.integers+repair.moves
        feasible = np.sum(neighbors**2, axis=1) <= encoded.latticeRadius**2
        self.assertTrue(np.all(model.pnl(neighbors[feasible]*repair.scale) >= result.pnl-1e-12))
        point = result.integers*repair.scale
        expected = model.pnl((result.integers+repair.moves)*repair.scale)-model.pnl(point)
        weighted = model.exposures*np.exp(model.center+model.directions@point)
        np.testing.assert_allclose(repair.increments@weighted, expected, atol=1e-14)
        # Feasible auxiliaries cancel all integer penalties in the original QUBO.
        energy = encoded.problem.energy(result.sample)
        self.assertAlmostEqual(energy, float(encoded.objective.value(point)), delta=1e-3)

    def test_integer_projection_is_feasible_idempotent_and_preserves_inside(self):
        for values in product(range(-5, 6), repeat=3):
            original = np.array(values)
            projected = projectIntegerBall(original, 3)
            self.assertLessEqual(int(projected@projected), 9)
            np.testing.assert_array_equal(projectIntegerBall(projected, 3), projected)
            np.testing.assert_array_equal(original, values)
            if original@original <= 9:
                np.testing.assert_array_equal(projected, original)
        largest = np.array([np.iinfo(np.int64).max, np.iinfo(np.int64).min], dtype=np.int64)
        projected = projectIntegerBall(largest, 127)
        self.assertLessEqual(int(projected@projected), 127**2)

    def test_all_small_bitstrings_repair_without_mutation(self):
        model, encoded = fixture(2, 2)
        repair = FactorStressRepair(model, encoded, FactorStressRepairConfig(maxSteps=0))
        coefficients = encoded.problem.linear.copy()
        for bits in product((0, 1), repeat=encoded.problem.variableCount):
            sample = np.array(bits, dtype=np.uint8)
            before = sample.copy()
            result = repair.repair(sample)
            self.assertTrue(encoded.diagnostics(result.sample)["encoding_feasible"])
            np.testing.assert_array_equal(sample, before)
            if encoded.diagnostics(sample)["scenario_feasible"]:
                np.testing.assert_array_equal(result.originalIntegers, result.integers)
                self.assertEqual(result.rawPnL, result.pnl)
                np.testing.assert_array_equal(sample[:encoded.scenarioBits], result.sample[:encoded.scenarioBits])
            np.testing.assert_array_equal(result.sample, result.projectedSample)
        np.testing.assert_array_equal(encoded.problem.linear, coefficients)

    def test_improvement_is_deterministic_monotone_and_locally_optimal(self):
        model, encoded = fixture()
        repair = FactorStressRepair(model, encoded)
        for start in ([3, 0, 0], [-3, 0, 0], [0, 0, 0], [1, 2, 1]):
            sample = encoded.encodeIntegers(np.array(start))
            sample[-1] ^= 1
            result = repair.repair(sample)
            again = repair.repair(sample)
            self.assertTrue(result.converged)
            np.testing.assert_array_equal(result.sample, again.sample)
            self.assertLessEqual(result.pnl, result.projectedPnL)
            self.assertTrue(encoded.diagnostics(result.sample)["encoding_feasible"])
            self.assertAlmostEqual(result.pnl, float(model.pnl(encoded.coordinates(result.sample))))
            for move in product((-1, 0, 1), repeat=3):
                point = result.integers+move
                if point@point <= encoded.latticeRadius**2:
                    self.assertGreaterEqual(float(model.pnl(point*repair.scale)), result.pnl-1e-12)
            with self.assertRaises(ValueError):
                result.sample.setflags(write=True)

    def test_diagonal_move_can_leave_an_axis_boundary(self):
        model = FactorStressModel(("A",), [1.], [0.], [[.01, .03]])
        encoded = FactorStressQUBO.build(QuadraticStressObjective(*model.quadraticCoefficients()),
                                         FactorStressQUBOConfig(3))
        result = FactorStressRepair(model, encoded).repair(encoded.encodeIntegers(np.array([-3, 0])))
        self.assertLess(result.pnl, result.projectedPnL)
        self.assertLess(result.integers[1], 0)
        self.assertGreater(result.integers[0], -3)

    def test_step_limit_and_exact_exponential_increments(self):
        model, encoded = fixture()
        repair = FactorStressRepair(model, encoded, FactorStressRepairConfig(maxSteps=1))
        point = np.array([1, 1, 1])
        weighted = model.exposures*np.exp(model.center+model.directions@(point*repair.scale))
        expected = model.pnl((point+repair.moves)*repair.scale)-model.pnl(point*repair.scale)
        np.testing.assert_allclose(repair.increments@weighted, expected, atol=1e-16, rtol=1e-12)
        result = repair.repair(encoded.encodeIntegers(point))
        self.assertEqual(result.steps, 1)
        self.assertFalse(result.converged)
        self.assertTrue(encoded.diagnostics(result.sample)["encoding_feasible"])

    def test_invalid_arguments(self):
        for coordinates in ([.1, .2], [], [[1, 2]]):
            with self.assertRaises(ValueError):
                projectIntegerBall(np.array(coordinates), 3)
        with self.assertRaises(ValueError):
            projectIntegerBall(np.array([1]), 0)
        with self.assertRaises(TypeError):
            FactorStressRepairConfig(maxSteps=True)
        for config in ({"maxSteps": -1}, {"improvementTolerance": -1}, {"improvementTolerance": float("nan")}):
            with self.assertRaises(ValueError):
                FactorStressRepairConfig(**config)
        model, encoding = fixture()
        other, _ = fixture(2)
        with self.assertRaises(ValueError):
            FactorStressRepair(other, encoding)
        with self.assertRaises(ValueError):
            FactorStressRepair(model, encoding).repair(np.full(encoding.problem.variableCount, .5))


if __name__ == "__main__":
    unittest.main()

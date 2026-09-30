"""Contract tests for the common library's application-facing adapters."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy
import torch

from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMResourcePlan, BQMSolverConfig, QUBOProblem,
)
from qubo_solvers import LIBRARY_SOLVERS, SOLVERS, create_bqm_solver, library_solver_class
from qubo_solvers.backends.library_solver import LibraryBQMSolver


def problem(*, grouped=False, seed=23):
    return QUBOProblem(
        [-3., 2., -.25], [0, 1, 0, 2, 1], [1, 0, 1, 2, 2],
        [4., -1., .5, 1.25, -.75], offset=7.5,
        oneHotGroups=((0, 1),) if grouped else (), seedOffset=seed,
    )


class LibraryBQMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_library_ids_preserve_factory_and_return_source_scores(self):
        self.assertEqual(len(SOLVERS), 28)
        self.assertEqual(len(LIBRARY_SOLVERS), 17)
        self.assertTrue(set(LIBRARY_SOLVERS).issubset(SOLVERS))
        original = problem()
        arrays = [value.copy() for value in (
            original.linear, original.quadraticHeads,
            original.quadraticTails, original.quadraticBiases,
        )]
        for name in LIBRARY_SOLVERS:
            with self.subTest(solver=name):
                solver = create_bqm_solver(name, {"device": "cpu"})
                self.assertIsInstance(solver, LibraryBQMSolver)
                options = {"steps": 2, "runs": 3, "run_batch_size": 2, "dtype": "float64"}
                first = solver.solve(original, options)
                second = solver.solve(original, options)
                self.assertEqual(first, second)
                self.assertEqual(len(first.sample), original.variableCount)
                self.assertEqual(first.energy, original.energy(first.sample))
        for before, after in zip(arrays, (
            original.linear, original.quadraticHeads,
            original.quadraticTails, original.quadraticBiases,
        )):
            numpy.testing.assert_array_equal(before, after)

    def test_pair_once_conversion_for_every_assignment(self):
        original = problem()
        solver = create_bqm_solver("lib_random_search")
        converted = solver._toLibraryProblem(original, "float64")
        for bits in itertools.product((0, 1), repeat=original.variableCount):
            expected = 7.5 - 3*bits[0] + 2*bits[1] + bits[2] + 3.5*bits[0]*bits[1] - .75*bits[1]*bits[2]
            self.assertEqual(float(converted.energy(torch.tensor(bits))), expected)
            self.assertEqual(original.energy(bits), expected)

    def test_library_candidates_are_rescored_in_source_float64(self):
        original = QUBOProblem([1.000000001, 1.], [], [], [], offset=8.)
        candidates = SimpleNamespace(
            best_assignments=torch.tensor([[1, 0], [0, 1]], dtype=torch.int8),
            final_assignments=None,
        )
        solver = create_bqm_solver("lib_random_search")
        with patch.object(library_solver_class(solver.solverId), "solve", return_value=candidates):
            result = solver.solve(original, {"steps": 0, "runs": 2, "dtype": "float32"})
        self.assertEqual(result.sample, (0, 1))
        self.assertEqual(result.energy, 9.)

    def test_one_hot_prefers_feasible_candidates_and_repairs_when_needed(self):
        original = problem(grouped=True)
        solver = create_bqm_solver("lib_random_search")
        options = {"steps": 0, "runs": 2}
        candidates = SimpleNamespace(
            best_assignments=torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.int8),
            final_assignments=None,
        )
        with patch.object(library_solver_class(solver.solverId), "solve", return_value=candidates):
            result = solver.solve(original, options)
        self.assertEqual(result.sample, (0, 1, 0))
        candidates.best_assignments = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.int8)
        with patch.object(library_solver_class(solver.solverId), "solve", return_value=candidates):
            result = solver.solve(original, options)
        self.assertEqual(sum(result.sample[:2]), 1)
        self.assertEqual(result.energy, original.energy(result.sample))
        with self.assertRaisesRegex(ValueError, "one-hot"):
            solver.solve(original, dict(options, best_only=True))

    def test_seed_offset_and_application_batches_preserve_results(self):
        solver = create_bqm_solver("lib_random_search")
        options = {"steps": 1, "runs": 7, "seed": 11}
        inputs = [problem(seed=offset) for offset in (4, 8, 3)]
        state = torch.random.get_rng_state().clone()
        expected = [solver.solve(item, options) for item in inputs]
        solver.beginSeries()
        actual = solver.solveMany(inputs[:1], options) + solver.solveMany(inputs[1:], options)
        solver.endSeries()
        self.assertEqual(expected, actual)
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        native = library_solver_class(solver.solverId)
        original_solve = native.solve
        seeds = []

        def capture_seed(instance, *args, **kwargs):
            seeds.append(kwargs["seed"])
            return original_solve(instance, *args, **kwargs)

        with patch.object(native, "solve", capture_seed):
            solver.solve(inputs[0], options)
        self.assertEqual(seeds, [15])

    def test_strict_configuration_and_native_aliases(self):
        solver = create_bqm_solver("lib_simulated_annealing")
        values = solver._getParameters({"steps": 3, "restarts": 2, "batch_size": 1})
        self.assertEqual((values["sweeps"], values["runs"], values["run_batch_size"]), (3, 2, 1))
        for invalid in (
            {"steps": 2, "sweeps": 2}, {"runs": 2, "restarts": 2},
            {"run_batch_size": 2, "batch_size": 2}, {"runs": 1.2}, {"seed": True},
            {"runs": 0}, {"dtype": "float16"}, {"best_only": "false"},
            {"memory_limit_bytes": 0}, {"steps": -1}, {"start_temperature": -1.},
            {"time_limit": 1}, {"device": "cpu"},
        ):
            with self.subTest(parameters=invalid), self.assertRaises((ValueError, TypeError)):
                solver._getParameters(invalid)
        with self.assertRaisesRegex(ValueError, "device"):
            create_bqm_solver("lib_random_search", {"device": "auto"})
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                create_bqm_solver("lib_random_search", {"device": "cuda:0"}).device

    def test_memory_estimate_is_allocation_free_and_tracks_batching(self):
        solver = create_bqm_solver("lib_replica_annealing")
        original = problem()
        with patch("numpy.zeros", side_effect=AssertionError("dense allocation")):
            full = solver.estimatedWorkingMemoryBytes(original, {"runs": 32})
            batched = solver.estimatedWorkingMemoryBytes(original, {"runs": 32, "run_batch_size": 2})
            more_replicas = solver.estimatedWorkingMemoryBytes(original, {"runs": 32, "replicas": 8})
        self.assertLess(batched, full)
        self.assertGreater(more_replicas, full)
        with patch.object(solver, "_toLibraryProblem", side_effect=AssertionError("dense allocation")):
            with self.assertRaises(MemoryError):
                solver.solve(original, {"memory_limit_bytes": 1})
            with self.assertRaises(MemoryError):
                solver.solve(original, {"memory_limit_bytes": 1000})

    def test_planned_execution_matches_ordered_single_worker_results(self):
        solver = create_bqm_solver("lib_random_search")
        inputs = [problem(seed=offset) for offset in (2, 4)]
        options = {"steps": 1, "runs": 2}
        plan = BQMResourcePlan.create([1000, 1000], 1)
        self.assertEqual(solver.solvePlanned(inputs, plan, options), solver.solveMany(inputs, options))
        with self.assertRaises(ValueError):
            solver.solvePlanned(inputs, BQMResourcePlan.create([1000, 1000], 2), options)

    def test_typed_and_yaml_configuration_reach_library_solver(self):
        config = BQMSolverConfig(
            solverType="lib_greedy_local_search", constructorParameters={"device": "cpu"},
            solverParameters={"steps": 4, "runs": 2},
        )
        result = config.createBQMSolver().solve(problem(), config.solverParameters)
        self.assertEqual(result.energy, problem().energy(result.sample))
        from margin_engine.yaml_application import MarginApplicationConfig

        application = MarginApplicationConfig.fromYamlText("""
marginDate: '2026-01-02'
portfolio:
  weights: {AAA: 1}
engine:
  marginCalculator:
    type: bqm
    solver:
      type: lib_greedy_local_search
      constructorParameters: {device: cpu}
      solverParameters: {steps: 4, runs: 2}
""", ".")
        parsed = application.engine.marginCalculator.solver
        self.assertEqual(parsed, config)
        self.assertIsInstance(parsed.createBQMSolver(), LibraryBQMSolver)


if __name__ == "__main__":
    unittest.main()

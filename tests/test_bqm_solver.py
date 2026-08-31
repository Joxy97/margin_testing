"""Tests for BQM solver interfaces and factories."""

import unittest
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any
from unittest.mock import patch

import dimod
import numpy

from margin_calculator.optimization.optimization_solver.bqm_solver import (
    AdaptiveTorchSBMBQMSolver,
    BQMSolver,
    BQMSolverFactory,
    BQMOptimizationResult,
    OptimizationProblem,
    OptimizationSolver,
    OptimizationSolverResult,
    PlanarGraphBQMSolver,
    QUBOProblem,
    RandomBQMSolver,
    SBMBQMSolver,
    SimulatedAnnealingBQMSolver,
    SteepestDescentBQMSolver,
    TabuBQMSolver,
    TorchSBMBQMSolver,
    TreeDecompositionBQMSolver,
    TreeDecompositionSamplerBQMSolver,
)


class StubBQMSolver(BQMSolver):
    def __init__(self, answer: str = "solved") -> None:
        self.answer = answer
        self.problem: QUBOProblem | None = None
        self.solverParameters: dict[str, Any] = {}

    def solve(
        self,
        problem: QUBOProblem,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> BQMOptimizationResult:
        self.problem = problem
        self.solverParameters = dict(solverParameters or {})
        return BQMOptimizationResult({"x": 1}, -1.0)


class BQMSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        BQMSolverFactory.registerSolver("stub", StubBQMSolver)

    def test_solver_receives_a_qubo_problem(self) -> None:
        problem = QUBOProblem(
            linear=numpy.array([-1.0]),
            quadraticHeads=numpy.array([], dtype=numpy.uint32),
            quadraticTails=numpy.array([], dtype=numpy.uint32),
            quadraticBiases=numpy.array([]),
        )
        solver = StubBQMSolver()

        result = solver.solve(problem, {"num_reads": 10})

        self.assertIsInstance(result, OptimizationSolverResult)
        self.assertIsInstance(result, BQMOptimizationResult)
        self.assertIs(solver.problem, problem)
        self.assertEqual(solver.solverParameters, {"num_reads": 10})
        self.assertIsInstance(problem, OptimizationProblem)
        self.assertIsInstance(solver, OptimizationSolver)

    def test_factory_creates_registered_solver_with_parameters(self) -> None:
        solver = BQMSolverFactory.createBQMSolver(
            "stub",
            {"answer": "custom"},
        )

        self.assertIsInstance(solver, StubBQMSolver)
        self.assertEqual(solver.answer, "custom")

    def test_factory_rejects_unknown_solver(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown BQM solver"):
            BQMSolverFactory.createBQMSolver("unknown")

    def test_factory_includes_all_bqm_solvers(self) -> None:
        expected_solvers = {
            "adaptive_torch_sbm": AdaptiveTorchSBMBQMSolver,
            "planar_graph": PlanarGraphBQMSolver,
            "random": RandomBQMSolver,
            "sbm": SBMBQMSolver,
            "simulated_annealing": SimulatedAnnealingBQMSolver,
            "steepest_descent": SteepestDescentBQMSolver,
            "tabu": TabuBQMSolver,
            "torch_sbm": TorchSBMBQMSolver,
            "tree_decomposition_solver": TreeDecompositionBQMSolver,
            "tree_decomposition_sampler": TreeDecompositionSamplerBQMSolver,
        }

        for name, solver_class in expected_solvers.items():
            self.assertIsInstance(
                BQMSolverFactory.createBQMSolver(name),
                solver_class,
            )

    def test_solver_selects_first_valid_one_hot_sample_by_energy(self) -> None:
        sample_set = dimod.SampleSet.from_samples(
            [
                {
                    0: 1,
                    1: 1,
                    2: 0,
                    3: 0,
                },
                {
                    0: 1,
                    1: 0,
                    2: 0,
                    3: 1,
                },
            ],
            vartype=dimod.BINARY,
            energy=[-10.0, -5.0],
        )

        problem = QUBOProblem(
            numpy.zeros(4),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([]),
            oneHotGroups=((0, 1), (2, 3)),
        )
        sample, energy = StubBQMSolver._selectBestSample(sample_set, problem)

        self.assertEqual(energy, -5.0)
        self.assertEqual(sample[0] + sample[1], 1)
        self.assertEqual(sample[2] + sample[3], 1)

    @unittest.skipUnless(
        SBMBQMSolver().libraryPath.is_file(),
        "build the sbm_python CMake target to run this integration test",
    )
    def test_sbm_solver_uses_the_cpp_kernel(self) -> None:
        problem = QUBOProblem(
            numpy.array([-1.0, 0.5]),
            numpy.array([0], dtype=numpy.uint32),
            numpy.array([1], dtype=numpy.uint32),
            numpy.array([-0.25]),
        )
        solver = BQMSolverFactory.createBQMSolver("sbm")

        result = solver.solve(
            problem,
            {"steps": 100, "runs": 4, "seed": 7},
        )

        self.assertAlmostEqual(result.energy, problem.energy(result.sample))
        self.assertEqual(len(result.sample), 2)

    @unittest.skipUnless(
        SBMBQMSolver().libraryPath.is_file(),
        "build the sbm_python CMake target to run this integration test",
    )
    def test_sbm_solver_batches_different_problem_sizes(self) -> None:
        problems = [
            QUBOProblem(
                numpy.array([-1.0, 0.5]),
                numpy.array([0], dtype=numpy.uint32),
                numpy.array([1], dtype=numpy.uint32),
                numpy.array([-0.25]),
            ),
            QUBOProblem(
                numpy.array([0.2, -0.4, -0.1]),
                numpy.array([0, 1], dtype=numpy.uint32),
                numpy.array([1, 2], dtype=numpy.uint32),
                numpy.array([0.3, -0.2]),
            ),
        ]

        results = SBMBQMSolver().solveMany(
            problems,
            {"steps": 100, "runs": 3, "seed": 9},
        )

        self.assertEqual(len(results), 2)
        for problem, result in zip(problems, results):
            self.assertEqual(len(result.sample), problem.variableCount)
            self.assertAlmostEqual(result.energy, problem.energy(result.sample))

    @unittest.skipUnless(
        SBMBQMSolver().libraryPath.is_file(),
        "build the sbm_python CMake target to run this integration test",
    )
    def test_sbm_adaptive_solver_returns_valid_warm_started_samples(self) -> None:
        problem = QUBOProblem(
            numpy.array([-1.0, -1.0]),
            numpy.array([0], dtype=numpy.uint32),
            numpy.array([1], dtype=numpy.uint32),
            numpy.array([2.0]),
            oneHotGroups=((0, 1),),
        )
        solver = SBMBQMSolver()
        parameters = {
            "steps": 100,
            "runs": 8,
            "adaptive": True,
            "min_runs": 2,
            "runs_per_batch": 2,
            "stability_batches": 1,
            "warm_start": True,
            "topology_cache_bytes": 1024 * 1024,
            "seed": 17,
        }

        solver.beginSeries()
        first = solver.solveMany([problem], parameters)[0]
        solver.endSeries()
        solver.beginSeries()
        second = solver.solveMany([problem], parameters)[0]
        solver.endSeries()

        self.assertEqual(sum(first.sample), 1)
        self.assertEqual(sum(second.sample), 1)
        self.assertGreaterEqual(solver.lastRunCounts[0], 2)
        self.assertLessEqual(solver.lastRunCounts[0], 8)

    def test_sbm_parameters_are_validated_before_native_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "steps and runs"):
            SBMBQMSolver._getParameters({"runs": 0})

    def test_sbm_adaptive_parameters_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_runs"):
            SBMBQMSolver._getParameters(
                {"adaptive": True, "runs": 4, "min_runs": 5}
            )

    def test_torch_sbm_parameters_are_validated_without_loading_torch(self) -> None:
        with self.assertRaisesRegex(ValueError, "dtype"):
            TorchSBMBQMSolver._getParameters({"dtype": "float16"})

    def test_torch_sbm_multi_device_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(TypeError, "sequence"):
            TorchSBMBQMSolver(devices="cuda:0")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            TorchSBMBQMSolver(devices=[])
        with self.assertRaisesRegex(ValueError, "both device and devices"):
            TorchSBMBQMSolver(device="cuda:0", devices=["cuda:1"])

        solver = TorchSBMBQMSolver(devices=["cuda:0", "cuda:0"])
        with patch.object(
            TorchSBMBQMSolver,
            "_resolveDevice",
            side_effect=lambda requested: requested,
        ):
            with self.assertRaisesRegex(ValueError, "unique"):
                _ = solver.devices

    def test_torch_sbm_distributes_ordered_batch_across_devices(self) -> None:
        problems = [
            QUBOProblem(
                numpy.array([float(index)]),
                numpy.array([], dtype=numpy.uint32),
                numpy.array([], dtype=numpy.uint32),
                numpy.array([]),
            )
            for index in range(5)
        ]
        calls: list[tuple[str, list[int], int]] = []

        def solve_batch(
            worker: TorchSBMBQMSolver,
            batch: list[QUBOProblem],
            parameters: Mapping[str, Any],
        ) -> list[BQMOptimizationResult]:
            identifiers = [int(problem.linear[0]) for problem in batch]
            calls.append((worker.device, identifiers, parameters["seed"]))
            return [
                BQMOptimizationResult((0,), float(identifier))
                for identifier in identifiers
            ]

        solver = TorchSBMBQMSolver(devices=["cuda:0", "cuda:1"])
        with (
            patch.object(
                TorchSBMBQMSolver,
                "_resolveDevice",
                side_effect=lambda requested: requested,
            ),
            patch.object(TorchSBMBQMSolver, "_solveBatch", solve_batch),
        ):
            results = solver.solveMany(problems, {"seed": 17})

        self.assertEqual([result.energy for result in results], list(range(5)))
        calls.sort(key=lambda call: call[1][0])
        self.assertEqual(calls[0][:2], ("cuda:0", [0, 1, 2]))
        self.assertEqual(calls[1][:2], ("cuda:1", [3, 4]))
        self.assertEqual(calls[0][2], 17)
        self.assertEqual(
            calls[1][2],
            (17 + 3 * 0x0D1B54A32D192ED03) % ((1 << 63) - 1),
        )

    def test_adaptive_torch_sbm_creates_adaptive_device_workers(self) -> None:
        solver = AdaptiveTorchSBMBQMSolver(devices=["cuda:0", "cuda:1"])
        solver._resolvedDevices = ("cuda:0", "cuda:1")

        workers = solver._getWorkerSolvers()

        self.assertEqual(len(workers), 2)
        self.assertTrue(
            all(isinstance(worker, AdaptiveTorchSBMBQMSolver) for worker in workers)
        )

    @unittest.skipUnless(find_spec("torch"), "install torch to run this test")
    def test_torch_sbm_accepts_rocm_device_aliases(self) -> None:
        torch = TorchSBMBQMSolver._torch()
        with (
            patch.object(torch.version, "hip", "7.2"),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            for alias in ("rocm", "amd", "hip"):
                self.assertEqual(
                    TorchSBMBQMSolver._resolveDevice(alias),
                    "cuda",
                )

    @unittest.skipUnless(find_spec("torch"), "install torch to run this test")
    def test_torch_sbm_explains_when_rocm_build_is_missing(self) -> None:
        torch = TorchSBMBQMSolver._torch()
        with patch.object(torch.version, "hip", None):
            with self.assertRaisesRegex(RuntimeError, "HIP support"):
                TorchSBMBQMSolver._resolveDevice("rocm")

    def test_adaptive_torch_sbm_parameters_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            AdaptiveTorchSBMBQMSolver._getParameters({"mode": "unknown"})
        with self.assertRaisesRegex(ValueError, "sampling_period"):
            AdaptiveTorchSBMBQMSolver._getParameters({"sampling_period": 0})

    def test_adaptive_torch_sbm_polish_preserves_one_hot_groups(self) -> None:
        problem = QUBOProblem(
            numpy.array([1.0, -2.0, 0.5, -1.0]),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([]),
            oneHotGroups=((0, 1), (2, 3)),
        )
        initial = BQMOptimizationResult((1, 0, 1, 0), 1.5)

        polished = AdaptiveTorchSBMBQMSolver._polish(
            problem,
            initial,
            sweeps=2,
            tolerance=1.0e-12,
        )

        self.assertEqual(polished.sample, (0, 1, 0, 1))
        self.assertAlmostEqual(polished.energy, -3.0)
        self.assertEqual(sum(polished.sample[:2]), 1)
        self.assertEqual(sum(polished.sample[2:]), 1)

    @unittest.skipUnless(find_spec("torch"), "install torch to run this test")
    def test_adaptive_torch_sbm_tracks_and_polishes_candidates(self) -> None:
        problem = QUBOProblem(
            numpy.array([-1.0, -0.5, -0.8, -0.2]),
            numpy.array([0, 0, 1, 2], dtype=numpy.uint32),
            numpy.array([2, 3, 2, 3], dtype=numpy.uint32),
            numpy.array([0.2, -0.1, -0.3, 0.4]),
            oneHotGroups=((0, 1), (2, 3)),
        )
        solver = AdaptiveTorchSBMBQMSolver("cpu")

        result = solver.solve(
            problem,
            {
                "steps": 60,
                "runs": 4,
                "dt": 0.1,
                "sampling_period": 5,
                "convergence_threshold": 2,
                "dtype": "float32",
                "seed": 37,
            },
        )

        self.assertEqual(sum(result.sample[:2]), 1)
        self.assertEqual(sum(result.sample[2:]), 1)
        self.assertAlmostEqual(result.energy, problem.energy(result.sample))
        self.assertLessEqual(solver.lastStepCount, 60)

    @unittest.skipUnless(find_spec("torch"), "install torch to run this test")
    def test_adaptive_torch_sbm_stops_when_agent_energies_stabilize(self) -> None:
        problem = QUBOProblem(
            numpy.zeros(3),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([]),
        )
        solver = AdaptiveTorchSBMBQMSolver("cpu")

        solver.solve(
            problem,
            {
                "steps": 100,
                "runs": 2,
                "sampling_period": 5,
                "convergence_threshold": 2,
                "early_stopping": True,
                "local_search_sweeps": 0,
                "seed": 41,
            },
        )

        self.assertLess(solver.lastStepCount, 100)

    def test_torch_sbm_qubo_to_ising_conversion_preserves_energies(self) -> None:
        problem = QUBOProblem(
            numpy.array([-1.2, 0.3, 0.7]),
            numpy.array([0, 1, 1, 2], dtype=numpy.uint32),
            numpy.array([1, 0, 2, 2], dtype=numpy.uint32),
            numpy.array([0.4, -0.1, 0.8, -0.25]),
            offset=0.6,
        )
        converted = TorchSBMBQMSolver._toIsingProblem(
            problem,
            numpy.float64,
            configuredC0=0.02,
        )
        differences = []
        for mask in range(1 << problem.variableCount):
            binary = numpy.array(
                [
                    (mask >> variable) & 1
                    for variable in range(problem.variableCount)
                ],
                dtype=numpy.uint8,
            )
            spins = 2.0 * binary - 1.0
            ising_energy = -converted.forceField @ spins - numpy.sum(
                converted.couplings
                * spins[converted.heads]
                * spins[converted.tails]
            )
            differences.append(problem.energy(binary) - ising_energy)

        numpy.testing.assert_allclose(
            differences,
            numpy.full(len(differences), differences[0]),
            atol=1e-12,
        )

    @unittest.skipUnless(find_spec("torch"), "install torch to run this test")
    def test_torch_sbm_batches_scenarios_and_trajectories(self) -> None:
        problems = [
            QUBOProblem(
                numpy.array([-1.0, -1.0]),
                numpy.array([0], dtype=numpy.uint32),
                numpy.array([1], dtype=numpy.uint32),
                numpy.array([2.0]),
                oneHotGroups=((0, 1),),
            ),
            QUBOProblem(
                numpy.array([-0.8, 0.2, -0.6]),
                numpy.array([0, 1], dtype=numpy.uint32),
                numpy.array([1, 2], dtype=numpy.uint32),
                numpy.array([0.3, -0.2]),
            ),
        ]
        parameters = {
            "steps": 100,
            "runs": 4,
            "run_batch_size": 2,
            "dt": 0.1,
            "dtype": "float64",
            "seed": 31,
        }

        results = TorchSBMBQMSolver("cpu").solveMany(problems, parameters)

        self.assertEqual(len(results), 2)
        for problem, result in zip(problems, results):
            self.assertEqual(len(result.sample), problem.variableCount)
            self.assertAlmostEqual(
                result.energy,
                problem.energy(result.sample),
                places=12,
            )
        self.assertEqual(sum(results[0].sample), 1)

    @unittest.skipUnless(find_spec("torch"), "install torch to run this test")
    def test_torch_sbm_run_batching_preserves_seeded_result(self) -> None:
        problem = QUBOProblem(
            numpy.array([-0.7, 0.2, -0.1]),
            numpy.array([0, 1], dtype=numpy.uint32),
            numpy.array([1, 2], dtype=numpy.uint32),
            numpy.array([0.5, -0.4]),
        )
        parameters = {
            "steps": 25,
            "runs": 4,
            "dt": 0.1,
            "dtype": "float64",
            "seed": 11,
        }

        together = TorchSBMBQMSolver("cpu").solve(
            problem,
            parameters,
        )
        split = TorchSBMBQMSolver("cpu").solve(
            problem,
            parameters | {"run_batch_size": 1},
        )

        self.assertEqual(tuple(together.sample), tuple(split.sample))
        self.assertAlmostEqual(together.energy, split.energy, places=12)

    def test_numeric_qubo_arrays_are_immutable(self) -> None:
        problem = QUBOProblem(
            numpy.array([-1.0]),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([], dtype=numpy.uint32),
            numpy.array([]),
        )

        with self.assertRaises(ValueError):
            problem.linear[0] = 2.0

    def test_base_solver_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BQMSolver()


if __name__ == "__main__":
    unittest.main()

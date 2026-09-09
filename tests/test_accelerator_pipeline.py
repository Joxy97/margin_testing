"""Numerical equivalence and bounded pipeline regression tests."""

from datetime import date
from importlib.util import find_spec
from threading import Event
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy
import pandas
import yaml

from margin_engine import MarginApplicationConfig
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolver, BatchBQMExecutionPolicy, TorchSVLBQMSolver,
)
from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_result import BQMOptimizationResult
from risk_state_generator import (
    NumpyPCABackend, TorchPCABackend, PCABackendConfig, ReturnsPCAGrid, ReturnsPCAKey,
)


def problem(shift=0):
    return QUBOProblem(numpy.array([-0.7, 0.2, -0.1]) + shift,
                       numpy.array([0, 1], dtype=numpy.uint32),
                       numpy.array([1, 2], dtype=numpy.uint32), numpy.array([0.5, -0.4]))


@unittest.skipUnless(find_spec("torch"), "requires torch")
class PCABackendTest(unittest.TestCase):
    def test_complete_local_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prices = pandas.DataFrame({"date": pandas.date_range("2024-01-01", periods=11),
                                       "A": [100, 101, 98, 103, 101, 105, 104, 108, 107, 110, 111]})
            prices.to_csv(root / "prices.csv", index=False)
            config = {
                "marginDate": "2024-01-11", "portfolio": {"weights": {"A": 10}},
                "engine": {
                    "downloadManager": {"providers": {"local": "local_csv"},
                                        "requestParameters": {"location": "prices.csv"}},
                    "riskStateGenerator": {"ew_window": 5, "components": 1,
                        "scenariosPerComponents": [3], "nZBins": 3, "allowEmptyBinFallback": True,
                        "pcaGridProvider": {"backend": {"type": "torch", "device": "cpu"}}},
                    "marginCalculator": {"type": "bqm",
                        "executionPolicy": {"type": "batch", "batchSize": 1, "prefetch": True},
                        "solver": {"type": "torch_svl", "constructorParameters": {"device": "cpu"},
                                   "solverParameters": {"steps": 10, "runs": 4}}},
                },
            }
            path = root / "margin.yaml"
            path.write_text(yaml.safe_dump(config))
            accelerated = MarginApplicationConfig.fromYaml(path).generateReport()
            config["engine"]["riskStateGenerator"]["pcaGridProvider"]["backend"]["type"] = "numpy"
            config["engine"]["marginCalculator"]["executionPolicy"]["prefetch"] = False
            path.write_text(yaml.safe_dump(config))
            baseline = MarginApplicationConfig.fromYaml(path).generateReport()
            self.assertAlmostEqual(accelerated.margin, baseline.margin, places=12)

    def check_device(self, device, dtype="auto"):
        for observations, assets in ((20, 5), (6, 15)):
            with self.subTest(device=device, shape=(observations, assets)):
                rng = numpy.random.default_rng(11)
                values = rng.normal(size=(observations, assets))
                weights = 0.94 ** numpy.arange(observations - 1, -1, -1)
                weights /= weights.sum()
                before = values.copy()
                expected = NumpyPCABackend().fit(values, weights, 3)
                result = TorchPCABackend(device, dtype).fit(values, weights, 3)
                single = dtype == "float32" or (dtype == "auto" and device.startswith("cuda"))
                for name in vars(expected):
                    actual = getattr(result, name)
                    self.assertEqual(actual.dtype, numpy.float32 if single else numpy.float64)
                    numpy.testing.assert_allclose(actual, getattr(expected, name),
                        rtol=2e-5 if single else 1e-7, atol=2e-5 if single else 1e-11)
                numpy.testing.assert_array_equal(values, before)

    def test_cpu_equivalence(self):
        self.check_device("cpu")

    def test_float32_equivalence_and_invalid_precision(self):
        self.check_device("cpu", "float32")
        for dtype in ("float16", "invalid"):
            with self.assertRaisesRegex(ValueError, "dtype"):
                TorchPCABackend("cpu", dtype)
        with self.assertRaisesRegex(TypeError, "dtype"):
            PCABackendConfig(type="torch", dtype=None)
        with self.assertRaisesRegex(ValueError, "NumPy"):
            PCABackendConfig(dtype="float32")
        for value in (1e40, 1e20):
            with self.assertRaisesRegex(ValueError, "finite"):
                TorchPCABackend("cpu", "float32").fit(
                    numpy.array([[value, 0.], [-value, 1.]]), numpy.array([.5, .5]), 1)

    def test_cuda_equivalence(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm unavailable")
        self.check_device("cuda:0")
        self.check_device("cuda:0", "float64")

    def test_grid_temporal_window_and_backend_injection(self):
        rng = numpy.random.default_rng(7)
        values = 100 * numpy.exp(rng.normal(0, .01, size=(12, 5)).cumsum(axis=0))
        data = pandas.DataFrame(values, columns=list("ABCDE"), index=pandas.date_range("2024-01-01", periods=12))
        key = ReturnsPCAKey(list("ABCDE"), 8, date(2024, 1, 10), .94, 2)
        baseline = ReturnsPCAGrid.construct(key, data)
        data.iloc[9:] = 1e50
        accelerated = ReturnsPCAGrid.construct(key, data, TorchPCABackend("cpu"))
        for name in ("lambdas", "loadings", "factors", "residuals", "logReturnMean", "logReturnScale"):
            numpy.testing.assert_allclose(getattr(baseline, name), getattr(accelerated, name), atol=1e-11)
        self.assertEqual(accelerated.calibrationEndDate, date(2024, 1, 9))

    def test_invalid_inputs(self):
        for backend in (NumpyPCABackend(), TorchPCABackend("cpu")):
            for values, weights, components in (
                (numpy.ones((1, 3)), numpy.ones(1), 1),
                (numpy.ones((3, 2)), numpy.ones(3)/3, 4),
                (numpy.ones((3, 2)), numpy.ones(2)/2, 1),
                (numpy.full((3, 2), numpy.nan), numpy.ones(3)/3, 1),
                (numpy.ones((3, 2)), numpy.ones(3)/3, 1),
            ):
                with self.assertRaises(ValueError):
                    backend.fit(values, weights, components)

    def test_yaml_and_strict_validation(self):
        config = {"marginDate": "2024-01-10", "portfolio": {"weights": {"A": 1}},
                  "engine": {"riskStateGenerator": {"pcaGridProvider": {
                      "backend": {"type": "torch", "device": "cpu"}}},
                      "marginCalculator": {"type": "bqm", "executionPolicy": {
                          "type": "batch", "prefetch": True}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.yaml"
            path.write_text(yaml.safe_dump(config))
            application = MarginApplicationConfig.fromYaml(path)
            provider = application.engine.riskStateGenerator.pcaGridProvider
            self.assertEqual(provider.backend, PCABackendConfig(type="torch", device="cpu"))
            self.assertTrue(application.engine.marginCalculator.executionPolicy.prefetch)
            config["engine"]["riskStateGenerator"]["pcaGridProvider"]["backend"]["typo"] = 1
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                MarginApplicationConfig.fromYaml(path)
        for settings in ({"type": "invalid"}, {"type": "numpy", "device": "cuda"}, {"device": "bad"}):
            with self.assertRaises(ValueError):
                PCABackendConfig(**settings)


@unittest.skipUnless(find_spec("torch"), "requires torch")
class TorchPipelineTest(unittest.TestCase):
    def test_float32_resident_coefficients_rank_original_float64_energy(self):
        import torch
        from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
        from margin_calculator.optimization.optimization_solver.bqm_solver.torch_qubo import TorchQUBO
        # Float32 collapses this difference; lexicographic ties would pick the wrong sample.
        p = QUBOProblem(numpy.array([-1.00000001, -1.]), numpy.array([], dtype=numpy.uint32),
                        numpy.array([], dtype=numpy.uint32), numpy.array([]), oneHotGroups=((0, 1),))
        coefficients = TorchQUBO(p, torch.tensor(p.linear, dtype=torch.float32),
            torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64), torch.empty(0))
        accumulator = TorchCandidateAccumulator(torch, p, "cpu", 1, BQMSolver._selectBestCandidates, coefficients)
        accumulator.add(torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8))
        self.assertEqual(accumulator.result(), ((1, 0), p.energy((1, 0))))

    def test_svl_invalid_numeric_parameters(self):
        for settings in ({"noise_chunk_size": 0}, {"temperature": float("nan")}, {"dt": float("inf")}):
            with self.assertRaises(ValueError):
                TorchSVLBQMSolver._getParameters(settings)

    def test_raw_svl_samples_preserve_run_and_problem_batching(self):
        collected = []
        original = TorchSVLBQMSolver._runTrajectories

        def capture(*args):
            samples = original(*args)
            collected.append(samples.cpu().numpy().copy())
            return samples

        params = {"steps": 19, "runs": 5, "noise_chunk_size": 8, "dtype": "float64", "seed": 13}
        problems = [problem(), problem(.2)]
        with patch.object(TorchSVLBQMSolver, "_runTrajectories", staticmethod(capture)):
            TorchSVLBQMSolver("cpu").solveMany(problems, params)
            together = collected.pop()
            TorchSVLBQMSolver("cpu").solveMany(problems, params | {"run_batch_size": 2})
            split = numpy.concatenate(collected, axis=1)
            collected.clear()
            for p in problems:
                TorchSVLBQMSolver("cpu").solve(p, params)
            separate = numpy.concatenate(collected, axis=0)
        numpy.testing.assert_array_equal(together, split)
        numpy.testing.assert_array_equal(together, separate)

    def test_candidate_accumulation_matches_full_selection(self):
        import torch
        from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
        p = QUBOProblem(numpy.array([-.7, .2, -.1]), numpy.array([0, 1], dtype=numpy.uint32),
                        numpy.array([1, 2], dtype=numpy.uint32), numpy.array([.5, -.4]),
                        oneHotGroups=((0, 1, 2),))
        for rows in ([[0, 0, 0], [1, 1, 1]],
                     [[0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1]],
                     [[1, 0, 0], [0, 0, 1], [1, 1, 1]]):
            accumulator = TorchCandidateAccumulator(torch, p, "cpu", 1, BQMSolver._selectBestCandidates)
            for sample in rows:
                accumulator.add(torch.tensor([sample], dtype=torch.uint8))
            expected = BQMSolver._selectBestCandidates(((s, p.energy(s)) for s in rows), p)
            self.assertEqual(accumulator.result(), expected)

    def test_memory_estimate_accounts_for_runs_and_noise(self):
        solver = TorchSVLBQMSolver()
        small = solver.estimatedWorkingMemoryBytes(problem(), {"runs": 1, "noise_chunk_size": 1})
        large = solver.estimatedWorkingMemoryBytes(problem(), {"runs": 16, "noise_chunk_size": 32})
        self.assertGreater(small, problem().numericMemoryBytes)
        self.assertGreater(large, small)

    def test_candidate_ties_and_original_diagonal_energy(self):
        import torch
        from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
        p = QUBOProblem(numpy.zeros(3), numpy.array([0, 0, 1], dtype=numpy.uint32),
                        numpy.array([0, 1, 0], dtype=numpy.uint32), numpy.array([-1., .5, -.5]), offset=2)
        rows = [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
        accumulator = TorchCandidateAccumulator(torch, p, "cpu", 1, BQMSolver._selectBestCandidates)
        accumulator.add(torch.tensor(rows, dtype=torch.uint8))
        self.assertEqual(accumulator.result(), ((1, 0, 0), 1.0))


class PipelineSchedulingTest(unittest.TestCase):
    def test_torch_rejects_wrong_problem_types_before_memory_estimation(self):
        from margin_calculator.optimization.optimization_solver.bqm_solver import TorchSBMBQMSolver
        with self.assertRaises(TypeError):
            TorchSBMBQMSolver(device="cpu").solveMany([None])


    def test_memory_observer_counts_active_and_prefetched_coefficients(self):
        measurements = []
        ready = Event()
        size = problem().numericMemoryBytes

        def observe(measurement):
            measurements.append(measurement)
            if measurement.retainedCoefficientBytes == 2 * size:
                ready.set()

        class Solver(BQMSolver):
            def solve(self, p, solverParameters=None):
                if not ready.wait(2):
                    raise AssertionError("prefetched retention was not accounted")
                return BQMOptimizationResult((0, 0, 0), 0)

        list(BatchBQMExecutionPolicy(batchSize=1, prefetch=True, memoryObserver=observe)
             .execute(Solver(), enumerate([problem(), problem()])))
        self.assertTrue(any(m.activeCoefficientBytes == size and
                            m.producerCoefficientBytes == size for m in measurements))
        self.assertEqual(measurements[-1].retainedCoefficientBytes, 0)
        self.assertEqual(measurements[-1].peakRetainedCoefficientBytes, 2 * size)


    def test_admitted_resource_plan_is_used_by_the_solver(self):
        class PlannedSolver(BQMSolver):
            @property
            def batchParallelism(self):
                return 2

            def estimatedWorkingMemoryBytes(self, p, solverParameters=None):
                return 1000

            def solve(self, p, solverParameters=None):
                raise AssertionError("resource assignment was discarded")

            def solvePlanned(self, problems, plan, solverParameters=None):
                for (start, stop), memory in zip(plan.shards, plan.workerBytes):
                    if stop - start > 1 or memory != 1000:
                        raise AssertionError("worker exceeds admitted memory")
                return [BQMOptimizationResult((0, 0, 0), p.energy((0, 0, 0))) for p in problems]

        result = list(BatchBQMExecutionPolicy(batchSize=5, maxBatchBytes=1500).execute(
            PlannedSolver(), enumerate([problem()] * 5)))
        self.assertEqual([context for context, _ in result], list(range(5)))

    def test_prefetch_exception_and_early_close_cleanup(self):
        for fail in (False, True):
            closed = []

            def items():
                try:
                    for i in range(4):
                        yield i, problem()
                finally:
                    closed.append("source")

            class Solver(BQMSolver):
                def solve(self, p, solverParameters=None):
                    if fail:
                        raise RuntimeError("solve failed")
                    return BQMOptimizationResult((0, 0, 0), 0)

                def endSeries(self):
                    closed.append("solver")

            iterator = BatchBQMExecutionPolicy(batchSize=1, prefetch=True).execute(Solver(), items())
            if fail:
                with self.assertRaisesRegex(RuntimeError, "solve failed"):
                    next(iterator)
            else:
                next(iterator)
                iterator.close()
            self.assertEqual(closed, ["source", "solver"])

    def test_heterogeneous_shard_budget(self):
        policy = BatchBQMExecutionPolicy(maxBatchBytes=100)
        self.assertTrue(policy._exceedsWorkerBudget([80, 80, 10, 10], 2))
        self.assertFalse(policy._exceedsWorkerBudget([80, 10, 80, 10], 2))

    def test_prefetch_overlaps_production_and_closes_series(self):
        ready = Event()
        closed = []

        def items():
            try:
                yield 0, problem()
                ready.set()
                yield 1, problem()
            finally:
                closed.append("source")

        class Solver(BQMSolver):
            def solve(self, p, solverParameters=None):
                if not ready.wait(2):
                    raise AssertionError("next batch was not produced during solve")
                return BQMOptimizationResult((0, 0, 0), 0)

            def endSeries(self):
                closed.append("solver")

        results = list(BatchBQMExecutionPolicy(batchSize=1, prefetch=True).execute(Solver(), items()))
        self.assertEqual([c for c, _ in results], [0, 1])
        self.assertEqual(closed, ["source", "solver"])

    def test_solver_estimate_limits_batches(self):
        batches = []

        class Solver(BQMSolver):
            def estimatedWorkingMemoryBytes(self, p, solverParameters=None):
                return 1000

            def solve(self, p, solverParameters=None):
                return BQMOptimizationResult((0, 0, 0), 0)

            def solveMany(self, problems, solverParameters=None):
                batches.append(len(problems))
                return super().solveMany(problems, solverParameters)

        list(BatchBQMExecutionPolicy(batchSize=10, maxBatchBytes=1500).execute(
            Solver(), enumerate([problem()] * 3)))
        self.assertEqual(batches, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()

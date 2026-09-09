"""Feasible categorical dynamics and source-energy semantics."""

from importlib.util import find_spec
from itertools import product
import tempfile
import unittest
from unittest.mock import patch

import numpy
import yaml

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory, TorchCategoricalBQMSolver
from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_categorical_bqm_solver import _CategoricalModel


def fixture():
    # Ragged noncontiguous groups, diagonals, reversed and duplicate edges,
    # cancelling edges, and a within-group interaction that must vanish.
    return QUBOProblem(numpy.array([.4, -.2, .1, .3, -.4]),
        numpy.array([0, 2, 1, 2, 0, 0, 4], dtype=numpy.uint32),
        numpy.array([0, 1, 2, 0, 3, 3, 4], dtype=numpy.uint32),
        numpy.array([-.7, .8, -.3, 300., 2., -2., .1]),
        oneHotGroups=((2, 0), (4, 1, 3)), offset=2., seedOffset=17)


class CategoricalModelTest(unittest.TestCase):
    def test_compiled_energy_equals_source_up_to_one_constant(self):
        p = fixture()
        model = _CategoricalModel.fromProblem(p)
        differences = []
        for choices in product(*model.groups):
            x = numpy.zeros(p.variableCount)
            x[list(choices)] = 1
            differences.append(p.energy(x) - (model.linear @ x + .5 * x @ model.adjacency @ x))
        numpy.testing.assert_allclose(differences, differences[0], rtol=0, atol=1e-12)
        for color in model.colors:
            for a, b in product(color, repeat=2):
                self.assertEqual(model.adjacency[list(model.groups[a])][:, list(model.groups[b])].nnz, 0)

    def test_complete_group_coverage_required(self):
        for groups in ((), ((0, 1),)):
            p = QUBOProblem(numpy.zeros(3), numpy.array([], dtype=numpy.uint32),
                numpy.array([], dtype=numpy.uint32), numpy.array([]), oneHotGroups=groups)
            with self.assertRaisesRegex(ValueError, 'covering every variable'):
                _CategoricalModel.fromProblem(p)
            with self.assertRaisesRegex(ValueError, 'covering every variable'):
                TorchCategoricalBQMSolver().estimatedWorkingMemoryBytes(p)

    def test_configuration_validation_and_factory(self):
        self.assertIsInstance(BQMSolverFactory.createBQMSolver('torch_categorical'), TorchCategoricalBQMSolver)
        for params in ({'steps': 0}, {'runs': -1}, {'seed': -1}, {'dtype': 'float16'},
                       {'temperature_start': float('nan')}, {'temperature_end': 2.},
                       {'noise_chunk_size': 0}, {'greedy_sweeps': -1}, {'run_batch_size': 0}, {'typo': 1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                TorchCategoricalBQMSolver._getParameters(params)
        with self.assertRaises(TypeError):
            TorchCategoricalBQMSolver._getParameters({'steps': 1.5})


@unittest.skipUnless(find_spec('torch'), 'requires Torch')
class TorchCategoricalTest(unittest.TestCase):
    def check_device(self, device):
        import torch
        p = fixture()
        original = p.linear.copy(), p.quadraticBiases.copy()
        calls = []
        scatter = torch.Tensor.scatter_
        add = TorchCandidateAccumulator.add
        def audit_scatter(tensor, dim, index, value, *args, **kwargs):
            result = scatter(tensor, dim, index, value, *args, **kwargs)
            if tensor.ndim == 2 and tensor.shape[0] == p.variableCount and value == 1.:
                for group in p.iterOneHotGroups():
                    self.assertTrue(bool((tensor[list(group)].sum(dim=0) == 1).all()))
                calls.append(1)
            return result
        def audit_candidates(accumulator, samples):
            self.assertTrue(all(bool((samples[:, list(g)].sum(dim=1) == 1).all()) for g in p.iterOneHotGroups()))
            return add(accumulator, samples)
        for dtype in ('float32', 'float64'):
            with patch.object(torch.Tensor, 'scatter_', audit_scatter), \
                 patch.object(TorchCandidateAccumulator, 'add', audit_candidates), \
                 patch.object(CandidateSelection, '_repairCandidate', side_effect=AssertionError('repair invoked')):
                result = TorchCategoricalBQMSolver(device).solve(p, {'steps': 16, 'runs': 5, 'run_batch_size': 2, 'dtype': dtype})
            energies = []
            for choices in product(*p.iterOneHotGroups()):
                sample = numpy.zeros(p.variableCount)
                sample[list(choices)] = 1
                energies.append(p.energy(sample))
            self.assertAlmostEqual(result.energy, min(energies), places=12)
            self.assertEqual(result.energy, p.energy(result.sample))
        self.assertGreater(len(calls), 32)
        numpy.testing.assert_array_equal(p.linear, original[0])
        numpy.testing.assert_array_equal(p.quadraticBiases, original[1])

    def test_cpu_feasible_at_every_update_and_exact_source_energy(self):
        self.check_device('cpu')

    def test_cuda_feasible_at_every_update_and_exact_source_energy(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('requires CUDA/ROCm')
        self.check_device('cuda:0')

    def test_run_and_problem_chunking_preserve_raw_candidates(self):
        p = fixture()
        solver = TorchCategoricalBQMSolver('cpu')
        captured = []
        add = TorchCandidateAccumulator.add
        def audit(accumulator, samples):
            captured.append(samples.cpu().numpy().copy())
            return add(accumulator, samples)
        params = {'steps': 19, 'runs': 5, 'seed': 13, 'noise_chunk_size': 8}
        with patch.object(TorchCandidateAccumulator, 'add', audit):
            together = solver.solveMany([p, p], params)
            all_samples = numpy.concatenate(captured)
            captured.clear()
            split = [solver.solve(p, params | {'run_batch_size': 2}) for _ in range(2)]
            split_samples = numpy.concatenate(captured)
        self.assertEqual(together, split)
        numpy.testing.assert_array_equal(all_samples, split_samples)

    def test_zero_temperature_matches_dense_categorical_descent(self):
        import torch
        from margin_calculator.optimization.optimization_solver.bqm_solver.torch_execution import _RUN_SEED_STRIDE, _MAX_TORCH_SEED
        p = fixture()
        model = _CategoricalModel.fromProblem(p)
        params = {'steps': 3, 'runs': 3, 'seed': 7, 'temperature_start': 0., 'temperature_end': 0., 'greedy_sweeps': 0, 'dtype': 'float64'}
        expected = []
        for run in range(3):
            seed = (7 + p.seedOffset + _RUN_SEED_STRIDE * run) % _MAX_TORCH_SEED
            u = torch.rand(len(model.groups), generator=torch.Generator().manual_seed(seed), dtype=torch.float64).numpy()
            x = numpy.zeros(p.variableCount)
            choices = [g[int(v * len(g))] for g, v in zip(model.groups, u)]
            x[choices] = 1
            for _ in range(3):
                for color in model.colors:
                    costs = model.linear + model.adjacency @ x
                    for group in color:
                        variables = model.groups[group]
                        selected = variables[int(numpy.argmin(costs[list(variables)]))]
                        previous = choices[group]
                        if costs[selected] < costs[previous] - 1e-12:
                            x[previous], x[selected] = 0, 1
                            choices[group] = selected
            expected.append(x)
        captured = []
        add = TorchCandidateAccumulator.add
        def audit(accumulator, samples):
            captured.extend(samples.cpu().numpy())
            return add(accumulator, samples)
        with patch.object(TorchCandidateAccumulator, 'add', audit):
            TorchCategoricalBQMSolver('cpu').solve(p, params)
        numpy.testing.assert_array_equal(captured, expected)

    def test_singleton_groups_and_no_cross_edges(self):
        p = QUBOProblem(numpy.array([3., -1., -2.]), numpy.array([], dtype=numpy.uint32),
            numpy.array([], dtype=numpy.uint32), numpy.array([]), oneHotGroups=((0,), (1, 2)))
        result = TorchCategoricalBQMSolver('cpu').solve(p, {'steps': 1, 'runs': 1})
        self.assertEqual(result.sample, (1, 0, 1))
        self.assertEqual(result.energy, 1.)

    def test_yaml_host_and_resident_margin_and_batch_policy(self):
        from tests.test_device_resident_pipeline import DeviceResidentPipelineTest
        from margin_engine import MarginApplicationConfig
        with tempfile.TemporaryDirectory() as directory:
            calculator = {'type': 'bqm', 'comparison': {'type': 'state_aware_greedy'},
                'solver': {'type': 'torch_categorical', 'constructorParameters': {'device': 'cpu'},
                    'solverParameters': {'steps': 4, 'runs': 3, 'temperature_start': .5, 'temperature_end': 0.}},
                'executionPolicy': {'type': 'batch', 'batchSize': 2}}
            config = DeviceResidentPipelineTest().configuration(directory, calculator)
            for resident in (True, False):
                if not resident:
                    del config['engine']['numericalExecution']
                with patch.object(CandidateSelection, '_repairCandidate', side_effect=AssertionError('repair invoked')):
                    report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
                self.assertAlmostEqual(report.margin, .1798027685847499, places=12)
                self.assertAlmostEqual(report.comparisonMargins['greedy'], report.margin, places=12)

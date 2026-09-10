"""Transverse-route equations, QUBO semantics, batching and YAML integration."""

from importlib.util import find_spec
from itertools import product
import tempfile
import unittest
from unittest.mock import patch

import numpy
import yaml

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolverFactory, TorchTransverseRouteBQMSolver,
)
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_transverse_route_bqm_solver import _FlowWorkspace


def fixture(seed=17, groups=()):
    return QUBOProblem(numpy.array([.4, -.2, .1, .3, -.4]),
        numpy.array([0, 2, 1, 2, 0, 0, 4], dtype=numpy.uint32),
        numpy.array([0, 1, 2, 0, 3, 3, 4], dtype=numpy.uint32),
        numpy.array([-.7, .8, -.3, 1.3, 2., -2., .1]),
        oneHotGroups=groups, offset=2., seedOffset=seed)


class TransverseRouteConfigTest(unittest.TestCase):
    def test_factory_aliases_and_invalid_parameters(self):
        self.assertIsInstance(BQMSolverFactory.create('torch_transverse_route'), TorchTransverseRouteBQMSolver)
        p = TorchTransverseRouteBQMSolver._getParameters({'agents': 7, 'max_steps': 3})
        self.assertEqual((p['runs'], p['steps']), (7, 3))
        for supplied in ({'steps': 0}, {'runs': -1}, {'seed': -1}, {'gamma': -1},
                         {'time_step': 0}, {'mobility': float('inf')}, {'route_strength': float('nan')},
                         {'kappa_final': -2}, {'dtype': 'float16'}, {'integrator': 'rk4'},
                         {'matrix_format': 'coo'}, {'sparse_threshold': 2}, {'typo': 2},
                         {'agents': 2, 'runs': 2}, {'run_batch_size': 0}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                TorchTransverseRouteBQMSolver._getParameters(supplied)
        for supplied in ({'steps': 1.5}, {'seed': True}, {'cuda_graph': 'false'}, {'gamma': '1'}):
            with self.subTest(supplied=supplied), self.assertRaises(TypeError):
                TorchTransverseRouteBQMSolver._getParameters(supplied)

    def test_normalization_preserves_qubo_energy_and_duplicates(self):
        q = fixture()
        linear = q.linear.copy()
        adjacency = numpy.zeros((5, 5))
        for i, j, bias in zip(q.quadraticHeads, q.quadraticTails, q.quadraticBiases):
            if i == j:
                linear[i] += bias
            else:
                adjacency[i, j] += bias / 4
                adjacency[j, i] += bias / 4
        h = linear / 2 + adjacency.sum(axis=1)
        scale = (numpy.abs(h) + numpy.abs(adjacency).sum(axis=1)).max()
        model = TorchTransverseRouteBQMSolver._toIsingProblem(q, numpy.float64, 1.)
        numpy.testing.assert_allclose(model.forceField, h / scale, atol=1e-15)
        differences = []
        for bits in product((0, 1), repeat=5):
            spins = 2 * numpy.array(bits) - 1
            energy = model.forceField @ spins + numpy.sum(model.couplings * spins[model.heads] * spins[model.tails])
            differences.append(q.energy(bits) - scale * energy)
        numpy.testing.assert_allclose(differences, differences[0], atol=1e-14)

    def test_memory_bounds_and_size_guards(self):
        solver, q = TorchTransverseRouteBQMSolver('cpu'), fixture()
        a = solver.estimatedWorkingMemoryBytes(q, {'candidate_batch_size': 16})
        b = solver.estimatedWorkingMemoryBytes(q, {'candidate_batch_size': 128})
        self.assertGreater(b, a)
        for p in ({'max_variables': 4}, {'matrix_format': 'dense', 'max_dense_variables': 4}):
            with self.assertRaises(ValueError):
                solver.estimatedWorkingMemoryBytes(q, p)


@unittest.skipUnless(find_spec('torch'), 'requires Torch')
class TransverseRouteDynamicsTest(unittest.TestCase):
    def test_rhs_and_integrators_against_numpy_reference(self):
        import torch
        matrix = numpy.array([[0., .3, -.2], [.3, 0., .1], [-.2, .1, 0.]])
        field = numpy.array([[.2], [-.1], [.3]])
        initial = numpy.array([[.2, -1.], [1.1, 2.4], [-2.2, .7]])
        for integrator, route, gamma in product(('euler', 'heun'), (0., 1.), (0., .4)):
            p = TorchTransverseRouteBQMSolver._getParameters({
                'integrator': integrator, 'route_strength': route, 'gamma': gamma, 'mobility': .7})
            theta = torch.tensor(initial)
            workspace = _FlowWorkspace(torch, torch.tensor(matrix), torch.tensor(field), theta, p)

            def rhs(t, kappa):
                x, y = numpy.cos(t), numpy.sin(t)
                return .7 * (y * (matrix @ x + field)
                    - route * (x*x - y*y) * (matrix @ (x*y))
                    - 3 * gamma * x*y*y * (matrix @ (y*y*y)) - kappa*x*y)

            first = rhs(initial, -.6)
            numpy.testing.assert_allclose(workspace.rhs(theta, -.6, workspace.first).numpy(), first, atol=1e-14)
            wrap = lambda t: (t + numpy.pi) % (2 * numpy.pi) - numpy.pi
            expected = wrap(initial + .05 * first)
            if integrator == 'heun':
                expected = wrap(initial + .025 * (first + rhs(expected, .2)))
            workspace.advance(torch.tensor(-.6, dtype=torch.float64), torch.tensor(.2, dtype=torch.float64))
            numpy.testing.assert_allclose(theta.numpy(), expected, atol=1e-14)

    def test_batches_chunks_and_rng_isolation(self):
        import torch
        solver = TorchTransverseRouteBQMSolver('cpu')
        problems = [fixture(17), fixture(18)]
        p = {'steps': 31, 'runs': 7, 'dtype': 'float64', 'candidate_interval': 6,
             'run_batch_size': 7, 'candidate_batch_size': 49}
        state = torch.random.get_rng_state().clone()
        originals = [q.quadraticBiases.copy() for q in problems]
        expected = [solver.solve(q, p) for q in problems]
        actual = solver.solveMany(problems, p)
        chunked = solver.solveMany(problems, {**p, 'run_batch_size': 3, 'candidate_batch_size': 3})
        dense = solver.solveMany(problems, {**p, 'matrix_format': 'dense'})
        self.assertEqual(actual, expected)
        self.assertEqual(chunked, expected)
        self.assertEqual(dense, expected)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for q, result, original in zip(problems, actual, originals):
            self.assertEqual(result.energy, q.energy(result.sample))
            numpy.testing.assert_array_equal(q.quadraticBiases, original)

    def test_zero_problem_and_one_hot_repair(self):
        solver = TorchTransverseRouteBQMSolver('cpu')
        empty = numpy.array([], dtype=numpy.uint32)
        q = QUBOProblem(numpy.zeros(3), empty, empty, numpy.array([]), offset=3.)
        self.assertEqual(solver.solve(q, {'steps': 2, 'runs': 2}).energy, 3.)
        q = fixture(groups=((2, 0), (4, 1, 3)))
        result = solver.solve(q, {'steps': 7, 'runs': 5, 'candidate_interval': 2})
        self.assertTrue(all(sum(result.sample[i] for i in group) == 1 for group in q.iterOneHotGroups()))
        self.assertEqual(result.energy, q.energy(result.sample))
        self.assertEqual(solver.solveMany([]), [])

    def test_duplicate_checkpoints_keep_the_same_repaired_winner(self):
        from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
        q = fixture(groups=((2, 0), (4, 1, 3)))
        solver = TorchTransverseRouteBQMSolver('cpu')
        p = {'steps': 9, 'runs': 4, 'candidate_interval': 1, 'candidate_batch_size': 16}
        counts = []
        original = TorchCandidateAccumulator.add

        def add(accumulator, samples):
            counts.append(len(samples))
            return original(accumulator, samples)

        # Once all trajectories reach one endpoint, checkpoints repeat exactly.
        def stationary(workspace, *_):
            workspace.theta.fill_(numpy.pi)

        with patch.object(_FlowWorkspace, 'advance', stationary), \
                patch.object(TorchCandidateAccumulator, 'add', add):
            expected = solver.solve(q, {**p, 'deduplicate_candidates': False})
            full_count = sum(counts)
            counts.clear()
            actual = solver.solve(q, p)
        self.assertEqual(actual, expected)
        self.assertLess(sum(counts), full_count)

    def test_yaml_host_and_resident_pipeline(self):
        from tests.test_device_resident_pipeline import DeviceResidentPipelineTest
        from margin_engine import MarginApplicationConfig
        with tempfile.TemporaryDirectory() as directory:
            calculator = {'type': 'bqm', 'comparison': {'type': 'state_aware_greedy'},
                'solver': {'type': 'torch_transverse_route', 'constructorParameters': {'device': 'cpu'},
                    'solverParameters': {'steps': 7, 'runs': 4, 'candidate_interval': 3}},
                'executionPolicy': {'type': 'batch', 'batchSize': 2}}
            config = DeviceResidentPipelineTest().configuration(directory, calculator)
            margins = []
            for resident in (True, False):
                if not resident:
                    del config['engine']['numericalExecution']
                report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
                margins.append(report.margin)
                self.assertAlmostEqual(report.margin, .1798027685847499, places=12)
            self.assertEqual(margins[0], margins[1])

    def test_cuda_graph_and_all_device_shards(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('requires CUDA')
        p = {'steps': 31, 'runs': 7, 'run_batch_size': 3, 'candidate_interval': 6,
             'graph_steps': 4, 'dtype': 'float64', 'gamma': .2, 'integrator': 'heun'}
        solver = TorchTransverseRouteBQMSolver('cuda:0')
        expected = solver.solve(fixture(), {**p, 'cuda_graph': False})
        self.assertEqual(solver.solve(fixture(), p), expected)
        count = torch.cuda.device_count()
        if count > 1:
            problems = [fixture(17 + index) for index in range(count)]
            single = solver.solveMany(problems, p)
            multi = TorchTransverseRouteBQMSolver(devices=[f'cuda:{i}' for i in range(count)])
            self.assertEqual(multi.solveMany(problems, p), single)

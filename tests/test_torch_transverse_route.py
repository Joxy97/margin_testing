"""Canonical transverse-route equations and application boundary checks."""
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from qubo_solvers import QUBO, TransverseRoute, create_bqm_solver


class TransverseRouteDynamicsTest(unittest.TestCase):
    def test_rhs_against_independent_numpy_reference(self):
        j = np.array([[0., .3, -.2], [.3, 0., .5], [-.2, .5, 0.]])
        h = np.array([-.2, .4, .1])
        angles = np.array([[.1, -.7, 2.], [-1., .5, -.2]])
        for gamma in (0., .6):
            solver = TransverseRoute(feature_strength=.7, locking=.3,
                                     locking_start=-.2, gamma=gamma)
            objective = SimpleNamespace(j=torch.tensor(j), scale=1.7,
                gradient=lambda x: 2*(x@torch.tensor(j)+torch.tensor(h))/1.7)
            actual = solver._rhs(torch.tensor(angles), .4, objective).numpy()
            x, y = np.cos(angles), np.sin(angles)
            expected = y*(2*(x@j+h)/1.7)-.7*(x*x-y*y)*(2*((x*y)@j)/1.7)
            expected -= 3*gamma*y*y*x*(2*((y**3)@j)/1.7)
            expected -= (-.2+(.3+.2)*.4)*x*y
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_library_integrators_preserve_objective_and_inputs(self):
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            problem = QUBO(torch.tensor([[-1., .25], [.25, -.5]], device=device), 2.)
            original = problem.Q.clone()
            for integrator in ('euler', 'heun'):
                solver = TransverseRoute(max_steps=4, integrator=integrator)
                result = solver.solve(problem, restarts=3, batch_size=2, seed=7)
                torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments))
                torch.testing.assert_close(problem.Q, original)

    def test_removed_private_execution_options_are_rejected(self):
        solver = create_bqm_solver('lib_transverse_route')
        for options in ({'candidate_interval': 3}, {'cuda_graph': True}, {'matrix_format': 'sparse'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                solver._getParameters(options)

    def test_yaml_host_and_resident_pipeline(self):
        from tests.test_device_resident_pipeline import DeviceResidentPipelineTest
        from margin_engine import MarginApplicationConfig
        with tempfile.TemporaryDirectory() as directory:
            calculator = {'type': 'bqm', 'comparison': {'type': 'state_aware_greedy'},
                'solver': {'type': 'lib_transverse_route', 'constructorParameters': {'device': 'cpu'},
                    'solverParameters': {'steps': 7, 'runs': 4, 'seed': 13, 'dtype': 'float64'}},
                'executionPolicy': {'type': 'batch', 'batchSize': 2}}
            config = DeviceResidentPipelineTest().configuration(directory, calculator)
            margins = []
            for resident in (True, False):
                if not resident:
                    del config['engine']['numericalExecution']
                report = MarginApplicationConfig.fromYamlText(yaml.safe_dump(config), directory).generateReport()
                margins.append(report.margin)
                self.assertTrue(np.isfinite(report.margin))
                self.assertGreaterEqual(report.margin, 0.)
                self.assertLessEqual(report.margin, report.comparisonMargins['greedy']+1e-12)
                self.assertAlmostEqual(report.comparisonMargins['greedy'], .1798027685847499, places=12)
            self.assertAlmostEqual(margins[0], margins[1], places=12)

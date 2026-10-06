"""Registry boundaries preserve objectives and unavailable diagnostics honestly."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from qubo_solvers import (
    QUBO, Ising, LIBRARY_SOLVERS, NATIVE_SOLVERS, BACKEND_SOLVERS, SOLVERS, BackendSolver,
    REMOVED_SOLVERS, GPU_SOLVERS, CPU_SOLVERS,
    TerminationReason, create_solver, create_bqm_solver,
    default_algorithm_parameters, library_solver_class,
)


def test_registry_has_all_implementations():
    assert len(LIBRARY_SOLVERS) == 17
    assert len(BACKEND_SOLVERS) == 11
    assert len(GPU_SOLVERS) == 27
    assert CPU_SOLVERS == ('lib_planar_graph',)
    assert len(SOLVERS) == len(set(SOLVERS)) == 28
    for name in LIBRARY_SOLVERS:
        solver = create_solver(name, **default_algorithm_parameters(name, 0))
        assert isinstance(solver, library_solver_class(name))
    for name in BACKEND_SOLVERS:
        assert isinstance(create_solver(name), BackendSolver)
    for function in (create_solver, library_solver_class, create_bqm_solver):
        with pytest.raises(ValueError, match='Unknown'):
            function('missing_solver')


@pytest.mark.parametrize('name', LIBRARY_SOLVERS)
@pytest.mark.parametrize('spin', [False, True])
def test_registered_native_algorithms_preserve_objective(name, spin, dtype):
    problem = QUBO(torch.tensor([[-1., .25], [.25, -2.]], dtype=dtype), 9.5)
    if spin:
        problem = problem.to_ising()
    result = create_solver(name, **default_algorithm_parameters(name, 2)).solve(
        problem, restarts=3, batch_size=2, seed=31, best_only=True)
    torch.testing.assert_close(result.best_energy, problem.energy(result.best_assignment))
    assert result.best_assignments.shape == (1, 2)


@pytest.mark.parametrize('spin', [False, True])
def test_backend_wrapper_conversion_batch_seed_and_unknown_diagnostics(monkeypatch, spin):
    import qubo_solvers.registry as registry
    problem = QUBO(torch.tensor([[-3., .75], [.75, -2.]], dtype=torch.float64), 7.25)
    if spin:
        problem = problem.to_ising()
    calls = []

    class Backend:
        def solve(self, bqm, parameters):
            calls.append(parameters)
            for assignment in itertools.product((0, 1), repeat=2):
                native = torch.tensor(assignment, dtype=torch.int8)
                if spin:
                    native = 2 * native - 1
                assert bqm.energy(assignment) == pytest.approx(float(problem.energy(native)))
            return SimpleNamespace(sample=(1, 1), energy=bqm.energy((1, 1)))

    def factory(name, constructor):
        assert name == 'lib_simulated_bifurcation'
        assert constructor == {'device': 'cpu'}
        return Backend()

    monkeypatch.setattr(registry, 'create_bqm_solver', factory)
    saved = (problem.J if spin else problem.Q).clone()
    result = create_solver('lib_simulated_bifurcation', device='cpu', precision='float64', steps=2).solve(
        problem, restarts=7, seed=13, best_only=True)
    assert calls == [dict(steps=2, runs=7, seed=13, dtype='float64')]
    torch.testing.assert_close(result.best_energy, problem.energy(result.best_assignment))
    torch.testing.assert_close(problem.J if spin else problem.Q, saved)
    assert result.iterations.tolist() == [-1]
    assert result.termination_reasons.tolist() == [TerminationReason.UNKNOWN]
    assert result.best_restart_index.item() == -1
    assert result.final_assignments is None
    assert result.energy_history is None


def test_backend_wrapper_rejects_unsupported_contracts_before_loading_backend():
    problem = QUBO(torch.eye(2))
    solver = create_solver('lib_simulated_bifurcation', steps=1)
    with pytest.raises(ValueError, match='best_only'):
        solver.solve(problem)
    for options in (dict(initial_assignments=torch.zeros(1, 2)),
                    dict(history_interval=1), dict(batch_size=1), dict(memory_limit_bytes=100)):
        with pytest.raises(ValueError, match='cannot provide'):
            solver.solve(problem, best_only=True, **options)
    with pytest.raises(ValueError, match='dtype'):
        create_solver('lib_simulated_bifurcation', precision='float64').solve(problem, best_only=True)
    with pytest.raises(ValueError, match='one-hot'):
        create_solver('lib_categorical').solve(problem, best_only=True)
    with pytest.raises(ValueError, match='CPU float64'):
        create_solver('lib_planar_graph').solve(problem, best_only=True)
    for option in ('seed', 'runs', 'num_reads', 'dtype', 'devices'):
        with pytest.raises(ValueError, match='Use solve'):
            create_solver('lib_simulated_bifurcation', **{option: 1})


def test_backend_wrapper_rejects_nonbinary_backend_result(monkeypatch):
    import qubo_solvers.registry as registry
    backend = SimpleNamespace(solve=lambda problem, parameters: SimpleNamespace(sample=(.2, 1), energy=0.))
    monkeypatch.setattr(registry, 'create_bqm_solver', lambda *args: backend)
    with pytest.raises(ValueError, match='invalid binary'):
        create_solver('lib_simulated_bifurcation').solve(QUBO(torch.eye(2)), best_only=True)


def test_bqm_factory_registry_matches_public_library():
    from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory
    assert all(BQMSolverFactory.create(name) is not None for name in SOLVERS)
    for name in LIBRARY_SOLVERS:
        solver = create_bqm_solver(name, {'device': 'cpu'})
        assert solver is not None


def test_actual_library_torch_bridge_scores_ising():
    problem = Ising(torch.tensor([[0., .25], [.25, 0.]], dtype=torch.float64),
                    torch.tensor([-.5, .75], dtype=torch.float64), 8.)
    result = create_solver('lib_simulated_bifurcation', steps=3).solve(problem, restarts=2, seed=9, best_only=True)
    torch.testing.assert_close(result.best_energy, problem.energy(result.best_assignment))


@pytest.mark.parametrize('old_name', REMOVED_SOLVERS)
def test_retired_names_are_not_silent_aliases(old_name):
    for factory in (create_solver, create_bqm_solver):
        with pytest.raises(ValueError, match='Retired solver ID'):
            factory(old_name)


def test_every_compact_solver_is_owned_by_library():
    import inspect
    for name in SOLVERS:
        solver = create_bqm_solver(name)
        assert type(solver).__module__.startswith('qubo_solvers.')
        assert 'qubo_solvers' in inspect.getfile(type(solver))


def test_library_creation_does_not_import_application():
    import os
    import subprocess
    import sys
    code = ('import sys; from qubo_solvers import SOLVERS, create_bqm_solver; '
            '[create_bqm_solver(name) for name in SOLVERS]; '
            'assert not any(n.startswith("margin_calculator") for n in sys.modules)')
    completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert completed.returncode == 0, completed.stderr


def test_application_solver_modules_contain_no_numerical_kernels():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    application = root/'src/margin_calculator/optimization/optimization_solver/bqm_solver'
    kernel_names = {'_runTrajectories', '_search', '_rhs', '_run', '_polish', '_toIsingProblem'}
    for path in application.glob('*.py'):
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        functions = {node.name for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)}
        assert not functions & kernel_names, path
    assert not (root/'src/sbm/cpu_solver.cpp').exists()
    assert not (root/'cuda/gpu_solver.cu').exists()

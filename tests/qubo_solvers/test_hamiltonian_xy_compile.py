"""Optional XY fusion preserves the eager root and does not specialize controls."""

import importlib.util
import math

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers import hamiltonian_discrete as discrete


def _inputs(device):
    a = torch.tensor([0., 0., -1., -1., 1., 1., -.3, .4, -10., 20., 1e-8],
                     device=device, dtype=torch.float64)
    b = a.new_tensor([0., 1., 1e-10, -1e-10, 1e-10, -1e-10, .7, -.8, -3., 2., .2])
    old = torch.linspace(-math.pi, math.pi, a.numel(), device=device, dtype=a.dtype)
    return a, b, old


@pytest.mark.parametrize('kappa', [0., .01, .2, 2.])
def test_tensor_control_candidate_equals_eager_minimizer(kappa, device):
    a, b, old = _inputs(device)
    actual = discrete._xy_convex_candidate(a, b, a.new_tensor(kappa), old)
    expected = discrete._xy_minimum(a, b, kappa, old)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not callable(getattr(torch, 'compile', None)), reason='torch.compile unavailable')
def test_fullgraph_compiles_once_across_scalar_control_values():
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(discrete._xy_convex_candidate, backend=backend,
                             fullgraph=True, dynamic=True)
    a, b, old = _inputs('cpu')
    for kappa in (0., .01, .2, 2.):
        actual = compiled(a, b, a.new_tensor(kappa), old)
        expected = discrete._xy_minimum(a, b, kappa, old)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(graphs) == 1


def test_compiler_factory_is_lazy_and_reuses_code_only(monkeypatch):
    calls = []

    def compile_function(function, **options):
        calls.append((function, options))
        return function

    discrete._compiled_xy_coordinates.cache_clear()
    monkeypatch.setattr(torch, 'compile', compile_function)
    try:
        first = discrete._compiled_xy_coordinates()
        assert discrete._compiled_xy_coordinates() is first
        assert calls == [(discrete._xy_convex_candidate, {'fullgraph': True, 'dynamic': True})]
    finally:
        discrete._compiled_xy_coordinates.cache_clear()


def test_optional_compile_flag_keeps_cpu_path_eager(monkeypatch):
    def fail():
        raise AssertionError('CPU execution must not initialize the GPU compiler')

    monkeypatch.setattr(discrete, '_compiled_xy_coordinates', fail)
    problem = QUBO(torch.tensor([[-1., 1.5, -.2], [1.5, -.4, .1], [-.2, .1, .3]],
                                dtype=torch.float64))
    ordinary = discrete.XYCoordinate(max_steps=2).solve(problem, restarts=2, seed=9, history_interval=1)
    requested = discrete.XYCoordinate(max_steps=2, compile_coordinates=True).solve(
        problem, restarts=2, seed=9, history_interval=1)
    torch.testing.assert_close(ordinary.final_assignments, requested.final_assignments, rtol=0, atol=0)
    torch.testing.assert_close(ordinary.best_energies, requested.best_energies, rtol=0, atol=0)
    torch.testing.assert_close(ordinary.energy_history, requested.energy_history, rtol=0, atol=0)


@pytest.mark.skipif(not (torch.cuda.is_available()
    and callable(getattr(torch, 'compile', None))
    and importlib.util.find_spec('triton') is not None), reason='CUDA torch.compile/Triton unavailable')
def test_inductor_gpu_candidate_preserves_conditional_energies_and_ties():
    a, b, old = _inputs('cuda')
    compiled = discrete._compiled_xy_coordinates()
    for kappa in (0., .01, .2, 2.):
        actual = compiled(a, b, a.new_tensor(kappa), old)
        expected = discrete._xy_minimum(a, b, kappa, old)
        actual_energy = -a * actual.cos() - b * actual.sin() + kappa * actual.sin().square()
        expected_energy = -a * expected.cos() - b * expected.sin() + kappa * expected.sin().square()
        torch.testing.assert_close(actual_energy, expected_energy, rtol=5e-12, atol=5e-12)
        assert bool(torch.isfinite(actual).all())
        if kappa == 0:
            assert actual[0] == old[0]


def test_compile_flag_is_explicit_boolean():
    assert discrete.XYCoordinate().compile_coordinates is False
    with pytest.raises(ValueError, match='compile_coordinates must be boolean'):
        discrete.XYCoordinate(compile_coordinates=1)

"""Exact elimination, Gibbs law, admission and explicit device contracts."""

import itertools
import math

import numpy as np
import pytest
import torch

from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.backends.tree_decomposition import (
    MAX_TREEWIDTH, TreeDecompositionBQMSolver, TreeDecompositionSamplerBQMSolver,
    TreewidthExceeded, elimination_preflight,
)


def fixture():
    # Deliberately include diagonal, reversed and duplicate pair terms; two
    # variables are disconnected from the 0--1--2 component.
    return QUBOProblem(
        np.array([-.5, 1., -1.25, .75, 0.]),
        np.array([0, 1, 0, 1, 2, 3]), np.array([1, 0, 1, 2, 2, 3]),
        np.array([1., -.5, .75, -2., .5, -.25]), 7.125, seedOffset=19,
    )


def enumeration(problem):
    assignments = list(itertools.product((0, 1), repeat=problem.variableCount))
    return assignments, np.array([problem.energy(sample) for sample in assignments])


@pytest.mark.parametrize('order', [None, [4, 3, 2, 1, 0], [1, 4, 0, 3, 2]])
def test_minimum_matches_exhaustive_source_with_duplicates_and_offsets(device, dtype, order):
    problem = fixture()
    _, energies = enumeration(problem)
    solver = TreeDecompositionBQMSolver(device=str(device))
    parameters = dict(runs=3, dtype=str(dtype).split('.')[-1], elimination_order=order)
    result = solver.sample_all(problem, parameters)
    assert result.assignments.device.type == device.type
    if device.index is not None:
        assert result.assignments.device.index == device.index
    assert result.assignments.dtype == torch.int8
    assert result.assignments.shape == (3, 5)
    assert result.energies.dtype == torch.float64
    for assignment, energy in zip(result.assignments.cpu().tolist(), result.energies.cpu().tolist()):
        assert energy == problem.energy(assignment) == energies.min()
    assert result.minimum_energy.item() == energies.min()
    application = solver.solve(problem, parameters)
    assert application.energy == problem.energy(application.sample) == energies.min()


@pytest.mark.parametrize('sampling', [False, True])
def test_no_interactions_and_disconnected_variables(device, dtype, sampling):
    problem = QUBOProblem(np.array([-2., 0., 3.]), [], [], [], offset=-5.5, seedOffset=0)
    solver = (TreeDecompositionSamplerBQMSolver if sampling else TreeDecompositionBQMSolver)(str(device))
    result = solver.sample_all(problem, dict(runs=7, dtype=str(dtype).split('.')[-1], beta=0.))
    assert result.plan.width == 0
    assert result.assignments.shape == (7, 3)
    for sample, energy in zip(result.assignments.cpu().tolist(), result.energies.cpu().tolist()):
        assert energy == problem.energy(sample)
    if sampling:
        assert result.log_partition.item() == pytest.approx(3 * math.log(2), rel=1e-6)
    else:
        assert result.assignments[0].cpu().tolist() == [1, 0, 0]
        assert result.minimum_energy.item() == -7.5


@pytest.mark.parametrize('beta', [0., .7, 3.])
def test_partition_and_empirical_boltzmann_distribution(device, beta):
    problem = QUBOProblem(np.array([-.5, .25, .75]), [0, 1], [1, 2], [1., -.5], 2.5, seedOffset=13)
    states, energies = enumeration(problem)
    log_weights = -beta * energies
    log_z = np.logaddexp.reduce(log_weights)
    probabilities = np.exp(log_weights - log_z)
    samples = TreeDecompositionSamplerBQMSolver(str(device)).sample_all(
        problem, dict(runs=40_000, seed=731, beta=beta, dtype='float64'))
    assert samples.log_partition.item() == pytest.approx(log_z, rel=1e-12, abs=1e-12)
    bitstrings = samples.assignments.cpu().numpy()
    codes = bitstrings @ np.array([4, 2, 1])
    observed = np.bincount(codes, minlength=8) / len(codes)
    # Six standard deviations plus a small finite-sample guard; fixed local seed.
    tolerance = 6 * np.sqrt(probabilities * (1 - probabilities) / len(codes)) + .001
    assert np.all(np.abs(observed - probabilities) < tolerance), (observed, probabilities)
    for index, state in enumerate(states):
        selected = samples.energies.cpu().numpy()[codes == index]
        assert np.all(selected == problem.energy(state))


def test_seed_repeatability_and_global_rng_isolation(device):
    problem = fixture()
    solver = TreeDecompositionSamplerBQMSolver(str(device))
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    first = solver.sample_all(problem, dict(runs=31, seed=53, beta=.7))
    second = solver.sample_all(problem, dict(runs=31, seed=53, beta=.7))
    torch.testing.assert_close(first.assignments, second.assignments, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(device), cuda_rng)


def test_dense_small_random_orders_match_exhaustive_min_and_partition(device, dtype):
    generator = np.random.default_rng(71)
    pairs = list(itertools.combinations(range(6), 2))
    for _ in range(5):
        problem = QUBOProblem(generator.integers(-8, 8, 6) / 4,
                              [i for i, _ in pairs], [j for _, j in pairs],
                              generator.integers(-4, 4, len(pairs)) / 4, 4.25)
        _, energies = enumeration(problem)
        parameters = dict(elimination_order=generator.permutation(6).tolist(),
                          dtype=str(dtype).split('.')[-1], beta=.7, runs=13)
        minimum = TreeDecompositionBQMSolver(str(device)).sample_all(problem, parameters)
        sampled = TreeDecompositionSamplerBQMSolver(str(device)).sample_all(problem, parameters)
        assert minimum.minimum_energy.item() == energies.min()
        assert minimum.energies.min().item() == energies.min()
        tolerance = 3e-6 if dtype == torch.float32 else 1e-12
        assert sampled.log_partition.item() == pytest.approx(
            np.logaddexp.reduce(-.7 * energies), rel=tolerance, abs=tolerance)


def test_width_and_memory_rejection_happen_before_torch_allocation(monkeypatch):
    edges = list(itertools.combinations(range(4), 2))
    problem = QUBOProblem(np.zeros(4), [i for i, _ in edges], [j for _, j in edges], np.ones(len(edges)))

    def forbidden(*args, **kwargs):
        raise AssertionError('Width/memory rejection must precede Torch allocation')

    monkeypatch.setattr(torch, 'tensor', forbidden)
    monkeypatch.setattr(torch, 'zeros', forbidden)
    with pytest.raises(TreewidthExceeded, match='minimum degree 3'):
        TreeDecompositionBQMSolver('cuda:0').sample_all(problem, dict(max_treewidth=2))
    with pytest.raises(MemoryError, match='Estimated tree'):
        TreeDecompositionBQMSolver('cuda:0').sample_all(problem, dict(memory_limit_bytes=1))
    assert MAX_TREEWIDTH == 25


def test_user_order_controls_width_and_invalid_parameters_reject():
    star = QUBOProblem(np.zeros(5), [0, 0, 0, 0], [1, 2, 3, 4], np.ones(4))
    assert elimination_preflight(star, dict(max_treewidth=1)).width == 1
    with pytest.raises(TreewidthExceeded, match='width 4'):
        elimination_preflight(star, dict(max_treewidth=1, elimination_order=[0, 1, 2, 3, 4]))
    invalid = [dict(elimination_order=[0, 0, 2, 3, 4]), dict(elimination_order=[0, 1]),
               dict(elimination_order=[False, 1, 2, 3, 4]), dict(max_treewidth=26),
               dict(runs=0), dict(runs=True), dict(beta=-1), dict(beta=float('inf')),
               dict(memory_limit_bytes=None), dict(dtype='float16'), dict(unknown=1),
               dict(num_reads=2, runs=2), dict(seed=2**63)]
    for parameters in invalid:
        with pytest.raises((ValueError, TypeError)):
            elimination_preflight(star, parameters)


def test_cancelling_pair_coefficients_do_not_create_edges():
    problem = QUBOProblem(np.array([-1., 2.]), [0, 1], [1, 0], [3., -3.], 7.)
    plan = elimination_preflight(problem, dict(max_treewidth=0))
    assert plan.width == 0
    assert TreeDecompositionBQMSolver().solve(problem).energy == 6.


def test_cuda_unavailable_is_explicit_and_preflight_is_hardware_independent(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    problem = fixture()
    solver = TreeDecompositionBQMSolver('cuda:0')
    assert solver.preflight(problem).width == 1
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        solver.sample_all(problem)


def test_sampler_application_result_uses_source_candidate_selection():
    problem = fixture()
    solver = TreeDecompositionSamplerBQMSolver()
    parameters = dict(runs=17, seed=2, beta=.5)
    all_samples = solver.sample_all(problem, parameters)
    selected = solver.solve(problem, parameters)
    assert selected.energy == min(problem.energy(x) for x in all_samples.assignments.tolist())
    assert selected.energy == problem.energy(selected.sample)

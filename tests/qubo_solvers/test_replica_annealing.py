"""Matched reference checks for replica candidate selection."""

import sys

import pytest
import torch

from qubo_solvers import QUBO, ReplicaAnnealing
from qubo_solvers._dynamics import SpinObjective, finite, normal, publish, spins


class _LegacyReplicaAnnealing(ReplicaAnnealing):
    """Original per-layer scoring for a matched seeded regression reference."""

    def _search(self, run, record):
        objective = SpinObjective(run)
        q = .1 * spins(run)[:, None, :].repeat(1, self.replicas, 1)
        q += .01 * normal(run, q.shape)
        for step in range(self.max_steps):
            progress = (step + 1) / max(self.max_steps, 1)
            gradient = (objective.gradient(q) + self.penalty * q * (q*q-1)) / self.replicas
            gradient -= self.coupling * progress * (q.roll(1, 1) + q.roll(-1, 1))
            q -= self.time_step * gradient
            candidates = (q >= 0).to(q.dtype).mul_(2).sub_(1)
            energies = objective.energy(candidates)
            for layer in range(self.replicas):
                publish(run, candidates[:, layer])
            publish(run, candidates[run.rows, energies.argmin(1)])
            run.iterations += 1
            record(step + 1)
        finite(q)
        return self.max_steps


def _assert_equal_results(actual, expected):
    for name in ('final_assignments', 'final_energies', 'best_assignments', 'best_energies',
                 'iterations', 'termination_reasons', 'energy_history', 'history_iterations',
                 'restart_indices'):
        left, right = getattr(actual, name), getattr(expected, name)
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('ising', [False, True])
@pytest.mark.parametrize('replicas', [3, 8])
@pytest.mark.parametrize('best_only', [False, True])
def test_replica_winner_publication_matches_legacy(device, dtype, ising, replicas, best_only):
    raw = torch.randn(7, 7, dtype=dtype, generator=torch.Generator().manual_seed(81))
    problem = QUBO(raw.to(device), 11.5)
    if ising:
        problem = problem.to_ising()
    settings = dict(max_steps=37, replicas=replicas, time_step=.2, coupling=.03, penalty=.2)
    options = dict(restarts=7, batch_size=3, seed=19, best_only=best_only)
    if not best_only:
        options['history_interval'] = 4
    actual = ReplicaAnnealing(**settings).solve(problem, **options)
    expected = _LegacyReplicaAnnealing(**settings).solve(problem, **options)
    _assert_equal_results(actual, expected)
    torch.testing.assert_close(problem.energy(actual.best_assignment), actual.best_energy)


@pytest.mark.parametrize('ising', [False, True])
def test_replica_ties_keep_first_final_layer_and_initial_incumbent(monkeypatch, device, dtype, ising):
    import qubo_solvers.replica_annealing as module

    # Distinct layers of a flat objective tie exactly. Large deterministic
    # perturbations ensure the first layer differs from the initial assignment.
    def chosen_noise(run, shape):
        return run.q.new_tensor([[[20., -20.], [-20., 20.], [20., 20.]]]).expand(shape)

    monkeypatch.setattr(module, 'normal', chosen_noise)
    monkeypatch.setattr(sys.modules[__name__], 'normal', chosen_noise)
    problem = QUBO(torch.zeros(2, 2, device=device, dtype=dtype), 7.)
    initial = torch.zeros(1, 2, device=device, dtype=dtype)
    expected_final = torch.tensor([[1, 0]], device=device, dtype=torch.int8)
    if ising:
        problem = problem.to_ising()
        initial = 2 * initial - 1
        expected_final = 2 * expected_final - 1
    options = dict(initial_assignments=initial, seed=7, history_interval=1)
    actual = ReplicaAnnealing(max_steps=1, replicas=3, coupling=0.).solve(problem, **options)
    expected = _LegacyReplicaAnnealing(max_steps=1, replicas=3, coupling=0.).solve(problem, **options)
    _assert_equal_results(actual, expected)
    torch.testing.assert_close(actual.final_assignments, expected_final)
    torch.testing.assert_close(actual.best_assignments, initial.to(torch.int8))


def test_replica_original_qubo_best_survives_transformed_energy_tie(monkeypatch, device):
    import qubo_solvers.replica_annealing as module

    def chosen_noise(run, shape):
        return run.q.new_tensor([[[-30., -10., -10.], [-30., 30., -10.],
                                  [10., 30., 30.]]]).expand(shape)

    monkeypatch.setattr(module, 'normal', chosen_noise)
    monkeypatch.setattr(sys.modules[__name__], 'normal', chosen_noise)
    problem = QUBO(torch.diag(torch.tensor([1e8, -.001, -1e8], device=device)), 11.5)
    options = dict(initial_assignments=torch.tensor([[1., 0., 0.]], device=device),
                   seed=7, history_interval=1)
    settings = dict(max_steps=1, replicas=3, coupling=0., time_step=.001)
    actual = ReplicaAnnealing(**settings).solve(problem, **options)
    expected = _LegacyReplicaAnnealing(**settings).solve(problem, **options)
    _assert_equal_results(actual, expected)
    # The converted Ising scores tie through cancellation, but the original
    # objective distinguishes the middle layer from the first/final layer.
    torch.testing.assert_close(actual.best_assignments,
                               torch.tensor([[0, 1, 0]], device=device, dtype=torch.int8))


@pytest.mark.parametrize('ising', [False, True])
def test_replica_publications_are_bounded_independently_of_layers(monkeypatch, ising):
    import qubo_solvers.replica_annealing as module

    published = []
    original = module.publish

    def counted(run, values):
        published.append(values.shape)
        return original(run, values)

    monkeypatch.setattr(module, 'publish', counted)
    problem = QUBO(torch.tensor([[-1., .5], [.5, -2.]]))
    if ising:
        problem = problem.to_ising()
    ReplicaAnnealing(max_steps=5, replicas=8).solve(problem, restarts=3, batch_size=2, seed=9)
    per_batch = 5 if ising else 10
    assert published == ([torch.Size([2, 2])] * per_batch
                         + [torch.Size([1, 2])] * per_batch)

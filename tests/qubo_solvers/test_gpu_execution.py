"""Device-resident selection and solve-local allocation reuse contracts."""

import pytest
import torch

from qubo_solvers import (
    QUBO, SimulatedAnnealing, GreedyLocalSearch, SpinVectorLangevin,
    AngularAnnealing, TransverseRoute, EasyAxisAnnealing, SpinCoherentAnnealing,
    VectorAmplitudeAnnealing, MeanFieldAnnealing, TAPAnnealing,
    SphericalAnnealing, ContactAnnealing, ReplicaAnnealing, HeatBathAnnealing,
    TabuSearch, RandomSearch, ExchangeCascade,
)
from qubo_solvers._dynamics import SpinObjective
from qubo_solvers.exchange_cascade import _ExchangeWorkspace
from qubo_solvers.solvers import _Run, _Output


SOLVERS = (
    SimulatedAnnealing(3, 2., .1), GreedyLocalSearch(3), SpinVectorLangevin(3),
    AngularAnnealing(3), TransverseRoute(3), EasyAxisAnnealing(3),
    SpinCoherentAnnealing(3), VectorAmplitudeAnnealing(3), MeanFieldAnnealing(3),
    TAPAnnealing(3), SphericalAnnealing(3), ContactAnnealing(3),
    ReplicaAnnealing(3), HeatBathAnnealing(3, 2., .1), TabuSearch(3),
    RandomSearch(3), ExchangeCascade(3),
)


@pytest.mark.parametrize('solver', SOLVERS, ids=lambda solver: type(solver).__name__)
def test_every_native_solver_batched_best_matches_full_on_device(solver, device, dtype):
    problem = QUBO(torch.tensor([[-1., .25, -.5], [.25, -2., .75], [-.5, .75, -.125]],
                                device=device, dtype=dtype), 9.)
    options = dict(restarts=7, batch_size=3, seed=311)
    full = solver.solve(problem, **options)
    best = solver.solve(problem, best_only=True, **options)
    torch.testing.assert_close(best.best_assignment, full.best_assignment, rtol=0, atol=0)
    torch.testing.assert_close(best.best_energy, full.best_energy, rtol=0, atol=0)
    torch.testing.assert_close(best.best_restart_index, full.best_restart_index)
    torch.testing.assert_close(problem.energy(best.best_assignment), best.best_energy)
    assert best.best_assignment.device == problem.Q.device
    assert best.best_assignment.dtype == torch.int8
    assert best.best_assignment.shape == (3,)
    # Best-only output owns just one row and never retains a batch-sized view.
    assert best.best_assignments.untyped_storage().nbytes() == 3


def scalar_extractions(profile):
    return sum(event.count for event in profile.key_averages()
               if event.key == 'aten::_local_scalar_dense')


def test_best_only_selection_has_no_host_scalar_extraction(device, dtype):
    problem = QUBO(torch.eye(4, device=device, dtype=dtype))
    run = _Run(problem, 3, None, torch.Generator(device=device).manual_seed(1),
               torch.Generator().manual_seed(1))
    output = _Output(problem, 3, True, None, 2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        output.finish_batch(run, 0, 2)
        result = output.result()
        assignment, restart = result.best_assignment, result.best_restart_index
    assert scalar_extractions(profile) == 0
    assert assignment.shape == (4,)
    assert restart.shape == ()


def test_exchange_setup_scale_stays_on_device_without_scalar_extraction(device, dtype):
    degree = torch.tensor([1., .7, .25], dtype=dtype, device=device)
    expected = 4. * max(float(degree.sum().item()) / 2, 1.)
    vectors = torch.zeros((3, 2, 3), dtype=dtype, device=device)
    matrix = torch.zeros((3, 3), dtype=dtype, device=device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        workspace = _ExchangeWorkspace(torch, matrix, degree, vectors,
                                       dict(order_seed=.05, order_strength=4.))
    assert scalar_extractions(profile) == 0
    assert workspace.orderScale.device == degree.device
    assert workspace.orderScale.item() == expected
    # A float64 scalar preserves float32 state arithmetic's scalar promotion.
    assert (workspace.orderScale * vectors).dtype == dtype


def test_spin_objective_reuses_immutable_tensors_only_within_one_solve(monkeypatch):
    original = SpinObjective.__init__
    observed = []

    def record(self, run, normalize=True):
        original(self, run, normalize)
        observed.append((self.j, self.h, self.scale))

    monkeypatch.setattr(SpinObjective, '__init__', record)
    problem = QUBO(torch.tensor([[-2., .5], [.5, -1.]]))
    saved = problem.Q.clone()
    solver = MeanFieldAnnealing(2)
    solver.solve(problem, restarts=7, batch_size=3, seed=31)
    first = observed[:]
    assert len(first) == 3
    for index in range(3):
        assert all(batch[index] is first[0][index] for batch in first)
    observed.clear()
    solver.solve(problem, restarts=7, batch_size=3, seed=31)
    assert observed[0][0] is not first[0][0]
    torch.testing.assert_close(problem.Q, saved)


def test_spin_objective_cache_keeps_normalized_and_unscaled_variants_distinct():
    problem = QUBO(torch.tensor([[10., 4.], [4., 6.]]))
    shared = {}
    first = _Run(problem, 1, None, torch.Generator(), torch.Generator(), shared)
    second = _Run(problem, 1, None, torch.Generator(), torch.Generator(), shared)
    normalized, raw = SpinObjective(first), SpinObjective(first, normalize=False)
    assert normalized.scale > 1
    assert raw.scale == 1
    assert SpinObjective(second).j is normalized.j
    assert SpinObjective(second, normalize=False).j is raw.j

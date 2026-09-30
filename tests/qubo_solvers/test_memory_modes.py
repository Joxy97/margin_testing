"""Behavioral checks for bounded batches, compact output, and native objectives."""
import pytest
import torch

from qubo_solvers import QUBO, Ising, GreedyLocalSearch, SimulatedAnnealing, estimate_memory


@pytest.mark.parametrize('solver', [GreedyLocalSearch(9), SimulatedAnnealing(5, 2., .1)])
@pytest.mark.parametrize('spin', [False, True])
def test_batched_best_matches_full(solver, spin, device, dtype):
    q = QUBO(torch.tensor([[-2., .5, 1.], [.5, -1., -.25], [1., -.25, -.5]],
                         device=device, dtype=dtype))
    problem = q.to_ising() if spin else q
    full = solver.solve(problem, restarts=11, batch_size=4, seed=17, history_interval=2)
    best = solver.solve(problem, restarts=11, batch_size=4, seed=17, best_only=True)
    again = solver.solve(problem, restarts=11, batch_size=4, seed=17, best_only=True)
    assert best.final_assignments is None and best.final_energies is None
    assert best.energy_history is None and best.history_iterations is None
    assert best.best_assignments.dtype == torch.int8
    assert best.best_assignments.shape == (1, 3)
    assert best.best_assignments.untyped_storage().nbytes() == 3
    torch.testing.assert_close(best.best_assignment, full.best_assignment)
    torch.testing.assert_close(best.best_energy, full.best_energy)
    assert best.best_restart_index == full.best_restart_index
    index = full.best_restart_index
    assert best.iterations[0] == full.iterations[index]
    assert best.termination_reasons[0] == full.termination_reasons[index]
    assert torch.equal(best.best_assignment, again.best_assignment)
    torch.testing.assert_close(full.energy_history[-1], full.final_energies)
    torch.testing.assert_close(problem.energy(full.best_assignments), full.best_energies)
    assert best.to(device='cpu').restart_indices.device.type == 'cpu'


def test_batched_greedy_history_and_early_stop(device, dtype):
    q = QUBO(-torch.eye(5, device=device, dtype=dtype))
    initial = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]], device=device, dtype=torch.int8)
    solver = GreedyLocalSearch(20)
    full = solver.solve(q, restarts=4, initial_assignments=initial, history_interval=2)
    batched = solver.solve(q, restarts=4, batch_size=2, initial_assignments=initial, history_interval=2)
    assert batched.history_iterations.tolist() == [0, 2, 4, 5]
    assert batched.iterations.tolist() == [0, 1, 3, 5]
    torch.testing.assert_close(batched.energy_history, full.energy_history)
    torch.testing.assert_close(batched.final_assignments, full.final_assignments)
    # Returned energy agrees with the single best assignment after every round.
    assert batched.energy_history[:, 0].tolist() == [-5., -5., -5., -5.]


@pytest.mark.parametrize('solver', [GreedyLocalSearch(0), SimulatedAnnealing(0, 1., 1.)])
def test_best_only_ties_choose_first_restart(solver, device, dtype):
    q = QUBO(torch.zeros(3, 3, device=device, dtype=dtype))
    result = solver.solve(q, restarts=7, batch_size=2, seed=4, best_only=True)
    assert result.best_restart_index.item() == 0
    assert result.iterations.tolist() == [0]


def test_estimate_and_explicit_limit():
    q = QUBO(torch.eye(100))
    full = estimate_memory(q, restarts=1000)
    batched = estimate_memory(q, restarts=1000, batch_size=10)
    best = estimate_memory(q, restarts=1000, batch_size=10, best_only=True)
    assert full.problem_bytes == batched.problem_bytes == best.problem_bytes == 40004
    assert full.workspace_bytes > batched.workspace_bytes == best.workspace_bytes
    assert full.output_bytes == batched.output_bytes > best.output_bytes
    assert estimate_memory(q, restarts=100000, batch_size=10, best_only=True) == best
    with pytest.raises(MemoryError, match='estimated tensor storage'):
        GreedyLocalSearch(1).solve(q, memory_limit_bytes=1)
    for kwargs in ({'batch_size': 0}, {'best_only': True, 'history_interval': 1},
                   {'memory_limit_bytes': 0}):
        with pytest.raises(ValueError):
            GreedyLocalSearch(1).solve(q, **kwargs)


def test_native_ising_does_not_convert(monkeypatch):
    problem = Ising(torch.tensor([[0., .5], [.5, 0.]]), torch.tensor([1., -2.]))
    def forbidden(*args):
        raise AssertionError('Ising solve must not allocate a QUBO conversion')
    monkeypatch.setattr(Ising, 'to_qubo', forbidden)
    for solver in (GreedyLocalSearch(5), SimulatedAnnealing(3, 2., .1)):
        result = solver.solve(problem, restarts=7, batch_size=3, seed=1)
        torch.testing.assert_close(problem.energy(result.best_assignments), result.best_energies)


@pytest.mark.parametrize('spin', [False, True])
def test_periodic_refresh_local_optimality(spin, device, dtype):
    g = torch.Generator().manual_seed(271)
    q = QUBO(torch.randn(12, 12, generator=g, dtype=dtype).to(device))
    p = q.to_ising() if spin else q
    result = GreedyLocalSearch(100, refresh_interval=7).solve(p, restarts=20, seed=1, batch_size=6)
    for column in range(12):
        neighbor = result.final_assignments.clone()
        neighbor[:, column] = -neighbor[:, column] if spin else 1 - neighbor[:, column]
        assert (p.energy(neighbor) >= result.final_energies - 1e-4).all()
    torch.testing.assert_close(p.energy(result.best_assignments), result.best_energies)


def test_late_batch_validation_preserves_inputs():
    q = QUBO(torch.eye(2))
    initial = torch.zeros(7, 2)
    initial[-1, -1] = 2
    saved = initial.clone()
    with pytest.raises(ValueError, match='outside the problem domain'):
        GreedyLocalSearch(2).solve(q, restarts=7, batch_size=2, initial_assignments=initial)
    assert torch.equal(saved, initial)


def test_allocation_failure_is_not_retried(monkeypatch):
    import qubo_solvers.solvers as implementation
    count = 0
    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        raise RuntimeError("DefaultCPUAllocator: can't allocate memory")
    monkeypatch.setattr(implementation, '_Output', fail)
    with pytest.raises(MemoryError, match='no automatic retry'):
        GreedyLocalSearch(2).solve(QUBO(torch.eye(2)))
    assert count == 1


def test_non_memory_runtime_errors_are_preserved(monkeypatch):
    import qubo_solvers.solvers as implementation
    def fail(*args, **kwargs):
        raise RuntimeError('unexpected tensor error')
    monkeypatch.setattr(implementation, '_Output', fail)
    with pytest.raises(RuntimeError, match='unexpected tensor error'):
        GreedyLocalSearch(2).solve(QUBO(torch.eye(2)))

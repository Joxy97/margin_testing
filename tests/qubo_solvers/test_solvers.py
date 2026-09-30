from itertools import product

import pytest
import torch

from qubo_solvers import GreedyLocalSearch, Ising, QUBO, SimulatedAnnealing, TerminationReason


@pytest.fixture(params=[GreedyLocalSearch(20), SimulatedAnnealing(8, 3., .1)])
def solver(request):
    return request.param


@pytest.mark.parametrize("spin", [False, True])
def test_solver_contract(solver, device, dtype, spin):
    q = QUBO(torch.tensor([[-2, 1, -.5], [1, 3, -1], [-.5, -1, 1]], device=device, dtype=dtype), 7.)
    problem = q.to_ising() if spin else q
    initial = torch.tensor(list(product([0, 1], repeat=3)), device=device, dtype=dtype)
    if spin:
        initial = 2 * initial - 1
    original = initial.clone()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    result = solver.solve(problem, restarts=8, initial_assignments=initial, seed=21, history_interval=3)
    again = solver.solve(problem, restarts=8, initial_assignments=initial, seed=21, history_interval=3)
    torch.testing.assert_close(result.final_assignments, again.final_assignments, rtol=0, atol=0)
    torch.testing.assert_close(result.energy_history, again.energy_history, rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert torch.equal(initial, original)
    torch.testing.assert_close(result.final_energies, problem.energy(result.final_assignments))
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments))
    assert (result.best_energies <= result.final_energies + 1e-5).all()
    assert (result.best_energies <= problem.energy(initial) + 1e-5).all()
    torch.testing.assert_close(result.energy_history[0], problem.energy(initial))
    torch.testing.assert_close(result.energy_history[-1], result.final_energies)
    torch.testing.assert_close(problem.energy(result.best_assignment), result.best_energy)
    moved = result.to(device="cpu")
    assert moved.final_assignments.device.type == "cpu"
    assert moved.iterations.dtype == torch.int64
    if isinstance(solver, GreedyLocalSearch):
        assert (result.termination_reasons == TerminationReason.LOCAL_OPTIMUM).all()
        for column in range(3):
            neighbor = result.final_assignments.clone()
            neighbor[:, column] = -neighbor[:, column] if spin else 1 - neighbor[:, column]
            assert (problem.energy(neighbor) >= result.final_energies - 1e-5).all()
    else:
        assert (result.iterations == solver.sweeps).all()


def test_random_initialization_is_seeded(solver, device, dtype):
    problem = QUBO(torch.eye(8, device=device, dtype=dtype))
    a = solver.solve(problem, restarts=9, seed=123, history_interval=1)
    b = solver.solve(problem, restarts=9, seed=123, history_interval=1)
    assert torch.equal(a.final_assignments, b.final_assignments)
    assert torch.equal(a.energy_history, b.energy_history)
    assert a.energy_history.shape[1] == 9


def test_greedy_updates_against_direct_enumeration(device, dtype):
    gen = torch.Generator().manual_seed(5)
    raw = torch.randn(5, 5, generator=gen, dtype=dtype).to(device)
    q = QUBO(raw)
    x = torch.tensor(list(product([0, 1], repeat=5)), device=device, dtype=dtype)
    reference = x.clone()
    for steps in range(1, 6):
        neighbors = reference[:, None, :].expand(-1, 5, -1).clone()
        indices = torch.arange(5, device=device)
        neighbors[:, indices, indices] = 1 - neighbors[:, indices, indices]
        energies = q.energy(neighbors)
        best_e, columns = energies.min(dim=1)
        improve = best_e < q.energy(reference)
        chosen = neighbors[torch.arange(32, device=device), columns]
        reference = torch.where(improve[:, None], chosen, reference)
        actual = GreedyLocalSearch(steps).solve(q, restarts=32, initial_assignments=x)
        torch.testing.assert_close(actual.final_assignments, reference.to(torch.int8))
        torch.testing.assert_close(actual.final_energies, q.energy(reference))


def test_ties_zero_budget_and_flat_problem(device, dtype):
    q = QUBO(-torch.eye(3, device=device, dtype=dtype))
    x = torch.zeros(1, 3, device=device, dtype=dtype)
    result = GreedyLocalSearch(1).solve(q, initial_assignments=x)
    assert result.final_assignments.tolist() == [[1, 0, 0]]
    assert result.iterations.item() == 1
    assert result.termination_reasons.item() == TerminationReason.BUDGET_EXHAUSTED
    for solver in (GreedyLocalSearch(0), SimulatedAnnealing(0, 1., 1.)):
        result = solver.solve(q, initial_assignments=x, history_interval=2)
        assert torch.equal(result.final_assignments, x)
        assert result.iterations.item() == 0
        assert result.history_iterations.tolist() == [0]
    flat = QUBO(torch.zeros(3, 3, device=device, dtype=dtype))
    result = GreedyLocalSearch(10).solve(flat)
    assert result.iterations.item() == 0
    assert result.termination_reasons.item() == TerminationReason.LOCAL_OPTIMUM


def test_annealing_keeps_intermediate_best(device, dtype):
    q = QUBO(-torch.eye(2, device=device, dtype=dtype))
    # Extremely high temperature accepts all flips: visits optimum then leaves it.
    result = SimulatedAnnealing(2, 1e20, 1e20).solve(
        q, initial_assignments=torch.zeros(1, 2, device=device, dtype=dtype), seed=1)
    assert result.best_energy.item() == -2
    assert result.final_energies.item() == 0


def test_invalid_configuration_and_initialization(solver):
    q = QUBO(torch.eye(2))
    for kwargs in ({"restarts": 0}, {"history_interval": 0},
                   {"initial_assignments": torch.ones(2)},
                   {"initial_assignments": torch.ones(1, 2) * 2}):
        with pytest.raises(ValueError):
            solver.solve(q, **kwargs)
    for build in (lambda: GreedyLocalSearch(-1), lambda: GreedyLocalSearch(True),
                  lambda: SimulatedAnnealing(2, .1, 1),
                  lambda: SimulatedAnnealing(2, float('inf'), 1)):
        with pytest.raises(ValueError):
            build()


def test_annealing_updates_against_direct_energy(device, dtype):
    """Independent reference recomputes each proposal's objective from scratch."""
    import math

    q = QUBO(torch.tensor([[-2., .5, 1.], [.5, 1., -.25], [1., -.25, -.5]],
                          device=device, dtype=dtype), 4.)
    initial = torch.tensor(list(product([0, 1], repeat=3)), device=device, dtype=dtype)
    x = initial.clone()
    best_x = x.clone()
    best = q.energy(x)
    draws = torch.Generator(device=device).manual_seed(7)
    order = torch.Generator().manual_seed(7)
    for sweep in range(4):
        temperature = math.exp((1 - sweep / 3) * math.log(2.) + sweep / 3 * math.log(.1))
        for column in torch.randperm(3, generator=order).tolist():
            proposal = x.clone()
            proposal[:, column] = 1 - proposal[:, column]
            delta = q.energy(proposal) - q.energy(x)
            uniform = torch.rand(8, generator=draws, device=device, dtype=dtype)
            accepted = (delta <= 0) | (uniform < torch.exp(-delta.clamp_min(0) / temperature))
            x = torch.where(accepted[:, None], proposal, x)
            energy = q.energy(x)
            best_x = torch.where((energy < best)[:, None], x, best_x)
            best = torch.minimum(best, energy)
    result = SimulatedAnnealing(4, 2., .1).solve(q, restarts=8, initial_assignments=initial, seed=7)
    torch.testing.assert_close(result.final_assignments, x.to(torch.int8))
    torch.testing.assert_close(result.best_assignments, best_x.to(torch.int8))
    torch.testing.assert_close(result.best_energies, best)

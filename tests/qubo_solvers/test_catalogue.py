import itertools

import pytest
import torch

from qubo_solvers import (
    QUBO, Ising, SpinVectorLangevin, AngularAnnealing, TransverseRoute,
    EasyAxisAnnealing, SpinCoherentAnnealing, VectorAmplitudeAnnealing,
    MeanFieldAnnealing, TAPAnnealing, SphericalAnnealing, ContactAnnealing,
    HeatBathAnnealing, TabuSearch, RandomSearch,
    ExchangeCascade, ReplicaAnnealing,
)
from qubo_solvers._dynamics import SpinObjective
from qubo_solvers.solvers import _Run


SOLVERS = [
    SpinVectorLangevin(4), SpinVectorLangevin(4, integrator='euler'),
    AngularAnnealing(4), TransverseRoute(4, gamma=.2),
    EasyAxisAnnealing(4), SpinCoherentAnnealing(4), VectorAmplitudeAnnealing(4),
    MeanFieldAnnealing(4), TAPAnnealing(4), SphericalAnnealing(4),
    ContactAnnealing(4), HeatBathAnnealing(4, 2., .1), TabuSearch(4), RandomSearch(4),
    ExchangeCascade(4), ReplicaAnnealing(4),
]


@pytest.mark.parametrize('solver', SOLVERS, ids=lambda s: repr(s))
@pytest.mark.parametrize('ising', [False, True])
def test_common_contract(solver, ising, device, dtype):
    q = torch.tensor([[-.7, .4], [.4, -.2]], dtype=dtype, device=device)
    problem = QUBO(q, 11.5)
    if ising:
        problem = problem.to_ising()
    initial = torch.tensor([[0, 0], [1, 0], [0, 1]], device=device, dtype=torch.int8)
    if ising:
        initial = 2*initial-1
    saved = initial.clone()
    global_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    options = dict(restarts=3, initial_assignments=initial, seed=47, batch_size=2)
    result = solver.solve(problem, history_interval=1, **options)
    repeated = solver.solve(problem, history_interval=1, **options)
    torch.testing.assert_close(result.final_assignments, repeated.final_assignments)
    torch.testing.assert_close(result.best_assignments, repeated.best_assignments)
    torch.testing.assert_close(initial, saved)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)
    if cuda_state is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(device), cuda_state)
    torch.testing.assert_close(q, torch.tensor([[-.7, .4], [.4, -.2]], dtype=dtype, device=device))
    for assignments, energies in ((result.final_assignments, result.final_energies),
                                   (result.best_assignments, result.best_energies)):
        assert assignments.dtype == torch.int8
        assert assignments.device == q.device
        assert energies.dtype == dtype
        torch.testing.assert_close(problem.energy(assignments), energies)
    assert (result.best_energies <= problem.energy(initial)+1e-5).all()
    assert (result.best_energies <= result.final_energies+1e-5).all()
    torch.testing.assert_close(result.energy_history[0], problem.energy(initial))
    torch.testing.assert_close(result.energy_history[-1], result.final_energies)
    best = solver.solve(problem, best_only=True, **options)
    torch.testing.assert_close(best.best_energy, result.best_energy)
    torch.testing.assert_close(best.best_assignment, result.best_assignment)
    assert best.best_restart_index == result.best_restart_index
    assert best.final_assignments is None
    assert best.to(device='cpu').best_assignment.device.type == 'cpu'


@pytest.mark.parametrize('solver', SOLVERS)
def test_budget_memory_and_validation(solver):
    problem = QUBO(torch.eye(2))
    with pytest.raises(MemoryError):
        solver.solve(problem, memory_limit_bytes=1)
    with pytest.raises(ValueError):
        solver.solve(problem, restarts=0)
    with pytest.raises(ValueError):
        solver.solve(problem, initial_assignments=torch.zeros(1, 3))
    with pytest.raises(ValueError):
        solver.solve(problem, best_only=True, history_interval=1)




def test_spin_objective_conversion_gradient():
    problem = QUBO(torch.tensor([[1., -2., .3], [-2., .4, .8], [.3, .8, -.5]], dtype=torch.float64))
    initial = torch.tensor(list(itertools.product([0., 1.], repeat=3)), dtype=torch.float64)
    run = _Run(problem, 8, initial, torch.Generator(), torch.Generator())
    obj = SpinObjective(run, normalize=False)
    s = 2*initial-1
    difference = problem.energy(initial)-obj.energy(s)
    torch.testing.assert_close(difference, difference[0].expand_as(difference))
    for i in range(3):
        flipped = s.clone()
        flipped[:, i] *= -1
        torch.testing.assert_close(obj.energy(flipped)-obj.energy(s), -2*s[:, i]*obj.gradient(s)[:, i])


def test_angular_force_matches_autograd():
    problem = Ising(torch.tensor([[0., .2], [.2, 0.]], dtype=torch.float64), torch.tensor([.3, -.4], dtype=torch.float64))
    run = _Run(problem, 1, torch.ones(1, 2, dtype=torch.float64), torch.Generator(), torch.Generator())
    obj = SpinObjective(run)
    theta = torch.tensor([[.3, 1.1]], dtype=torch.float64, requires_grad=True)
    solver = AngularAnnealing(feature_strength=.7, locking=1.3, quartic_penalty=.8)
    x, y = theta.cos(), theta.sin()
    r, t = x*y, .6
    energy = (t*obj.energy(x)/obj.scale + .7*((r @ obj.j)*r).sum()/obj.scale
              + .5*1.3*t*(y*y).sum() + .25*.8*(y**4).sum() - (1-t)*y.sum())
    gradient, = torch.autograd.grad(energy.sum(), theta)
    torch.testing.assert_close(solver._rhs(theta, t, obj), -gradient)




@pytest.mark.parametrize('solver', [SpinVectorLangevin(0), AngularAnnealing(0),
    EasyAxisAnnealing(0), MeanFieldAnnealing(0), SphericalAnnealing(0), ContactAnnealing(0),
    ReplicaAnnealing(0), ExchangeCascade(0)])
def test_zero_budget_preserves_start(solver, dtype):
    problem = QUBO(torch.eye(2, dtype=dtype), 4.)
    initial = torch.tensor([[0, 1], [1, 0]])
    result = solver.solve(problem, restarts=2, initial_assignments=initial, seed=2, history_interval=1)
    torch.testing.assert_close(result.final_assignments, initial.to(torch.int8))
    assert (result.iterations == 0).all()
    torch.testing.assert_close(result.energy_history[0], problem.energy(initial))




@pytest.mark.parametrize('solver', [TabuSearch(128),
    MeanFieldAnnealing(128), EasyAxisAnnealing(128), ExchangeCascade(128)])
def test_longer_random_trajectories(solver, device, dtype):
    generator = torch.Generator().manual_seed(321)
    q = torch.randn(20, 20, dtype=dtype, generator=generator).to(device)
    problem = QUBO(q, 19.)
    result = solver.solve(problem, restarts=7, batch_size=3, seed=13, history_interval=7)
    torch.testing.assert_close(result.final_energies, problem.energy(result.final_assignments))
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments))
    assert (result.best_energies <= result.final_energies+1e-4).all()


def test_biqmac_symmetric_entries(tmp_path):
    from qubo_benchmark.formats import parseMatrix
    problem = parseMatrix(b'2 3\n1 1 2\n1 2 -3\n2 2 1\n').canonical()
    assert problem.score([1, 1]) == -3
    # Published symmetric orientations represent the same matrix entry.
    mirrored = parseMatrix(b'2 2\n1 2 -3\n2 1 -3\n').canonical()
    assert mirrored.score([1, 1]) == -6
    with pytest.raises(ValueError, match='Duplicate'):
        parseMatrix(b'2 2\n1 2 -3\n1 2 -3\n')


def test_exchange_propagator_against_matrix_exponential(device, dtype):
    from qubo_solvers.exchange_cascade import _fieldPropagators
    a = torch.tensor([[.2, .3, .1], [0., 0., 0.], [1., 0., 0.]], device=device, dtype=dtype)
    b = torch.tensor([[.1, -.2, .4], [0., 0., 0.], [2., 0., 0.]], device=device, dtype=dtype)
    k, xi, dt = 1., 4., .1
    decay, response, _ = _fieldPropagators(torch, a, b, k, xi, dt)
    identity = torch.eye(3, device=device, dtype=dtype).expand(3, 3, 3)
    matrix = k*identity+xi*(a[:, :, None]*a[:, None, :]+b[:, :, None]*b[:, None, :])
    expected = torch.matrix_exp(-dt*matrix)
    torch.testing.assert_close(decay, expected)
    torch.testing.assert_close(response, torch.linalg.solve(matrix, identity-expected))


@pytest.mark.parametrize('solver', [EasyAxisAnnealing(), SpinCoherentAnnealing(), VectorAmplitudeAnnealing()])
def test_vector_forces_match_potential(solver):
    problem = Ising(torch.tensor([[0., .2], [.2, 0.]], dtype=torch.float64), torch.tensor([.3, -.4], dtype=torch.float64))
    run = _Run(problem, 1, torch.ones(1, 2, dtype=torch.float64), torch.Generator(), torch.Generator())
    obj = SpinObjective(run)
    vector = torch.tensor([[[.2, .5, -.7], [.8, -.1, .3]]], dtype=torch.float64, requires_grad=True)
    t = .6
    if isinstance(solver, SpinCoherentAnnealing):
        energy = t*obj.energy(vector[..., 2]).sum()/obj.scale-(1-t)*vector[..., 0].sum()
        energy -= .5*solver.penalty*t*vector[..., 2].square().sum()
    else:
        energy = (torch.einsum('bid,ij,bjd->b', vector, obj.j, vector)
                  +(vector[..., 2]*obj.h).sum(1)).sum()/obj.scale
        if isinstance(solver, EasyAxisAnnealing):
            energy += .5*solver.penalty*t*vector[..., :2].square().sum()
        else:
            energy += .25*(vector.square().sum(-1)-1).square().sum()
            anchor = vector.new_tensor([[[0., 0., 1.]]])
            augmented = torch.cat((anchor, vector), dim=1)
            cross = torch.linalg.cross(augmented[:, :, None, :].expand(-1, -1, 3, -1),
                                       augmented[:, None, :, :].expand(-1, 3, -1, -1))
            energy += .25*solver.penalty*t/2*cross.square().sum()
    gradient, = torch.autograd.grad(energy, vector)
    torch.testing.assert_close(solver._gradient(vector, t, obj), gradient)


def test_solver_specific_estimate_scales_with_replicas():
    problem = QUBO(torch.eye(16))
    small = ReplicaAnnealing(replicas=3).estimate_memory(problem, restarts=100, batch_size=5)
    large = ReplicaAnnealing(replicas=9).estimate_memory(problem, restarts=100, batch_size=5)
    assert large.workspace_bytes > small.workspace_bytes
    assert large.output_bytes == small.output_bytes
    with pytest.raises(MemoryError):
        ReplicaAnnealing(replicas=9).solve(problem, restarts=100, batch_size=5,
                                           memory_limit_bytes=small.total_bytes)


def test_transverse_route_force_matches_source_potential():
    problem = Ising(torch.tensor([[0., .2], [.2, 0.]], dtype=torch.float64), torch.tensor([.3, -.4], dtype=torch.float64))
    run = _Run(problem, 1, torch.ones(1, 2, dtype=torch.float64), torch.Generator(), torch.Generator())
    obj = SpinObjective(run)
    angle = torch.tensor([[.3, 1.1]], dtype=torch.float64, requires_grad=True)
    solver = TransverseRoute(feature_strength=.7, gamma=.4, locking_start=-.5, locking=1.3)
    x, y = angle.cos(), angle.sin()
    route, cubic, t = x*y, y**3, .6
    locking = solver.locking_start+(solver.locking-solver.locking_start)*t
    energy = (obj.energy(x)/obj.scale+.7*((route @ obj.j)*route).sum()/obj.scale
              +.4*((cubic @ obj.j)*cubic).sum()/obj.scale+.5*locking*y.square().sum())
    gradient, = torch.autograd.grad(energy.sum(), angle)
    torch.testing.assert_close(solver._rhs(angle, t, obj), -gradient)

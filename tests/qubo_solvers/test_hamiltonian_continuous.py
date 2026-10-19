"""Hamiltonian identities, geometry and finite-step acceptance for eight models."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers._population import PopulationObjective, PopulationRun
from qubo_solvers.hamiltonian_continuous import (
    HAMILTONIAN_CONTINUOUS_SOLVERS, BoseHubbard, Dicke, HaldaneShastry,
    Inozemtsev, JaynesCummings, JaynesCummingsHubbard, NVector,
    SSHDimerOscillator, _Ring, _Target, _backtrack, _hs_spectra,
    _inozemtsev_spectrum, _normalize, _project, _spin_velocity,
)


CLASSES = tuple(HAMILTONIAN_CONTINUOUS_SOLVERS.values())


def _fixture(device, *, sparse=False, spin=False):
    q = torch.tensor([[-1., .25, 0., -.125], [.25, -2., .375, 0.],
                      [0., .375, .5, -.2], [-.125, 0., -.2, -.75]],
                     device=device, dtype=torch.float64)
    problem = QUBO(q, offset=3.25)
    if spin:
        problem = problem.to_ising()
    objective = PopulationObjective.from_problem(problem, precision=torch.float64,
                                                 normalize=False, threshold=1. if sparse else 0.)
    generator = torch.Generator(device=device).manual_seed(93)
    run = PopulationRun(objective, 2, generator)
    return problem, run, _Target(objective), _Ring(run)


def _difference(function, state, direction, epsilon=1e-6):
    plus = tuple(value+epsilon*delta for value, delta in zip(state, direction))
    minus = tuple(value-epsilon*delta for value, delta in zip(state, direction))
    return (function(plus)-function(minus))/(2*epsilon)


@pytest.mark.parametrize('spin', [False, True])
@pytest.mark.parametrize('sparse', [False, True])
def test_source_extension_endpoints_gradients_and_coordinate_fields(device, spin, sparse):
    problem, run, target, _ = _fixture(device, sparse=sparse, spin=spin)
    corners = torch.tensor(list(itertools.product((-1., 1.), repeat=4)),
                           dtype=torch.float64, device=device)
    assignments = corners if spin else (corners+1)/2
    torch.testing.assert_close(target.energy(corners, include_offset=True), problem.energy(assignments))
    values = run.normal((2, 4))*.25
    direction = run.normal(values.shape)
    derivative = _difference(lambda state: target.energy(state[0]), (values,), (direction,))
    expected = (target.exact_gradient(values)*direction).sum(-1)
    torch.testing.assert_close(derivative, expected, rtol=2e-8, atol=2e-8)
    field = target.exact_gradient(values)
    delta = values.new_tensor([.3, -.7])
    target.update_field(field, 2, delta)
    values[:, 2] += delta
    torch.testing.assert_close(field, target.exact_gradient(values), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('sparse', [False, True])
def test_conditional_readout_is_no_worse_than_multilinear_energy(device, sparse):
    _, run, target, _ = _fixture(device, sparse=sparse)
    soft = run.normal((2, 4)).tanh()
    bits = target.readout(soft, sweeps=1, conditional=True)
    assert ((bits == -1) | (bits == 1)).all()
    assert (target.energy(bits) <= target.energy(soft)+1e-14).all()
    polished = target.readout(bits, sweeps=2)
    assert (target.energy(polished) <= target.energy(bits)+1e-14).all()


def test_ssh_forces_are_negative_full_potential_gradient(device):
    _, run, target, ring = _fixture(device)
    solver = SSHDimerOscillator()
    a, d = run.normal((2, 4))*.2, run.normal((2, 4))*.2
    velocity = (run.normal((2, 4))*.1, run.normal((2, 4))*.1)
    r, eta = -.4, .7
    state = (a, d, *velocity)
    da, dd = run.normal(a.shape), run.normal(d.shape)
    direction = (da, dd, torch.zeros_like(a), torch.zeros_like(d))
    fa = (-target.gradient(a)-solver.kappa*a*(a.square()-r)
          +eta*ring.dimer(d, solver.intra_hopping, solver.inter_hopping))
    fd = (-solver.kappa*d*(d.square()-r)
          +eta*ring.dimer(a, solver.intra_hopping, solver.inter_hopping, True))
    actual = _difference(lambda values: solver._energy(values, target, ring, r, eta), state, direction)
    expected = -(fa*da+fd*dd).sum(-1)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)


def test_bose_projected_wirtinger_derivative_and_terminal_identity(device):
    problem, run, target, ring = _fixture(device)
    solver = BoseHubbard()
    vector = run.normal((2, 4, 2, 2))
    vector /= vector.square().sum((-1, -2), keepdim=True).sqrt()
    direction = run.normal(vector.shape)
    direction -= vector*(vector*direction).sum((-1, -2), keepdim=True)
    eta = .8
    actual = _difference(lambda state: solver._energy(state[0], target, ring, eta),
                         (vector,), (direction,))
    tangent = solver._gradient(vector, target, ring, eta)
    expected = 2*(tangent*direction).sum((-1, -2, -3))
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)
    derivative = _difference(lambda state: solver._energy(state[0], target, ring, eta),
                             (vector,), (-solver.relaxation*tangent,))
    torch.testing.assert_close(derivative, -2*solver.relaxation*tangent.square().sum((-1,-2,-3)),
                               rtol=2e-8, atol=2e-8)
    corners = torch.tensor(list(itertools.product((0., 1.), repeat=4)),
                           dtype=torch.float64, device=device)
    encoded = torch.zeros((16, 4, 2, 2), dtype=torch.float64, device=device)
    encoded[..., 0, 0] = 1-corners
    encoded[..., 1, 0] = corners
    identity = _Ring(SimpleNamespace(x=corners, normal=lambda shape: torch.zeros_like(corners)), False)
    torch.testing.assert_close(solver._energy(encoded, target, identity, 0.)+problem.offset,
                               problem.energy(corners))


@pytest.mark.parametrize('n', [1, 2, 4, 7])
def test_haldane_shastry_fft_spectrum_and_positive_mobility(device, n):
    generator = torch.Generator(device=device).manual_seed(7)
    normal = lambda shape: torch.randn(shape, device=device, dtype=torch.float64, generator=generator)
    ring = _Ring(SimpleNamespace(x=torch.zeros((2, n), device=device), normal=normal), False)
    value = normal((2, n, 3))
    spectrum, laplacian = _hs_spectra(n, value)
    index = torch.arange(n, dtype=torch.float64, device=device)
    distance = (index[:, None]-index[None]).remainder(n)
    kernel = (torch.pi/n)**2/torch.sin(torch.pi*distance/n).square()
    kernel.diagonal().zero_()
    expected = torch.einsum('ij,bjd->bid', kernel, value)
    torch.testing.assert_close(ring.convolve(value, spectrum), expected, rtol=1e-13, atol=1e-13)
    mobility = 1/(1+.7*laplacian)
    degree = kernel.sum(1)
    matrix = torch.eye(n, device=device, dtype=torch.float64)+.7*(torch.diag(degree)-kernel)
    direct = torch.linalg.solve(matrix, value.transpose(0, 1).reshape(n, -1)).reshape(n, 2, 3).transpose(0, 1)
    filtered = ring.convolve(value, mobility)
    torch.testing.assert_close(filtered, direct, rtol=1e-13, atol=1e-13)
    assert (value*filtered).sum() >= 0
    torch.testing.assert_close(ring.convolve(torch.ones_like(value), mobility), torch.ones_like(value))


@pytest.mark.parametrize('interaction_range', [.25, 1., 4.])
def test_inozemtsev_image_sum_matches_long_reference(device, interaction_range):
    n = 7
    like = torch.empty((), device=device, dtype=torch.float64)
    spectrum = _inozemtsev_spectrum(n, interaction_range, like)
    kernel = torch.fft.irfft(spectrum, n=n)
    reference = torch.zeros(n, device=device, dtype=torch.float64)
    r = torch.arange(1, n, device=device, dtype=torch.float64)
    for image in range(-40, 41):
        reference[1:] += torch.sinh(like.new_tensor(interaction_range)).square()/torch.sinh(
            interaction_range*(r+image*n)).square()
    torch.testing.assert_close(kernel, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('cls', [Inozemtsev, HaldaneShastry])
def test_exchange_spin_field_and_dissipation(device, cls):
    _, run, target, ring = _fixture(device)
    solver = cls()
    vector = _normalize(run.normal((2, 4, 3)))
    direction = _project(vector, run.normal(vector.shape))
    spectrum = solver._spectrum(4, vector)
    controls = (.37, .4, .25)
    field = solver._field(vector, target, ring, spectrum, controls)
    actual = _difference(lambda state: solver._energy(state[0], target, ring, spectrum, controls),
                         (vector,), (direction,))
    torch.testing.assert_close(actual, -(field*direction).sum((-1, -2)), rtol=2e-8, atol=2e-8)
    if cls is HaldaneShastry:
        _, laplacian = _hs_spectra(4, vector)
        tangent = _project(vector, field)
        filtered = ring.convolve(tangent, 1/(1+solver.mobility*laplacian))
        velocity = torch.linalg.cross(vector, field)+solver.damping*_project(vector, filtered)
        expected = -solver.damping*(tangent*filtered).sum((-1, -2))
    else:
        velocity = _spin_velocity(vector, field, solver.damping)
        expected = -solver.damping*torch.linalg.cross(vector, field).square().sum((-1, -2))
    derivative = _difference(lambda state: solver._energy(state[0], target, ring, spectrum, controls),
                             (vector,), (velocity,))
    torch.testing.assert_close(derivative, expected, rtol=2e-8, atol=2e-8)
    torch.testing.assert_close((vector*velocity).sum(-1), torch.zeros_like(vector[..., 0]), atol=1e-14, rtol=0)


def test_dicke_joint_dissipation_and_cavity_square(device):
    _, run, target, _ = _fixture(device)
    solver = Dicke()
    spin = _normalize(run.normal((2, 4, 3)))
    cavity = run.normal((2, 2))*.2
    state, controls = (spin, cavity), (.4, .5, .3)
    rhs = solver._rhs(state, target, controls)
    derivative = _difference(lambda values: solver._energy(values, target, controls), state, rhs)
    tau, gamma, kappa = controls
    cg = (2/4)**.5*solver.coupling
    field = torch.zeros_like(spin)
    field[..., 0] = gamma-tau*cg*cavity[:, 0, None]
    field[..., 2] = -target.gradient(spin[..., 2])+2*kappa*spin[..., 2]-tau*solver.atomic_frequency/2
    expected = (-solver.cavity_damping*solver.cavity_frequency*cavity[:, 1].square()
                -solver.damping*torch.linalg.cross(spin, field).square().sum((-1, -2)))
    torch.testing.assert_close(derivative, expected, rtol=2e-8, atol=2e-8)
    qstar = -tau*cg*spin[..., 0].sum(-1)/solver.cavity_frequency
    optimal = torch.stack((qstar, torch.zeros_like(qstar)), -1)
    displacement = solver.cavity_frequency/2*((cavity[:, 0]-qstar).square()+cavity[:, 1].square())
    torch.testing.assert_close(solver._energy(state, target, controls)
        -solver._energy((spin, optimal), target, controls), displacement)


@pytest.mark.parametrize('cls', [JaynesCummings, JaynesCummingsHubbard])
def test_photon_spin_real_imaginary_gradients_and_dissipation(device, cls):
    _, run, target, ring = _fixture(device)
    solver = cls()
    spin = _normalize(run.normal((2, 4, 3)))
    photon = run.normal((2, 4, 2))*.2
    state, controls = (spin, photon), (.4, .3)
    ds = _project(spin, run.normal(spin.shape))
    da = run.normal(photon.shape)
    tau, kappa = controls
    gradient = solver._photon_action(photon, ring)
    gradient[..., 0] += tau*solver.coupling/2*spin[..., 0]
    gradient[..., 1] -= tau*solver.coupling/2*spin[..., 1]
    field = torch.empty_like(spin)
    field[..., 0] = -tau*solver.coupling*photon[..., 0]
    field[..., 1] = tau*solver.coupling*photon[..., 1]
    field[..., 2] = -target.gradient(spin[..., 2])+2*kappa*spin[..., 2]-tau*solver.atomic_frequency/2
    actual = _difference(lambda values: solver._energy(values, target, ring, controls), state, (ds, da))
    expected = -(field*ds).sum((-1, -2))+2*(gradient*da).sum((-1, -2))
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)
    rhs = solver._rhs(state, target, ring, controls)
    derivative = _difference(lambda values: solver._energy(values, target, ring, controls), state, rhs)
    expected = (-2*solver.photon_damping*gradient.square().sum((-1, -2))
                -solver.damping*torch.linalg.cross(spin, field).square().sum((-1, -2)))
    torch.testing.assert_close(derivative, expected, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize('dimension', [2, 3, 8])
def test_nvector_tangent_gradient_and_deterministic_descent(device, dimension):
    _, run, target, ring = _fixture(device)
    solver = NVector(dimension=dimension)
    vector = _normalize(run.normal((2, 4, dimension)))
    tau, kappa = .2, .4
    field = solver._field(vector, target, ring, tau, kappa)
    tangent = _project(vector, field)
    derivative = _difference(lambda values: solver._energy(values[0], target, ring, tau, kappa),
                             (vector,), (tangent,))
    torch.testing.assert_close(derivative, -tangent.square().sum((-1, -2)), rtol=2e-8, atol=2e-8)
    new = _normalize(vector+1e-3*tangent)
    torch.testing.assert_close(new.norm(dim=-1), torch.ones_like(vector[..., 0]), rtol=1e-14, atol=1e-14)
    assert (solver._energy(new, target, ring, tau, kappa) <= solver._energy(vector, target, ring, tau, kappa)).all()


def test_backtracking_retains_entire_failed_replica_and_armijo_decreases(device):
    state = (torch.tensor([[1.], [2.]], dtype=torch.float64, device=device),
             torch.tensor([[3.], [4.]], dtype=torch.float64, device=device))
    energy = lambda values: sum(value.square().sum(-1) for value in values)
    wrong = lambda old, dt: tuple(value*(1+dt[:, None]) for value in old)
    result, failed = _backtrack(state, wrong, energy, .1, tolerance=0)
    assert failed.all()
    for old, actual in zip(state, result):
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
    right = lambda old, dt: tuple(value*(1-dt[:, None]) for value in old)
    result, failed = _backtrack(state, right, energy, 3., tolerance=0, decrease=energy(state))
    assert not failed.any()
    assert (energy(result) < energy(state)).all()


@pytest.mark.parametrize('cls', CLASSES)
def test_small_solve_source_scores_history_reproducibility_and_zero_steps(device, dtype, cls):
    problem = _fixture(device)[0].to(dtype=dtype)
    initial = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], device=device, dtype=dtype)
    solver = cls(max_steps=4, dynamics_dtype=dtype)
    kwargs = dict(restarts=2, initial_assignments=initial, seed=17, history_interval=1)
    result = solver.solve(problem, **kwargs)
    repeated = solver.solve(problem, **kwargs)
    torch.testing.assert_close(result.best_assignments, repeated.best_assignments, rtol=0, atol=0)
    source = problem.to(dtype=torch.float64)
    torch.testing.assert_close(result.best_energies, source.energy(result.best_assignments))
    torch.testing.assert_close(result.final_energies, source.energy(result.final_assignments))
    assert (result.best_energies <= source.energy(initial)).all()
    assert result.energy_history.shape == (5, 2)
    assert torch.isfinite(result.energy_history).all()
    zero = cls(max_steps=0).solve(problem, **kwargs)
    torch.testing.assert_close(zero.final_assignments.to(dtype), initial)
    assert zero.iterations.eq(0).all()


@pytest.mark.parametrize('cls', CLASSES)
def test_terminal_pole_energy_identity(device, cls):
    problem, _, target, _ = _fixture(device)
    corners = torch.tensor(list(itertools.product((0., 1.), repeat=4)), device=device, dtype=torch.float64)
    signs = 2*corners-1
    run = SimpleNamespace(x=signs, normal=lambda shape: torch.zeros(shape, device=device, dtype=torch.float64))
    ring = _Ring(run, False)
    solver = cls()
    spin = torch.zeros((16, 4, 3), device=device, dtype=torch.float64)
    spin[..., 2] = signs
    if cls is SSHDimerOscillator:
        state = (signs, signs, torch.zeros_like(signs), torch.zeros_like(signs))
        energy = solver._energy(state, target, ring, 1., 0.)
    elif cls is BoseHubbard:
        z = torch.zeros((16, 4, 2, 2), device=device, dtype=torch.float64)
        z[..., 0, 0], z[..., 1, 0] = 1-corners, corners
        energy = solver._energy(z, target, ring, 0.)
    elif cls in (Inozemtsev, HaldaneShastry):
        energy = solver._energy(spin, target, ring, solver._spectrum(4, spin), (0., 0., .55))
    elif cls is Dicke:
        energy = solver._energy((spin, torch.zeros((16, 2), device=device, dtype=torch.float64)),
                                target, (0., 0., .55))
    elif cls in (JaynesCummings, JaynesCummingsHubbard):
        energy = solver._energy((spin, torch.zeros((16, 4, 2), device=device, dtype=torch.float64)),
                                target, ring, (0., .5))
    else:
        spin[..., 0], spin[..., 2] = signs, 0.
        energy = solver._energy(spin, target, ring, 0., 2.)
    torch.testing.assert_close(energy+problem.offset, problem.energy(corners), rtol=1e-14, atol=1e-14)


def test_compact_duplicates_selfloops_and_sparse_memory_scale(device):
    np = pytest.importorskip('numpy')
    from qubo_solvers.backends.problem import QUBOProblem
    compact = QUBOProblem(np.array([-1., -2., 0., 0.]), np.array([0, 0, 2]),
                          np.array([1, 1, 2]), np.array([1., 2., -.3]), offset=4.)
    objective = PopulationObjective.from_compact(compact, device=device, precision=torch.float64,
                                                 normalize=False, threshold=1.)
    target = _Target(objective)
    z = torch.tensor([[.2, -.4, .6, .8]], dtype=torch.float64, device=device)
    p = (z+1)/2
    expected = -p[:, 0]-2*p[:, 1]+3*p[:, 0]*p[:, 1]-.3*p[:, 2]+4
    torch.testing.assert_close(target.energy(z, include_offset=True), expected)
    direction = torch.tensor([[.3, -.1, .7, -.3]], dtype=torch.float64, device=device)
    torch.testing.assert_close(_difference(lambda state: target.energy(state[0]), (z,), (direction,)),
                               (target.exact_gradient(z)*direction).sum(-1), rtol=1e-8, atol=1e-8)
    solver = NVector(max_steps=3)
    result = solver.solve_compact(compact, device=device, restarts=2, seed=2)
    assert torch.isfinite(result.best_energies).all()
    # An edgeless compact source must stay linear in n in the admission estimate.
    one = QUBOProblem(np.zeros(100), np.array([], dtype=int), np.array([], dtype=int), np.array([]))
    two = QUBOProblem(np.zeros(200), np.array([], dtype=int), np.array([], dtype=int), np.array([]))
    assert solver.estimate_compact_memory(two).total_bytes < 2.2*solver.estimate_compact_memory(one).total_bytes


@pytest.mark.parametrize('cls', CLASSES)
def test_constant_target_returns_without_steps_and_fills_history(device, cls):
    problem = QUBO(torch.zeros((3, 3), device=device, dtype=torch.float64), offset=-7.25)
    result = cls(max_steps=7).solve(problem, restarts=2, history_interval=3, seed=3)
    assert result.iterations.eq(0).all()
    torch.testing.assert_close(result.energy_history, torch.full((4, 2), -7.25,
                               device=device, dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize('n', [1, 2, 7])
def test_local_ring_neighbours_and_discrete_nvector(device, n):
    normal = lambda shape: torch.randn(shape, dtype=torch.float64, device=device)
    value = normal((3, n))
    ring = _Ring(SimpleNamespace(x=value, normal=normal))
    all_neighbours = ring.neighbours(value)
    for site in range(n):
        torch.testing.assert_close(ring.local_neighbours(value, site), all_neighbours[:, site])
    # S^0 has discrete heat-bath motion, including the zero-temperature case.
    problem = QUBO(-torch.eye(n, dtype=torch.float64, device=device))
    result = NVector(dimension=1, initial_temperature=0, final_temperature=0,
                     max_steps=3).solve(problem, restarts=2, seed=9)
    torch.testing.assert_close(result.best_energies, torch.full((2,), -float(n),
                               dtype=torch.float64, device=device), rtol=0, atol=0)


@pytest.mark.parametrize('cls', [BoseHubbard, Inozemtsev, HaldaneShastry, Dicke,
                                 JaynesCummings, JaynesCummingsHubbard, NVector])
def test_trajectory_preserves_cell_or_spin_norms(device, cls, monkeypatch):
    import qubo_solvers.hamiltonian_continuous as implementation
    problem = _fixture(device)[0]
    count = 0
    original_backtrack = implementation._backtrack

    def checked_backtrack(state, trial, energy, time_step, **kwargs):
        nonlocal count

        def checked_trial(old, dt):
            nonlocal count
            result = trial(old, dt)
            value = result[0]
            length = (value.square().sum((-1, -2)).sqrt() if cls is BoseHubbard
                      else value.norm(dim=-1))
            torch.testing.assert_close(length, torch.ones_like(length), rtol=1e-13, atol=1e-13)
            count += 1
            return result

        return original_backtrack(state, checked_trial, energy, time_step, **kwargs)

    monkeypatch.setattr(implementation, '_backtrack', checked_backtrack)
    kwargs = {'initial_temperature': 0, 'final_temperature': 0} if cls is NVector else {}
    cls(max_steps=3, dynamics_dtype=torch.float64, **kwargs).solve(problem, restarts=2, seed=5)
    assert count >= 3


@pytest.mark.parametrize('cls', CLASSES)
def test_rejects_invalid_common_options(cls):
    for kwargs in ({'max_steps': -1}, {'time_step': 0}, {'dynamics_dtype': 'float16'},
                   {'sparse_threshold': 1.1}, {'polish_sweeps': -1}, {'normalize': 1}):
        with pytest.raises(ValueError):
            cls(**kwargs)


def test_model_specific_invalid_parameters():
    with pytest.raises(ValueError):
        JaynesCummingsHubbard(cavity_frequency=.5, hopping=.25)
    with pytest.raises(ValueError):
        NVector(dimension=0)
    with pytest.raises(ValueError):
        NVector(initial_temperature=0, final_temperature=.1)

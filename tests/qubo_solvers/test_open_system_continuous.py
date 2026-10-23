"""Compendium equations, limits, source scoring and bounded solver smoke tests."""
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers._population import PopulationObjective, PopulationRun
from qubo_solvers.hamiltonian_continuous import _Target
from qubo_solvers.open_system_continuous import (
    OPEN_CONTINUOUS_CLASSES, CavityAtomCooling, CavityOptomechanics,
    DopplerCooling, OptomechanicalCooling, QuadratureSqueezing,
    SteadyStateSqueezing, TomonagaLuttinger, _oscillator, _positive_filter,
    _ring_laplacian, _rotation, _sideband_rates, _thermostat, _wall,
)


CLASSES = tuple(OPEN_CONTINUOUS_CLASSES.values())


def _fixture(device, sparse=False):
    q = torch.tensor([[-1., .3, 0., -.2], [.3, -2., .4, 0.],
                      [0., .4, .5, -.1], [-.2, 0., -.1, -.75]],
                     dtype=torch.float64, device=device)
    problem = QUBO(q, offset=2.125)
    objective = PopulationObjective.from_problem(problem, precision=torch.float64,
                                                  threshold=1. if sparse else 0.)
    generator = torch.Generator(device=device).manual_seed(133)
    run = PopulationRun(objective, 3, generator)
    return problem, run, _Target(objective)


def test_passive_rotation_and_exact_oscillator_invariants(device):
    p = torch.tensor([[.2, -.6, 1.2]], dtype=torch.float64, device=device)
    r = torch.tensor([[-.8, .4, .1]], dtype=torch.float64, device=device)
    pp, rr = _rotation(p, r, .73)
    torch.testing.assert_close(pp.square()+rr.square(), p.square()+r.square())
    p0, r0 = _rotation(pp, rr, -.73)
    torch.testing.assert_close(p0, p)
    torch.testing.assert_close(r0, r)
    assert ((.8*pp).square()+(.6*rr).square() <= p.square()+r.square()).all()
    for frequency in (.1, 1., 7.):
        rr, pp = _oscillator(r, p, frequency, .3)
        torch.testing.assert_close(pp.square()+frequency**2*rr.square(),
                                   p.square()+frequency**2*r.square())


def test_inelastic_walls_only_remove_crossing_momentum(device):
    z = torch.tensor([[-2., -1., .2, 1., 1.01]], device=device)
    p = torch.tensor([[3., 2., -1., -2., -3.]], device=device)
    zz, pp = _wall(z, p)
    torch.testing.assert_close(zz, z.clamp(-1, 1))
    torch.testing.assert_close(pp, torch.tensor([[0., 2., -1., -2., 0.]], device=device))
    bad_z = torch.tensor([[float('inf')]], device=device)
    zz, pp = _wall(bad_z, torch.zeros_like(bad_z))
    assert torch.isnan(zz).all() and torch.isnan(pp).all()


def test_exact_ou_stationary_variance_and_lag(device):
    generator = torch.Generator(device=device).manual_seed(8)
    run = SimpleNamespace(normal=lambda shape: torch.randn(shape, generator=generator,
                          dtype=torch.float64, device=device))
    temperature, decay = .7, .6
    p = math.sqrt(temperature)*run.normal((100000,))
    result = _thermostat(p, decay, math.sqrt(1-decay*decay), temperature, run)
    assert abs(float(result.square().mean())-temperature) < .012
    assert abs(float((result*p).mean()/p.square().mean())-decay) < .012


@pytest.mark.parametrize('n', [1, 2, 5])
def test_positive_polynomial_matches_dense_operator_and_preserves_zero_mode(device, n):
    identity = torch.eye(n, dtype=torch.float64, device=device)
    laplacian = _ring_laplacian(identity)
    length, terms = .7, 4
    omega = 1/(1+4*length*length)
    a = identity+length*length*laplacian
    explicit = torch.zeros_like(identity)
    power = identity
    for _ in range(terms):
        explicit += omega*power
        power = power @ (identity-omega*a)
    actual = _positive_filter(identity, length, terms)
    torch.testing.assert_close(actual, explicit)
    assert (torch.linalg.eigvalsh(actual) > 0).all()
    torch.testing.assert_close(actual.sum(1), torch.full((n,), 1-(1-omega)**terms,
                               dtype=torch.float64, device=device))


def test_sideband_resonance_ratio_and_zero_coupling(device):
    frequency = torch.tensor([.1, .7, 2.], dtype=torch.float64, device=device)
    minus, plus = _sideband_rates(frequency, .8, .3)
    assert (minus > plus).all()
    torch.testing.assert_close(plus/(minus-plus), (.8/(4*frequency)).square())
    minus, plus = _sideband_rates(frequency, .8, 0.)
    assert not minus.any() and not plus.any()


def test_compensated_optomechanical_split_decreases_extended_energy(device):
    _, run, target = _fixture(device)
    solver = CavityOptomechanics(temperature=0., time_step=.5)
    z = run.normal((3, 4)).tanh()*.6
    _, cache = solver._prepare(run, z)
    state = (z, run.normal(z.shape)*.2, run.normal(z.shape)*.1)
    anneal = .6

    def energy(values):
        zz, q, p = values
        return (target.energy(zz) + solver.pinning*anneal*(1-zz.square()).sum(-1)
                + .5*(p.square()+solver.frequency**2*(q-solver.coupling*zz).square()).sum(-1))

    force = -target.gradient(z)+2*solver.pinning*anneal*z
    advanced = solver._advance(state, force, 0., anneal, 1, run, cache)
    assert (energy(advanced) <= energy(state)+1e-12).all()
    # Compensation really makes q=c*z minimize the auxiliary potential.
    q = solver.coupling*z
    square = .5*solver.frequency**2*(q-solver.coupling*z).square()
    expanded = (.5*solver.frequency**2*q.square() - solver.frequency**2*solver.coupling*q*z
                + .5*solver.frequency**2*solver.coupling**2*z.square())
    torch.testing.assert_close(square, expanded)


@pytest.mark.parametrize('cls', [SteadyStateSqueezing, TomonagaLuttinger])
def test_preconditioned_deterministic_step_descends_target(device, cls):
    _, run, target = _fixture(device, sparse=True)
    solver = cls(temperature=0.)
    z = run.normal((3, 4)).tanh()*.5
    state, cache = solver._prepare(run, z)
    anneal = .7
    force = -target.gradient(z)+2*solver.pinning*anneal*z
    advanced = solver._advance(state, force, 0., anneal, 1, run, cache)
    zz = solver._position(advanced)
    old = target.energy(z)+solver.pinning*anneal*(1-z.square()).sum(-1)
    new = target.energy(zz)+solver.pinning*anneal*(1-zz.square()).sum(-1)
    assert (new <= old+1e-12).all()
    if cls is SteadyStateSqueezing:
        mobility = cache['mobility']
        assert float(mobility.median()) == 1.
        assert (mobility >= solver.mobility_min).all() and (mobility <= solver.mobility_max).all()


@pytest.mark.parametrize('variant', [CavityAtomCooling(coupling=0.),
                                    OptomechanicalCooling(optical_coupling=0.),
                                    QuadratureSqueezing(squeeze=0., temperature=.1)])
def test_zero_driver_limit_is_exact_doppler_trajectory(device, variant):
    problem, _, _ = _fixture(device)
    from dataclasses import replace
    settings = dict(max_steps=7, candidate_interval=1, polish_sweeps=0)
    baseline = DopplerCooling(**settings).solve(problem, restarts=3, seed=42, history_interval=1)
    actual = replace(variant, **settings).solve(problem, restarts=3, seed=42, history_interval=1)
    torch.testing.assert_close(actual.best_assignments, baseline.best_assignments, rtol=0, atol=0)
    torch.testing.assert_close(actual.energy_history, baseline.energy_history, rtol=0, atol=0)


def test_quadrature_inverse_signs_and_zero_force_pulse_order(device):
    _, run, _ = _fixture(device)
    z = torch.tensor([[.1, -.2, .3, -.4]], dtype=torch.float64, device=device)
    p = z.flip(1)*.1
    stretch = math.exp(.2)
    torch.testing.assert_close((z*stretch)/stretch, z)
    torch.testing.assert_close((p/stretch)*stretch, p)
    assert torch.equal((z*stretch).sign(), z.sign())
    solver = QuadratureSqueezing(damping=0., squeeze=.2, pulse_interval=1)
    _, cache = solver._prepare(run, z)
    zz, pp = solver._advance((z, p), torch.zeros_like(z), 0., 0., 1, run, cache)
    torch.testing.assert_close(zz, (z+solver.time_step*p)*stretch)
    torch.testing.assert_close(pp, p/stretch)


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('sparse', [False, True])
def test_small_search_source_score_and_monotone_archive(device, cls, sparse):
    problem, _, _ = _fixture(device, sparse=sparse)
    solver = cls(max_steps=6, candidate_interval=2, sparse_threshold=1. if sparse else 0.)
    result = solver.solve(problem, restarts=4, seed=16, history_interval=2)
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments.to(torch.float64)))
    assert torch.isfinite(result.best_energies).all()
    initial_energy = result.energy_history[0]
    assert (result.best_energies <= initial_energy).all()
    assignments = torch.tensor(list(itertools.product((0., 1.), repeat=4)),
                               dtype=torch.float64, device=device)
    optimum = problem.energy(assignments).min()
    assert (result.best_energies >= optimum-1e-12).all()
    assert result.best_energies.min() <= optimum+1e-12


@pytest.mark.parametrize('cls', CLASSES)
def test_one_variable_exact_with_offset_and_zero_steps(device, cls):
    for coefficient in (-2., 3.):
        problem = QUBO(torch.tensor([[coefficient]], dtype=torch.float64, device=device), offset=7.)
        result = cls(max_steps=2).solve(problem, restarts=2, seed=1)
        assert float(result.best_energies.min()) == 7.+min(0., coefficient)
        zero = cls(max_steps=0).solve(problem, restarts=2, seed=1)
        torch.testing.assert_close(zero.best_energies, problem.energy(zero.best_assignments.double()))


@pytest.mark.parametrize('cls,kwargs', [
    (DopplerCooling, {'damping': float('nan')}),
    (CavityAtomCooling, {'memory_damping': -1}),
    (CavityOptomechanics, {'frequency': 0}),
    (OptomechanicalCooling, {'linewidth': 0}),
    (SteadyStateSqueezing, {'mobility_min': 2}),
    (TomonagaLuttinger, {'filter_terms': 0}),
    (TomonagaLuttinger, {'coordinate_cap': 1000}),
    (QuadratureSqueezing, {'squeeze': 1000}),
    (DopplerCooling, {'terminal_fraction': 1.}),
])
def test_invalid_controls_fail_before_allocation(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)

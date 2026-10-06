"""Opt-in execution-clock schedules without real wall-clock waits."""
import pytest
import torch

from qubo_solvers import (
    QUBO, AngularAnnealing, EasyAxisAnnealing, HeatBathAnnealing,
    MeanFieldAnnealing, ReplicaAnnealing, SimulatedAnnealing,
    SphericalAnnealing, SpinCoherentAnnealing, SpinVectorLangevin,
    TAPAnnealing, TransverseRoute, VectorAmplitudeAnnealing, create_bqm_solver,
)
from qubo_solvers import observation
from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.backends.torch_physics import smoothSchedule


class FakeClockObserver:
    """A deterministic clock-control substitute, with inert candidate hooks."""
    def __init__(self, fraction=.37, enabled=True):
        self.fraction = fraction
        self.wall_schedule = enabled
        self.requests = []

    def schedule_fraction(self, legacy_value):
        self.requests.append(legacy_value)
        return self.fraction

    def poll(self):
        pass

    def capture(self, *args, **kwargs):
        pass

    def prepared(self, *args, **kwargs):
        pass


def problem():
    return QUBO(torch.tensor([
        [-.7, .4, -.1], [.4, -.2, .3], [-.1, .3, -.9],
    ], dtype=torch.float64), 11.5)


def solve(solver):
    return solver.solve(problem(), restarts=2, seed=81, history_interval=1)


def assert_identical_results(left, right):
    for name in ('final_assignments', 'final_energies', 'best_assignments', 'best_energies',
                 'iterations', 'termination_reasons', 'energy_history', 'history_iterations'):
        torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)


LINEAR_END = [.25, .5, .75, 1.]
LINEAR_START = [0., 1/3, 2/3, 1.]
HEUN = [0., .25, .25, .5, .5, .75, .75, 1.]
NATIVE_CASES = [
    (EasyAxisAnnealing(4), LINEAR_END),
    (SpinCoherentAnnealing(4), LINEAR_END),
    (VectorAmplitudeAnnealing(4), LINEAR_END),
    (AngularAnnealing(4), HEUN),
    (TransverseRoute(4), HEUN),
    (SpinVectorLangevin(4), [0., 1/3, 1/3, 2/3, 2/3, 1., 1., 1.]),
    (SpinVectorLangevin(4, integrator='euler'), LINEAR_START),
    (MeanFieldAnnealing(4), LINEAR_START),
    (ReplicaAnnealing(4), LINEAR_END),
    (SphericalAnnealing(4, penalty=1.3), [None]*4),
    (HeatBathAnnealing(4, 2., .1), LINEAR_START),
]


@pytest.mark.parametrize('solver,expected', NATIVE_CASES, ids=lambda value: type(value).__name__)
def test_native_default_fractions_and_disabled_observer_are_exact(monkeypatch, solver, expected):
    calls = []
    original = observation.schedule_fraction

    def recorded(value):
        result = original(value)
        calls.append((value, result))
        return result

    monkeypatch.setattr(observation, 'schedule_fraction', recorded)
    baseline = solve(solver)
    assert calls == list(zip(expected, expected))
    calls.clear()
    disabled = FakeClockObserver(fraction=.99, enabled=False)
    with observation.observing(disabled):
        observed = solve(solver)
    assert calls == list(zip(expected, expected))
    assert disabled.requests == []
    assert_identical_results(baseline, observed)


@pytest.mark.parametrize('solver,expected', NATIVE_CASES, ids=lambda value: type(value).__name__)
def test_native_enabled_observer_controls_every_schedule_value(monkeypatch, solver, expected):
    applied = []
    original = observation.schedule_fraction

    def recorded(value):
        result = original(value)
        applied.append(result)
        return result

    monkeypatch.setattr(observation, 'schedule_fraction', recorded)
    clock = FakeClockObserver()
    with observation.observing(clock):
        result = solve(solver)
    assert clock.requests == expected
    assert applied == [clock.fraction] * len(expected)
    torch.testing.assert_close(problem().energy(result.best_assignments), result.best_energies)
    assert torch.isfinite(result.final_energies).all()


@pytest.mark.parametrize('solver', [EasyAxisAnnealing(4), SpinCoherentAnnealing(4),
                                  VectorAmplitudeAnnealing(4), AngularAnnealing(4), TransverseRoute(4)])
def test_injected_fraction_reaches_the_existing_force_equations(monkeypatch, solver):
    method = '_rhs' if hasattr(solver, '_rhs') else '_gradient'
    original = getattr(type(solver), method)
    fractions = []

    def recorded(self, state, fraction, objective):
        fractions.append(fraction)
        return original(self, state, fraction, objective)

    monkeypatch.setattr(type(solver), method, recorded)
    clock = FakeClockObserver(fraction=.625)
    with observation.observing(clock):
        solve(solver)
    assert len(fractions) in (4, 8)
    assert fractions == [clock.fraction] * len(fractions)


@pytest.mark.parametrize('steps', [1, 2, 4, 37, 1000])
def test_physics_legacy_smooth_schedule_preserves_exact_formula(steps):
    for step in (0., .5, 1., steps-1., float(steps)):
        legacy = min(1., step / max(1., .8*(steps-1)))
        expected = legacy*legacy*(3-2*legacy)
        assert smoothSchedule(step, steps) == expected
        with observation.observing(FakeClockObserver(enabled=False)):
            assert smoothSchedule(step, steps) == expected


@pytest.mark.parametrize('fraction,expected', [(0., 0.), (.4, .5), (.8, 1.), (1., 1.)])
def test_physics_wall_schedule_keeps_eighty_percent_ramp(fraction, expected):
    clock = FakeClockObserver(fraction=fraction)
    with observation.observing(clock):
        assert smoothSchedule(0, 1000) == expected
        assert smoothSchedule(999, 1000) == expected
    assert clock.requests == [None, None]


@pytest.mark.parametrize('solver', [TAPAnnealing(4), SimulatedAnnealing(4, 2., .1)])
def test_protected_tap_and_simulated_annealing_keep_their_schedule(solver):
    baseline = solve(solver)
    clock = FakeClockObserver(fraction=.99)
    with observation.observing(clock):
        observed = solve(solver)
    assert clock.requests == []
    assert_identical_results(baseline, observed)


def backend_run(solver_name):
    source = QUBOProblem([-1., .5, -2.], [0, 1], [1, 2], [.5, -.75])
    solver = create_bqm_solver(solver_name, {'device': 'cpu'})
    settings = dict(steps=4, runs=1, run_batch_size=1, seed=19, dtype='float64',
                    candidate_interval=1, conditional_rounding=False,
                    local_search_steps=0, matrix_format='dense')
    result = solver.solve(source, settings)
    assert result.energy == source.energy(result.sample)
    return result


def test_dynamical_geometry_midpoint_and_smooth_controls_use_fake_clock():
    clock = FakeClockObserver(fraction=.4)
    with observation.observing(clock):
        backend_run('lib_dynamical_geometry')
    assert clock.requests == [.125, None, .375, None, .625, None, .875, None]


@pytest.mark.parametrize('solver', ['lib_altermagnet', 'lib_geometric', 'lib_supersymmetric'])
def test_shared_physics_schedule_has_no_real_clock_dependency(solver):
    baseline = backend_run(solver)
    disabled = FakeClockObserver(enabled=False)
    with observation.observing(disabled):
        observed = backend_run(solver)
    assert disabled.requests == []
    assert baseline.sample == observed.sample and baseline.energy == observed.energy
    clock = FakeClockObserver(fraction=.4)
    with observation.observing(clock):
        backend_run(solver)
    assert clock.requests == [None]*4

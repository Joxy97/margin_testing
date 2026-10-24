"""The tuning controls preserve defaults and the physical state domain."""

import inspect
import textwrap

import pytest
import torch

from qubo_solvers import QUBO
from qubo_solvers import compendium_iii_a as a
from qubo_solvers import compendium_iii_c as c


def _fixture():
    generator = torch.Generator().manual_seed(20261024)
    return QUBO(torch.randn((9, 9), generator=generator, dtype=torch.float64))


@pytest.mark.parametrize('cls', [a.C3AubryAndre, a.C3Haldane,
    a.C3DzyaloshinskiiMoriya, a.C3FrohlichPolaron])
def test_a_default_seeded_trajectory_matches_original_literals(cls, monkeypatch):
    solver = cls(max_steps=11, candidate_interval=3, polish_sweeps=0)
    problem = _fixture()
    actual = solver.solve(problem, restarts=3, seed=109, history_interval=1)
    original = textwrap.dedent(inspect.getsource(a._Continuous._search))
    original = original.replace('self.initial_scale *', '.05 *').replace(
        'self.initial_noise *', '.01 *')
    original = original.replace('    margin = torch.finfo(z.dtype).eps\n', '').replace(
        '    z = z.clamp(-1 + margin, 1 - margin)\n', '')
    namespace = dict(vars(a))
    exec(compile(original, '<original_compendium_iii_initialization>', 'exec'), namespace)
    monkeypatch.setattr(a._Continuous, '_search', namespace['_search'])
    expected = solver.solve(problem, restarts=3, seed=109, history_interval=1)
    for name in ('best_assignments', 'best_energies', 'final_assignments',
                 'final_energies', 'energy_history'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)


@pytest.mark.parametrize('cls', [c.C3Rashba, c.C3QuantumRabi, c.C3FeshbachTwoChannel])
def test_c_default_seeded_trajectory_matches_original_schedule(cls, monkeypatch):
    solver = cls(max_steps=11, candidate_interval=3, polish_steps=2)
    problem = _fixture()
    actual = solver.solve(problem, restarts=3, seed=109, history_interval=1)
    def original_controls(self, step):
        fraction = c.observation.schedule_fraction(step / max(1, self.max_steps))
        fraction = min(1., max(0., fraction / .85))
        ramp = fraction * fraction * (3 - 2 * fraction)
        return 1 - ramp, self.pinning * ramp
    monkeypatch.setattr(c._C3Base, '_controls', original_controls)
    expected = solver.solve(problem, restarts=3, seed=109, history_interval=1)
    for name in ('best_assignments', 'best_energies', 'final_assignments',
                 'final_energies', 'energy_history'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)


@pytest.mark.parametrize('terminal', [0., .05, .15, .4, .8])
def test_terminal_schedule_has_monotone_turnoff_and_exact_endpoint(terminal):
    solver = c.C3Rashba(max_steps=100, terminal_fraction=terminal)
    values = [solver._controls(step) for step in range(101)]
    assert values[0] == (1., 0.)
    assert values[-1] == (0., solver.pinning)
    assert all(x[0] >= y[0] and x[1] <= y[1] for x, y in zip(values, values[1:]))
    for step, value in enumerate(values):
        if step / 100 >= 1 - terminal:
            assert value == (0., solver.pinning)


@pytest.mark.parametrize('terminal', [-.1, 1., float('inf'), float('nan'), True])
def test_invalid_terminal_fraction_rejected(terminal):
    with pytest.raises(ValueError):
        c.C3Rashba(terminal_fraction=terminal)


@pytest.mark.parametrize('kwargs', [{'initial_scale': 0.}, {'initial_scale': 1.},
    {'initial_scale': float('nan')}, {'initial_noise': -.1}, {'initial_noise': True}])
def test_invalid_initial_cloud_rejected(kwargs):
    with pytest.raises(ValueError):
        a.C3Haldane(**kwargs)


def test_wide_initial_cloud_stays_finite_for_tanh_carrier():
    result = a.C3Haldane(max_steps=5, initial_scale=.99, initial_noise=3.,
                        polish_sweeps=0).solve(_fixture(), restarts=4, seed=1)
    assert bool(torch.isfinite(result.best_energies).all())

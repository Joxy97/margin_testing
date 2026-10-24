"""Physical schedule controls preserve scoring and valid state spaces."""
import itertools

import pytest
import torch

from qubo_solvers import QUBO, create_solver


DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
CASES = [
    ('lib_aklt', dict(initial_vacancy_probability=1., cooling_fraction=.5)),
    ('lib_noisy_excitation_transport', dict(cooling_fraction=.5)),
    ('lib_moore_read', dict(random_triple_interval=1)),
    ('lib_tj_vacancy', dict(initial_vacancy_probability=1., single_move_probability=.2,
                           chemical_start=-2., chemical_end=.5,
                           start_temperature=.3, end_temperature=.001)),
    ('lib_potts', dict(start_temperature=.2, end_temperature=.002, final_pinning=.2)),
    ('lib_zn_chiral', dict(start_temperature=.2, end_temperature=.002, final_pinning=.2)),
    ('lib_xy_coordinate', dict(cold_start_temperature=.05, cold_end_temperature=.0001,
                               hot_start_temperature=.5, hot_end_temperature=.01,
                               final_pinning=.2)),
    ('lib_quantum_dimer', dict(single_slice_probability=.2, full_worldline_probability=.5)),
]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('solver_id,parameters', CASES)
def test_new_controls_preserve_binary_source_scoring(device, solver_id, parameters):
    q = torch.tensor([[-1., .7, -.5], [.7, -.1, .3], [-.5, .3, -.2]],
                     device=device, dtype=torch.float64)
    problem = QUBO(q, offset=1.3)
    solver = create_solver(solver_id, max_steps=3, **parameters)
    result = solver.solve(problem, restarts=3, seed=43, history_interval=1)
    repeat = solver.solve(problem, restarts=3, seed=43)
    states = torch.tensor(list(itertools.product([0., 1.], repeat=3)),
                          device=device, dtype=torch.float64)
    assert bool(((result.best_assignments == 0) | (result.best_assignments == 1)).all())
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments.double()))
    torch.testing.assert_close(result.best_energies, repeat.best_energies)
    assert bool((result.best_energies >= problem.energy(states).min() - 1e-12).all())


@pytest.mark.parametrize('solver_id,parameters', [
    ('lib_aklt', {'initial_vacancy_probability': 1.1}),
    ('lib_aklt', {'cooling_fraction': 0.}),
    ('lib_noisy_excitation_transport', {'cooling_fraction': 1.1}),
    ('lib_moore_read', {'random_triple_interval': 0}),
    ('lib_tj_vacancy', {'chemical_end': float('inf')}),
    ('lib_tj_vacancy', {'single_move_probability': -1.}),
    ('lib_potts', {'start_temperature': 0.}),
    ('lib_zn_chiral', {'final_pinning': -1.}),
    ('lib_xy_coordinate', {'cold_start_temperature': 2.}),
    ('lib_xy_coordinate', {'hot_end_temperature': 0.}),
    ('lib_quantum_dimer', {'single_slice_probability': .9, 'full_worldline_probability': .2}),
    ('lib_quantum_dimer', {'single_slice_probability': float('nan')}),
])
def test_new_control_validation(solver_id, parameters):
    with pytest.raises(ValueError):
        create_solver(solver_id, **parameters)

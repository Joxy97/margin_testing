"""Public tensor, compact, resident and runtime contracts for the 23 additions."""
from dataclasses import fields

import numpy as np
import pytest
import torch

import qubo_solvers as library
from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.backends.resource_plan import BQMResourcePlan
from qubo_solvers.backends.torch_qubo import TorchQUBO
from qubo_benchmark.runtime.worker import native_input_dtype
from qubo_benchmark.runtime.continuous_calibration import estimate_worker_memory


DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


def options(name):
    names = {item.name for item in fields(library.library_solver_class(name))}
    values = dict(max_steps=2, dynamics_dtype='float32')
    if 'polish_sweeps' in names:
        values['polish_sweeps'] = 0
    if 'polish_steps' in names:
        values['polish_steps'] = 0
    if 'slices' in names:
        values['slices'] = 4
    return values


@pytest.mark.parametrize('name', library.HAMILTONIAN_SOLVERS)
def test_public_factory_and_native_admission_match(name):
    cls = library.library_solver_class(name)
    assert cls.__name__ in library.__all__
    assert getattr(library, cls.__name__) is cls
    assert library.create_solver(name).max_steps == cls().max_steps
    assert library.create_bqm_solver(name)._getParameters()['max_steps'] == cls().max_steps
    assert native_input_dtype(dict(solver=name, parameters={'dtype': 'float32'})) == 'float64'
    solver = library.create_solver(name, **options(name))
    problem = library.QUBO(torch.zeros((8, 8), dtype=torch.float64))
    params = dict(options(name), runs=3, run_batch_size=2, dtype='float32', best_only=True)
    admitted = estimate_worker_memory(8, 9, params, True, name)
    actual = solver.estimate_memory(problem, restarts=3, batch_size=2, best_only=True)
    assert admitted['search_matrix_bytes'] + admitted['workspace_bytes'] == actual.total_bytes


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', library.HAMILTONIAN_SOLVERS)
def test_compact_and_resident_original_precision_without_dense_adapter(name, device, monkeypatch):
    source = QUBOProblem(np.array([16777217., -3., -.25]), [0, 1, 0, 2], [1, 0, 0, 2],
                        [.5, -.125, .25, -.75], offset=2.75)
    resident = TorchQUBO(source, *(torch.tensor(array, dtype=dtype, device=device)
        for array, dtype in ((source.linear, torch.float64), (source.quadraticHeads, torch.int64),
                            (source.quadraticTails, torch.int64), (source.quadraticBiases, torch.float64))))
    adapter = library.create_bqm_solver(name, {'device': device})
    def forbid(*args, **kwargs):
        pytest.fail('Hamiltonian solvers must use source-precision compact/resident operators')
    monkeypatch.setattr(adapter, '_toLibraryProblem', forbid)
    monkeypatch.setattr(adapter, '_toLibraryResident', forbid)
    params = dict(options(name), runs=3, run_batch_size=2, dtype='float32', seed=7,
                  memory_limit_bytes=128 * 1024**2)
    ordinary = adapter.solve(source, params)
    resident_result, = adapter.solveResidentPlanned([resident], BQMResourcePlan.create([1], 1), params)
    for result in (ordinary, resident_result):
        assert result.energy == source.energy(result.sample)
        assert set(result.sample).issubset({0, 1})
    assert ordinary.energy == resident_result.energy
    assert ordinary.sample == resident_result.sample


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', library.HAMILTONIAN_SOLVERS)
def test_direct_ising_batched_history_and_initial_incumbent(name, device):
    problem = library.QUBO(torch.tensor([[-1., .375, 0.], [.375, .25, -.125], [0., -.125, -2.]],
                                      device=device, dtype=torch.float64), 4.25).to_ising()
    initial = torch.tensor([[-1, 1, -1], [1, -1, 1], [1, 1, -1]], device=device, dtype=torch.int8)
    result = library.create_solver(name, **options(name)).solve(problem, restarts=3, batch_size=2,
                    initial_assignments=initial, seed=21, history_interval=1)
    torch.testing.assert_close(result.best_energies, problem.energy(result.best_assignments))
    torch.testing.assert_close(result.final_energies, problem.energy(result.final_assignments))
    assert bool((result.best_energies <= problem.energy(initial) + 1e-12).all())
    assert bool(torch.isfinite(result.energy_history).all())
    assert result.best_assignments.device == initial.device

"""Registration, immutable presets and original-coefficient integration."""

from copy import deepcopy

import numpy as np
import pytest
import torch

import qubo_solvers
from qubo_solvers import (
    BACKEND_SOLVERS, GPU_SOLVERS, NATIVE_SOLVERS, SOLVERS, create_bqm_solver,
    create_solver, default_algorithm_parameters, library_solver_class, solver_capabilities,
)
from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.backends.resource_plan import BQMResourcePlan
from qubo_solvers.backends.torch_qubo import TorchQUBO
from qubo_benchmark.adapters import GPU_SOLVERS as BENCHMARK_SOLVERS
from qubo_benchmark.runtime.common import CONFIG, ROOT, read
from qubo_benchmark.runtime.continuous_cli import _parameters
from qubo_benchmark.runtime.continuous_calibration import estimate_worker_memory
from qubo_benchmark.runtime.selection import (
    inventory, resolve_legacy_exchange_defaults, resolve_new_solver_defaults, select_solvers,
)
from qubo_benchmark.runtime.worker import WarmWorker, native_input_dtype


NEW_SOLVERS = ('lib_su2_chiral_isospin', 'lib_gravity_tunneling')
OLD_NATIVE = (
    'lib_simulated_annealing', 'lib_greedy_local_search', 'lib_spin_vector_langevin',
    'lib_angular_annealing', 'lib_transverse_route', 'lib_easy_axis_annealing',
    'lib_spin_coherent_annealing', 'lib_vector_amplitude_annealing',
    'lib_mean_field_annealing', 'lib_tap_annealing', 'lib_spherical_annealing',
    'lib_contact_annealing', 'lib_replica_annealing', 'lib_heat_bath_annealing',
    'lib_tabu_search', 'lib_random_search', 'lib_exchange_cascade',
)


def test_exactly_two_native_additions_keep_existing_registry_order():
    assert NATIVE_SOLVERS == OLD_NATIVE + NEW_SOLVERS
    assert len(BACKEND_SOLVERS) == 11
    assert len(SOLVERS) == len(set(SOLVERS)) == 30
    assert len(GPU_SOLVERS) == 29
    assert len(BENCHMARK_SOLVERS) == 25
    for name, class_name in zip(NEW_SOLVERS, ('SU2ChiralIsospin', 'GravityTunneling')):
        cls = getattr(qubo_solvers, class_name)
        assert class_name in qubo_solvers.__all__
        assert library_solver_class(name) is cls
        assert create_solver(name).max_steps == cls().max_steps == 512
        assert create_bqm_solver(name)._getParameters()['max_steps'] == 512
        assert isinstance(create_solver(name, max_steps=0), cls)
        assert default_algorithm_parameters(name, steps=7) == {'max_steps': 7}
        cap = solver_capabilities(name)
        assert cap['gpu'] and cap['native_tensor_contract']
        assert cap['resident_preparation_mode'] == 'device'
        assert cap['restriction'] is None
    for name in OLD_NATIVE:
        algorithm = create_solver(name)
        assert getattr(algorithm,'max_steps',getattr(algorithm,'sweeps',None)) == 1000


def test_new_presets_append_without_mutating_or_retuning_old_settings():
    baseline = read(CONFIG/'benchmark_solvers.json')['solvers']
    snapshot = deepcopy(baseline)
    legacy = resolve_legacy_exchange_defaults(baseline)
    resolved = resolve_new_solver_defaults(legacy)
    assert baseline == snapshot and len(baseline) == 23
    assert tuple(resolved)[-2:] == NEW_SOLVERS
    assert {name: resolved[name] for name in legacy} == legacy
    rows = inventory(baseline)
    assert len(rows) == 30
    assert len(select_solvers(rows)) == 25
    for name in NEW_SOLVERS:
        params = resolved[name]
        actual = create_bqm_solver(name, {'device': 'cpu'})._getParameters(params)
        actual.pop('seed')
        assert actual == params
        assert params['runs'] == params['run_batch_size'] == 16
        assert params['best_only'] and params['dtype'] == 'float32'
        assert params['memory_limit_bytes'] == 512*1024**2
        assert next(row for row in rows if row['solver_id'] == name)['status'] == 'runnable'


@pytest.mark.parametrize('size', [100, 200, 500, 1000, 5000, 10000])
def test_continuous_presets_keep_historical_entries_and_resolve_new_guards(size):
    config = read(CONFIG/'benchmark_solvers_selected_20s.json')
    snapshot = deepcopy(config)
    profile = config.get('size_configurations', {}).get(str(size))
    old = profile.get('solvers', profile) if profile is not None else config['solvers']
    legacy = resolve_legacy_exchange_defaults(old)
    resolved = _parameters(config, size)
    assert config == snapshot
    # Existing large-size override rules are covered by the continuous tests.
    for name in legacy:
        expected = dict(legacy[name])
        if profile is None and size >= 5000:
            from qubo_benchmark.runtime.continuous_worker import PROTECTED
            if name not in PROTECTED and 'max_dense_variables' in expected:
                expected['max_dense_variables'] = 16384
            if name in PROTECTED and config.get('protected_memory_guard_override') and 'memory_limit_bytes' in expected:
                expected['memory_limit_bytes'] = 8*1024**3
        assert resolved[name] == expected
    for name in NEW_SOLVERS:
        assert resolved[name]['memory_limit_bytes'] == (8*1024**3 if size >= 5000 else 512*1024**2)


def test_explicit_new_configuration_is_preserved_and_incomplete_one_rejected():
    resolved = resolve_new_solver_defaults(read(CONFIG/'benchmark_solvers.json')['solvers'])
    resolved[NEW_SOLVERS[0]]['max_steps'] = 3
    repeated = resolve_new_solver_defaults(resolved)
    assert repeated == resolved
    repeated[NEW_SOLVERS[0]].pop('max_steps')
    row = next(row for row in inventory(repeated) if row['solver_id'] == NEW_SOLVERS[0])
    assert row['status'] == 'unavailable'


@pytest.mark.parametrize('filename,count', [
    ('full.json', 25), ('full_compatible.json', 25), ('full_all.json', 30), ('smoke_all.json', 30),
])
def test_enumerated_benchmark_configs_include_both_new_methods(filename, count):
    config = read(ROOT/'benchmark_configs'/filename)
    ids = [spec['solver'] for spec in config['solvers']]
    assert len(ids) == len(set(ids)) == count
    assert tuple(ids[-2:]) == NEW_SOLVERS


@pytest.mark.parametrize('name', NEW_SOLVERS)
def test_new_compact_adapter_preserves_source_and_bypasses_dense_conversion(name, monkeypatch):
    source = QUBOProblem(
        [-1., -.25, .5], [0, 1, 0, 2], [1, 0, 2, 2],
        [.5, -.25, .125, -.75], offset=7.25, seedOffset=19,
    )
    adapter = create_bqm_solver(name, {'device': 'cpu'})
    def forbid(*args, **kwargs):
        pytest.fail('The new methods must preserve original compact coefficients')
    monkeypatch.setattr(adapter, '_toLibraryProblem', forbid)
    result = adapter.solve(source, dict(steps=2, runs=3, run_batch_size=2, dtype='float32', seed=11))
    assert result.energy == source.energy(result.sample)
    assert all(bit in (0, 1) for bit in result.sample)


@pytest.mark.parametrize('name', NEW_SOLVERS)
def test_sparse_compact_memory_estimate_avoids_dense_quadratic_storage(name):
    source = QUBOProblem(np.zeros(10000), [0], [9999], [-1.])
    adapter = create_bqm_solver(name, {'device': 'cpu'})
    estimate = adapter.estimatedWorkingMemoryBytes(source, dict(
        steps=1, runs=16, run_batch_size=16, dtype='float32', best_only=True,
    ))
    assert 0 < estimate < 256*1024**2


@pytest.mark.parametrize('name', NEW_SOLVERS)
def test_new_resident_adapter_bypasses_dense_conversion(name, monkeypatch):
    source = QUBOProblem([-1., -.25, .5], [0, 1], [1, 2], [.5, -.25], offset=7.25)
    resident = TorchQUBO(source, *(torch.tensor(array, dtype=dtype) for array, dtype in (
        (source.linear, torch.float64), (source.quadraticHeads, torch.int64),
        (source.quadraticTails, torch.int64), (source.quadraticBiases, torch.float64),
    )))
    adapter = create_bqm_solver(name, {'device': 'cpu'})
    def forbid(*args, **kwargs):
        pytest.fail('The new methods must retain resident original coefficients')
    monkeypatch.setattr(adapter, '_toLibraryResident', forbid)
    result, = adapter.solveResidentPlanned([resident], BQMResourcePlan.create([1], 1), dict(
        steps=2, runs=3, run_batch_size=2, dtype='float32', seed=11,
    ))
    assert result.energy == source.energy(result.sample)


@pytest.mark.parametrize('name', NEW_SOLVERS)
def test_new_worker_input_retains_fp64_coefficients(name):
    assert native_input_dtype(dict(solver=name, parameters={'dtype': 'float32'})) == 'float64'
    assert native_input_dtype(dict(solver='lib_random_search', parameters={'dtype': 'float32'})) == 'float32'


@pytest.mark.parametrize('name', NEW_SOLVERS)
def test_native_worker_cache_keeps_original_fp64_across_solver_and_precision_changes(name, monkeypatch):
    from qubo_benchmark.model import Problem as BenchmarkProblem
    original = BenchmarkProblem(2,np.array([0,1]),np.array([0,1]),
                                np.array([16777217.,-1.]),0.)
    monkeypatch.setattr(BenchmarkProblem,'load',lambda path:original)
    worker = WarmWorker.__new__(WarmWorker)
    worker.torch, worker.device, worker.threads = torch, 'cpu', 1
    worker.cache_key, worker.trials, worker.warmup_s = None, 0, 0.
    worker.worker_id, worker.pid = 'fixture', 1
    job = dict(npz='immutable-fixture',solver=name,parameters={'dtype':'float32'},
               device='cpu',cpu_threads=1)
    first = worker.prepare(job)
    assert not first['input_cache_hit']
    assert worker.prepared.Q.dtype == torch.float64
    assert float(worker.prepared.Q[0,0]) == 16777217.
    shared = worker.prepared
    second = worker.prepare(dict(job,parameters={'dtype':'float64'}))
    assert second['input_cache_hit'] and worker.prepared is shared
    previous = worker.prepare(dict(job,solver='lib_random_search'))
    assert not previous['input_cache_hit'] and worker.prepared.Q.dtype == torch.float32
    rebuilt = worker.prepare(job)
    assert not rebuilt['input_cache_hit'] and worker.prepared.Q.dtype == torch.float64
    assert float(worker.prepared.Q[0,0]) == 16777217.


@pytest.mark.parametrize('name', NEW_SOLVERS)
def test_new_admission_accounts_for_fp64_source_and_explicit_dynamics_precision(name):
    params = dict(runs=16, run_batch_size=16, dtype='float32')
    new = estimate_worker_memory(1000, 10, params, True, name)
    assert new['native_input_bytes'] == 8*1000**2+8
    assert new['operator_workspace_bytes'] == (32+2*4)*1000**2
    assert new['search_matrix_bytes'] == (8+32+2*4)*1000**2+8
    assert new['host_per_worker_estimate']-new['workspace_bytes'] == (
        3*(32*10+8*1000)+16*1000**2+1024**3)
    precise = estimate_worker_memory(1000, 10, dict(params, dynamics_dtype='float64'), True, name)
    assert precise['native_input_bytes'] == new['native_input_bytes']
    assert precise['operator_workspace_bytes'] == (32+2*8)*1000**2
    assert precise['workspace_bytes'] > new['workspace_bytes']


@pytest.mark.parametrize('name', NEW_SOLVERS)
@pytest.mark.parametrize('precision', ['float32','float64'])
@pytest.mark.parametrize('best_only', [False,True])
@pytest.mark.parametrize('coo_count', [10,2080])
def test_new_continuous_admission_covers_direct_native_estimate(name, precision, best_only, coo_count):
    n, runs, batch = 64, 19, 16
    params = dict(runs=runs,run_batch_size=batch,dtype=precision,best_only=best_only)
    admission = estimate_worker_memory(n,coo_count,params,True,name)
    source = qubo_solvers.QUBO(torch.zeros((n,n),dtype=torch.float64),7.25)
    algorithm = create_solver(name,max_steps=3,dynamics_dtype=precision)
    native = algorithm.estimate_memory(source,restarts=runs,batch_size=batch,best_only=best_only)
    assert admission['search_matrix_bytes']+admission['workspace_bytes'] == native.total_bytes
    assert admission['workspace_bytes'] == (
        admission['common_workspace_bytes']+admission['population_workspace_bytes']+admission['output_bytes'])
    assert admission['gpu_per_worker_estimate'] >= (
        native.total_bytes+admission['independent_scoring_bytes']+1024**3)


def test_legacy_continuous_admission_formula_is_preserved():
    params = dict(runs=19,run_batch_size=16,dtype='float32')
    admission = estimate_worker_memory(1000,10,params,True,'lib_spin_vector_langevin')
    assert admission['search_matrix_bytes'] == 6*4*1000**2
    assert admission['workspace_bytes'] == 4*(108*16*1000+288*16)
    assert admission['host_per_worker_estimate'] == (
        3*(32*10+8*1000)+1000**2*(8+4)+admission['workspace_bytes']+1024**3)


@pytest.mark.parametrize('name', NEW_SOLVERS)
def test_historical_runtime_admits_dense_native_cache_on_sparse_input(name, monkeypatch):
    from qubo_benchmark.model import Problem as BenchmarkProblem
    from qubo_benchmark.runtime import selection
    n = 1000
    source = BenchmarkProblem(n,np.array([0]),np.array([n-1]),np.array([-.5]),0.)
    monkeypatch.setattr(BenchmarkProblem,'load',lambda path:source)
    monkeypatch.setattr(selection,'host_memory_available',lambda:128*1024**3)
    monkeypatch.setattr(selection,'cpu_capacity',lambda:{'effective_cpus':8})
    parameters = resolve_new_solver_defaults({})[name]
    report = selection.worker_capacity([{'npz':'immutable-fixture'}],{name:parameters},
                                      'cpu',selection.PROTOCOL['execution'])
    algorithm = create_solver(name,max_steps=parameters['max_steps'],dynamics_dtype='float32')
    problem = qubo_solvers.QUBO(torch.zeros((n,n),dtype=torch.float64))
    expected = algorithm.estimate_memory(problem,restarts=16,batch_size=16,best_only=True)
    assert report['algorithm_working_bytes_estimate'] >= expected.total_bytes
    compact = create_bqm_solver(name).estimatedWorkingMemoryBytes(
        QUBOProblem(np.zeros(n),[0],[n-1],[-1.]),parameters)
    assert report['algorithm_working_bytes_estimate'] > compact

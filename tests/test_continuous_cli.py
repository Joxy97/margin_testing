"""Named selection and frozen protected settings for the existing entrypoint."""
from copy import deepcopy
import queue
import runpy
import sys
import pytest
from qubo_benchmark.runtime import continuous_cli
from qubo_benchmark.runtime.continuous_cli import parser,_parameters
from qubo_benchmark.runtime.common import CONFIG,ROOT,filehash,read


def test_named_size_density_and_raw_flags():
    args=parser().parse_args(['--variable-count','5000','--dense','--no-keep-raw-results'])
    assert (args.variable_count,args.density,args.keep_raw_results)==(5000,'dense',False)
    with pytest.raises(SystemExit):
        parser().parse_args(['--sparse','--dense'])


@pytest.mark.parametrize('flag', ['--seed-workers','--gpu-workers','--workers-per-gpu'])
def test_worker_mode_aliases_and_optimum_stop_toggle(flag):
    args=parser().parse_args(['--variable-count','100','--dense',flag,'auto','--no-stop-on-optimum'])
    assert args.seed_workers=='auto' and args.stop_on_optimum is False


def test_new_protocol_refuses_discarding_postrun_hit_energies():
    args=parser().parse_args(['--variable-count','100','--sparse','--no-keep-raw-results','--no-plots'])
    with pytest.raises(ValueError,match='Retain checkpoint energies'):
        continuous_cli.prepare(args)


def fake_prepared():
    source=continuous_cli.source_snapshot_hash()
    return dict(manifest=dict(solvers={'lib_random_search':{}},
        problems=[dict(id='fixture',n=100,density='sparse',reference=-10,
                       reference_type='OPTIMUM',integer_objective=True)],
        inputs={'fixture':'input-hash'},source=dict(snapshot=source),execution={}),
        entry={'instance_id':'fixture'},npz='unused.npz',settings={'lib_random_search':{}},
        solvers=['lib_random_search'],devices=['cpu'],seeds=[0,1])


def test_group_execution_commits_optimum_times_and_percentage_progress(tmp_path,monkeypatch,capsys):
    from qubo_benchmark.runtime import continuous_pool
    from qubo_benchmark.runtime.compact_storage import CompactStore
    prepared=fake_prepared()
    args=parser().parse_args(['--variable-count','100','--sparse','--output',str(tmp_path),
                              '--no-plots'])
    args.keep_raw_results=True
    args.stop_on_optimum=True
    def plan(p,a):
        value={'lib_random_search':{'cpu':1}}
        p['manifest']['execution']['worker_plan']=value
        return value
    monkeypatch.setattr(continuous_cli,'_worker_plan',plan)
    def group(job,devices,workers,seeds,on_result):
        assert job['reference']==-10 and job['reference_type']=='OPTIMUM'
        assert job['stop_on_optimum'] and job['tolerance']==0
        assert workers=={'cpu':1} and seeds==[0,1]
        for seed in seeds:
            on_result(dict(seed=seed,device='cpu',status='complete',
                energies=[-9.]+[-10.]*8,time_to_optimum_s=.075,error=None))
        return [],.1
    monkeypatch.setattr(continuous_pool,'run_group',group)
    root,code=continuous_cli.execute(prepared,args)
    assert code==0
    stderr=capsys.readouterr().err
    assert all(text in stderr for text in ('0.00%','50.00%','100.00%','optimum at 0.075000s'))
    reopened=CompactStore(root,prepared['manifest'],seeds=[0,1])
    assert reopened._group('lib_random_search','fixture')['time_to_optimum_s']==[.075,.075]


def test_source_change_before_commit_leaves_seed_pending(tmp_path,monkeypatch):
    from qubo_benchmark.runtime import continuous_pool
    prepared=fake_prepared()
    args=parser().parse_args(['--variable-count','100','--sparse','--output',str(tmp_path),
                              '--no-plots'])
    args.keep_raw_results=True
    args.stop_on_optimum=True
    monkeypatch.setattr(continuous_cli,'_worker_plan',lambda p,a:{'lib_random_search':{'cpu':1}})
    def group(job,devices,workers,seeds,on_result):
        monkeypatch.setattr(continuous_cli,'source_snapshot_hash',lambda:'changed source')
        on_result(dict(seed=0,device='cpu',status='complete',energies=[-9.]*9))
    monkeypatch.setattr(continuous_pool,'run_group',group)
    with pytest.raises(continuous_cli.SourceIdentityError):
        continuous_cli.execute(prepared,args)
    progress=read(tmp_path/'progress/lib_random_search/fixture.json')['payload']
    assert progress['status']==[None,None] and progress['time_to_optimum_s']==[None,None]


def test_protected_numerics_and_authorized_large_memory_guard():
    baseline=read(CONFIG/'benchmark_solvers.json')
    config=deepcopy(baseline)
    assert _parameters(config,1000)['lib_simulated_annealing']==baseline['solvers']['lib_simulated_annealing']
    config['protected_memory_guard_override']=True
    for name in ('lib_simulated_annealing','lib_greedy_local_search','lib_tabu_search','lib_tap_annealing'):
        config['solvers'][name]['memory_limit_bytes']=8*1024**3
    _parameters(config,10000)
    config['solvers']['lib_simulated_annealing']['sweeps']+=1
    with pytest.raises(ValueError,match='Protected'):
        _parameters(config,10000)


@pytest.mark.parametrize('equals_form', [False, True])
def test_entrypoint_routes_both_resume_forms_to_compact_protocol(tmp_path, monkeypatch, equals_form):
    from qubo_benchmark.runtime import cli as legacy_cli
    (tmp_path/'manifest.json').write_text('{}', encoding='utf-8')
    selected=[]
    monkeypatch.setattr(continuous_cli,'main',lambda: selected.append('compact') or 0)
    monkeypatch.setattr(legacy_cli,'main',lambda: selected.append('legacy') or 0)
    arguments=[f'--resume={tmp_path}'] if equals_form else ['--resume',str(tmp_path)]
    monkeypatch.setattr(sys,'argv',[str(ROOT/'run_benchmark.py'),*arguments])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT/'run_benchmark.py'),run_name='__main__')
    assert result.value.code==0 and selected==['compact']


class FakeWarmWorker:
    def __init__(self, mutate=None):
        self.cache_key=None
        self.device='cpu'
        self.torch=object()
        self.native=True
        self.source=object()
        self.mutate=mutate
        self.prepare_calls=0

    def prepare(self,job):
        from qubo_solvers import LIBRARY_SOLVERS
        self.prepare_calls+=1
        self.native=job['solver'] in LIBRARY_SOLVERS
        self.cache_key=(job['npz'],job['parameters']['dtype'],self.native)
        if self.mutate:
            self.mutate()


def fake_job(tmp_path):
    path=tmp_path/'input.npz'
    path.write_bytes(b'original immutable input')
    return dict(npz=str(path),input_sha256=filehash(path),solver='lib_random_search',seed=0,
                parameters={'dtype':'float32'},device='cpu',cpu_threads=1)


def fake_setup(monkeypatch):
    from qubo_benchmark.runtime import worker as worker_module
    constructed=[]
    monkeypatch.setattr(continuous_cli,'ObjectiveScorer',
                        lambda *args,**kwargs: constructed.append((args,kwargs)) or object())
    monkeypatch.setattr(worker_module,'synchronize',lambda *args: None)
    return constructed


def test_cache_miss_input_identity_is_verified_before_and_after_setup(tmp_path, monkeypatch):
    job=fake_job(tmp_path)
    built=fake_setup(monkeypatch)
    original=continuous_cli.filehash
    checks=[]
    monkeypatch.setattr(continuous_cli,'filehash',lambda path: checks.append(path) or original(path))
    worker=FakeWarmWorker()
    continuous_cli._prepare_job(worker,job)
    assert checks==[job['npz'],job['npz']]
    assert worker.prepare_calls==1 and len(built)==1
    continuous_cli._prepare_job(worker,job)
    assert len(checks)==2 and len(built)==1
    changed=dict(job,input_sha256='different manifest input')
    with pytest.raises(continuous_cli.InputIdentityError,match='Cached immutable input'):
        continuous_cli._prepare_job(worker,changed)


def test_compact_to_sbm_preparation_rebuilds_scoring_operator_without_input_reload(tmp_path, monkeypatch):
    job = dict(fake_job(tmp_path), solver='lib_altermagnet')
    built = fake_setup(monkeypatch)
    worker = FakeWarmWorker()
    continuous_cli._prepare_job(worker, job)
    compact_key = worker.cache_key
    assert built[-1][1]['native'] is False
    continuous_cli._prepare_job(worker, dict(job, solver='lib_simulated_bifurcation'))
    assert worker.cache_key == compact_key
    assert len(built) == 2 and built[-1][1]['native'] is True
    continuous_cli._prepare_job(worker, dict(job, solver='lib_simulated_bifurcation'))
    assert len(built) == 2


def test_exchange_graph_stream_is_prepared_once_and_shared_only_as_infrastructure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    job = dict(fake_job(tmp_path), solver='lib_exchange_cascade')
    job['parameters']['graph_block'] = 8
    fake_setup(monkeypatch)
    worker = FakeWarmWorker()
    worker.device = 'cuda:2'
    created = []
    def stream(*, device):
        token = object()
        created.append((device, token))
        return token
    worker.torch = SimpleNamespace(cuda=SimpleNamespace(Stream=stream))
    continuous_cli._prepare_job(worker, job)
    first = worker._continuous_graph_stream
    continuous_cli._prepare_job(worker, dict(job, seed=1))
    assert created == [('cuda:2', first)]
    assert worker._continuous_graph_stream is first


def test_missing_plot_dependency_fails_before_input_admission(monkeypatch):
    args = parser().parse_args(['--variable-count', '100', '--dense'])
    monkeypatch.setattr(continuous_cli, 'find_spec', lambda name: None)
    with pytest.raises(ValueError, match='matplotlib.*--no-plots'):
        continuous_cli.prepare(args)


def test_no_plots_does_not_require_plot_dependency(monkeypatch):
    from qubo_benchmark import continuous_catalog
    args = parser().parse_args(['--variable-count', '100', '--dense', '--no-plots'])

    def unexpected_dependency_probe(name):
        raise AssertionError('No plotting dependency is needed for --no-plots')

    def input_admission(path):
        raise RuntimeError('Input admission reached')

    monkeypatch.setattr(continuous_cli, 'find_spec', unexpected_dependency_probe)
    monkeypatch.setattr(continuous_catalog, 'load_continuous_catalog', input_admission)
    with pytest.raises(RuntimeError, match='Input admission reached'):
        continuous_cli.prepare(args)


def test_input_change_during_setup_leaves_seed_fatal_and_pending(tmp_path, monkeypatch):
    from qubo_benchmark.runtime import worker as worker_module
    job=fake_job(tmp_path)
    built=fake_setup(monkeypatch)
    worker=FakeWarmWorker(lambda: (tmp_path/'input.npz').write_bytes(b'changed input'))
    monkeypatch.setattr(worker_module,'WarmWorker',lambda job: worker)
    invoked=[]
    monkeypatch.setattr(continuous_cli,'run_continuous_trial',lambda *args: invoked.append(True))
    incoming,outgoing=queue.Queue(),queue.Queue()
    incoming.put(job)
    incoming.put(None)
    continuous_cli._worker(0,'cpu',incoming,outgoing)
    messages=[]
    while not outgoing.empty():
        messages.append(outgoing.get_nowait())
    assert [message['kind'] for message in messages]==['ready','started','fatal']
    assert 'InputIdentityError' in messages[-1]['error']
    assert invoked==[] and built==[]


def test_prepared_signal_follows_all_immutable_setup_and_precedes_solve(tmp_path, monkeypatch):
    from qubo_benchmark.runtime import worker as worker_module
    job=fake_job(tmp_path)
    built=fake_setup(monkeypatch)
    worker=FakeWarmWorker()
    monkeypatch.setattr(worker_module,'WarmWorker',lambda job: worker)
    incoming,outgoing=queue.Queue(),queue.Queue()
    incoming.put(job)
    incoming.put(None)
    def solve(instance,trial):
        assert instance is worker and len(built)==1
        assert outgoing.queue[-1]['kind']=='prepared'
        return dict(status='complete',energies=[0.]*9)
    monkeypatch.setattr(continuous_cli,'run_continuous_trial',solve)
    continuous_cli._worker(0,'cpu',incoming,outgoing)
    messages=[]
    while not outgoing.empty():
        messages.append(outgoing.get_nowait())
    assert [message['kind'] for message in messages]==['ready','started','prepared','done']


def test_prepared_signal_resets_separate_setup_and_solve_watchdogs():
    class Process:
        def is_alive(self):return True
    lanes=[(Process(),None,'cpu')]
    active={0:dict(case=('lib_random_search','example',7),phase='setup',started=0.)}
    continuous_cli._check_watchdogs(active,lanes,119.)
    with pytest.raises(RuntimeError,match='setup watchdog'):
        continuous_cli._check_watchdogs(active,lanes,121.)
    message=dict(worker=0,solver='lib_random_search',seed=7)
    continuous_cli._mark_prepared(active,message,119.)
    continuous_cli._check_watchdogs(active,lanes,143.)
    with pytest.raises(RuntimeError,match='solve watchdog'):
        continuous_cli._check_watchdogs(active,lanes,145.)
    with pytest.raises(RuntimeError,match='unexpected continuous seed'):
        continuous_cli._mark_prepared(active,dict(message,seed=8),146.)


def test_old_independent_operator_is_released_before_cache_miss_setup(tmp_path, monkeypatch):
    job=fake_job(tmp_path)
    fake_setup(monkeypatch)
    worker=FakeWarmWorker()
    worker._continuous_scorer=object()
    worker._continuous_score_key='old large operator'
    original=worker.prepare
    def checked(trial):
        assert worker._continuous_scorer is None
        assert worker._continuous_score_key is None
        original(trial)
    worker.prepare=checked
    continuous_cli._prepare_job(worker,job)
    assert worker._continuous_scorer is not None
    assert worker._continuous_input_sha256==job['input_sha256']


@pytest.mark.parametrize('kill_succeeds', [True,False])
def test_worker_shutdown_escalates_from_terminate_to_owned_process_kill(kill_succeeds):
    class Process:
        pid=1234
        alive=True
        calls=[]
        def join(self,timeout):self.calls.append(('join',timeout))
        def is_alive(self):return self.alive
        def terminate(self):self.calls.append(('terminate',))
        def kill(self):
            self.calls.append(('kill',))
            self.alive=not kill_succeeds
    process=Process()
    if kill_succeeds:
        continuous_cli._stop_process(process)
        assert not process.is_alive()
    else:
        with pytest.raises(RuntimeError,match='PID 1234 survived shutdown'):
            continuous_cli._stop_process(process)
    assert process.calls==[('join',2.),('terminate',),('join',5.),('kill',),('join',2.)]

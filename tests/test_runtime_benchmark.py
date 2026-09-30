"""Runtime protocol contracts, controlled timing, persistence and real CPU/GPU seams."""
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from qubo_benchmark.model import Problem
from qubo_benchmark.runtime.common import read,digest,atomic,CONFIG
from qubo_benchmark.runtime.selection import select,seeds,budget,grid,REPRESENTATIVES,inventory,environment,preflight
from qubo_benchmark.runtime.metrics import quality,evaluate
from qubo_benchmark.runtime.worker import Observer
from qubo_solvers.observation import SolveInterrupted,observing,current
from qubo_benchmark.runtime import engine,storage,supervisor
from qubo_benchmark.runtime.cli import parser,resolve

def objective():
    return Problem(2,np.array([0,1]),np.array([0,1]),np.array([-10,-1]),0)

def event(bits,elapsed,phase='checkpoint',raw=999):
    return dict(bitstring=bits,elapsed_s=elapsed,phase=phase,raw=raw,iteration=1)

def test_all_selection_and_frozen_protocol():
    counts={(200,'sparse'):3,(200,'dense'):10,(500,'sparse'):10,(500,'dense'):2,(1000,'sparse'):10,(1000,'dense'):2}
    for group,count in counts.items():
        assert len(select(*group))==count
        assert select(*group,'representative')[0]['instance_id']==REPRESENTATIVES[f'{group[0]}_{group[1]}']
    assert len(grid())==36 and len(set(grid()))==36
    assert seeds()==list(range(100))
    assert budget('1')==budget('1.0')==1.
    for t in (.05,.1,.5,1.,5.,10.):assert budget(str(t))==t
    for t in ('0.01','nan','inf','-1','abc'):
        with pytest.raises(ValueError):budget(t)
    for spec in ((10,'sparse'),(200,'medium')):
        with pytest.raises(ValueError):select(*spec)
    with pytest.raises(ValueError):select(200,'sparse',instance='bqp500-1')
    settings=read(CONFIG/'benchmark_solvers.json')['solvers'];before=digest(settings)
    assert len(settings)==23
    for n,d,t in grid():
        for p in settings.values():assert 'seed' not in p and 'time_limit' not in p
    assert digest(settings)==before
    rows=inventory(settings)
    assert sum(r['status']=='runnable' for r in rows)==23
    assert sum(r['status']=='excluded' for r in rows)==5
    missing=dict(settings);missing.pop(next(iter(missing)))
    assert sum(r['status']=='unavailable' for r in inventory(missing))==1

def test_seed_file_validation(tmp_path):
    path=tmp_path/'seeds.json'
    for values in ([0,0],[0],[True,1],[-1,1],[0,2**63]):
        atomic(path,values)
        with pytest.raises(ValueError):seeds(2,path=path)
    atomic(path,[17,21]);assert seeds(2,path=path)==[17,21]

@pytest.mark.parametrize('value,gap,success',[(-1010,-1.,True),(-990,1.,True),(-989,1.1,False)])
def test_directional_gap(value,gap,success):
    q=quality(value,-1000)
    assert q['signed_gap_percent']==gap and q['success_1pct']==success

def test_zero_integer_and_alarm():
    assert quality(0,0)['signed_gap_percent'] is None
    assert quality(1,0)['success_1pct'] is False
    assert quality(-10,-10)['near_target_objective']==-9.9
    assert quality(-9,-10)['success_1pct'] is False
    alarm=quality(-11,-10,True)
    assert alarm['optimum_validation_alarm'] and not alarm['success_1pct']
    assert quality(-10,-10,True)['exact_optimum_reached']

def test_hit_then_improve_keeps_search_trace_and_late_candidate_separate():
    result,solutions,trace=evaluate(objective(),[event('00',0,'initial'),event('10',.01),
        event('11',.03),event('11',.06,'final_returned')],.05,-10,'best_published')
    assert result['best_objective_within_budget']==-11
    assert result['ttt_1pct_s']==.01 and result['time_to_best_s']==.03
    assert result['final_returned_objective']==-11 and not result['final_returned_within_budget']
    assert result['solver_reported_raw_min']==999
    assert [r['independently_validated_best_so_far'] for r in trace]==[0,-10,-11,-11]
    assert next(s for s in solutions if s['solution_id']==result['best_solution_id'])['bitstring']=='11'

@pytest.mark.parametrize('events,error,expected',[
    ([event('01',.01)],None,'completed'),
    ([event('10',.06)],None,'no_in_budget_candidate'),
    ([event('0.5',.01)],None,'invalid_output'),
    ([event('1',.01)],None,'invalid_output'),
    ([event('10',.01)],'worker_crash','worker_crash'),
    ([event('10',.01)],'oom','oom'),
    ([event('10',.01)],'watchdog_timeout','watchdog_timeout'),
    ([],None,'no_in_budget_candidate')])
def test_failure_unknown_censoring_categories(events,error,expected):
    result,_,_=evaluate(objective(),events,.05,-10,'best_published',error)
    assert result['result_status']==expected
    if error:assert result['ttt_1pct_s'] is None and not result['success_1pct']

def test_no_history_does_not_invent_first_hit_and_proven_alarm_quarantined():
    result,_,_=evaluate(objective(),[event('10',.02,'final_returned')],.05,-10,'published_proven_optimum')
    assert result['success_reference'] and result['exact_optimum_reached']
    assert not result['history_available'] and result['ttt_reference_s'] is None
    assert result['reference_time_status']=='history_unavailable'
    result,_,_=evaluate(objective(),[event('11',.02)],.05,-10,'published_proven_optimum')
    assert result['optimum_validation_alarm'] and not result['success_reference']
    with pytest.raises(ValueError,match='Nonmonotonic'):
        evaluate(objective(),[event('00',.02),event('10',.01)],.05,-10,'best_published')

def test_capture_timestamp_after_immutable_copy_and_clean_scope():
    clock=[0];messages=[]
    observer=Observer(.05,messages.append,clock=lambda:clock[0])
    tensor=torch.tensor([[0.,1.]])
    with observing(observer):
        assert current() is observer
        clock[0]=10_000_000
        observer.capture(tensor,phase='initial')
        tensor.fill_(1)
        assert messages[0]['events'][0]['bitstring']=='01'
        clock[0]=51_000_000
        with pytest.raises(SolveInterrupted):observer.poll()
        # Actual final returns are preserved as diagnostics even when late.
        observer.capture(tensor,phase='final_returned')
    assert current() is None
    assert messages[-1]['events'][0]['elapsed_s']==.051
    class Slow:
        def detach(self):return self
        def clone(self):return self
        def cpu(self):clock[0]=70_000_000;return self
        def tolist(self):return [[0,0]]
    clock[0]=0;late=[];obs=Observer(.05,late.append,clock=lambda:clock[0])
    with pytest.raises(SolveInterrupted):obs.capture(Slow())
    assert late[0]['events'][0]['elapsed_s']==.07

def args_for(root):
    return resolve(parser().parse_args(['200','sparse','0.05','--instances','representative',
        '--runs','2','--device','cpu','--solvers','lib_greedy_local_search','--output-root',str(root)]))

def fake_trial(job,heartbeat,stopped):
    heartbeat('solving')
    return dict(events=[event('0'*200,.01,'initial'),event('0'*200,.02,'final_returned')],
        done=dict(actual_solve_wall_s=.02,stop_reason='natural_return',initialization_hash=str(job['seed'])),
        setup=dict(worker_ready_wall_s=.001,transfer_s=.0001),started_at_utc='2026-09-30T00:00:00+00:00',
        finished_at_utc='2026-09-30T00:00:01+00:00',cleanup_s=.001,worker_end_to_end_s=.025,
        cpu_rss_start_bytes=100,cpu_rss_peak_sampled_bytes=120,abort_gpu=False)

def test_durable_resume_csv_repair_retries_identity_and_partial_denominators(tmp_path,monkeypatch):
    args=args_for(tmp_path);prepared=preflight(args);calls=[]
    def run(*a):calls.append(a[0]['seed']);return fake_trial(*a)
    monkeypatch.setattr(engine,'trial',run)
    directory,code=engine.execute(prepared,args)
    assert code==0 and calls==[0,1]
    exp=read(directory/'experiment.json');attempts=storage.load_attempts(directory)
    rows=storage.summaries(exp,attempts[:1])
    assert rows[0]['success_denominator']==2 and rows[0]['success_rate'] is None
    own=directory/'solvers/lib_greedy_local_search'
    with (own/'solutions.csv').open(newline='') as f:
        bits=list(csv.DictReader(f))[0]['bitstring']
    assert bits=='0'*200
    assert set(storage.RUN_FIELDS)==set(attempts[0]['run'])
    # Restore a truncated export from authoritative checksummed attempts on resume.
    (own/'runs.csv').write_text('broken,truncation')
    resume=resolve(parser().parse_args(['--resume',str(directory)]))
    same=preflight(resume);assert same['identity']==prepared['identity']
    engine.execute(same,resume);assert calls==[0,1]
    assert len(list(csv.DictReader((own/'runs.csv').open())))==2
    changed=dict(same,identity='changed')
    with pytest.raises(ValueError,match='identity mismatch'):engine.execute(changed,resume)
    with pytest.raises(ValueError,match='already committed'):storage.commit(directory,attempts[0])
    broken=read(next((directory/'attempts').glob('*.json')));broken['payload']['run']['seed']=999
    atomic(next((directory/'attempts').glob('*.json')),broken)
    with pytest.raises(ValueError,match='Corrupt'):storage.load_attempts(directory)

def test_retry_keeps_failure_primary_and_interrupted_run_pending(tmp_path,monkeypatch):
    args=args_for(tmp_path);args.runs=1;prepared=preflight(args)
    def failure(*a):
        outcome=fake_trial(*a);outcome['done']['error']='oom';return outcome
    monkeypatch.setattr(engine,'trial',failure)
    directory,code=engine.execute(prepared,args);assert code==1
    resume=resolve(parser().parse_args(['--resume',str(directory),'--retry-failed']))
    monkeypatch.setattr(engine,'trial',fake_trial)
    _,code=engine.execute(preflight(resume),resume)
    attempts=storage.load_attempts(directory);assert len(attempts)==2 and code==1
    assert list(storage.primary(attempts).values())[0]['run']['result_status']=='oom'
    attempts[0]['run']['result_status']='interrupted'
    assert list(storage.primary(attempts).values())[0]['attempt_number']==2

def ignoring_worker(connection,stop,job):
    connection.send(dict(kind='ready',setup={}))
    connection.recv()
    connection.send(dict(kind='started',origin_ns=time.perf_counter_ns(),started_at_utc='test'))
    connection.send(dict(kind='events',events=[event('00',.001,'initial')]))
    time.sleep(10)

def test_watchdog_ignores_uncooperative_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,'execute',ignoring_worker)
    job=dict(device='cpu',budget=.05,watchdog_grace_s=.03,setup_timeout_s=20,
             memory_sampling_interval_s=.01,heartbeat_interval_s=1.)
    result=supervisor.trial(job,lambda phase:None,lambda:False)
    assert result['done']['error']=='watchdog_timeout'
    assert result['events'][0]['bitstring']=='00'
    assert not result['abort_gpu']

@pytest.mark.parametrize('device',['cpu',pytest.param('cuda:0',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA hardware unavailable; no timing claim'))])
def test_real_ready_worker_reset_and_device_metrics(tmp_path,device):
    path=tmp_path/'tiny.npz';objective().save(path)
    p=read(CONFIG/'benchmark_solvers.json')['solvers']['lib_random_search'].copy()
    p['max_steps']=1
    base=dict(solver='lib_random_search',seed=7,parameters=p,npz=str(path),device=device,
         cpu_threads=1,watchdog_grace_s=.5,setup_timeout_s=30,memory_sampling_interval_s=.01,heartbeat_interval_s=1.)
    results=[supervisor.trial(dict(base,budget=t),lambda phase:None,lambda:False) for t in (.5,1.)]
    assert all(r['done'].get('error') is None for r in results)
    assert results[0]['done']['initialization_hash']==results[1]['done']['initialization_hash']
    assert results[0]['events'][0]['phase']=='initial'
    assert results[0]['setup']['worker_ready_wall_s']>0
    if device.startswith('cuda'):
        assert results[0]['done']['gpu_allocated_peak_bytes']>=results[0]['done']['gpu_allocated_start_bytes']

def test_require_gpu_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError):environment('cuda:0')
    with pytest.raises(ValueError):environment('auto',True)

def test_hook_does_not_change_math_without_deadline():
    from qubo_solvers import QUBO,create_solver,LIBRARY_SOLVERS,default_algorithm_parameters
    problem=QUBO(torch.tensor([[-1.,.25],[.25,-2.]]))
    for name in LIBRARY_SOLVERS:
        solver=create_solver(name,**default_algorithm_parameters(name,steps=3))
        original=solver.solve(problem,restarts=2,seed=7)
        observer=Observer(100,lambda message:None)
        with observing(observer):observed=solver.solve(problem,restarts=2,seed=7)
        assert torch.equal(original.best_assignments,observed.best_assignments),name
        assert torch.equal(original.best_energies,observed.best_energies),name
def test_interruption_resume_restarts_original_seed_and_uses_new_attempt(tmp_path,monkeypatch):
    args=args_for(tmp_path);prepared=preflight(args)
    def interrupted(*a):
        result=fake_trial(*a);result['done']['error']='interrupted';return result
    monkeypatch.setattr(engine,'trial',interrupted)
    directory,code=engine.execute(prepared,args)
    assert code==130 and read(directory/'status.json')['pending']==2
    resume=resolve(parser().parse_args(['--resume',str(directory)]))
    called=[]
    def complete(*a):called.append(a[0]['seed']);return fake_trial(*a)
    monkeypatch.setattr(engine,'trial',complete)
    _,code=engine.execute(preflight(resume),resume)
    assert code==0 and called==[0,1]
    attempts=storage.load_attempts(directory)
    assert len(attempts)==3 and len(storage.primary(attempts))==2
    assert attempts[0]['run']['result_status']=='interrupted'
    assert attempts[1]['attempt_number']==2

def test_live_lock_and_resume_policy_mismatches(tmp_path):
    from qubo_benchmark.runtime.common import DirectoryLock
    with DirectoryLock(tmp_path):
        with pytest.raises(ValueError,match='locked'):
            with DirectoryLock(tmp_path):pass
    args=args_for(tmp_path)
    prepared=preflight(args)
    for key,value in [('time_limit','.1'),('runs',3),('seed_start',1)]:
        other=SimpleNamespace(**vars(args));setattr(other,key,value)
        assert preflight(other)['identity']!=prepared['identity']

def test_monitor_and_aggregator_preserve_failures_and_quoted_solutions(tmp_path,monkeypatch,capsys):
    from qubo_benchmark.runtime.monitor import main as monitor
    from qubo_benchmark.runtime.aggregate import main as aggregate
    args=args_for(tmp_path/'results');args.runs=1
    monkeypatch.setattr(engine,'trial',fake_trial)
    directory,_=engine.execute(preflight(args),args)
    assert monitor([str(directory),'--once'])==0
    out=tmp_path/'analysis'
    assert aggregate([str(tmp_path/'results'),'--output',str(out)])==0
    assert read(out/'aggregation.json')['experiments']==1
    assert next(csv.DictReader((out/'solutions.csv').open()))['bitstring']=='0'*200
    assert next(csv.DictReader((out/'summary.csv').open()))['planned']=='1'

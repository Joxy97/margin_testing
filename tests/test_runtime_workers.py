"""Bounded checks for reusable workers, parallel seed isolation and solve-only clocks."""
import os
import hashlib
import time
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from qubo_benchmark.model import Problem
from qubo_benchmark.runtime import pool,engine,storage
from qubo_benchmark.runtime.common import CONFIG,read
from qubo_benchmark.runtime.selection import preflight,select_solvers,worker_capacity,PROTOCOL
from qubo_benchmark.runtime.cli import parser,resolve
from qubo_benchmark.runtime.worker import Observer


def controlled_server(connection,stop):
    count=0
    try:
        while True:
            command=connection.recv()
            if command['action']=='shutdown':return
            job=command['job']
            if not count:time.sleep(.15)  # Setup exceeds the 50 ms budget.
            connection.send(dict(kind='ready',setup=dict(worker_pid=os.getpid(),worker_reused=count>0)))
            if connection.recv()['action']=='shutdown':return
            origin=time.perf_counter_ns()
            connection.send(dict(kind='started',origin_ns=origin,started_at_utc='controlled'))
            if job.get('hang'):time.sleep(10)
            if job.get('fail'):
                connection.send(dict(kind='fatal',error_message='controlled crash'));return
            time.sleep(.025)
            elapsed=(time.perf_counter_ns()-origin)/1e9
            connection.send(dict(kind='events',events=[dict(bitstring=str(job['seed']%2)*2,
                elapsed_s=elapsed,phase='initial',raw=None,iteration=0)]))
            if job.get('slow_cleanup'):
                connection.send(dict(kind='solved',actual_solve_wall_s=elapsed,finished_at_utc='controlled'))
                time.sleep(.25)
            count+=1
            connection.send(dict(kind='done',actual_solve_wall_s=elapsed,error=None,
                origin_ns=origin,worker_trial_index=count,stop_reason='natural_return'))
    finally:connection.close()


def base_job(**overrides):
    return dict(dict(device='cpu',budget=.05,watchdog_grace_s=.1,setup_timeout_s=30,
        memory_sampling_interval_s=.01,heartbeat_interval_s=1.,seed=0),**overrides)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.int8,torch.bool])
def test_vectorized_capture_preserves_bits_and_initialization_hash(dtype):
    samples=torch.tensor([[0,1,0],[1,0,1]],dtype=dtype)
    messages=[];observer=Observer(10,messages.append)
    observer.capture(samples,phase='initial')
    expected=hashlib.sha256(repr(samples.tolist()).encode()).hexdigest()
    assert observer.initialization_hash==expected
    samples.fill_(0)
    assert [e['bitstring'] for e in messages[0]['events']]==['010','101']


def test_vectorized_capture_keeps_invalid_values_visible():
    messages=[];observer=Observer(10,messages.append)
    observer.capture(torch.tensor([[.5,1.],[float('nan'),0.]]))
    assert [e['bitstring'] for e in messages[0]['events']]==['[0.5, 1.0]','[nan, 0.0]']


def test_parallel_readiness_and_setup_outside_budget(monkeypatch):
    monkeypatch.setattr(pool,'serve',controlled_server)
    workers=pool.SeedPool(2,'persistent',lambda:False)
    try:
        results=dict(workers.wave([base_job(seed=0),base_job(seed=1)],lambda _:None))
        assert all(r['done'].get('error') is None for r in results.values()), results
        assert all(r['setup']['worker_ready_wall_s']>.15 for r in results.values())
        assert all(r['done']['actual_solve_wall_s']<.05 for r in results.values())
        assert results[0]['events'][0]['bitstring']=='00'
        assert results[1]['events'][0]['bitstring']=='11'
        starts=[results[i]['done']['origin_ns']/1e9 for i in range(2)]
        ends=[starts[i]+results[i]['done']['actual_solve_wall_s'] for i in range(2)]
        assert max(starts)<min(ends), 'Seed solve intervals must overlap'
        again=dict(workers.wave([base_job(seed=2),base_job(seed=3)],lambda _:None))
        for i in range(2):
            assert again[i]['setup']['worker_pid']==results[i]['setup']['worker_pid']
            assert again[i]['setup']['worker_reused']
        cleanup=dict(workers.wave([base_job(slow_cleanup=True)],lambda _:None))[0]
        assert cleanup['done']['error'] is None
        assert cleanup['done']['actual_solve_wall_s']<.05
        assert cleanup['worker_end_to_end_s']>.25
    finally:lifetimes=workers.close()
    assert len(lifetimes)==2 and all(r['exit_code']==0 for r in lifetimes)


def test_pool_watchdog_and_fatal_gpu_abort(monkeypatch):
    monkeypatch.setattr(pool,'serve',controlled_server)
    workers=pool.SeedPool(1,'persistent',lambda:False)
    try:
        result=dict(workers.wave([base_job(hang=True)],lambda _:None))[0]
        assert result['done']['error']=='watchdog_timeout'
        assert not result['abort_gpu'] and workers.slots[0].process is None
        # Controlled IPC-only CUDA job exercises failure policy without GPU hardware.
        result=dict(workers.wave([base_job(device='cuda:0',fail=True)],lambda _:None))[0]
        assert result['abort_gpu'] and workers.abort.is_set()
    finally:workers.close()


def test_buffered_completion_is_drained_before_watchdog(monkeypatch):
    """Parent scheduling delays must not turn a completed seed into GPU loss."""
    monkeypatch.setattr(pool,'serve',controlled_server)
    workers=pool.SeedPool(1,'persistent',lambda:False)
    slot=workers.slots[0];receive=slot.receive
    def delayed_receive():
        message=receive()
        if message is not None and message['kind']=='started':
            # Child completes in 25 ms; parent resumes after the 150 ms
            # watchdog deadline with events and done already waiting in IPC.
            time.sleep(.25)
        return message
    monkeypatch.setattr(slot,'receive',delayed_receive)
    try:
        result=dict(workers.wave([base_job(device='cuda:0')],lambda _:None))[0]
        assert result['done']['error'] is None
        assert result['done']['actual_solve_wall_s']<.05
        assert result['events'][0]['bitstring']=='00'
        assert not result['abort_gpu']
        assert result['worker_end_to_end_s']>.25
    finally:workers.close()


def test_slot_runs_jobs_without_a_wave_barrier(monkeypatch):
    monkeypatch.setattr(pool,'serve',controlled_server)
    workers=pool.SeedPool(1,'persistent',lambda:False)
    try:
        first=workers.slots[0].run_job(base_job(seed=0))
        second=workers.slots[0].run_job(base_job(seed=1))
        assert first['done']['error'] is None and second['done']['error'] is None
        assert first['setup']['worker_pid']==second['setup']['worker_pid']
        assert second['setup']['worker_reused']
        assert first['events'][0]['bitstring']!=second['events'][0]['bitstring']
    finally:workers.close()


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda:0',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA hardware unavailable; no GPU performance claim'))])
def test_real_reuse_resets_rng_and_retains_only_immutable_input(tmp_path,device):
    path=tmp_path/'tiny.npz'
    Problem(12,np.arange(12),np.arange(12),-np.arange(1,13),0).save(path)
    parameters=read(CONFIG/'benchmark_solvers.json')['solvers']['lib_random_search'].copy()
    parameters['max_steps']=1
    job=base_job(device=device,budget=.5,npz=str(path),parameters=parameters,
                 solver='lib_random_search',cpu_threads=1)
    workers=pool.SeedPool(1,'persistent',lambda:False)
    try:
        results=[dict(workers.wave([dict(job,seed=s)],lambda _:None))[0] for s in (7,8,7)]
        assert all(r['done'].get('error') is None for r in results)
        assert len({r['setup']['worker_id'] for r in results})==1
        assert [r['setup']['input_cache_hit'] for r in results]==[False,True,True]
        assert [r['done']['worker_trial_index'] for r in results]==[1,2,3]
        assert results[0]['done']['initialization_hash']==results[2]['done']['initialization_hash']
        assert results[0]['done']['initialization_hash']!=results[1]['done']['initialization_hash']
        assert [r['bitstring'] for r in results[0]['events']]==[r['bitstring'] for r in results[2]['events']]
        if device.startswith('cuda'):
            assert all(r['done']['gpu_allocated_peak_bytes']>=r['done']['gpu_allocated_start_bytes'] for r in results)
    finally:workers.close()


def test_execution_policy_frozen_and_resume_restores_workers(tmp_path,monkeypatch):
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(available=64*2**30,total=64*2**30))
    args=resolve(parser().parse_args(['200','sparse','.05','--instances','representative','--runs','2',
        '--solvers','lib_random_search','--device','cpu','--seed-workers','2']))
    prepared=preflight(args);directory,experiment=engine.initialize(prepared,tmp_path)
    resumed=resolve(parser().parse_args(['--resume',str(directory)]))
    assert resumed.seed_workers==2 and resumed.worker_mode=='persistent'
    assert preflight(resumed)['identity']==prepared['identity']
    args.seed_workers=1;isolated=preflight(args)
    assert isolated['identity']!=prepared['identity']
    assert isolated['core']['solvers']==prepared['core']['solvers']
    summary=storage.summaries(experiment,[])[0]
    assert summary['seed_workers']==2 and summary['execution_mode']=='parallel_shared_device'
    for count in (0,101):
        args.seed_workers=count
        with pytest.raises(ValueError,match='1 through 100'):preflight(args)


def test_capacity_is_memory_estimate_and_empty_inventory_reports_backend_error(monkeypatch):
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(available=8*2**30))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(4*2**30,4*2**30))
    cap=worker_capacity([],{},'cuda:0',PROTOCOL['execution'])
    assert cap['maximum_workers_estimate']==3
    rows=[dict(solver_id='lib_random_search',status='unavailable',reason="No module named 'scipy'")]
    with pytest.raises(ValueError,match="No eligible runnable solvers[\\s\\S]*scipy"):
        select_solvers(rows)
    rows[0]['status']='runnable'
    assert select_solvers(rows)==['lib_random_search']
    with pytest.raises(ValueError,match='unique'):select_solvers(rows,'lib_random_search, lib_random_search')

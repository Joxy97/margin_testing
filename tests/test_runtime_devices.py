"""Device discovery, aggregate admission, container limits and frozen resume."""
import copy
from types import SimpleNamespace

import pytest
import torch

from qubo_benchmark.runtime import selection
from qubo_benchmark.runtime.cli import parser,resolve
from qubo_benchmark.runtime.common import atomic,read


def fake_cuda(monkeypatch,count=4):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:count>0)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:count)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda index:SimpleNamespace(
        name='RTX fixture',uuid=f'GPU-{index}',total_memory=24*2**30))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(20*2**30,24*2**30))


def test_auto_uses_all_visible_devices_and_explicit_selection_is_strict(monkeypatch):
    fake_cuda(monkeypatch)
    assert selection.device_selection()==['cuda:0','cuda:1','cuda:2','cuda:3']
    assert selection.device_selection('cuda:2')==['cuda:2']
    assert selection.device_selection('cuda:1,cuda:3')==['cuda:1','cuda:3']
    for devices in ('cuda:0,cuda:0','cuda:0,cuda:00','cuda:4','cuda:-1','cuda:0,cpu','cuda:0,'):
        with pytest.raises(ValueError):selection.device_selection(devices)
    fake_cuda(monkeypatch,0)
    assert selection.device_selection()==['cpu']
    with pytest.raises(ValueError,match='unavailable'):selection.device_selection('cuda:0')
    with pytest.raises(ValueError,match='require-gpu'):selection.environment('auto',True)


def test_hardware_identity_records_every_selected_gpu(monkeypatch):
    fake_cuda(monkeypatch)
    monkeypatch.setattr(selection.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='fixture-driver'))
    env=selection.environment()
    assert len(env['hardware']['gpus'])==4
    assert [gpu['uuid'] for gpu in env['hardware']['gpus']]==[f'GPU-{i}' for i in range(4)]
    assert env['hardware']['device']=='cuda:0'
    single=selection.environment('cuda:0')
    assert single['hardware_id']!=env['hardware_id']


def test_cgroup_v2_cpu_and_memory_respect_parent_limits(tmp_path,monkeypatch):
    root=tmp_path/'cgroup';child=root/'parent'/'job';child.mkdir(parents=True)
    membership=tmp_path/'self.cgroup';membership.write_text('0::/parent/job\n')
    (root/'cpu.max').write_text('max 100000')
    (child.parent/'cpu.max').write_text('250000 100000')
    (child/'cpu.max').write_text('400000 100000')
    (child.parent/'memory.max').write_text(str(8*2**30))
    (child.parent/'memory.current').write_text(str(3*2**30))
    monkeypatch.setattr(selection.os,'cpu_count',lambda:64)
    monkeypatch.setattr('psutil.Process',lambda:SimpleNamespace(cpu_affinity=lambda:list(range(32))))
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(available=128*2**30))
    cap=selection.cpu_capacity(root,membership)
    assert cap['effective_cpus']==cap['cgroup_cpu_quota']==2.5
    assert selection.host_memory_available(root,membership)==5*2**30


def test_cgroup_v1_quotas_and_memory(tmp_path,monkeypatch):
    root=tmp_path/'cgroup';cpu=root/'cpu';memory=root/'memory'
    cpu.mkdir(parents=True);memory.mkdir()
    membership=tmp_path/'self.cgroup';membership.write_text('2:cpu,cpuacct:/\n3:memory:/\n')
    (cpu/'cpu.cfs_quota_us').write_text('175000')
    (cpu/'cpu.cfs_period_us').write_text('100000')
    (memory/'memory.limit_in_bytes').write_text('1000000')
    (memory/'memory.usage_in_bytes').write_text('750000')
    monkeypatch.setattr(selection.os,'cpu_count',lambda:64)
    monkeypatch.setattr('psutil.Process',lambda:SimpleNamespace(cpu_affinity=lambda:list(range(32))))
    monkeypatch.setattr('psutil.virtual_memory',lambda:SimpleNamespace(available=128*2**30))
    assert selection.cpu_capacity(root,membership)['effective_cpus']==1.75
    assert selection.host_memory_available(root,membership)==250000


def test_admission_accounts_for_host_memory_shared_by_all_gpus(monkeypatch):
    fake_cuda(monkeypatch)
    monkeypatch.setattr(selection,'host_memory_available',lambda:10*2**30)
    monkeypatch.setattr(selection,'cpu_capacity',lambda:dict(effective_cpus=40.96))
    cap=selection.worker_capacity([],{},[f'cuda:{i}' for i in range(4)],selection.PROTOCOL['execution'])
    assert cap['maximum_workers_estimate']==8
    assert cap['cpu_worker_limit']==40
    assert cap['workers_per_device']['cuda:0']==8
    # Four independent per-device checks would otherwise incorrectly admit 32 workers.
    prepared=dict(core=dict(actual_devices=list(cap['workers_per_device']),seeds=list(range(100)),
        protocol=dict(execution={}),source=dict(code_commit='fixture',code_snapshot_hash='fixture')),capacity=cap)
    with pytest.raises(ValueError,match='aggregate'):
        selection.finalize_execution(prepared,{device:4 for device in cap['workers_per_device']})
    selection.finalize_execution(prepared,{device:2 for device in cap['workers_per_device']})
    assert prepared['core']['protocol']['execution']['seed_workers']==8
    assert prepared['core']['protocol']['execution']['execution_mode']=='parallel_shared_device'


def test_cli_defaults_all_devices_auto_workers_and_explicit_subset():
    args=resolve(parser().parse_args(['200','sparse','1']))
    assert args.device=='auto' and args.seed_workers=='auto' and args.max_auto_workers==4
    args=resolve(parser().parse_args(['200','sparse','1','--devices','cuda:0,cuda:2','--seed-workers','3']))
    assert args.device=='cuda:0,cuda:2' and args.seed_workers==3
    with pytest.raises(SystemExit):parser().parse_args(['--seed-workers','0'])


def test_tuned_counts_are_identity_but_measurements_are_not(tmp_path,monkeypatch):
    monkeypatch.setattr(selection,'host_memory_available',lambda:64*2**30)
    monkeypatch.setattr(selection,'cpu_capacity',lambda:dict(logical_cpus=8,cpu_affinity=list(range(8)),
                                                          cgroup_cpu_quota=None,effective_cpus=8.))
    args=resolve(parser().parse_args(['200','sparse','.05','--device','cpu','--runs','3',
        '--solvers','lib_random_search','--instances','representative']))
    prepared=selection.preflight(args)
    selection.finalize_execution(prepared,{'cpu':2},{'measurement_s':12})
    first=prepared['identity']
    selection.finalize_execution(prepared,{'cpu':2},{'measurement_s':21})
    assert prepared['identity']==first
    root=tmp_path/'result';root.mkdir()
    atomic(root/'experiment.json',dict(core=prepared['core']))
    atomic(root/'seeds.json',prepared['core']['seeds'])
    atomic(root/'solver_configuration.json',prepared['configuration'])
    resume=resolve(parser().parse_args(['--resume',str(root)]))
    assert resume.seed_workers=='auto'
    resumed=selection.preflight(resume)
    assert resumed['core']['protocol']['execution']['workers_per_device']=={'cpu':2}
    assert resumed['identity']==first
    resume.seed_workers=1
    assert selection.preflight(resume)['identity']!=first


def test_distinct_gpus_without_sharing_have_separate_timing_label():
    prepared=dict(core=dict(actual_devices=['cuda:0','cuda:1'],seeds=list(range(100)),
        protocol=dict(execution={}),source=dict(code_commit='fixture',code_snapshot_hash='fixture')),
        capacity=dict(workers_per_device={'cuda:0':4,'cuda:1':4},maximum_workers_estimate=8))
    selection.finalize_execution(prepared,{'cuda:0':1,'cuda:1':1})
    assert prepared['core']['protocol']['execution']['execution_mode']=='multi_gpu_isolated'


def calibration_fixture(monkeypatch,durations,fail_at=None,incomplete_at=None):
    from qubo_benchmark.runtime import autotune
    now=[0.];pools=[]
    class ControlledPool:
        def __init__(self,size,mode,interrupted):
            self.size=size;self.closed=False;self.jobs=[];pools.append(self)
            assert mode=='persistent'
        def wave(self,jobs,heartbeat):
            self.jobs.extend(jobs)
            count=self.size//2
            now[0]+=durations[count]
            returned=jobs[:-1] if count==incomplete_at else jobs
            for i,job in enumerate(returned):
                yield i,dict(done=dict(error='watchdog_timeout' if count==fail_at else None,
                    actual_solve_wall_s=job['budget'],gpu_reserved_peak_bytes=1234))
        def close(self):self.closed=True
    monkeypatch.setattr(autotune,'SeedPool',ControlledPool)
    monkeypatch.setattr(autotune.time,'perf_counter',lambda:now[0])
    devices=['cuda:0','cuda:1']
    execution=copy.deepcopy(selection.PROTOCOL['execution'])
    execution.update(autotune_enabled=True,seed_workers_requested='auto')
    prepared=dict(core=dict(actual_devices=devices,actual_device=devices[0],seeds=list(range(8)),budget_s=1.,
        solvers={'lib_greedy_local_search':{'steps':123},'lib_simulated_bifurcation':{'steps':321}},
        protocol=dict(execution=execution),source=dict(code_commit='fixture',code_snapshot_hash='fixture')),
        capacity=dict(workers_per_device={d:4 for d in devices},maximum_workers_estimate=8),
        selected=[dict(n_couplings=123,npz='unused-fixture.npz',entry=dict(instance_id='fixture'))])
    selection.finalize_execution(prepared,{d:1 for d in devices})
    args=SimpleNamespace(resume=None,dry_run=False)
    return autotune,prepared,args,pools


def test_auto_calibration_selects_measured_gain_without_changing_algorithms(monkeypatch):
    autotune,prepared,args,pools=calibration_fixture(monkeypatch,{1:1.,2:1.5,4:1.6})
    solvers=copy.deepcopy(prepared['core']['solvers']);original_id=prepared['identity']
    autotune.calibrate(prepared,args)
    assert [p.size for p in pools]==[2,4,8]
    assert all(p.closed for p in pools)
    assert prepared['core']['protocol']['execution']['workers_per_device']=={'cuda:0':4,'cuda:1':4}
    assert prepared['core']['protocol']['execution']['execution_mode']=='parallel_shared_device'
    assert prepared['core']['solvers']==solvers and prepared['core']['budget_s']==1.
    assert prepared['identity']!=original_id
    assert len(prepared['tuning']['attempts'])==3
    assert all(job['seed']>=2**31 for p in pools for job in p.jobs)
    assert all(job['budget']==.5 for p in pools for job in p.jobs)


def test_auto_calibration_stops_when_contention_erases_throughput_gain(monkeypatch):
    autotune,prepared,args,pools=calibration_fixture(monkeypatch,{1:1.,2:2.,4:3.})
    original_id=prepared['identity']
    autotune.calibrate(prepared,args)
    assert [p.size for p in pools]==[2,4]
    assert all(p.closed for p in pools)
    assert prepared['core']['protocol']['execution']['workers_per_device']=={'cuda:0':1,'cuda:1':1}
    assert prepared['core']['protocol']['execution']['execution_mode']=='multi_gpu_isolated'
    assert prepared['identity']==original_id  # Different measurements do not alter execution identity.


def test_auto_calibration_seeds_do_not_overlap_custom_campaign_seeds(monkeypatch):
    autotune,prepared,args,pools=calibration_fixture(monkeypatch,{1:1.,2:2.})
    prepared['core']['seeds']=[2**31,2**31+1,2**31+2,2**31+3]
    autotune.calibrate(prepared,args)
    assert set(job['seed'] for pool in pools for job in pool.jobs).isdisjoint(prepared['core']['seeds'])


def test_auto_calibration_uses_frozen_minimum_gain_threshold(monkeypatch):
    autotune,prepared,args,pools=calibration_fixture(monkeypatch,{1:1.,2:2/1.075,4:4.})
    prepared['core']['protocol']['execution']['auto_concurrency_min_speedup']=1.1
    autotune.calibrate(prepared,args)
    assert [p.size for p in pools]==[2,4]
    assert prepared['core']['protocol']['execution']['workers_per_device']=={'cuda:0':1,'cuda:1':1}


@pytest.mark.parametrize('failure',['timeout','incomplete'])
def test_auto_calibration_rejects_failing_concurrency_and_closes_workers(monkeypatch,failure):
    autotune,prepared,args,pools=calibration_fixture(monkeypatch,{1:1.,2:.5,4:.5},
        fail_at=2 if failure=='timeout' else None,incomplete_at=2 if failure=='incomplete' else None)
    autotune.calibrate(prepared,args)
    assert [p.size for p in pools]==[2,4] and all(p.closed for p in pools)
    assert prepared['core']['protocol']['execution']['workers_per_device']=={'cuda:0':1,'cuda:1':1}
    assert prepared['tuning']['attempts'][-1]['failed']


def test_auto_calibration_rejects_broken_single_worker_baseline(monkeypatch):
    autotune,prepared,args,pools=calibration_fixture(monkeypatch,{1:1.},fail_at=1)
    with pytest.raises(ValueError,match='Single-worker GPU calibration failed'):
        autotune.calibrate(prepared,args)
    assert len(pools)==1 and pools[0].closed


@pytest.mark.parametrize('reason',['dry_run','resume','explicit','disabled','cpu'])
def test_auto_calibration_skipped_when_policy_requires_it(monkeypatch,reason):
    autotune,prepared,args,pools=calibration_fixture(monkeypatch,{})
    if reason=='dry_run':args.dry_run=True
    elif reason=='resume':args.resume='saved-results'
    elif reason=='explicit':prepared['core']['protocol']['execution']['seed_workers_requested']=2
    elif reason=='disabled':prepared['core']['protocol']['execution']['autotune_enabled']=False
    elif reason=='cpu':prepared['core']['actual_devices']=['cpu']
    autotune.calibrate(prepared,args)
    assert pools==[] and 'tuning' not in prepared

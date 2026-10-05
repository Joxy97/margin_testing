"""Incremental exports stay equivalent to journal recovery with bounded history RAM."""
import csv
import gc
import shutil
import tracemalloc

import pytest

from qubo_benchmark.runtime import storage
from qubo_benchmark.runtime.aggregate import main as aggregate
from qubo_benchmark.runtime.common import SCHEMA,atomic,digest,read


def fixture(count=2):
    policy=dict(worker_mode='persistent',seed_workers=4,execution_mode='multi_gpu_throughput')
    settings={'dtype':'float32'}
    schedule=[dict(run_id=f'run-{seed}',solver_id='lib_example',instance_id='example',
                   seed_index=seed,seed=seed,execution_order_index=seed) for seed in range(count)]
    experiment=dict(test_id='test',campaign_id='campaign',
        core=dict(solvers={'lib_example':settings},budget_s=1.,protocol=dict(execution=policy)),
        environment=dict(hardware_id='hardware'),execution_plan=schedule)
    payloads=[]
    for scheduled in schedule:
        row=dict.fromkeys(storage.RUN_FIELDS)
        row.update(scheduled,schema_version=SCHEMA,attempt_id=scheduled['run_id']+'-a0001',
            test_id='test',campaign_id='campaign',result_status='completed',
            actual_device=f"cuda:{scheduled['seed']%2}",workers_on_device=2,
            requested_devices_json=['cuda:0','cuda:1'],parameters_hash=digest(settings),
            execution_policy_hash=digest(policy),**policy,
            success_1pct=True,success_reference=False,exact_optimum_reached=None,
            history_available=True,in_budget_score_available=True,
            best_objective_within_budget=-10,signed_gap=1,signed_gap_percent=10,
            actual_solve_wall_s=.95,overshoot_s=0,ttt_1pct_s=.9)
        solution=dict.fromkeys(storage.SOLUTION_FIELDS)
        solution.update(schema_version=SCHEMA,run_id=row['run_id'],attempt_id=row['attempt_id'],
            solution_id='solution',bitstring='0001',valid=True)
        trace=dict.fromkeys(storage.TRACE_FIELDS)
        trace.update(schema_version=SCHEMA,run_id=row['run_id'],attempt_id=row['attempt_id'],
                     solution_id='solution',event_index=0,elapsed_s=.1,within_budget=True)
        payloads.append(dict(run=row,attempt_number=1,solutions=[solution],trace=[trace],
            warmup=dict(run_id=row['run_id'],attempt_id=row['attempt_id'],worker_id='worker',
                        future_setup_measurement=dict(bytes=123))))
    return experiment,payloads


def csv_rows(path):
    with path.open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))


def test_incremental_exports_equal_rebuild_and_group_devices(tmp_path):
    experiment,payloads=fixture()
    storage.materialize(tmp_path,experiment,[])
    for payload in payloads:
        storage.commit(tmp_path,payload)
        storage.append_projection(tmp_path,experiment,payload)
        storage.append_finalization(tmp_path,payload['run']['attempt_id'],
            dict(serialization_s=.002,trial_end_to_end_s=1.5,note='measured',future_metric=42))
    summary=storage.refresh_summaries(tmp_path,experiment,[storage.compact_attempt(a) for a in payloads])
    assert len(summary)==1 and summary[0]['terminal']==2
    assert summary[0]['actual_devices_json']==['cuda:0','cuda:1']
    before={p.relative_to(tmp_path).as_posix():csv_rows(p) for p in tmp_path.rglob('*.csv')}
    storage.materialize(tmp_path,experiment,storage.load_attempts(tmp_path,compact=True))
    assert before=={p.relative_to(tmp_path).as_posix():csv_rows(p) for p in tmp_path.rglob('*.csv')}
    assert before['solvers/lib_example/solutions.csv'][0]['bitstring']=='0001'
    assert 'future_setup_measurement' in before['warmups.csv'][0]['extra_json']
    assert 'future_metric' in before['finalization.csv'][0]['extra_json']


def test_crash_between_commit_and_partial_csv_append_recovers_once(tmp_path):
    experiment,payloads=fixture()
    storage.materialize(tmp_path,experiment,[])
    for payload in payloads:storage.commit(tmp_path,payload)
    storage.append_projection(tmp_path,experiment,payloads[0])
    own=tmp_path/'solvers/lib_example'
    # A process dies while exporting the next journal, then resume repairs it.
    with (own/'runs.csv').open('a',encoding='utf-8') as handle:handle.write('"partial')
    attempts=storage.load_attempts(tmp_path,compact=True)
    storage.check_resume_exports(tmp_path,experiment,attempts)
    storage.materialize(tmp_path,experiment,attempts)
    assert len(csv_rows(own/'runs.csv'))==2
    assert len(csv_rows(own/'solutions.csv'))==2
    assert len(csv_rows(own/'trace.csv'))==2
    assert 'rebuilding' in (tmp_path/'csv_recovery.log').read_text()
    # Rebuilding again never duplicates entries, and absent finalization stays absent.
    storage.materialize(tmp_path,experiment,attempts)
    assert len(csv_rows(own/'runs.csv'))==2 and csv_rows(tmp_path/'finalization.csv')==[]


def test_compact_history_memory_and_corrupt_journal_detection(tmp_path):
    _,payloads=fixture(4)
    for payload in payloads:
        payload['solutions']=[dict(payload['solutions'][0],solution_id=str(i),bitstring='0'*1000+str(i%2))
                              for i in range(256)]
        storage.commit(tmp_path,payload)
    def retained(compact):
        gc.collect();tracemalloc.start()
        try:
            records=storage.load_attempts(tmp_path,compact=compact)
            retained_bytes=tracemalloc.get_traced_memory()[0]
            assert len(records)==4
            if compact:assert all(set(a)=={'run','attempt_number'} for a in records)
            return retained_bytes
        finally:tracemalloc.stop()
    full_bytes=retained(False);compact_bytes=retained(True)
    assert compact_bytes<full_bytes/3
    path=next((tmp_path/'attempts').glob('*.json'));wrapper=read(path)
    wrapper['payload']['solutions'][0]['bitstring']='111';atomic(path,wrapper)
    with pytest.raises(ValueError,match='Corrupt'):storage.load_attempts(tmp_path,compact=True)


def test_streaming_aggregate_deduplicates_full_histories_and_rejects_conflicts(tmp_path):
    experiment,payloads=fixture()
    source=tmp_path/'results'/'original';atomic(source/'experiment.json',experiment)
    for payload in payloads:storage.commit(source,payload)
    duplicate=tmp_path/'results'/'copy';shutil.copytree(source,duplicate)
    out=tmp_path/'analysis'
    assert aggregate([str(tmp_path/'results'),'--output',str(out)])==0
    assert len(csv_rows(out/'runs.csv'))==2 and len(csv_rows(out/'solutions.csv'))==2
    assert len(read(out/'aggregation.json')['duplicate_copies_ignored'])==1
    assert csv_rows(out/'solutions.csv')[0]['actual_device'].startswith('cuda:')
    path=next((duplicate/'attempts').glob('*.json'));wrapper=read(path)
    wrapper['payload']['solutions'][0]['bitstring']='1010'
    wrapper['sha256']=digest(wrapper['payload']);atomic(path,wrapper)
    rejected=tmp_path/'rejected'
    with pytest.raises(ValueError,match='Conflicting experiment'):
        aggregate([str(tmp_path/'results'),'--output',str(rejected)])
    assert not (rejected/'runs.csv').exists()


def test_engine_out_of_order_results_resume_interrupted_seed_once(tmp_path,monkeypatch):
    from qubo_benchmark.runtime import engine
    from qubo_benchmark.runtime.cli import parser,resolve
    from qubo_benchmark.runtime.selection import preflight

    args=resolve(parser().parse_args(['200','sparse','.05','--instances','representative',
        '--solvers','lib_random_search','--runs','3','--device','cpu','--seed-workers','2',
        '--output-root',str(tmp_path)]))
    prepared=preflight(args)
    monkeypatch.setattr(engine,'source_snapshot_hash',lambda:prepared['core']['source']['code_snapshot_hash'])
    dispatched=[];invocations=[]

    class OutOfOrderScheduler:
        def __init__(self,*_):self.index=len(invocations);invocations.append(self)
        def snapshot(self):return []
        def close(self):return dict(workers=[],dispatched=len(dispatched))
        def outcomes(self,planned,job_for,heartbeat):
            # First invocation finishes seed2, interrupts seed1, never starts0.
            # Resume finishes the two remaining seeds in reverse plan order.
            chosen=[planned[2],planned[1]] if self.index==0 else list(reversed(planned))
            for row in chosen:
                job=job_for(row,'cpu');dispatched.append(job['seed'])
                error='interrupted' if self.index==0 and row['seed']==1 else None
                yield row,dict(events=[dict(bitstring='0'*200,elapsed_s=.01,phase='initial',raw=0,iteration=0)],
                    done=dict(error=error,actual_solve_wall_s=.02,stop_reason='deadline'),
                    setup=dict(worker_ready_wall_s=.001),actual_device='cpu',
                    started_at_utc='2026-10-05T00:00:00+00:00',finished_at_utc='2026-10-05T00:00:01+00:00',
                    cleanup_s=0.,worker_end_to_end_s=.025,cpu_rss_start_bytes=100,
                    cpu_rss_peak_sampled_bytes=120,abort_gpu=False)

    monkeypatch.setattr(engine,'DeviceScheduler',OutOfOrderScheduler)
    directory,code=engine.execute(prepared,args)
    assert code==130 and dispatched==[2,1]
    assert read(directory/'status.json')['pending']==2
    args.resume=str(directory)
    _,code=engine.execute(prepared,args)
    assert code==0 and dispatched==[2,1,1,0]
    attempts=storage.load_attempts(directory,compact=True)
    assert len(attempts)==4 and len(storage.primary(attempts))==3
    assert [a['attempt_number'] for a in attempts if a['run']['seed']==1]==[1,2]
    assert all(a['run']['result_status']=='completed' for a in storage.primary(attempts).values())
    assert len(csv_rows(directory/'solvers/lib_random_search/runs.csv'))==4
    assert len(csv_rows(directory/'finalization.csv'))==4

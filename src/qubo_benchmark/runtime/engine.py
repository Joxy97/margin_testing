"""Execution plan, durable trial commits and live progress."""
from pathlib import Path
import shutil
import signal
import time
import uuid
from contextlib import closing
from functools import lru_cache
from .common import ROOT,CONFIG,SCHEMA,utc,digest,atomic,read,table,flattened,DirectoryLock,source_snapshot_hash
from .selection import preflight
from .storage import (RUN_FIELDS,plan,load_attempts,primary,commit,materialize,check_resume_exports,
                      append_projection,append_finalization,compact_attempt,refresh_summaries)
from .metrics import evaluate
from .supervisor import trial
from .scheduler import DeviceScheduler
from .telemetry import Telemetry

def initialize(prepared,output_root):
    core=prepared['core'];test_id=uuid.uuid4().hex[:12]
    stamp=utc().replace(':','').replace('-','').replace('.','')[:15]
    policy=core['protocol']['execution']
    name=f"n{core['n']}_{core['density']}_t{core['budget_s']:.3f}s_{core['instance_mode']}_g{len(core['actual_devices'])}_w{policy['seed_workers']}_{policy['worker_mode']}_{stamp}_{test_id}".replace('.','p')
    directory=Path(output_root)/name;directory.mkdir(parents=True,exist_ok=False)
    campaign=digest(dict(solvers=core['solvers'],seeds=core['seeds'],protocol=core['protocol'],
                        reference=core['reference_snapshot_sha256'],environment=core['environment_hash'],
                        code=core['source']['code_snapshot_hash']))
    experiment=dict(schema_version=SCHEMA,test_id=test_id,campaign_id=campaign,created_at=utc(),
                    identity=prepared['identity'],core=core,environment=prepared['environment'],
                    execution_plan=plan(core),label=('STANDARD' if core['standard'] else 'NONSTANDARD SMOKE')+
                    ' '+policy['execution_mode'].upper())
    atomic(directory/'experiment.json',experiment)
    atomic(directory/'environment.json',prepared['environment'])
    atomic(directory/'worker_capacity.json',prepared['capacity'])
    atomic(directory/'autotune.json',prepared.get('tuning',dict(note='Calibration not requested; concurrency explicit, resumed or default single-worker')))
    atomic(directory/'reference_snapshot.json',prepared['catalog'])
    atomic(directory/'solver_configuration.json',prepared['configuration'])
    atomic(directory/'seeds.json',core['seeds'])
    shutil.copyfile(ROOT/'BENCHMARK_DEFINITIONS.txt',directory/'BENCHMARK_DEFINITIONS.txt')
    table(directory/'experiment_metadata.csv',flattened({k:v for k,v in experiment.items() if k not in ('core','execution_plan','environment')}))
    table(directory/'environment.csv',flattened(prepared['environment']))
    table(directory/'solver_registry.csv',prepared['registry'])
    table(directory/'execution_plan.csv',experiment['execution_plan'])
    table(directory/'seed_schedule.csv',[dict(seed_index=i,seed=s) for i,s in enumerate(core['seeds'])])
    table(directory/'instances.csv',[dict(instance_id=i['entry']['instance_id'],n_variables=i['entry']['binary_variables'],
          problem_family=i['entry']['problem_family'],density_category=i['entry']['density_category'],
          normalized_problem_sha256=i['normalized_problem_sha256'],reference_objective=i['entry']['normalized_reference_objective_min'],
          reference_status=i['entry']['reference_status'],n_couplings=i['n_couplings'],measured_density=i['measured_density'])
          for i in prepared['selected']])
    for solver,parameters in core['solvers'].items():
        path=directory/'solvers'/solver
        atomic(path/'parameters.json',dict(parameters=parameters,parameters_hash=digest(parameters)))
        table(path/'parameters.csv',[dict(row,configuration_hash=digest(parameters)) for row in flattened(parameters)])
    materialize(directory,experiment,[])
    return directory,experiment

@lru_cache(maxsize=4)
def scoring_problem(path):
    from ..model import Problem
    return Problem.load(path)


def assemble(experiment,item,scheduled,outcome,attempt_number,session,audit_started):
    core=experiment['core'];entry=item['entry'];meta=item['metadata']
    problem=scoring_problem(item['npz']);done=outcome['done'];elapsed=done.get('actual_solve_wall_s')
    device=outcome.get('actual_device',core['actual_device'])
    metrics,solutions,trace=evaluate(problem,outcome['events'],core['budget_s'],
                          entry['normalized_reference_objective_min'],entry['reference_status'],done.get('error'))
    audit=time.perf_counter()-audit_started
    solver=scheduled['solver_id'];p=core['solvers'][solver];attempt_id=scheduled['run_id']+f'-a{attempt_number:04d}'
    row={field:None for field in RUN_FIELDS}
    row.update(schema_version=SCHEMA,campaign_id=experiment['campaign_id'],test_id=experiment['test_id'],
        **scheduled,attempt_id=attempt_id,problem_family=entry['problem_family'],
        n_variables=problem.n,density_category=entry['density_category'],measured_density=item['measured_density'],
        n_couplings=item['n_couplings'],nnz_diagonal=item['nnz_diagonal'],nnz_matrix=item['nnz_matrix'],
        raw_source_sha256=item['raw_source_sha256'],normalized_problem_sha256=item['normalized_problem_sha256'],
        catalog_sha256=core['catalog_sha256'],reference_snapshot_sha256=core['reference_snapshot_sha256'],
        index_mapping_id=digest(meta['source_labels']),objective_offset=problem.offset,
        solver_version='0.1.0',adapter_version='runtime-v4',**core['source'],
        parameters_hash=digest(p),parameters_json=p,execution_policy_hash=digest(core['protocol']['execution']),
        rng_backend='Python Random; NumPy legacy + explicit per-trajectory Generators; Torch local Generators',
        initialization_policy='fresh solver object and reseeded RNGs inside solve clock; immutable input cache; compact seedOffset=0',
        initialization_hash=done.get('initialization_hash'),seed_effective=True,
        deterministic_mode='stochastic seeded; wall-clock truncation and CUDA arithmetic are not bitwise reproducibility guarantees',
        population_size=p['runs'],replicas=p.get('replicas'),internal_batch_size=p.get('run_batch_size'),
        worker_mode=core['protocol']['execution']['worker_mode'],
        seed_workers=core['protocol']['execution']['seed_workers'],
        workers_on_device=core['protocol']['execution']['workers_per_device'][device],
        requested_devices_json=core['actual_devices'],
        execution_mode=core['protocol']['execution']['execution_mode'],
        worker_trial_index=done.get('worker_trial_index'),
        restarts=p['runs'],cpu_threads=core['protocol']['execution']['cpu_threads'],
        requested_device=core['requested_device'],actual_device=device,backend='torch',precision=p['dtype'],
        hardware_id=experiment['environment']['hardware_id'],environment_hash=core['environment_hash'],
        warmup_id=attempt_id+'-warmup',static_setup_id=attempt_id+'-setup',session_id=session,
        requested_time_s=core['budget_s'],timing_scope=core['protocol']['execution']['timing_scope'],
        started_at_utc=outcome['started_at_utc'],finished_at_utc=outcome['finished_at_utc'],
        trial_reset_init_s=done.get('trial_reset_init_s'),actual_solve_wall_s=elapsed,
        static_setup_s=outcome['setup'].get('worker_ready_wall_s'),
        transfer_s=outcome['setup'].get('transfer_s'),audit_s=audit,cleanup_s=outcome['cleanup_s'],
        trial_end_to_end_s=outcome['worker_end_to_end_s']+audit,
        post_budget_return_s=elapsed if elapsed is not None and elapsed>core['budget_s'] else None,
        overshoot_s=max(0.,elapsed-core['budget_s']) if elapsed is not None else None,
        watchdog_grace_s=core['protocol']['execution']['watchdog_grace_s'],
        deadline_status=done.get('stop_reason',done.get('error')),
        budget_compliance='candidate_eligibility_uses_completed_capture_time',
        stop_reason=done.get('stop_reason'),iterations=done.get('iterations'),
        reference_objective=entry['normalized_reference_objective_min'],reference_status=entry['reference_status'],
        reference_gap_label='gap_to_optimum' if entry['reference_status']=='published_proven_optimum' else 'gap_to_published_reference',
        reference_evidence_checked=meta['reference']['evidence_checked'],
        reference_vector_validated=meta['reference']['reference_vector_evaluated'],
        target_fraction=.01,arithmetic_tolerance=0,source_problem_valid=True,
        solver_raw_energy_convention='native diagnostic only; may omit offset; never used for benchmark quality',
        error_type=done.get('error_type'),error_message=done.get('error_message'),
        observation_end_s=min(elapsed,core['budget_s']) if elapsed is not None else None,
        censor_time_s=min(elapsed,core['budget_s']) if elapsed is not None and not done.get('error') else None,
        time_measurement_qualifier='host-observed completed captures; TTT is an observation upper bound, not exact discovery time',
        cpu_rss_start_bytes=outcome['cpu_rss_start_bytes'],cpu_rss_peak_sampled_bytes=outcome['cpu_rss_peak_sampled_bytes'],
        cpu_memory_method='supervisor samples worker process tree RSS; may miss transient peaks',
        cpu_memory_sampling_interval_s=core['protocol']['execution']['memory_sampling_interval_s'],
        gpu_memory_method='Torch allocator peak counters' if device.startswith('cuda') else 'not_applicable',
        peak_memory_scope='timed reset/search/capture plus persistent input baseline; RSS sampled through return',
        unsupported_measurements='algorithm_preprocess_s not isolated; reset field excludes algorithm-specific initialization; '
          'serialization/end-to-end finalization in finalization.csv; GPU event/process sampling unavailable; evaluation counters unavailable')
    row.update(metrics)
    if not metrics['history_available'] or not metrics['in_budget_score_available'] or done.get('error'):
        row['censor_time_s']=None
    for field in ('gpu_allocated_start_bytes','gpu_allocated_peak_bytes','gpu_reserved_start_bytes','gpu_reserved_peak_bytes'):
        row[field]=done.get(field)
    for field in ('matrix_storage_format','matrix_storage_bytes'):row[field]=outcome['setup'].get(field)
    for field in ('worker_id','worker_pid','worker_reused','input_cache_hit'):row[field]=outcome['setup'].get(field)
    row.update(done.get('preparation',{}))
    if row['resolved_structure_json'] is None:
        row['resolved_structure_json']=dict(original_variables=problem.n,
            note='native dense representation prepared before clock, or preprocessing interrupted before matrix became ready')
    linkage=dict(schema_version=SCHEMA,run_id=scheduled['run_id'],attempt_id=attempt_id)
    return dict(run=row,solutions=[dict(s,**linkage) for s in solutions],trace=[dict(t,**linkage) for t in trace],
                attempt_number=attempt_number,warmup=dict(linkage,**dict(outcome['setup'],device=device)))


def execute(prepared,args):
    if args.resume:
        directory=Path(args.resume).resolve();experiment=read(directory/'experiment.json')
        if experiment['identity']!=prepared['identity']:raise ValueError('Resume identity mismatch: data/reference/seeds/settings/code/hardware/policy changed')
        if read(directory/'reference_snapshot.json')!=prepared['catalog']:raise ValueError('Reference snapshot changed')
        from .common import filehash
        if filehash(directory/'BENCHMARK_DEFINITIONS.txt')!=experiment['core']['definitions_sha256']:
            raise ValueError('Protocol definitions copy changed')
    else:directory,experiment=initialize(prepared,args.output_root)
    print(f"Output: {directory.resolve()} [{experiment['label']}]",flush=True)
    interrupted=[False]
    def request_stop(*_):interrupted[0]=True
    old_handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    scheduler=None;telemetry=None
    try:
        with DirectoryLock(directory):
            attempts=load_attempts(directory,compact=True)
            check_resume_exports(directory,experiment,attempts)
            materialize(directory,experiment,attempts)
            core=experiment['core'];session=uuid.uuid4().hex;policy=core['protocol']['execution']
            items={i['entry']['instance_id']:i for i in prepared['selected']}
            registry=prepared['registry'];eligible=sum(r['status']=='runnable' for r in registry)
            current=None;selected=primary(attempts);total=len(experiment['execution_plan'])
            telemetry=Telemetry(core['actual_devices'],output_path=directory/'hardware_telemetry.csv',
                gpu_metadata=experiment['environment']['hardware'].get('gpus')).start()
            begun=time.perf_counter();next_source_check=0.;next_summary=0.
            counts={}
            for attempt in attempts:
                key=attempt['run']['run_id'];counts[key]=max(counts.get(key,0),attempt['attempt_number'])
            def progress(phase):
                nonlocal next_source_check
                now=time.perf_counter()
                if now>=next_source_check:
                    next_source_check=now+1.
                    if source_snapshot_hash()!=core['source']['code_snapshot_hash']:
                        interrupted[0]=True;phase='source_changed'
                active=scheduler.snapshot() if scheduler is not None else []
                terminal=len(selected);successes=sum(a['run']['success_1pct'] for a in selected.values())
                own=[a['run'] for a in selected.values() if current and
                     a['run']['solver_id']==current['solver_id'] and a['run']['instance_id']==current['instance_id']]
                gaps=[r['signed_gap_percent'] for r in own if r['signed_gap_percent'] is not None]
                status=dict(test_id=experiment['test_id'],label=experiment['label'],phase=phase,heartbeat_at=utc(),
                    planned=total,terminal=terminal,pending=total-terminal,attempts=len(attempts),
                    percentage=100*terminal/total,verified_valid=sum(a['run']['result_status']=='completed' for a in selected.values()),
                    failures=sum(a['run']['result_status']!='completed' for a in selected.values()),
                    near_target_hits=successes,success_denominator=total,rate_status='final' if terminal==total else 'provisional',
                    discovered_solvers=len(registry),eligible_solvers=eligible,
                    unavailable_solvers=sum(r['status']=='unavailable' for r in registry),
                    excluded_solvers=sum(r['status']=='excluded' for r in registry),
                    output=str(directory.resolve()),current=current,latest_gap_percent=gaps[-1] if gaps else None,
                    active_trials=active,active_seeds=[r['seed'] for r in active],seed_workers=policy['seed_workers'],
                    workers_per_device=policy['workers_per_device'],actual_devices=core['actual_devices'],
                    execution_mode=policy['execution_mode'],worker_mode=policy['worker_mode'],
                    best_gap_percent=min(gaps) if gaps else None,current_instance_terminal=len(own),
                    current_solver_terminal=sum(a['run']['solver_id']==current['solver_id'] for a in selected.values()) if current else 0,
                    session_elapsed_s=now-begun,session_trials_per_s=(len(attempts)-initial_attempts)/max(now-begun,1e-9))
                status['hardware_telemetry']=telemetry.snapshot()
                atomic(directory/'status.json',status)
                line=(f"{phase} | {terminal}/{total} ({status['percentage']:.2f}%) | "
                      f"solver {list(core['solvers']).index(current['solver_id'])+1}/{len(core['solvers'])} {current['solver_id']} | "
                      f"instance {list(items).index(current['instance_id'])+1}/{len(items)} {current['instance_id']} | "
                      f"seed run {current['seed_index']+1}/{len(core['seeds'])} seed={current['seed']} | "
                      f"GPUs/devices={len(core['actual_devices'])} workers={policy['seed_workers']} active={len(active)} | "
                      f"latest gap={status['latest_gap_percent']}% best gap={status['best_gap_percent']}% | "
                      f"hits {successes}/{terminal}") if current else f"{phase}: {terminal}/{total}, active={len(active)}"
                print(line,flush=True)
                with (directory/'progress.log').open('a',encoding='utf-8') as handle:handle.write(utc()+' '+line+'\n')
            initial_attempts=len(attempts);progress('ready')
            # Retry targets frozen once; failed primaries are never silently replaced.
            todo=[r for r in experiment['execution_plan'] if r['run_id'] not in selected or
                  (args.retry_failed and selected[r['run_id']]['run']['result_status']!='completed')]
            def job_for(row,device):
                return dict(solver=row['solver_id'],seed=row['seed'],budget=core['budget_s'],
                    parameters=core['solvers'][row['solver_id']],npz=items[row['instance_id']]['npz'],device=device,
                    **{k:policy[k] for k in ('cpu_threads','watchdog_grace_s','setup_timeout_s',
                                            'memory_sampling_interval_s','heartbeat_interval_s')})
            def fresh_outcomes():
                for row in todo:
                    if interrupted[0]:return
                    outcome=trial(job_for(row,core['actual_device']),progress,lambda:interrupted[0])
                    yield row,outcome
            if todo and not (policy['worker_mode']=='fresh' and policy['seed_workers']==1):
                scheduler=DeviceScheduler(policy['workers_per_device'],policy['worker_mode'],lambda:interrupted[0])
            stream=(scheduler.outcomes(todo,job_for,progress) if scheduler is not None else fresh_outcomes())
            try:
                with closing(stream):
                    for current,outcome in stream:
                        trial_started=outcome.get('parent_trial_started',time.perf_counter()-outcome['worker_end_to_end_s'])
                        if source_snapshot_hash()!=core['source']['code_snapshot_hash']:
                            outcome['done'].update(error='code_changed',error_message='Source changed during campaign')
                            interrupted[0]=True
                        key=current['run_id'];number=counts.get(key,0)+1;counts[key]=number
                        payload=assemble(experiment,items[current['instance_id']],current,outcome,number,session,time.perf_counter())
                        payload['run'].update(trial_end_to_end_s=time.perf_counter()-trial_started,wave_id=None,wave_size=None)
                        serial_start=time.perf_counter();commit(directory,payload)
                        append_projection(directory,experiment,payload)
                        compact=compact_attempt(payload);attempts.append(compact)
                        if payload['run']['result_status']!='interrupted':selected.setdefault(key,compact)
                        if time.perf_counter()>=next_summary:
                            refresh_summaries(directory,experiment,attempts);next_summary=time.perf_counter()+5.
                        serialization=time.perf_counter()-serial_start
                        append_finalization(directory,payload['run']['attempt_id'],
                            dict(serialization_s=serialization,trial_end_to_end_s=time.perf_counter()-trial_started,
                                 note='Includes preparation, result-queue waiting, audit, journal and CSV append; excludes final pool shutdown'))
                        if outcome['done'].get('error')=='interrupted':interrupted[0]=True
                        if outcome['abort_gpu']:interrupted[0]=True;progress('gpu_worker_lost_resume_required')
                        progress('trial_saved')
            finally:
                if scheduler is not None:atomic(directory/'worker_sessions'/f'{session}.json',scheduler.close())
            materialize(directory,experiment,attempts)
            if len(selected)<total:interrupted[0]=True
            unavailable=any(r['status']=='unavailable' for r in registry)
            failures=any(a['run']['result_status']!='completed' for a in selected.values())
            progress('interrupted' if interrupted[0] else 'complete_with_unavailable_solvers' if unavailable else 'complete_with_failures' if failures else 'complete')
            return directory,130 if interrupted[0] else 1 if unavailable or failures else 0
    finally:
        if telemetry is not None:telemetry.close()
        scoring_problem.cache_clear()
        for s,handler in old_handlers.items():signal.signal(s,handler)

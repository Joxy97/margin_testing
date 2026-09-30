"""Execution plan, durable trial commits and live progress."""
from pathlib import Path
import shutil
import signal
import time
import uuid
from .common import ROOT,CONFIG,SCHEMA,utc,digest,atomic,read,table,flattened,DirectoryLock,source_identity
from .selection import preflight
from .storage import RUN_FIELDS,plan,load_attempts,primary,commit,materialize,check_resume_exports
from .metrics import evaluate
from .supervisor import trial

def initialize(prepared,output_root):
    core=prepared['core'];test_id=uuid.uuid4().hex[:12]
    stamp=utc().replace(':','').replace('-','').replace('.','')[:15]
    name=f"n{core['n']}_{core['density']}_t{core['budget_s']:.3f}s_{core['instance_mode']}_{stamp}_{test_id}".replace('.','p')
    directory=Path(output_root)/name;directory.mkdir(parents=True,exist_ok=False)
    campaign=digest(dict(solvers=core['solvers'],seeds=core['seeds'],protocol=core['protocol'],
                        reference=core['reference_snapshot_sha256'],environment=core['environment_hash'],
                        code=core['source']['code_snapshot_hash']))
    experiment=dict(schema_version=SCHEMA,test_id=test_id,campaign_id=campaign,created_at=utc(),
                    identity=prepared['identity'],core=core,environment=prepared['environment'],
                    execution_plan=plan(core),label='STANDARD' if core['standard'] else 'NONSTANDARD SMOKE')
    atomic(directory/'experiment.json',experiment)
    atomic(directory/'environment.json',prepared['environment'])
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

def assemble(experiment,item,scheduled,outcome,attempt_number,session,audit_started):
    from ..model import Problem
    core=experiment['core'];entry=item['entry'];meta=item['metadata']
    problem=Problem.load(item['npz']);done=outcome['done'];elapsed=done.get('actual_solve_wall_s')
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
        solver_version='0.1.0',adapter_version='runtime-v2',**core['source'],
        parameters_hash=digest(p),parameters_json=p,execution_policy_hash=digest(core['protocol']['execution']),
        rng_backend='Python Random; NumPy legacy + explicit per-trajectory Generators; Torch local Generators',
        initialization_policy='fresh process; reseed after infrastructure warmup; native algorithm initialization; compact seedOffset=0',
        initialization_hash=done.get('initialization_hash'),seed_effective=True,
        deterministic_mode='stochastic seeded; wall-clock truncation and CUDA arithmetic are not bitwise reproducibility guarantees',
        population_size=p['runs'],replicas=p.get('replicas'),internal_batch_size=p.get('run_batch_size'),
        restarts=p['runs'],cpu_threads=core['protocol']['execution']['cpu_threads'],
        requested_device=core['requested_device'],actual_device=core['actual_device'],backend='torch',precision=p['dtype'],
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
        gpu_memory_method='Torch allocator peak counters' if core['actual_device'].startswith('cuda') else 'not_applicable',
        peak_memory_scope='timed reset/search/capture plus persistent input baseline; RSS sampled through return',
        unsupported_measurements='algorithm_preprocess_s not isolated; reset field excludes algorithm-specific initialization; '
          'serialization/end-to-end finalization in finalization.csv; GPU event/process sampling unavailable; evaluation counters unavailable')
    row.update(metrics)
    if not metrics['history_available'] or not metrics['in_budget_score_available'] or done.get('error'):
        row['censor_time_s']=None
    for field in ('gpu_allocated_start_bytes','gpu_allocated_peak_bytes','gpu_reserved_start_bytes','gpu_reserved_peak_bytes'):
        row[field]=done.get(field)
    for field in ('matrix_storage_format','matrix_storage_bytes'):row[field]=outcome['setup'].get(field)
    row.update(done.get('preparation',{}))
    if row['resolved_structure_json'] is None:
        row['resolved_structure_json']=dict(original_variables=problem.n,
            note='native dense representation prepared before clock, or preprocessing interrupted before matrix became ready')
    linkage=dict(schema_version=SCHEMA,run_id=scheduled['run_id'],attempt_id=attempt_id)
    return dict(run=row,solutions=[dict(s,**linkage) for s in solutions],trace=[dict(t,**linkage) for t in trace],
                attempt_number=attempt_number,warmup=dict(linkage,**outcome['setup']))

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
    try:
        with DirectoryLock(directory):
            attempts=load_attempts(directory)
            check_resume_exports(directory,experiment,attempts)
            materialize(directory,experiment,attempts)
            core=experiment['core'];session=uuid.uuid4().hex
            items={i['entry']['instance_id']:i for i in prepared['selected']}
            registry=prepared['registry'];eligible=sum(r['status']=='runnable' for r in registry)
            current=None
            def progress(phase):
                selected=primary(attempts);terminal=len(selected);total=len(experiment['execution_plan'])
                successes=sum(a['run']['success_1pct'] for a in selected.values())
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
                    best_gap_percent=min(gaps) if gaps else None,current_instance_terminal=len(own),
                    current_solver_terminal=sum(a['run']['solver_id']==current['solver_id'] for a in selected.values()) if current else 0)
                atomic(directory/'status.json',status)
                line=(f"{phase} | {terminal}/{total} ({status['percentage']:.2f}%) | "
                      f"solver {list(core['solvers']).index(current['solver_id'])+1}/{len(core['solvers'])} {current['solver_id']} | "
                      f"instance {list(items).index(current['instance_id'])+1}/{len(items)} {current['instance_id']} | "
                      f"seed run {current['seed_index']+1}/{len(core['seeds'])} seed={current['seed']} | "
                      f"latest gap={status['latest_gap_percent']}% best gap={status['best_gap_percent']}% | "
                      f"hits {successes}/{terminal} {'FINAL' if terminal==total else 'PROVISIONAL'}") if current else f"{phase}: {terminal}/{total}"
                print(line,flush=True)
                with (directory/'progress.log').open('a',encoding='utf-8') as handle:handle.write(utc()+' '+line+'\n')
            progress('ready')
            # Freeze retry targets at invocation; never loop retrying failures.
            selected=primary(attempts)
            todo=[r for r in experiment['execution_plan'] if r['run_id'] not in selected or
                  (args.retry_failed and selected[r['run_id']]['run']['result_status']!='completed')]
            for current in todo:
                if interrupted[0]:break
                trial_started=time.perf_counter()
                if source_identity()['code_snapshot_hash']!=core['source']['code_snapshot_hash']:
                    progress('source_changed');raise ValueError('Source changed during campaign; no new trial scheduled')
                progress('starting')
                number=1+sum(a['run']['run_id']==current['run_id'] for a in attempts)
                item=items[current['instance_id']];policy=core['protocol']['execution']
                job=dict(solver=current['solver_id'],seed=current['seed'],budget=core['budget_s'],
                    parameters=core['solvers'][current['solver_id']],npz=item['npz'],device=core['actual_device'],
                    **{k:policy[k] for k in ('cpu_threads','watchdog_grace_s','setup_timeout_s',
                                            'memory_sampling_interval_s','heartbeat_interval_s')})
                outcome=trial(job,progress,lambda:interrupted[0])
                if source_identity()['code_snapshot_hash']!=core['source']['code_snapshot_hash']:
                    outcome['done'].update(error='code_changed',error_message='Source changed during trial')
                    interrupted[0]=True
                payload=assemble(experiment,item,current,outcome,number,session,time.perf_counter())
                payload['run']['trial_end_to_end_s']=time.perf_counter()-trial_started
                serial_start=time.perf_counter();commit(directory,payload);attempts.append(payload)
                materialize(directory,experiment,attempts)
                serialization=time.perf_counter()-serial_start
                atomic(directory/'finalization'/f"{payload['run']['attempt_id']}.json",
                       dict(serialization_s=serialization,trial_end_to_end_s=time.perf_counter()-trial_started,
                            note='includes atomic attempt commit and CSV export; excludes this finalization file and progress update'))
                if outcome['done'].get('error')=='interrupted':interrupted[0]=True
                if outcome['abort_gpu']:
                    interrupted[0]=True
                    progress('gpu_worker_lost_resume_required');break
                progress('trial_saved')
                if interrupted[0]:break
            materialize(directory,experiment,attempts)
            if len(primary(attempts))<len(experiment['execution_plan']):interrupted[0]=True
            unavailable=any(r['status']=='unavailable' for r in registry)
            progress('interrupted' if interrupted[0] else 'complete_with_unavailable_solvers' if unavailable else 'complete_with_failures' if any(
                a['run']['result_status']!='completed' for a in primary(attempts).values()) else 'complete')
            return directory,130 if interrupted[0] else 1 if unavailable or any(
                a['run']['result_status']!='completed' for a in primary(attempts).values()) else 0
    finally:
        for s,handler in old_handlers.items():signal.signal(s,handler)

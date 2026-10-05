"""Durable per-attempt records are authoritative; CSVs are repairable materializations."""
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager, ExitStack
import statistics
import csv
import os
from .common import SCHEMA,atomic,read,digest,table,canonical,utc

RUN_FIELDS = """
schema_version campaign_id test_id run_id attempt_id instance_id problem_family n_variables density_category
measured_density n_couplings nnz_diagonal nnz_matrix matrix_storage_format matrix_storage_bytes
raw_source_sha256 normalized_problem_sha256 catalog_sha256 reference_snapshot_sha256 index_mapping_id objective_offset
solver_id solver_version adapter_version code_commit code_dirty code_snapshot_hash parameters_hash parameters_json
execution_policy_hash seed_index seed rng_backend initialization_policy initialization_hash seed_effective
worker_mode seed_workers execution_mode worker_id worker_pid worker_reused input_cache_hit worker_trial_index wave_id wave_size
workers_on_device requested_devices_json
deterministic_mode population_size replicas internal_batch_size restarts cpu_threads requested_device actual_device
backend precision hardware_id environment_hash warmup_id execution_order_index session_id requested_time_s timing_scope
started_at_utc finished_at_utc trial_reset_init_s algorithm_preprocess_s actual_solve_wall_s gpu_elapsed_s
static_setup_id static_setup_s transfer_s audit_s serialization_s cleanup_s trial_end_to_end_s
last_candidate_elapsed_s post_budget_return_s overshoot_s watchdog_grace_s deadline_status budget_compliance
stop_reason iterations evaluations completed_work_units reference_objective reference_status reference_evidence_checked
reference_vector_validated target_fraction near_target_objective arithmetic_tolerance solver_reported_raw_min
solver_raw_energy_convention raw_binary_min_objective best_objective_within_budget final_returned_objective
final_returned_within_budget postprocessed_objective signed_gap signed_gap_percent reference_beaten
optimum_validation_alarm success_1pct success_reference reference_equal exact_optimum_reached candidate_valid
source_problem_valid in_budget_score_available score_audit_status result_status error_type error_message
history_available history_mode observation_resolution_s time_to_first_candidate_s time_to_best_s ttt_1pct_s
ttt_reference_s ttt_optimum_s near_target_event reference_target_event observation_end_s censor_time_s
near_target_time_status reference_time_status optimum_time_status time_measurement_qualifier trace_truncated
cpu_rss_start_bytes cpu_rss_peak_sampled_bytes cpu_memory_method cpu_memory_sampling_interval_s
gpu_allocated_start_bytes gpu_allocated_peak_bytes gpu_reserved_start_bytes gpu_reserved_peak_bytes
gpu_process_peak_sampled_bytes gpu_memory_method gpu_memory_sampling_interval_s peak_memory_scope
initial_solution_id best_solution_id final_solution_id solution_sha256 trace_event_count
unsupported_measurements reference_gap_label resolved_structure_json
""".split()
SOLUTION_FIELDS="""schema_version run_id attempt_id solution_id role n_variables index_mapping bitstring
independent_objective completed_elapsed_s within_budget valid solution_sha256""".split()
TRACE_FIELDS="""schema_version run_id attempt_id event_index solution_id elapsed_s iteration evaluation_count phase
solver_reported_energy independently_evaluated_objective independently_validated_best_so_far signed_gap
signed_gap_percent near_target_hit reference_target_hit within_budget timestamp_kind valid""".split()
WARMUP_FIELDS="""schema_version run_id attempt_id static_setup_s transfer_s warmup_s warmup_policy
matrix_storage_format matrix_storage_bytes worker_id worker_pid worker_reused input_cache_hit
worker_ready_wall_s device extra_json""".split()
FINALIZATION_FIELDS=['attempt_id','serialization_s','trial_end_to_end_s','note','extra_json']

def plan(core):
    rows=[]
    for solver in core['solvers']:
        for item in core['instances']:
            name=item['entry']['instance_id']
            for index,seed in enumerate(core['seeds']):
                key=dict(solver=solver,instance=name,seed=seed,budget=core['budget_s'],
                         parameters=digest(core['solvers'][solver]),
                         problem=item['normalized_problem_sha256'],policy=digest(core['protocol']['execution']),
                         reference=core['reference_snapshot_sha256'],environment=core['environment_hash'],
                         code=core['source']['code_snapshot_hash'])
                rows.append(dict(run_id=digest(key)[:24],solver_id=solver,instance_id=name,
                                 seed_index=index,seed=seed,execution_order_index=len(rows)))
    return rows

def _read_attempt(path):
    wrapper=read(path)
    if wrapper.get('sha256')!=digest(wrapper['payload']):raise ValueError(f'Corrupt attempt journal: {path}')
    payload=wrapper['payload']
    if payload['run'].get('schema_version')!=SCHEMA or set(payload['run'])!=set(RUN_FIELDS):
        raise ValueError(f'Attempt schema conflict: {path}')
    return payload

def iter_attempts(directory):
    """Read and validate at most one complete solution/history payload at a time."""
    for path in sorted((Path(directory)/'attempts').glob('*.json')):
        yield _read_attempt(path)

def compact_attempt(payload):
    """Keep scientific run metadata in RAM; large traces live in the durable journal."""
    return dict(run=payload['run'],attempt_number=payload['attempt_number'])

def load_attempts(directory,compact=False):
    records=[compact_attempt(a) if compact else a for a in iter_attempts(directory)]
    records.sort(key=lambda r:(r['run']['execution_order_index'],r['attempt_number']))
    return records

def primary(attempts):
    selected={}
    for item in attempts:
        row=item['run']
        # Interrupted partial work is never a terminal scientific trial.
        if row['result_status']!='interrupted':selected.setdefault(row['run_id'],item)
    return selected

def _attempt_path(directory,payload):
    return Path(directory)/'attempts'/f"{payload['run']['run_id']}__{payload['attempt_number']:04d}.json"

def commit(directory,payload):
    path=_attempt_path(directory,payload)
    if path.exists():raise ValueError('Attempt already committed; refusing overwrite')
    atomic(path,dict(sha256=digest(payload),payload=payload))

def statistics_fields(prefix,values):
    values=[v for v in values if v is not None]
    return {prefix+'_'+k:v for k,v in dict(count=len(values),min=min(values) if values else None,
        median=statistics.median(values) if values else None,mean=statistics.mean(values) if values else None,
        std=statistics.pstdev(values) if values else None,max=max(values) if values else None).items()}

def summaries(experiment,attempts):
    selected=primary(attempts);groups=defaultdict(list)
    attempts_by_group=defaultdict(list)
    for attempt in attempts:
        attempts_by_group[(attempt['run']['solver_id'],attempt['run']['instance_id'])].append(attempt)
    for row in experiment['execution_plan']:groups[(row['solver_id'],row['instance_id'])].append(row)
    result=[]
    for (solver,instance),scheduled in groups.items():
        records=[selected[r['run_id']]['run'] for r in scheduled if r['run_id'] in selected]
        valid=[r for r in records if r['result_status']=='completed']
        hits=sum(bool(r['success_1pct']) for r in valid);total=len(scheduled);terminal=len(records)
        all_attempts=attempts_by_group[(solver,instance)]
        row=dict(schema_version=SCHEMA,test_id=experiment['test_id'],campaign_id=experiment['campaign_id'],
            solver_id=solver,instance_id=instance,budget_s=experiment['core']['budget_s'],
            execution_policy_hash=digest(experiment['core']['protocol']['execution']),
            **{k:experiment['core']['protocol']['execution'][k] for k in ('worker_mode','seed_workers','execution_mode')},
            actual_devices_json=sorted({r['actual_device'] for r in records if r.get('actual_device')}),
            hardware_id=experiment['environment']['hardware_id'],parameters_hash=digest(experiment['core']['solvers'][solver]),
            planned=total,terminal=terminal,pending=total-terminal,attempted=len(all_attempts),
            completed_valid=len(valid),failures=terminal-len(valid),skipped=0,
            history_available_count=sum(bool(r['history_available']) for r in records),
            valid_score_count=sum(bool(r['in_budget_score_available']) for r in records),
            near_target_hits=hits,success_denominator=total,success_rate=hits/total if terminal==total else None,
            provisional_hits_per_terminal=hits/terminal if terminal else None,
            completion_coverage=terminal/total,rate_status='final' if terminal==total else 'provisional',
            conditional_success_among_valid=hits/len(valid) if valid else None,
            reference_hits=sum(bool(r['success_reference']) for r in valid),
            optimum_hits=sum(bool(r['exact_optimum_reached']) for r in valid),
            reference_hit_rate=sum(bool(r['success_reference']) for r in valid)/total if terminal==total else None,
            optimum_hit_rate=(sum(bool(r['exact_optimum_reached']) for r in valid)/total
                              if terminal==total and all(r['exact_optimum_reached'] is not None for r in records) else None),
            validation_alarms=sum(bool(r['optimum_validation_alarm']) for r in records),
            candidate_improvements=sum(bool(r['reference_beaten']) for r in records),
            retry_attempts=sum(a['attempt_number']>1 for a in all_attempts),
            ttt_1pct_conditional_median=statistics.median([r['ttt_1pct_s'] for r in valid if r['ttt_1pct_s'] is not None])
                                      if any(r['ttt_1pct_s'] is not None for r in valid) else None,
            ttt_1pct_hit_count=sum(r['ttt_1pct_s'] is not None for r in valid))
        for field in ('best_objective_within_budget','signed_gap','signed_gap_percent','actual_solve_wall_s','overshoot_s',
                      'cpu_rss_peak_sampled_bytes','gpu_allocated_peak_bytes','gpu_reserved_peak_bytes'):
            row.update(statistics_fields(field,[r[field] for r in (valid if field in
                       ('best_objective_within_budget','signed_gap','signed_gap_percent') else records)]))
        result.append(row)
    return result

def _csv_row(row):
    return {key:canonical(value) if isinstance(value,(dict,list,tuple)) else value
            for key,value in row.items()}

def _extensible_row(row,fields):
    """Keep setup/diagnostic extensions without growing CSV headers mid-campaign."""
    known={key:row.get(key) for key in fields if key!='extra_json'}
    known['extra_json']={key:value for key,value in row.items() if key not in fields}
    if row.get('extra_json'):known['extra_json'].update(row['extra_json'])
    return known

def append_csv(path,rows,fields):
    """Append only new rows. A single campaign writer owns these repairable exports."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists() and path.stat().st_size>0
    if exists:
        with path.open(newline='',encoding='utf-8') as handle:
            if next(csv.reader(handle),None)!=fields:
                raise ValueError(f'CSV header conflict; rebuild projections from journal: {path}')
    with path.open('a',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,quoting=csv.QUOTE_ALL,extrasaction='raise')
        if not exists:writer.writeheader()
        for row in rows:writer.writerow(_csv_row(row))
        handle.flush();os.fsync(handle.fileno())

def _error_line(payload):
    row=payload['run']
    return f"{row['attempt_id']} {row['result_status']}: {row['error_message']}\n"

def append_projection(directory,experiment,payload):
    """Call once after commit; recovery always rebuilds CSVs from authoritative JSON.

    Do not retry an interrupted append in place: some files may already contain
    it. Resuming/rebuilding first repairs both missing rows and partial appends.
    """
    directory=Path(directory);solver=payload['run']['solver_id']
    if solver not in experiment['core']['solvers']:raise ValueError('Unknown solver in attempt projection')
    path=directory/'solvers'/solver
    append_csv(path/'runs.csv',[payload['run']],RUN_FIELDS)
    append_csv(path/'solutions.csv',payload['solutions'],SOLUTION_FIELDS)
    append_csv(path/'trace.csv',payload['trace'],TRACE_FIELDS)
    append_csv(directory/'warmups.csv',[_extensible_row(payload['warmup'],WARMUP_FIELDS)],WARMUP_FIELDS)
    if payload['run']['result_status']!='completed':
        with (path/'errors.log').open('a',encoding='utf-8') as handle:
            handle.write(_error_line(payload));handle.flush();os.fsync(handle.fileno())

def append_finalization(directory,attempt_id,info):
    """Durably record measured overhead after the attempt/projection writes finish."""
    directory=Path(directory)
    atomic(directory/'finalization'/f'{attempt_id}.json',info)
    append_csv(directory/'finalization.csv',
        [_extensible_row(dict(attempt_id=attempt_id,**info),FINALIZATION_FIELDS)],FINALIZATION_FIELDS)

def refresh_summaries(directory,experiment,attempts):
    summary=summaries(experiment,attempts);directory=Path(directory)
    for solver in experiment['core']['solvers']:
        table(directory/'solvers'/solver/'summary.csv',[s for s in summary if s['solver_id']==solver])
    table(directory/'summary.csv',summary)
    return summary

@contextmanager
def csv_stream(path,fields):
    """Atomic streaming replacement without retaining the table's rows in RAM."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    try:
        with temp.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,quoting=csv.QUOTE_ALL,extrasaction='raise')
            writer.writeheader()
            yield lambda row:writer.writerow(_csv_row(row))
            handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)

def materialize(directory,experiment,attempts):
    """Full projection repair at startup/finalization, never after every trial.

    Compact records are hydrated one at a time, so resident memory is independent
    of the campaign's total solution/history size.
    """
    directory=Path(directory)
    with ExitStack() as stack:
        writers={};errors={}
        for solver in experiment['core']['solvers']:
            path=directory/'solvers'/solver
            writers[solver]={name:stack.enter_context(csv_stream(path/(name+'.csv'),fields))
                for name,fields in (('runs',RUN_FIELDS),('solutions',SOLUTION_FIELDS),('trace',TRACE_FIELDS))}
            errors[solver]=stack.enter_context((path/'errors.log').open('w',encoding='utf-8'))
        warmup=stack.enter_context(csv_stream(directory/'warmups.csv',WARMUP_FIELDS))
        finalization=stack.enter_context(csv_stream(directory/'finalization.csv',FINALIZATION_FIELDS))
        for item in attempts:
            payload=item if 'trace' in item else _read_attempt(_attempt_path(directory,item))
            own=writers[payload['run']['solver_id']]
            own['runs'](payload['run'])
            for row in payload['solutions']:own['solutions'](row)
            for row in payload['trace']:own['trace'](row)
            warmup(_extensible_row(payload['warmup'],WARMUP_FIELDS))
            if payload['run']['result_status']!='completed':errors[payload['run']['solver_id']].write(_error_line(payload))
            aid=payload['run']['attempt_id'];path=directory/'finalization'/f'{aid}.json'
            if path.exists():finalization(_extensible_row(dict(attempt_id=aid,**read(path)),FINALIZATION_FIELDS))
    return refresh_summaries(directory,experiment,attempts)

def check_resume_exports(directory,experiment,attempts):
    """Validate immutable journal identity and detect damaged materialized CSVs."""
    expected={r['run_id']:r for r in experiment['execution_plan']};seen=set()
    for attempt in attempts:
        row=attempt['run'];key=(row['run_id'],attempt['attempt_number'])
        if key in seen:raise ValueError('Duplicate attempt identity')
        seen.add(key)
        if row['run_id'] not in expected or any(row[k]!=v for k,v in expected[row['run_id']].items()):
            raise ValueError('Attempt differs from immutable execution plan')
        if row['parameters_hash']!=digest(experiment['core']['solvers'][row['solver_id']]):
            raise ValueError('Attempt parameter hash mismatch')
    for solver in experiment['core']['solvers']:
        for filename,fields in [('runs.csv',RUN_FIELDS),('solutions.csv',SOLUTION_FIELDS),('trace.csv',TRACE_FIELDS)]:
            path=Path(directory)/'solvers'/solver/filename
            issue=None
            if path.exists():
                try:
                    with path.open(newline='',encoding='utf-8') as handle:
                        rows=csv.reader(handle,strict=True)
                        if next(rows,None)!=fields:issue='schema/header conflict'
                        elif any(len(row)!=len(fields) for row in rows):issue='truncated/malformed row'
                except (csv.Error,UnicodeError):issue='invalid CSV encoding/quoting'
            else:issue='missing CSV'
            if issue:
                with (Path(directory)/'csv_recovery.log').open('a',encoding='utf-8') as handle:
                    handle.write(f'{utc()} {path.name}: {issue}; rebuilding from validated attempt journal\n')

"""Durable per-attempt records are authoritative; CSVs are repairable materializations."""
from collections import defaultdict
from pathlib import Path
import statistics
import csv
from .common import SCHEMA,atomic,read,digest,table,flattened,utc

RUN_FIELDS = """
schema_version campaign_id test_id run_id attempt_id instance_id problem_family n_variables density_category
measured_density n_couplings nnz_diagonal nnz_matrix matrix_storage_format matrix_storage_bytes
raw_source_sha256 normalized_problem_sha256 catalog_sha256 reference_snapshot_sha256 index_mapping_id objective_offset
solver_id solver_version adapter_version code_commit code_dirty code_snapshot_hash parameters_hash parameters_json
execution_policy_hash seed_index seed rng_backend initialization_policy initialization_hash seed_effective
worker_mode seed_workers execution_mode worker_id worker_pid worker_reused input_cache_hit worker_trial_index wave_id wave_size
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

def load_attempts(directory):
    records=[]
    for path in sorted((Path(directory)/'attempts').glob('*.json')):
        wrapper=read(path)
        if wrapper.get('sha256')!=digest(wrapper['payload']):raise ValueError(f'Corrupt attempt journal: {path}')
        payload=wrapper['payload']
        if payload['run'].get('schema_version')!=SCHEMA or set(payload['run'])!=set(RUN_FIELDS):
            raise ValueError(f'Attempt schema conflict: {path}')
        records.append(payload)
    records.sort(key=lambda r:(r['run']['execution_order_index'],r['attempt_number']))
    return records

def primary(attempts):
    selected={}
    for item in attempts:
        row=item['run']
        # Interrupted partial work is never a terminal scientific trial.
        if row['result_status']!='interrupted':selected.setdefault(row['run_id'],item)
    return selected

def commit(directory,payload):
    path=Path(directory)/'attempts'/f"{payload['run']['run_id']}__{payload['attempt_number']:04d}.json"
    if path.exists():raise ValueError('Attempt already committed; refusing overwrite')
    atomic(path,dict(sha256=digest(payload),payload=payload))

def statistics_fields(prefix,values):
    values=[v for v in values if v is not None]
    return {prefix+'_'+k:v for k,v in dict(count=len(values),min=min(values) if values else None,
        median=statistics.median(values) if values else None,mean=statistics.mean(values) if values else None,
        std=statistics.pstdev(values) if values else None,max=max(values) if values else None).items()}

def summaries(experiment,attempts):
    selected=primary(attempts);groups=defaultdict(list)
    for row in experiment['execution_plan']:groups[(row['solver_id'],row['instance_id'])].append(row)
    result=[]
    for (solver,instance),scheduled in groups.items():
        records=[selected[r['run_id']]['run'] for r in scheduled if r['run_id'] in selected]
        valid=[r for r in records if r['result_status']=='completed']
        hits=sum(bool(r['success_1pct']) for r in valid);total=len(scheduled);terminal=len(records)
        ids={r['run_id'] for r in scheduled}
        all_attempts=[a for a in attempts if a['run']['run_id'] in ids]
        row=dict(schema_version=SCHEMA,test_id=experiment['test_id'],campaign_id=experiment['campaign_id'],
            solver_id=solver,instance_id=instance,budget_s=experiment['core']['budget_s'],
            execution_policy_hash=digest(experiment['core']['protocol']['execution']),
            **{k:experiment['core']['protocol']['execution'][k] for k in ('worker_mode','seed_workers','execution_mode')},
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

def materialize(directory,experiment,attempts):
    directory=Path(directory);summary=summaries(experiment,attempts)
    for solver in experiment['core']['solvers']:
        own=[a for a in attempts if a['run']['solver_id']==solver];path=directory/'solvers'/solver
        table(path/'runs.csv',[a['run'] for a in own],RUN_FIELDS)
        table(path/'solutions.csv',[s for a in own for s in a['solutions']],SOLUTION_FIELDS)
        table(path/'trace.csv',[t for a in own for t in a['trace']],TRACE_FIELDS)
        table(path/'summary.csv',[s for s in summary if s['solver_id']==solver])
        with (path/'errors.log').open('w',encoding='utf-8') as handle:
            for a in own:
                if a['run']['result_status']!='completed':
                    handle.write(f"{a['run']['attempt_id']} {a['run']['result_status']}: {a['run']['error_message']}\n")
    table(directory/'summary.csv',summary)
    table(directory/'warmups.csv',[a['warmup'] for a in attempts])
    # Durations are measured after the atomic attempt commit; never guessed in a row.
    table(directory/'finalization.csv',[dict(attempt_id=a['run']['attempt_id'],**read(path))
          for a in attempts if (path:=directory/'finalization'/f"{a['run']['attempt_id']}.json").exists()])
    return summary

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

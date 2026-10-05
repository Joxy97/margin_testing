"""Offline per-instance aggregation, keeping configuration/hardware identities separate."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from .common import read,table,atomic,digest
from .storage import (iter_attempts,compact_attempt,summaries,csv_stream,
                      RUN_FIELDS,TRACE_FIELDS,SOLUTION_FIELDS)

IDENTITY_FIELDS='''test_id run_id attempt_id instance_id solver_id hardware_id parameters_hash
execution_policy_hash execution_mode seed_workers worker_mode actual_device workers_on_device'''.split()
QUALITY_FIELDS=IDENTITY_FIELDS+'''requested_time_s best_objective_within_budget signed_gap_percent result_status'''.split()
TIMING_FIELDS=IDENTITY_FIELDS+'''requested_time_s actual_solve_wall_s ttt_1pct_s ttt_reference_s ttt_optimum_s
near_target_event censor_time_s near_target_time_status result_status'''.split()
MEMORY_FIELDS=IDENTITY_FIELDS+'''n_variables n_couplings cpu_rss_peak_sampled_bytes gpu_allocated_peak_bytes
gpu_reserved_peak_bytes result_status'''.split()
LINK_FIELDS='''test_id campaign_id execution_policy_hash execution_mode seed_workers worker_mode
actual_device workers_on_device hardware_id parameters_hash instance_id solver_id requested_time_s'''.split()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root');p.add_argument('--output',required=True)
    args=p.parse_args(argv);root=Path(args.root);out=Path(args.output)
    if out.exists() and any(out.iterdir()):p.error('Choose a new/empty analysis directory')
    experiments=sorted(root.rglob('experiment.json'))
    rows=[];seen={};duplicates=[];run_count=0
    # Only one experiment's small run records and one attempt's history are resident.
    # Temporary CSVs replace outputs only after every journal passed validation.
    with ExitStack() as stack:
        runs=stack.enter_context(csv_stream(out/'runs.csv',RUN_FIELDS))
        trace=stack.enter_context(csv_stream(out/'trace.csv',TRACE_FIELDS+LINK_FIELDS))
        solutions=stack.enter_context(csv_stream(out/'solutions.csv',SOLUTION_FIELDS+LINK_FIELDS))
        quality=stack.enter_context(csv_stream(out/'quality.csv',QUALITY_FIELDS))
        timing=stack.enter_context(csv_stream(out/'timing.csv',TIMING_FIELDS))
        memory=stack.enter_context(csv_stream(out/'memory.csv',MEMORY_FIELDS))
        for path in experiments:
            experiment=read(path);key=experiment['test_id'];attempts=[];hashes=[]
            duplicate=key in seen
            for attempt in iter_attempts(path.parent):
                row=attempt['run'];attempts.append(compact_attempt(attempt))
                hashes.append((row['execution_order_index'],attempt['attempt_number'],digest(attempt)))
                if duplicate:continue
                runs(row);run_count+=1
                link={k:row.get(k) for k in LINK_FIELDS}
                for event in attempt['trace']:trace(dict(event,**link))
                for solution in attempt['solutions']:solutions(dict(solution,**link))
                quality({k:row.get(k) for k in QUALITY_FIELDS})
                timing({k:row.get(k) for k in TIMING_FIELDS})
                memory({k:row.get(k) for k in MEMORY_FIELDS})
            identity=digest(dict(experiment=experiment,attempts=sorted(hashes)))
            if duplicate:
                if seen[key]!=identity:raise ValueError(f'Conflicting experiment identity {key}')
                duplicates.append(str(path));continue
            seen[key]=identity
            attempts.sort(key=lambda a:(a['run']['execution_order_index'],a['attempt_number']))
            rows.extend(summaries(experiment,attempts))
    table(out/'summary.csv',rows)
    atomic(out/'aggregation.json',dict(experiments=len(seen),duplicate_copies_ignored=duplicates,
        partial_groups=sum(r['pending']>0 for r in rows),
        policy='Per-instance/solver/budget/parameter/hardware summaries only. First non-interrupted attempt primary; retries retained as diagnostics. No pooled scaling claim.'))
    print(f'{len(seen)} experiments; {run_count} attempts; output={out.resolve()}')
    return 0

"""Offline per-instance aggregation, keeping configuration/hardware identities separate."""
import argparse
from pathlib import Path
from .common import read,table,atomic,digest
from .storage import load_attempts,summaries

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root');p.add_argument('--output',required=True)
    args=p.parse_args(argv);root=Path(args.root);out=Path(args.output)
    if out.exists() and any(out.iterdir()):p.error('Choose a new/empty analysis directory')
    experiments=sorted(root.rglob('experiment.json'))
    rows=[];runs=[];trace=[];solutions=[];seen={};duplicates=[]
    for path in experiments:
        experiment=read(path);attempts=load_attempts(path.parent);key=experiment['test_id']
        identity=digest(dict(experiment=experiment,attempts=attempts))
        if key in seen:
            if seen[key]!=identity:raise ValueError(f'Conflicting experiment identity {key}')
            duplicates.append(str(path));continue
        seen[key]=identity
        rows.extend(summaries(experiment,attempts))
        for attempt in attempts:
            runs.append(attempt['run'])
            link=dict(test_id=key,campaign_id=experiment['campaign_id'],
                      hardware_id=experiment['environment']['hardware_id'],
                      parameters_hash=attempt['run']['parameters_hash'],instance_id=attempt['run']['instance_id'],
                      solver_id=attempt['run']['solver_id'],requested_time_s=attempt['run']['requested_time_s'])
            trace.extend(dict(r,**link) for r in attempt['trace'])
            solutions.extend(dict(r,**link) for r in attempt['solutions'])
    table(out/'summary.csv',rows);table(out/'runs.csv',runs)
    table(out/'trace.csv',trace);table(out/'solutions.csv',solutions)
    table(out/'quality.csv',[{k:r.get(k) for k in ('test_id','run_id','attempt_id','instance_id','solver_id','hardware_id',
        'parameters_hash','requested_time_s','best_objective_within_budget','signed_gap_percent','result_status')} for r in runs])
    table(out/'timing.csv',[{k:r.get(k) for k in ('test_id','run_id','attempt_id','instance_id','solver_id','hardware_id',
        'parameters_hash','requested_time_s','actual_solve_wall_s','ttt_1pct_s','ttt_reference_s','ttt_optimum_s',
        'near_target_event','censor_time_s','near_target_time_status','result_status')} for r in runs])
    table(out/'memory.csv',[{k:r.get(k) for k in ('test_id','run_id','attempt_id','instance_id','solver_id','hardware_id',
        'parameters_hash','n_variables','n_couplings','cpu_rss_peak_sampled_bytes','gpu_allocated_peak_bytes',
        'gpu_reserved_peak_bytes','result_status')} for r in runs])
    atomic(out/'aggregation.json',dict(experiments=len(seen),duplicate_copies_ignored=duplicates,
        partial_groups=sum(r['pending']>0 for r in rows),
        policy='Per-instance/solver/budget/parameter/hardware summaries only. First non-interrupted attempt primary; retries retained as diagnostics. No pooled scaling claim.'))
    print(f'{len(seen)} experiments; {len(runs)} attempts; output={out.resolve()}')
    return 0

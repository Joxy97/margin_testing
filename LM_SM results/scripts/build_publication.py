"""Validate saved campaigns; export settings and six SM23 figures. No trials run."""
from pathlib import Path
import argparse
import csv
import json
import hashlib
import numpy as np
from campaign_validation import load_campaign, checked

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader(); writer.writerows(rows)

def figures(campaign):
    # Use the original renderer so typography, layout and scales match LM23.
    import subprocess
    p = campaign['manifest']['problems'][0]
    subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                    '-File', str(ROOT/'scripts/plot_sm23.ps1'),
                    '-VariableCount', str(p['n']), '-DensityCategory', p['density'],
                    '-BaseDir', str(ROOT), '-OutputDir', str(ROOT/'PLots')], check=True)
    return [f"PLots/200_{p['density']}_{metric}_SM23.png" for metric in
            ['exact_hits', 'hits_within_1pct', 'gap_to_reference_percent']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only',action='store_true'); args=parser.parse_args()
    summary=[]; settings_rows=[]; outputs=[]
    paths=(sorted(ROOT.glob('*_*/manifest.json')) +
           sorted(ROOT.glob('*_*/LM23/manifest.json')) +
           sorted(ROOT.glob('*_*/SM23/manifest.json')))
    for path in paths:
        m=json.loads(path.read_bytes()); c=load_campaign(path,set(m['solvers']))
        name=path.parent.relative_to(ROOT).as_posix(); p=m['problems'][0]
        summary.append(dict(campaign=name,solvers=len(m['solvers']),trials=c['terminal'],failures=c['failures'],
                            manifest_sha256=c['identity'],complete=c['complete']))
        print(name,c['terminal'],'trials;',c['failures'],'failures',flush=True)
        if args.validate_only: continue
        directory=path.parent/'solver_settings'
        write_json(directory/'experiment.json',dict(manifest_sha256=c['identity'],**m))
        write_json(directory/'solver_parameters.json',dict(schema_version=2,
                   description='Exact recorded per-solver parameters; see experiment.json for scheduling and environment.',solvers=m['solvers']))
        for solver, params in m['solvers'].items():
            for parameter,value in params.items():
                settings_rows.append(dict(campaign=name,solver=solver,parameter=parameter,value_json=json.dumps(value),
                    trials=len(m['seeds']),budget_seconds=m['execution']['budget_seconds'],
                    wall_schedule=solver in m['execution']['wall_schedule_solvers']))
        if path.parent.name=='SM23':
            near=[]; timings=[]
            for solver in sorted(m['solvers']):
                for t,hits in zip(m['checkpoints_seconds'],c['curves'][solver]['near_hits']):
                    near.append(dict(solver=solver,problem=p['id'],checkpoint_seconds=t,hits=int(hits),planned_seeds=100,hit_probability=float(hits)/100))
                progress=checked(path.parent/'progress'/solver/(p['id']+'.json'))
                for i,seed in enumerate(m['seeds']):
                    timings.append(dict(solver=solver,problem=p['id'],seed=seed,time_to_optimum_s=progress['time_to_optimum_s'][i],
                                        status=progress['status'][i],reference_type=p['reference_type']))
            write_csv(path.parent/'postprocessed/hits_within_1pct.csv',list(near[0]),near)
            write_csv(path.parent/'postprocessed/time_to_optimum.csv',list(timings[0]),timings)
            outputs.extend(figures(c))
    if not args.validate_only:
        write_csv(ROOT/'reproducibility/all_solver_settings.csv',list(settings_rows[0]),settings_rows)
        write_json(ROOT/'reproducibility/validation_report.json',dict(campaigns=summary,total_trials=sum(c['trials'] for c in summary),
            total_failures=sum(c['failures'] for c in summary),new_figures=outputs,
            checks=['manifest and checked metadata identities','all raw row SHA256','aggregate SHA256',
                    'aggregate metrics recomputed from raw','monotone checkpoint energies','optimum timing and carry-forward',
                    '100 seeds and nine checkpoints per solver']))

if __name__=='__main__': main()

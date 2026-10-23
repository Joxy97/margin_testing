"""Audit the completed SM20 addition and refresh its publication indexes only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np
from campaign_validation import load_campaign, checked, digest

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    choices = [ROOT / name / 'SM20' for name in ('codev4_200_dense', 'dense_200')]
    candidates = [p for p in choices if (p / 'manifest.json').is_file()]
    if len(candidates) != 1:
        raise ValueError('Expected one SM20 dense_200 campaign')
    folder = candidates[0]
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    campaign = load_campaign(folder / 'manifest.json', set(manifest['solvers']))
    assert len(manifest['solvers']) == 20 and campaign['terminal'] == 2000
    assert not campaign['failures'] and campaign['complete']
    settings = json.loads((folder / 'solver_settings/experiment.json').read_bytes())
    assert settings == dict(manifest_sha256=digest(manifest), **manifest)
    parameters = json.loads((folder / 'solver_settings/solver_parameters.json').read_bytes())
    assert parameters['solvers'] == manifest['solvers']
    problem = manifest['problems'][0]
    assert problem['id'] == 'gka5e' and problem['reference'] == -35154
    reference = problem['reference']
    quality = []
    for solver in manifest['solvers']:
        matrix = np.load(folder / 'raw' / solver / 'gka5e.npy', allow_pickle=False)
        final = matrix[:, -1]
        assert np.all(np.isfinite(final))
        gap = 100 * (final-reference) / abs(reference)
        quality.append(dict(solver=solver,problem='gka5e',checkpoint_seconds=20,
            planned_seeds=100,valid_seeds=100,exact_hits=int((final==reference).sum()),
            hits_within_1pct=int((final<=reference+.01*abs(reference)).sum()),
            best_energy=float(final.min()),mean_energy=float(final.mean()),worst_energy=float(final.max()),
            best_gap_percent=float(gap.min()),mean_gap_percent=float(gap.mean()),worst_gap_percent=float(gap.max())))
    with (folder/'postprocessed/hits_within_1pct.csv').open(newline='',encoding='utf-8') as handle:
        near = list(csv.DictReader(handle))
    expected_near = {(solver,float(t)):int(count) for solver in manifest['solvers']
        for t,count in zip(manifest['checkpoints_seconds'],campaign['curves'][solver]['near_hits'])}
    assert len(near)==180
    assert {(r['solver'],float(r['checkpoint_seconds'])):int(r['hits']) for r in near}==expected_near
    with (folder/'postprocessed/time_to_optimum.csv').open(newline='',encoding='utf-8') as handle:
        timings = list(csv.DictReader(handle))
    assert len(timings)==2000
    for row in timings:
        progress=checked(folder/'progress'/row['solver']/'gka5e.json')
        i=manifest['seeds'].index(int(row['seed']))
        actual=None if row['time_to_optimum_s']=='' else float(row['time_to_optimum_s'])
        assert actual==progress['time_to_optimum_s'][i] and row['status']==progress['status'][i]
    summary=dict(campaign=folder.relative_to(ROOT).as_posix(),solvers=20,trials=2000,
        failures=0,manifest_sha256=campaign['identity'],complete=True)
    if args.validate_only:
        print(json.dumps(dict(status='passed',**summary),indent=2))
        return
    write_csv(folder/'postprocessed/final_quality_summary.csv',list(quality[0]),quality)
    fields=['campaign','solver','parameter','value_json','trials','budget_seconds','wall_schedule']
    settings_path=ROOT/'reproducibility/all_solver_settings.csv'
    with settings_path.open(newline='',encoding='utf-8') as handle:
        rows=[r for r in csv.DictReader(handle) if r['campaign']!=summary['campaign']]
    for solver,params in manifest['solvers'].items():
        for parameter,value in params.items():
            rows.append(dict(campaign=summary['campaign'],solver=solver,parameter=parameter,
                value_json=json.dumps(value),trials=100,budget_seconds=20.0,wall_schedule=False))
    write_csv(settings_path,fields,rows)
    report_path=ROOT/'reproducibility/validation_report.json'
    report=json.loads(report_path.read_bytes())
    report['campaigns']=[c for c in report['campaigns'] if c['campaign']!=summary['campaign']]+[summary]
    report['total_trials']=sum(c['trials'] for c in report['campaigns'])
    report['total_failures']=sum(c['failures'] for c in report['campaigns'])
    for metric in ('exact_hits','hits_within_1pct','gap_to_reference_percent'):
        name=f'PLots/200_dense_{metric}_SM20.png'
        assert (ROOT/name).is_file()
        if name not in report['new_figures']: report['new_figures'].append(name)
    write_json(report_path,report)
    print(json.dumps(dict(status='passed',**summary,total_published_trials=report['total_trials'],
        exact_hits=sum(r['exact_hits'] for r in quality),
        hits_within_1pct=sum(r['hits_within_1pct'] for r in quality)),indent=2))


if __name__=='__main__':
    main()

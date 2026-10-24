"""Read-only SM86 collection audit; no solver trials or result writes.

Validates four preserved original campaigns and the non-resumable flat view.
Run on either the local results layout or the published density_size layout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from campaign_validation import CHECKPOINTS, checked, digest, load_campaign

SCHEMA = 'sm86_collection_v1'


INPUT_SHA = '3a2d6521caefa62a867a7b96dae169388b14088b1ba309dbcdef989b9460d375'


SOURCE_SHA = 'ecdeb6896f4cc85ecf8e3b47abd7836f39ecf15c9c73dfa99609320018f61ede'


HIT_COLUMNS = ('solver', 'problem', 'checkpoint_seconds', 'hits', 'planned_seeds', 'hit_probability')


TIME_COLUMNS = ('solver', 'problem', 'seed', 'time_to_optimum_s', 'status', 'reference_type')


QUALITY_COLUMNS = ('solver', 'problem', 'checkpoint_seconds', 'planned_seeds', 'valid_seeds',
                   'exact_hits', 'hits_within_1pct', 'best_energy', 'mean_energy', 'worst_energy',
                   'best_gap_percent', 'mean_gap_percent', 'worst_gap_percent')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, name):
    result = (root / name).resolve()
    require(result.is_relative_to(root.resolve()), f'Path escapes archive root: {name}')
    return result


def verify_inventory(folder):
    inventory = read(folder / 'download_manifest.json')
    require(isinstance(inventory.get('files'), dict), 'Missing transport file inventory')
    for name, expected in inventory['files'].items():
        path = inside(folder, name)
        require(path.is_file() and not path.is_symlink(), f'Missing inventory file: {path}')
        require(sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes'],
                f'Transport file checksum/size mismatch: {path}')
    return inventory


def derive_exports(manifest, target):
    hits, timings, quality = [], [], []
    reference = manifest['problems'][0]['reference']
    for solver in sorted(manifest['solvers']):
        matrix = np.load(target / 'raw' / solver / 'gka5e.npy', allow_pickle=False)
        progress = checked(target / 'progress' / solver / 'gka5e.json')
        near = (np.isfinite(matrix) & (matrix <= reference + .01 * abs(reference))).sum(axis=0)
        for seconds, count in zip(CHECKPOINTS, near):
            hits.append(dict(zip(HIT_COLUMNS, (solver, 'gka5e', seconds, int(count), 100, int(count) / 100))))
        for seed, timing, status in zip(manifest['seeds'], progress['time_to_optimum_s'], progress['status']):
            timings.append(dict(zip(TIME_COLUMNS, (solver, 'gka5e', seed, timing, status, 'OPTIMUM'))))
        final = matrix[:, -1]
        require(np.isfinite(final).all(), f'Completed successful solver has missing final energy: {solver}')
        gap = 100 * (final - reference) / abs(reference)
        quality.append(dict(zip(QUALITY_COLUMNS, (
            solver, 'gka5e', 20, 100, 100, int((final == reference).sum()), int(near[-1]),
            float(final.min()), float(final.mean()), float(final.max()),
            float(gap.min()), float(gap.mean()), float(gap.max())))))
    return hits, timings, quality


def validate_collection(target):
    target = Path(target).resolve()
    manifest = read(target / 'manifest.json')
    require(manifest['schema_version'] == SCHEMA and manifest['resumable'] is False,
            'SM86 must be an explicitly non-resumable derived collection')
    require(len(manifest['solvers']) == 78 and len(manifest['partitions']) == 4, 'Incorrect collection size')
    require(manifest['inputs'] == {'gka5e':INPUT_SHA} and manifest['source']['snapshot'] == SOURCE_SHA,
            'Collection source/input identity differs')
    require(manifest['seeds'] == list(range(100)) and tuple(manifest['checkpoints_seconds']) == CHECKPOINTS,
            'Collection sampling differs')
    identity, seen, total, failures, preserved = digest(manifest), set(), 0, 0, 0
    state, completion = checked(target / 'state.json'), checked(target / 'completion.json')
    require(state['manifest_sha256'] == completion['manifest_sha256'] == identity,
            'Derived collection checked metadata differs')
    require(state['complete'] and completion['complete'] and state['schema_version'] == completion['schema_version'] == SCHEMA,
            'Derived completion record is invalid')
    for part in manifest['partitions']:
        original_root = inside(target, part['results_path'])
        c = load_campaign(original_root / 'manifest.json', set(part['solvers']))
        require(c['identity'] == part['manifest_sha256'] and sha(original_root/'manifest.json') == part['manifest_file_sha256'],
                'Original manifest identity differs from collection provenance')
        require(all(part[key] == c['manifest'][key] for key in
                    ('source','configuration','catalog','environment','execution')),
                'Partition settings or environment differ from original manifest')
        require(c['complete'] and not c['failures'], 'Original partition is not successfully complete')
        require(not seen & set(c['manifest']['solvers']), 'Duplicate solver between partition originals')
        seen.update(c['manifest']['solvers'])
        total += c['terminal']; failures += c['failures']
        inventory_root = original_root.parent.parent
        verify_inventory(inventory_root)
        for solver, params in c['manifest']['solvers'].items():
            require(manifest['solvers'][solver] == params, 'Copied solver parameters changed')
            origin = manifest['solver_origin'][solver]
            require(origin['partition_id'] == part['partition_id'] and origin['manifest_sha256'] == c['identity']
                    and origin['results_path'] == part['results_path'],
                    'Solver origin does not identify its original partition')
            for directory, suffix in (('raw','.npy'),('progress','.json'),('aggregated','.csv')):
                relative = Path(directory)/solver/('gka5e'+suffix)
                require(sha(target/relative) == sha(original_root/relative), 'Flat copy differs: '+str(relative))
                preserved += 1
            require(completion['aggregates'][solver]['gka5e'] == sha(target/'aggregated'/solver/'gka5e.csv'),
                    'Derived aggregate checksum differs')
    require(seen == set(manifest['solvers']) and total == 7800 and failures == 0, 'Collection coverage differs')
    require(state['planned'] == state['terminal'] == completion['planned'] == completion['terminal'] == total,
            'Derived completed trial counts differ')
    require(state['failures'] == completion['failures'] == failures, 'Derived failure counts differ')
    for directory, suffix in (('raw','.npy'),('progress','.json'),('aggregated','.csv')):
        actual = {p.parent.name for p in (target/directory).glob('*/*') if p.name == 'gka5e'+suffix}
        require(actual == seen, 'Flat result inventory differs: '+directory)
    transport = read(target/'reproducibility/download_identity.json')['partitions']
    require(len(transport) == 4 and {p['partition_id'] for p in transport} == {1,2,3,4},
            'Transport archive identities do not cover four partitions')
    for record in transport:
        archive = inside(target.parents[1]/'reproducibility', record['archive'])
        require(archive.is_file() and sha(archive) == record['sha256'] and archive.stat().st_size == record['bytes'],
                'Preserved original transport archive differs: '+record['archive'])
        inventory_file = target/'reproducibility/campaign'/('partition_'+str(record['partition_id']))/'download_manifest.json'
        require(sha(inventory_file) == record['download_manifest_sha256'], 'Original transport inventory identity differs')
    excluded = read(target/'unsupported.json')['unsupported']
    require(excluded == manifest['unsupported_models'] and len(excluded) == 8, 'Unsupported records differ')
    require(not set(r['solver_id'] for r in excluded) & seen, 'Unsupported entries have trial results')
    require(read(target/'solver_settings/experiment.json') == dict(manifest_sha256=identity,**manifest),
            'Experiment settings differ from collection manifest')
    require(read(target/'solver_settings/solver_parameters.json')['solvers'] == manifest['solvers'],
            'Exact solver parameter export differs')
    rows = derive_exports(manifest, target)
    for name, columns, expected in zip(('hits_within_1pct','time_to_optimum','final_quality_summary'),
                                      (HIT_COLUMNS,TIME_COLUMNS,QUALITY_COLUMNS), rows):
        path = target/'postprocessed'/(name+'.csv')
        with path.open(newline='',encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            require(tuple(reader.fieldnames) == columns, 'Postprocessed CSV columns differ')
            actual = list(reader)
        require(len(actual) == len(expected), 'Postprocessed CSV row count differs')
        for left,right in zip(actual,expected):
            require(left == {key:'' if right[key] is None else str(right[key]) for key in columns},
                    'Postprocessed CSV does not reproduce retained raw data: '+name)
    quality = rows[2]
    return dict(status='passed',schema_version=SCHEMA,resumable=False,collection='SM86',
        partition_count=4,collection_models=86,supported_solvers=78,unsupported_models=8,
        trials=7800,failures=0,complete=True,manifest_sha256=identity,
        source_snapshot=SOURCE_SHA,input_sha256=INPUT_SHA,raw_arrays_verified=78,
        raw_row_checksums_verified=7800,flat_files_byte_identical=preserved,
        raw_numerical_payload_bytes=78*100*9*8,
        postprocessed_rows=dict(hits_within_1pct=702,time_to_optimum=7800,final_quality_summary=78),
        exact_optimum_hits=sum(q['exact_hits'] for q in quality),
        hits_within_one_percent=sum(q['hits_within_1pct'] for q in quality),
        checks=['all original download inventory SHA256 and sizes','four independent original manifest identities',
                'all raw row SHA256','aggregate SHA256 and metric recomputation',
                'monotone incumbents and no below-optimum energies','optimum timing and carry-forward',
                'exact frozen settings and per-partition execution environments',
                '78 disjoint solvers with 100 seeds and nine checkpoints; eight explicit exclusions',
                'flat copies remain byte-identical to their original host result files',
                'postprocessed CSVs independently reconstructed from retained raw data'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, help='SM86 collection directory; default: discover under this results root')
    args = parser.parse_args(argv)
    target = args.target
    if target is None:
        choices = [SCRIPTS.parent / name / 'SM86' for name in ('codev4_200_dense', 'dense_200')]
        present = [path for path in choices if (path / 'manifest.json').is_file()]
        if len(present) != 1:
            parser.error('Expected exactly one local or published SM86 dense_200 directory; supply --target')
        target = present[0]
    try:
        print(json.dumps(validate_collection(target), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'SM86 validation: {error}\n')


if __name__ == '__main__':
    main()

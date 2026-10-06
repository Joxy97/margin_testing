"""Read-only integrity/schema/metric audit of a compact result directory."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from qubo_benchmark.runtime.common import digest, filehash, read
from qubo_benchmark.runtime.compact_metrics import AGGREGATE_COLUMNS, aggregate_energies
from qubo_benchmark.runtime.compact_storage import _read_checked, _row_hash, SUCCESS_STATUSES, TERMINAL_STATUSES


def validate(root):
    root = Path(root)
    manifest = read(root/'manifest.json')
    identity = digest(manifest)
    completion = _read_checked(root/'completion.json')
    state = _read_checked(root/'state.json')
    if (completion['manifest_sha256'] != identity or state['manifest_sha256'] != identity
            or not state['complete']):
        raise ValueError('Result directory is incomplete')
    arrays, payload, has_failures = 0, 0, False
    for solver in manifest['solvers']:
        for problem in manifest['problems']:
            key = problem['id']
            group = _read_checked(root/'progress'/solver/(key+'.json'))
            if (group['manifest_sha256'] != identity or group['solver_id'] != solver or group['problem_id'] != key
                    or len(group['status']) != len(manifest['seeds'])
                    or any(s not in TERMINAL_STATUSES for s in group['status'])):
                raise ValueError('Invalid or incomplete seed progress')
            has_failures |= any(s not in SUCCESS_STATUSES for s in group['status'])
            path = root/'aggregated'/solver/(key+'.csv')
            if filehash(path) != group['aggregate_sha256'] or group['aggregate_sha256'] != completion['aggregates'][solver][key]:
                raise ValueError('Completed aggregate checksum differs')
            with path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != AGGREGATE_COLUMNS:
                    raise ValueError('Extra or missing aggregate metric')
                actual = np.array([[float(row[k]) for k in AGGREGATE_COLUMNS] for row in reader])
            if actual.shape != (9, 4) or not np.array_equal(actual[:, 0], manifest['checkpoints_seconds']):
                raise ValueError('Wrong aggregate checkpoints')
            raw = root/'raw'/solver/(key+'.npy')
            if raw.exists() != bool(manifest['keep_raw_results']):
                raise ValueError('Raw retention does not match completed manifest')
            if raw.exists():
                matrix = np.load(raw, allow_pickle=False)
                if matrix.dtype != np.float64 or matrix.shape != (len(manifest['seeds']), 9):
                    raise ValueError('Raw energies have wrong dtype/shape')
                for index, row in enumerate(matrix):
                    if _row_hash(row) != group['row_sha256'][index]:
                        raise ValueError('Committed energy row checksum differs')
                    present = row[np.isfinite(row)]
                    if np.any(np.diff(present) > 0):
                        raise ValueError('Incumbent energy worsens within a seed')
                rows = aggregate_energies(matrix, manifest['checkpoints_seconds'], problem['reference'],
                    problem['reference_type'], tolerance=0 if problem.get('integer_objective', True) else 1e-9)
                expected = np.array([[row[k] for k in AGGREGATE_COLUMNS] for row in rows])
                if not np.array_equal(actual, expected, equal_nan=True):
                    raise ValueError('Aggregate differs from exact energy recomputation')
                arrays += 1
                payload += matrix.nbytes
    files = list(root.rglob('*'))
    # Only documented metric/progress/plot artifacts are accepted; source
    # witnesses and matrices belong outside this result directory.
    allowed = {'.json', '.npy', '.csv', '.png'}
    for path in files:
        if not path.is_file() or path.name.startswith('.'):
            continue
        if path.suffix not in allowed or any(word in path.name.lower() for word in
                ('bitstring', 'trajectory', 'population', 'telemetry', 'solution_vector')):
            raise ValueError(f'Unexpected result artifact: {path}')
    return dict(status='passed', arrays=arrays, numerical_payload_bytes=payload,
                disk_bytes=sum(p.stat().st_size for p in files if p.is_file()),
                has_failures=bool(has_failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True, allow_nan=False))

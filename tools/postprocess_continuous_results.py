"""Export near-reference hits and recorded optimum times from completed results."""
import argparse
import math
from numbers import Real
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from qubo_benchmark.runtime.common import read, table
from qubo_benchmark.runtime.compact_metrics import CHECKPOINTS
from qubo_benchmark.runtime.compact_storage import _identifier, _read_checked, SUCCESS_STATUSES
from tools.validate_continuous_results import validate

HIT_COLUMNS = ('solver', 'problem', 'checkpoint_seconds', 'hits', 'planned_seeds', 'hit_probability')
TIME_COLUMNS = ('solver', 'problem', 'seed', 'time_to_optimum_s', 'status', 'reference_type')


def _inside(root, path):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Result path escapes its directory: {path}')
    return path


def export_results(root, output, hit_percent=1.):
    """Validate inputs without modifying them, then atomically write two CSVs."""
    if (not isinstance(hit_percent, Real) or isinstance(hit_percent, bool)
            or not math.isfinite(hit_percent) or hit_percent < 0):
        raise ValueError('Hit percent must be finite and nonnegative')
    root = Path(root).resolve()
    manifest = read(_inside(root, root / 'manifest.json'))
    if manifest.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported compact result schema')
    seeds = manifest['seeds']
    if (not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Invalid planned seed schedule')
    if tuple(manifest['checkpoints_seconds']) != CHECKPOINTS:
        raise ValueError('Expected the nine compact checkpoints')
    solvers = [_identifier(solver) for solver in manifest['solvers']]
    problems = manifest['problems']
    if not solvers or not problems or len({_identifier(p['id']) for p in problems}) != len(problems):
        raise ValueError('Invalid solver/problem plan')
    for name in ('state.json', 'completion.json'):
        _inside(root, root / name)
    for problem in problems:
        if problem['reference_type'] not in ('OPTIMUM', 'BKS') or not math.isfinite(problem['reference']):
            raise ValueError('Invalid problem reference')
        for solver in solvers:
            key = problem['id']
            for folder, suffix in (('progress', '.json'), ('raw', '.npy'), ('aggregated', '.csv')):
                _inside(root, root / folder / solver / (key + suffix))
            if not (root / 'raw' / solver / (key + '.npy')).is_file():
                raise ValueError('Retained raw energies are required; aggregate curves cannot recover near-reference hits')
            group = _read_checked(root / 'progress' / solver / (key + '.json'))
            checksums = group.get('row_sha256')
            if not isinstance(checksums, list) or len(checksums) != len(seeds):
                raise ValueError('Seed checksum count differs from the planned schedule')
    validate(root)
    state = _read_checked(root / 'state.json')
    completion = _read_checked(root / 'completion.json')
    if any(record.get('schema_version') != manifest['schema_version'] for record in (state, completion)):
        raise ValueError('Completion/state schema differs from the manifest')
    planned = len(solvers) * len(problems) * len(seeds)
    if state.get('planned') != planned or state.get('terminal') != planned:
        raise ValueError('Global seed counts differ from completed progress')
    hits, timings, failures = [], [], 0
    for solver in solvers:
        for problem in problems:
            key, reference = problem['id'], problem['reference']
            group = _read_checked(root / 'progress' / solver / (key + '.json'))
            matrix = np.load(root / 'raw' / solver / (key + '.npy'), allow_pickle=False)
            recorded = group.get('time_to_optimum_s', [None] * len(seeds) if manifest['schema_version'] == 1 else None)
            if not isinstance(recorded, list) or len(recorded) != len(seeds):
                raise ValueError('Malformed optimum timing progress')
            tolerance = 0. if problem.get('integer_objective', True) else float(manifest.get('tolerance', 1e-9))
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError('Invalid objective tolerance')
            for index, (seed, timing, status) in enumerate(zip(seeds, recorded, group['status'])):
                failures += status not in SUCCESS_STATUSES
                if status not in SUCCESS_STATUSES and not np.isnan(matrix[index]).all():
                    raise ValueError('Failed seed has available energies')
                if timing is not None:
                    if (not isinstance(timing, Real) or isinstance(timing, bool) or not math.isfinite(timing)
                            or not 0 <= timing <= 20 or status not in SUCCESS_STATUSES
                            or problem['reference_type'] != 'OPTIMUM'):
                        raise ValueError('Invalid recorded optimum time')
                    exact = np.isfinite(matrix[index]) & (np.abs(matrix[index] - reference) <= tolerance)
                    checkpoints = np.asarray(CHECKPOINTS)
                    if not exact.any() or np.any(exact & (checkpoints < timing)) or not exact[checkpoints > timing].all():
                        raise ValueError('Optimum time differs from checkpoint history')
                timings.append(dict(zip(TIME_COLUMNS, (solver, key, seed, timing, status, problem['reference_type']))))
            threshold = reference + hit_percent / 100 * abs(reference)
            if not math.isfinite(threshold):
                raise ValueError('Near-reference threshold is not finite')
            counts = (np.isfinite(matrix) & (matrix <= threshold)).sum(axis=0)
            for seconds, count in zip(CHECKPOINTS, counts):
                hits.append(dict(zip(HIT_COLUMNS, (solver, key, seconds, int(count), len(seeds), int(count) / len(seeds)))))
    if state.get('failures') != failures:
        raise ValueError('Global failure count differs from seed progress')
    output = Path(output).resolve() if output is not None else root / 'postprocessed'
    paths = (output / 'hits_within_1pct.csv', output / 'time_to_optimum.csv')
    table(paths[0], hits, HIT_COLUMNS)
    table(paths[1], timings, TIME_COLUMNS)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resultdir', type=Path)
    parser.add_argument('--hit-percent', '--hit-percent1', type=float, default=1., dest='hit_percent')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        paths = export_results(args.resultdir, args.output, args.hit_percent)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()

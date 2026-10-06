"""Small retained-energy fixtures for post-run reference-hit exports."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qubo_benchmark.runtime.common import atomic, digest, read
from qubo_benchmark.runtime.compact_storage import CompactStore


def helper():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'postprocess_continuous_results.py'
    spec = importlib.util.spec_from_file_location('continuous_postprocess', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root, keep=True):
    manifest = dict(solvers={'example': {}}, problems=[
        dict(id='negative', n=2, density='dense', reference=-100., reference_type='BKS'),
        dict(id='zero', n=2, density='dense', reference=0., reference_type='BKS'),
        dict(id='optimum', n=2, density='dense', reference=-10., reference_type='OPTIMUM')])
    store = CompactStore(root, manifest, seeds=(4, 7), keep_raw_results=keep)
    store.commit_seed('example', 'negative', 4, [-99.] * 9)
    store.commit_seed('example', 'negative', 7, [-98., -98.] + [-101.] * 7)
    store.commit_seed('example', 'zero', 4, [.001, .001] + [0.] * 7)
    store.commit_seed('example', 'zero', 7, [np.nan] * 9, status='failed')
    store.commit_seed('example', 'optimum', 4, [np.nan, -9.] + [-10.] * 7,
                      time_to_optimum_s=.15000000012345)
    store.commit_seed('example', 'optimum', 7, [-9.] * 9)
    assert store.finalize()
    return store


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def rewrite_checked(path, change):
    payload = read(path)['payload']
    change(payload)
    atomic(path, dict(sha256=digest(payload), payload=payload))


def test_exports_one_percent_hits_and_exact_recorded_times_without_input_mutation(tmp_path):
    root = tmp_path / 'results'
    fixture(root)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in root.rglob('*') if path.is_file()}
    hit_path, time_path = helper().export_results(root, None)
    assert hit_path.parent == time_path.parent == root / 'postprocessed'
    hits, times = rows(hit_path), rows(time_path)
    assert len(hits) == 27 and len(times) == 6
    assert tuple(hits[0]) == helper().HIT_COLUMNS
    assert tuple(times[0]) == helper().TIME_COLUMNS
    negative = [row for row in hits if row['problem'] == 'negative']
    assert [int(row['hits']) for row in negative] == [1, 1] + [2] * 7
    assert [float(row['hit_probability']) for row in negative] == [.5, .5] + [1.] * 7
    zero = [row for row in hits if row['problem'] == 'zero']
    assert [int(row['hits']) for row in zero] == [0, 0] + [1] * 7
    assert [float(row['hit_probability']) for row in zero] == [0., 0.] + [.5] * 7
    assert all(row['planned_seeds'] == '2' for row in hits)
    exact = next(row for row in times if row['problem'] == 'optimum' and row['seed'] == '4')
    assert exact['time_to_optimum_s'] == '0.15000000012345'
    assert all(row['time_to_optimum_s'] == '' for row in times if row is not exact)
    assert next(row for row in times if row['problem'] == 'zero' and row['seed'] == '7')['status'] == 'failed'
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize('percent, expected', [(0., [0, 0] + [1] * 7), (2., [2] * 9)])
def test_custom_threshold_uses_absolute_negative_reference(tmp_path, percent, expected):
    fixture(tmp_path / 'results')
    hit_path, _ = helper().export_results(tmp_path / 'results', tmp_path / 'exports', percent)
    negative = [row for row in rows(hit_path) if row['problem'] == 'negative']
    assert [int(row['hits']) for row in negative] == expected


@pytest.mark.parametrize('percent', [-1., float('nan'), float('inf'), True])
def test_bad_percent_does_not_create_exports(tmp_path, percent):
    fixture(tmp_path / 'results')
    with pytest.raises(ValueError, match='finite and nonnegative'):
        helper().export_results(tmp_path / 'results', tmp_path / 'exports', percent)
    assert not (tmp_path / 'exports').exists()


def test_raw_discard_cannot_fall_back_to_aggregate_curves(tmp_path):
    fixture(tmp_path / 'results', keep=False)
    assert not list((tmp_path / 'results').rglob('*.npy'))
    with pytest.raises(ValueError, match='Retained raw energies are required'):
        helper().export_results(tmp_path / 'results', tmp_path / 'exports')
    assert not (tmp_path / 'exports').exists()


@pytest.mark.parametrize('corruption', ['raw', 'progress', 'completion', 'state_count', 'progress_count',
                                       'reference', 'timing', 'missing_timing', 'checksum_count',
                                       'failure_count', 'completion_schema'])
def test_corrupt_or_incomplete_inputs_are_rejected_before_exports(tmp_path, corruption):
    root = tmp_path / 'results'
    store = fixture(root)
    progress = root / 'progress' / 'example' / 'optimum.json'
    if corruption == 'raw':
        raw = store.raw_path('example', 'negative')
        matrix = np.load(raw, allow_pickle=False)
        matrix[0, 0] = -98.
        np.save(raw, matrix, allow_pickle=False)
    elif corruption == 'progress':
        wrapper = read(progress)
        wrapper['payload']['time_to_optimum_s'][0] = .16
        atomic(progress, wrapper)
    elif corruption == 'completion':
        rewrite_checked(root / 'completion.json', lambda payload: payload.update(manifest_sha256='other root'))
    elif corruption == 'state_count':
        rewrite_checked(root / 'state.json', lambda payload: payload.update(terminal=5))
    elif corruption == 'progress_count':
        rewrite_checked(progress, lambda payload: payload['status'].pop())
    elif corruption == 'checksum_count':
        rewrite_checked(progress, lambda payload: payload['row_sha256'].pop())
    elif corruption == 'failure_count':
        rewrite_checked(root / 'state.json', lambda payload: payload.update(failures=0))
    elif corruption == 'completion_schema':
        rewrite_checked(root / 'completion.json', lambda payload: payload.update(schema_version=1))
    elif corruption == 'reference':
        manifest = read(root / 'manifest.json')
        manifest['problems'][0]['reference'] = -101.
        atomic(root / 'manifest.json', manifest)
    elif corruption == 'timing':
        rewrite_checked(progress, lambda payload: payload['time_to_optimum_s'].__setitem__(0, .8))
    else:
        rewrite_checked(progress, lambda payload: payload.pop('time_to_optimum_s'))
    with pytest.raises(ValueError):
        helper().export_results(root, tmp_path / 'exports')
    assert not (tmp_path / 'exports').exists()


def test_legacy_schema_one_is_read_only_and_has_unavailable_optimum_times(tmp_path):
    root = tmp_path / 'results'
    fixture(root)
    manifest = read(root / 'manifest.json')
    manifest['schema_version'] = 1
    atomic(root / 'manifest.json', manifest)
    identity = digest(manifest)
    for path in (root / 'state.json', root / 'completion.json'):
        rewrite_checked(path, lambda payload: payload.update(schema_version=1, manifest_sha256=identity))
    for path in (root / 'progress').rglob('*.json'):
        def legacy(payload):
            payload['manifest_sha256'] = identity
            payload.pop('time_to_optimum_s')
        rewrite_checked(path, legacy)
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    _, time_path = helper().export_results(root, tmp_path / 'exports')
    assert all(row['time_to_optimum_s'] == '' for row in rows(time_path))
    assert before == {path: path.read_bytes() for path in before}


def test_unsafe_identifier_is_rejected_without_following_its_path(tmp_path):
    root = tmp_path / 'results'
    fixture(root)
    manifest = read(root / 'manifest.json')
    manifest['problems'][0]['id'] = '../outside'
    atomic(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Unsafe solver/problem identifier'):
        helper().export_results(root, tmp_path / 'exports')


def test_cli_exports_to_explicit_output_and_accepts_threshold_option(tmp_path, capsys):
    root, output = tmp_path / 'results', tmp_path / 'exports'
    fixture(root)
    helper().main([str(root), '--hit-percent', '2', '--output', str(output)])
    assert (output / 'hits_within_1pct.csv').exists() and (output / 'time_to_optimum.csv').exists()
    assert str(output / 'time_to_optimum.csv') in capsys.readouterr().out

"""Bounded tests for checkpoint statistics and crash-safe compact persistence."""
import csv
import json
import math

import numpy as np
import pytest

from qubo_benchmark.runtime.common import atomic, digest, read
from qubo_benchmark.runtime.compact_metrics import AGGREGATE_COLUMNS, CHECKPOINTS, aggregate_energies, tts99
from qubo_benchmark.runtime.compact_plots import plot_results
from qubo_benchmark.runtime.compact_storage import CompactStore


def manifest(n=200, reference_type='BKS', integer_objective=True):
    return dict(solvers={'lib_example': {'dtype': 'float32'}},
                problems=[dict(id='example', n=n, density='sparse', reference=-10,
                               reference_type=reference_type, source_url='https://example.org/input',
                               integer_objective=integer_objective)],
                source='source-content-hash', configuration='configuration-hash',
                inputs={'example': 'input-hash'}, environment='environment-hash')


def store(root, **kwargs):
    return CompactStore(root, manifest(), seeds=(0, 1), **kwargs)


def row(value):
    return np.full(len(CHECKPOINTS), value, dtype=np.float64)


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == AGGREGATE_COLUMNS
        return list(reader)


def test_metrics_use_signed_raw_gap_reference_hits_and_literal_tts_limits():
    matrix = np.array([[-12., -10., -8.], [-8., -10., -8.]])
    result = aggregate_energies(matrix, (.05, .1, .2), -10, 'BKS')
    assert [item['gap_to_solution'] for item in result] == [0., 0., 2.]
    assert [item['hit_probability'] for item in result] == [.5, 1., 0.]
    assert result[0]['TTS99'] == pytest.approx(.05 * math.log(.01) / math.log(.5))
    assert result[1]['TTS99'] == 0.
    assert math.isinf(result[2]['TTS99'])
    assert tts99(20, 1) == 0 and tts99(20, 0) == math.inf
    with pytest.raises(ValueError, match='Incomplete'):
        aggregate_energies(matrix, (.05, .1, .2), -10, complete=False)


def test_unavailable_energy_is_miss_and_mean_gap_is_nan():
    matrix = np.array([[-10., -9.], [np.nan, -10.], [np.nan, -12.]])
    result = aggregate_energies(matrix, (.05, .1), -10)
    assert math.isnan(result[0]['gap_to_solution'])
    assert result[0]['hit_probability'] == pytest.approx(1 / 3)
    assert result[1]['gap_to_solution'] == pytest.approx(-1 / 3)
    assert result[1]['hit_probability'] == pytest.approx(2 / 3)


def test_bks_improvements_keep_negative_gap_in_raw_objective_units():
    result = aggregate_energies(np.array([[-12., -13.], [-14., -15.]]), (.05, .1), -10, 'BKS')
    assert [item['gap_to_solution'] for item in result] == [-3., -4.]
    assert [item['hit_probability'] for item in result] == [1., 1.]


def test_proven_optimum_and_integer_tolerance_are_enforced(tmp_path):
    instance = CompactStore(tmp_path / 'integer', manifest(reference_type='OPTIMUM'), seeds=(0,))
    with pytest.raises(ValueError, match='proven optimum'):
        instance.commit_seed('lib_example', 'example', 0, row(-10.0000000001))
    assert instance.pending() == [('lib_example', 'example', 0)]
    floating = CompactStore(tmp_path / 'float', manifest(reference_type='OPTIMUM', integer_objective=False), seeds=(0,))
    floating.commit_seed('lib_example', 'example', 0, row(-10.0000000001))
    assert floating.aggregate('lib_example', 'example')[0]['hit_probability'] == 1
    with pytest.raises(ValueError, match='proven optimum'):
        aggregate_energies(np.array([[-11.]]), (1.,), -10, 'OPTIMUM', tolerance=1e-9)


def test_defaults_store_one_small_matrix_and_seed_time_arrays_once(tmp_path):
    first = CompactStore(tmp_path / 'n200', manifest(200))
    second = CompactStore(tmp_path / 'n1000', manifest(1000))
    saved = read(first.root / 'manifest.json')
    assert saved['seeds'] == list(range(100))
    assert saved['checkpoints_seconds'] == list(CHECKPOINTS)
    assert saved['keep_raw_results'] is True
    assert saved['tolerance'] == 1e-9
    arrays = list(first.root.rglob('*.npy'))
    assert len(arrays) == 1
    matrix = np.load(arrays[0], allow_pickle=False)
    assert matrix.shape == (100, 9) and matrix.dtype == np.float64
    assert matrix.nbytes == 7200 and np.isnan(matrix).all()
    assert arrays[0].stat().st_size == second.raw_path('lib_example', 'example').stat().st_size
    assert first.planned_count == 100 and first.completed_count == 0
    assert 'seeds' not in repr(first.state) and 'checkpoint_seconds' not in repr(first.state)
    assert len(list(first.root.rglob('*.*'))) == 4
    assert (first.root / 'state.json').stat().st_size < 600
    assert 'groups' not in read(first.root / 'state.json')['payload']


def test_partial_groups_do_not_publish_final_rates_and_terminal_failures_count(tmp_path):
    instance = store(tmp_path)
    instance.commit_seed('lib_example', 'example', 0, row(-10))
    assert instance.aggregate('lib_example', 'example') is None
    assert instance.finalize() is False
    assert not instance.aggregate_path('lib_example', 'example').exists()
    instance.commit_seed('lib_example', 'example', 1, row(-10), status='failed')
    assert np.isnan(np.load(instance.raw_path('lib_example', 'example'))[1]).all()
    assert instance.has_failures and instance.completed_count == 2
    assert instance.finalize() is True
    result = csv_rows(instance.aggregate_path('lib_example', 'example'))
    assert len(result) == 9
    assert all(item['gap_to_solution'] == 'NaN' for item in result)
    assert all(float(item['hit_probability']) == .5 for item in result)
    assert instance.raw_path('lib_example', 'example').exists()
    reopened = store(tmp_path)
    assert reopened.pending() == [] and reopened.state['complete']
    with pytest.raises(ValueError, match='already committed'):
        reopened.commit_seed('lib_example', 'example', 0, row(-8))


def test_interrupted_seed_reruns_in_full_without_saving_partial_energies(tmp_path):
    instance = store(tmp_path)
    assert instance.commit_seed('lib_example', 'example', 0, row(-10), status='interrupted') is False
    assert np.isnan(np.load(instance.raw_path('lib_example', 'example'))).all()
    assert store(tmp_path).pending() == [('lib_example', 'example', 0), ('lib_example', 'example', 1)]


def test_crash_after_matrix_before_progress_discards_orphan_and_reruns(tmp_path, monkeypatch):
    instance = store(tmp_path)
    def crash(*args):
        raise OSError('simulated progress-write crash')
    monkeypatch.setattr(instance, '_write_group', crash)
    with pytest.raises(OSError, match='simulated'):
        instance.commit_seed('lib_example', 'example', 0, row(-10))
    assert np.load(instance.raw_path('lib_example', 'example'))[0, 0] == -10
    reopened = store(tmp_path)
    assert np.isnan(np.load(reopened.raw_path('lib_example', 'example'))).all()
    assert len(reopened.pending()) == 2
    reopened.commit_seed('lib_example', 'example', 0, row(-9))
    assert store(tmp_path).pending() == [('lib_example', 'example', 1)]


def test_optimum_times_share_atomic_progress_and_survive_resume(tmp_path):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0, 1))
    instance.commit_seed('lib_example', 'example', 0, [-9.] + [-10.] * 8,
                         time_to_optimum_s=.08)
    instance.commit_seed('lib_example', 'example', 1, row(-9.))
    progress_path = tmp_path / 'progress' / 'lib_example' / 'example.json'
    progress = read(progress_path)
    assert progress['sha256'] == digest(progress['payload'])
    assert progress['payload']['time_to_optimum_s'] == [.08, None]
    assert progress['payload']['status'] == ['complete', 'complete']
    assert len(list(tmp_path.rglob('*.npy'))) == 1
    reopened = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0, 1))
    assert reopened.pending() == []
    assert reopened.state['groups']['lib_example']['example']['time_to_optimum_s'] == [.08, None]
    assert reopened.aggregate('lib_example', 'example')[0]['hit_probability'] == 0
    assert reopened.aggregate('lib_example', 'example')[1]['hit_probability'] == .5


@pytest.mark.parametrize('timing', [float('nan'), float('inf'), -1., 20.1, '.01', True])
def test_commit_rejects_malformed_optimum_time_without_writing(tmp_path, timing):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))
    with pytest.raises(ValueError, match='finite number'):
        instance.commit_seed('lib_example', 'example', 0, row(-10.), time_to_optimum_s=timing)
    assert instance.completed_count == 0
    assert np.isnan(np.load(instance.raw_path('lib_example', 'example'))).all()


@pytest.mark.parametrize('energies,timing', [
    (row(-10.), .08),
    ([-9.] + [-10.] * 7 + [-9.], .08),
    ([-9.] + [-10.] * 7 + [np.nan], .08),
    (row(-9.), .01),
])
def test_commit_rejects_optimum_time_inconsistent_with_checkpoints(tmp_path, energies, timing):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))
    with pytest.raises(ValueError, match='inconsistent with seed checkpoints'):
        instance.commit_seed('lib_example', 'example', 0, energies, time_to_optimum_s=timing)
    assert instance.completed_count == 0


def test_boundary_optimum_time_keeps_previously_recorded_checkpoint(tmp_path):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0, 1))
    instance.commit_seed('lib_example', 'example', 0, [-9., -9.] + [-10.] * 7,
                         time_to_optimum_s=.1)
    instance.commit_seed('lib_example', 'example', 1, [-9.] * 8 + [-10.],
                         time_to_optimum_s=20.)
    reopened = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0, 1))
    assert reopened.state['groups']['lib_example']['example']['time_to_optimum_s'] == [.1, 20.]


@pytest.mark.parametrize('failed_status', ['failed', 'error', 'validation_error', 'oom'])
def test_bks_never_accepts_optimum_time_and_terminal_failure_clears_it(tmp_path, failed_status):
    bks = store(tmp_path / 'bks')
    with pytest.raises(ValueError, match='successful OPTIMUM'):
        bks.commit_seed('lib_example', 'example', 0, row(-10.), time_to_optimum_s=.01)
    instance = CompactStore(tmp_path / 'optimum', manifest(reference_type='OPTIMUM'), seeds=(0, 1))
    instance.commit_seed('lib_example', 'example', 0, row(-10.), status=failed_status, time_to_optimum_s=.01)
    assert instance.commit_seed('lib_example', 'example', 1, row(-10.), status='interrupted',
                                time_to_optimum_s=.01) is False
    progress = read(instance.root / 'progress' / 'lib_example' / 'example.json')['payload']
    assert progress['time_to_optimum_s'] == [None, None]
    assert progress['status'] == [failed_status, None]
    assert np.isnan(np.load(instance.raw_path('lib_example', 'example'))).all()


def test_optimum_time_is_pending_when_progress_commit_crashes(tmp_path, monkeypatch):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))
    def crash(*args):
        raise OSError('simulated timing progress crash')
    monkeypatch.setattr(instance, '_write_group', crash)
    with pytest.raises(OSError, match='timing progress crash'):
        instance.commit_seed('lib_example', 'example', 0, row(-10.), time_to_optimum_s=.01)
    assert instance.state['groups']['lib_example']['example']['time_to_optimum_s'] == [None]
    assert np.load(instance.raw_path('lib_example', 'example'))[0, 0] == -10.
    reopened = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))
    assert reopened.pending() == [('lib_example', 'example', 0)]
    assert reopened.state['groups']['lib_example']['example']['time_to_optimum_s'] == [None]
    assert np.isnan(np.load(reopened.raw_path('lib_example', 'example'))).all()


def test_optimum_time_commits_before_global_counter_crash(tmp_path, monkeypatch):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))
    def crash():
        raise OSError('simulated timing counter crash')
    monkeypatch.setattr(instance, '_write_state', crash)
    with pytest.raises(OSError, match='timing counter crash'):
        instance.commit_seed('lib_example', 'example', 0, row(-10.), time_to_optimum_s=.01)
    reopened = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))
    assert reopened.pending() == []
    assert reopened.state['groups']['lib_example']['example']['time_to_optimum_s'] == [.01]


@pytest.mark.parametrize('mutation', ['missing', 'short', 'nan', 'infinite', 'negative', 'late',
                                      'string', 'boolean', 'pending', 'failed', 'back_credit'])
def test_resume_rejects_malformed_or_inconsistent_optimum_times(tmp_path, mutation):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))
    instance.commit_seed('lib_example', 'example', 0, row(-10.), time_to_optimum_s=.01)
    path = tmp_path / 'progress' / 'lib_example' / 'example.json'
    payload = read(path)['payload']
    if mutation == 'missing':
        del payload['time_to_optimum_s']
    elif mutation == 'short':
        payload['time_to_optimum_s'] = []
    elif mutation in ('pending', 'failed'):
        payload['status'][0] = None if mutation == 'pending' else 'failed'
    else:
        payload['time_to_optimum_s'][0] = dict(nan=float('nan'), infinite=float('inf'), negative=-.01,
            late=20.1, string='.01', boolean=True, back_credit=.08)[mutation]
    if mutation in ('nan', 'infinite'):
        # Production canonical JSON cannot serialize nonfinite values. Exercise
        # recovery from malformed external JSON rather than its writer guard.
        path.write_text(json.dumps(dict(sha256='invalid', payload=payload)), encoding='utf-8')
    else:
        atomic(path, dict(sha256=digest(payload), payload=payload))
    with pytest.raises(ValueError, match='[Oo]ptimum timing|optimum timing state|Corrupt compact state'):
        CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0,))


def test_optimum_times_survive_raw_discard_and_validate_before_recovery(tmp_path):
    instance = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0, 1), keep_raw_results=False)
    instance.commit_seed('lib_example', 'example', 0, row(-10.), time_to_optimum_s=.01)
    instance.commit_seed('lib_example', 'example', 1, row(-10.), status='failed', time_to_optimum_s=.01)
    raw_path = instance.raw_path('lib_example', 'example')
    raw_bytes = raw_path.read_bytes()
    instance.finalize()
    reopened = CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0, 1), keep_raw_results=False)
    progress = read(tmp_path / 'progress' / 'lib_example' / 'example.json')['payload']
    assert progress['row_sha256'] is None and progress['time_to_optimum_s'] == [.01, None]
    assert reopened.pending() == [] and not raw_path.exists()
    raw_path.write_bytes(raw_bytes)
    progress['time_to_optimum_s'][1] = .01
    atomic(tmp_path / 'progress' / 'lib_example' / 'example.json',
           dict(sha256=digest(progress), payload=progress))
    with pytest.raises(ValueError, match='successful OPTIMUM'):
        CompactStore(tmp_path, manifest(reference_type='OPTIMUM'), seeds=(0, 1), keep_raw_results=False)
    assert raw_path.exists()


def test_schema_one_results_are_preserved_but_cannot_resume_with_timing_protocol(tmp_path):
    instance = store(tmp_path)
    old_manifest = read(tmp_path / 'manifest.json')
    old_manifest['schema_version'] = 1
    atomic(tmp_path / 'manifest.json', old_manifest)
    original_raw = instance.raw_path('lib_example', 'example').read_bytes()
    with pytest.raises(ValueError, match='resume identity mismatch'):
        store(tmp_path)
    assert instance.raw_path('lib_example', 'example').read_bytes() == original_raw


@pytest.mark.parametrize('component', ['source', 'configuration', 'inputs', 'environment', 'reference'])
def test_resume_identity_includes_all_required_components(tmp_path, component):
    store(tmp_path)
    changed = manifest()
    if component == 'reference':
        changed['problems'][0]['reference'] = -11
    else:
        changed[component] = 'different identity'
    with pytest.raises(ValueError, match='identity mismatch'):
        CompactStore(tmp_path, changed, seeds=(0, 1))


def test_resume_rejects_protocol_and_raw_retention_changes(tmp_path):
    store(tmp_path)
    with pytest.raises(ValueError, match='identity mismatch'):
        CompactStore(tmp_path, manifest(), seeds=(0, 2))
    with pytest.raises(ValueError, match='identity mismatch'):
        store(tmp_path, keep_raw_results=False)
    with pytest.raises(ValueError, match='exactly'):
        CompactStore(tmp_path, manifest(), seeds=(0, 1), checkpoints=(.05, .1))


def test_resume_detects_stale_or_partial_matrix_and_corrupt_state(tmp_path):
    instance = store(tmp_path / 'stale')
    instance.commit_seed('lib_example', 'example', 0, row(-10))
    matrix = np.load(instance.raw_path('lib_example', 'example'))
    matrix[0, 0] = -9
    with instance.raw_path('lib_example', 'example').open('wb') as handle:
        np.save(handle, matrix, allow_pickle=False)
    with pytest.raises(ValueError, match='Stale/corrupt committed'):
        store(instance.root)
    partial = store(tmp_path / 'partial')
    with partial.raw_path('lib_example', 'example').open('wb') as handle:
        handle.write(b'partial')
    with pytest.raises(ValueError, match='Partial/corrupt'):
        store(partial.root)
    corrupt = store(tmp_path / 'state')
    wrapper = read(corrupt.root / 'state.json')
    wrapper['payload']['complete'] = True
    atomic(corrupt.root / 'state.json', wrapper)
    with pytest.raises(ValueError, match='Corrupt compact state'):
        store(corrupt.root)


def test_crash_after_durable_group_before_global_counters_keeps_completed_seed(tmp_path, monkeypatch):
    instance = store(tmp_path)
    def crash():
        raise OSError('simulated counter-write crash')
    monkeypatch.setattr(instance, '_write_state', crash)
    with pytest.raises(OSError, match='simulated'):
        instance.commit_seed('lib_example', 'example', 0, row(-10))
    reopened = store(tmp_path)
    assert reopened.pending() == [('lib_example', 'example', 1)]
    assert read(tmp_path / 'state.json')['payload']['terminal'] == 1


def test_raw_discard_occurs_only_after_durable_aggregate_and_completion(tmp_path, monkeypatch):
    instance = store(tmp_path, keep_raw_results=False)
    for seed in (0, 1):
        instance.commit_seed('lib_example', 'example', seed, row(-10))
    original = instance._write_state
    def crash_after_completion():
        if (instance.root / 'completion.json').exists():
            raise OSError('simulated completion-state crash')
        original()
    monkeypatch.setattr(instance, '_write_state', crash_after_completion)
    with pytest.raises(OSError, match='simulated'):
        instance.finalize()
    assert instance.raw_path('lib_example', 'example').exists()
    assert instance.aggregate_path('lib_example', 'example').exists()
    reopened = store(tmp_path, keep_raw_results=False)
    assert reopened.pending() == [] and reopened.state['complete']
    assert not reopened.raw_path('lib_example', 'example').exists()
    assert read(tmp_path / 'progress' / 'lib_example' / 'example.json')['payload']['row_sha256'] is None
    assert reopened.finalize() is True
    assert reopened.aggregate('lib_example', 'example')[0]['TTS99'] == 0
    assert store(tmp_path, keep_raw_results=False).pending() == []


def test_incomplete_no_raw_retention_still_retains_resume_matrix(tmp_path):
    instance = store(tmp_path, keep_raw_results=False)
    instance.commit_seed('lib_example', 'example', 0, row(-10))
    assert instance.finalize() is False
    assert instance.raw_path('lib_example', 'example').exists()
    reopened = store(tmp_path, keep_raw_results=False)
    assert reopened.pending() == [('lib_example', 'example', 1)]


def test_durable_completion_can_repeat_raw_removal_after_crash(tmp_path):
    instance = store(tmp_path, keep_raw_results=False)
    for seed in (0, 1):
        instance.commit_seed('lib_example', 'example', seed, row(-10))
    raw_path = instance.raw_path('lib_example', 'example')
    saved_matrix = raw_path.read_bytes()
    instance.finalize()
    # Simulate a directory unlink that did not survive a sudden host failure.
    raw_path.write_bytes(saved_matrix)
    reopened = store(tmp_path, keep_raw_results=False)
    assert reopened.pending() == [] and reopened.state['complete']
    assert not raw_path.exists()


def test_raw_discard_compacts_checksums_and_preserves_failure_statuses(tmp_path):
    instance = store(tmp_path, keep_raw_results=False)
    instance.commit_seed('lib_example', 'example', 0, row(-10))
    instance.commit_seed('lib_example', 'example', 1, row(np.nan), status='failed')
    instance.finalize()
    progress = read(tmp_path / 'progress' / 'lib_example' / 'example.json')['payload']
    assert progress['row_sha256'] is None
    assert progress['status'] == ['complete', 'failed']
    assert progress['raw_discarded'] is True
    reopened = store(tmp_path, keep_raw_results=False)
    assert reopened.completed_count == 2 and reopened.has_failures
    assert reopened.pending() == []
    assert read(tmp_path / 'state.json')['payload']['failures'] == 1
    assert reopened.aggregate('lib_example', 'example')[0]['hit_probability'] == .5


def test_compacted_progress_requires_valid_completion_before_raw_recovery(tmp_path):
    instance = store(tmp_path, keep_raw_results=False)
    for seed in (0, 1):
        instance.commit_seed('lib_example', 'example', seed, row(-10))
    saved_raw = instance.raw_path('lib_example', 'example').read_bytes()
    instance.finalize()
    raw_path = instance.raw_path('lib_example', 'example')
    raw_path.write_bytes(saved_raw)
    completion = read(tmp_path / 'completion.json')
    completion['payload']['manifest_sha256'] = 'changed'
    atomic(tmp_path / 'completion.json', completion)
    with pytest.raises(ValueError, match='Corrupt compact state'):
        store(tmp_path, keep_raw_results=False)
    assert raw_path.exists()


def test_discarded_raw_does_not_hide_corrupt_aggregate(tmp_path):
    instance = store(tmp_path, keep_raw_results=False)
    for seed in (0, 1):
        instance.commit_seed('lib_example', 'example', seed, row(-10))
    instance.finalize()
    with instance.aggregate_path('lib_example', 'example').open('a', encoding='utf-8') as handle:
        handle.write('partial')
    with pytest.raises(ValueError, match='Missing/corrupt completed'):
        store(tmp_path, keep_raw_results=False)


def test_three_png_plot_types_are_generated_without_raw_arrays(tmp_path):
    pytest.importorskip('matplotlib')
    instance = store(tmp_path, keep_raw_results=False)
    for seed in (0, 1):
        instance.commit_seed('lib_example', 'example', seed, row(-10 + seed))
    instance.finalize()
    assert not list(tmp_path.rglob('*.npy'))
    paths = plot_results(tmp_path)
    assert {path.name for path in paths} == {'gap_to_solution.png', 'hit_probability.png', 'TTS99.png'}
    assert len(paths) == len(list(tmp_path.rglob('*.png'))) == 3
    for path in paths:
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert path.stat().st_size < 500_000

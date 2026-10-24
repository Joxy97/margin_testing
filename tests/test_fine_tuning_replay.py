"""CPU-only replay validation and deadline tests; no GPU campaign is launched."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qubo_benchmark.model import Problem
from qubo_benchmark.runtime.continuous_worker import ScalarObserver
from qubo_solvers.observation import SolveInterrupted

spec = importlib.util.spec_from_file_location('fine_tuning_replay', Path(__file__).parents[1]/'fine_tuning/run.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path/'repo'; package = root/'fine_tuning'; package.mkdir(parents=True)
    source = root/'src/fixture.py'; source.parent.mkdir(); source.write_text('# frozen source\n')
    files = {'src/fixture.py': replay.sha(source)}
    parameters = {'runs': 16, 'run_batch_size': 16, 'max_steps': 128, 'dtype': 'float32',
                  'best_only': True, 'memory_limit_bytes': None, 'nested': {'value': [1, 2]}}
    eligible = ['lib_fixture']+[f'lib_fixture_{i}' for i in range(129)]
    excluded = [f'lib_excluded_{i}' for i in range(14)]
    input_path = tmp_path/'gka5e.npz'
    Problem(200, np.array([0]), np.array([0]), np.array([-35154], dtype=np.int64)).save(input_path)
    # Resolve the real recovery function before ROOT is replaced by the fixture.
    fingerprint = replay.canonical_input(input_path)
    original_canonical = replay.canonical_input
    actual_root = replay.ROOT
    def canonical(path):
        current = replay.ROOT
        try:
            replay.ROOT = actual_root
            return original_canonical(path)
        finally:
            replay.ROOT = current
    settings = dict(status='validated_complete', pending_eligible_solvers=[],
        solvers={sid: copy.deepcopy(parameters) for sid in eligible+excluded}, excluded_unchanged=excluded,
        schedule_by_solver={sid: 'wall' if sid == 'lib_fixture' else 'finite_step' for sid in eligible+excluded},
        wall_schedule_solvers=['lib_fixture'], source_sha256=replay.digest(files),
        problem='gka5e', n_variables=200, budget_seconds=20., checkpoints_seconds=replay.CHECKPOINTS,
        input_sha256=replay.sha(input_path))
    put(package/'selected_settings.json', settings)
    put(package/'source_manifest.json', dict(source_files=files, source_sha256=replay.digest(files),
        selected_settings_sha256=replay.sha(package/'selected_settings.json')))
    protocol = dict(problem=dict(instance_id='gka5e', n_variables=200, reference=-35154., reference_type='OPTIMUM',
        numeric_coo_sha256=fingerprint, npz_sha256=replay.sha(input_path)),
        budget_seconds=20., checkpoints_seconds=replay.CHECKPOINTS)
    put(package/'protocol.json', protocol)
    monkeypatch.setattr(replay, 'ROOT', root)
    monkeypatch.setattr(replay, 'BUNDLE', package)
    monkeypatch.setattr(replay, 'canonical_input', canonical)
    return root, package, input_path, settings


def test_dry_run_validates_on_cpu_without_outputs_or_gpu_execution(bundle, tmp_path, monkeypatch, capsys):
    root, package, input_path, settings = bundle
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(replay, 'execute', lambda *a: pytest.fail('Dry run invoked a GPU solve.'))
    output = tmp_path/'not_created'
    assert replay.main(['--solver', 'lib_fixture', '--seed', '1001', '--input', str(input_path),
                        '--output', str(output), '--dry-run']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'DRY_RUN_VERIFIED' and report['schedule'] == 'wall'
    assert not output.exists()
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['source', 'settings', 'input'])
def test_modified_source_settings_or_input_are_rejected(bundle, change):
    root, package, input_path, _ = bundle
    if change == 'source':
        (root/'src/fixture.py').write_text('# modified\n')
    elif change == 'settings':
        row = replay.read(package/'selected_settings.json'); row['solvers']['lib_fixture']['runs'] = 9
        put(package/'selected_settings.json', row)
    else:
        Problem(200, np.array([0]), np.array([0]), np.array([-35153], dtype=np.int64)).save(input_path)
    with pytest.raises(ValueError, match='differs|differ'):
        replay.validate_bundle(input_path, 'lib_fixture', [1001])


def test_equal_numeric_arrays_with_different_npz_container_are_allowed_and_recorded(bundle, tmp_path):
    _, _, input_path, _ = bundle
    source = Problem.load(input_path)
    repacked = tmp_path/'repacked.npz'
    np.savez(repacked, n=source.n, rows=source.rows, cols=source.cols, values=source.values, offset=source.offset)
    _, _, evidence = replay.validate_bundle(repacked, 'lib_fixture', [1001])
    assert evidence['actual_npz_sha256'] != evidence['original_npz_sha256']
    assert evidence['original_npz_bytes_equal'] is False


def test_exclusions_and_duplicate_or_invalid_seeds_are_rejected(bundle):
    _, _, input_path, _ = bundle
    for solver, seeds in [('lib_excluded_0', [1]), ('missing', [1]), ('lib_fixture', [1, 1]),
                          ('lib_fixture', [-1]), ('lib_fixture', [2**32]), ('lib_fixture', list(range(101)))]:
        with pytest.raises(ValueError):
            replay.validate_bundle(input_path, solver, seeds)


def test_each_seed_gets_independent_parameters_and_exact_schedule(bundle):
    _, _, input_path, settings = bundle
    original = copy.deepcopy(settings)
    first = replay.make_job(settings, 'lib_fixture', 1, input_path, 'cuda:0')
    first['parameters']['nested']['value'].append(99)
    second = replay.make_job(settings, 'lib_fixture', 2, input_path, 'cuda:0')
    finite = replay.make_job(settings, 'lib_fixture_0', 3, input_path, 'cuda:0')
    assert settings == original and second['parameters'] == original['solvers']['lib_fixture']
    assert first['wall_schedule'] and not finite['wall_schedule']
    assert second['budget'] == 20. and second['checkpoints'] == replay.CHECKPOINTS
    assert second['seed'] == 2 and second['stop_on_optimum']


class Clock:
    seconds = 0.
    def __call__(self):
        return int(self.seconds*1e9)


def observer(clock, scorer=None):
    return replay.immediate_observer(ScalarObserver)(scorer or SimpleNamespace(), clock=clock,
        reference=-35154., reference_type='OPTIMUM', tolerance=0., stop_on_optimum=True)


def test_natural_return_forward_fills_without_search_or_sleep():
    clock = Clock(); record = observer(clock)
    clock.seconds = .07; record.capture_objectives(-35000.)
    clock.seconds = .12
    values = record.finish()
    assert np.isnan(values[0])  # The .05 checkpoint predates the first certified candidate.
    assert np.all(values[1:] == -35000.) and clock.seconds == .12


def test_candidate_finishing_after_deadline_cannot_improve_any_checkpoint():
    clock = Clock()
    class DelayedScorer:
        def score(self, samples, raw):
            clock.seconds = 20.001
            return -35154.
    record = observer(clock, DelayedScorer())
    clock.seconds = .01; record.capture_objectives(-35000.)
    clock.seconds = 19.99
    with pytest.raises(SolveInterrupted, match='deadline'):
        record.capture([[0]])
    assert record.certified == -35000. and record.time_to_optimum_s is None
    assert np.all(record.finish() == -35000.)


def test_exact_optimum_before_deadline_stops_and_forward_fills():
    clock = Clock(); record = observer(clock)
    clock.seconds = .01; record.capture_objectives(-35000.)
    clock.seconds = .4
    with pytest.raises(SolveInterrupted, match='optimum'):
        record.capture_objectives(-35154.)
    values = record.finish()
    assert list(values[:3]) == [-35000.]*3 and np.all(values[3:] == -35154.)
    assert record.time_to_optimum_s == .4

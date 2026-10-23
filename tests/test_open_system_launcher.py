"""Exercise the frozen campaign wrapper without dispatching any solver trials."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from qubo_benchmark.runtime import continuous_cli as cli
from qubo_benchmark.runtime.common import atomic, filehash, read
from qubo_benchmark.runtime.compact_metrics import CHECKPOINTS
from qubo_benchmark.runtime.compact_storage import CompactStore


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / 'run_open_system_dense.py'
    spec = importlib.util.spec_from_file_location('open_system_launcher_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_fixture(launcher):
    solvers = list(launcher.OPEN_SYSTEM_SOLVERS)
    return dict(
        solvers=solvers, settings={solver: {} for solver in solvers}, npz='unused.npz',
        manifest=dict(
            solvers={solver: {} for solver in solvers}, inputs={'gka5e': 'input-sha'},
            problems=[dict(id='gka5e', n=200, density='dense', reference=-35154,
                           reference_type='OPTIMUM', integer_objective=True)],
            execution={'wall_schedule_solvers': solvers.copy()}))


@pytest.mark.parametrize('raise_after_commit', [False, True])
def test_fixed_plan_commits_resumes_and_restores_runtime_hooks(
        launcher, tmp_path, monkeypatch, capsys, raise_after_commit):
    evidence = dict(input_files={'input/gka5e.npz': 'input-sha'}, launcher_sha256='launcher-sha')
    monkeypatch.setattr(launcher, 'checked_identity', lambda folder: evidence)
    prepared = prepared_fixture(launcher)
    base_prepare = lambda args: prepared
    monkeypatch.setattr(cli, 'prepare', base_prepare)
    base_job, base_commit = cli._job, CompactStore.commit_seed
    output = tmp_path / 'results'
    args = SimpleNamespace(campaign=tmp_path, output=None, devices='cuda:0,cuda:1',
                           resume=False, dry_run=True)

    def fake_main(argv):
        parsed = cli.parser().parse_args(argv)
        assert (parsed.variable_count, parsed.density, parsed.runs, parsed.seed_start) == (200, 'dense', 100, 0)
        assert parsed.solvers.split(',') == list(launcher.OPEN_SYSTEM_SOLVERS)
        assert len(parsed.solvers.split(',')) == 20
        assert parsed.seed_workers == '1' and parsed.no_autotune and parsed.require_gpu
        assert parsed.stop_on_optimum and parsed.keep_raw_results and not parsed.plots
        assert parsed.dry_run and parsed.json
        assert parsed.devices == 'cuda:0,cuda:1'
        assert Path(parsed.output_root) == output
        frozen = cli.prepare(parsed)
        execution = frozen['manifest']['execution']
        assert execution['wall_schedule_solvers'] == []
        assert execution['open_system_campaign_identity'] == evidence
        assert execution['open_system_campaign_identity'] is not evidence
        solver = frozen['solvers'][0]
        job = cli._job(frozen, parsed, solver)
        assert job['budget'] == 20 and job['checkpoints'] == CHECKPOINTS
        assert CHECKPOINTS == (.05, .1, .2, .5, 1., 2., 5., 10., 20.)
        assert job['wall_schedule'] is False and job['stop_on_optimum'] is True
        assert job['reference_type'] == 'OPTIMUM' and job['reference'] == -35154

        # Synthetic commits validate persistence/progress without calling a solver.
        store = CompactStore(output, frozen['manifest'], seeds=range(100))
        assert store.planned_count == 2000
        assert store.commit_seed(solver, 'gka5e', 0, [-35100.] + [-35154.] * 8,
                                 time_to_optimum_s=.075)
        assert not store.commit_seed(solver, 'gka5e', 1, [None] * 9, status='interrupted')
        assert store.commit_seed(solver, 'gka5e', 2, [0.] * 9, status='error')
        progress = read(output / 'campaign_progress.json')
        assert progress['completed'] == 2 and progress['planned'] == 2000
        assert progress['percent'] == .1 and progress['has_failures']
        assert progress['last_status'] == 'error' and progress['seed'] == 2
        assert progress['eta_s'] >= 0

        resumed = CompactStore(output, frozen['manifest'], seeds=range(100))
        assert resumed.completed_count == 2
        assert resumed._group(solver, 'gka5e')['time_to_optimum_s'][0] == .075
        assert (solver, 'gka5e', 1) in resumed.pending()
        changed = deepcopy(frozen['manifest'])
        changed['execution']['open_system_campaign_identity']['launcher_sha256'] = 'changed'
        with pytest.raises(ValueError, match='resume identity mismatch'):
            CompactStore(output, changed, seeds=range(100))
        if raise_after_commit:
            raise RuntimeError('synthetic controller interruption')
        return 0

    monkeypatch.setattr(cli, 'main', fake_main)
    if raise_after_commit:
        with pytest.raises(RuntimeError, match='synthetic controller interruption'):
            launcher.run(args)
    else:
        assert launcher.run(args) == 0
    assert cli.prepare is base_prepare and cli._job is base_job
    assert CompactStore.commit_seed is base_commit
    assert '0.10%' in capsys.readouterr().out


def test_resume_targets_same_output(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'checked_identity', lambda folder: {})
    seen = []
    monkeypatch.setattr(cli, 'main', lambda argv: seen.append(cli.parser().parse_args(argv)) or 0)
    output = tmp_path / 'custom_results'
    args = SimpleNamespace(campaign=tmp_path, output=output, devices='auto', resume=True, dry_run=False)
    assert launcher.run(args) == 0
    assert seen[0].resume == str(output) == seen[0].output_root
    assert not seen[0].dry_run


@pytest.mark.parametrize('mismatch', ['solver', 'input'])
def test_prepared_selection_mismatch_is_rejected(launcher, tmp_path, monkeypatch, mismatch):
    evidence = dict(input_files={'gka5e.npz': 'input-sha'})
    monkeypatch.setattr(launcher, 'checked_identity', lambda folder: evidence)
    prepared = prepared_fixture(launcher)
    if mismatch == 'solver':
        prepared['solvers'] = ['lib_simulated_bifurcation']
    else:
        prepared['manifest']['inputs']['gka5e'] = 'different-input'
    base_prepare = lambda args: prepared
    monkeypatch.setattr(cli, 'prepare', base_prepare)
    monkeypatch.setattr(cli, 'main', lambda argv: cli.prepare(cli.parser().parse_args(argv)))
    args = SimpleNamespace(campaign=tmp_path, output=None, devices='auto', resume=False, dry_run=True)
    with pytest.raises(ValueError, match='Unexpected solver|Selected dense input'):
        launcher.run(args)
    assert cli.prepare is base_prepare


@pytest.mark.parametrize('tamper', ['protocol', 'source', 'launcher', 'campaign', 'input'])
def test_campaign_identity_detects_changed_artifacts(launcher, tmp_path, monkeypatch, tamper):
    repo, campaign = tmp_path / 'repo', tmp_path / 'campaign'
    repo.mkdir()
    campaign.mkdir()
    (repo / 'gka5e.npz').write_bytes(b'input fixture')
    (campaign / 'config.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(launcher, 'ROOT', repo)
    monkeypatch.setattr(launcher, 'source_snapshot_hash', lambda: 'source-sha')
    identity = dict(protocol=launcher.protocol(), source_snapshot_sha256='source-sha',
                    launcher_sha256=filehash(launcher.__file__),
                    campaign_files={'config.json': filehash(campaign / 'config.json')},
                    input_files={'gka5e.npz': filehash(repo / 'gka5e.npz')})
    atomic(campaign / 'identity.json', identity)
    assert launcher.checked_identity(campaign) == identity
    if tamper == 'protocol':
        identity['protocol']['seeds'] = [0]
    elif tamper == 'source':
        identity['source_snapshot_sha256'] = 'changed'
    elif tamper == 'launcher':
        identity['launcher_sha256'] = 'changed'
    elif tamper == 'campaign':
        (campaign / 'config.json').write_text('{"changed": true}', encoding='utf-8')
    else:
        (repo / 'gka5e.npz').write_bytes(b'changed')
    atomic(campaign / 'identity.json', identity)
    with pytest.raises(ValueError, match='differs|changed'):
        launcher.checked_identity(campaign)

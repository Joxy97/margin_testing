"""Frozen new20 campaign: gka5e, seeds 0..99, nine checkpoints, 20s cap.

Prepare once with ``prepare --input /path/to/gka5e.npz``. Then use ``run``,
``run --dry-run``, ``run --resume`` or ``status --watch``. Preparation and
dry-run never dispatch solver trials. Generated campaign files stay outside
the source checkout by default.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN = ROOT.parent / 'open_system_dense_campaign_20261023'
sys.path.insert(0, str(ROOT / 'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

from qubo_benchmark.runtime.common import atomic, filehash, read, source_snapshot_hash
from qubo_benchmark.runtime.compact_metrics import CHECKPOINTS
from qubo_solvers._open_system_catalog import OPEN_SYSTEM_SOLVERS


def protocol():
    return dict(problem='gka5e', variables=200, density='dense',
                solvers=list(OPEN_SYSTEM_SOLVERS), seeds=list(range(100)),
                budget_seconds=20.0, checkpoints=list(CHECKPOINTS),
                workers_per_gpu=1, wall_schedule=False, stop_on_optimum=True,
                tuning='Initial candidate defaults; no broad parameter tuning.',
                natural_return='Retain final incumbent through 20s; no restarts.',
                optimum_policy='Stop on independently rescored published optimum; carry to later checkpoints.')


def checked_identity(folder):
    identity = read(folder / 'identity.json')
    if identity['protocol'] != protocol():
        raise ValueError('Prepared campaign protocol differs from this launcher')
    if identity['source_snapshot_sha256'] != source_snapshot_hash():
        raise ValueError('Solver/runtime source changed after campaign preparation')
    if identity['launcher_sha256'] != filehash(__file__):
        raise ValueError('Launcher changed after campaign preparation')
    for name, expected in identity['campaign_files'].items():
        if filehash(folder / name) != expected:
            raise ValueError('Prepared campaign file changed: ' + name)
    for name, expected in identity['input_files'].items():
        if filehash(ROOT / name) != expected:
            raise ValueError('Prepared input or provenance changed: ' + name)
    return identity


def prepare(folder, supplied):
    from qubo_benchmark.continuous_catalog import verify_continuous_problem
    if (folder / 'identity.json').exists():
        checked_identity(folder)
        print(json.dumps(dict(status='already_prepared', campaign=str(folder)), indent=2))
        return 0
    catalog = read(ROOT / 'configs/continuous_catalog.windows.json')
    entry = next(e for e in catalog['instances'] if e['instance_id'] == 'gka5e')
    expected = entry['normalized_sha256']
    if filehash(supplied) != expected:
        raise ValueError('Input differs from the existing n200 dense gka5e input')
    target = ROOT / entry['normalized_path']
    if target.exists() and filehash(target) != expected:
        raise ValueError('An input with a different checksum already occupies ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(supplied, target)
    original_provenance = read(ROOT / entry['verification_path'])
    original_provenance['normalized_sha256'] = expected
    entry['verification_path'] = 'benchmark_data/continuous12/open_system_dense/gka5e.json'
    atomic(ROOT / entry['verification_path'], original_provenance)
    verification = verify_continuous_problem(entry, target, independent=True)
    for other in catalog['instances']:
        if other is not entry:
            other.update(preparation_status='unavailable',
                         unavailable_reason='This campaign selects only the existing n200 dense gka5e input.')
    # The generic CLI validates protected historical defaults even when none
    # are selected. Keep them exact, then explicitly select only the new20.
    config = read(ROOT / 'configs/benchmark_solvers.json')
    new = read(ROOT / 'configs/benchmark_solvers_open_system.json')['solvers']
    if set(new) != set(OPEN_SYSTEM_SOLVERS) or len(new) != 20:
        raise ValueError('Expected exactly the twenty open-system solver settings')
    config['solvers'].update(new)
    config['description'] = 'Frozen open-system dense campaign; only the explicit new20 are run.'
    atomic(folder / 'config.json', config)
    atomic(folder / 'catalog.json', catalog)
    atomic(folder / 'input_proof.json', verification)
    identity = dict(schema_version=1, protocol=protocol(),
                    source_snapshot_sha256=source_snapshot_hash(),
                    launcher_sha256=filehash(__file__),
                    campaign_files={name:filehash(folder / name) for name in
                                    ('config.json', 'catalog.json', 'input_proof.json')},
                    input_files={name:filehash(ROOT / name) for name in
                                 (entry['normalized_path'], entry['verification_path'])})
    atomic(folder / 'identity.json', identity)
    print(json.dumps(dict(status='prepared', campaign=str(folder), planned_trials=2000,
                          input=verification['instance_id'], production_started=False), indent=2))
    return 0


def run(args):
    from qubo_benchmark.runtime import continuous_cli as cli
    folder = args.campaign
    evidence = checked_identity(folder)
    output = (args.output or folder / 'results').resolve()
    original_prepare, original_job = cli.prepare, cli._job
    original_commit = cli.CompactStore.commit_seed
    started, fresh = time.monotonic(), 0

    def frozen_prepare(argv):
        prepared = original_prepare(argv)
        if prepared['solvers'] != list(OPEN_SYSTEM_SOLVERS):
            raise ValueError('Unexpected solver selection')
        if prepared['manifest']['inputs']['gka5e'] != next(
                value for key,value in evidence['input_files'].items() if key.endswith('.npz')):
            raise ValueError('Selected dense input changed')
        execution = prepared['manifest']['execution']
        execution['wall_schedule_solvers'] = []
        execution['open_system_schedule'] = 'finite fixed_steps; same settings as implementation smoke'
        execution['open_system_campaign_identity'] = deepcopy(evidence)
        return prepared

    def frozen_job(prepared, argv, solver):
        return dict(original_job(prepared, argv, solver), wall_schedule=False)

    def commit(store, *values, **kwargs):
        nonlocal fresh
        committed = original_commit(store, *values, **kwargs)
        if not committed:
            return committed
        fresh += 1
        elapsed = time.monotonic() - started
        remaining = store.planned_count - store.completed_count
        status = dict(completed=store.completed_count, planned=store.planned_count,
                      percent=100 * store.completed_count / store.planned_count,
                      elapsed_s=elapsed, eta_s=remaining * elapsed / fresh,
                      eta_policy='Measured rate since this process started; includes setup overhead.',
                      solver=values[0], seed=values[2],
                      last_status=kwargs.get('status', 'complete'),
                      has_failures=store.has_failures,
                      note='Completion counts all terminal attempts, including failures.')
        atomic(output / 'campaign_progress.json', status)
        print(f"Campaign: {status['percent']:.2f}% ({status['completed']}/2000), "
              f"ETA {status['eta_s']/60:.1f} min, failures={status['has_failures']}", flush=True)
        return committed

    argv = ['--continuous', '--variable-count', '200', '--density', 'dense',
            '--config', str(folder / 'config.json'), '--catalog', str(folder / 'catalog.json'),
            '--solvers', ','.join(OPEN_SYSTEM_SOLVERS), '--runs', '100', '--seed-start', '0',
            '--devices', args.devices, '--require-gpu', '--seed-workers', '1', '--no-autotune',
            '--stop-on-optimum', '--keep-raw-results', '--no-plots', '--output-root', str(output)]
    if args.resume:
        argv += ['--resume', str(output)]
    if args.dry_run:
        argv += ['--dry-run', '--json']
    cli.prepare, cli._job, cli.CompactStore.commit_seed = frozen_prepare, frozen_job, commit
    try:
        return cli.main(argv)
    finally:
        cli.prepare, cli._job, cli.CompactStore.commit_seed = original_prepare, original_job, original_commit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    stage = sub.add_parser('prepare', help='Freeze settings and verify the existing input; no trials')
    stage.add_argument('--input', type=Path, required=True)
    launch = sub.add_parser('run', help='Run the fixed 2,000-trial plan (or --dry-run)')
    launch.add_argument('--devices', default='auto', help='Default: all visible CUDA GPUs')
    launch.add_argument('--resume', action='store_true', help='Resume exactly this frozen campaign')
    launch.add_argument('--dry-run', action='store_true', help='Validate inputs and GPU admission without trials')
    status = sub.add_parser('status', help='Read-only progress and ETA')
    status.add_argument('--watch', action='store_true')
    for child in (stage, launch, status):
        child.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    for child in (launch, status):
        child.add_argument('--output', type=Path, help='Default: CAMPAIGN/results')
    args = parser.parse_args(argv)
    args.campaign = args.campaign.resolve()
    try:
        if args.command == 'prepare':
            return prepare(args.campaign, args.input)
        if args.command == 'run':
            return run(args)
        path = ROOT / 'tools/monitor_open_system_campaign.py'
        spec = importlib.util.spec_from_file_location('campaign_monitor', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(['--output', str(args.output or args.campaign / 'results')]
                           + (['--watch'] if args.watch else []))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Open-system campaign: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())

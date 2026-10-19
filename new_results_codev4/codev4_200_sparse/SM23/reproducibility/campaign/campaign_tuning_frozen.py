"""Bounded parameter tuning and opt-in canonical continuous production runs.

No production run starts from ``tune``. Numerical code is imported from --repo.
Tuning uses independent seeds; production delegates to the published runner.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import statistics
import sys
import time


def bootstrap(repo):
    sys.path.insert(0, str(Path(repo).resolve() / 'src'))


def _worker(index, device, incoming, outgoing, repo):
    bootstrap(repo)
    from qubo_benchmark.runtime.worker import WarmWorker
    from qubo_benchmark.runtime.continuous_cli import _prepare_job
    from qubo_benchmark.runtime import continuous_worker as continuous
    from qubo_solvers.observation import SolveInterrupted
    import gc
    worker = WarmWorker(dict(device=device, cpu_threads=1))
    outgoing.put(dict(kind='ready', worker=index))
    for job in iter(incoming.get, None):
        job = dict(job, device=device)
        outgoing.put(dict(kind='started', worker=index, key=job['key']))
        original = continuous.ScalarObserver
        observers = []

        class PilotObserver(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.iterations = 0
                self.iteration_elapsed = 0.
                self.natural_elapsed = None
                observers.append(self)

            def capture(self, samples, raw=None, *, phase='checkpoint', iteration=None):
                super().capture(samples, raw, phase=phase, iteration=iteration)
                if iteration is not None and iteration > self.iterations:
                    self.iterations = int(iteration)
                    self.iteration_elapsed = self.elapsed()

            def finish(self):
                if self.natural_elapsed is None:
                    self.natural_elapsed = self.elapsed()
                return super().finish()

        try:
            _prepare_job(worker, job)
            if job.get('pilot'):
                continuous.ScalarObserver = PilotObserver
            result = continuous.run_continuous_trial(worker, job)
            if observers:
                observer = observers[0]
                result.update(iterations=observer.iterations,
                              iteration_elapsed=observer.iteration_elapsed,
                              natural_elapsed=observer.natural_elapsed)
        except Exception as exc:
            result = dict(status='error', error=f'{type(exc).__name__}: {exc}',
                          energies=[None] * len(job['checkpoints']))
        finally:
            continuous.ScalarObserver = original
        outgoing.put(dict(kind='done', worker=index, key=job['key'],
                          result=dict(result, seed=job['seed'], device=device)))
        gc.collect()


class TuningPool:
    """Four persistent independent workers, bounded queues, durable controller."""
    def __init__(self, repo, devices, directory, identity):
        from qubo_benchmark.runtime.common import atomic, read, digest
        self.atomic, self.read, self.digest = atomic, read, digest
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        identity_path = self.directory / 'identity.json'
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError('Tuning resume identity differs; use a fresh output directory')
        atomic(identity_path, identity)
        self.identity = identity
        context = mp.get_context('spawn')
        self.outgoing = context.Queue()
        self.lanes = []
        for index, device in enumerate(devices):
            incoming = context.Queue(maxsize=1)
            process = context.Process(target=_worker,
                args=(index, device, incoming, self.outgoing, str(repo)))
            process.start()
            self.lanes.append((process, incoming))
        ready = set()
        started = time.monotonic()
        while len(ready) != len(self.lanes):
            try:
                message = self.outgoing.get(timeout=1.)
            except queue.Empty:
                if time.monotonic() - started > 120 or any(not p.is_alive() for p, _ in self.lanes):
                    self.close()
                    raise RuntimeError('Tuning worker startup failed')
                continue
            if message['kind'] == 'ready':
                ready.add(message['worker'])

    def run(self, jobs, phase):
        from qubo_benchmark.runtime.common import source_snapshot_hash
        jobs = list(jobs)
        rows, pending = {}, deque()
        for job in jobs:
            path = self.directory / 'trials' / (job['key'] + '.json')
            if path.exists():
                saved = self.read(path)
                if saved['job'] != json.loads(json.dumps(job)):
                    raise ValueError('Tuning job changed during resume')
                rows[job['key']] = saved['result']
            else:
                pending.append(job)
        active = {}
        started, already = time.monotonic(), len(rows)
        try:
            while pending or active:
                for index, (process, incoming) in enumerate(self.lanes):
                    if not process.is_alive():
                        raise RuntimeError('Tuning worker exited; pending work remains resumable')
                    if index not in active and pending:
                        job = pending.popleft()
                        active[index] = (job, time.monotonic())
                        incoming.put(job)
                try:
                    message = self.outgoing.get(timeout=1.)
                except queue.Empty:
                    if any(time.monotonic() - began > max(120., job['budget'] + 60.)
                           for job, began in active.values()):
                        raise RuntimeError('Tuning worker watchdog expired; active jobs remain pending')
                    continue
                if message['kind'] != 'done':
                    continue
                job, _ = active.pop(message['worker'])
                if job['key'] != message['key']:
                    raise RuntimeError('Worker returned a mismatched tuning job')
                if source_snapshot_hash() != self.identity['source_snapshot']:
                    raise RuntimeError('Numerical source changed during tuning')
                result = message['result']
                self.atomic(self.directory / 'trials' / (job['key'] + '.json'),
                            dict(job=job, result=result))
                rows[job['key']] = result
                fresh = len(rows) - already
                elapsed = time.monotonic() - started
                progress = dict(phase=phase, completed=len(rows), planned=len(jobs),
                    percent=100. * len(rows) / len(jobs), elapsed_s=elapsed,
                    eta_s=(len(jobs) - len(rows)) * elapsed / fresh if fresh else None,
                    solver=job['solver'], candidate=job['candidate'], status=result['status'])
                self.atomic(self.directory / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        except BaseException:
            self.close()
            raise
        return rows

    def close(self):
        for process, incoming in self.lanes:
            if process.is_alive():
                try:
                    incoming.put_nowait(None)
                except queue.Full:
                    pass
        for process, _ in self.lanes:
            process.join(timeout=2.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3.)


def quality(rows, reference):
    """Paired final objective then full checkpoint mean; failures are worst."""
    if any(row['status'] != 'complete' or row['energies'][-1] is None for row in rows):
        return (float('inf'), float('inf'))
    final = statistics.mean(row['energies'][-1] for row in rows)
    path = statistics.mean(statistics.mean(value if value is not None else abs(reference)
                                           for value in row['energies']) for row in rows)
    return final, path


def tune(args):
    from qubo_benchmark.continuous_catalog import (
        load_continuous_catalog, select_continuous_problem, verify_continuous_problem)
    from qubo_benchmark.runtime.common import atomic, read, filehash, source_snapshot_hash, digest
    from qubo_solvers._hamiltonian_catalog import HAMILTONIAN_SOLVERS
    from qubo_benchmark.runtime.compact_metrics import CHECKPOINTS
    candidates = read(args.candidates)
    if set(candidates['solvers']) != set(HAMILTONIAN_SOLVERS):
        raise ValueError('Exactly the 23 new Hamiltonian solvers must be configured')
    entry, npz = select_continuous_problem(200, args.density, load_continuous_catalog(args.catalog))
    if verify_continuous_problem(entry, npz, independent=True)['status'] != 'passed':
        raise ValueError('Input verification failed')
    output = Path(args.output or Path(__file__).resolve().parent / args.density / 'tuning')
    devices = args.devices.split(',')
    reference = entry['normalized_reference_objective_min']
    identity = dict(source_snapshot=source_snapshot_hash(), input_sha256=filehash(npz),
                    candidates_sha256=filehash(args.candidates), devices=devices,
                    density=args.density, pilot_seconds=args.pilot_seconds,
                    screen_seconds=args.screen_seconds, holdout_seconds=20.,
                    screen_seeds=[1000, 1001], holdout_seeds=list(range(2001, 2001 + args.holdout_seeds)),
                    schedule='fixed_steps', helper_sha256=filehash(__file__))
    common = candidates['common']

    def job(solver, candidate, parameters, phase, seed, budget, pilot=False):
        return dict(key=f'{phase}__{solver}__{candidate}__{seed}',
            candidate=candidate, pilot=pilot, solver=solver, parameters=parameters,
            seed=seed, cpu_threads=1, npz=str(Path(npz).resolve()),
            input_sha256=identity['input_sha256'], budget=budget,
            checkpoints=list(CHECKPOINTS) if budget == 20. else [budget],
            wall_schedule=False, reference=reference,
            reference_type='OPTIMUM' if entry['reference_status'] in
                ('published_proven_optimum', 'independently_verified_optimum') else 'BKS',
            tolerance=0. if entry.get('integer_objective', True) else 1e-9,
            stop_on_optimum=not pilot)

    pool = TuningPool(args.repo, devices, output, identity)
    try:
        pilots = []
        for solver, options in candidates['solvers'].items():
            parameters = dict(common, **options[0]['parameters'], max_steps=16)
            pilots.append(job(solver, 'baseline', parameters, 'pilot', 9000,
                              args.pilot_seconds, True))
        pilot_rows = pool.run(pilots, 'pilot')
        steps = {}
        for item in pilots:
            row = pilot_rows[item['key']]
            count = row.get('iterations', 0)
            duration = row.get('iteration_elapsed', 0.)
            if count and duration > 0:
                steps[item['solver']] = max(1, min(20000, math.floor(18. * count / duration)))
            else:
                steps[item['solver']] = 1
        atomic(output / 'timing_steps.json', steps)
        screening = []
        full_parameters = {}
        for solver, options in candidates['solvers'].items():
            full_parameters[solver] = {}
            for option in options:
                params = dict(common, **option['parameters'], max_steps=steps[solver])
                full_parameters[solver][option['id']] = params
                short = dict(params, max_steps=max(1, math.floor(steps[solver] * args.screen_seconds / 20.)))
                screening.extend(job(solver, option['id'], short, 'screen', seed,
                                     args.screen_seconds) for seed in (1000, 1001))
        screen_rows = pool.run(screening, 'screen')
        winners = {}
        holdout = []
        for solver, options in candidates['solvers'].items():
            def rank(option):
                rows = [screen_rows[j['key']] for j in screening
                        if j['solver'] == solver and j['candidate'] == option['id']]
                return quality(rows, reference)
            winner = min(options, key=rank)['id']
            winners[solver] = winner
            baseline = options[0]['id']
            for candidate in dict.fromkeys((baseline, winner)):
                holdout.extend(job(solver, candidate, full_parameters[solver][candidate],
                                   'holdout', seed, 20.) for seed in identity['holdout_seeds'])
        atomic(output / 'screen_winners.json', winners)
        holdout_rows = pool.run(holdout, 'holdout')
        baseline_config = read(Path(args.repo) / 'configs' / 'benchmark_solvers.json')
        config = deepcopy(baseline_config)
        selections = {}
        for solver, options in candidates['solvers'].items():
            baseline, winner = options[0]['id'], winners[solver]
            def held(candidate):
                return quality([holdout_rows[j['key']] for j in holdout
                    if j['solver'] == solver and j['candidate'] == candidate], reference)
            selected = winner if held(winner) < held(baseline) else baseline
            if not math.isfinite(held(selected)[0]):
                atomic(output / 'validation_failure.json', dict(solver=solver,
                    reason='Neither baseline nor screened candidate passed heldout validation',
                    trials=[dict(job=j, result=holdout_rows[j['key']])
                            for j in holdout if j['solver'] == solver]))
                raise RuntimeError(f'No validated production configuration for {solver}')
            config['solvers'][solver] = full_parameters[solver][selected]
            selections[solver] = dict(selected=selected, screened=winner,
                baseline_score=[v if math.isfinite(v) else None for v in held(baseline)],
                selected_score=[v if math.isfinite(v) else None for v in held(selected)])
        atomic(output.parent / 'selected_config.json', config)
        report = dict(identity=identity, complete=True, production_started=False,
            selections=selections, pilot_trials=len(pilots), screen_trials=len(screening),
            holdout_trials=len(holdout), total_trials=len(pilots)+len(screening)+len(holdout),
            failures=[dict(key=k, status=v['status'], error=v.get('error'))
                for rows in (pilot_rows, screen_rows, holdout_rows) for k, v in rows.items()
                if v['status'] != 'complete'],
            limitation=f'Short exploratory tuning: two screening seeds and {args.holdout_seeds} fresh validation seeds; validation gates acceptance against baseline and is not an unbiased performance estimate. Not statistical optimality evidence.')
        atomic(output / 'report.json', report)
        print(json.dumps(dict(phase='tuning_complete', solvers=len(selections),
                             configuration=str(output.parent / 'selected_config.json'),
                             failures=len(report['failures']), production_started=False)), flush=True)
    finally:
        pool.close()


def production(args):
    """Keep existing script/storage semantics; restrict methods and fixed steps."""
    from qubo_benchmark.runtime import continuous_cli as cli
    from qubo_benchmark.runtime.common import atomic, read
    from qubo_solvers._hamiltonian_catalog import HAMILTONIAN_SOLVERS
    config = Path(args.config or Path(__file__).resolve().parent / args.density / 'selected_config.json')
    output = Path(args.output or Path(__file__).resolve().parent / args.density / 'production')
    original_prepare, original_job = cli.prepare, cli._job
    original_commit = cli.CompactStore.commit_seed
    started = time.monotonic()
    fresh = 0

    def prepare(argv):
        prepared = original_prepare(argv)
        prepared['manifest']['execution']['wall_schedule_solvers'] = []
        prepared['manifest']['execution']['hamiltonian_schedule'] = 'finite tuned fixed_steps'
        return prepared

    def make_job(prepared, argv, solver):
        return dict(original_job(prepared, argv, solver), wall_schedule=False)

    def commit(store, *values, **kwargs):
        nonlocal fresh
        result = original_commit(store, *values, **kwargs)
        fresh += 1
        elapsed = time.monotonic() - started
        progress = dict(phase='production', density=args.density,
            completed=store.completed_count, planned=store.planned_count,
            percent=100. * store.completed_count / store.planned_count,
            elapsed_s=elapsed, eta_s=(store.planned_count-store.completed_count)*elapsed/fresh)
        atomic(output / 'campaign_progress.json', progress)
        print(json.dumps(progress), flush=True)
        return result

    cli.prepare, cli._job, cli.CompactStore.commit_seed = prepare, make_job, commit
    argv = ['--continuous', '--variable-count', '200', '--density', args.density,
            '--config', str(config), '--catalog', args.catalog,
            '--solvers', ','.join(HAMILTONIAN_SOLVERS), '--runs', str(args.runs),
            '--seed-start', '0', '--devices', args.devices, '--require-gpu',
            '--seed-workers', '1', '--no-autotune', '--stop-on-optimum',
            '--keep-raw-results', '--no-plots', '--output-root', str(output)]
    if args.resume:
        argv.extend(['--resume', str(output)])
    if args.dry_run:
        argv.append('--dry-run')
    try:
        return cli.main(argv)
    finally:
        cli.prepare, cli._job, cli.CompactStore.commit_seed = original_prepare, original_job, original_commit


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['tune', 'run'])
    parser.add_argument('--repo', default=str(root / 'repo'))
    parser.add_argument('--catalog', default=str(root / 'catalog.json'))
    parser.add_argument('--candidates', default=str(root / 'tuning_candidates.json'))
    parser.add_argument('--density', choices=['sparse', 'dense'], required=True)
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3')
    parser.add_argument('--output')
    parser.add_argument('--config')
    parser.add_argument('--pilot-seconds', type=float, default=1.)
    parser.add_argument('--screen-seconds', type=float, default=2.)
    parser.add_argument('--holdout-seeds', type=int, choices=(1, 2), default=1)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    bootstrap(args.repo)
    if args.command == 'tune':
        tune(args)
        return 0
    return production(args)


if __name__ == '__main__':
    raise SystemExit(main())

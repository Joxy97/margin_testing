"""Frozen 130-preset dense campaign: one persistent worker per GPU, seeds 0..99.

Run only an explicitly assigned host partition. The 20-second limit governs
certified quality, not hard GPU termination. A separate 300-second watchdog
records failed attempts without retrying seeds or restarting GPU workers.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time
import traceback
import uuid

COMMIT = '05a630dd865fe0d062af70a68a61bbde8af31bed'
CHECKPOINTS = [.05, .1, .2, .5, 1., 2., 5., 10., 20.]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable(path, value):
    if path.exists():
        require(read(path) == value, 'Existing terminal evidence differs: '+str(path))
    else:
        atomic(path, value)


def load_replay(repo):
    sys.path.insert(0, str(repo/'src'))
    spec = importlib.util.spec_from_file_location('_published_fine_tuning_replay', repo/'fine_tuning/run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_plan(plan, settings, host):
    eligible = set(settings['solvers'])-set(settings['excluded_unchanged'])
    assignments = plan['assignments']; ordered = [s for group in assignments.values() for s in group]
    require(plan['schema_version'] == 1 and plan['git_commit'] == COMMIT, 'Unexpected campaign schema or published commit.')
    require(len(eligible) == 130 and len(plan['solver_ids']) == 130 and set(plan['solver_ids']) == eligible,
            'Campaign must cover exactly the 130 tuned solvers.')
    require(set(plan['excluded_solvers']) == set(settings['excluded_unchanged']) and len(plan['excluded_solvers']) == 14,
            'The 14 excluded presets must remain excluded.')
    require(len(assignments) == 7 and sorted(map(len, assignments.values())) == [10]+[20]*6
            and len(ordered) == len(set(ordered)) == 130 and set(ordered) == eligible and host in assignments,
            'Expected six disjoint 20-solver partitions and one 10-solver partition.')
    require(plan['seeds'] == list(range(100)) and plan['budget_seconds'] == 20.
            and plan['checkpoints'] == CHECKPOINTS and plan['reference'] == -35154., 'Frozen seed or timing protocol differs.')
    require(plan['workers_per_gpu'] == 1 and plan['cpu_threads'] == 1 and plan['tf32'] is False,
            'Worker, thread or precision protocol differs.')
    require(set(plan['gpu_counts']) == set(assignments) and all(type(n) is int and n > 0 for n in plan['gpu_counts'].values()),
            'Every host needs its expected visible GPU count.')
    require(plan['watchdog_seconds'] >= 300 and math.isfinite(plan['watchdog_seconds']),
            'Independent hang watchdog must allow at least 300 seconds.')


def preflight(args):
    plan = read(args.plan); replay = load_replay(args.repo)
    require(sha(args.repo/'fine_tuning/run.py') == plan['replay_sha256'], 'Published replay source differs.')
    head = subprocess.run(['git', '-c', 'safe.directory='+args.repo.as_posix(), '-C', str(args.repo), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    require(head == COMMIT, 'Repository is not at the published preset commit.')
    args.input = args.input or args.repo/read(args.repo/'fine_tuning/protocol.json')['problem']['repository_npz']
    first_solver = plan['assignments'][args.host][0]
    settings, protocol, evidence = replay.validate_bundle(args.input, first_solver, plan['seeds'])
    validate_plan(plan, settings, args.host)
    from qubo_solvers import create_bqm_solver
    for sid in plan['assignments'][args.host]:
        create_bqm_solver(sid, {'device': 'cpu'})._getParameters(copy.deepcopy(settings['solvers'][sid]))
    for key in ('source_sha256', 'selected_settings_sha256', 'protocol_sha256'):
        require(plan[key] == evidence[key], 'Campaign/published bundle identity differs: '+key)
    require(plan['input']['numeric_coo_sha256'] == evidence['numeric_coo_sha256']
            and plan['input']['original_npz_sha256'] == evidence['original_npz_sha256'], 'Campaign input identity differs.')
    base = dict(schema_version=1, campaign=plan['campaign'], plan_sha256=digest(plan), host=args.host,
        git_commit=COMMIT, launcher_sha256=sha(Path(__file__)), replay_sha256=plan['replay_sha256'],
        solvers=plan['assignments'][args.host], seeds=plan['seeds'], input=evidence,
        parameters={s: settings['solvers'][s] for s in plan['assignments'][args.host]},
        schedules={s: settings['schedule_by_solver'][s] for s in plan['assignments'][args.host]},
        checkpoints=CHECKPOINTS, budget_seconds=20., reference=-35154., workers_per_gpu=1, cpu_threads=1, tf32=False)
    return plan, settings, base


def task_id(identity, solver, seed):
    return digest(dict(campaign=identity['digest'], solver=solver, seed=seed,
        parameters=identity['parameters'][solver], schedule=identity['schedules'][solver]))


def expected_tasks(identity):
    return [(sid, seed) for sid in identity['solvers'] for seed in identity['seeds']]


def validate_trial(row, identity, sid, seed):
    require(row['status'] in {'complete', 'error', 'oom', 'validation_error', 'invalid_output', 'no_in_budget_candidate',
                             'setup_error', 'interrupted', 'worker_error', 'watchdog_timeout'}, 'Unknown terminal trial status.')
    require(row['solver'] == sid and row['seed'] == seed and row['identity'] == identity['digest']
            and row['task_identity'] == task_id(identity, sid, seed)
            and row['parameters'] == identity['parameters'][sid]
            and row['wall_schedule'] == (identity['schedules'][sid] == 'wall'), 'Trial identity or parameters differ.')
    hardware = identity['hardware'][row['device']]
    require(row['hardware_digest'] == digest(hardware), 'Trial GPU/environment identity differs.')
    energies = row['energies']; observed = [e for e in energies if e is not None]
    require(len(energies) == len(CHECKPOINTS), 'Trial checkpoint count differs.')
    require(all(isinstance(e, (int, float)) and math.isfinite(e) and e >= -35154. for e in observed)
            and all(a >= b for a, b in zip(observed, observed[1:])), 'Invalid or nonmonotone source-scored energy.')
    if observed and row['status'] == 'complete':
        start = next(i for i, e in enumerate(energies) if e is not None)
        require(all(e is not None for e in energies[start:]), 'Missing checkpoint after the first certified candidate.')
    require(math.isfinite(row['actual_solve_wall_s']) and row['actual_solve_wall_s'] >= 0,
            'Invalid solver return time.')
    require(math.isfinite(row['caller_wall_seconds']) and row['caller_wall_seconds'] >= 0,
            'Invalid caller duration.')
    if row['status'] == 'complete':
        require(energies[-1] is not None, 'Successful trial lacks a final in-budget candidate.')
        optimum = row.get('time_to_optimum_s')
        require((optimum is not None) == (energies[-1] == -35154.), 'Optimum timestamp and final energy disagree.')
        if optimum is not None:
            require(0 <= optimum < 20. and optimum <= row['actual_solve_wall_s'] + 1e-8,
                    'An optimum was credited after its certification deadline.')
            require(all(e == -35154. for point, e in zip(CHECKPOINTS, energies) if point > optimum),
                    'Known optimum was not propagated to later checkpoints.')
            require(all(e != -35154. for point, e in zip(CHECKPOINTS, energies) if point < optimum),
                    'An optimum was credited before its certified timestamp.')


def scan(output, identity):
    expected = set(expected_tasks(identity)); rows = {}; hashes = {}
    for path in (output/'trials').glob('*/*.json'):
        key = path.parent.name, int(path.stem)
        require(key in expected and key not in rows, 'Unexpected or duplicate campaign trial.')
        row = read(path); validate_trial(row, identity, *key)
        rows[key] = row; hashes[path.relative_to(output).as_posix()] = sha(path)
    return rows, hashes


def completed_result(output, identity, rows, hashes):
    path = output/'completion.json'
    if not path.exists():
        return None
    report = read(path)
    require(report['identity'] == identity['digest'] and report['trial_files_sha256'] == digest(hashes)
            and report['attempted'] == report['planned'] == len(expected_tasks(identity)) == len(rows)
            and report['failed'] == sum(r['status'] != 'complete' for r in rows.values())
            and report['status'] in {'complete', 'completed_with_errors'}
            and set(report['exit_codes']) == {str(i) for i in range(len(identity['hardware']))},
            'Saved completion differs from the verified terminal evidence.')
    if report['status'] == 'complete':
        require(not report['failed'] and not report['watchdog_killed']
                and all(code == 0 for code in report['exit_codes'].values()), 'Completion claims success despite controller failures.')
    return report


def require_idle_gpus():
    # The approved host is dedicated to this campaign. Checking all compute
    # processes also catches orphan workers whose controller lock is stale.
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=30)
    require(not result.stdout.strip(), 'GPU compute processes are already active; refusing overlapping workers or unsafe resume.')


def summary(identity, rows, active=(), duration_hints=None, active_gpu_count=None):
    per_solver = {}
    hints = duration_hints or {}
    for sid in identity['solvers']:
        subset = [rows[sid, seed] for seed in identity['seeds'] if (sid, seed) in rows]
        successful = [r for r in subset if r['status'] == 'complete']
        times = [r['caller_wall_seconds'] for r in successful]
        duration = sum(times)/len(times) if times else hints.get(sid, 20.1)
        per_solver[sid] = dict(attempted=len(subset), successful=len(successful), failed=len(subset)-len(successful),
            exact=sum(r['energies'][-1] == -35154. for r in successful),
            within_1pct=sum(r['energies'][-1] <= -35154.+351.54 for r in successful),
            mean_gap_percent=(sum(100*(r['energies'][-1]+35154.)/35154. for r in successful)/len(successful) if successful else None),
            completed_all_100=len(subset) == 100, successful_all_100=len(successful) == 100,
            estimated_remaining_worker_seconds=(100-len(subset))*duration)
    planned = len(identity['solvers'])*100
    failures = sum(s['failed'] for s in per_solver.values())
    return dict(updated_at_utc=utc(), host=identity['host'], identity=identity['digest'], attempted=len(rows),
        planned=planned, successful=len(rows)-failures, failed=failures, percent=100*len(rows)/planned,
        completed_solvers=sum(s['completed_all_100'] for s in per_solver.values()), solver_count=len(per_solver),
        eta_seconds=sum(s['estimated_remaining_worker_seconds'] for s in per_solver.values())/max(1, active_gpu_count if active_gpu_count is not None else len(identity['hardware'])),
        eta_policy='Remaining jobs times per-solver observed mean caller time, otherwise plan estimate/20.1s, divided by GPU count; cooperative return can exceed20s.',
        active=list(active), solvers=per_solver)


def failed_claim(claim, identity, reason):
    return dict(claim, status=reason, error='The started attempt ended without a committed solver result; it will not be rerun automatically.',
        energies=[None]*len(CHECKPOINTS), time_to_optimum_s=None, actual_solve_wall_s=0., caller_wall_seconds=0.,
        completed_at_utc=utc(), actual_iterations=None)


def recover_claims(output, identity, rows, reason='interrupted'):
    for path in (output/'claims').glob('*/*.json'):
        sid, seed = path.parent.name, int(path.stem)
        claim = read(path)
        require((sid, seed) in set(expected_tasks(identity)) and claim['task_identity'] == task_id(identity, sid, seed),
                'An existing attempt claim belongs to another task.')
        if (sid, seed) not in rows:
            failure = reason.get(claim['device'], 'interrupted') if isinstance(reason, dict) else reason
            row = failed_claim(claim, identity, failure); validate_trial(row, identity, sid, seed)
            immutable(output/'trials'/sid/f'{seed}.json', row); rows[sid, seed] = row


def acquire_lock(output):
    import psutil
    lock = output/'.run.lock'
    if lock.exists():
        old = read(lock)
        require(old['hostname'] == socket.gethostname(), 'Controller lock belongs to another machine.')
        try:
            active = abs(psutil.Process(old['pid']).create_time()-old['process_created']) < .01
        except psutil.NoSuchProcess:
            active = False
        require(not active, 'Another campaign controller is active.')
        archive = output/'stale_locks'/('lock_'+uuid.uuid4().hex+'.json'); archive.parent.mkdir(exist_ok=True)
        os.rename(lock, archive)
    value = dict(pid=os.getpid(), process_created=psutil.Process().create_time(), hostname=socket.gethostname())
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
    return lock


def hardware_record(torch, device):
    import numpy
    import scipy
    properties = torch.cuda.get_device_properties(device)
    smi = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    return dict(device=device, host=socket.gethostname(), gpu_name=properties.name,
        gpu_uuid=str(getattr(properties, 'uuid', 'unavailable')), total_memory=properties.total_memory,
        python=sys.version, python_executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        numpy=numpy.__version__, scipy=scipy.__version__, nvidia_smi=smi, cpu_threads=1,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)


def worker_main(payload, gpu, tasks, events, start, stop):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    args = argparse.Namespace(**payload); args.repo = Path(args.repo); args.output = Path(args.output); args.input = Path(args.input)
    replay = load_replay(args.repo)
    import torch
    import qubo_solvers
    import qubo_benchmark.runtime.continuous_worker as cw
    from qubo_benchmark.runtime.worker import WarmWorker
    require(Path(qubo_solvers.__file__).resolve().is_relative_to((args.repo/'src').resolve())
            and Path(cw.__file__).resolve().is_relative_to((args.repo/'src').resolve()), 'Wrong installed source checkout.')
    device = f'cuda:{gpu}'; worker = WarmWorker(dict(device=device, cpu_threads=1))
    hardware = hardware_record(torch, device)
    events.put(dict(kind='ready', gpu=gpu, hardware=hardware))
    while not start.wait(1):
        if stop.is_set():
            return
    if stop.is_set():
        return
    identity = read(args.output/'identity.json'); settings = read(args.repo/'fine_tuning/selected_settings.json')
    require(identity['hardware'][device] == hardware, 'GPU/environment differs from frozen campaign identity.')
    cw.ScalarObserver = replay.immediate_observer(cw.ScalarObserver)
    while not stop.is_set():
        task = tasks.get()
        if task is None:
            break
        sid, seed = task
        job = replay.make_job(settings, sid, seed, args.input, device)
        claim = dict(identity=identity['digest'], task_identity=task_id(identity, sid, seed), solver=sid, seed=seed,
            parameters=copy.deepcopy(job['parameters']), wall_schedule=job['wall_schedule'], device=device,
            hardware_digest=digest(hardware), pid=os.getpid(), started_at_utc=utc(), status='running')
        require(not (args.output/'trials'/sid/f'{seed}.json').exists()
                and not (args.output/'claims'/sid/f'{seed}.json').exists(), 'Task was already attempted; refusing an implicit rerun.')
        immutable(args.output/'claims'/sid/f'{seed}.json', claim)
        events.put(dict(kind='active', gpu=gpu, solver=sid, seed=seed))
        began = time.perf_counter()
        # An integrity change is fatal; ordinary solver failures remain terminal
        # attempts while unrelated queued work continues on all healthy GPUs.
        require(sha(args.input) == identity['input']['actual_npz_sha256'], 'Input bytes changed during the campaign.')
        version = None
        try:
            setup = worker.prepare(job); version = worker.prepared.Q._version if worker.native else None
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            result = cw.run_continuous_trial(worker, job, stop)
            torch.cuda.synchronize(device)
            result.update(setup=setup, source_integer_scoring_exact=worker._continuous_scorer.exact_float32,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device), gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        except Exception as error:
            result = dict(status='setup_error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                energies=[None]*len(CHECKPOINTS), actual_solve_wall_s=0., time_to_optimum_s=None)
        # A corrupted cached input must never reach another seed. Let this
        # worker exit; healthy GPU workers keep draining the shared queue.
        require(version is None or worker.prepared.Q._version == version, 'Solver modified immutable input.')
        row = dict(claim, **{k: v for k, v in result.items() if k != 'status'}, status=result['status'],
            caller_wall_seconds=time.perf_counter()-began, completed_at_utc=utc(), actual_iterations=None,
            iteration_note='Configured ceilings are retained; realized iteration counts are not exposed by the scalar runtime.')
        validate_trial(row, identity, sid, seed)
        immutable(args.output/'trials'/sid/f'{seed}.json', row)
        events.put(dict(kind='committed', gpu=gpu, solver=sid, seed=seed))
        if row['status'] in ('oom', 'setup_error'):
            import gc
            gc.collect(); torch.cuda.empty_cache()
    events.put(dict(kind='done', gpu=gpu))


def run(args, plan, base):
    import torch
    require(torch.cuda.is_available(), 'CUDA GPUs are required; CPU fallback is forbidden.')
    count = torch.cuda.device_count()
    require(count == plan['gpu_counts'][args.host], 'Visible GPU count differs from the approved host plan.')
    require(args.resume or not args.output.exists(), 'Output already exists; use explicit --resume for the exact campaign.')
    args.output.mkdir(parents=True, exist_ok=True); lock = acquire_lock(args.output)
    ctx = mp.get_context('spawn'); tasks = ctx.Queue(); events = ctx.Queue(); start = ctx.Event(); stop = ctx.Event()
    payload = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    processes = []; active = {}; watchdog = []; identity = None; rows = {}
    try:
        require_idle_gpus()
        for gpu in range(count):
            process = ctx.Process(target=worker_main, args=(payload, gpu, tasks, events, start, stop), name=f'cuda_{gpu}')
            process.start(); processes.append(process)
        hardware = {}; deadline = time.monotonic()+plan['watchdog_seconds']
        while len(hardware) < count:
            require(time.monotonic() < deadline and all(p.is_alive() for p in processes), 'GPU worker initialization failed or timed out.')
            try:
                event = events.get(timeout=1)
            except queue.Empty:
                continue
            require(event['kind'] == 'ready' and event['gpu'] not in hardware, 'Unexpected worker readiness event.')
            hardware[event['gpu']] = event['hardware']
        identity = dict(base, hardware={f'cuda:{i}': hardware[i] for i in range(count)})
        identity['digest'] = digest(identity)
        if (args.output/'identity.json').exists():
            require(read(args.output/'identity.json') == identity, 'Resume source, plan, input, settings, launcher or hardware identity differs.')
        else:
            require(not any((args.output/'trials').glob('*/*.json')), 'Existing trials lack a campaign identity.')
            immutable(args.output/'identity.json', identity)
        rows, hashes = scan(args.output, identity)
        finished = completed_result(args.output, identity, rows, hashes)
        if finished is not None:
            print(json.dumps({k: v for k, v in finished.items() if k != 'solvers'}, indent=2), flush=True)
            return 0 if finished['status'] == 'complete' else 1
        recover_claims(args.output, identity, rows)
        pending = [task for task in expected_tasks(identity) if task not in rows]
        for task in pending:
            tasks.put(task)
        for _ in range(count):
            tasks.put(None)
        start.set(); last_update = 0.
        while any(p.is_alive() for p in processes):
            try:
                event = events.get(timeout=1); gpu = event['gpu']
                if event['kind'] == 'active':
                    active[gpu] = dict(solver=event['solver'], seed=event['seed'], started=time.monotonic())
                elif event['kind'] == 'committed':
                    key = event['solver'], event['seed']
                    row = read(args.output/'trials'/key[0]/f'{key[1]}.json'); validate_trial(row, identity, *key)
                    rows[key] = row; active.pop(gpu, None)
                elif event['kind'] == 'done':
                    active.pop(gpu, None)
            except queue.Empty:
                pass
            for gpu, process in enumerate(processes):
                if gpu in active and process.is_alive() and time.monotonic()-active[gpu]['started'] > plan['watchdog_seconds']:
                    watchdog.append(dict(gpu=gpu, **active[gpu])); process.terminate()
            if time.monotonic()-last_update >= 5:
                report = summary(identity, rows, [dict(gpu=gpu, solver=t['solver'], seed=t['seed']) for gpu, t in active.items()],
                                 plan.get('expected_caller_seconds'), sum(p.is_alive() for p in processes))
                atomic(args.output/'progress.json', report)
                print(json.dumps({k: v for k, v in report.items() if k != 'solvers'}), flush=True)
                last_update = time.monotonic()
        for process in processes:
            process.join()
        rows, hashes = scan(args.output, identity)
        recover_claims(args.output, identity, rows, {f'cuda:{entry["gpu"]}': 'watchdog_timeout' for entry in watchdog})
        rows, hashes = scan(args.output, identity)
        report = summary(identity, rows, duration_hints=plan.get('expected_caller_seconds'))
        report.update(status=('completed_with_errors' if report['failed'] or any(p.exitcode != 0 for p in processes) else 'complete') if report['attempted'] == report['planned'] else 'interrupted',
            exit_codes={str(i): p.exitcode for i, p in enumerate(processes)}, watchdog_killed=watchdog,
            trial_files_sha256=digest(hashes), no_automatic_seed_retries=True)
        atomic(args.output/'progress.json', report); atomic(args.output/'run_end.json', report)
        if report['attempted'] == report['planned']:
            immutable(args.output/'completion.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'solvers'}), flush=True)
        return 0 if report['status'] == 'complete' else 1
    finally:
        stop.set(); start.set()
        grace = time.monotonic()+plan['watchdog_seconds']
        for process in processes:
            if process.is_alive():
                process.join(timeout=max(0., grace-time.monotonic()))
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
        if lock.exists() and read(lock).get('pid') == os.getpid():
            lock.unlink()
        close_queues(tasks, events)


def close_queues(*queues):
    # Pending jobs may remain when a worker fails. Do not let the parent's
    # feeder wait forever for an abandoned pipe during interpreter shutdown.
    for channel in queues:
        channel.cancel_join_thread()
        channel.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'status', 'validate'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve(); args.plan = args.plan.resolve(); args.output = args.output.resolve()
    if args.input:
        args.input = args.input.resolve()
    plan, settings, base = preflight(args)
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN_VERIFIED', host=args.host, solvers=len(base['solvers']),
            planned_trials=len(base['solvers'])*100, plan_sha256=base['plan_sha256'], gpu_execution=False), indent=2))
        return 0
    if args.command == 'run':
        return run(args, plan, base)
    require((args.output/'identity.json').exists(), 'Campaign has no saved identity yet.')
    identity = read(args.output/'identity.json')
    require({k: v for k, v in identity.items() if k not in {'hardware', 'digest'}} == base
            and digest({k: v for k, v in identity.items() if k != 'digest'}) == identity['digest'], 'Saved campaign identity differs.')
    rows, hashes = scan(args.output, identity); report = summary(identity, rows, duration_hints=plan.get('expected_caller_seconds'))
    finished = completed_result(args.output, identity, rows, hashes)
    report['status'] = ('FAIL' if report['failed'] or (finished and finished['status'] != 'complete') else
                        'PASS' if finished and report['attempted'] == report['planned'] else 'PENDING')
    if args.command == 'validate':
        report['trial_files_sha256'] = digest(hashes); report['full_reread'] = True
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'PASS' else 1 if report['status'] == 'FAIL' else 2
    print(json.dumps({k: v for k, v in report.items() if k != 'solvers'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Existing benchmark's compact continuous mode: one problem, one 20s run/seed."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
from importlib.util import find_spec
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import sys
import time

import numpy as np

from .common import ROOT, CONFIG, DirectoryLock, atomic, digest, filehash, read, source_snapshot_hash
from .compact_metrics import CHECKPOINTS
from .compact_storage import CompactStore
from .continuous_worker import ObjectiveScorer, PROTECTED, run_continuous_trial, _scoring_operator_required


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=
        'Old positional N DENSITY BUDGET commands remain available as the historical vector-output protocol.')
    p.add_argument('--continuous', action='store_true', help='Compact continuous protocol (also selected by named size flags)')
    p.add_argument('--variable-count', '--n', type=int, choices=(100,200,500,1000,5000,10000))
    density = p.add_mutually_exclusive_group()
    density.add_argument('--density', choices=('sparse','dense'))
    density.add_argument('--sparse', dest='density', action='store_const', const='sparse')
    density.add_argument('--dense', dest='density', action='store_const', const='dense')
    p.add_argument('--config', help='Frozen size-aware 20s preset JSON')
    p.add_argument('--catalog', help='Verified twelve-slot public input catalog')
    p.add_argument('--solvers', help='Comma-separated canonical IDs; default all23')
    p.add_argument('--runs', type=int, help='Seeds per solver, default100; smaller values are bounded smoke tests')
    p.add_argument('--seed-start', type=int, help='First deterministic seed, default0')
    p.add_argument('--devices', '--device', default=None, help='auto, cpu, cuda:N, or comma-separated CUDA devices')
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--seed-workers', '--gpu-workers', '--workers-per-gpu', default=None,
                   help='Workers per GPU: 1 (default), auto (quality-aware calibration), or explicit N')
    p.add_argument('--no-autotune', action='store_true', default=None,
                   help='With auto workers, use one per GPU without calibration')
    p.add_argument('--max-auto-workers', type=int, default=None, help='Calibration ceiling, default4')
    p.add_argument('--calibration-seeds', type=int, default=None, help='Paired screening seeds, default4')
    p.add_argument('--calibration-holdout-seeds', type=int, default=None, help='Fresh paired holdout seeds, default4')
    p.add_argument('--calibration-only', action='store_true', help='Measure/freeze worker policy, do not start production seeds')
    p.add_argument('--calibration-cache', default=None, help='Verified machine/input/solver-specific calibration cache')
    p.add_argument('--recalibrate', action='store_true', help='Ignore cached evidence for a new run; never on resume')
    p.add_argument('--stop-on-optimum', action=argparse.BooleanOptionalAction, default=None,
                   help='Stop after a verified OPTIMUM hit and save its time (default); BKS always continues')
    p.add_argument('--output-root', '--output', default=None, help='One compact result directory')
    p.add_argument('--resume', help='Resume this exact compact directory; interrupted seeds restart from their original seed')
    p.add_argument('--keep-raw-results', action=argparse.BooleanOptionalAction, default=None,
                   help='Keep checkpoint energies for post-run1%% hits (required in this protocol)')
    p.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True,
                   help='Generate only Gap, Hit probability and TTS99 PNGs after completion')
    p.add_argument('--dry-run', action='store_true', help='Verify inputs/settings/admission without starting any solver')
    p.add_argument('--list-instances', action='store_true')
    p.add_argument('--list-solvers', action='store_true')
    p.add_argument('--doctor', action='store_true', help='Read-only offline data/configuration readiness')
    p.add_argument('--json', action='store_true', help='Stable JSON summary on stdout; progress/errors on stderr')
    return p


def _parameters(config, n):
    profiles = config.get('size_configurations', {})
    profile = profiles.get(str(n))
    settings = profile.get('solvers', profile) if profile is not None else config['solvers']
    from .selection import resolve_legacy_exchange_defaults
    settings = resolve_legacy_exchange_defaults(settings)
    baseline = read(CONFIG/'benchmark_solvers.json')['solvers']
    if profile is None and n>=5000:
        for solver,parameters in settings.items():
            if solver not in PROTECTED and 'max_dense_variables' in parameters:
                parameters['max_dense_variables']=16384
            if solver in PROTECTED and config.get('protected_memory_guard_override') and 'memory_limit_bytes' in parameters:
                parameters['memory_limit_bytes']=8*1024**3
    for solver in PROTECTED:
        expected = dict(baseline[solver])
        if n>=5000 and config.get('protected_memory_guard_override') and 'memory_limit_bytes' in expected:
            expected['memory_limit_bytes'] = 8*1024**3
        if digest(settings[solver])!=digest(expected):
            raise ValueError(f'Protected numerical parameters changed: {solver}')
    return settings


class InputIdentityError(ValueError):
    """Immutable input changed; the corresponding seed must remain pending."""
    fatal_calibration = True


class SourceIdentityError(RuntimeError):
    """Source changes abort calibration and production, never become pilot misses."""
    fatal_calibration = True


def _verify_job_input(job):
    try:
        actual = filehash(job['npz'])
    except OSError as exc:
        raise InputIdentityError('Immutable continuous input is missing/unreadable') from exc
    if actual != job['input_sha256']:
        raise InputIdentityError('Immutable continuous input differs from manifest checksum')


def _prepare_job(worker, job):
    """Verify cache misses and prepare all immutable data before solve timing."""
    from qubo_solvers import LIBRARY_SOLVERS
    key = (job['npz'], job['parameters']['dtype'], job['solver'] in LIBRARY_SOLVERS)
    cache_miss = key != worker.cache_key
    if cache_miss:
        _verify_job_input(job)
        # Release the previous large independent scoring operator before a new
        # source/native input is allocated. Only immutable caches cross seeds.
        worker._continuous_scorer = None
        worker._continuous_score_key = None
    elif getattr(worker, '_continuous_input_sha256', None) != job['input_sha256']:
        raise InputIdentityError('Cached immutable input differs from manifest checksum')
    try:
        worker.prepare(job)
    finally:
        if cache_miss:
            _verify_job_input(job)
    worker._continuous_input_sha256 = job['input_sha256']
    # Infrastructure only: cuBLAS retains a workspace for each (handle, stream).
    # Reuse one worker-owned capture stream across fresh Exchange solves instead
    # of warming new BLAS workspaces with every graph/seed. Graphs and all search
    # state remain solve-local. Prepare it outside the measured solve budget.
    if (worker.device.startswith('cuda') and job['solver']=='lib_exchange_cascade'
            and job['parameters'].get('graph_block',0)
            and getattr(worker,'_continuous_graph_stream',None) is None):
        worker._continuous_graph_stream = worker.torch.cuda.Stream(device=worker.device)
    needs_operator = _scoring_operator_required(worker, job)
    score_key = (worker.cache_key, job['parameters']['dtype'], needs_operator)
    if getattr(worker, '_continuous_score_key', None) != score_key:
        worker._continuous_scorer = ObjectiveScorer(worker.source, worker.device,
            native=needs_operator, precision=job['parameters']['dtype'])
        worker._continuous_score_key = score_key
    from .worker import synchronize
    synchronize(worker.torch, worker.device)


def _mark_prepared(active, message, now):
    index = message['worker']
    job = active.get(index)
    if job is None or (message['solver'], message['seed']) != (job['case'][0], job['case'][2]):
        raise RuntimeError('Worker prepared an unexpected continuous seed')
    job['phase'], job['started'] = 'solve', now


def _check_watchdogs(active, lanes, now):
    for index, job in active.items():
        if not lanes[index][0].is_alive():
            raise RuntimeError('Worker lost; active seeds remain pending for safe resume')
        phase = job['phase']
        limit = 25. if phase == 'solve' else 120.
        if now - job['started'] > limit:
            raise RuntimeError(f'Continuous {phase} watchdog; active seeds remain pending')


def _stop_process(process):
    """Stop only a process created by this controller, escalating if needed."""
    process.join(timeout=2.)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.)
    if process.is_alive():
        raise RuntimeError(f'Owned continuous worker PID {process.pid} survived shutdown; '
                           'stop it before resuming this result directory')


def _worker(index, device, incoming, outgoing):
    from .worker import WarmWorker
    import gc
    try:
        worker = WarmWorker(dict(device=device,cpu_threads=1))
        outgoing.put(dict(kind='ready',worker=index))
        for job in iter(incoming.get, None):
            outgoing.put(dict(kind='started',worker=index,seed=job['seed'],solver=job['solver']))
            try:
                _prepare_job(worker, job)
                outgoing.put(dict(kind='prepared',worker=index,seed=job['seed'],solver=job['solver']))
                result = run_continuous_trial(worker, job)
            except InputIdentityError:
                raise
            except Exception as exc:
                result = dict(energies=[None]*9,status='error',error=f'{type(exc).__name__}: {exc}')
            outgoing.put(dict(kind='done',worker=index,seed=job['seed'],solver=job['solver'],result=result))
            gc.collect()
    except BaseException as exc:
        outgoing.put(dict(kind='fatal',worker=index,error=f'{type(exc).__name__}: {exc}'))


def _emit(value, json_mode):
    print(json.dumps(value,sort_keys=True,allow_nan=False) if json_mode else str(value),flush=True)


def prepare(args):
    if args.plots and find_spec('matplotlib') is None:
        raise ValueError('Plot generation requires matplotlib; install benchmark dependencies or use --no-plots')
    from ..continuous_catalog import load_continuous_catalog, select_continuous_problem, verify_continuous_problem
    from .selection import environment, inventory
    if args.resume:
        saved = read(Path(args.resume)/'manifest.json')
        invocation = saved['invocation']
        for key in ('variable_count','density','config','catalog','solvers','runs','seed_start','devices',
                    'seed_workers','max_auto_workers','calibration_seeds','calibration_holdout_seeds',
                    'no_autotune','stop_on_optimum'):
            current = getattr(args,key)
            if current is None:
                setattr(args,key,invocation.get(key))
        if args.keep_raw_results is None:
            args.keep_raw_results = saved['keep_raw_results']
        args.output_root = args.resume
    if args.variable_count is None or args.density is None:
        raise ValueError('Provide --variable-count and --sparse/--dense or --density')
    args.runs = 100 if args.runs is None else args.runs
    args.seed_start = 0 if args.seed_start is None else args.seed_start
    if not 1<=args.runs<=100 or args.seed_start<0 or args.seed_start+args.runs>=2**63:
        raise ValueError('Expected1..100 seeds and a nonnegative63-bit seed range')
    args.devices = args.devices or 'auto'
    args.config = str(Path(args.config or CONFIG/'benchmark_solvers_20s.json').resolve())
    args.catalog = str(Path(args.catalog or CONFIG/'continuous_catalog.json').resolve())
    args.keep_raw_results = True if args.keep_raw_results is None else args.keep_raw_results
    if not args.keep_raw_results:
        raise ValueError('Retain checkpoint energies for post-run1% hits: omit --no-keep-raw-results. '
                         'Earlier aggregate-only results remain intact.')
    from .selection import worker_argument
    from .continuous_calibration import calibration_options
    args.seed_workers = worker_argument(1 if args.seed_workers is None else args.seed_workers)
    args.no_autotune = bool(args.no_autotune)
    args.stop_on_optimum = True if args.stop_on_optimum is None else args.stop_on_optimum
    args.max_auto_workers = 4 if args.max_auto_workers is None else args.max_auto_workers
    args.calibration_seeds = 4 if args.calibration_seeds is None else args.calibration_seeds
    args.calibration_holdout_seeds = 4 if args.calibration_holdout_seeds is None else args.calibration_holdout_seeds
    calibration_options(args)
    if args.resume and (args.recalibrate or args.calibration_only):
        raise ValueError('Resume restores the frozen worker policy; recalibration requires a new result directory')
    catalog = load_continuous_catalog(args.catalog)
    entry, npz = select_continuous_problem(args.variable_count,args.density,catalog)
    verification = verify_continuous_problem(entry,npz,independent=True)
    if verification.get('status')!='passed':
        raise ValueError(f'Independent public input validation failed: {verification}')
    config = read(args.config)
    settings = _parameters(config,args.variable_count)
    selected = list(settings) if args.solvers in (None,'all') else args.solvers.split(',')
    if not selected or len(set(selected))!=len(selected) or any(s not in settings for s in selected):
        raise ValueError('Select unique supported canonical solver IDs')
    args.solvers = ','.join(selected)
    env = environment(args.devices,args.require_gpu)
    devices = env['hardware']['devices']
    registry = {r['solver_id']:r for r in inventory(settings,devices[0])}
    unavailable = [s for s in selected if registry[s]['status']!='runnable']
    if unavailable:
        raise ValueError(f'Unavailable solver configurations: {unavailable}')
    reference_type = ('OPTIMUM' if entry['reference_status'] in
                      ('published_proven_optimum','independently_verified_optimum') else 'BKS')
    problem = dict(id=entry['instance_id'],n=entry['binary_variables'],density=entry['density_category'],
        reference=entry['normalized_reference_objective_min'],reference_type=reference_type,
        source_url=entry['reference_sources'][0],integer_objective=entry.get('integer_objective',True))
    # Only fixed configuration/environment identity is recorded, never GPU
    # activity, memory peaks, timing curves, vectors or source matrices.
    execution = dict(devices=devices,cpu_threads=1,tf32=False,seed_workers_requested=args.seed_workers,
        budget_seconds=20.,wall_schedule_solvers=[s for s in selected if s not in PROTECTED],
        checkpoint_policy='last completed independently scored incumbent at or before each boundary',
        natural_return_policy='retain same result to20s unless verified optimum; do not restart or retune protected solvers',
        stop_on_optimum=args.stop_on_optimum,bks_policy='continue after matching or improving BKS',
        optimum_time_policy='first observed completed independently scored optimum, seconds from solve origin',
        scheduling_policy='homogeneous solver groups; all immutable inputs ready before first dispatch',
        calibration_policy='paired full20s screening and fresh holdout; no observed quality degradation and >=10% throughput gain')
    invocation = {key:getattr(args,key) for key in ('variable_count','density','config','catalog',
                    'solvers','runs','seed_start','devices','seed_workers','max_auto_workers',
                    'calibration_seeds','calibration_holdout_seeds','no_autotune','stop_on_optimum')}
    manifest = dict(solvers={s:settings[s] for s in selected},problems=[problem],
        source=dict(snapshot=source_snapshot_hash()),configuration=filehash(args.config),
        inputs={entry['instance_id']:filehash(npz)},catalog=filehash(args.catalog),
        environment=dict(os=env['os'],python=env['python'],packages=env['packages'],
                         torch_cuda=env['torch_cuda'],driver=env['driver'],hardware_id=env['hardware_id']),
        execution=execution,invocation=invocation,
        metrics=['Gap to solution','Hit probability','TTS99'],
        tts99_convention='literal continuous formula: p0=Inf,p1=0')
    prepared = dict(manifest=manifest,entry=entry,npz=str(npz),settings=settings,environment=env,
                    solvers=selected,devices=devices,seeds=list(range(args.seed_start,args.seed_start+args.runs)))
    if args.resume:
        prepared['saved_worker_plan'] = saved['execution'].get('worker_plan')
        if prepared['saved_worker_plan'] is None:
            raise ValueError('This earlier protocol has no optimum timing/worker policy; start a new result directory')
    return prepared


def _job(prepared,args,solver):
    problem = prepared['manifest']['problems'][0]
    return dict(cpu_threads=1,solver=solver,parameters=prepared['settings'][solver],
        npz=prepared['npz'],input_sha256=prepared['manifest']['inputs'][problem['id']],
        budget=20.,checkpoints=CHECKPOINTS,wall_schedule=solver not in PROTECTED,
        reference=problem['reference'],reference_type=problem['reference_type'],
        tolerance=0. if problem.get('integer_objective',True) else 1e-9,
        stop_on_optimum=args.stop_on_optimum)


def _capacity_snapshot(prepared):
    from .selection import cpu_capacity,host_memory_available
    prepared['host_available_bytes'] = host_memory_available()
    prepared['cpu_allowed'] = cpu_capacity()['effective_cpus']
    available = {}
    if all(device=='cpu' for device in prepared['devices']):
        prepared['gpu_available_bytes'] = available
        return
    result = subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if result.returncode:
        raise ValueError('Cannot verify current GPU free memory for worker admission')
    readings = {}
    for line in result.stdout.splitlines():
        uuid,free = line.split(',',1)
        readings[uuid.strip().lower().removeprefix('gpu-')] = int(free.strip())*1024**2
    for gpu in prepared['environment']['hardware']['gpus']:
        uuid = gpu['uuid'].lower().removeprefix('gpu-')
        if uuid not in readings:
            raise ValueError(f"Cannot match GPU memory reading to device {gpu['device']}")
        available[gpu['device']] = readings[uuid]
    prepared['gpu_available_bytes'] = available


def _worker_plan(prepared,args):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from .continuous_calibration import admission_capacity,calibration_cache_key,calibration_options,calibrate_solver
    from .continuous_pool import run_group
    devices = prepared['devices']
    _capacity_snapshot(prepared)
    options = calibration_options(args)
    cache = Path(args.calibration_cache or ROOT/'results'/'continuous_calibration').resolve()
    saved = prepared.get('saved_worker_plan')
    if saved is not None and args.recalibrate:
        raise ValueError('Cannot recalibrate a resumed experiment')
    plan, evidence = {}, {}
    stop_event = mp.get_context('spawn').Event()
    for solver in prepared['solvers']:
        capacity = admission_capacity(prepared,solver,devices)
        limits = capacity['workers_per_device']
        if any(limit<1 for limit in limits.values()):
            raise ValueError(f'One unchanged {solver} worker is not memory/CPU admitted: {limits}')
        if saved is not None:
            counts = saved[solver]
            if set(counts)!=set(devices) or any(not isinstance(c,int) or isinstance(c,bool) or
                not 1<=c<=limits[d] for d,c in counts.items()):
                raise ValueError(f'Frozen worker count is not currently admitted: {solver}')
            plan[solver] = counts
            continue
        automatic = args.seed_workers=='auto' and not args.no_autotune and devices!=['cpu']
        if not automatic:
            count = 1 if args.seed_workers=='auto' else args.seed_workers
            if devices==['cpu'] and count!=1:
                raise ValueError('CPU compact mode uses one worker; GPU sharing requires CUDA')
            counts = {d:min(count,len(prepared['seeds'])) for d in devices}
            if any(c>limits[d] for d,c in counts.items()):
                raise ValueError(f'Requested {count} workers/GPU exceeds unchanged {solver} admission {limits}')
            plan[solver] = counts
            continue

        def measure(device):
            key = calibration_cache_key(prepared,solver,[device],options)
            path = cache/(key+'.json')
            if path.exists() and not args.recalibrate:
                wrapper = read(path)
                report = wrapper['payload']
                if wrapper['sha256']!=digest(report) or report.get('cache_key')!=key:
                    raise ValueError(f'Corrupt or incompatible worker calibration cache: {path}')
                counts = report['selected']
                if set(counts)=={device} and isinstance(counts[device],int) and not isinstance(counts[device],bool) and 1<=counts[device]<=limits[device]:
                    print(f'Calibration cache: {solver} {device}, workers={counts[device]}',file=sys.stderr,flush=True)
                    report_sha = digest(report)
                    immutable = cache/(key+'_'+report_sha+'.json')
                    if not immutable.exists():
                        atomic(immutable,wrapper)
                    return device,counts[device],report,dict(cache_key=key,report_sha256=report_sha,
                                                           report_path=str(immutable))
            view = dict(prepared,capacity=capacity)
            def wave(solver_id,wave_device,count,seeds,calibration=True):
                if source_snapshot_hash()!=prepared['manifest']['source']['snapshot']:
                    raise SourceIdentityError('Source changed during worker calibration')
                print(f'Calibration: {solver_id} {wave_device}, workers={count}, paired seeds={len(seeds)}',
                      file=sys.stderr,flush=True)
                def check_source(result):
                    if source_snapshot_hash()!=prepared['manifest']['source']['snapshot']:
                        stop_event.set()
                        raise SourceIdentityError('Source changed during worker calibration')
                return run_group(_job(prepared,args,solver_id),[wave_device],
                                 {wave_device:count},seeds,calibration=calibration,
                                 stop_event=stop_event,on_result=check_source)
            counts,report = calibrate_solver(view,args,solver,[device],wave)
            report_sha = digest(report)
            wrapper = dict(sha256=report_sha,payload=report)
            immutable = cache/(key+'_'+report_sha+'.json')
            if immutable.exists() and read(immutable)!=wrapper:
                raise ValueError('Conflicting immutable calibration evidence')
            if not immutable.exists():
                atomic(immutable,wrapper)
            atomic(path,wrapper)
            return device,counts[device],report,dict(cache_key=key,report_sha256=report_sha,
                                                   report_path=str(immutable))

        pool = ThreadPoolExecutor(max_workers=len(devices))
        try:
            futures = [pool.submit(measure,device) for device in devices]
            reports = [future.result() for future in as_completed(futures)]
        except BaseException:
            stop_event.set()
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        plan[solver] = {d:min(count,len(prepared['seeds'])) for d,count,_,_ in reports}
        evidence[solver] = {d:key for d,_,_,key in reports}
    prepared['manifest']['execution']['worker_plan'] = plan
    prepared['manifest']['execution']['execution_mode'] = (
        'parallel_shared_device' if any(c>1 for counts in plan.values() for c in counts.values())
        else 'multi_gpu_isolated' if len(devices)>1 else 'isolated_latency')
    if saved is not None:
        evidence = read(Path(args.resume)/'manifest.json')['execution'].get('calibration_evidence',{})
    prepared['manifest']['execution']['calibration_evidence'] = evidence
    return plan


def execute(prepared,args):
    from .continuous_pool import run_group
    root = Path(args.output_root or ROOT/'results'/'continuous'/
                f'n{args.variable_count}_{args.density}').resolve()
    root.mkdir(parents=True,exist_ok=True)
    with DirectoryLock(root):
        if (root/'manifest.json').exists() and not args.resume:
            raise ValueError('This result directory already exists; use --resume or choose a new --output')
        if not args.resume and not args.calibration_only and any(p.name!='.lock' for p in root.iterdir()):
            raise ValueError('Choose an empty output directory for a new production run')
        plan = _worker_plan(prepared,args)
        if args.calibration_only:
            atomic(root/'calibration_manifest.json',prepared['manifest'])
            return root,0
        store = CompactStore(root,prepared['manifest'],prepared['seeds'],CHECKPOINTS,True)
        print(f'Progress: {100*store.completed_count/store.planned_count:.2f}% '
              f'({store.completed_count}/{store.planned_count})',file=sys.stderr,flush=True)
        for solver in prepared['solvers']:
            pending = [seed for _,_,seed in store.pending(solver)]
            if not pending:
                continue
            if source_snapshot_hash()!=prepared['manifest']['source']['snapshot']:
                raise RuntimeError('Source changed during continuous campaign')
            problem = prepared['manifest']['problems'][0]['id']
            def commit(result):
                if source_snapshot_hash()!=prepared['manifest']['source']['snapshot']:
                    raise SourceIdentityError('Source changed during continuous campaign; active seeds remain pending')
                status = result['status']
                if status=='interrupted':
                    raise KeyboardInterrupt
                values = [np.nan if x is None else x for x in result['energies']]
                store.commit_seed(solver,problem,result['seed'],values,status=status,
                                  time_to_optimum_s=result.get('time_to_optimum_s'))
                detail = f": {result['error']}" if result.get('error') else ''
                timing = result.get('time_to_optimum_s')
                hit = f', optimum at {timing:.6f}s' if timing is not None else ''
                print(f'Progress: {100*store.completed_count/store.planned_count:.2f}% '
                      f'({store.completed_count}/{store.planned_count}) {solver}, seed={result["seed"]}, '
                      f'{result["device"]}, workers={plan[solver][result["device"]]}, {status}{hit}{detail}',
                      file=sys.stderr,flush=True)
            run_group(_job(prepared,args,solver),prepared['devices'],plan[solver],pending,on_result=commit)
        store.finalize()
        if args.plots:
            from .compact_plots import plot_results
            plot_results(root)
    return root,int(store.has_failures)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.doctor or args.list_instances:
            from ..continuous_catalog import load_continuous_catalog
            catalog = load_continuous_catalog(args.catalog)
            rows = [dict(id=e['instance_id'],n=e['binary_variables'],density=e['density_category'],
                         status=e['preparation_status']) for e in catalog['instances']]
            _emit(dict(auth_required=False,offline=True,instances=rows),args.json)
            return 0
        if args.list_solvers:
            config = read(args.config or CONFIG/'benchmark_solvers_20s.json')
            _emit(dict(solvers=list(config['solvers']),protected=list(sorted(PROTECTED))),args.json)
            return 0
        prepared = prepare(args)
        if args.dry_run:
            from .continuous_calibration import admission_capacity
            _capacity_snapshot(prepared)
            admission = {solver:admission_capacity(prepared,solver,prepared['devices'])['workers_per_device']
                         for solver in prepared['solvers']}
            if args.seed_workers!='auto' and any(min(args.seed_workers,len(prepared['seeds']))>limit
                    for counts in admission.values() for limit in counts.values()):
                raise ValueError('Requested workers exceed memory/CPU admission; use fewer workers or auto')
            _emit(dict(problem=prepared['entry']['instance_id'],solvers=prepared['solvers'],
                seeds=prepared['seeds'],devices=prepared['devices'],checkpoints=CHECKPOINTS,
                workers_requested=args.seed_workers,admission_maximum_workers_per_gpu=admission,
                calibration_executed=False,stop_on_optimum=args.stop_on_optimum,
                expected_raw_payload_bytes=8*9*len(prepared['seeds'])*len(prepared['solvers'])),args.json)
            return 0
        root,code = execute(prepared,args)
        if args.calibration_only:
            _emit(dict(output=str(root),status='calibrated',production_started=False,
                       worker_plan=prepared['manifest']['execution']['worker_plan']),args.json)
            return code
        _emit(dict(output=str(root),complete=True,status='failed_attempts' if code else 'complete',
                   metrics=['Gap to solution','Hit probability','TTS99']),args.json)
        return code
    except KeyboardInterrupt:
        print('Interrupted; active seeds remain pending and restart from their original seeds on resume.',file=sys.stderr)
        return 130
    except (ValueError,OSError,RuntimeError,ImportError) as exc:
        if args.json:
            _emit(dict(error=type(exc).__name__,message=str(exc)),True)
        else:
            print(f'Continuous benchmark error: {exc}',file=sys.stderr)
        return 2

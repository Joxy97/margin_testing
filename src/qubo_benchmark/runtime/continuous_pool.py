"""Bounded spawned workers for one compact solver/input group.

Only immutable input and infrastructure survive between seeds. All workers
prepare their input and independent scorer before the measured group begins.
Imports in the controller stay lightweight: Torch is imported only in children.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time


SETUP_TIMEOUT_S = 120.
SOLVE_TIMEOUT_S = 25.
POLL_INTERVAL_S = .05
_FAILURES = frozenset(('error', 'oom', 'invalid_output', 'validation_error',
                       'no_in_budget_candidate', 'failed', 'invalid'))


class FatalWorkerError(RuntimeError):
    """Unsafe execution or immutable identity failure; never reject and retry."""
    fatal_calibration = True


class UnsafeWorkerShutdown(FatalWorkerError):
    """An owned child survived bounded termination and must block new pools."""


class _StopSignal:
    def __init__(self, owned, shared=None):
        self.owned, self.shared = owned, shared

    def is_set(self):
        return self.owned.is_set() or (self.shared is not None and self.shared.is_set())


def _raise_if_stopped(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt


def _verify_input(job, stop_event=None):
    """Check the immutable identity without importing a numerical package."""
    checksum = hashlib.sha256()
    try:
        with Path(job['npz']).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                _raise_if_stopped(stop_event)
                checksum.update(chunk)
    except OSError as exc:
        raise FatalWorkerError('Immutable continuous input is missing/unreadable; seeds remain pending') from exc
    if checksum.hexdigest() != job['input_sha256']:
        raise FatalWorkerError('Immutable continuous input differs from manifest checksum; seeds remain pending')


def _compact_result(result, seed, device, checkpoint_count):
    """Whitelist scalar output; never allow solver state into the result queue."""
    status = result['status']
    if status == 'interrupted':
        return dict(status=status, energies=[None]*checkpoint_count, seed=seed, device=device)
    if status != 'complete' and status not in _FAILURES:
        raise ValueError(f'Unexpected continuous terminal status: {status}')
    energies = result['energies']
    if len(energies) != checkpoint_count:
        raise ValueError('Continuous result has an unexpected checkpoint count')
    values = []
    for value in energies:
        if value is None:
            values.append(None)
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('Continuous result contains a nonfinite energy')
            values.append(value)
    if status != 'complete':
        values = [None]*checkpoint_count
    row = dict(energies=values, status=status, seed=seed, device=device)
    for name in ('stop_reason', 'error', 'actual_solve_wall_s', 'time_to_optimum_s'):
        if name in result:
            row[name] = result[name]
    return row


def _host_rss_peak_bytes():
    """Process high-water RSS, including immutable preparation and CUDA context."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == 'darwin' else peak*1024)
    except (ImportError, AttributeError, OSError):
        try:
            import psutil
            memory = psutil.Process().memory_info()
            return int(getattr(memory, 'peak_wset', getattr(memory, 'peak_rss', memory.rss)))
        except (ImportError, OSError):
            return None


def _reset_calibration_memory(worker):
    if worker.device.startswith('cuda'):
        worker.torch.cuda.reset_peak_memory_stats(worker.device)


def _calibration_memory(worker):
    memory = dict(host_rss_peak_bytes=_host_rss_peak_bytes(),
                  gpu_allocated_peak_bytes=None, gpu_reserved_peak_bytes=None)
    if worker.device.startswith('cuda'):
        memory['gpu_allocated_peak_bytes'] = int(worker.torch.cuda.max_memory_allocated(worker.device))
        memory['gpu_reserved_peak_bytes'] = int(worker.torch.cuda.max_memory_reserved(worker.device))
    return memory


def _serve(index, device, template, incoming, outgoing, stop, calibration, stop_event=None):
    """One device context, one immutable input, and fresh mutable state per seed."""
    try:
        from .worker import WarmWorker
        from .continuous_cli import _prepare_job
        from .continuous_worker import run_continuous_trial

        stop = _StopSignal(stop, stop_event)
        job = dict(template, device=device, cpu_threads=1)
        worker = WarmWorker(job)
        _prepare_job(worker, job)
        outgoing.put(dict(kind='ready', worker=index))
        while not stop.is_set():
            try:
                command = incoming.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                continue
            if command is None:
                return
            seed = command['seed']
            job = dict(template, device=device, cpu_threads=1, seed=seed)
            if calibration:
                _reset_calibration_memory(worker)
            outgoing.put(dict(kind='started', worker=index, seed=seed,
                              solver=job['solver'], started=time.monotonic()))
            result = run_continuous_trial(worker, job, stop)
            row = _compact_result(result, seed, device, len(job['checkpoints']))
            if calibration:
                row.update(_calibration_memory(worker))
            outgoing.put(dict(kind='done', worker=index, seed=seed,
                              solver=job['solver'], result=row))
    except Exception as exc:
        outgoing.put(dict(kind='fatal', worker=index, error=f'{type(exc).__name__}: {exc}',
                          fatal_calibration=bool(getattr(exc, 'fatal_calibration', False)
                                                or type(exc).__name__ == 'InputIdentityError')))


def _stop_process(process):
    """Reap only this pool's child, with a bounded escalation at each stage."""
    process.join(timeout=2.)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.)
    if process.is_alive():
        raise UnsafeWorkerShutdown(f'Owned continuous worker PID {process.pid} survived shutdown; '
                                   'stop it before resuming this result directory')


def _close_queue(channel):
    # A killed consumer must not leave the parent's feeder join blocked.
    channel.cancel_join_thread()
    channel.close()


def _check_workers(lanes, active, now):
    for index, (process, _, _) in enumerate(lanes):
        if not process.is_alive():
            raise FatalWorkerError('Continuous worker lost; active seeds remain pending for safe resume')
        job = active.get(index)
        if job is None:
            continue
        limit = SOLVE_TIMEOUT_S if job['phase'] == 'solve' else SETUP_TIMEOUT_S
        if now-job['started'] > limit:
            raise RuntimeError(f"Continuous {job['phase']} watchdog; active seeds remain pending")


def _device_lanes(devices, workers_per_device):
    devices = list(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Select at least one unique continuous device')
    if set(workers_per_device) != set(devices):
        raise ValueError('Worker counts must cover exactly the selected devices')
    lanes = []
    for device in devices:
        count = workers_per_device[device]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100:
            raise ValueError('Expected 1 through 100 workers per continuous device')
        lanes.extend([device]*count)
    return lanes


def run_group(template_job, devices, workers_per_device, seeds, on_result=None, calibration=False,
              stop_event=None):
    """Run every planned seed once and stream terminal rows to ``on_result``.

    Returns ``(rows, measured_wave_seconds)``. Rows contain only checkpoint
    energies and scalar operational metadata. Setup, pool shutdown, and input
    identity checks are outside wave timing; scheduling, drain, and callbacks
    are included. Interrupted/lost-worker attempts raise and remain pending.
    A fresh group owns a fresh pool so dormant solver caches cannot fill a GPU.
    """
    assignments = _device_lanes(devices, workers_per_device)
    seeds = list(seeds)
    if len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int)
                                         or s < 0 or s >= 2**63 for s in seeds):
        raise ValueError('Expected unique nonnegative 63-bit continuous seeds')
    if not seeds:
        return [], 0.
    _raise_if_stopped(stop_event)
    if template_job.get('cpu_threads', 1) != 1:
        raise ValueError('Continuous workers require one CPU thread per process')
    template = dict(template_job, cpu_threads=1, seed=seeds[0])
    _verify_input(template, stop_event)
    context = mp.get_context('spawn')
    outgoing = context.Queue(maxsize=max(4, 2*len(assignments)))
    stop = context.Event()
    lanes, active, rows = [], {}, []
    pending = deque(seeds)
    try:
        began = time.monotonic()
        for index, device in enumerate(assignments):
            _raise_if_stopped(stop_event)
            incoming = context.Queue(maxsize=1)
            arguments = (index, device, template, incoming, outgoing, stop, bool(calibration))
            if stop_event is not None:
                arguments += (stop_event,)
            process = context.Process(target=_serve, args=arguments)
            lanes.append((process, incoming, device))
            process.start()
        ready = set()
        while len(ready) < len(lanes):
            _raise_if_stopped(stop_event)
            try:
                message = outgoing.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                _raise_if_stopped(stop_event)
                _check_workers(lanes, {}, time.monotonic())
                if time.monotonic()-began > SETUP_TIMEOUT_S:
                    raise RuntimeError('Continuous immutable setup watchdog; no seeds dispatched')
                continue
            if message['kind'] == 'fatal':
                failure = FatalWorkerError if message.get('fatal_calibration') else RuntimeError
                raise failure(message['error'])
            index = message['worker']
            if message['kind'] != 'ready' or index not in range(len(lanes)) or index in ready:
                raise RuntimeError('Unexpected continuous worker readiness response')
            ready.add(index)
        _raise_if_stopped(stop_event)
        _check_workers(lanes, {}, time.monotonic())
        _verify_input(template, stop_event)
        wave_started = time.monotonic()

        def dispatch(index):
            _raise_if_stopped(stop_event)
            if pending:
                seed = pending.popleft()
                active[index] = dict(seed=seed, phase='dispatch', started=time.monotonic())
                lanes[index][1].put(dict(seed=seed))

        for index in range(len(lanes)):
            dispatch(index)
        while active or pending:
            _raise_if_stopped(stop_event)
            try:
                message = outgoing.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                message = None
            available = []
            # Fetch after each callback so any completions queued during slow
            # persistence are drained before watchdog checks or redispatch.
            while message is not None:
                _raise_if_stopped(stop_event)
                kind, index = message['kind'], message['worker']
                if kind == 'fatal':
                    failure = FatalWorkerError if message.get('fatal_calibration') else RuntimeError
                    raise failure(message['error'])
                trial = active.get(index)
                if (trial is None or message.get('seed') != trial['seed']
                        or message.get('solver') != template['solver']):
                    raise RuntimeError('Worker returned an unexpected continuous seed')
                if kind == 'started':
                    if trial['phase'] != 'dispatch':
                        raise RuntimeError('Worker started the same continuous seed twice')
                    trial.update(phase='solve', started=message['started'])
                elif kind == 'done':
                    row = message['result']
                    if row['seed'] != trial['seed'] or row['device'] != lanes[index][2]:
                        raise RuntimeError('Continuous result seed/device differs from its assignment')
                    if row['status'] == 'interrupted':
                        raise KeyboardInterrupt
                    if on_result is not None:
                        on_result(row)
                    rows.append(row)
                    del active[index]
                    available.append(index)
                else:
                    raise RuntimeError(f'Unexpected continuous worker response: {kind}')
                try:
                    message = outgoing.get_nowait()
                except queue.Empty:
                    message = None
            if active or pending:
                _raise_if_stopped(stop_event)
                _check_workers(lanes, active, time.monotonic())
            for index in available:
                dispatch(index)
        elapsed = time.monotonic()-wave_started
        if pending or len(rows) != len(seeds):
            raise RuntimeError('Continuous group did not return every planned seed')
        order = {seed: index for index, seed in enumerate(seeds)}
        rows.sort(key=lambda row: order[row['seed']])
        return rows, elapsed
    finally:
        stop.set()
        for process, incoming, _ in lanes:
            if process.pid is not None and process.is_alive():
                try:
                    incoming.put_nowait(None)
                except queue.Full:
                    pass
        errors = []
        for process, incoming, _ in lanes:
            try:
                if process.pid is not None:
                    _stop_process(process)
                process.close()
            except RuntimeError as exc:
                errors.append(str(exc))
            finally:
                _close_queue(incoming)
        _close_queue(outgoing)
        if errors:
            raise UnsafeWorkerShutdown('; '.join(errors))


run_wave = run_group

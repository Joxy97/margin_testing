"""Bounded supervision checks; controlled children perform no optimization."""
import hashlib
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from qubo_benchmark.runtime import continuous_pool as pool


def template(tmp_path, **overrides):
    path = tmp_path/'input.npz'
    path.write_bytes(b'controlled immutable input')
    return dict(dict(npz=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cpu_threads=1, solver='lib_random_search', parameters={'dtype': 'float32'},
        budget=20., checkpoints=(.05, .1, .2, .5, 1., 2., 5., 10., 20.),
        wall_schedule=False, reference=0., reference_type='BKS', tolerance=0.), **overrides)


class FakeQueue(queue.Queue):
    def __init__(self, context, maxsize):
        super().__init__(maxsize=maxsize)
        self.context, self.process = context, None
        self.closed = self.cancelled = False

    def put(self, value, block=True, timeout=None):
        if self.process is not None:
            if value is None:
                return
            self.process.dispatch(value)
        else:
            super().put(value, block=block, timeout=timeout)

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled = True


class FakeProcess:
    def __init__(self, context, target, args):
        self.context, self.target, self.args = context, target, args
        self.index, self.device, self.job, incoming, self.outgoing, self.stop, self.calibration = args[:7]
        self.shared_stop = args[7] if len(args) > 7 else None
        incoming.process = self
        self.pid, self.alive, self.closed = None, False, False
        self.seeds, self.calls = [], []

    def start(self):
        self.pid, self.alive = 1000+self.index, True
        self.context.prepared.add(self.index)
        if self.context.start_fatal == self.index:
            self.outgoing.put(dict(kind='fatal', worker=self.index, error='InputIdentityError: changed input',
                                  fatal_calibration=True))
        elif self.context.ready:
            self.outgoing.put(dict(kind='ready', worker=self.index))
        if self.context.after_prepare is not None:
            self.context.after_prepare(self.index)

    def dispatch(self, command):
        assert len(self.context.prepared) == len(self.context.processes), 'Dispatch preceded all-worker barrier'
        seed = command['seed']
        self.seeds.append(seed)
        self.outgoing.put(dict(kind='started', worker=self.index, solver=self.job['solver'],
                              seed=seed, started=time.monotonic()-self.context.solve_age))
        if seed in self.context.lost:
            self.alive = False
            return
        if seed in self.context.hang:
            return
        result = dict(status='interrupted' if seed in self.context.interrupt else 'complete',
                      energies=[float(seed)]*9, actual_solve_wall_s=.01, time_to_optimum_s=None,
                      vectors=['must not escape the compact whitelist'])
        row = pool._compact_result(result, seed, self.device, 9)
        if self.calibration:
            row.update(gpu_reserved_peak_bytes=100, gpu_allocated_peak_bytes=80, host_rss_peak_bytes=200)
        self.outgoing.put(dict(kind='done', worker=self.index, solver=self.job['solver'], seed=seed, result=row))

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.calls.append(('join', timeout))
        if self.stop.is_set():
            self.alive = False

    def terminate(self):
        self.calls.append(('terminate',))
        self.alive = False

    def kill(self):
        self.calls.append(('kill',))
        self.alive = False

    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self):
        self.processes, self.channels, self.prepared = [], [], set()
        self.ready, self.start_fatal, self.solve_age = True, None, 0.
        self.after_prepare = None
        self.lost, self.hang, self.interrupt = set(), set(), set()

    def Queue(self, maxsize):
        channel = FakeQueue(self, maxsize)
        self.channels.append(channel)
        return channel

    def Event(self):
        return threading.Event()

    def Process(self, target, args):
        process = FakeProcess(self, target, args)
        self.processes.append(process)
        return process


def fake_context(monkeypatch):
    context = FakeContext()
    def get_context(mode):
        assert mode == 'spawn'
        return context
    monkeypatch.setattr(pool.mp, 'get_context', get_context)
    return context


def controlled_serve(index, device, job, incoming, outgoing, stop, calibration):
    # Spawn/import and immutable preparation exceed each controlled solve.
    time.sleep(.15)
    outgoing.put(dict(kind='ready', worker=index))
    while not stop.is_set():
        try:
            command = incoming.get(timeout=.02)
        except queue.Empty:
            continue
        if command is None:
            return
        seed, started = command['seed'], time.monotonic()
        outgoing.put(dict(kind='started', worker=index, seed=seed,
                          solver=job['solver'], started=started))
        random.seed(seed)
        value = random.random()
        time.sleep(.025)
        row = dict(seed=seed, device=device, status='complete', energies=[value]*9,
                   actual_solve_wall_s=time.monotonic()-started, time_to_optimum_s=None)
        outgoing.put(dict(kind='done', worker=index, seed=seed, solver=job['solver'], result=row))


def test_full_primary_group_covers_every_seed_and_reuses_bounded_device_lanes(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    committed = []
    devices, counts = ['cuda:0', 'cuda:1'], {'cuda:0': 2, 'cuda:1': 1}
    rows, elapsed = pool.run_group(template(tmp_path), devices, counts, range(100), committed.append)
    assert len(rows) == len(committed) == 100
    assert [row['seed'] for row in rows] == list(range(100))
    assert sorted(seed for process in context.processes for seed in process.seeds) == list(range(100))
    assert [process.device for process in context.processes] == ['cuda:0', 'cuda:0', 'cuda:1']
    assert all(len(process.seeds) > 1 for process in context.processes)
    assert elapsed >= 0.
    assert context.channels[0].maxsize == 6
    assert all(channel.maxsize == 1 for channel in context.channels[1:])
    assert all(channel.closed and channel.cancelled for channel in context.channels)
    assert all(process.closed and not process.alive for process in context.processes)
    assert all(set(row) == {'seed', 'device', 'status', 'energies', 'actual_solve_wall_s',
                            'time_to_optimum_s'} for row in rows)


def test_real_spawn_excludes_setup_and_resets_independent_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(pool, '_serve', controlled_serve)
    committed = []
    began = time.monotonic()
    rows, wave = pool.run_group(template(tmp_path), ['cpu'], {'cpu': 2}, [7, 8, 9, 10], committed.append)
    whole = time.monotonic()-began
    assert [row['seed'] for row in rows] == [7, 8, 9, 10]
    assert len(committed) == 4 and whole-wave >= .15
    assert all(row['actual_solve_wall_s'] < .15 for row in rows)
    assert [row['energies'][0] for row in rows] == [random.Random(seed).random() for seed in (7, 8, 9, 10)]


def test_actual_cpu_workers_run_all_short_seeds_with_prepared_scoring(tmp_path):
    import numpy as np
    from qubo_benchmark.model import Problem
    from qubo_benchmark.runtime.common import CONFIG, filehash, read
    path = tmp_path/'tiny.npz'
    Problem(12, np.arange(12), np.arange(12), -np.arange(1, 13), 0).save(path)
    parameters = dict(read(CONFIG/'benchmark_solvers.json')['solvers']['lib_random_search'], max_steps=1)
    job = dict(npz=str(path), input_sha256=filehash(path), cpu_threads=1,
               solver='lib_random_search', parameters=parameters, budget=.2,
               checkpoints=(.05, .1, .2), reference=-78., reference_type='BKS', tolerance=0.)
    committed = []
    shared = pool.mp.get_context('spawn').Event()
    rows, elapsed = pool.run_group(job, ['cpu'], {'cpu': 2}, [7, 8, 9, 10], committed.append,
                                   stop_event=shared)
    assert len(rows) == len(committed) == 4 and elapsed >= .4
    assert not shared.is_set()
    assert all(row['status'] == 'complete' and len(row['energies']) == 3 for row in rows)
    assert all(all(value is not None for value in row['energies']) for row in rows)
    assert all(row['energies'] == sorted(row['energies'], reverse=True) for row in rows)


def test_module_import_and_pure_helpers_do_not_import_torch():
    source = ('import sys; from qubo_benchmark.runtime import continuous_pool as p; '
              "assert p._device_lanes(['cpu'], {'cpu': 1}) == ['cpu']; "
              "assert 'torch' not in sys.modules")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    result = subprocess.run([sys.executable, '-c', source], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_calibration_metadata_is_optional_and_never_in_production(tmp_path, monkeypatch):
    fake_context(monkeypatch)
    job = template(tmp_path)
    production, _ = pool.run_group(job, ['cpu'], {'cpu': 1}, [0])
    fake_context(monkeypatch)
    calibration, _ = pool.run_group(job, ['cpu'], {'cpu': 1}, [1], calibration=True)
    for key in ('gpu_reserved_peak_bytes', 'gpu_allocated_peak_bytes', 'host_rss_peak_bytes'):
        assert key not in production[0] and isinstance(calibration[0][key], int)


def test_scalar_whitelist_and_failed_rows_exclude_partial_quality():
    result = dict(status='oom', energies=[-3.]*9, error='allocation failed', stop_reason='oom',
                  actual_solve_wall_s=.2, time_to_optimum_s=None, states=[object()], vectors=[object()])
    row = pool._compact_result(result, 8, 'cpu', 9)
    assert row['energies'] == [None]*9 and row['error'] == 'allocation failed'
    assert 'states' not in row and 'vectors' not in row
    with pytest.raises(ValueError, match='checkpoint count'):
        pool._compact_result(dict(status='complete', energies=[0.]), 8, 'cpu', 9)


def test_callback_completions_are_drained_before_watchdog(tmp_path, monkeypatch):
    fake_context(monkeypatch)
    monkeypatch.setattr(pool, 'SOLVE_TIMEOUT_S', .001)
    committed = []
    def slow_commit(row):
        committed.append(row['seed'])
        time.sleep(.01)
    rows, elapsed = pool.run_group(template(tmp_path), ['cpu'], {'cpu': 2}, [0, 1, 2, 3], slow_commit)
    assert len(rows) == 4 and committed == [0, 1, 2, 3]
    assert elapsed >= .04


@pytest.mark.parametrize('failure', ['lost', 'interrupt', 'hang', 'callback'])
def test_incomplete_active_seeds_never_get_a_completion_callback(tmp_path, monkeypatch, failure):
    context = fake_context(monkeypatch)
    if failure != 'callback':
        getattr(context, failure).add(1)
    if failure == 'hang':
        context.solve_age = 1.
        monkeypatch.setattr(pool, 'SOLVE_TIMEOUT_S', .001)
    committed = []
    def commit(row):
        if failure == 'callback' and row['seed'] == 1:
            raise OSError('durable commit failed')
        committed.append(row['seed'])
    exception = KeyboardInterrupt if failure == 'interrupt' else OSError if failure == 'callback' else RuntimeError
    with pytest.raises(exception):
        pool.run_group(template(tmp_path), ['cpu'], {'cpu': 1}, [0, 1, 2], commit)
    assert committed == [0]
    assert context.processes[0].seeds == [0, 1]
    assert all(channel.closed for channel in context.channels)
    assert all(not process.alive for process in context.processes)


def test_identity_failure_precedes_all_timed_dispatch(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    job = template(tmp_path)
    Path(job['npz']).write_bytes(b'changed after manifest')
    with pytest.raises(RuntimeError, match='manifest checksum'):
        pool.run_group(job, ['cpu'], {'cpu': 1}, [0])
    assert context.processes == []


def test_child_preparation_fatal_keeps_entire_group_pending(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    context.start_fatal = 1
    with pytest.raises(RuntimeError, match='InputIdentityError'):
        pool.run_group(template(tmp_path), ['cpu'], {'cpu': 2}, [0, 1])
    assert all(process.seeds == [] for process in context.processes)
    assert all(not process.alive for process in context.processes)


def test_input_change_during_readiness_is_fatal_before_dispatch(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    job = template(tmp_path)
    def mutate(index):
        if index == 1:
            Path(job['npz']).write_bytes(b'input changed during immutable preparation')
    context.after_prepare = mutate
    with pytest.raises(pool.FatalWorkerError, match='manifest checksum'):
        pool.run_group(job, ['cpu'], {'cpu': 2}, [0, 1])
    assert all(process.seeds == [] for process in context.processes)
    assert all(not process.alive for process in context.processes)


def test_setup_watchdog_is_independent_from_solve_watchdog(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    context.ready = False
    monkeypatch.setattr(pool, 'SETUP_TIMEOUT_S', .001)
    monkeypatch.setattr(pool, 'POLL_INTERVAL_S', .002)
    with pytest.raises(RuntimeError, match='immutable setup watchdog'):
        pool.run_group(template(tmp_path), ['cpu'], {'cpu': 1}, [0])
    assert context.processes[0].seeds == []


@pytest.mark.parametrize('survives_kill', [False, True])
def test_shutdown_escalates_only_for_the_owned_process(survives_kill):
    class Process:
        pid = 4321
        alive = True
        def __init__(self): self.calls = []
        def join(self, timeout): self.calls.append(('join', timeout))
        def is_alive(self): return self.alive
        def terminate(self): self.calls.append(('terminate',))
        def kill(self):
            self.calls.append(('kill',))
            self.alive = survives_kill
    process = Process()
    if survives_kill:
        with pytest.raises(pool.UnsafeWorkerShutdown, match='PID 4321 survived shutdown'):
            pool._stop_process(process)
    else:
        pool._stop_process(process)
    assert process.calls == [('join', 2.), ('terminate',), ('join', 5.), ('kill',), ('join', 2.)]


def test_allocator_peaks_are_per_assigned_device(monkeypatch):
    calls = []
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda device: calls.append(('reset', device)),
                           max_memory_allocated=lambda device: calls.append(('allocated', device)) or 40,
                           max_memory_reserved=lambda device: calls.append(('reserved', device)) or 60)
    worker = SimpleNamespace(device='cuda:2', torch=SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(pool, '_host_rss_peak_bytes', lambda: 80)
    pool._reset_calibration_memory(worker)
    assert pool._calibration_memory(worker) == dict(gpu_allocated_peak_bytes=40,
        gpu_reserved_peak_bytes=60, host_rss_peak_bytes=80)
    assert calls == [('reset', 'cuda:2'), ('allocated', 'cuda:2'), ('reserved', 'cuda:2')]


def test_child_prepares_scorer_once_and_keeps_only_immutable_worker_between_seeds(tmp_path, monkeypatch):
    events, instances = [], []
    def warm_worker(job):
        worker = SimpleNamespace(device=job['device'], ready=False)
        instances.append(worker)
        events.append('context')
        return worker
    def prepare(worker, job):
        worker.ready = True
        events.append('scorer')
    def solve(worker, job, stop):
        assert worker is instances[0] and worker.ready
        assert list(outgoing.queue)[0]['kind'] == 'ready'
        events.append(job['seed'])
        return dict(status='complete', energies=[float(job['seed'])]*9)
    monkeypatch.setitem(sys.modules, 'qubo_benchmark.runtime.worker', SimpleNamespace(WarmWorker=warm_worker))
    monkeypatch.setitem(sys.modules, 'qubo_benchmark.runtime.continuous_cli', SimpleNamespace(_prepare_job=prepare))
    monkeypatch.setitem(sys.modules, 'qubo_benchmark.runtime.continuous_worker',
                        SimpleNamespace(run_continuous_trial=solve))
    incoming, outgoing = queue.Queue(), queue.Queue()
    incoming.put(dict(seed=7))
    incoming.put(dict(seed=8))
    incoming.put(None)
    pool._serve(0, 'cpu', dict(template(tmp_path), seed=7), incoming, outgoing, threading.Event(), False)
    assert len(instances) == 1 and events == ['context', 'scorer', 7, 8]
    messages = list(outgoing.queue)
    assert [message['kind'] for message in messages] == ['ready', 'started', 'done', 'started', 'done']
    assert [message['result']['seed'] for message in messages if message['kind'] == 'done'] == [7, 8]


@pytest.mark.parametrize('devices, counts', [([], {}), (['cpu', 'cpu'], {'cpu': 1}),
    (['cpu'], {}), (['cpu'], {'cpu': 0}), (['cpu'], {'cpu': True})])
def test_worker_assignment_validation(devices, counts):
    with pytest.raises(ValueError):
        pool._device_lanes(devices, counts)


def test_shared_stop_aborts_pending_dispatch_and_reaps_owned_children(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    shared = threading.Event()
    committed = []
    def commit(row):
        committed.append(row['seed'])
        shared.set()
    with pytest.raises(KeyboardInterrupt):
        pool.run_group(template(tmp_path), ['cpu'], {'cpu': 1}, [0, 1, 2], commit, stop_event=shared)
    assert committed == [0] and context.processes[0].seeds == [0]
    assert context.processes[0].shared_stop is shared
    assert all(not process.alive and process.closed for process in context.processes)


def test_normal_pool_cleanup_does_not_cancel_another_shared_pool(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    shared = threading.Event()
    rows, _ = pool.run_group(template(tmp_path), ['cpu'], {'cpu': 1}, [0], stop_event=shared)
    assert len(rows) == 1 and not shared.is_set()
    assert context.processes[0].shared_stop is shared


def test_shared_stop_before_start_creates_no_children(tmp_path, monkeypatch):
    context = fake_context(monkeypatch)
    shared = threading.Event()
    shared.set()
    with pytest.raises(KeyboardInterrupt):
        pool.run_group(template(tmp_path), ['cpu'], {'cpu': 1}, [0], stop_event=shared)
    assert context.processes == []


def test_worker_stop_signal_combines_pool_cleanup_and_shared_cancel():
    owned, shared = threading.Event(), threading.Event()
    signal = pool._StopSignal(owned, shared)
    assert not signal.is_set()
    shared.set()
    assert signal.is_set()
    shared.clear()
    owned.set()
    assert signal.is_set()
    assert pool.UnsafeWorkerShutdown.fatal_calibration and pool.FatalWorkerError.fatal_calibration

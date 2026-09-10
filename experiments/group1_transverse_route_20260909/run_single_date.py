"""Time one complete Group 1 calculation, with live solver/scenario progress."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import threading
import time
import traceback

import torch

from margin_engine import MarginApplicationConfig
from margin_calculator.optimization.optimization_solver.bqm_solver import TorchTransverseRouteBQMSolver
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
LOCK = threading.Lock()
START = time.perf_counter()
STATE = {'state': 'initializing', 'generated_scenarios': 0, 'solved_scenarios': 0,
         'active': {}, 'completed_batches': []}
STOP = threading.Event()


def emit(message):
    print(f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {message}', flush=True)


def snapshot():
    with LOCK:
        value = dict(STATE, elapsed_seconds=time.perf_counter() - START)
        tmp = OUT / 'status.json.tmp'
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(OUT / 'status.json')


def heartbeat():
    while not STOP.wait(15):
        snapshot()
        with LOCK:
            emit(f"HEARTBEAT elapsed={time.perf_counter() - START:.1f}s generated={STATE['generated_scenarios']} solved={STATE['solved_scenarios']} active={json.dumps(STATE['active'])}")


def main():
    torch.set_num_threads(1)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    config = HERE / 'transverse_route.yaml'
    app = MarginApplicationConfig.fromYaml(config)
    engine = app.createEngine()
    devices = engine.marginCalculator.bqmSolver.devices
    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)
    original_batch = TorchTransverseRouteBQMSolver._solveBatch
    original_add = TorchCandidateAccumulator.add
    original_states = engine.riskStateGenerator.getRiskStates

    def states(context):
        for index, state in enumerate(original_states(context), 1):
            with LOCK:
                STATE['generated_scenarios'] = index
            emit(f'SCENARIO_READY index={index}')
            yield state

    def solve_batch(solver, problems, parameters):
        device = solver.device
        started = time.perf_counter()
        description = dict(phase='preparation/integration', problems=len(problems),
            variables=[p.variableCount for p in problems], edges=[p.interactionCount for p in problems],
            started_elapsed_seconds=started - START)
        with LOCK:
            STATE['active'][device] = description
        emit(f'SOLVE_START device={device} {json.dumps(description)}')
        result = original_batch(solver, problems, parameters)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        with LOCK:
            STATE['solved_scenarios'] += len(problems)
            STATE['active'].pop(device, None)
            STATE['completed_batches'].append(dict(device=device, seconds=elapsed,
                problems=len(problems), energies=[r.energy for r in result]))
        emit(f'SOLVE_END device={device} seconds={elapsed:.6f} energies={[r.energy for r in result]}')
        return result

    def add_candidates(accumulator, samples):
        device = str(samples.device)
        started = time.perf_counter()
        with LOCK:
            if device in STATE['active']:
                STATE['active'][device]['phase'] = f'scoring/repair ({len(samples)} candidates)'
        emit(f'CANDIDATES_START device={device} variables={accumulator.problem.variableCount} candidates={len(samples)}')
        result = original_add(accumulator, samples)
        emit(f'CANDIDATES_END device={device} seconds={time.perf_counter() - started:.6f}')
        with LOCK:
            if device in STATE['active']:
                STATE['active'][device]['phase'] = 'integration/collection'
        return result

    engine.riskStateGenerator.getRiskStates = states
    TorchTransverseRouteBQMSolver._solveBatch = solve_batch
    TorchCandidateAccumulator.add = add_candidates
    hardware = {d: torch.cuda.get_device_name(d) for d in devices}
    emit(f'START date={app.marginDate} steps=1000 runs=4 hardware={hardware}')
    with LOCK:
        STATE['state'] = 'running'
    snapshot()
    synchronize()
    started = time.perf_counter()
    try:
        report = engine.generateReport(app.portfolio, app.marginDate)
        synchronize()
        wall = time.perf_counter() - started
        row = dict(date=str(app.marginDate), group='US_group_1', assets=8590,
            solver='torch_transverse_route', steps=1000, runs=4, seed=1,
            solver_dtype='float32', n_devices=len(devices), margin=report.margin,
            wall_seconds=wall, **asdict(report.timings))
        (OUT / 'report.json').write_text(json.dumps(dict(row,
            comparisonMargins=dict(report.comparisonMargins),
            numericalDiagnostics=dict(report.numericalDiagnostics),
            hardware=hardware, torch=torch.__version__, cuda=torch.version.cuda,
            python=platform.python_version(),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            batches=STATE['completed_batches'],
            timing_note='Inclusive pipeline wall time; generation overlaps calculation. Per-device batch durations overlap across devices.'), indent=2) + '\n')
        with (OUT / 'benchmark_single_date.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        with LOCK:
            STATE.update(state='completed', margin=report.margin, wall_seconds=wall)
        emit(f'FINISHED {json.dumps(row)}')
    except BaseException:
        with LOCK:
            STATE.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        STOP.set()
        thread.join()
        snapshot()


if __name__ == '__main__':
    main()

"""Logged full Group 1 categorical TRF margin run with a hard no-repair guard."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

import torch

from margin_engine import MarginApplicationConfig
from margin_calculator.optimization.optimization_solver.bqm_solver import TorchCategoricalTRFBQMSolver
from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
LOCK = threading.Lock()
STOP = threading.Event()
START = time.perf_counter()
STATE = dict(state='initializing', generated=0, solved=0, active={}, batches=[],
             candidate_batches=0, candidates=0, repair_calls=0)


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def snapshot():
    with LOCK:
        payload = dict(STATE, elapsed_seconds=time.perf_counter() - START)
        temporary = OUT / 'status.json.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(OUT / 'status.json')


def heartbeat():
    while not STOP.wait(15):
        snapshot()
        with LOCK:
            log(f"HEARTBEAT elapsed={time.perf_counter() - START:.1f}s "
                f"generated={STATE['generated']}/105 solved={STATE['solved']}/105 "
                f"candidates={STATE['candidates']} active={STATE['active']}")


def reject_repair(*args, **kwargs):
    with LOCK:
        STATE['repair_calls'] += 1
    raise RuntimeError('Categorical TRF benchmark attempted candidate repair')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        config = HERE / 'categorical_trf.yaml'
        shutil.copy2(config, OUT / 'configuration.yaml')
        app = MarginApplicationConfig.fromYaml(config)
        engine = app.createEngine()
        devices = engine.marginCalculator.bqmSolver.devices
        original_states = engine.riskStateGenerator.getRiskStates
        original_solve = TorchCategoricalTRFBQMSolver._solveBatch
        original_add = TorchCandidateAccumulator.add

        def states(context):
            for state in original_states(context):
                with LOCK:
                    STATE['generated'] += 1
                    index = STATE['generated']
                log(f'SCENARIO_READY index={index}')
                yield state

        def solve(solver, problems, parameters):
            device = str(solver.device)
            with LOCK:
                STATE['active'][device] = dict(phase='preparation/integration',
                    problems=len(problems), variables=[p.variableCount for p in problems])
            log(f'SOLVE_START device={device} variables={[p.variableCount for p in problems]}')
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = original_solve(solver, problems, parameters)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            with LOCK:
                STATE['solved'] += len(problems)
                STATE['active'].pop(device, None)
                STATE['batches'].append(dict(device=device, problems=len(problems), seconds=elapsed,
                    energies=[r.energy for r in result]))
            log(f'SOLVE_END device={device} problems={len(problems)} seconds={elapsed:.3f}')
            return result

        def add(accumulator, samples):
            device = str(samples.device)
            started = time.perf_counter()
            with LOCK:
                if device in STATE['active']:
                    STATE['active'][device]['phase'] = 'float64 scoring (no repair)'
            # Shared selection checks group feasibility; any repair request is fatal.
            original_add(accumulator, samples)
            if not accumulator.feasible:
                raise RuntimeError('No feasible categorical candidate in checkpoint batch')
            with LOCK:
                STATE['candidate_batches'] += 1
                STATE['candidates'] += len(samples)
                if device in STATE['active']:
                    STATE['active'][device]['phase'] = 'integration'
            log(f'CANDIDATES device={device} count={len(samples)} feasible=true '
                f'best_energy={accumulator.best[1]:.9g} seconds={time.perf_counter() - started:.3f}')

        CandidateSelection._repairCandidate = staticmethod(reject_repair)
        engine.riskStateGenerator.getRiskStates = states
        TorchCategoricalTRFBQMSolver._solveBatch = solve
        TorchCandidateAccumulator.add = add
        hardware = {str(d): torch.cuda.get_device_name(d) for d in devices}
        log(f'START solver=torch_categorical_trf date={app.marginDate} runs=4 steps=10000 hardware={hardware}')
        with LOCK:
            STATE['state'] = 'running'
        for device in devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        report = engine.generateReport(app.portfolio, app.marginDate)
        for device in devices:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        row = dict(date=str(app.marginDate), assets=8590, solver='torch_categorical_trf',
            steps=10000, runs=4, seed=1, dtype='float32', devices=len(devices), margin=report.margin,
            wall_seconds=elapsed, scenarios=STATE['solved'], repair_calls=STATE['repair_calls'],
            **asdict(report.timings))
        with (OUT / 'benchmark_single_date.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        payload = dict(row, hardware=hardware, torch_version=torch.__version__, cuda_version=torch.version.cuda,
            configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), batches=STATE['batches'],
            candidate_batches=STATE['candidate_batches'], candidates=STATE['candidates'],
            notes=['Experimental categorical mirror-flow TRF variant; not the original binary angular TRF.',
                   'All checkpoint candidates are one-hot; any repair invocation aborts the run.',
                   'Float32 dynamics; original float64 QUBO scoring and standard margin decoding.',
                   'Risk generation and calculation timings overlap; concurrent batch times are not wall time.',
                   'Cold calculation includes preparation, graph capture, scoring and logging; no speedup claimed.'])
        (OUT / 'report.json').write_text(json.dumps(payload, indent=2) + '\n')
        with LOCK:
            STATE.update(state='completed', margin=report.margin, wall_seconds=elapsed)
        log('FINISHED ' + json.dumps(row))
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

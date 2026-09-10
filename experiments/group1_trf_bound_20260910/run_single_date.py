"""Binary TRF with repair and a sufficient source-QUBO one-hot penalty bound."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
import traceback

import numpy as np
import torch

from margin_engine import MarginApplicationConfig
from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_solver.bqm_solver import TorchTransverseRouteBQMSolver
from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator


HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
LOCK = threading.Lock()
STOP = threading.Event()
START = time.perf_counter()
STATE = dict(state='initializing', generated=0, solved=0, active={}, batches=[],
             candidate_batches=0, candidates=0, repair_calls=0, repair_seconds=0.)
PENALTIES = []
THREAD_STATE = threading.local()


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def snapshot():
    with LOCK:
        temporary = OUT / 'status.json.tmp'
        temporary.write_text(json.dumps(dict(STATE, elapsed_seconds=time.perf_counter() - START), indent=2) + '\n')
        temporary.replace(OUT / 'status.json')


def heartbeat():
    while not STOP.wait(15):
        snapshot()
        with LOCK:
            log(f"HEARTBEAT elapsed={time.perf_counter() - START:.1f}s generated={STATE['generated']}/105 "
                f"solved={STATE['solved']}/105 repairs={STATE['repair_calls']} active={STATE['active']}")


def bounded_problem(objective, penalty_floor):
    """Accept a zero-one-hot-penalty portfolio QUBO and add the bounded penalty.

    The dedicated visitor call uses lambdaOneHot=0, retaining its zero-valued
    clique topology. All diagonal terms are included in the objective bound.
    Summing absolute duplicate terms is conservative even without aggregation.
    """
    groups = tuple(tuple(group) for group in objective.iterOneHotGroups())
    if not groups or sum(map(len, groups)) != objective.variableCount:
        raise ValueError('Bounded penalty requires complete disjoint one-hot groups')
    owner = np.empty(objective.variableCount, dtype=np.int64)
    for index, group in enumerate(groups):
        owner[list(group)] = index
    heads, tails, biases = objective.quadraticHeads, objective.quadraticTails, objective.quadraticBiases
    diagonal = heads == tails
    within = (owner[heads] == owner[tails]) & ~diagonal
    # This helper deliberately targets PortfolioRiskStateBQMVisitor, whose
    # within-group objective interactions are zero and topology is canonical.
    expected_pairs = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    if int(within.sum()) != expected_pairs or np.any(biases[within] != 0):
        raise ValueError('Expected zero-penalty portfolio clique topology')
    objective_linear = objective.linear.copy()
    np.add.at(objective_linear, heads[diagonal], biases[diagonal])
    bounds = np.abs(objective_linear)
    cross = ~diagonal & ~within
    magnitudes = np.abs(biases[cross])
    np.add.at(bounds, heads[cross], magnitudes)
    np.add.at(bounds, tails[cross], magnitudes)
    bound = float(bounds.max(initial=0.))
    penalty = max(float(penalty_floor), float(np.nextafter(1.01 * bound, np.inf)), 1e-6)
    if not math.isfinite(penalty) or penalty <= bound:
        raise ValueError('Unable to construct a finite strict penalty bound')
    new_biases = biases.copy()
    new_biases[within] = 2. * penalty
    digest = hashlib.blake2b(digest_size=8, person=b'onehot-bound')
    digest.update(int(objective.seedOffset).to_bytes(8, 'little'))
    digest.update(np.float64(penalty).tobytes())
    problem = QUBOProblem(linear=objective.linear - penalty,
        quadraticHeads=heads, quadraticTails=tails, quadraticBiases=new_biases,
        offset=objective.offset + penalty * len(groups),
        oneHotGroups=groups, seedOffset=int.from_bytes(digest.digest(), 'little'))
    detail = dict(variables=problem.variableCount, groups=len(groups), edges=problem.interactionCount,
        lambda_floor=float(penalty_floor), nonpenalty_row_bound=bound, lambda_oneHot=penalty,
        lambdaCompat=.1, max_abs_objective_linear=float(np.abs(objective_linear).max(initial=0.)),
        max_abs_cross_coefficient=float(magnitudes.max(initial=0.)),
        float32_penalty_ulp=float(np.spacing(np.float32(penalty))))
    return problem, detail


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    penalty_stream = None
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        config = HERE / 'transverse_route.yaml'
        shutil.copy2(config, OUT / 'configuration.yaml')
        app = MarginApplicationConfig.fromYaml(config)
        engine = app.createEngine()
        calculator = engine.marginCalculator
        devices = calculator.bqmSolver.devices
        model = dict(calculator.modelParameters)
        floor = float(model.get('lambdaOneHot', 1.))
        if not math.isfinite(floor) or floor < 0:
            raise ValueError('lambdaOneHot floor must be finite and nonnegative')
        objective_parameters = dict(model, lambdaOneHot=0.)
        original_solve = TorchTransverseRouteBQMSolver._solveBatch
        original_add = TorchCandidateAccumulator.add
        original_repair = CandidateSelection._repairCandidate
        penalty_stream = (OUT / 'penalties.csv').open('w', newline='')
        penalty_writer = None

        def encode(risk_state, portfolio):
            nonlocal penalty_writer
            started = time.perf_counter()
            objective = calculator.bqmVisitor.createBQM(risk_state, portfolio, objective_parameters)
            problem, detail = bounded_problem(objective, floor)
            with LOCK:
                STATE['generated'] += 1
                detail = dict(scenario=STATE['generated'], **detail,
                              encoding_seconds=time.perf_counter() - started)
                detail['lambdaCompat'] = float(model.get('lambdaCompat', .1))
                PENALTIES.append(detail)
                if penalty_writer is None:
                    penalty_writer = csv.DictWriter(penalty_stream, fieldnames=list(detail))
                    penalty_writer.writeheader()
                penalty_writer.writerow(detail)
                penalty_stream.flush()
            log('QUBO_BOUND ' + json.dumps(detail))
            return risk_state, problem

        def solve(solver, problems, parameters):
            device = str(solver.device)
            THREAD_STATE.device = device
            with LOCK:
                STATE['active'][device] = dict(phase='preparation/integration', problems=len(problems))
            log(f'SOLVE_START device={device} variables={[p.variableCount for p in problems]}')
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            results = original_solve(solver, problems, parameters)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            with LOCK:
                STATE['solved'] += len(problems)
                STATE['active'].pop(device, None)
                STATE['batches'].append(dict(device=device, problems=len(problems), seconds=elapsed,
                                             energies=[r.energy for r in results]))
            log(f'SOLVE_END device={device} seconds={elapsed:.3f} problems={len(problems)}')
            return results

        def repair(*args, **kwargs):
            device = getattr(THREAD_STATE, 'device', 'unknown')
            with LOCK:
                STATE['repair_calls'] += 1
                number = STATE['repair_calls']
                if device in STATE['active']:
                    STATE['active'][device]['phase'] = f'CPU repair #{number}'
            if number <= 8 or number % 10 == 0:
                log(f'REPAIR_START device={device} total_calls={number}')
            started = time.perf_counter()
            result = original_repair(*args, **kwargs)
            elapsed = time.perf_counter() - started
            with LOCK:
                STATE['repair_seconds'] += elapsed
            if number <= 8 or number % 10 == 0:
                log(f'REPAIR_END device={device} seconds={elapsed:.3f}')
            return result

        def add(accumulator, samples):
            device = str(samples.device)
            started = time.perf_counter()
            with LOCK:
                STATE['candidate_batches'] += 1
                STATE['candidates'] += len(samples)
                if device in STATE['active']:
                    STATE['active'][device]['phase'] = 'candidate scoring/repair'
            log(f'CANDIDATES_START device={device} count={len(samples)}')
            original_add(accumulator, samples)
            with LOCK:
                if device in STATE['active']:
                    STATE['active'][device]['phase'] = 'integration'
            log(f'CANDIDATES_END device={device} raw_feasible_seen={accumulator.feasible} '
                f'best_energy={accumulator.best[1]:.9g} seconds={time.perf_counter() - started:.3f}')

        calculator._encodeRiskState = encode
        TorchTransverseRouteBQMSolver._solveBatch = solve
        TorchCandidateAccumulator.add = add
        CandidateSelection._repairCandidate = staticmethod(repair)
        hardware = {str(d): torch.cuda.get_device_name(d) for d in devices}
        with LOCK:
            STATE['state'] = 'running'
        log(f'START binary TRF WITH REPAIR date={app.marginDate} steps=10000 runs=4 hardware={hardware}')
        for device in devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        report = engine.generateReport(app.portfolio, app.marginDate)
        for device in devices:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        row = dict(date=str(app.marginDate), assets=8590, solver='torch_transverse_route',
            steps=10000, runs=4, seed=1, dtype='float32', devices=len(devices),
            penalty_rule='max(floor, nextafter(1.01 * max_absolute_nonpenalty_row_sum, +inf))',
            lambda_min=min(p['lambda_oneHot'] for p in PENALTIES),
            lambda_max=max(p['lambda_oneHot'] for p in PENALTIES),
            margin=report.margin, wall_seconds=elapsed, scenarios=STATE['solved'],
            repair_calls=STATE['repair_calls'], repair_seconds=STATE['repair_seconds'], **asdict(report.timings))
        with (OUT / 'benchmark_single_date.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        payload = dict(row, hardware=hardware, torch_version=torch.__version__, cuda_version=torch.version.cuda,
            configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), penalties=PENALTIES,
            batches=STATE['batches'], candidates=STATE['candidates'],
            notes=['Original binary TRF; normal shared categorical repair remains enabled.',
                   'Penalty bound applies to the float64 source objective, not finite-time trajectory feasibility.',
                   'Large penalties can worsen float32 objective resolution and dynamics conditioning.',
                   'Each QUBO is encoded without one-hot penalties before the bound is computed.',
                   'Stage times overlap; concurrent repair/batch seconds are not end-to-end wall time.'])
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
        if penalty_stream is not None:
            penalty_stream.close()
        snapshot()


if __name__ == '__main__':
    main()

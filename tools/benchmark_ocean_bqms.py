"""Stream local Ocean QUBOs through the canonical solver library.

Solver time and explicit application one-hot repair time are measured separately.
No compiled benchmark executable or replacement data are required.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path
import dimod
import numpy as np
import torch
from qubo_solvers import create_bqm_solver
from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.backends.candidate_selection import CandidateSelection

LABEL = re.compile(r'x_(\d+)_(\d+)$')


def load_problem(path):
    started = time.perf_counter()
    with path.open('rb') as stream:
        model = dimod.BinaryQuadraticModel.from_file(stream)
    if model.vartype is not dimod.BINARY:
        raise ValueError(f'{path} is not a binary BQM')
    indexed = []
    for label in model.variables:
        match = LABEL.fullmatch(str(label))
        if match is None:
            raise ValueError(f'Unsupported variable label {label!r}')
        indexed.append((int(match[1]), int(match[2]), label))
    indexed.sort()
    order = [row[2] for row in indexed]
    groups = []
    for asset in sorted({row[0] for row in indexed}):
        groups.append(tuple(i for i, row in enumerate(indexed) if row[0] == asset))
    linear, quadratic, offset = model.to_numpy_vectors(variable_order=order)
    heads, tails, biases = quadratic
    problem = QUBOProblem(linear, heads, tails, biases, float(offset))
    grouped = QUBOProblem(linear, heads, tails, biases, float(offset), oneHotGroups=groups)
    return problem, grouped, time.perf_counter()-started


def benchmark(path, device, steps, greedy_steps, sa_sweeps, seed, penalty):
    problem, grouped, load_seconds = load_problem(path)
    groups = tuple(grouped.iterOneHotGroups())
    records = []
    budgets = {'lib_greedy_local_search': greedy_steps,
               'lib_simulated_annealing': sa_sweeps, 'lib_simulated_bifurcation': steps}
    for solver_id, budget in budgets.items():
        solver = create_bqm_solver(solver_id, {'device': device})
        parameters = {'steps': budget, 'runs': 1, 'seed': seed, 'dtype': 'float64'}
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        result = solver.solve(problem, parameters)
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        solve_seconds = time.perf_counter()-before
        raw = np.asarray(result.sample)
        if raw.shape != (problem.variableCount,) or not np.isin(raw, [0, 1]).all():
            raise ValueError(f'{solver_id} returned an invalid binary sample')
        raw = raw.astype(np.uint8)
        np.testing.assert_allclose(problem.energy(raw), result.energy, rtol=0, atol=1e-9)
        violations = sum(int(raw[list(group)].sum()) != 1 for group in groups)
        before = time.perf_counter()
        selection = CandidateSelection(grouped)
        selection.add([(raw, result.energy)])
        sample, energy = selection.result()
        repair_seconds = time.perf_counter()-before
        portfolio_return = float((problem.linear+penalty) @ np.asarray(sample))
        records.append({'scenario': path.stem, 'solver': solver_id, 'variables': problem.variableCount,
            'interactions': problem.interactionCount, 'assets': len(groups),
            'load_seconds': load_seconds, 'solve_seconds': solve_seconds,
            'repair_seconds': repair_seconds, 'raw_energy': result.energy,
            'energy': energy, 'raw_one_hot_violations': violations,
            'one_hot_violations': sum(sum(sample[i] for i in group) != 1 for group in groups),
            'portfolio_return': portfolio_return, 'margin': max(0., -portfolio_return),
            'configuration': json.dumps(solver._getParameters(parameters), sort_keys=True)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--greedy-steps', type=int, default=100)
    parser.add_argument('--sa-sweeps', type=int, default=8)
    parser.add_argument('--lambda-one-hot', type=float, default=1.)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=20260827)
    args = parser.parse_args()
    if min(args.steps, args.greedy_steps, args.sa_sweeps) < 1:
        parser.error('solver budgets must be positive')
    paths = sorted(args.input_dir.glob('scenario_*.bqm'), key=lambda p: int(p.stem.removeprefix('scenario_')))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error('input_dir contains no scenario_*.bqm files')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'results.csv').open('w', newline='') as stream:
        writer = None
        for path in paths:
            records = benchmark(path, args.device, args.steps, args.greedy_steps,
                                args.sa_sweeps, args.seed, args.lambda_one_hot)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
            writer.writerows(records)
            stream.flush()
            print(path.name, {r['solver']: r['energy'] for r in records}, flush=True)


if __name__ == '__main__':
    main()

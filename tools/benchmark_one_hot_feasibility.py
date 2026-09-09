"""Measure raw SBM/SVL feasibility, repair work, and original-QUBO quality.

Fixtures model signed return exposure plus nonnegative pairwise compatibility.
Penalty sweeps hold input seedOffset fixed so lambda does not change RNG streams.
"""

import argparse
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy
import torch

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_solver.bqm_solver import TorchSBMBQMSolver, TorchSVLBQMSolver
from margin_calculator.optimization.optimization_solver.bqm_solver.candidate_selection import CandidateSelection
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_execution import TorchExecution


def fixture(groups, states, exposureScale):
    rng = numpy.random.default_rng(710 + groups)
    z = numpy.linspace(-2., 2., states)
    weights = rng.uniform(5, 15, groups) * numpy.where(numpy.arange(groups) % 2, -1., 1.) * exposureScale
    linear = (weights[:, None] * numpy.expm1(rng.uniform(-.001, .001, (groups, 1)) + .02 * z)).ravel()
    pairs = sorted({tuple(sorted((a, (a + delta) % groups))) for a in range(groups) for delta in (1, 3)
                    if a != (a + delta) % groups})
    heads, tails, biases = [], [], []
    for a, b in pairs:
        rho = rng.uniform(-.8, .8)
        for i, j in itertools.product(range(states), repeat=2):
            heads.append(a * states + i)
            tails.append(b * states + j)
            biases.append(.1 * (z[i]**2 - 2 * rho * z[i] * z[j] + z[j]**2) / (1 - rho**2))
    return QUBOProblem(linear, numpy.array(heads, dtype=numpy.uint32), numpy.array(tails, dtype=numpy.uint32),
        numpy.array(biases), groupOffsets=numpy.arange(groups + 1) * states, seedOffset=710 + groups)


def sufficientPenalty(base):
    # Any correcting single-bit flip improves the penalty by at least lambda.
    # Its objective cost is bounded by |linear| + sum |incident quadratic terms|.
    bound = numpy.abs(base.linear).copy()
    numpy.add.at(bound, base.quadraticHeads, numpy.abs(base.quadraticBiases))
    numpy.add.at(bound, base.quadraticTails, numpy.abs(base.quadraticBiases))
    return float(bound.max()) * 1.01 + 1e-6


def penalized(base, strength):
    heads, tails = [], []
    for group in base.iterOneHotGroups():
        for a, b in itertools.combinations(group, 2):
            heads.append(a)
            tails.append(b)
    return QUBOProblem(base.linear - strength,
        numpy.concatenate((base.quadraticHeads, numpy.asarray(heads, dtype=numpy.uint32))),
        numpy.concatenate((base.quadraticTails, numpy.asarray(tails, dtype=numpy.uint32))),
        numpy.concatenate((base.quadraticBiases, numpy.full(len(heads), 2 * strength))),
        offset=strength * len(tuple(base.iterOneHotGroups())), groupOffsets=base.groupOffsets,
        seedOffset=base.seedOffset)


def exactReference(base):
    groups = tuple(tuple(g) for g in base.iterOneHotGroups())
    if numpy.prod([len(g) for g in groups], dtype=float) > 100000:
        return None
    best = float('inf')
    for choices in itertools.product(*groups):
        sample = numpy.zeros(base.variableCount, dtype=numpy.uint8)
        sample[list(choices)] = 1
        best = min(best, base.energy(sample))
    return best


def measure(solver, problem, parameters, repeats, reference, base):
    device = torch.device(solver.device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    solver.solve(problem, parameters)
    times = []
    for _ in range(repeats):
        sync()
        start = perf_counter()
        result = solver.solve(problem, parameters)
        sync()
        times.append(perf_counter() - start)
    captured, repair_times = [], []
    original_add = TorchCandidateAccumulator.add
    original_repair = CandidateSelection._repairCandidate
    original_model = CandidateSelection._repairModel
    model_times = []
    def add(accumulator, samples):
        captured.append(samples.cpu().numpy().copy())
        return original_add(accumulator, samples)
    def repair(*args):
        start = perf_counter()
        value = original_repair(*args)
        repair_times.append(perf_counter() - start)
        return value
    def model(*args):
        start = perf_counter()
        value = original_model(*args)
        model_times.append(perf_counter() - start)
        return value
    with patch.object(TorchCandidateAccumulator, 'add', add), \
         patch.object(CandidateSelection, '_repairCandidate', staticmethod(repair)), \
         patch.object(CandidateSelection, '_repairModel', staticmethod(model)):
        instrumented = solver.solve(problem, parameters)
    assert instrumented == result
    samples = numpy.concatenate(captured)
    groups = tuple(tuple(g) for g in problem.iterOneHotGroups())
    counts = numpy.stack([samples[:, g].sum(axis=1) for g in groups], axis=1)
    feasible = (counts == 1).all(axis=1)
    raw_energy = TorchExecution._energies(problem, samples, 10000)
    raw_objective = TorchExecution._energies(base, samples, 10000)
    # Independent diagnostic: repair ALL raw samples, including feasible ones.
    # This is separated from production repair timing and does not alter selection.
    start = perf_counter()
    adjacency, linear = original_model(problem)
    polished = [original_repair(s, problem, groups, adjacency, linear)[1] for s in samples]
    polish_time = perf_counter() - start
    assert all(sum(result.sample[i] for i in g) == 1 for g in groups)
    numpy.testing.assert_allclose(result.energy, problem.energy(result.sample), rtol=0, atol=1e-10)
    return {'seconds': times, 'medianSeconds': median(times), 'rawFeasibleFraction': float(feasible.mean()),
        'rawViolatedGroupFraction': float((counts != 1).mean()), 'rawEmptyGroupFraction': float((counts == 0).mean()),
        'rawMultipleGroupFraction': float((counts > 1).mean()), 'rawBestEnergy': float(raw_energy.min()),
        'rawBestFeasibleEnergy': float(raw_energy[feasible].min()) if feasible.any() else None,
        'rawEnergies': raw_energy.tolist(), 'rawObjectives': raw_objective.tolist(),
        'rawPenaltyEnergies': (raw_energy - raw_objective).tolist(),
        'productionRepairCalls': len(repair_times), 'productionRepairSeconds': sum(repair_times),
        'productionRepairModelSeconds': sum(model_times), 'allSamplePolishSeconds': polish_time,
        'allSamplePolishedEnergies': polished, 'returnedEnergy': result.energy,
        'exactFeasibleEnergy': reference, 'exactGap': None if reference is None else result.energy - reference,
        'returnedSampleSha256': hashlib.sha256(bytes(result.sample)).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--groups', type=int, nargs='+', default=[4, 64])
    parser.add_argument('--states', type=int, default=4)
    parser.add_argument('--exposure-scales', type=float, nargs='+', default=[1., 10.])
    parser.add_argument('--penalties', type=float, nargs='+', default=[.5, 2., 8.])
    parser.add_argument('--steps', type=int, nargs='+', default=[256, 1024])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 13, 31])
    parser.add_argument('--runs', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--solvers', nargs='+', default=['torch_sbm', 'torch_svl'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.groups, args.states, *args.steps, args.runs, args.repeats) < 1:
        parser.error('sizes and repeats must be positive')
    if any(not numpy.isfinite(v) or v < 0 for v in [*args.exposure_scales, *args.penalties]):
        parser.error('scales and penalties must be finite and nonnegative')
    torch.set_num_threads(1)
    from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory
    classes = [TorchSBMBQMSolver, TorchSVLBQMSolver, CandidateSelection, TorchExecution, TorchCandidateAccumulator]
    result = {'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': numpy.__version__,
        'python': platform.python_version(), 'hardware': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else platform.processor(),
        'arguments': vars(args) | {'output': str(args.output)},
        'sourceHashes': {c.__name__: hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in classes}}, 'measurements': []}
    for groups, scale in itertools.product(args.groups, args.exposure_scales):
        base = fixture(groups, args.states, scale)
        exact = exactReference(base)
        bound = sufficientPenalty(base)
        for strength, steps, name, seed in itertools.product([*args.penalties, bound], args.steps, args.solvers, args.seeds):
            problem = penalized(base, strength)
            solver = BQMSolverFactory.createBQMSolver(name, {'device': args.device})
            parameters = {'steps': steps, 'runs': args.runs, 'seed': seed, 'dtype': 'float32'}
            measurement = measure(solver, problem, parameters, args.repeats, exact, base)
            measurement.update({'groups': groups, 'states': args.states, 'exposureScale': scale, 'penalty': strength,
                'sufficientPenalty': bound, 'solver': name, 'parameters': solver._getParameters(parameters)})
            result['measurements'].append(measurement)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Completed groups={groups}, exposure={scale}', flush=True)


if __name__ == '__main__':
    main()

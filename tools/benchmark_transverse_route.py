"""Profile the canonical transverse-route library solver on local fixtures.

Use --device cuda:0 for synchronized GPU measurements. Search implementation
and restart batching are owned exclusively by qubo_solvers.
"""
import argparse
import json
from pathlib import Path
import torch
from qubo_solvers import create_bqm_solver
from benchmark_torch_solvers import makeProblems, measure
from benchmark_biqmac import librarySourceHashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--variables', nargs='+', type=int, default=[64, 1024])
    parser.add_argument('--problems', type=int, default=1)
    parser.add_argument('--degree', type=int, default=8)
    parser.add_argument('--one-hot', action='store_true')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--runs', type=int, default=16)
    parser.add_argument('--run-batch-size', type=int, default=8)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--integrator', choices=['euler', 'heun'], default='heun')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--profile-directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.variables, args.problems, args.degree, args.steps, args.runs,
           args.run_batch_size, args.repeats) < 1 or args.warmups < 0:
        parser.error('sizes and repeats must be positive; warmups nonnegative')
    torch.set_num_threads(1)
    constructor = {'devices': args.devices} if args.devices else {'device': args.device}
    solver = create_bqm_solver('lib_transverse_route', constructor)
    parameters = dict(steps=args.steps, runs=args.runs, run_batch_size=args.run_batch_size,
                      dtype=args.dtype, integrator=args.integrator, seed=13)
    output = {'solver': 'lib_transverse_route', 'devices': list(solver.devices),
              'parameters': solver._getParameters(parameters), 'measurements': [],
              'library_source_sha256': librarySourceHashes()}
    for n in args.variables:
        problems = makeProblems(n, args.problems, args.degree, args.one_hot)
        trace = args.profile_directory / f'transverse_route_{n}.json' if args.profile_directory else None
        row = measure(solver, problems, parameters, args.warmups, args.repeats, trace)
        row.update(variables=n, problems=len(problems))
        output['measurements'].append(row)
        print(json.dumps({'variables': n, 'median_seconds': row['median_seconds']}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Compare pinned upstream flow, optimized solves, and multi-GPU throughput.

Use --upstream-root for a checkout of Joxy97/solver_testing. The matched reference
uses upstream equations, COO multiplication and allocating integration with this
repository's identical seeds, normalization and candidate-selection contract.
The original upstream solve can also be measured separately; its global RNG and
lack of one-hot repair make its quality results non-identical to this adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy
import torch

from margin_calculator.optimization.optimization_solver.bqm_solver import TorchTransverseRouteBQMSolver
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_transverse_route_bqm_solver import _FlowWorkspace
from benchmark_torch_solvers import makeProblems


def referenceClass(upstream):
    class ReferenceWorkspace(_FlowWorkspace):
        def __init__(self, torch, matrix, field, theta, parameters):
            if matrix.layout == torch.sparse_csr:
                matrix = matrix.to_sparse_coo()
            super().__init__(torch, matrix, field, theta, parameters)

        def advance(self, kappa, nextKappa):
            p, theta = self.parameters, self.theta
            first = upstream._flow_rhs(torch, self.matrix, self.field[:, 0], theta,
                kappa=kappa, mobility=p['mobility'], route_strength=p['route_strength'], gamma=p['gamma'])
            if p['integrator'] == 'heun':
                predictor = upstream._wrap_angles(torch, theta + p['time_step'] * first)
                second = upstream._flow_rhs(torch, self.matrix, self.field[:, 0], predictor,
                    kappa=nextKappa, mobility=p['mobility'], route_strength=p['route_strength'], gamma=p['gamma'])
                result = upstream._wrap_angles(torch, theta + .5 * p['time_step'] * (first + second))
            else:
                result = upstream._wrap_angles(torch, theta + p['time_step'] * first)
            theta.copy_(result)

    class ReferenceSolver(TorchTransverseRouteBQMSolver):
        _workspaceClass = ReferenceWorkspace

    return ReferenceSolver


def parity(upstream, device):
    """Compare trajectories from identical angles, including Euler and Heun."""
    rng = numpy.random.default_rng(712)
    matrix = rng.normal(0, .03, (23, 23))
    matrix = matrix + matrix.T
    numpy.fill_diagonal(matrix, 0.)
    field, initial = rng.normal(0, .1, (23, 1)), rng.uniform(-numpy.pi, numpy.pi, (23, 7))
    rows = []
    for dtype in (torch.float32, torch.float64):
        for integrator in ('euler', 'heun'):
            for route, gamma in ((0., 0.), (1., 0.), (1., .3)):
                p = TorchTransverseRouteBQMSolver._getParameters({
                    'integrator': integrator, 'route_strength': route, 'gamma': gamma})
                theta = torch.tensor(initial, dtype=dtype, device=device)
                reference = torch.tensor(initial, dtype=dtype)
                workspace = _FlowWorkspace(torch, torch.tensor(matrix, dtype=dtype, device=device),
                    torch.tensor(field, dtype=dtype, device=device), theta, p)
                cpu_matrix, cpu_field = torch.tensor(matrix, dtype=dtype), torch.tensor(field[:, 0], dtype=dtype)
                with torch.inference_mode():
                    for step in range(64):
                        kappa, next_kappa = -1 + 3 * step / 64, -1 + 3 * (step + 1) / 64
                        first = upstream._flow_rhs(torch, cpu_matrix, cpu_field, reference,
                            kappa=kappa, mobility=1., route_strength=route, gamma=gamma)
                        if integrator == 'heun':
                            pred = upstream._wrap_angles(torch, reference + .05 * first)
                            second = upstream._flow_rhs(torch, cpu_matrix, cpu_field, pred,
                                kappa=next_kappa, mobility=1., route_strength=route, gamma=gamma)
                            reference = upstream._wrap_angles(torch, reference + .025 * (first + second))
                        else:
                            reference = upstream._wrap_angles(torch, reference + .05 * first)
                        workspace.advance(kappa, next_kappa)
                difference = (theta.cpu().numpy() - reference.numpy() + numpy.pi) % (2 * numpy.pi) - numpy.pi
                error = float(numpy.abs(difference).max())
                tolerance = 3e-5 if dtype == torch.float32 else 1e-12
                if error > tolerance:
                    raise AssertionError(f'Upstream trajectory mismatch: {dtype}, {integrator}, {route}, {gamma}: {error}')
                rows.append(dict(dtype=str(dtype), integrator=integrator, route_strength=route,
                                 gamma=gamma, max_angle_error=error, tolerance=tolerance))
    return rows


def measure(call, devices, repeats, warmups, problems, profilePath=None):
    def sync():
        for device in devices:
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)

    sync()
    started = perf_counter()
    call()
    sync()
    cold = perf_counter() - started
    for _ in range(warmups):
        call()
    sync()
    for device in devices:
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        sync()
        started = perf_counter()
        results = call()
        sync()
        times.append(perf_counter() - started)
    signatures = []
    for problem, result in zip(problems, results):
        sample, energy = result.sample, result.energy
        assert len(sample) == problem.variableCount
        assert all(v in (0, 1) for v in sample)
        assert all(sum(sample[i] for i in group) == 1 for group in problem.iterOneHotGroups())
        numpy.testing.assert_allclose(energy, problem.energy(sample), atol=1e-12, rtol=0)
        signatures.append({'energy': energy, 'sample_sha256': hashlib.sha256(bytes(sample)).hexdigest()})
    peak = {d: torch.cuda.max_memory_allocated(d) for d in devices if d.startswith('cuda')}
    profile_text = None
    if profilePath:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if any(d.startswith('cuda') for d in devices):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, profile_memory=True) as prof:
            call()
            sync()
        profilePath.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(profilePath))
        profile_text = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=15)
    return dict(cold_seconds=cold, seconds=times, median_seconds=median(times),
                peak_cuda_allocated_bytes=peak, results=signatures, profile=profile_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--variables', nargs='+', type=int, default=[64, 1024, 8192])
    parser.add_argument('--problems', type=int, default=1)
    parser.add_argument('--degree', type=int, default=8)
    parser.add_argument('--one-hot', action='store_true')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--runs', type=int, default=16)
    parser.add_argument('--run-batch-size', type=int, default=16)
    parser.add_argument('--candidate-interval', type=int, default=25)
    parser.add_argument('--candidate-batch-size', type=int, default=128)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--integrator', choices=['euler', 'heun'], default='euler')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--upstream-root', type=Path)
    parser.add_argument('--profile-directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.variables, args.problems, args.degree, args.steps, args.runs,
           args.run_batch_size, args.candidate_interval, args.candidate_batch_size, args.repeats) < 1 or args.warmups < 0:
        parser.error('sizes and repeats must be positive; warmups must be nonnegative')
    torch.set_num_threads(1)
    devices = args.devices or [args.device]
    single = TorchTransverseRouteBQMSolver(device=devices[0])
    p = dict(steps=args.steps, runs=args.runs, run_batch_size=args.run_batch_size,
             candidate_interval=args.candidate_interval, candidate_batch_size=args.candidate_batch_size,
             dtype=args.dtype, integrator=args.integrator, seed=13)
    output = {'environment': {'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'numpy': numpy.__version__, 'devices': devices,
        'hardware': [torch.cuda.get_device_name(d) if d.startswith('cuda') else platform.processor() for d in devices],
        'source_sha256': hashlib.sha256(Path(inspect.getfile(TorchTransverseRouteBQMSolver)).read_bytes()).hexdigest(),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
        'parameters': single._getParameters(p), 'measurements': []}
    reference = None
    if args.upstream_root:
        sys.path.insert(0, str(args.upstream_root.resolve()))
        from solvers import transverse_route as upstream
        output['environment']['upstream_sha256'] = hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()
        output['parity'] = parity(upstream, devices[0])
        reference = referenceClass(upstream)(devices[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n in args.variables:
        problems = makeProblems(n, args.problems, args.degree, args.one_hot)
        variants = [('optimized_eager', single, {**p, 'cuda_graph': False})]
        if reference:
            variants.insert(0, ('matched_upstream_flow', reference,
                               {**p, 'cuda_graph': False, 'deduplicate_candidates': False}))
        if devices[0].startswith('cuda'):
            variants.append(('optimized_graph', single, {**p, 'cuda_graph': True}))
        if len(devices) > 1:
            variants.append(('multi_gpu_graph', TorchTransverseRouteBQMSolver(devices=devices), {**p, 'cuda_graph': True}))
        for name, solver, parameters in variants:
            trace = args.profile_directory / f'{name}_{n}.json' if args.profile_directory else None
            row = measure(lambda: solver.solveMany(problems, parameters), solver.devices,
                          args.repeats, args.warmups, problems, trace)
            row.update(variant=name, variables=n, problems=len(problems), interactions=[q.interactionCount for q in problems])
            output['measurements'].append(row)
            args.output.write_text(json.dumps(output, indent=2) + '\n')
            print(json.dumps({k: row[k] for k in ('variant', 'variables', 'problems', 'median_seconds')}), flush=True)


if __name__ == '__main__':
    main()

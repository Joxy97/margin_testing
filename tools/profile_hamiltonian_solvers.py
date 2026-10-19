"""Reproducible bounded end-to-end probes for the 23 Hamiltonian candidates.

Example (write generated results outside the checkout)::

    python tools/profile_hamiltonian_solvers.py --device cuda:0 --sizes 32,128 \
        --steps 2 --runs 4 --repeat 3 --output /tmp/hamiltonian-probe.json

One warmup precedes independently seeded measured solves. Timings include input
transfer/preparation, search, readout, and solver candidate capture. Independent
CPU rescoring and cleanup are excluded. CUDA event spans include launch gaps;
they are not a sum of kernel durations. These short synthetic probes establish
bounded execution, scoring correctness and memory use, not convergence or a
time-to-target advantage. CPU comparison checks the same output contract;
CPU/GPU random streams and returned strings need not agree.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

# Support an uninstalled checkout without changing installed package behavior.
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))
from qubo_solvers import create_solver  # noqa: E402
from qubo_solvers.backends.problem import QUBOProblem  # noqa: E402
from qubo_solvers.hamiltonians import HAMILTONIAN_CLASSES, HAMILTONIAN_SOLVERS  # noqa: E402


def fixture(n, kind, seed):
    """Integer coefficients make independent FP64 score comparisons exact."""
    rng = np.random.default_rng(seed + n + (1_000_003 if kind == 'dense' else 0))
    linear = rng.integers(-5, 6, n).astype(np.float64)
    maximum = n * (n - 1) // 2
    if kind == 'dense' or maximum <= 4 * n:
        heads, tails = np.triu_indices(n, k=1)
    else:
        target = min(4 * n, maximum)
        pairs = np.empty((0, 2), dtype=np.int64)
        while len(pairs) < target:
            trial = np.sort(rng.integers(0, n, (2 * target, 2)), axis=1)
            trial = trial[trial[:, 0] != trial[:, 1]]
            pairs = np.unique(np.concatenate((pairs, trial)), axis=0)
        pairs = pairs[rng.permutation(len(pairs))[:target]]
        heads, tails = pairs.T
    weights = rng.integers(-5, 5, len(heads))
    weights += weights >= 0
    return QUBOProblem(linear, heads.astype(np.uint32), tails.astype(np.uint32),
                       weights.astype(np.float64), offset=7.)


def _digest(source):
    result = hashlib.sha256()
    for array in (source.linear, source.quadraticHeads, source.quadraticTails,
                  source.quadraticBiases, np.asarray(source.offset, dtype=np.float64)):
        result.update(np.asarray(array).tobytes())
    return result.hexdigest()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _timed(operation, device):
    synchronize(device)
    if device.type == 'cuda':
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    if device.type == 'cuda':
        first.record()
    result = operation()
    if device.type == 'cuda':
        last.record()
    synchronize(device)
    return result, time.perf_counter() - started, (first.elapsed_time(last) if device.type == 'cuda' else None)


def verify(source, result, runs):
    """Rescore original pair-once coefficients using an independent CPU loop."""
    errors, best = [], []
    for label, assignments, energies in (
            ('best', result.best_assignments, result.best_energies),
            ('final', result.final_assignments, result.final_energies)):
        if assignments is None or energies is None:
            raise AssertionError(f'{label} outputs missing in full-result mode')
        bits, reported = assignments.detach().cpu().numpy(), energies.detach().cpu().numpy()
        if bits.shape != (runs, source.variableCount) or reported.shape != (runs,):
            raise AssertionError(f'invalid {label} output shapes: {bits.shape}, {reported.shape}')
        if not np.isin(bits, (0, 1)).all() or not np.isfinite(reported).all():
            raise AssertionError(f'invalid {label} bits or nonfinite energies')
        independent = np.asarray([source.energy(row) for row in bits], dtype=np.float64)
        discrepancy = np.abs(independent - reported)
        if not np.array_equal(independent, reported):
            raise AssertionError(f'{label} source energy mismatch; maximum error {discrepancy.max()}')
        errors.extend(discrepancy.tolist())
        if label == 'best':
            best = independent.tolist()
    return dict(passed=True, maximum_absolute_error=max(errors, default=0.),
                returned_best_energies=best, checked_candidates=2 * runs,
                expected_assignment_shape=[runs, source.variableCount],
                scoring='Independent CPU original compact coefficients; exact integer fixture comparison')


def _parameters(name, steps, dtype, compile_xy=False):
    available = {field.name for field in fields(HAMILTONIAN_CLASSES[name])}
    overrides = dict(max_steps=steps, dynamics_dtype=dtype)
    # Reduce only the declared smoke configuration, never the class defaults.
    for option in ('polish_sweeps', 'polish_steps', 'terminal_steps'):
        if option in available:
            overrides[option] = 0
    if name == 'lib_quantum_dimer':
        overrides['slices'] = 8
    if name == 'lib_xy_coordinate' and compile_xy:
        overrides['compile_coordinates'] = True
    return overrides


def _memory_limit(device, explicit_mib):
    if explicit_mib is not None:
        return int(explicit_mib * 2**20)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        reclaimable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return max(1, int(.8 * (free + reclaimable)))
    return 4 * 2**30


@torch.no_grad()
def probe(source, name, args, device):
    overrides = _parameters(name, args.steps, args.dtype, args.compile_xy)
    solver = create_solver(name, **overrides)
    estimate = solver.estimate_compact_memory(source, precision=args.dtype, restarts=args.runs,
                                             batch_size=args.runs, best_only=False)
    limit = _memory_limit(device, args.memory_limit_mib)
    output = dict(device=str(device), solver=name, solver_class=type(solver).__name__,
                  parameter_overrides=overrides, effective_parameters=asdict(solver),
                  runs=args.runs, batch_size=args.runs, repeat=args.repeat,
                  memory_admission=dict(estimated_problem_bytes=estimate.problem_bytes,
                      estimated_output_bytes=estimate.output_bytes,
                      estimated_workspace_bytes=estimate.workspace_bytes,
                      estimated_total_bytes=estimate.total_bytes, limit_bytes=limit,
                      policy='Explicit cap' if args.memory_limit_mib else
                             '80% currently available/reclaimable CUDA bytes' if device.type == 'cuda' else
                             'Default CPU tensor estimate cap of 4 GiB'), measurements=[])
    if estimate.total_bytes > limit:
        output.update(status='memory_guard', reason=f'estimate {estimate.total_bytes} exceeds cap {limit}')
        return output

    def operation(seed):
        return solver.solve_compact(source, device=device, precision=args.dtype,
            restarts=args.runs, batch_size=args.runs, best_only=False, seed=seed,
            memory_limit_bytes=limit)

    warmup, seconds, span = _timed(lambda: operation(args.seed), device)
    output['warmup'] = dict(wall_seconds=seconds, cuda_event_span_ms=span,
                            verification=verify(source, warmup, args.runs))
    del warmup
    gc.collect()
    synchronize(device)
    for repeat in range(args.repeat):
        seed = args.seed + repeat + 1
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
        result, seconds, span = _timed(lambda: operation(seed), device)
        row = dict(seed=seed, end_to_end_wall_seconds=seconds, cuda_event_span_ms=span,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
                   baseline_allocated_bytes=baseline if device.type == 'cuda' else None,
                   baseline_reserved_bytes=reserved if device.type == 'cuda' else None,
                   verification=verify(source, result, args.runs))
        del result
        gc.collect()
        synchronize(device)
        row['after_cleanup_allocated_bytes'] = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
        output['measurements'].append(row)
    times = [row['end_to_end_wall_seconds'] for row in output['measurements']]
    energies = [value for row in output['measurements']
                for value in row['verification']['returned_best_energies']]
    output.update(status='passed', median_end_to_end_wall_seconds=statistics.median(times),
                  minimum_end_to_end_wall_seconds=min(times), mean_best_energy=statistics.mean(energies),
                  minimum_best_energy=min(energies), scoring_pass=True,
                  reported_score_dtype='torch.float64',
                  energy_note='Raw original-scale feasible energies; no optimum or convergence claim')
    return output


def _save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sizes', default='32,128')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--repeat', '--repeats', type=int, default=3)
    parser.add_argument('--solvers', help='Comma-separated canonical IDs; default all 23 Hamiltonian solvers')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--cpu-comparison', action='store_true')
    parser.add_argument('--compile-xy', action='store_true',
                        help='Enable the optional fused torch.compile XY coordinate kernel')
    parser.add_argument('--memory-limit-mib', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        sizes = list(dict.fromkeys(int(value.strip()) for value in args.sizes.split(',')))
    except ValueError:
        parser.error('--sizes must contain comma-separated positive integers')
    if not sizes or min(sizes) < 1 or args.steps < 0 or args.runs < 1 or args.repeat < 1 or args.seed < 0:
        parser.error('sizes/runs/repeat must be positive; steps/seed must be nonnegative')
    if args.memory_limit_mib is not None and (not np.isfinite(args.memory_limit_mib) or args.memory_limit_mib <= 0):
        parser.error('--memory-limit-mib must be finite and positive')
    selected = (list(dict.fromkeys(value.strip() for value in args.solvers.split(',')))
                if args.solvers else list(HAMILTONIAN_SOLVERS))
    if set(selected) - set(HAMILTONIAN_SOLVERS):
        parser.error(f'unknown Hamiltonian solver(s): {sorted(set(selected) - set(HAMILTONIAN_SOLVERS))}')
    output = args.output.expanduser().resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        parser.error('--output must be outside the checkout to keep generated results out of the repository')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('--device must select cpu or cuda')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable; no CPU fallback')
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda,
        device=str(device), requested_dynamics_dtype=args.dtype, independent_score_dtype='float64',
        cpu_threads=torch.get_num_threads(), warmup_solves_per_case=1,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        scope='Bounded synthetic implementation profile; not a quality campaign or convergence claim',
        timing_scope='Synchronized complete solve_compact call, including input transfer and preparation; independent verification/cleanup excluded',
        cpu_gpu_comparison='Output domain/shape/source-score contract only; different random streams are allowed',
        memory_scope='PyTorch allocated/reserved tensor bytes; excludes CUDA context and external library allocations',
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (REPOSITORY / 'src' / 'qubo_solvers').glob('hamiltonian*.py')})
    metadata['source_sha256']['_population.py'] = hashlib.sha256(
        (REPOSITORY / 'src' / 'qubo_solvers' / '_population.py').read_bytes()).hexdigest()
    metadata['source_sha256'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        metadata.update(cuda_device_count=torch.cuda.device_count(),
                        cuda_total_memory_bytes=properties.total_memory,
                        cuda_compute_capability=list(torch.cuda.get_device_capability(device)))
    report = dict(metadata=metadata, arguments={**vars(args), 'output': str(output)}, cases=[])
    failed = 0
    for n in sizes:
        for kind in ('sparse', 'dense'):
            source = fixture(n, kind, args.seed)
            for name in selected:
                row = dict(n=n, kind=kind, interactions=source.interactionCount,
                           actual_pair_density=(2 * source.interactionCount / (n * (n - 1)) if n > 1 else 0.),
                           source_sha256=_digest(source))
                try:
                    row.update(probe(source, name, args, device))
                    if args.cpu_comparison and device.type != 'cpu':
                        row['cpu_comparison'] = probe(source, name, args, torch.device('cpu'))
                except (MemoryError, torch.OutOfMemoryError) as error:
                    row.update(solver=name, status='allocation_failed', error=str(error))
                    failed += 1
                except Exception as error:
                    row.update(solver=name, status='failed', error_type=type(error).__name__, error=str(error))
                    failed += 1
                report['cases'].append(row)
                report['summary'] = {status: sum(case.get('status') == status for case in report['cases'])
                                     for status in ('passed', 'memory_guard', 'allocation_failed', 'failed')}
                _save(output, report)
                print(json.dumps(dict(solver=name, n=n, kind=kind, status=row['status'],
                    median_seconds=row.get('median_end_to_end_wall_seconds'),
                    mean_best_energy=row.get('mean_best_energy'))), flush=True)
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

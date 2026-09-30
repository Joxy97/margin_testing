"""Offline CPU/CUDA characterization of the 17 native tensor solvers.

Example: python tools/profile_solver_gpu.py --device cuda:0 --solvers all
             --instances gka1e bqp500-1 bqp1000-1 --output gpu-profile.json

This is a bounded profiling tool, not the full benchmark campaign. CUDA is
explicit, synchronized, and never replaced with CPU when unavailable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))


def synchronize(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def scalar_count(profile):
    return sum(event.count for event in profile.key_averages()
               if event.key == 'aten::_local_scalar_dense')


def selection_probe(torch, device, dtype):
    """Compare the prior indexing mechanism with the actual best-only reducer."""
    from qubo_solvers import QUBO
    from qubo_solvers.solvers import _Output, _Run

    problem = QUBO(torch.eye(4, device=device, dtype=dtype))
    run = _Run(problem, 3, None, torch.Generator(device=device).manual_seed(1),
               torch.Generator().manual_seed(1))
    output = _Output(problem, 3, True, None, 2)
    # Five scalar-indexing operations used by the previous reducer. This only
    # reproduces the indexing mechanism; no second solver implementation exists.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as before:
        index = run.energies.argmin()
        for tensor in (run.energies, run.best_x, run.energies, run.iterations, run.reasons):
            tensor[index]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as after:
        output.finish_batch(run, 0, 2)
        result = output.result()
        result.best_assignment
        result.best_restart_index
    synchronize(torch, device)
    return dict(prior_scalar_indexing_mechanism=scalar_count(before),
                current_reduction_and_result_access=scalar_count(after),
                meaning='aten::_local_scalar_dense operator count; this measures host scalar extraction, not GPU speedup')


def tensor_bytes(torch, result):
    tensors = [getattr(result, field.name) for field in fields(result)
               if isinstance(getattr(result, field.name), torch.Tensor)]
    return dict(logical_bytes=sum(t.numel() * t.element_size() for t in tensors),
                owned_storage_bytes=sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                                         for t in tensors}.values()))


def cpu_memory():
    try:
        import psutil
    except ImportError:
        return None
    information = psutil.Process().memory_info()
    return dict(rss_bytes=information.rss,
                lifetime_peak_working_set_bytes=getattr(information, 'peak_wset', None),
                note='Process-level readings, not isolated live tensor peaks; allocator caches and earlier work contribute')


def characterize(torch, source, cpu_problem, solver, device, options, warmups, repeats, profile_ops):
    resident_problem = cpu_problem.to(device=device)
    for _ in range(warmups):
        solver.solve(resident_problem, **options)
    synchronize(torch, device)
    baseline_cuda = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        baseline_cuda = torch.cuda.memory_allocated(device)
    memory_before = cpu_memory()
    resident_times, end_to_end_times, objectives = [], [], []
    measured_storage = None
    for _ in range(repeats):
        synchronize(torch, device)
        started = time.perf_counter()
        result = solver.solve(resident_problem, **options)
        synchronize(torch, device)
        resident_times.append(time.perf_counter() - started)
        sample = result.best_assignment.detach().cpu().numpy()
        objective = source.score(sample)
        if objective != float(result.best_energy.detach().cpu()):
            raise ValueError('Independent source scoring disagrees with the native solver energy')
        objectives.append(objective)
        measured_storage = tensor_bytes(torch, result)
        del result

        synchronize(torch, device)
        started = time.perf_counter()
        transferred = cpu_problem.to(device=device)
        result = solver.solve(transferred, **options).to(device='cpu')
        synchronize(torch, device)
        end_to_end_times.append(time.perf_counter() - started)
        if source.score(result.best_assignment.numpy()) != float(result.best_energy):
            raise ValueError('Independent scoring disagrees after result transfer')
        del result, transferred
    memory_after = cpu_memory()
    gpu_memory = None
    if device.type == 'cuda':
        gpu_memory = dict(baseline_allocated_bytes=baseline_cuda,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          note='Peak spans both resident and end-to-end trials; includes resident problem storage')
    record = dict(resident_seconds=resident_times, end_to_end_seconds=end_to_end_times,
                  resident_median_seconds=statistics.median(resident_times),
                  end_to_end_median_seconds=statistics.median(end_to_end_times),
                  independently_scored_objectives=objectives, result_storage=measured_storage,
                  cpu_memory_before=memory_before, cpu_memory_after=memory_after,
                  cuda_memory=gpu_memory)
    if profile_ops:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
            solver.solve(resident_problem, **options)
            synchronize(torch, device)
        record['operator_profile'] = [dict(name=event.key, count=event.count,
                                          self_cpu_time_us=event.self_cpu_time_total,
                                          self_cpu_memory_bytes=event.self_cpu_memory_usage)
                                      for event in profile.key_averages()]
        record['operator_memory_note'] = 'Operator allocation/deallocation traffic is not peak live memory'
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True, help='cpu or cuda[:index]; no automatic fallback')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--solvers', nargs='+', help='Native canonical solver names, or all; default greedy local search')
    parser.add_argument('--instances', nargs='+', default=['gka1e'])
    parser.add_argument('--data', type=Path, default=ROOT / 'benchmark_data/qubo37')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--restarts', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--full-result', action='store_true', help='Retain every restart instead of best-only')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--profile-ops', action='store_true', help='Run one additional untimed operator/memory profile')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    from qubo_solvers import QUBO, LIBRARY_SOLVERS, create_solver, default_algorithm_parameters, library_solver_class

    if args.device != 'cpu' and not (args.device == 'cuda' or
                                     args.device.startswith('cuda:') and args.device[5:].isdigit()):
        parser.error('device must be cpu or cuda[:index]')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA was requested but is unavailable; CPU fallback is disabled')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            parser.error(f'CUDA device {device.index} does not exist')
    if min(args.restarts, args.batch_size, args.repeats, args.cpu_threads) < 1 or min(args.steps, args.warmups) < 0:
        parser.error('steps/warmups must be nonnegative; restarts/batch-size/repeats/threads must be positive')
    if args.output.exists():
        parser.error(f'Refusing to overwrite an existing profile: {args.output}')
    torch.set_num_threads(args.cpu_threads)
    names = args.solvers or [name for name in LIBRARY_SOLVERS
                             if library_solver_class(name).__name__ == 'GreedyLocalSearch']
    if names == ['all']:
        names = list(LIBRARY_SOLVERS)
    if len(set(names)) != len(names) or any(name not in LIBRARY_SOLVERS for name in names):
        parser.error('Choose distinct native tensor solvers: ' + ', '.join(LIBRARY_SOLVERS))

    # Optional benchmark dependencies load only when the offline input is used.
    from qubo_benchmark.catalog import loadCatalog, selectEntries
    from qubo_benchmark.pipeline import loadPrepared

    options = dict(restarts=args.restarts, batch_size=args.batch_size,
                   best_only=not args.full_result, seed=31)
    records = []
    for entry in selectEntries(loadCatalog(args.data), args.instances):
        source, metadata = loadPrepared(entry, args.data)
        cpu_problem = QUBO(torch.tensor(source.dense(), dtype=getattr(torch, args.dtype)), source.offset)
        for name in names:
            solver = create_solver(name, **default_algorithm_parameters(name, args.steps))
            record = characterize(torch, source, cpu_problem, solver, device, options,
                                  args.warmups, args.repeats, args.profile_ops)
            record.update(instance=entry['instance_id'], n=source.n,
                          normalized_sha256=metadata['normalized_sha256'], solver=name,
                          algorithm_parameters=asdict(solver), solve_parameters=options,
                          algorithm_sha256=hashlib.sha256(Path(inspect.getfile(type(solver))).read_bytes()).hexdigest())
            records.append(record)
            print(f'{name} {entry["instance_id"]}: resident={record["resident_median_seconds"]:.6f}s '
                  f'end_to_end={record["end_to_end_median_seconds"]:.6f}s', flush=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], device=str(device), dtype=args.dtype,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  platform=platform.platform(), python=platform.python_version(),
                  cpu_threads=torch.get_num_threads(), warmups=args.warmups, repeats=args.repeats,
                  method='Independent source scoring outside timing. Resident timing excludes transfers; end-to-end includes problem and result transfer. CUDA synchronized before/after each timed region. Optional operator profiling is separate from timings.',
                  scope='17 native tensor algorithms; backend-specific solvers use the main benchmark runner',
                  selection_scalar_extraction_probe=selection_probe(torch, device, getattr(torch, args.dtype)),
                  gpu_validation='measured on requested CUDA device' if device.type == 'cuda' else 'pending; CPU characterization does not establish GPU speedup',
                  records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Bounded synchronized correctness/memory probe, not a quality campaign.

Example: python tools/profile_new_physics_solvers.py --device cuda:0 --output probe.json
Output separates one-time preparation from resident search. CUDA event spans
include launch gaps; they are not the sum of kernel durations. No known optima
or reference assignments enter the search. Scientific Agent Skills influenced
the device-resident and measured-memory workflow: doi:10.48550/arXiv.2609.00065.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from qubo_solvers import GravityTunneling, SU2ChiralIsospin
from qubo_solvers._population import PopulationObjective
from qubo_solvers.backends.problem import QUBOProblem


def fixture(n, density):
    rng = np.random.default_rng(20261014 + n)
    linear = rng.integers(-5, 6, n).astype(np.float64)
    if density == 'dense':
        heads, tails = np.triu_indices(n, k=1)
    else:
        heads = rng.integers(0, n, 4 * n)
        tails = rng.integers(0, n, 4 * n)
    biases = rng.integers(-5, 6, len(heads)).astype(np.float64)
    return QUBOProblem(linear, heads.astype(np.uint32), tails.astype(np.uint32), biases)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed(call, device):
    synchronize(device)
    start = time.perf_counter()
    if device.type == 'cuda':
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record()
    value = call()
    if device.type == 'cuda':
        last.record()
    synchronize(device)
    elapsed = time.perf_counter() - start
    return value, elapsed, first.elapsed_time(last) if device.type == 'cuda' else None


def verify(source, result):
    errors = []
    for assignments, energies in ((result.best_assignments, result.best_energies),
                                  (result.final_assignments, result.final_energies)):
        if assignments is None:
            continue
        for bits, reported in zip(assignments.cpu().numpy(), energies.cpu().numpy()):
            raw = source.energy(bits)
            error = abs(raw - float(reported))
            errors.append(error)
            # These bounded integer fixtures have exactly representable scores.
            if error != 0:
                raise AssertionError(f'original-source scoring mismatch: {raw} != {reported}')
    return max(errors, default=0.)


@torch.no_grad()
def probe(source, solver, device, runs, repeats):
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
    objective, prepare, prepare_span = timed(lambda: PopulationObjective.from_compact(
        source, device=device, precision=torch.float32, normalize=solver.normalize,
        threshold=solver.sparse_threshold), device)
    prepared = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
    preparation_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    operation = lambda: solver._execute(objective, runs, runs, None, 42, True, None)
    warmup = operation()
    verify(source, warmup)
    del warmup
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations, event_spans, residuals, score_errors, best = [], [], [], [], []
    for _ in range(repeats):
        result, duration, span = timed(operation, device)
        durations.append(duration)
        event_spans.append(span)
        score_errors.append(verify(source, result))
        best.append(float(result.best_energies.min().cpu()))
        del result
        gc.collect()
        synchronize(device)
        residuals.append(torch.cuda.memory_allocated(device) if device.type == 'cuda' else None)
    search_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    estimate = solver.estimate_compact_memory(source, restarts=runs, batch_size=runs, best_only=True)
    output = dict(solver=type(solver).__name__, n=source.variableCount,
        interactions=source.interactionCount, runs=runs, batch_size=runs,
        steps=solver.max_steps, candidate_interval=solver.candidate_interval,
        dynamics_dtype=str(objective.dtype), source_score_dtype='float64',
        force_layout=str(objective.matrix.layout), preparation_wall_seconds=prepare,
        preparation_cuda_event_span_ms=prepare_span,
        resident_search_wall_seconds=durations, resident_search_cuda_event_span_ms=event_spans,
        source_score_max_absolute_error=max(score_errors), best_energies=best,
        initial_allocated_bytes=before, prepared_allocated_bytes=prepared,
        preparation_peak_allocated_bytes=preparation_peak, search_peak_allocated_bytes=search_peak,
        peak_reserved_bytes=reserved, post_search_allocated_bytes=residuals,
        estimated_total_bytes=estimate.total_bytes)
    if device.type == 'cuda' and len(set(residuals)) != 1:
        raise AssertionError(f'live tensor memory grew across repeated solves: {residuals}')
    del objective
    gc.collect()
    synchronize(device)
    output['after_cleanup_allocated_bytes'] = (torch.cuda.memory_allocated(device)
                                              if device.type == 'cuda' else None)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sizes', default='256,1000,5000')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--runs', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-10000', action='store_true')
    parser.add_argument('--cpu-comparison', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    report = dict(python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, device=str(device), warmup_searches=1,
        hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        memory_note='Tensor allocator only; reserved blocks and CUDA context are not leaks.',
        scope='Synthetic correctness/scaling smoke; not 100-seed TTS or quality evidence.', cases=[])
    repository = Path(__file__).resolve().parents[1]
    report['source_sha256'] = {name: hashlib.sha256(
        (repository / 'src' / 'qubo_solvers' / name).read_bytes()).hexdigest()
        for name in ('_population.py', 'su2_chiral_isospin.py', 'gravity_tunneling.py')}
    sizes = [int(value) for value in args.sizes.split(',')]
    cases = [(n, density) for n in sizes for density in ('sparse', 'dense')]
    if args.include_10000:
        cases.append((10000, 'sparse'))
    for n, density in cases:
        source = fixture(n, density)
        for cls in (SU2ChiralIsospin, GravityTunneling):
            solver = cls(max_steps=args.steps, dynamics_dtype='float32')
            row = probe(source, solver, device, args.runs, args.repeats)
            row['density'] = density
            if args.cpu_comparison and device.type != 'cpu':
                row['cpu_comparison'] = probe(source, solver, torch.device('cpu'), args.runs, args.repeats)
                row['cpu_to_gpu_search_speedup'] = (
                    np.median(row['cpu_comparison']['resident_search_wall_seconds']) /
                    np.median(row['resident_search_wall_seconds']))
            report['cases'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(dict(solver=row['solver'], n=n, density=density,
                search_seconds=float(np.median(row['resident_search_wall_seconds'])),
                peak_MiB=None if row['search_peak_allocated_bytes'] is None else
                row['search_peak_allocated_bytes'] / 2**20)), flush=True)


if __name__ == '__main__':
    main()

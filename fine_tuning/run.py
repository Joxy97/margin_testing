"""Replay one selected dense-gka5e preset with explicit seeds on one CUDA GPU.

The 20-second limit is a certified-quality deadline. GPU work and return are
cooperative; late candidates receive no quality credit. No library default is
changed. Use --dry-run to verify the preset, source and input without CUDA work.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'fine_tuning'
sys.path.insert(0, str(ROOT / 'src'))
CHECKPOINTS = [.05, .1, .2, .5, 1., 2., 5., 10., 20.]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_input(path):
    # Reuse the recovery tool's compression-independent, ordered numeric digest.
    spec = importlib.util.spec_from_file_location('_replay_input_fingerprint', ROOT/'tools/stage_continuous_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.canonical_fingerprint(path)


def validate_bundle(input_path, solver, seeds):
    settings, protocol, manifest = (read(BUNDLE/name) for name in
        ('selected_settings.json', 'protocol.json', 'source_manifest.json'))
    require(manifest['selected_settings_sha256'] == sha(BUNDLE/'selected_settings.json'), 'Selected settings checksum differs.')
    require(digest(manifest['source_files']) == manifest['source_sha256'] == settings['source_sha256'],
            'Selected settings and numerical source identity differ.')
    for name, expected in manifest['source_files'].items():
        path = (ROOT/name).resolve()
        require(name.startswith('src/') and path.is_relative_to((ROOT/'src').resolve())
                and path.is_file() and sha(path) == expected, 'Numerical source differs: '+name)
    require(settings['status'] == 'validated_complete' and not settings['pending_eligible_solvers'], 'Presets are incomplete.')
    require(len(settings['solvers']) == 144 and len(settings['excluded_unchanged']) == 14
            and set(settings['schedule_by_solver']) == set(settings['solvers']), 'Incomplete preset inventory.')
    require(solver in settings['solvers'] and solver not in settings['excluded_unchanged'],
            'Choose one of the 130 tuned canonical solver IDs; excluded or untested solvers cannot use this replay CLI.')
    require(len(seeds) == len(set(seeds)) and 1 <= len(seeds) <= 100
            and all(0 <= seed < 2**32 for seed in seeds), 'Specify 1–100 distinct seeds in [0, 2**32).')
    problem = protocol['problem']
    require(problem['instance_id'] == settings['problem'] == 'gka5e'
            and problem['n_variables'] == settings['n_variables'] == 200
            and problem['reference'] == -35154. and problem['reference_type'] == 'OPTIMUM'
            and protocol['budget_seconds'] == settings['budget_seconds'] == 20.
            and protocol['checkpoints_seconds'] == settings['checkpoints_seconds'] == CHECKPOINTS,
            'The published dense-gka5e timing or objective protocol differs.')
    require(settings['schedule_by_solver'][solver] in ('wall', 'finite_step'), 'Unknown solver schedule.')
    require(settings['wall_schedule_solvers'] == sorted(s for s, mode in settings['schedule_by_solver'].items() if mode == 'wall'),
            'Per-solver schedules disagree with the wall-schedule inventory.')
    from qubo_benchmark.model import Problem
    source = Problem.load(input_path)
    require(source.n == 200, 'Expected the dense 200-variable gka5e input.')
    fingerprint, actual_sha = canonical_input(input_path), sha(input_path)
    require(fingerprint == problem['numeric_coo_sha256'], 'Input ordered numeric arrays differ from the tested gka5e problem.')
    require(problem['npz_sha256'] == settings['input_sha256'], 'Original input provenance differs.')
    return settings, protocol, dict(source_sha256=manifest['source_sha256'], selected_settings_sha256=manifest['selected_settings_sha256'],
        protocol_sha256=sha(BUNDLE/'protocol.json'), source_manifest_sha256=sha(BUNDLE/'source_manifest.json'),
        original_npz_sha256=problem['npz_sha256'], actual_npz_sha256=actual_sha,
        numeric_coo_sha256=fingerprint, original_npz_bytes_equal=actual_sha == problem['npz_sha256'])


def immediate_observer(base):
    """Same natural-return forwarding used by the frozen tuning runner."""
    class ImmediateFinishObserver(base):
        def finish(self):
            if self._optimum_stopped:
                value = self._optimum_energy
            else:
                self._checkpoint(self.elapsed())
                value = self.certified
                if hasattr(value, 'detach'):
                    value = float(value.detach().cpu())
                if value is None:
                    value = float('nan')
                self.invalid |= not math.isfinite(value) and not math.isnan(value)
            self.energies[self.cursor:] = value
            self.cursor = len(self.checkpoints)
            return self.energies.copy()
    return ImmediateFinishObserver


def make_job(settings, solver, seed, input_path, device):
    return dict(device=device, cpu_threads=1, solver=solver, parameters=copy.deepcopy(settings['solvers'][solver]),
        npz=str(input_path.resolve()), seed=seed, budget=20., checkpoints=list(CHECKPOINTS),
        wall_schedule=settings['schedule_by_solver'][solver] == 'wall', reference=-35154.,
        reference_type='OPTIMUM', tolerance=0., stop_on_optimum=True)


def execute(args, settings, protocol, evidence):
    import numpy
    import scipy
    import torch
    import qubo_solvers
    import qubo_benchmark
    import qubo_benchmark.runtime.continuous_worker as cw
    from qubo_benchmark.runtime.common import atomic
    from qubo_benchmark.runtime.worker import WarmWorker
    for module in (qubo_solvers, qubo_benchmark, cw):
        require(Path(module.__file__).resolve().is_relative_to((ROOT/'src').resolve()),
                'An installed package from a different checkout was imported.')
    require(torch.cuda.is_available(), 'CUDA is unavailable; this replay CLI does not fall back to CPU.')
    require(int(args.device.split(':')[1]) < torch.cuda.device_count(), 'Requested CUDA device is unavailable.')
    require(not args.output.exists(), 'Choose a new output directory; replay results are never overwritten.')
    worker = WarmWorker(dict(device=args.device, cpu_threads=1))
    props = torch.cuda.get_device_properties(args.device)
    hardware = dict(host=socket.gethostname(), device=args.device, name=props.name,
        uuid=str(getattr(props, 'uuid', 'unavailable')), total_memory=props.total_memory,
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        numpy=numpy.__version__, scipy=scipy.__version__, cpu_threads=1, tf32=False)
    identity = dict(**evidence, solver=args.solver, seeds=args.seeds, hardware=hardware,
        parameters=settings['solvers'][args.solver], schedule=settings['schedule_by_solver'][args.solver],
        budget_seconds=20., checkpoints_seconds=CHECKPOINTS, replay_cli_sha256=sha(Path(__file__)),
        note='New replay evidence on the recorded hardware; historical GPU identity is not required. Hardware changes may change timings and outcomes.')
    identity['digest'] = digest(identity)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic(args.output/'identity.json', identity)
    original_observer = cw.ScalarObserver
    cw.ScalarObserver = immediate_observer(original_observer)
    completed = 0
    try:
        for seed in args.seeds:
            job = make_job(settings, args.solver, seed, args.input, args.device)
            before = time.perf_counter()
            try:
                require(sha(args.input) == evidence['actual_npz_sha256'], 'Input file changed after validation.')
                setup = worker.prepare(job)
                version = worker.prepared.Q._version if worker.native else None
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                result = cw.run_continuous_trial(worker, job)
                torch.cuda.synchronize(args.device)
                require(not worker.native or worker.prepared.Q._version == version, 'Solver modified the immutable QUBO input.')
                final = result['energies'][-1]
                require(final is None or (math.isfinite(final) and final >= -35154.), 'Invalid energy below the proven optimum.')
                result.update(setup=setup, source_integer_scoring_exact=worker._continuous_scorer.exact_float32,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
            except Exception as error:
                result = dict(status='setup_error', error=f'{type(error).__name__}: {error}',
                              energies=[None]*len(CHECKPOINTS), actual_solve_wall_s=0.)
            result.update(solver=args.solver, seed=seed, parameters=job['parameters'], wall_schedule=job['wall_schedule'],
                identity=identity['digest'], device=args.device, caller_wall_seconds=time.perf_counter()-before,
                completed_at_utc=datetime.now(timezone.utc).isoformat(), actual_iterations=None,
                iteration_note='Configured ceilings are preserved; realized iterations are not exposed by the scalar runtime.')
            atomic(args.output/'trials'/f'{seed}.json', result)
            completed += 1
            progress = dict(completed=completed, requested=len(args.seeds), seed=seed, status=result['status'],
                            final_energy=result['energies'][-1], identity=identity['digest'])
            atomic(args.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
            if result['status'] != 'complete':
                return 1
        atomic(args.output/'completion.json', dict(status='complete', seeds=args.seeds, identity=identity['digest']))
        return 0
    finally:
        cw.ScalarObserver = original_observer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.seeds = args.seeds if args.seeds is not None else [args.seed]
    require(re.fullmatch(r'cuda:\d+', args.device), 'Use one explicit CUDA device, such as cuda:0.')
    settings, protocol, evidence = validate_bundle(args.input, args.solver, args.seeds)
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN_VERIFIED', solver=args.solver, seeds=args.seeds, device=args.device,
            schedule=settings['schedule_by_solver'][args.solver], parameters=settings['solvers'][args.solver],
            evidence=evidence, gpu_execution=False, output_created=False), indent=2, allow_nan=False))
        return 0
    return execute(args, settings, protocol, evidence)


if __name__ == '__main__':
    raise SystemExit(main())

"""Bounded deployment check: four one-second seeds, separate from production."""
from pathlib import Path
import argparse
import importlib.util
import json
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing deployment evidence will not be overwritten')
    sys.path.insert(0, str(args.repo / 'src'))
    import torch
    from qubo_benchmark.runtime.worker import WarmWorker
    from qubo_benchmark.runtime.continuous_worker import run_continuous_trial
    from qubo_benchmark.runtime.common import source_snapshot_hash
    spec = importlib.util.spec_from_file_location('c3_launch', args.repo / 'run_compendium_iii_dense.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    identity = launcher.checked_identity(args.campaign)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Expected all four CUDA devices')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = json.loads((args.campaign / 'solver_settings.json').read_text())
    solvers = identity['protocol']['solvers']
    chosen = [solvers[index] for index in (0, len(solvers)//3, 2*len(solvers)//3, len(solvers)-1)]
    report = dict(scope='Four one-second deployment smoke trials; not production or quality tuning.',
                  python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                  source_snapshot_sha256=source_snapshot_hash(), partition=identity['partition_id'],
                  trials=[], production_started=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for gpu, solver in enumerate(chosen):
        device = f'cuda:{gpu}'
        worker = WarmWorker(dict(device=device, cpu_threads=1))
        job = dict(device=device, cpu_threads=1, solver=solver, parameters=settings[solver],
                   npz=str(args.repo/'benchmark_data/continuous12/normalized/gka5e.npz'),
                   seed=91000+gpu, budget=1.0, checkpoints=[.05, .1, .2, .5, 1.0],
                   wall_schedule=False, reference=-35154, reference_type='OPTIMUM',
                   tolerance=0.0, stop_on_optimum=True)
        worker.prepare(job)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = run_continuous_trial(worker, job)
        torch.cuda.synchronize(device)
        result.update(solver=solver, device=device, gpu=torch.cuda.get_device_name(device),
                      seed=job['seed'], wall_s=time.perf_counter()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        report['trials'].append(result)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(solver=solver, device=device, status=result['status'],
                              final_energy=result.get('energies', [None])[-1])), flush=True)
        if result['status'] != 'complete':
            raise RuntimeError('Deployment smoke failed; retained evidence contains details')
        del worker
        torch.cuda.empty_cache()
    if (args.campaign / 'results' / 'manifest.json').exists():
        raise RuntimeError('Unexpected production manifest exists before user launch')
    print(json.dumps(dict(status='passed', cuda_smokes=4, production_started=False)), flush=True)


if __name__ == '__main__':
    main()

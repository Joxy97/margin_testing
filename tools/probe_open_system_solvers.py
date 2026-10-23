"""Very small, explicit open-system/SBM smoke: two n200 inputs, two seeds, 1s.

Uses the production continuous scorer and deadline machinery. Does not tune,
launch a full campaign or claim a statistically meaningful solver ranking.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense-input', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    import torch
    from qubo_solvers import create_bqm_solver
    from qubo_solvers._open_system_catalog import OPEN_SYSTEM_SOLVERS
    from qubo_benchmark.runtime.worker import WarmWorker
    from qubo_benchmark.runtime.continuous_worker import run_continuous_trial
    from qubo_benchmark.runtime.common import source_snapshot_hash
    frozen=json.loads((ROOT/'configs/benchmark_solvers_selected_20s.json').read_bytes())
    seeds=[7101,7102]; budget=1.
    report=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        device=args.device,gpu=torch.cuda.get_device_name(args.device),source_snapshot=source_snapshot_hash(),
        budget_seconds=budget,seeds=seeds,trials=[],scope='bounded smoke; no tuning or statistical ranking',
        setup_policy='context and immutable input/scorer setup excluded; RNG, search, readout and certification included',
        postprocessing_policy='each method retains its own recorded library defaults; no extra polish or restart',
        tf32=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    worker=WarmWorker(dict(device=args.device,cpu_threads=1))
    for label,path,reference in [('sparse',ROOT/'benchmark_data/qubo37/normalized/gka1e.npz',-16464),
                                 ('dense',args.dense_input,-35154)]:
        for solver in (*OPEN_SYSTEM_SOLVERS,'lib_simulated_bifurcation'):
            factory=create_bqm_solver(solver,{'device':args.device})
            config=(dict(frozen['solvers'][solver]) if solver=='lib_simulated_bifurcation' else
                    dict(runs=16,run_batch_size=16,best_only=True,dtype='float32',memory_limit_bytes=512*1024**2))
            parameters=factory._getParameters(config)
            for seed in seeds:
                job=dict(device=args.device,cpu_threads=1,solver=solver,parameters=parameters,npz=str(path),
                    seed=seed,budget=budget,checkpoints=[.05,.1,.2,.5,1.],wall_schedule=False,
                    reference=reference,reference_type='OPTIMUM',tolerance=0.,stop_on_optimum=True)
                setup=worker.prepare(job)
                torch.cuda.reset_peak_memory_stats(args.device)
                began=time.perf_counter()
                result=run_continuous_trial(worker,job)
                result.update(solver=solver,problem=label,seed=seed,parameters=parameters,setup=setup,
                    caller_wall_s=time.perf_counter()-began,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
                report['trials'].append(result)
                args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(label,solver,seed,result['status'],result['energies'][-1],flush=True)
    if any(row['status']!='complete' for row in report['trials']):
        raise SystemExit('One or more smoke trials failed; inspect the saved report.')


if __name__=='__main__': main()

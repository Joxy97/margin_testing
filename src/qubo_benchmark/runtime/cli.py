"""Public positional CLI; all matching instances and 100 seeds are defaults."""
import argparse
from pathlib import Path
from .common import CONFIG,read
from .selection import select,preflight,inventory,environment,worker_argument
from .engine import execute

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('n',type=int,nargs='?');p.add_argument('density',nargs='?')
    p.add_argument('time_limit',nargs='?')
    p.add_argument('--instances',choices=['all','representative'])
    p.add_argument('--instance-id');p.add_argument('--runs',type=int)
    p.add_argument('--seed-start',type=int);p.add_argument('--seeds-file')
    p.add_argument('--config');p.add_argument('--solvers')
    p.add_argument('--device','--devices',dest='device',
                   help='auto uses all visible GPUs (CPU fallback); cpu, cuda:N, or cuda:0,cuda:1')
    p.add_argument('--seed-workers',type=worker_argument,
                   help='Concurrent processes per device: auto (measured default) or 1..100')
    p.add_argument('--no-autotune',action='store_true',default=None,
                   help='With auto workers, keep one worker per device without calibration')
    p.add_argument('--max-auto-workers',type=int,
                   help='Upper bound for measured concurrency per device (default 4)')
    p.add_argument('--worker-mode',choices=['persistent','fresh'],help='Reuse warm workers (default) or spawn per trial')
    p.add_argument('--output-root',default='results');p.add_argument('--resume')
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--list-solvers',action='store_true');p.add_argument('--list-instances',action='store_true')
    p.add_argument('--require-gpu',action='store_true')
    return p

def resolve(args):
    if args.resume:
        root=Path(args.resume);saved=read(root/'experiment.json')['core']
        defaults=dict(n=saved['n'],density=saved['density'],time_limit=saved['budget_s'],
            instances='all' if saved['instance_mode']=='single' else saved['instance_mode'],
            instance_id=saved['instances'][0]['entry']['instance_id'] if saved['instance_mode']=='single' else None,
            runs=len(saved['seeds']),seed_start=0,seeds_file=str(root/'seeds.json'),
            config=str(root/'solver_configuration.json'),device=saved['requested_device'],
            solvers=','.join(saved['solvers']),
            seed_workers=saved['protocol']['execution'].get('seed_workers_requested',1),
            no_autotune=not saved['protocol']['execution'].get('autotune_enabled',False),
            max_auto_workers=saved['protocol']['execution'].get('auto_concurrency_max_per_device',4),
            worker_mode=saved['protocol']['execution'].get('worker_mode','fresh'))
    else:defaults=dict(instances='all',runs=100,seed_start=0,device='auto',solvers='all',
                       seed_workers='auto',worker_mode='persistent',no_autotune=False,max_auto_workers=4)
    for key,value in defaults.items():
        if getattr(args,key) is None:setattr(args,key,value)
    if args.retry_failed and not args.resume:raise ValueError('--retry-failed requires --resume')
    return args

def main(argv=None):
    p=parser()
    try:
        args=resolve(p.parse_args(argv))
        if args.list_instances:
            catalog=read(CONFIG/'qubo_benchmark_catalog_200_500_1000_v2.json')
            for entry in catalog['instances']:
                print(entry['instance_id'],entry['binary_variables'],entry['density_category'],
                      entry['normalized_reference_objective_min'],entry['reference_status'])
            return 0
        if args.list_solvers:
            config=read(args.config or CONFIG/'benchmark_solvers.json')
            env=environment(args.device,args.require_gpu)
            for row in inventory(config['solvers'],env['hardware']['device']):
                print(row['solver_id'],row['status'],row['reason'] or 'GPU-capable; timed-candidate availability is hardware dependent')
            return 0
        if None in (args.n,args.density,args.time_limit):p.error('Provide N DENSITY TIME_LIMIT')
        prepared=preflight(args)
        from .autotune import calibrate
        calibrate(prepared,args)
        for row in prepared['registry']:
            if row['status']=='unavailable':print(f"UNAVAILABLE: {row['solver_id']}: {row['reason']}",flush=True)
        print(f"Plan: {len(prepared['core']['solvers'])} solvers x {len(prepared['selected'])} instances x "
              f"{len(prepared['core']['seeds'])} seeds = {prepared['total']} trials; "
              f"devices={','.join(prepared['core']['actual_devices'])}; "
              f"{'STANDARD' if prepared['core']['standard'] else 'NONSTANDARD SMOKE'}",flush=True)
        print('Instances: '+', '.join(e['entry']['instance_id'] for e in prepared['selected']),flush=True)
        print('Solvers: '+', '.join(prepared['core']['solvers']),flush=True)
        print(f"Workers: {prepared['core']['protocol']['execution']['workers_per_device']} "
              f"{args.worker_mode}; total memory/CPU-admitted maximum: "
              f"{prepared['capacity']['maximum_workers_estimate']} (not a speedup guarantee)",flush=True)
        if args.seed_workers=='auto' and args.dry_run:
            print('Auto concurrency starts at one worker per device; bounded calibration is skipped by --dry-run.',flush=True)
        print('Timing mode: '+prepared['core']['protocol']['execution']['execution_mode']+
              '; budget covers solver reset, preprocessing, search and completed capture only',flush=True)
        if args.dry_run:return 0
        _,code=execute(prepared,args);return code
    except (ValueError,OSError,ImportError) as exc:
        p.exit(2,f'Preflight/runtime error: {exc}\n')

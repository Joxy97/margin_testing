"""Offline preflight, immutable settings and complete registry inventory."""
import importlib.metadata
import itertools
import math
import os
import platform
import socket
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from .common import ROOT,CONFIG,read,digest,filehash,source_identity
from ..catalog import DEFAULT_DATA,loadCatalog
from ..pipeline import loadPrepared,readSource

PROTOCOL=read(CONFIG/'benchmark_protocol.json')
REPRESENTATIVES=PROTOCOL['representatives']
BUDGETS=tuple(PROTOCOL['budgets_s'])

def budget(text):
    try:value=Decimal(str(text))
    except InvalidOperation:raise ValueError('Budget must be one of 0.05, 0.1, 0.5, 1, 5, 10 seconds')
    if value not in [Decimal(str(v)) for v in BUDGETS]:
        raise ValueError('Unsupported budget; choose 0.05, 0.1, 0.5, 1, 5 or 10 seconds')
    return float(value)

def seeds(runs=100,start=0,path=None):
    if isinstance(runs,bool) or runs<1:raise ValueError('runs must be positive')
    values=read(path) if path else list(range(start,start+runs))
    if not isinstance(values,list) or len(values)!=runs:
        raise ValueError('Seed file must be a JSON array of runs unique seeds')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 or v>=2**63-1 for v in values):
        raise ValueError('Seeds must be integers in [0,2**63-1)')
    if len(set(values))!=runs:raise ValueError('Seed file must contain unique seeds')
    return values

def select(n,density,mode='all',instance=None):
    if n not in PROTOCOL['variable_counts'] or density not in PROTOCOL['densities']:
        raise ValueError('N must be 200, 500 or 1000; density must be sparse or dense')
    entries=[e for e in read(CONFIG/'qubo_benchmark_catalog_200_500_1000_v2.json')['instances']
             if e['binary_variables']==n and e['density_category']==density]
    if instance:
        entries=[e for e in entries if e['instance_id']==instance]
    elif mode=='representative':
        entries=[e for e in entries if e['instance_id']==REPRESENTATIVES[f'{n}_{density}']]
    elif mode!='all':raise ValueError('Unknown instance mode')
    if not entries:raise ValueError('No matching catalog instance; check N/density/instance ID')
    return entries

def grid():return list(itertools.product(PROTOCOL['variable_counts'],PROTOCOL['densities'],BUDGETS))

def environment(requested='auto',require_gpu=False):
    import torch
    import psutil
    if requested=='auto':device='cuda:0' if torch.cuda.is_available() else 'cpu'
    else:device=requested
    if device!='cpu' and not device.startswith('cuda:'):raise ValueError('Use auto, cpu or indexed cuda:N')
    if require_gpu and not device.startswith('cuda'):raise ValueError('--require-gpu needs an available CUDA device')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable; no CPU fallback')
        index=int(device.split(':')[1])
        if index<0 or index>=torch.cuda.device_count():raise ValueError('CUDA device index unavailable')
        p=torch.cuda.get_device_properties(index)
        gpu=dict(name=p.name,uuid=str(getattr(p,'uuid','unavailable')),total_memory_bytes=p.total_memory)
    else:gpu=None
    driver=None
    if gpu:
        try:
            driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                                  capture_output=True,text=True,timeout=5).stdout.strip()
        except (OSError,subprocess.TimeoutExpired):pass
    packages={}
    for name in ('torch','numpy','scipy','psutil','threadpoolctl','dimod','dwave-samplers'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    import numpy as np
    try:blas=np.__config__.show(mode='dicts')
    except TypeError:blas='numpy configuration interface unavailable'
    hardware=dict(host=socket.gethostname(),cpu=platform.processor(),logical_cpus=os.cpu_count(),
                  physical_cpus=psutil.cpu_count(logical=False),host_ram_bytes=psutil.virtual_memory().total,
                  gpu=gpu,device=device,
                  cpu_affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(),'cpu_affinity') else None)
    return dict(hardware=hardware,hardware_id=digest(hardware),os=platform.platform(),python=platform.python_version(),
                packages=packages,torch_cuda=torch.version.cuda,driver=driver,blas=blas,
                cpu_threads=PROTOCOL['execution']['cpu_threads'],blas_thread_limit=PROTOCOL['execution']['cpu_threads'],tf32=False,
                environment_allowlist={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
                        'MKL_NUM_THREADS','NVIDIA_VISIBLE_DEVICES') if k in os.environ},
                container_image=None,container_image_reason='not available without an explicitly supplied deployment manifest')

def inventory(settings,device='cpu'):
    from qubo_solvers import SOLVERS,solver_capabilities,create_bqm_solver
    from ..adapters import GPU_SOLVERS
    rows=[]
    for name in SOLVERS:
        cap=solver_capabilities(name);eligible=name in GPU_SOLVERS
        row=dict(solver_id=name,version='0.1.0',adapter_version='runtime-v2',backend=cap['backend'],
                 supported_devices='cpu,cuda' if cap['gpu'] else 'cpu',supported_precision=cap['precisions'],
                 eligible=eligible,status='excluded',reason=cap['restriction'],
                 seed_support=name not in ('lib_planar_graph','lib_tree_decomposition_solver'),deadline_support='cooperative checkpoints plus process watchdog' if eligible else None,
                 history_support='completed binary checkpoints' if eligible else None,
                 timing_50ms='conditional: no in-budget score if setup/search/capture exceeds cutoff' if eligible else 'out of scope',
                 license_readiness='local open-source backend; no credentials')
        if eligible:
            try:
                if name not in settings:raise ValueError('Missing frozen solver configuration')
                obj=create_bqm_solver(name,{'device':device})
                resolved=obj._getParameters(settings[name]);resolved.pop('seed',None)
                if resolved!=settings[name]:raise ValueError('Configuration must contain all resolved defaults')
                row.update(status='runnable',reason=None,entry_point=f'{type(obj).__module__}.{type(obj).__name__}')
            except Exception as exc:row.update(status='unavailable',reason=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    return rows

def preflight(args):
    import numpy as np
    settings=read(args.config or CONFIG/'benchmark_solvers.json')
    if settings.get('schema_version')!=2:raise ValueError('Unsupported solver configuration schema')
    if any('seed' in p or 'device' in p for p in settings['solvers'].values()):
        raise ValueError('Seeds/devices belong to execution policy, not algorithm configuration')
    catalog=read(CONFIG/'qubo_benchmark_catalog_200_500_1000_v2.json')
    if catalog['instances']!=loadCatalog()['instances']:
        raise ValueError('v2 catalog differs from validated prepared input catalog')
    entries=select(args.n,args.density,args.instances,args.instance_id)
    selected=[]
    for entry in entries:
        problem,meta=loadPrepared(entry)
        source,_=readSource(entry,DEFAULT_DATA)
        rng=np.random.default_rng(20260929)
        checks=[np.zeros(problem.n,dtype=int),np.ones(problem.n,dtype=int),
                np.eye(1,problem.n,problem.n-1,dtype=int)[0]]+[rng.integers(0,2,problem.n) for _ in range(4)]
        if any(problem.score(x)!=source.score(x) for x in checks):raise ValueError('Independent source objective mismatch')
        if not math.isfinite(entry['normalized_reference_objective_min']):raise ValueError('Nonfinite reference')
        off=problem.rows!=problem.cols
        selected.append(dict(entry=entry,metadata=meta,npz=str(DEFAULT_DATA/'normalized'/f"{entry['instance_id']}.npz"),
            normalized_problem_sha256=meta['normalized_sha256'],raw_source_sha256=meta['raw_source']['sha256'],
            n_couplings=int(off.sum()),nnz_diagonal=int((~off).sum()),
            nnz_matrix=int((~off).sum()+2*off.sum()),measured_density=meta['measured_interaction_density_percent']))
    env=environment(args.device,args.require_gpu);device=env['hardware']['device']
    rows=inventory(settings['solvers'],device)
    available=[r['solver_id'] for r in rows if r['status']=='runnable']
    requested=available if args.solvers=='all' else args.solvers.split(',')
    if len(set(requested))!=len(requested) or not requested:raise ValueError('Solver IDs must be nonempty and unique')
    if set(requested)-set(available):raise ValueError(f'Unavailable/ineligible solvers: {sorted(set(requested)-set(available))}')
    if not available:raise ValueError('No eligible runnable solvers')
    schedule=seeds(args.runs,args.seed_start,args.seeds_file)
    core=dict(schema_version=2,n=args.n,density=args.density,budget_s=budget(args.time_limit),
         instance_mode='single' if args.instance_id else args.instances,
         instances=[{k:v for k,v in row.items() if k!='npz'} for row in selected],
         solvers={name:settings['solvers'][name] for name in requested},seeds=schedule,
         protocol=PROTOCOL,protocol_sha256=filehash(CONFIG/'benchmark_protocol.json'),
         definitions_sha256=filehash(ROOT/'BENCHMARK_DEFINITIONS.txt'),
         catalog_sha256=filehash(CONFIG/'qubo_benchmark_catalog_200_500_1000_v2.json'),
         reference_snapshot_sha256=digest(catalog),environment_hash=digest(env),
         source=source_identity(),requested_device=args.device,actual_device=device)
    core['standard']=schedule==list(range(100)) and args.instances=='all' and not args.instance_id
    # Dirty flag can change with docs/artifacts; numerical source content governs resume.
    identity=dict(core,source={'code_commit':core['source']['code_commit'],
                              'code_snapshot_hash':core['source']['code_snapshot_hash']})
    return dict(core=core,identity=digest(identity),environment=env,registry=rows,
                selected=selected,catalog=catalog,configuration=settings,
                total=len(selected)*len(requested)*len(schedule))

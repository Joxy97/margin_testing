"""Offline preflight, immutable settings and complete registry inventory."""
import importlib.metadata
import itertools
import math
import os
import platform
import socket
import copy
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from .common import ROOT,CONFIG,SCHEMA,read,digest,filehash,source_identity
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

def device_selection(requested='auto'):
    """Resolve visible logical CUDA indices; never fall back for explicit CUDA."""
    import torch
    if requested in ('auto','all'):
        devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        if devices:return devices
        if requested=='all':raise ValueError('CUDA requested but unavailable; no CPU fallback')
        return ['cpu']
    devices=[value.strip() for value in str(requested).split(',')]
    if devices==['cpu']:return devices
    if not devices or len(set(devices))!=len(devices):
        raise ValueError('Devices must be nonempty and unique')
    for device in devices:
        if not device.startswith('cuda:') or not device[5:].isdigit():
            raise ValueError('Use auto, cpu, indexed cuda:N, or a comma-separated CUDA device list')
    if not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable; no CPU fallback')
    indices=[int(device[5:]) for device in devices]
    if len(set(indices))!=len(indices):raise ValueError('Devices must be nonempty and unique')
    if any(index>=torch.cuda.device_count() for index in indices):raise ValueError('CUDA device index unavailable')
    return [f'cuda:{index}' for index in indices]


def _cgroup_directories(root=Path('/sys/fs/cgroup'),membership=Path('/proc/self/cgroup')):
    """Include namespaced roots and readable current-group ancestors (v1/v2)."""
    directories=[(root,'unified'),(root/'cpu','cpu'),(root/'cpu,cpuacct','cpu'),(root/'memory','memory')]
    try:lines=membership.read_text().splitlines()
    except OSError:lines=[]
    for line in lines:
        fields=line.split(':',2)
        if len(fields)!=3:continue
        _,controllers,relative=fields
        parts=Path(relative.lstrip('/')).parts
        if '..' in parts:continue
        names=controllers.split(',') if controllers else ['unified']
        for name in names:
            if name not in ('unified','cpu','memory'):continue
            bases=[root] if name=='unified' else [root/name,root/controllers]
            for base in bases:
                current=base.joinpath(*parts)
                while True:
                    directories.append((current,name))
                    if current==base:break
                    current=current.parent
    return list(dict.fromkeys(directories))


def cpu_capacity(root=Path('/sys/fs/cgroup'),membership=Path('/proc/self/cgroup')):
    """Logical affinity and container CPU quota, rather than physical host count."""
    import psutil
    logical=os.cpu_count() or 1
    try:affinity=psutil.Process().cpu_affinity()
    except (AttributeError,OSError,psutil.Error):affinity=None
    usable=float(min(logical,len(affinity))) if affinity else float(logical)
    quotas=[]
    for directory,kind in _cgroup_directories(root,membership):
        try:
            if kind=='unified':
                quota,period=(directory/'cpu.max').read_text().split()
                if quota!='max' and int(quota)>0 and int(period)>0:quotas.append(int(quota)/int(period))
            elif kind=='cpu':
                quota=int((directory/'cpu.cfs_quota_us').read_text())
                period=int((directory/'cpu.cfs_period_us').read_text())
                if quota>0 and period>0:quotas.append(quota/period)
        except (OSError,ValueError):continue
    quota=min(quotas) if quotas else None
    return dict(logical_cpus=logical,cpu_affinity=affinity,cgroup_cpu_quota=quota,
                effective_cpus=min(usable,quota) if quota is not None else usable)


def host_memory_available(root=Path('/sys/fs/cgroup'),membership=Path('/proc/self/cgroup')):
    import psutil
    available=psutil.virtual_memory().available
    for directory,kind in _cgroup_directories(root,membership):
        names=('memory.max','memory.current') if kind=='unified' else (
            'memory.limit_in_bytes','memory.usage_in_bytes') if kind=='memory' else None
        if names is None:continue
        try:
            limit=(directory/names[0]).read_text().strip()
            if limit!='max':available=min(available,max(0,int(limit)-int((directory/names[1]).read_text())))
        except (OSError,ValueError):continue
    return available


def environment(requested='auto',require_gpu=False):
    import torch
    import psutil
    devices=device_selection(requested);device=devices[0]
    if require_gpu and not device.startswith('cuda'):raise ValueError('--require-gpu needs an available CUDA device')
    gpus=[]
    for selected in devices:
        if selected=='cpu':continue
        index=int(selected.split(':')[1])
        p=torch.cuda.get_device_properties(index)
        gpus.append(dict(device=selected,name=p.name,uuid=str(getattr(p,'uuid','unavailable')),total_memory_bytes=p.total_memory))
    gpu=gpus[0] if gpus else None
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
    hardware=dict(host=socket.gethostname(),cpu=platform.processor(),**cpu_capacity(),
                  physical_cpus=psutil.cpu_count(logical=False),host_ram_bytes=psutil.virtual_memory().total,
                  gpu=gpu,gpus=gpus,device=device,devices=devices)
    return dict(hardware=hardware,hardware_id=digest(hardware),os=platform.platform(),python=platform.python_version(),
                packages=packages,torch_cuda=torch.version.cuda,driver=driver,blas=blas,
                cpu_threads=PROTOCOL['execution']['cpu_threads'],blas_thread_limit=PROTOCOL['execution']['cpu_threads'],tf32=False,
                environment_allowlist={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
                        'MKL_NUM_THREADS','NVIDIA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY',
                        'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE') if k in os.environ},
                container_image=None,container_image_reason='not available without an explicitly supplied deployment manifest')

def inventory(settings,device='cpu'):
    from qubo_solvers import SOLVERS,solver_capabilities,create_bqm_solver
    from ..adapters import GPU_SOLVERS
    rows=[]
    for name in SOLVERS:
        cap=solver_capabilities(name);eligible=name in GPU_SOLVERS
        row=dict(solver_id=name,version='0.1.0',adapter_version='runtime-v4',backend=cap['backend'],
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

def select_solvers(rows,selection='all'):
    """Keep backend failures distinct from invalid explicit solver IDs."""
    by_name={row['solver_id']:row for row in rows}
    available=[row['solver_id'] for row in rows if row['status']=='runnable']
    if selection=='all':
        if available:return available
        details='\n'.join(f"  {row['solver_id']}: {row['reason'] or row['status']}"
                          for row in rows if row['status']=='unavailable')
        raise ValueError('No eligible runnable solvers. Backend availability checks failed:\n'
                         +(details or '  No eligible solver entries were discovered.')
                         +'\nRun python run_benchmark.py --list-solvers for the inventory.'
                         +'\nFor missing dependencies, install them in this same Python environment with:'
                         +'\n  python -m pip install -e ".[benchmark]"')
    requested=[name.strip() for name in selection.split(',')]
    if any(not name for name in requested) or len(set(requested))!=len(requested):
        raise ValueError('Solver IDs must be nonempty and unique')
    unavailable=[name for name in requested if name not in available]
    if unavailable:
        details='\n'.join(f"  {name}: "+(by_name[name]['reason'] or by_name[name]['status']
                         if name in by_name else 'unknown solver ID') for name in unavailable)
        raise ValueError('Unavailable/ineligible solvers:\n'+details)
    return requested


def preflight(args):
    import numpy as np
    settings=read(args.config or CONFIG/'benchmark_solvers.json')
    if settings.get('schema_version')!=2:raise ValueError('Unsupported solver configuration schema')
    if any('seed' in p or 'device' in p for p in settings['solvers'].values()):
        raise ValueError('Seeds/devices belong to execution policy, not algorithm configuration')
    env=environment(args.device,args.require_gpu);device=env['hardware']['device']
    devices=env['hardware']['devices']
    rows=inventory(settings['solvers'],device)
    requested=select_solvers(rows,args.solvers)
    workers=worker_argument(getattr(args,'seed_workers','auto'))
    mode=getattr(args,'worker_mode','persistent')
    if mode not in ('persistent','fresh'):raise ValueError('Invalid worker mode')
    max_auto=getattr(args,'max_auto_workers',4)
    if isinstance(max_auto,bool) or not isinstance(max_auto,int) or not 1<=max_auto<=100:
        raise ValueError('--max-auto-workers must be an integer from 1 through 100')
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
    schedule=seeds(args.runs,args.seed_start,args.seeds_file)
    policy=copy.deepcopy(PROTOCOL)
    policy['execution'].update(worker_mode=mode,seed_workers_requested=workers,
        autotune_enabled=not bool(getattr(args,'no_autotune',False)),auto_concurrency_max_per_device=max_auto,
        worker_policy=('persistent spawned workers; immutable input reuse; new solver and RNG/state per trial'
                       if mode=='persistent' else 'fresh spawned worker per trial'))
    capacity=worker_capacity(selected,{name:settings['solvers'][name] for name in requested},devices,policy['execution'])
    counts={name:min(1 if workers=='auto' else workers,len(schedule)) for name in devices}
    # Auto calibration is performed once, before experiment identity is frozen.
    # A resumed run must use its saved counts, never re-tune against current load.
    if getattr(args,'resume',None):
        saved=read(Path(args.resume)/'experiment.json')['core']
        previous=saved['protocol']['execution']
        if workers==previous.get('seed_workers_requested',1):
            counts=previous.get('workers_per_device',{saved['actual_device']:previous['seed_workers']})
    core=dict(schema_version=SCHEMA,n=args.n,density=args.density,budget_s=budget(args.time_limit),
         instance_mode='single' if args.instance_id else args.instances,
         instances=[{k:v for k,v in row.items() if k!='npz'} for row in selected],
         solvers={name:settings['solvers'][name] for name in requested},seeds=schedule,
         protocol=policy,protocol_sha256=filehash(CONFIG/'benchmark_protocol.json'),
         definitions_sha256=filehash(ROOT/'BENCHMARK_DEFINITIONS.txt'),
         catalog_sha256=filehash(CONFIG/'qubo_benchmark_catalog_200_500_1000_v2.json'),
         reference_snapshot_sha256=digest(catalog),environment_hash=digest(env),
         source=source_identity(),requested_device=args.device,actual_device=device,actual_devices=devices)
    core['standard']=schedule==list(range(100)) and args.instances=='all' and not args.instance_id
    prepared=dict(core=core,environment=env,registry=rows,
                selected=selected,catalog=catalog,configuration=settings,capacity=capacity,
                total=len(selected)*len(requested)*len(schedule))
    finalize_execution(prepared,counts)
    return prepared


def worker_argument(value):
    if value=='auto':return value
    if isinstance(value,str) and value.isdigit():value=int(value)
    if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=100:
        raise ValueError('--seed-workers must be auto or an integer from 1 through 100 per device')
    return value


def refresh_identity(prepared):
    """Refresh only after all execution settings have been selected."""
    core=prepared['core']
    identity=dict(core,source={'code_commit':core['source']['code_commit'],
                              'code_snapshot_hash':core['source']['code_snapshot_hash']})
    prepared['identity']=digest(identity)
    return prepared


def finalize_execution(prepared,workers_per_device,tuning_metadata=None):
    """Validate/freeze concurrency; measurements are metadata, not resume identity."""
    core=prepared['core'];capacity=prepared['capacity'];counts=dict(workers_per_device)
    if set(counts)!=set(core['actual_devices']):raise ValueError('Worker device inventory differs from selected devices')
    for device,count in counts.items():
        if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=len(core['seeds']):
            raise ValueError('Each selected device needs 1..number-of-seeds workers')
        maximum=capacity['workers_per_device'][device]
        if count>maximum:
            raise ValueError(f'Requested {count} workers on {device} exceeds the memory/CPU-admitted maximum {maximum}. '
                             'Reduce --seed-workers; solver parameters will not be changed.')
    total=sum(counts.values())
    if total>capacity['maximum_workers_estimate']:
        raise ValueError(f'Requested {total} total workers exceeds the aggregate memory/CPU-admitted maximum '
                         f"{capacity['maximum_workers_estimate']}. Reduce --seed-workers.")
    shared=any(count>1 for count in counts.values())
    mode='parallel_shared_device' if shared else 'multi_gpu_isolated' if len(counts)>1 else 'isolated_latency'
    core['protocol']['execution'].update(workers_per_device=counts,seed_workers=total,execution_mode=mode,
                                        scheduling_policy='device-affine persistent lanes; solver/instance groups interleaved')
    if tuning_metadata is not None:prepared['tuning']=tuning_metadata
    return refresh_identity(prepared)


def worker_capacity(selected,settings,device,policy):
    """Conservative admission estimate, not an automatically tuned concurrency."""
    import torch
    from ..model import Problem
    from ..adapters import toSolverProblem
    from qubo_solvers import create_bqm_solver
    peak=0
    for item in selected:
        problem=toSolverProblem(Problem.load(item['npz']))
        for name,parameters in settings.items():
            solver=create_bqm_solver(name,{'device':'cpu'})
            peak=max(peak,solver.estimatedWorkingMemoryBytes(problem,parameters))
    fraction=policy['memory_admission_fraction']
    devices=[device] if isinstance(device,str) else list(device)
    host_free=host_memory_available()
    host_per_worker=policy['worker_host_reserve_bytes']+peak
    host_limit=max(0,int(host_free*fraction)//host_per_worker)
    cpus=cpu_capacity()
    # Keep at least one lane per selected GPU; additional processes must fit
    # CPU affinity/quota as well as aggregate host and per-device memory.
    cpu_limit=max(len(devices),int(cpus['effective_cpus']//policy['cpu_threads']))
    per_device={};gpu_memories={}
    for selected_device in devices:
        limit=min(100,host_limit,cpu_limit)
        gpu_free=None;gpu_per_worker=None
        if selected_device.startswith('cuda'):
            gpu_free,_=torch.cuda.mem_get_info(selected_device)
            gpu_per_worker=policy['worker_cuda_reserve_bytes']+peak
            limit=min(limit,int(gpu_free*fraction)//gpu_per_worker)
        per_device[selected_device]=max(0,limit)
        gpu_memories[selected_device]=dict(free_bytes=gpu_free,per_worker_estimate=gpu_per_worker)
    first=gpu_memories[devices[0]]
    return dict(maximum_workers_estimate=max(0,min(host_limit,cpu_limit,sum(per_device.values()))),
                workers_per_device=per_device,devices=gpu_memories,cpu=cpus,cpu_worker_limit=cpu_limit,
                host_worker_limit=host_limit,algorithm_working_bytes_estimate=peak,
                host_available_bytes=host_free,host_per_worker_estimate=host_per_worker,
                gpu_free_bytes=first['free_bytes'],gpu_per_worker_estimate=first['per_worker_estimate'],
                memory_fraction=fraction,
                note='Conservative allocation/context estimates, not measured peaks or optimal throughput. '
                     'Host memory and CPU quotas are shared across devices. Actual CUDA context overhead '
                     'and other workloads can still cause OOM; auto tuning stays within these limits.')

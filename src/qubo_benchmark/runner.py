"""Spawn-isolated deadline enforcement; offline scoring and reproducible results."""
import hashlib
import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import numpy as np
from .adapters import (Adapter, SOLVERS, GPU_SOLVERS, TORCH_SOLVERS, CLASSICAL_SOLVERS,
                       CATEGORICAL_SOLVERS, DETERMINISTIC_SOLVERS,
                       UnsupportedProblem, BackendUnavailable)
from .catalog import ROOT, DEFAULT_DATA, loadCatalog, selectEntries
from .model import Problem
from .pipeline import loadPrepared, paths, readSource
from .storage import readJson, writeJson, sha256, stamp


def worker(connection, path, specification, seed, config):
    try:
        started=time.perf_counter();problem=Problem.load(path);loaded=time.perf_counter()
        if specification['solver'] in TORCH_SOLVERS:
            import torch
            torch.set_num_threads(config.get('cpu_threads',1))
        adapter=Adapter(problem,specification['solver'],seed,config['device'],config['precision'],specification.get('parameters',{}),specification.get('constructor'))
        adapter.synchronize()
        connection.send(dict(kind='ready',load_seconds=loaded-started,
                             setup_seconds=time.perf_counter()-loaded,adapter=adapter.describe()))
        if connection.recv() != 'start': return
        if adapter.device.startswith('cuda'):
            import torch
            torch.cuda.reset_peak_memory_stats(adapter.device)
        started=time.perf_counter();sample,reported=adapter.solve();adapter.synchronize()
        solveSeconds=time.perf_counter()-started
        memory={}
        if adapter.device.startswith('cuda'):
            memory.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(adapter.device))
        connection.send(dict(kind='result',sample=sample.tolist(),solver_reported_energy=reported,
                             solve_seconds=solveSeconds,memory=memory))
    except UnsupportedProblem as exc:
        connection.send(dict(kind='error',status='unsupported',error=str(exc)))
    except (BackendUnavailable,ImportError,OSError) as exc:
        connection.send(dict(kind='error',status='unavailable',error=str(exc)))
    except BaseException:
        connection.send(dict(kind='error',error=traceback.format_exc()))
    finally: connection.close()


def supervise(path,specification,seed,config,workerFunction=worker):
    context=mp.get_context('spawn');parent,child=context.Pipe()
    process=context.Process(target=workerFunction,args=(child,str(path),specification,seed,config))
    started=time.perf_counter();ready=None;result=None
    process.start();child.close();deadline=started+config['setup_limit_s'];phase='setup'
    try:
        while True:
            remaining=deadline-time.perf_counter()
            if remaining <= 0:
                result=dict(status='timeout',timeout_phase=phase,error=f'{phase} deadline exceeded; no completed-call incumbent available');break
            if parent.poll(min(remaining,.05)):
                try:message=parent.recv()
                except EOFError:
                    result=dict(status='error',error='Worker exited without a result');break
                if message['kind']=='ready':
                    ready=message;phase='solve';deadline=time.perf_counter()+config['time_limit_s'];parent.send('start')
                elif message['kind']=='result':
                    result=dict(status='completed',**message);break
                else:
                    result=dict(status=message.get('status','error'),error=message['error']);break
            elif not process.is_alive():
                result=dict(status='error',error=f'Worker exited with code {process.exitcode}');break
    finally:
        if process.is_alive(): process.terminate()
        process.join(timeout=1)
        if process.is_alive(): process.kill();process.join(timeout=1)
        parent.close()
    return dict(result,worker_setup=ready,worker_end_to_end_seconds=time.perf_counter()-started)


def codeIdentity():
    digest=hashlib.sha256()
    roots=[ROOT/'src'/'qubo_benchmark',ROOT/'src'/'qubo_solvers',ROOT/'src'/'margin_calculator'/'optimization']
    for root in roots:
        for path in sorted(root.rglob('*.py')):
            digest.update(path.relative_to(ROOT).as_posix().encode());digest.update(path.read_bytes())
    try: head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): head=None
    return dict(git_head=head,python_sources_sha256=digest.hexdigest(),
                source_hash_note='Includes uncommitted and untracked Python under benchmark, qubo_solvers and optimization packages.')


def checkConfig(config):
    if config.get('device')=='auto' or not isinstance(config.get('device'),str):raise ValueError('An explicit device is required')
    if config.get('precision') not in ('float32','float64'):raise ValueError('Precision must be float32 or float64')
    if not isinstance(config.get('seeds'),list) or not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Provide a nonempty list of distinct explicit seeds')
    if any(type(s) is not int or s < 0 or s >= 2**63-1 for s in config['seeds']): raise ValueError('Invalid seeds')
    for key in ('setup_limit_s','time_limit_s'):
        if not np.isfinite(config[key]) or config[key] <= 0: raise ValueError(f'Invalid {key}')
    if not isinstance(config.get('solvers'),list) or not config['solvers']: raise ValueError('No solvers selected')
    ids=[s['id'] for s in config['solvers']]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate solver variant ID')
    for spec in config['solvers']:
        if spec['solver'] not in SOLVERS: raise ValueError(f'Unsupported solver: {spec["solver"]}')
        if spec.get('trial_kind') not in ('independent_seeded_start','deterministic_timing_repetition'):
            raise ValueError('Declare trial_kind for each solver')
        if not spec.get('output_variant'): raise ValueError('Describe raw/bundled postprocessing in output_variant')
        effective=effectiveConfig(config,spec)
        name=spec['solver'];device=effective['device'];precision=effective['precision']
        if precision not in ('float32','float64'):raise ValueError('Invalid per-solver precision')
        if device != 'cpu' and not (isinstance(device,str) and (device=='cuda' or (device.startswith('cuda:') and device[5:].isdigit()))):raise ValueError('Use explicit cpu or cuda[:index] device')
        if name in CLASSICAL_SOLVERS and device!='cpu':raise ValueError(f'{name} only has a CPU entry point')
        if name in CLASSICAL_SOLVERS and precision!='float64':raise ValueError('Classical sampler wrappers require float64')
        if name in DETERMINISTIC_SOLVERS and spec['trial_kind']!='deterministic_timing_repetition':raise ValueError(f'{name} repetitions are deterministic, not independent seeded trials')
    return config


def effectiveConfig(config,specification):
    """Per-solver CPU/GPU/precision overrides are explicit, never fallbacks."""
    return dict(config,**{key:specification[key] for key in ('device','precision') if key in specification})


def comparison(value,entry):
    ref=entry['normalized_reference_objective_min'];delta=value-ref
    proven=entry['reference_status']=='published_proven_optimum'
    return dict(delta=delta,relative_delta_percent=100*delta/max(1,abs(ref)),
                comparison_label='gap to published optimum (catalog claim)' if proven else 'gap to published reference',
                alarm=('below_published_proven_optimum' if proven else 'candidate_improvement') if delta < 0 else None,
                reference_attained=delta <= 0,time_to_target_seconds=None,
                time_to_target_note='Solver exposes no timed incumbent history; final score is not a time-to-target observation.')


def run(config,directory=DEFAULT_DATA,output=None):
    config=checkConfig(config);entries=selectEntries(loadCatalog(directory),config.get('instances'))
    output=Path(output or ROOT/'benchmark_results'/'runs'/time.strftime('%Y%m%d-%H%M%S'))
    if output.exists() and any(output.iterdir()): raise ValueError(f'Refusing to overwrite existing run: {output}')
    output.mkdir(parents=True,exist_ok=True);writeJson(output/'config.json',config)
    identity=codeIdentity();rows=[]
    hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),python=platform.python_version())
    for entry in entries:
        start=time.perf_counter()
        try:
            problem,metadata=loadPrepared(entry,directory);source,_=readSource(entry,directory)
        except (OSError,ValueError,KeyError) as exc:
            for spec in config['solvers']:
                for seed in config['seeds']:
                    row=dict(instance_id=entry['instance_id'],solver_id=spec['id'],seed=seed,status='data_error',
                             error=str(exc),n=entry['binary_variables'],density_category=entry['density_category'],
                             problem_family=entry['problem_family'],configuration=config,code=identity,
                             objective=None,bitstring=None)
                    rows.append(row);writeJson(output/f'run-{len(rows):06d}.json',row)
            print(f"{entry['instance_id']}: data_error: {exc}",flush=True)
            continue
        loadSeconds=time.perf_counter()-start
        for spec in config['solvers']:
            effective=effectiveConfig(config,spec)
            for seed in config['seeds']:
                started=time.perf_counter()
                row=dict(instance_id=entry['instance_id'],solver_id=spec['id'],solver=spec['solver'],seed=seed,
                         n=problem.n,density_category=entry['density_category'],problem_family=entry['problem_family'],
                         instance_sha256=metadata['normalized_sha256'],raw_sha256=metadata['raw_source']['sha256'],
                         code=identity,configuration=config,solver_specification=spec,hardware=hardware,
                         requested_device=effective['device'],precision=effective['precision'],reference=metadata['reference'],
                         created_at=stamp(),load_validation_seconds=loadSeconds,incumbent_history=None,
                         objective=None,bitstring=None,time_to_target_seconds=None)
                row.update(supervise(paths(directory,entry['instance_id'])[0],spec,seed,effective))
                if row['status']=='completed':
                    try:
                        sample=row.pop('sample');value=problem.score(sample)
                        if source.score(sample) != value: raise ValueError('Source/canonical result mismatch')
                        row.update(objective=value,bitstring=''.join(map(str,sample)),**comparison(value,entry))
                        if hasattr(source,'details'): row['clique']=source.details(sample)
                    except Exception as exc: row.update(status='invalid_output',error=str(exc))
                row['end_to_end_seconds']=time.perf_counter()-started+loadSeconds
                row['timing_note']='Per-instance load validation amortized across trials, shown in each row; worker setup separately recorded; transfers included in solve.'
                rows.append(row);writeJson(output/f'run-{len(rows):06d}.json',row)
                print(f"{entry['instance_id']} {spec['id']} seed={seed}: {row['status']} objective={row['objective']}"+
                      (f" ({row['error']})" if row['status'] in ('unsupported','unavailable') else ''),flush=True)
    report=summarize(output);writeJson(output/'summary.json',report);return report


def summarize(output):
    rows=[readJson(p) for p in sorted(Path(output).glob('run-*.json'))]
    groups={};families={}
    for row in rows:
        for table,key in ((groups,f"{row['n']}_{row['density_category']}"),(families,row['problem_family'])):
            key += ':'+row['solver_id'];group=table.setdefault(key,dict(total=0,completed=0,failures=0,unsupported=0,unavailable=0,deltas=[],validation_alarms=0,candidate_improvements=0))
            group['total']+=1
            if row['status']=='completed':
                group['completed']+=1;group['deltas'].append(row['delta'])
                group['validation_alarms']+=row.get('alarm')=='below_published_proven_optimum'
                group['candidate_improvements']+=row.get('alarm')=='candidate_improvement'
            elif row['status']=='unsupported':group['unsupported']+=1
            else:
                group['failures']+=1
                group['unavailable']+=row['status']=='unavailable'
    for group in list(groups.values())+list(families.values()):
        values=group.pop('deltas');group['mean_signed_delta']=sum(values)/len(values) if values else None
    return dict(total=len(rows),completed=sum(r['status']=='completed' for r in rows),
                unsupported=sum(r['status']=='unsupported' for r in rows),unavailable=sum(r['status']=='unavailable' for r in rows),
                groups=groups,families=families,
                per_instance=[{k:r.get(k) for k in ('instance_id','solver_id','seed','status','objective','delta','alarm','error')} for r in rows])

"""Warm readiness handshake and strict completed-candidate timing."""
import hashlib
import math
import random
import time
import traceback
from .common import utc
from qubo_solvers.observation import SolveInterrupted, observing

class Observer:
    def __init__(self,budget,send,stop=None,clock=time.perf_counter_ns):
        self.budget=budget;self.send=send;self.stop=stop;self.clock=clock
        self.origin=clock();self.seen=set();self.initialization_hash=None;self.events=0
        self.last_iteration=None
        self.preparation={}
    def elapsed(self):return (self.clock()-self.origin)/1e9
    def poll(self):
        if self.stop is not None and self.stop.is_set():raise SolveInterrupted('interrupted')
        if self.elapsed()>=self.budget:raise SolveInterrupted('deadline')
    def prepared(self,matrix,**details):
        import torch
        if matrix.layout==torch.strided:
            storage=matrix.numel()*matrix.element_size();form='dense_tensor'
        elif matrix.layout==torch.sparse_csr:
            storage=sum(v.numel()*v.element_size() for v in (matrix.values(),matrix.col_indices(),matrix.crow_indices()))
            form='sparse_csr'
        else:storage=None;form=str(matrix.layout)
        self.preparation=dict(matrix_storage_format=form,matrix_storage_bytes=storage,
                              resolved_structure_json=dict(details,matrix_shape=list(matrix.shape)))
    def capture(self,samples,raw=None,*,phase='checkpoint',iteration=None):
        # Do not decode a new candidate after a known cutoff.
        if phase!='final_returned':self.poll()
        if hasattr(samples,'detach'):
            samples=samples.detach().clone().cpu().tolist()
        else:samples=[list(row) for row in samples]
        if hasattr(raw,'detach'):raw=raw.detach().cpu().reshape(-1).tolist()
        raw=[None]*len(samples) if raw is None else list(raw)
        batch=[]
        for index,row in enumerate(samples):
            # Invalid values remain visible; no coercion of .5 or NaN into valid bits.
            bits=''.join(str(int(v)) for v in row) if all(v in (0,1) for v in row) else repr(row)
            if phase!='final_returned' and bits in self.seen:continue
            self.seen.add(bits)
            value=raw[index] if index<len(raw) else None
            if isinstance(value,(int,float)) and not math.isfinite(value):value=None
            batch.append(dict(bitstring=bits,raw=value,phase=phase,iteration=iteration))
        if phase=='initial' and self.initialization_hash is None:
            self.initialization_hash=hashlib.sha256(repr(samples).encode()).hexdigest()
        # Timestamp after decoding, clone, CPU completion, text construction and hashing.
        elapsed=self.elapsed()
        for event in batch:event['elapsed_s']=elapsed
        if batch:self.send(dict(kind='events',events=batch))
        self.events+=len(batch);self.last_iteration=iteration
        if phase!='final_returned':self.poll()

def synchronize(torch,device):
    if device.startswith('cuda'):torch.cuda.synchronize(device)

def execute(connection,stop,job):
    setup_started=time.perf_counter()
    try:
        import numpy as np
        import torch
        # These backends import SciPy lazily. Importing is infrastructure; do
        # not charge module loading to the first timed calibration or solve.
        import scipy.linalg
        import scipy.sparse
        from threadpoolctl import threadpool_limits
        from dataclasses import fields
        from qubo_solvers import QUBO,LIBRARY_SOLVERS,create_solver,create_bqm_solver,library_solver_class
        from qubo_solvers.backends.problem import QUBOProblem
        from ..model import Problem
        from ..adapters import toSolverProblem
        device=job['device'];parameters=dict(job['parameters']);name=job['solver']
        torch.set_num_threads(job['cpu_threads'])
        thread_limits=threadpool_limits(limits=job['cpu_threads'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        source=Problem.load(job['npz'])  # Contains no reference or quality target.
        compact=toSolverProblem(source)
        # Trial seed is effective, not mixed with a problem-dependent hidden offset.
        compact=QUBOProblem(compact.linear,compact.quadraticHeads,compact.quadraticTails,
                            compact.quadraticBiases,compact.offset,seedOffset=0)
        native=name in LIBRARY_SOLVERS
        solver=create_bqm_solver(name,{'device':device})
        transfer_started=time.perf_counter()
        prepared=(QUBO(torch.as_tensor(source.dense().copy(),dtype=getattr(torch,parameters['dtype']),device=device),
                       source.offset) if native else None)
        synchronize(torch,device)
        transfer_s=time.perf_counter()-transfer_started
        # Infrastructure-only warmup uses no coefficients, states or candidates from the instance.
        warm_started=time.perf_counter()
        z=torch.zeros((8,8),device=device);torch.mm(z,z);synchronize(torch,device);del z
        warm_s=time.perf_counter()-warm_started
        algorithm=None
        if native:
            names={f.name for f in fields(library_solver_class(name))}
            algorithm=create_solver(name,**{k:v for k,v in parameters.items() if k in names})
        setup=dict(static_setup_s=time.perf_counter()-setup_started,transfer_s=transfer_s if native else None,
                   warmup_s=warm_s,warmup_policy='context and 8x8 zero GEMM only; no instance optimization',
                   matrix_storage_format='dense_tensor' if native else 'compact_source_then_native_preprocessing',
                   matrix_storage_bytes=(prepared.Q.numel()*prepared.Q.element_size() if native else compact.numericMemoryBytes))
        connection.send(dict(kind='ready',setup=setup))
        if connection.recv()!='go':return
        memory={}
        if device.startswith('cuda'):
            synchronize(torch,device);torch.cuda.reset_peak_memory_stats(device)
            memory=dict(gpu_allocated_start_bytes=torch.cuda.memory_allocated(device),
                        gpu_reserved_start_bytes=torch.cuda.memory_reserved(device))
        observer=Observer(job['budget'],connection.send,stop)
        connection.send(dict(kind='started',origin_ns=observer.origin,started_at_utc=utc()))
        error=None;error_type=None;error_message=None;reason='natural_return';result=None
        reset_s=None
        try:
            with observing(observer):
                random.seed(job['seed']);np.random.seed(job['seed']%(2**32))
                torch.manual_seed(job['seed'])
                if device.startswith('cuda'):torch.cuda.manual_seed_all(job['seed'])
                # Explicit local Generators are seeded by each unchanged algorithm.
                reset_s=observer.elapsed();observer.poll()
                if native:
                    result=algorithm.solve(prepared,restarts=parameters['runs'],seed=job['seed'],
                        batch_size=parameters['run_batch_size'],best_only=parameters['best_only'],
                        memory_limit_bytes=parameters['memory_limit_bytes'])
                    observer.capture(result.best_assignments,result.best_energies,phase='final_returned')
                else:
                    result=solver.solve(compact,dict(parameters,seed=job['seed']))
                    observer.capture([result.sample],[result.energy],phase='final_returned')
        except SolveInterrupted as exc:reason=str(exc)
        except Exception as exc:
            error_type=type(exc).__name__;error_message=str(exc)
            error='oom' if isinstance(exc,(MemoryError,torch.OutOfMemoryError)) or 'out of memory' in str(exc).lower() else 'error'
            reason=error
        finally:
            synchronize(torch,device)
        elapsed=observer.elapsed()
        if device.startswith('cuda'):
            memory.update(gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
                          gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
        connection.send(dict(kind='done',actual_solve_wall_s=elapsed,finished_at_utc=utc(),
            stop_reason=reason,error=error,error_type=error_type,error_message=error_message,
            trial_reset_init_s=reset_s,initialization_hash=observer.initialization_hash,
            iterations=observer.last_iteration,preparation=observer.preparation,**memory))
    except BaseException as exc:
        try:connection.send(dict(kind='fatal',error_type=type(exc).__name__,error_message=str(exc),
                                 traceback=traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:connection.close()

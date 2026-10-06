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
        arrays=False
        if hasattr(samples,'detach'):
            captured=samples.detach().clone().cpu()
            if hasattr(captured,'numpy'):
                try:samples=captured.numpy();arrays=True
                except TypeError:samples=captured.tolist()  # E.g. bfloat16 diagnostics.
            else:samples=captured.tolist()
        else:samples=[list(row) for row in samples]
        if arrays:
            import numpy as np
            # Validate before casting: fractional values and NaNs must remain
            # invalid. Encoding a completed host copy in C avoids thousands of
            # Python objects and per-bit conversions at every solver checkpoint.
            valid=((samples==0)|(samples==1)).all(axis=1)
            bitstrings=[(row.astype(np.uint8)+48).tobytes().decode('ascii') if yes
                        else repr(row.tolist()) for row,yes in zip(samples,valid)]
        else:
            bitstrings=[''.join(str(int(v)) for v in row) if all(v in (0,1) for v in row)
                        else repr(row) for row in samples]
        if hasattr(raw,'detach'):raw=raw.detach().cpu().reshape(-1).tolist()
        raw=[None]*len(samples) if raw is None else list(raw)
        batch=[]
        for index,bits in enumerate(bitstrings):
            # Invalid values remain visible; no coercion of .5 or NaN into valid bits.
            if phase!='final_returned' and bits in self.seen:continue
            self.seen.add(bits)
            value=raw[index] if index<len(raw) else None
            if isinstance(value,(int,float)) and not math.isfinite(value):value=None
            batch.append(dict(bitstring=bits,raw=value,phase=phase,iteration=iteration))
        if phase=='initial' and self.initialization_hash is None:
            initial=samples.tolist() if arrays else samples
            self.initialization_hash=hashlib.sha256(repr(initial).encode()).hexdigest()
        # Timestamp after decoding, clone, CPU completion, text construction and hashing.
        elapsed=self.elapsed()
        for event in batch:event['elapsed_s']=elapsed
        if batch:self.send(dict(kind='events',events=batch))
        self.events+=len(batch);self.last_iteration=iteration
        if phase!='final_returned':self.poll()

def synchronize(torch,device):
    if device.startswith('cuda'):torch.cuda.synchronize(device)

class WarmWorker:
    """Own only infrastructure and one immutable input cache between trials."""
    def __init__(self,job):
        import os
        import uuid
        import numpy as np
        import torch
        import scipy.linalg
        import scipy.sparse
        from threadpoolctl import threadpool_limits
        self.torch=torch;self.np=np;self.device=job['device']
        self.threads=job['cpu_threads'];self.worker_id=uuid.uuid4().hex
        self.pid=os.getpid();self.trials=0;self.cache_key=None
        self.source=self.compact=self.prepared=None
        if self.device.startswith('cuda'):
            # CUDA APIs which omit a device must address this worker's GPU,
            # rather than silently creating a second context on cuda:0.
            torch.cuda.set_device(self.device)
        torch.set_num_threads(self.threads)
        self.thread_limits=threadpool_limits(limits=self.threads)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        began=time.perf_counter()
        z=torch.zeros((8,8),device=self.device);torch.mm(z,z)
        synchronize(torch,self.device);del z
        self.warmup_s=time.perf_counter()-began

    def prepare(self,job):
        from qubo_solvers import QUBO,LIBRARY_SOLVERS
        from qubo_solvers.backends.problem import QUBOProblem
        from ..model import Problem
        from ..adapters import toSolverProblem
        if job['device']!=self.device or job['cpu_threads']!=self.threads:
            raise ValueError('Cannot change device/thread policy in a warm worker')
        began=time.perf_counter();native=job['solver'] in LIBRARY_SOLVERS
        key=(job['npz'],job['parameters']['dtype'],native)
        hit=key==self.cache_key
        transfer=0.
        if not hit:
            self.prepared=self.compact=self.source=None
            self.source=source=Problem.load(job['npz'])  # No references/targets.
            compact=toSolverProblem(source)
            self.compact=QUBOProblem(compact.linear,compact.quadraticHeads,compact.quadraticTails,
                                    compact.quadraticBiases,compact.offset,seedOffset=0)
            copied=time.perf_counter()
            if native:
                tensor=self.torch.as_tensor(source.dense().copy(),
                    dtype=getattr(self.torch,job['parameters']['dtype']),device=self.device)
                self.prepared=QUBO(tensor,source.offset)
            synchronize(self.torch,self.device)
            transfer=time.perf_counter()-copied
            self.cache_key=key
        self.native=native
        return dict(static_setup_s=time.perf_counter()-began,transfer_s=transfer if native else None,
            warmup_s=self.warmup_s if not self.trials else 0.,
            warmup_policy='context and 8x8 zero GEMM once per worker; no instance optimization',
            matrix_storage_format='dense_tensor' if native else 'compact_source_then_native_preprocessing',
            matrix_storage_bytes=(self.prepared.Q.numel()*self.prepared.Q.element_size()
                                  if native else self.compact.numericMemoryBytes),
            worker_id=self.worker_id,worker_pid=self.pid,worker_reused=self.trials>0,
            input_cache_hit=hit)

    def run(self,connection,stop,job):
        import gc
        from dataclasses import fields
        from qubo_solvers import create_solver,create_bqm_solver,library_solver_class
        torch=self.torch;parameters=dict(job['parameters']);device=self.device
        memory={}
        synchronize(torch,device)
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
            memory=dict(gpu_allocated_start_bytes=torch.cuda.memory_allocated(device),
                        gpu_reserved_start_bytes=torch.cuda.memory_reserved(device))
        version=self.prepared.Q._version if self.native else None
        observer=Observer(job['budget'],connection.send,stop)
        connection.send(dict(kind='started',origin_ns=observer.origin,started_at_utc=utc()))
        error=None;error_type=None;error_message=None;reason='natural_return'
        result=algorithm=solver=None;reset_s=None
        try:
            with observing(observer):
                random.seed(job['seed']);self.np.random.seed(job['seed']%(2**32))
                torch.manual_seed(job['seed'])
                if device.startswith('cuda'):torch.cuda.manual_seed_all(job['seed'])
                reset_s=observer.elapsed();observer.poll()
                # Fresh objects keep all mutable solver state inside this trial.
                # Construction, algorithmic preprocessing and RNG reset are timed.
                if self.native:
                    names={f.name for f in fields(library_solver_class(job['solver']))}
                    algorithm=create_solver(job['solver'],**{k:v for k,v in parameters.items() if k in names})
                    result=algorithm.solve(self.prepared,restarts=parameters['runs'],seed=job['seed'],
                        batch_size=parameters['run_batch_size'],best_only=parameters['best_only'],
                        memory_limit_bytes=parameters['memory_limit_bytes'])
                    observer.capture(result.best_assignments,result.best_energies,phase='final_returned')
                else:
                    solver=create_bqm_solver(job['solver'],{'device':device})
                    result=solver.solve(self.compact,dict(parameters,seed=job['seed']))
                    observer.capture([result.sample],[result.energy],phase='final_returned')
        except SolveInterrupted as exc:reason=str(exc)
        except Exception as exc:
            error_type=type(exc).__name__;error_message=str(exc)
            error='oom' if isinstance(exc,(MemoryError,torch.OutOfMemoryError)) or 'out of memory' in str(exc).lower() else 'error'
            reason=error
        finally:
            # Solver work has unwound, but queued device operations still belong
            # to the measured solve. This marker lets supervision distinguish a
            # device drain from a solver which never observes its deadline.
            connection.send(dict(kind='stopping',stop_reason=reason,
                solver_return_wall_s=observer.elapsed()))
            synchronize(torch,device)
        elapsed=observer.elapsed()
        if self.native and self.prepared.Q._version!=version:
            raise RuntimeError('Solver mutated cached immutable input; worker cannot be reused')
        if device.startswith('cuda'):
            memory.update(gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
                          gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
        finished=utc()
        connection.send(dict(kind='solved',actual_solve_wall_s=elapsed,finished_at_utc=finished))
        cleanup_started=time.perf_counter()
        result=algorithm=solver=None
        gc.collect()  # Outside solver budget; no mutable state survives into next seed.
        self.trials+=1
        connection.send(dict(kind='done',actual_solve_wall_s=elapsed,finished_at_utc=finished,
            stop_reason=reason,error=error,error_type=error_type,error_message=error_message,
            trial_reset_init_s=reset_s,initialization_hash=observer.initialization_hash,
            iterations=observer.last_iteration,preparation=observer.preparation,
            worker_cleanup_s=time.perf_counter()-cleanup_started,worker_trial_index=self.trials,**memory))


def _fatal(connection,exc):
    try:connection.send(dict(kind='fatal',error_type=type(exc).__name__,error_message=str(exc),
                             traceback=traceback.format_exc()))
    except (BrokenPipeError,EOFError,OSError):pass


def execute(connection,stop,job):
    """Single-use worker retained for fresh-process timing comparisons."""
    began=time.perf_counter()
    try:
        worker=WarmWorker(job)
        setup=worker.prepare(job);setup['static_setup_s']=time.perf_counter()-began
        connection.send(dict(kind='ready',setup=setup))
        if connection.recv()=='go':worker.run(connection,stop,job)
    except BaseException as exc:_fatal(connection,exc)
    finally:connection.close()


def serve(connection,stop):
    """Persistent server: prepare -> readiness barrier -> timed run -> idle."""
    worker=None
    try:
        while True:
            command=connection.recv()
            if command['action']=='shutdown':break
            if command['action']!='prepare':raise ValueError('Expected prepare command')
            job=command['job'];began=time.perf_counter()
            if worker is None:worker=WarmWorker(job)
            setup=worker.prepare(job);setup['static_setup_s']=time.perf_counter()-began
            connection.send(dict(kind='ready',setup=setup))
            command=connection.recv()
            if command['action']=='shutdown':break
            if command['action']!='run':raise ValueError('Expected run command')
            worker.run(connection,stop,job)
    except EOFError:pass
    except BaseException as exc:_fatal(connection,exc)
    finally:connection.close()

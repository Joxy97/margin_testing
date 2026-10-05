"""Warm process isolation, readiness barriers and independently supervised seeds."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import threading
import time
from .common import utc
from .worker import serve


class WorkerSlot:
    def __init__(self,context,mode,stopped):
        self.context=context;self.mode=mode;self.stopped=stopped
        self.process=None;self.connection=None;self.stop=None;self.lifetimes=[]

    def close(self):
        if self.process is None:return 0.
        if self.process.pid is None:
            self.connection.close();self.process.close();self.process=None;self.connection=None
            return 0.
        began=time.perf_counter();forced=False
        if self.process.is_alive():
            try:self.connection.send(dict(action='shutdown'))
            except (BrokenPipeError,EOFError,OSError):pass
            self.process.join(2)
            if self.process.is_alive():
                forced=True;self.process.terminate();self.process.join(2)
            if self.process.is_alive():self.process.kill();self.process.join()
        else:self.process.join()
        duration=time.perf_counter()-began
        self.lifetimes.append(dict(worker_pid=self.process.pid,exit_code=self.process.exitcode,
            shutdown_s=duration,forced=forced,finished_at_utc=utc()))
        self.connection.close();self.process.close();self.process=None;self.connection=None
        return duration

    def sample(self):
        import psutil
        try:
            proc=psutil.Process(self.process.pid)
            rss=sum(p.memory_info().rss for p in [proc]+proc.children(recursive=True))
            if self.rss_start is None:self.rss_start=rss
            self.rss_peak=max(self.rss_peak or rss,rss)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass

    def receive(self):
        if self.connection.poll(.005):return self.connection.recv()
        if not self.process.is_alive():raise EOFError(f'Worker exit code {self.process.exitcode}')
        return None

    def finish(self,done,*,forced=False):
        cleanup=done.get('worker_cleanup_s',0.)
        if forced and self.process is not None and self.process.is_alive():self.process.terminate()
        failed=forced or bool(done.get('error'))
        if failed or self.mode=='fresh':cleanup+=self.close()
        if self.stopped() or done.get('stop_reason')=='interrupted':done['error']='interrupted'
        return dict(events=self.events,setup=self.setup,done=done,started_at_utc=self.started,
            finished_at_utc=done.get('finished_at_utc',utc()),cleanup_s=cleanup,
            worker_end_to_end_s=time.perf_counter()-self.began,parent_trial_started=self.began,
            cpu_rss_start_bytes=self.rss_start,cpu_rss_peak_sampled_bytes=self.rss_peak,
            abort_gpu=self.job['device'].startswith('cuda') and failed)

    def prepare(self,job):
        self.job=job;self.began=time.perf_counter();self.setup={};self.events=[]
        self.started=None;self.rss_start=None;self.rss_peak=None
        try:
            if self.process is None:
                parent,child=self.context.Pipe();self.connection=parent;self.stop=self.context.Event()
                self.process=self.context.Process(target=serve,args=(child,self.stop))
                self.process.start();child.close()
            self.stop.clear()
            self.connection.send(dict(action='prepare',job=job))
            while True:
                if self.stopped():return self.finish(dict(error='interrupted'),forced=True)
                message=self.receive()
                if message:
                    if message['kind']=='fatal':
                        return self.finish(dict(message,error='setup_error'),forced=True)
                    if message['kind']!='ready':raise ValueError('Worker did not acknowledge preparation')
                    self.setup=message['setup'];self.setup['worker_ready_wall_s']=time.perf_counter()-self.began
                    self.sample();return None
                if time.perf_counter()-self.began>job['setup_timeout_s']:
                    return self.finish(dict(error='setup_timeout',error_message='Worker setup timed out'),forced=True)
        except (EOFError,OSError,ValueError) as exc:
            return self.finish(dict(error='worker_crash',error_message=str(exc)),forced=True)

    def run(self):
        origin=None;dispatched=time.perf_counter();next_sample=0.;solved=None;cleanup_started=None
        try:
            if self.stopped():return self.finish(dict(error='interrupted'))
            self.connection.send(dict(action='run'))
            while True:
                now=time.perf_counter()
                if self.stopped():self.stop.set()
                if now>=next_sample:
                    self.sample();next_sample=now+self.job['memory_sampling_interval_s']
                message=self.receive()
                if message:
                    kind=message.pop('kind')
                    if kind=='started':origin=message['origin_ns'];self.started=message['started_at_utc']
                    elif kind=='events':self.events.extend(message['events'])
                    elif kind=='solved':solved=message;cleanup_started=time.perf_counter()
                    elif kind=='done':return self.finish(message)
                    elif kind=='fatal':return self.finish(dict(message,error='worker_error'),forced=True)
                    else:raise ValueError(f'Unexpected worker response {kind}')
                elapsed=(time.perf_counter_ns()-origin)/1e9 if origin is not None else None
                if solved is not None:
                    if now-cleanup_started>self.job['setup_timeout_s']:
                        return self.finish(dict(solved,error='cleanup_timeout',
                            error_message='Worker cleanup timed out after solver returned'),forced=True)
                    continue
                if ((elapsed is not None and elapsed>self.job['budget']+self.job['watchdog_grace_s']) or
                    (origin is None and now-dispatched>self.job['setup_timeout_s'])):
                    return self.finish(dict(error='watchdog_timeout',stop_reason='watchdog',
                        error_message='Supervisor cutoff; no post-deadline quality credit',actual_solve_wall_s=elapsed),forced=True)
        except (EOFError,OSError,ValueError) as exc:
            return self.finish(dict(error='worker_crash',error_message=str(exc)),forced=True)


class SeedPool:
    """One process per concurrent seed; never share RNGs or mutable solver objects."""
    def __init__(self,workers,mode,stop_requested):
        self.abort=threading.Event();self.stop_requested=stop_requested
        context=mp.get_context('spawn')
        self.slots=[WorkerSlot(context,mode,self.stopped) for _ in range(workers)]
        self.executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='seed-supervisor')

    def stopped(self):return self.abort.is_set() or self.stop_requested()

    def completed(self,futures,heartbeat,phase):
        pending=set(futures);next_heartbeat=0.
        while pending:
            if time.perf_counter()>=next_heartbeat:
                heartbeat(phase);next_heartbeat=time.perf_counter()+1.
            finished,pending=wait(pending,timeout=.05,return_when=FIRST_COMPLETED)
            for future in finished:
                result=future.result()
                if result is not None and result['abort_gpu']:self.abort.set()
                yield futures[future],result

    def wave(self,jobs,heartbeat):
        """All inputs are ready before starting any solver clock in this wave."""
        if len(jobs)>len(self.slots):raise ValueError('Wave exceeds worker count')
        futures={self.executor.submit(slot.prepare,job):i for i,(slot,job) in enumerate(zip(self.slots,jobs))}
        ready=[];failures=[]
        try:
            for index,result in self.completed(futures,heartbeat,'worker_setup'):
                if result is None:ready.append(index)
                else:failures.append((index,result))
            futures={self.executor.submit(self.slots[i].run):i for i in ready}
            for result in failures:yield result
            yield from self.completed(futures,heartbeat,'solving')
        finally:
            # Also covers generator.close(), scoring failures and Ctrl+C during export.
            if any(not f.done() for f in futures):
                self.abort.set()
                wait(futures)

    def close(self):
        self.abort.set();self.executor.shutdown(wait=True)
        for slot in self.slots:slot.close()
        return [row for slot in self.slots for row in slot.lifetimes]

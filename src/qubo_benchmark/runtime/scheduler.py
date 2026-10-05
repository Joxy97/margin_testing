"""Bounded, device-affine workers fed by a shared solver/instance task queue."""
from collections import deque,OrderedDict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import queue
import threading
import time
from .pool import WorkerSlot


class WorkQueue:
    """Prefer resident groups; interleave solvers and steal unclaimed tail seeds."""
    def __init__(self,rows):
        groups=OrderedDict()
        for row in rows:groups.setdefault((row['solver_id'],row['instance_id']),deque()).append(row)
        solvers=list(dict.fromkeys(k[0] for k in groups))
        instances=list(dict.fromkeys(k[1] for k in groups))
        self.groups=groups
        self.waiting=deque((s,i) for i in instances for s in solvers if (s,i) in groups)
        self.affinity={};self.lock=threading.Lock()

    def take(self,lane):
        with self.lock:
            key=self.affinity.get(lane)
            if key is None or not self.groups[key]:
                while self.waiting and not self.groups[self.waiting[0]]:self.waiting.popleft()
                if self.waiting:key=self.waiting.popleft()
                else:key=max(self.groups,key=lambda k:len(self.groups[k]),default=None)
            if key is None or not self.groups[key]:return None
            self.affinity[lane]=key
            return self.groups[key].popleft()


class DeviceScheduler:
    """One supervisor thread per spawned process, one serial result consumer.

    Worker count and result backlog bound resident input/context and event memory.
    Workers can start their next trial while previous results are scored/written.
    """
    def __init__(self,workers_per_device,mode,stop_requested):
        self.devices=[d for d,count in workers_per_device.items() for _ in range(count)]
        self.abort=threading.Event();self.consumer_closed=threading.Event();self.stop_requested=stop_requested
        self.slots=[WorkerSlot(mp.get_context('spawn'),mode,self.stopped) for _ in self.devices]
        self.messages=queue.Queue(maxsize=max(1,2*len(self.slots)))
        self.active={};self.lock=threading.Lock();self.executor=None;self.futures=[]
        self.backlog_peak=0;self.dispatched=0;self.barrier=None

    def stopped(self):return self.abort.is_set() or self.stop_requested()

    def snapshot(self):
        with self.lock:return [dict(v) for v in self.active.values()]

    def _put(self,value):
        while True:
            try:
                self.messages.put(value,timeout=.05)
                self.backlog_peak=max(self.backlog_peak,self.messages.qsize());return
            except queue.Full:
                # On consumer failure, release all writers so close() cannot deadlock.
                if self.consumer_closed.is_set():return

    def _lane(self,index,work,job_for,barrier):
        slot=self.slots[index];first=True
        try:
            while not self.stopped():
                row=work.take(index)
                if row is None:
                    if first:
                        try:barrier.wait()
                        except threading.BrokenBarrierError:
                            if not self.stopped():raise
                    break
                job=job_for(row,self.devices[index])
                with self.lock:
                    self.dispatched+=1
                    self.active[index]=dict(row,device=self.devices[index],lane=index,phase='worker_setup')
                outcome=slot.prepare(job)
                if first:
                    # Nothing timed while other CUDA contexts are still starting.
                    try:barrier.wait()
                    except threading.BrokenBarrierError:
                        if not self.stopped():raise
                        if outcome is None:outcome=slot.finish(dict(error='interrupted'))
                    first=False
                with self.lock:self.active[index]['phase']='solving'
                if outcome is None:outcome=slot.run()
                outcome['actual_device']=self.devices[index];outcome['lane']=index
                if outcome['abort_gpu'] or outcome['done'].get('error')=='interrupted':self.abort.set()
                self._put((row,outcome))
        except BaseException as exc:
            self.abort.set()
            try:barrier.abort()
            except threading.BrokenBarrierError:pass
            self._put(exc)
        finally:
            if first and self.stopped():barrier.abort()
            with self.lock:self.active.pop(index,None)

    def outcomes(self,rows,job_for,heartbeat):
        work=WorkQueue(rows)
        self.executor=ThreadPoolExecutor(max_workers=len(self.slots),thread_name_prefix='gpu-lane')
        barrier=self.barrier=threading.Barrier(len(self.slots),timeout=180)
        self.futures=[self.executor.submit(self._lane,i,work,job_for,barrier) for i in range(len(self.slots))]
        next_heartbeat=0.
        try:
            while True:
                if time.perf_counter()>=next_heartbeat:
                    heartbeat('running');next_heartbeat=time.perf_counter()+1.
                try:result=self.messages.get(timeout=.05)
                except queue.Empty:
                    if all(f.done() for f in self.futures):break
                    continue
                if isinstance(result,BaseException):raise result
                yield result
            for future in self.futures:future.result()
        finally:
            self.consumer_closed.set();self.abort.set();barrier.abort()

    def close(self):
        self.consumer_closed.set();self.abort.set()
        if self.barrier is not None:self.barrier.abort()
        if self.executor is not None:self.executor.shutdown(wait=True)
        for slot in self.slots:slot.close()
        return dict(workers=[r for slot in self.slots for r in slot.lifetimes],
                    result_queue_limit=self.messages.maxsize,result_queue_peak=self.backlog_peak,
                    dispatched=self.dispatched)

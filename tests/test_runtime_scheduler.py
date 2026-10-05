"""Thread-only scheduling tests: no CUDA, child processes, or production solves."""
from collections import Counter
import threading
import time

from qubo_benchmark.runtime import scheduler


def rows(solvers=3,instances=2,seeds=25):
    return [dict(run_id=f'{s}-{i}-{seed}',solver_id=f'solver-{s}',instance_id=f'instance-{i}',seed=seed)
            for s in range(solvers) for i in range(instances) for seed in range(1,seeds+1)]


def install_slots(monkeypatch,prepare=None,delay=.0002):
    """Record resource use while exercising the real queue/supervisor threads."""
    state=dict(lock=threading.Lock(),slots=[],prepared=[],started=[],finished=[],active=Counter(),
               maximum=Counter(),total_peak=0,overlap=False)

    class FakeSlot:
        def __init__(self,context,mode,stopped):
            self.stopped=stopped;self.lifetimes=[];self.closed=False;self.count=0
            self.index=len(state['slots']);state['slots'].append(self)

        def prepare(self,job):
            self.job=job
            if prepare:prepare(self,state)
            with state['lock']:state['prepared'].append((self.index,job['run_id'],time.perf_counter()))
            return None

        def run(self):
            job=dict(self.job);device=job['device']
            with state['lock']:
                state['started'].append((self.index,job['run_id'],device,time.perf_counter()))
                state['active'][device]+=1
                state['maximum'][device]=max(state['maximum'][device],state['active'][device])
                state['total_peak']=max(state['total_peak'],sum(state['active'].values()))
                state['overlap']|=sum(state['active'].values())>1
            if delay:time.sleep(delay)
            with state['lock']:
                state['active'][device]-=1
                state['finished'].append(job['run_id'])
            self.count+=1
            return dict(done=dict(error='interrupted' if self.stopped() else None),
                        abort_gpu=False,events=[],parent_trial_started=time.perf_counter())

        def close(self):
            self.closed=True;self.lifetimes.append(dict(lane=self.index,trials=self.count))

        def finish(self,done,**kwargs):
            return dict(done=done,abort_gpu=False,events=[],parent_trial_started=time.perf_counter())

    monkeypatch.setattr(scheduler,'WorkerSlot',FakeSlot)
    return state


def job_for(row,device):return dict(row,device=device)


def test_all_solver_instance_seeds_exactly_once_with_device_limits(monkeypatch):
    state=install_slots(monkeypatch)
    limits={'cuda:0':2,'cuda:1':1,'cuda:2':1,'cuda:3':2}
    workers=scheduler.DeviceScheduler(limits,'persistent',lambda:False)
    planned=rows(solvers=23,instances=3,seeds=25)
    try:
        results=list(workers.outcomes(planned,job_for,lambda _:None))
        assert Counter(row['run_id'] for row,_ in results)==Counter(row['run_id'] for row in planned)
        assert len(results)==1725
        assert {outcome['actual_device'] for _,outcome in results}==set(limits)
        assert all(state['maximum'][d]<=limit for d,limit in limits.items())
        assert state['overlap'] and state['total_peak']<=sum(limits.values())
        # One lane's immutable cache affinity stays within a solver/instance group
        # until that group's assigned seed queue is exhausted.
        by_lane={index:[] for index in range(len(workers.slots))}
        for index,run_id,_,_ in state['started']:by_lane[index].append(run_id.rsplit('-',1)[0])
        for groups in by_lane.values():
            transitions=[key for i,key in enumerate(groups) if i==0 or key!=groups[i-1]]
            assert len(transitions)==len(set(transitions))
    finally:report=workers.close()
    assert report['dispatched']==1725 and all(s.closed for s in state['slots'])


def test_initial_readiness_barrier_and_simultaneous_distinct_solvers(monkeypatch):
    def prepare(slot,state):
        if slot.count==0:time.sleep(.005*(slot.index+1))
    state=install_slots(monkeypatch,prepare,delay=.01)
    workers=scheduler.DeviceScheduler({'cuda:0':2,'cuda:1':2},'persistent',lambda:False)
    try:
        results=list(workers.outcomes(rows(solvers=4,instances=1,seeds=2),job_for,lambda _:None))
        first_prepared={}
        for lane,_,stamp in state['prepared']:first_prepared.setdefault(lane,stamp)
        assert min(item[3] for item in state['started'])>=max(first_prepared.values())
        assert len({run_id.split('-')[0] for _,run_id,_,_ in state['started'][:4]})==4
        assert len(results)==8 and state['total_peak']==4
    finally:workers.close()


def test_slow_consumer_bounds_finished_backlog_and_keeps_workers_reusable(monkeypatch):
    state=install_slots(monkeypatch)
    workers=scheduler.DeviceScheduler({'cuda:0':2},'persistent',lambda:False)
    stream=workers.outcomes(rows(solvers=2,instances=1,seeds=25),job_for,lambda _:None)
    try:
        first=next(stream);time.sleep(.05)
        # Two completed results per lane in the queue, one pending result per
        # lane, and the one result already consumed: memory never grows with N.
        assert workers.messages.qsize()==workers.messages.maxsize==4
        assert len(state['started'])<=3*len(workers.slots)+1
        results=[first,*stream]
        assert len(results)==50 and len(state['slots'])==2
        assert all(slot.count>1 for slot in state['slots'])
    finally:stream.close();workers.close()


def test_operational_abort_drains_all_finished_results_with_full_queue(monkeypatch):
    state=install_slots(monkeypatch)
    workers=scheduler.DeviceScheduler({'cuda:0':2},'persistent',lambda:False)
    stream=workers.outcomes(rows(solvers=2,instances=1,seeds=25),job_for,lambda _:None)
    try:
        first=next(stream);time.sleep(.05)
        assert workers.messages.full()
        # Operational abort still has a live serial consumer: completed seeds
        # must be persisted, even though no new jobs should start.
        workers.abort.set();time.sleep(.07)
        results=[first,*stream]
        assert Counter(row['run_id'] for row,_ in results)==Counter(state['finished'])
        assert len(results)<50
    finally:stream.close();workers.close()


def test_stop_during_initial_setup_and_consumer_close_join_workers(monkeypatch):
    stop=threading.Event()
    def prepare(slot,state):stop.set()
    state=install_slots(monkeypatch,prepare)
    # Bound a regression's wait without making the test hang for180seconds.
    original_barrier=threading.Barrier
    monkeypatch.setattr(scheduler.threading,'Barrier',lambda parties,timeout:original_barrier(parties,timeout=.4))
    workers=scheduler.DeviceScheduler({'cuda:0':4},'persistent',stop.is_set)
    began=time.perf_counter()
    try:list(workers.outcomes(rows(),job_for,lambda _:None))
    finally:workers.close()
    assert time.perf_counter()-began<.3
    assert all(f.done() for f in workers.futures)
    assert all(slot.closed for slot in state['slots'])

    # A consumer exception/early close with a full queue must release blocked
    # producers and leave no live supervisor threads or process slots.
    state=install_slots(monkeypatch)
    workers=scheduler.DeviceScheduler({'cuda:0':2},'persistent',lambda:False)
    stream=workers.outcomes(rows(),job_for,lambda _:None)
    next(stream);time.sleep(.05);began=time.perf_counter()
    stream.close();workers.close()
    assert time.perf_counter()-began<.3
    assert all(f.done() for f in workers.futures)
    assert all(slot.closed for slot in state['slots'])

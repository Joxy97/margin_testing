"""Bounded measured worker admission; pilot trials never enter benchmark statistics."""
from contextlib import closing
import time
from .pool import SeedPool
from .selection import finalize_execution
from .common import utc


def candidates(limit):
    value=1
    while value<=limit:
        yield value
        value*=2


def calibrate(prepared,args):
    core=prepared['core'];policy=core['protocol']['execution'];devices=core['actual_devices']
    if (args.resume or args.dry_run or policy['seed_workers_requested']!='auto' or
        not policy['autotune_enabled'] or not all(d.startswith('cuda') for d in devices)):
        return prepared
    limits=prepared['capacity']
    maximum=min(policy['auto_concurrency_max_per_device'],len(core['seeds']),
                limits['maximum_workers_estimate']//len(devices),
                *[limits['workers_per_device'][d] for d in devices])
    if maximum<=1:return prepared
    campaign_seeds=set(core['seeds']);pilot_seeds=[];candidate_seed=2**31
    while len(pilot_seeds)<2*maximum*len(devices):
        if candidate_seed not in campaign_seeds:pilot_seeds.append(candidate_seed)
        candidate_seed+=1
    priority=['lib_greedy_local_search','lib_simulated_bifurcation','lib_random_search']
    solvers=[s for s in priority if s in core['solvers']][:2] or [next(iter(core['solvers']))]
    item=max(prepared['selected'],key=lambda i:i['n_couplings'])
    # A brief workload sample; the final campaign retains its exact requested budget.
    pilot_budget=min(.5,max(.1,core['budget_s']))
    attempts=[];best_count=1;best_rate=0.;started=time.perf_counter()
    print(f'Auto calibration: {devices}; candidates={list(candidates(maximum))} per GPU; '
          f'pilot budget={pilot_budget}s; pilot results excluded from statistics',flush=True)
    for count in candidates(maximum):
        pool=SeedPool(count*len(devices),'persistent',lambda:False)
        records=[];warm_fail=False;measured=0.;valid=0
        try:
            for repeat in range(2):  # First pass warms infrastructure; second is measured.
                for solver in solvers:
                    jobs=[dict(solver=solver,parameters=core['solvers'][solver],npz=item['npz'],
                        seed=pilot_seeds[repeat*maximum*len(devices)+i],budget=pilot_budget,device=device,
                        **{k:policy[k] for k in ('cpu_threads','watchdog_grace_s','setup_timeout_s',
                           'memory_sampling_interval_s','heartbeat_interval_s')})
                        for i,device in enumerate(d for d in devices for _ in range(count))]
                    began=time.perf_counter()
                    with closing(pool.wave(jobs,lambda phase:None)) as stream:results=list(stream)
                    elapsed=time.perf_counter()-began
                    failed=(len(results)!=len(jobs) or {i for i,_ in results}!=set(range(len(jobs))) or
                            any(r['done'].get('error') for _,r in results))
                    if repeat:
                        measured+=elapsed
                        valid+=sum(not r['done'].get('error') for _,r in results)
                    records.extend(dict(device=jobs[i]['device'],solver=solver,warmup=not repeat,
                        error=r['done'].get('error'),solve_s=r['done'].get('actual_solve_wall_s'),
                        gpu_peak=r['done'].get('gpu_reserved_peak_bytes')) for i,r in results)
                    if failed:warm_fail=True;break
                if warm_fail:break
        finally:pool.close()
        rate=valid/measured if measured else 0.
        attempts.append(dict(workers_per_device=count,valid_rate=rate,measured_s=measured,
                             failed=warm_fail,trials=records))
        print(f'Auto calibration: workers/GPU={count}, completed trials/s={rate:.2f}, '
              f'worker failure={warm_fail}',flush=True)
        if warm_fail:
            if count==1:raise ValueError('Single-worker GPU calibration failed; inspect GPU environment before benchmarking')
            break
        # A small gain does not justify more contention and memory use.
        if rate>best_rate*policy.get('auto_concurrency_min_speedup',1.1):best_count=count;best_rate=rate
        else:break
    counts={d:best_count for d in devices}
    metadata=dict(created_at=utc(),seconds=time.perf_counter()-started,selected=counts,
        pilot_budget_s=pilot_budget,solvers=solvers,instance=item['entry']['instance_id'],attempts=attempts,
        qualifier='Bounded throughput sample; not a global optimum. Candidate quality is not used to choose concurrency. '
                  'Campaign parameters and per-seed wall budgets are unchanged; concurrent timing includes contention.')
    finalize_execution(prepared,counts,metadata)
    print(f'Auto calibration selected {counts}',flush=True)
    return prepared

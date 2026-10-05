"""Sequential isolated workers with a ready barrier, live heartbeat and watchdog."""
import multiprocessing as mp
import time
import uuid
from .common import utc
from .worker import execute

def trial(job,heartbeat,stop_requested):
    import psutil
    context=mp.get_context('spawn')
    parent,child=context.Pipe()
    stop=context.Event()
    process=context.Process(target=execute,args=(child,stop,job))
    began=time.perf_counter();events=[];setup={};done={};origin=None;started=None
    rss_start=None;rss_peak=None;next_sample=0.;next_heartbeat=0.;forced=False;ready=False
    solved=None;cleanup_origin=None;stopping=None
    process.start();child.close()
    try:
        while True:
            now=time.perf_counter()
            if stop_requested():stop.set()
            if now>=next_sample:
                try:
                    p=psutil.Process(process.pid)
                    rss=sum(x.memory_info().rss for x in [p]+p.children(recursive=True))
                    if ready and rss_start is None:rss_start=rss
                    if ready:rss_peak=max(rss_peak or rss,rss)
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                next_sample=now+job['memory_sampling_interval_s']
            if now>=next_heartbeat:
                heartbeat('solving' if origin else 'worker_setup')
                next_heartbeat=now+job['heartbeat_interval_s']
            if parent.poll(.005):
                try:message=parent.recv()
                except EOFError:
                    if not done:done=dict(error='worker_crash',error_message='Worker closed pipe without a result')
                    break
                kind=message.pop('kind')
                if kind=='ready':
                    ready=True;setup=message['setup'];setup['worker_ready_wall_s']=time.perf_counter()-began
                    try:
                        proc=psutil.Process(process.pid)
                        rss_start=sum(x.memory_info().rss for x in [proc]+proc.children(recursive=True))
                        rss_peak=rss_start
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                    parent.send('go')
                elif kind=='started':origin=message['origin_ns'];started=message['started_at_utc']
                elif kind=='events':events.extend(message['events'])
                elif kind=='stopping':stopping=message
                elif kind=='solved':solved=message;cleanup_origin=time.perf_counter()
                elif kind=='done':done=message;break
                elif kind=='fatal':
                    done=dict(message,error='setup_error' if not ready else 'worker_error');break
            elif not process.is_alive():
                done=dict(error='worker_crash',error_message=f'Worker exit code {process.exitcode}');break
            # A completed result can be buffered behind captures while the parent
            # is busy with progress/OS scheduling. Consume it before enforcing
            # the watchdog; the immutable capture timestamps still decide credit.
            if parent.poll():continue
            now=time.perf_counter()
            elapsed=(time.perf_counter_ns()-origin)/1e9 if origin else None
            if solved is not None:
                if now-cleanup_origin>job['setup_timeout_s']:
                    forced=True;process.terminate();process.join(2)
                    done=dict(solved,error='cleanup_timeout',error_message='Worker cleanup timed out after solver returned')
                    break
                continue
            if ((origin is not None and elapsed>job['budget']+job['watchdog_grace_s']) or
                (origin is None and now-began>job['setup_timeout_s'])):
                forced=True;stop.set();process.terminate();process.join(2)
                if process.is_alive():process.kill();process.join(2)
                done=dict(error='watchdog_timeout' if origin else 'setup_timeout',
                          error_message=('Supervisor cutoff during '+('device drain' if stopping else 'solver work')+
                                         '; no post-deadline quality credit'),
                          stop_reason='watchdog',actual_solve_wall_s=elapsed)
                break
        cleanup_started=time.perf_counter()
        process.join(2)
        if process.is_alive():
            forced=True;process.terminate();process.join(2)
            if process.is_alive():process.kill();process.join()
        cleanup=time.perf_counter()-cleanup_started
    finally:
        if process.is_alive():process.terminate();process.join()
        parent.close()
    if stop_requested() or done.get('stop_reason')=='interrupted':done['error']='interrupted'
    return dict(events=events,done=done,setup=setup,started_at_utc=started,
                finished_at_utc=done.get('finished_at_utc',utc()),cleanup_s=cleanup+done.get('worker_cleanup_s',0.),
                worker_end_to_end_s=time.perf_counter()-began,cpu_rss_start_bytes=rss_start,
                cpu_rss_peak_sampled_bytes=rss_peak,
                abort_gpu=job['device'].startswith('cuda') and (forced or done.get('error') in ('worker_crash','worker_error')))

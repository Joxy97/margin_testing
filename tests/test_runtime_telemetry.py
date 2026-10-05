"""Hardware telemetry must never block progress or require CUDA initialization."""
import csv
import subprocess
import threading
import time
from types import SimpleNamespace

from qubo_benchmark.runtime.telemetry import Telemetry,parse_nvidia_csv


OUTPUT='0, GPU-first, RTX 3090, 87, 22, 1024, 24576, 211.5\n1, GPU-second, RTX 3090, 0, 0, 8, 24576, [N/A]\n'


def wait_for_sample(sampler):
    deadline=time.monotonic()+2
    while sampler.snapshot()['sampled_at_utc'] is None and time.monotonic()<deadline:time.sleep(.005)
    return sampler.snapshot()


def test_gpu_csv_parsing_and_uuid_mapping(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1,0')
    calls=[]
    def query(command,**kwargs):
        calls.append((command,kwargs));return SimpleNamespace(stdout=OUTPUT)
    monkeypatch.setattr(subprocess,'run',query)
    path=tmp_path/'hardware.csv'
    sampler=Telemetry(['cuda:0'],path,gpu_metadata=[dict(device='cuda:0',uuid='second')]).start()
    try:sample=wait_for_sample(sampler)
    finally:sampler.close()
    assert sample['status']=='ok' and sample['gpus'][0]['index']==1
    assert sample['gpus'][0]['device_mapping']=='uuid'
    assert sample['gpus'][0]['power_watts'] is None
    assert 'not measured TFLOPS' in sample['metric_qualifier']
    assert calls[0][1]['timeout']==2. and calls[0][0][0]=='nvidia-smi'
    rows=list(csv.DictReader(path.open()))
    assert len(rows)==1 and rows[0]['device']=='cuda:0'
    assert parse_nvidia_csv(OUTPUT)[0]['memory_used_bytes']==1024*2**20
    sample['gpus'].clear()
    assert len(sampler.snapshot()['gpus'])==1  # Caller cannot mutate retained state.


def test_slow_query_does_not_block_snapshot(monkeypatch):
    entered=threading.Event();release=threading.Event()
    def query(*args,**kwargs):
        entered.set();release.wait(1);return SimpleNamespace(stdout=OUTPUT)
    monkeypatch.setattr(subprocess,'run',query)
    sampler=Telemetry(['cuda:0']).start()
    try:
        assert entered.wait(1)
        began=time.monotonic();sample=sampler.snapshot()
        assert time.monotonic()-began<.1 and sample['status']=='pending'
    finally:release.set();sampler.close()


def test_failed_gpu_query_is_nonfatal(monkeypatch):
    def query(*args,**kwargs):raise subprocess.TimeoutExpired('nvidia-smi',2)
    monkeypatch.setattr(subprocess,'run',query)
    sampler=Telemetry(['cuda:0']).start()
    try:sample=wait_for_sample(sampler)
    finally:sampler.close()
    assert sample['status']=='unavailable' and 'GPU telemetry unavailable' in sample['error']
    assert sample['parent_rss_bytes']>0


def test_cpu_sampler_never_queries_nvidia_and_appends_on_resume(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('CPU run must not query nvidia-smi')
    monkeypatch.setattr(subprocess,'run',forbidden)
    path=tmp_path/'hardware.csv'
    for _ in range(2):
        sampler=Telemetry(['cpu'],path).start()
        try:assert wait_for_sample(sampler)['status']=='ok'
        finally:sampler.close()
    assert len(list(csv.DictReader(path.open())))==2

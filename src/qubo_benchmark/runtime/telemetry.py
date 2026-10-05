"""Optional, bounded hardware observations outside all trial timing paths.

GPU utilization measures device activity, not achieved or theoretical TFLOPS.
The sampler owns its subprocess and CSV writes; progress reads a small snapshot.
"""
import copy
import csv
import io
import os
from pathlib import Path
import subprocess
import threading
import time

from .common import utc


QUALIFIER='GPU utilization is activity percentage, not measured TFLOPS; device metrics include other processes.'
QUERY='index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw'
FIELDS=['sampled_at_utc','status','device','index','uuid','name','gpu_utilization_percent',
        'memory_utilization_percent','memory_used_bytes','memory_total_bytes','power_watts',
        'parent_cpu_percent','parent_rss_bytes','error','metric_qualifier']


def _number(value,scale=1):
    try:return float(value.strip())*scale
    except (TypeError,ValueError):return None


def parse_nvidia_csv(output):
    """Parse nounits output, retaining unsupported metrics as nulls."""
    rows=[]
    for row in csv.reader(io.StringIO(output)):
        if not row:continue
        if len(row)!=8:raise ValueError('Unexpected nvidia-smi telemetry column count')
        index,uuid,name,gpu,memory,used,total,power=(item.strip() for item in row)
        rows.append(dict(index=int(index),uuid=uuid,name=name,
            gpu_utilization_percent=_number(gpu),memory_utilization_percent=_number(memory),
            memory_used_bytes=_number(used,2**20),memory_total_bytes=_number(total,2**20),
            power_watts=_number(power)))
    return rows


class Telemetry:
    """One background sampler; retains only the latest snapshot.

    Pass environment['gpus'] as gpu_metadata to match CUDA logical devices by
    UUID. Without UUID metadata the CUDA_VISIBLE_DEVICES/index mapping is best
    effort and is explicitly labeled as such. No CUDA context is created here.
    """
    def __init__(self,devices,output_path=None,*,interval_s=2.,gpu_metadata=None):
        if interval_s<=0:raise ValueError('Telemetry interval must be positive')
        self.devices=list(devices);self.output_path=Path(output_path) if output_path else None
        self.interval_s=float(interval_s);self._stop=threading.Event();self._lock=threading.Lock()
        self._thread=None;self._sampled_monotonic=None;self._process=None;self._csv_checked=False
        self._latest=dict(sampled_at_utc=None,status='pending',gpus=[],parent_cpu_percent=None,
            parent_rss_bytes=None,error=None,metric_qualifier=QUALIFIER)
        self._uuids={row['device']:str(row['uuid']) for row in (gpu_metadata or [])
                     if row.get('uuid') and row['uuid']!='unavailable'}
        self._visible=os.environ.get('CUDA_VISIBLE_DEVICES')

    def start(self):
        if self._thread is None:
            self._thread=threading.Thread(target=self._loop,name='hardware-telemetry',daemon=True)
            self._thread.start()
        return self

    def snapshot(self):
        with self._lock:
            snapshot=copy.deepcopy(self._latest);sampled=self._sampled_monotonic
        snapshot['sample_age_s']=None if sampled is None else max(0.,time.monotonic()-sampled)
        return snapshot

    def close(self):
        self._stop.set()
        if self._thread is not None:self._thread.join(timeout=3.)
        return self.snapshot()

    def _selected(self,rows):
        selected=[];missing=[]
        tokens=self._visible.split(',') if self._visible is not None else None
        for device in self.devices:
            if not device.startswith('cuda'):continue
            logical=int(device.split(':',1)[1]) if ':' in device else 0
            uuid=self._uuids.get(device)
            token=tokens[logical].strip() if tokens is not None and logical<len(tokens) else str(logical)
            if uuid:
                # Torch properties omit NVIDIA's "GPU-" UUID prefix.
                canonical=uuid.removeprefix('GPU-').lower()
                matches=[row for row in rows if row['uuid'].removeprefix('GPU-').lower()==canonical]
                mapping='uuid'
            elif token.startswith(('GPU-','MIG-')):
                matches=[row for row in rows if row['uuid'].startswith(token)]
                mapping='CUDA_VISIBLE_DEVICES uuid'
            else:
                matches=[row for row in rows if str(row['index'])==token]
                mapping='CUDA_VISIBLE_DEVICES/index best effort'
            if len(matches)==1:selected.append(dict(matches[0],device=device,device_mapping=mapping))
            else:missing.append(device)
        return selected,missing

    def _sample(self):
        snapshot=dict(sampled_at_utc=utc(),status='ok',gpus=[],parent_cpu_percent=None,
            parent_rss_bytes=None,error=None,metric_qualifier=QUALIFIER)
        errors=[]
        try:
            import psutil
            if self._process is None:self._process=psutil.Process()
            snapshot.update(parent_cpu_percent=self._process.cpu_percent(interval=None),
                            parent_rss_bytes=self._process.memory_info().rss)
        except Exception as exc:errors.append('CPU telemetry unavailable: '+str(exc)[:300])
        if any(device.startswith('cuda') for device in self.devices):
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu='+QUERY,'--format=csv,noheader,nounits'],
                    capture_output=True,text=True,check=True,timeout=2.)
                snapshot['gpus'],missing=self._selected(parse_nvidia_csv(result.stdout))
                if missing:errors.append('No unambiguous GPU telemetry for '+', '.join(missing))
            except (OSError,subprocess.SubprocessError,ValueError) as exc:
                errors.append('GPU telemetry unavailable: '+str(exc)[:300])
        if errors:snapshot.update(status='partial' if snapshot['gpus'] else 'unavailable',error='; '.join(errors))
        return snapshot

    def _append(self,snapshot):
        if self.output_path is None:return
        self.output_path.parent.mkdir(parents=True,exist_ok=True)
        existing=self.output_path.exists() and self.output_path.stat().st_size>0
        if existing and not self._csv_checked:
            with self.output_path.open(newline='',encoding='utf-8') as source:
                if next(csv.reader(source),None)!=FIELDS:raise ValueError('Telemetry CSV header mismatch')
        self._csv_checked=True
        with self.output_path.open('a',newline='',encoding='utf-8') as target:
            writer=csv.DictWriter(target,fieldnames=FIELDS,extrasaction='ignore')
            if not existing:writer.writeheader()
            common={key:value for key,value in snapshot.items() if key!='gpus'}
            for gpu in snapshot['gpus'] or [{}]:writer.writerow(dict(common,**gpu))

    def _loop(self):
        while not self._stop.is_set():
            began=time.monotonic()
            try:snapshot=self._sample()
            except Exception as exc:
                snapshot=dict(sampled_at_utc=utc(),status='unavailable',gpus=[],
                    parent_cpu_percent=None,parent_rss_bytes=None,error='Telemetry failed: '+str(exc)[:300],
                    metric_qualifier=QUALIFIER)
            try:self._append(snapshot)
            except (OSError,ValueError) as exc:
                snapshot['status']='partial'
                snapshot['error']='; '.join(filter(None,[snapshot.get('error'),'Telemetry CSV: '+str(exc)[:300]]))
            with self._lock:
                self._latest=snapshot;self._sampled_monotonic=time.monotonic()
            self._stop.wait(max(0.,self.interval_s-(time.monotonic()-began)))

"""Protocol, identities and crash-safe serialization."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'configs'
SCHEMA = 4

def utc():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)
    if os.name!='nt':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def table(path, rows, fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    fields=fields or sorted({key for row in rows for key in row}) or ['schema_version']
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,quoting=csv.QUOTE_ALL,extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({key:(canonical(value) if isinstance(value,(dict,list,tuple)) else value)
                             for key,value in row.items()})
        handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)

def flattened(value, prefix=''):
    for key,item in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):yield from flattened(item,name)
        else:yield dict(path=name,type=type(item).__name__,value=item)

def source_snapshot_hash():
    files=[]
    for directory in ('src/qubo_solvers','src/qubo_benchmark'):
        files.extend((ROOT/directory).rglob('*.py'))
    files.extend(ROOT/name for name in ('run_benchmark.py','run_benchmark_grid.py',
                                        'monitor_benchmark.py','aggregate_benchmarks.py'))
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):filehash(p) for p in sorted(files) if p.exists()}
    return digest(hashes)

def source_identity():
    snapshot=source_snapshot_hash()
    def git(*args):
        result=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        return result.stdout.strip() if result.returncode==0 else None
    return dict(code_commit=git('rev-parse','HEAD'),code_dirty=bool(git('status','--porcelain')),
                code_snapshot_hash=snapshot)

class DirectoryLock:
    def __init__(self,path): self.path=Path(path)/'.lock'
    def __enter__(self):
        if self.path.exists():
            info=read(self.path)
            if info['host']!=socket.gethostname():raise ValueError('Foreign-host lock; verify old runner has stopped')
            import psutil
            if psutil.pid_exists(info['pid']):raise ValueError('Test directory is locked by a live process')
            self.path.unlink()
        fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w') as handle:
            json.dump(dict(pid=os.getpid(),host=socket.gethostname(),created_at=utc()),handle)
            handle.flush();os.fsync(handle.fileno())
        return self
    def __exit__(self,*args): self.path.unlink(missing_ok=True)

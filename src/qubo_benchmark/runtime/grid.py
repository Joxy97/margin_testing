"""Sequential 36-configuration campaign. Never schedules concurrent device trials."""
import argparse
from pathlib import Path
import subprocess
import sys
import uuid
from .common import ROOT,CONFIG,read,atomic,digest,DirectoryLock,source_identity
from .selection import grid,environment

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instances',choices=['all','representative'],default='all')
    p.add_argument('--config',default=str(CONFIG/'benchmark_solvers.json'))
    p.add_argument('--device',default='auto');p.add_argument('--solvers',default='all')
    p.add_argument('--output-root',default='results');p.add_argument('--runs',type=int,default=100)
    p.add_argument('--seed-start',type=int,default=0);p.add_argument('--seeds-file')
    p.add_argument('--require-gpu',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume');p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args(argv)
    common=['--instances',args.instances,'--config',str(Path(args.config).resolve()),'--device',args.device,
            '--solvers',args.solvers,'--runs',str(args.runs),'--seed-start',str(args.seed_start)]
    if args.seeds_file:common+=['--seeds-file',str(Path(args.seeds_file).resolve())]
    if args.require_gpu:common+=['--require-gpu']
    commands=[[sys.executable,str(ROOT/'run_benchmark.py'),str(n),density,str(t),*common] for n,density,t in grid()]
    if args.dry_run:
        for command in commands:print(subprocess.list2cmdline(command+['--dry-run']))
        print('36 sequential configurations; no solver executed')
        return 0
    env=environment(args.device,args.require_gpu)
    contract=dict(common=common,configuration=read(args.config),environment=env,source=source_identity(),
                  seeds=read(args.seeds_file) if args.seeds_file else list(range(args.seed_start,args.seed_start+args.runs)),
                  protocol=read(CONFIG/'benchmark_protocol.json'))
    contract['source'].pop('code_dirty',None)
    directory=Path(args.resume) if args.resume else Path(args.output_root)/('campaign_'+uuid.uuid4().hex[:12])
    directory.mkdir(parents=True,exist_ok=bool(args.resume))
    manifest=dict(identity=digest(contract),contract=contract,configurations=[list(c) for c in grid()])
    if args.resume:
        if read(directory/'campaign.json')!=manifest:raise ValueError('Campaign resume identity mismatch; use the original arguments')
    else:atomic(directory/'campaign.json',manifest)
    failures=False
    with DirectoryLock(directory):
        for index,command in enumerate(commands):
            root=directory/f'configuration_{index:02d}'
            existing=list(root.glob('*/experiment.json')) if root.exists() else []
            if len(existing)>1:raise ValueError('Ambiguous child experiments')
            invocation=command+['--output-root',str(root)]
            if existing:
                invocation+=['--resume',str(existing[0].parent)]
                if args.retry_failed:invocation+=['--retry-failed']
            result=subprocess.run(invocation,cwd=ROOT)
            atomic(directory/'status.json',dict(configuration_index=index,exit_code=result.returncode,total=36))
            if result.returncode not in (0,1):return result.returncode
            failures |= result.returncode==1
    return 1 if failures else 0

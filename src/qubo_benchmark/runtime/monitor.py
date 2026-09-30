"""Read-only saved-status monitor; never imports a solver or queries a GPU."""
import argparse
import json
import time
from pathlib import Path
from .common import read

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory');p.add_argument('--once',action='store_true')
    p.add_argument('--interval',type=float,default=2)
    args=p.parse_args(argv)
    if args.interval<=0:p.error('interval must be positive')
    previous=None
    try:
        while True:
            path=Path(args.directory)/'status.json'
            if path.exists():
                status=read(path)
                if status!=previous:print(json.dumps(status,indent=2),flush=True);previous=status
                if args.once or status['phase'] in ('complete','complete_with_failures','complete_with_unavailable_solvers','interrupted','gpu_worker_lost_resume_required','source_changed'):return 0
            elif args.once:p.error('status.json does not exist')
            time.sleep(args.interval)
    except KeyboardInterrupt:return 0

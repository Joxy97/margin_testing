"""Run with PYTHONPATH=src python -m qubo_benchmark COMMAND."""
import argparse
import json
from .catalog import DEFAULT_DATA, loadCatalog
from .pipeline import download, normalize, validate
from .runner import run, summarize
from .storage import readJson


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default=str(DEFAULT_DATA))
    commands=parser.add_subparsers(dest='command',required=True)
    sub=commands.add_parser('download');sub.add_argument('--offline',action='store_true')
    sub.add_argument('--recover-graph-ranges',action='store_true',help='Explicitly recover invalid Glasgow graphs using HTTP byte ranges from the same publisher')
    for name in ('normalize','validate','list','solvers'): commands.add_parser(name)
    sub=commands.add_parser('run');sub.add_argument('--config',required=True);sub.add_argument('--output')
    sub=commands.add_parser('summarize');sub.add_argument('output')
    args=parser.parse_args()
    if args.command=='download':
        if args.recover_graph_ranges:
            if args.offline:parser.error('Range recovery requires network')
            from .storage import SourceCache
            from .formats import parseGraph
            for entry in loadCatalog(args.data)['instances']:
                if entry['reference_solution_vector_url']:
                    cache=SourceCache(args.data);url=entry['input_url']
                    try:parseGraph(cache.read(url)[0])
                    except (OSError,ValueError):
                        try:
                            _,record=cache.recoverRanges(url,url.replace('https:','http:',1),parseGraph)
                            print(f"Recovered {entry['instance_id']}: {record['bytes']} bytes",flush=True)
                        except Exception as exc:print(f"Recovery failed {entry['instance_id']}: {exc}",flush=True)
        result=download(args.data,args.offline);success=all(r['status']!='error' for r in result)
    elif args.command=='normalize':
        result=normalize(args.data);success=all(r['status']=='ready' for r in result['instances'])
    elif args.command=='validate':
        result=validate(args.data);success=result['passed']==37
    elif args.command=='list':result=loadCatalog(args.data);success=True
    elif args.command=='solvers':
        from .adapters import solverCapabilities
        result=solverCapabilities();success=True
    elif args.command=='run':
        result=run(readJson(args.config),args.data,args.output);success=result['completed']+result['unsupported']==result['total']
    else:result=summarize(args.output);success=True
    print(json.dumps(result,indent=2));return 0 if success else 1


if __name__=='__main__': raise SystemExit(main())

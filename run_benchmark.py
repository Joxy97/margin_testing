"""Offline QUBO runtime benchmark entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))

if __name__ == '__main__':
    flags = ('--continuous','--variable-count','--n','--density','--sparse','--dense',
             '--doctor','--json','--keep-raw-results','--no-keep-raw-results')
    compact = not sys.argv[1:] or any(arg.split('=')[0] in flags for arg in sys.argv[1:])
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        compact = True
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == '--resume' and index+1 < len(sys.argv):
            resume_path = sys.argv[index+1]
        elif arg.startswith('--resume='):
            resume_path = arg.partition('=')[2]
        else:
            continue
        if (Path(resume_path)/'manifest.json').exists():
            compact = True
    if compact:
        from qubo_benchmark.runtime.continuous_cli import main
    else:
        from qubo_benchmark.runtime.cli import main
    raise SystemExit(main())

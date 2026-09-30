"""Offline QUBO runtime benchmark entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))

if __name__ == '__main__':
    from qubo_benchmark.runtime.aggregate import main
    raise SystemExit(main())

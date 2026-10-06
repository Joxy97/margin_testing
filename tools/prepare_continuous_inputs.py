"""Prepare only the explicitly catalogued public continuous benchmark inputs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qubo_benchmark.continuous_catalog import prepare_continuous_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Require existing hash-verified public source caches')
    args = parser.parse_args()
    report = prepare_continuous_inputs(offline=args.offline)
    print(json.dumps(report, indent=2))
    return 0 if all(r['status'] == 'ready' for r in report) else 2


if __name__ == '__main__':
    raise SystemExit(main())

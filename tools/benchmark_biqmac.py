"""Persistent, resumable SBM/SVL/Transverse Route sweep over BiqMac MaxCut."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import queue
import re
from statistics import median
import tarfile
import time
import traceback

import numpy as np

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem


SOLVERS = {'SBM': 'torch_sbm', 'SVL': 'torch_svl', 'TRF': 'torch_transverse_route'}
ARCHIVE_URL = 'https://biqmac.aau.at/library/tar_files/mac_all.tar.gz'
REFERENCE_URL = 'https://biqmac.aau.at/biqmaclib.tex'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomicJson(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomicCsv(path, rows, fields):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def readGraph(path):
    """Encode minus the cut, including signed/parallel edges and ignored loops."""
    lines = [line.split('#', 1)[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    if not lines or len(lines[0].split()) != 2:
        raise ValueError(f'{path}: expected vertex and edge counts')
    n, m = map(int, lines[0].split())
    if n < 1 or m < 0 or len(lines) != m + 1:
        raise ValueError(f'{path}: invalid dimensions or edge count')
    values = np.array([list(map(float, line.split())) for line in lines[1:]], dtype=float).reshape(m, 3)
    if not np.isfinite(values).all():
        raise ValueError(f'{path}: non-finite edge data')
    if m and (np.any(values[:, :2] != np.floor(values[:, :2]))
              or np.min(values[:, :2]) < 1 or np.max(values[:, :2]) > n):
        raise ValueError(f'{path}: invalid 1-based vertex index')
    heads, tails = values[:, 0].astype(np.uint32) - 1, values[:, 1].astype(np.uint32) - 1
    weights = values[:, 2].copy()
    keep = heads != tails
    heads, tails, weights = heads[keep], tails[keep], weights[keep]
    linear = np.zeros(n)
    np.add.at(linear, heads, -weights)
    np.add.at(linear, tails, -weights)
    problem = QUBOProblem(linear, heads, tails, 2 * weights)
    return problem, heads, tails, weights


def prepare(archive, references, destination):
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) != 2 or parts[0] not in ('rudy', 'ising') or '..' in parts:
                raise ValueError(f'Unexpected archive member: {member.name}')
            if member.size > 10_000_000:
                raise ValueError('Unexpectedly large graph file')
            payload = bundle.extractfile(member).read()
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            problem, heads, tails, weights = readGraph(target)
            # Independent direct-cut check before any timed solver invocation.
            bits = np.random.default_rng(13).integers(0, 2, problem.variableCount)
            cut = float(weights[bits[heads] != bits[tails]].sum())
            np.testing.assert_allclose(problem.energy(bits), -cut, rtol=0, atol=1e-6)
            entries.append(dict(name=parts[1], family=parts[0], path=member.name,
                vertices=problem.variableCount, edges=len(weights),
                sha256=hashlib.sha256(payload).hexdigest()))
    names = {entry['name'] for entry in entries}
    if len(names) != len(entries):
        raise ValueError('Graph names must be unique')
    optimum = {}
    for line in references.read_text().replace('\\_', '_').splitlines():
        parts = line.split('&')
        if len(parts) == 2 and parts[0].strip() in names:
            match = re.fullmatch(r'\s*(-?\d+)\s*\\\\\s*', parts[1])
            if match:
                optimum[parts[0].strip()] = int(match.group(1))
    if optimum.keys() != names:
        raise ValueError(f'Missing reference optima: {sorted(names - optimum.keys())}')
    for entry in entries:
        entry['reference_cut'] = optimum[entry['name']]
    manifest = dict(archive_url=ARCHIVE_URL, reference_url=REFERENCE_URL,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(references.read_bytes()).hexdigest(),
        instances=sorted(entries, key=lambda e: (e['vertices'], e['family'], e['name'])))
    atomicJson(destination / 'manifest.json', manifest)
    print(f'[{stamp()}] PREPARED {len(entries)} verified MaxCut instances; all reference optima found', flush=True)


def parameters(solver, runs, steps, seed):
    p = dict(runs=runs, steps=steps, seed=seed, dtype='float32', run_batch_size=runs,
             energy_chunk_size=8192)
    if solver == 'TRF':
        p.update(cuda_graph=True, candidate_interval=25)
    return p


def worker(device, tasks, events, inputDirectory, repeats, seed):
    try:
        import torch
        from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        solvers = {name: BQMSolverFactory.create(kind, {'device': device}) for name, kind in SOLVERS.items()}
        cache = {}
        def sync():
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
        # Warm up every solver/context outside the measurement interval.
        warm = QUBOProblem(np.array([-1., -1.]), np.array([0], dtype=np.uint32),
                           np.array([1], dtype=np.uint32), np.array([2.]))
        for name, solver in solvers.items():
            events.put(dict(type='warmup', device=device, solver=name))
            solver.solve(warm, parameters(name, 2, 16, seed))
        sync()
        while True:
            task = tasks.get()
            if task is None:
                return
            entry, runs, steps, index = task
            if entry['path'] not in cache:
                cache[entry['path']] = readGraph(Path(inputDirectory) / entry['path'])
            problem, heads, tails, weights = cache[entry['path']]
            results = {}
            order = list(SOLVERS)
            order = order[index % 3:] + order[:index % 3]
            for name in order:
                trials = []
                try:
                    supplied = parameters(name, runs, steps, seed)
                    for repeat in range(repeats):
                        events.put(dict(type='start', device=device, instance=entry['name'],
                            runs=runs, steps=steps, solver=name, repeat=repeat + 1, repeats=repeats))
                        sync()
                        started = time.perf_counter()
                        result = solvers[name].solve(problem, supplied)
                        sync()
                        elapsed = time.perf_counter() - started
                        bits = np.asarray(result.sample, dtype=np.uint8)
                        if bits.shape != (problem.variableCount,) or not np.isin(bits, (0, 1)).all():
                            raise ValueError('Solver returned a non-binary or incorrectly sized sample')
                        cut = float(weights[bits[heads] != bits[tails]].sum())
                        np.testing.assert_allclose(result.energy, -cut, rtol=0, atol=1e-5)
                        if cut > entry['reference_cut'] + 1e-5:
                            raise ValueError('Cut exceeds the published optimum; check graph/reference mapping')
                        trial = dict(seconds=elapsed, cut=cut, energy=result.energy,
                            sample=''.join(map(str, bits.tolist())),
                            sample_sha256=hashlib.sha256(bits.tobytes()).hexdigest())
                        trials.append(trial)
                        events.put(dict(type='trial', device=device, instance=entry['name'],
                            runs=runs, steps=steps, solver=name, repeat=repeat + 1,
                            seconds=elapsed, cut=cut, reference_cut=entry['reference_cut']))
                    results[name] = dict(status='ok', trials=trials,
                        parameters=solvers[name]._getParameters(supplied))
                except Exception:
                    error = traceback.format_exc()
                    results[name] = dict(status='error', error=error, trials=trials)
                    events.put(dict(type='solver_error', device=device, instance=entry['name'],
                        runs=runs, steps=steps, solver=name, error=error))
            events.put(dict(type='result', device=device, instance=entry, runs=runs,
                            steps=steps, solvers=results, seed=seed))
    except BaseException:
        events.put(dict(type='fatal', device=device, error=traceback.format_exc()))
        raise


def resultRow(record):
    entry = record['instance']
    row = dict(instance=entry['name'], family=entry['family'], vertices=entry['vertices'],
        edges=entry['edges'], reference_cut=entry['reference_cut'], runs=record['runs'],
        steps=record['steps'], seed=record['seed'], device=record['device'])
    for name in SOLVERS:
        result = record['solvers'][name]
        prefix = name.lower()
        row[prefix + '_status'] = result['status']
        for suffix in ('best_cut', 'median_seconds', 'min_seconds', 'max_seconds',
                       'gap_to_reference', 'gap_percent', 'sample_sha256', 'trial_seconds', 'trial_cuts', 'error'):
            row[prefix + '_' + suffix] = ''
        if result['status'] != 'ok':
            row[prefix + '_error'] = result['error']
            continue
        best = min(result['trials'], key=lambda t: (-t['cut'], t['sample']))
        times = [t['seconds'] for t in result['trials']]
        row.update({prefix + '_best_cut': best['cut'], prefix + '_median_seconds': median(times),
            prefix + '_min_seconds': min(times), prefix + '_max_seconds': max(times),
            prefix + '_gap_to_reference': entry['reference_cut'] - best['cut'],
            prefix + '_gap_percent': 100 * (entry['reference_cut'] - best['cut']) / max(1, abs(entry['reference_cut'])),
            prefix + '_sample_sha256': best['sample_sha256'],
            prefix + '_trial_seconds': json.dumps(times),
            prefix + '_trial_cuts': json.dumps([t['cut'] for t in result['trials']])})
    row.update(best_cut='', quality_winners='', fastest_solver='', winner='', status='error')
    if all(record['solvers'][name]['status'] == 'ok' for name in SOLVERS):
        best = max(row[name.lower() + '_best_cut'] for name in SOLVERS)
        winners = [name for name in SOLVERS if abs(row[name.lower() + '_best_cut'] - best) <= 1e-6]
        fastest = lambda choices: min(choices, key=lambda name: (row[name.lower() + '_median_seconds'], name))
        row.update(best_cut=best, quality_winners='|'.join(winners), fastest_solver=fastest(SOLVERS),
                   winner=fastest(winners), status='ok')
    return row


def publish(output, records, settings, expected):
    # Construct the stable CSV header even before any instance has completed.
    dummy = dict(instance=dict(name='', family='', vertices=0, edges=0, reference_cut=0),
                 runs=0, steps=0, seed=0, device='', solvers={n: dict(status='error', error='') for n in SOLVERS})
    fields = list(resultRow(dummy))
    summaries = []
    all_rows = [resultRow(record) for record in records.values()]
    for runs, steps in settings:
        rows = sorted((row for row in all_rows if row['runs'] == runs and row['steps'] == steps), key=lambda r: r['instance'])
        atomicCsv(output / f'runs_{runs}_steps_{steps}.csv', rows, fields)
        valid = [row for row in rows if row['status'] == 'ok']
        setting_rows = []
        for name in SOLVERS:
            prefix = name.lower()
            eligible = [row for row in rows if row[prefix + '_status'] == 'ok']
            points = sum(1 / len(row['quality_winners'].split('|')) for row in valid if name in row['quality_winners'].split('|'))
            setting_rows.append(dict(runs=runs, steps=steps, solver=name, completed_instances=len(rows),
                expected_instances=expected, setting_complete=len(rows) == expected,
                successful_comparisons=len(valid), quality_win_points=points,
                outright_quality_wins=sum(row['quality_winners'] == name for row in valid),
                quality_ties=sum(name in row['quality_winners'].split('|') and '|' in row['quality_winners'] for row in valid),
                quality_then_time_wins=sum(row['winner'] == name for row in valid),
                reference_hits=sum(abs(row[prefix + '_gap_to_reference']) <= 1e-6 for row in eligible),
                mean_gap_percent=float(np.mean([row[prefix + '_gap_percent'] for row in eligible])) if eligible else '',
                total_median_seconds=sum(row[prefix + '_median_seconds'] for row in eligible),
                solver_errors=sum(row[prefix + '_status'] != 'ok' for row in rows)))
        winner = min(setting_rows, key=lambda r: (-r['quality_win_points'], r['total_median_seconds'], r['solver']))['solver'] if valid else ''
        summaries.extend(dict(row, setting_winner=winner) for row in setting_rows)
    atomicCsv(output / 'summary.csv', summaries, list(summaries[0]))
    overall = []
    for name in SOLVERS:
        subset = [row for row in summaries if row['solver'] == name]
        overall.append(dict(solver=name, quality_win_points=sum(row['quality_win_points'] for row in subset),
            quality_then_time_wins=sum(row['quality_then_time_wins'] for row in subset),
            reference_hits=sum(row['reference_hits'] for row in subset),
            total_median_seconds=sum(row['total_median_seconds'] for row in subset),
            solver_errors=sum(row['solver_errors'] for row in subset)))
    overall.sort(key=lambda r: (-r['quality_win_points'], r['total_median_seconds'], r['solver']))
    for rank, row in enumerate(overall, 1):
        row['rank'] = rank
    atomicCsv(output / 'overall.csv', overall, list(overall[0]))
    return overall


def run(args):
    import torch
    from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory
    from margin_calculator.optimization.optimization_solver.bqm_solver.torch_execution import TorchExecution
    from margin_calculator.optimization.optimization_solver.bqm_solver.torch_candidates import TorchCandidateAccumulator
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A second coordinator must not race checkpoint/CSV publication.
    import fcntl
    lock = (output / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data_path = args.inputs.resolve() / 'manifest.json'
    data = json.loads(data_path.read_text())
    entries = data['instances'][:args.limit] if args.limit else data['instances']
    if not entries:
        raise ValueError('No instances to benchmark')
    if len(set(args.devices)) != len(args.devices):
        raise ValueError('Each worker must have a distinct device')
    for entry in entries:
        if hashlib.sha256((args.inputs / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Input checksum changed: {entry['name']}")
    classes = [BQMSolverFactory.create(kind).__class__ for kind in SOLVERS.values()]
    files = {Path(inspect.getfile(c)).resolve() for c in classes + [TorchExecution, TorchCandidateAccumulator, QUBOProblem]}
    files.add(Path(__file__).resolve())
    signature = dict(runs=args.runs, steps=args.steps, repeats=args.repeats, seed=args.seed,
        devices=args.devices, input_manifest_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        instances=[entry['name'] for entry in entries],
        code_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)})
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['signature'] != signature:
            raise ValueError('Resume configuration, inputs or source changed; use a new output directory')
    else:
        atomicJson(manifest_path, dict(created=stamp(), signature=signature, sources=data,
            torch=torch.__version__, cuda=torch.version.cuda,
            hardware={d: torch.cuda.get_device_name(d) if d.startswith('cuda') else 'CPU' for d in args.devices},
            comparison='Native solver defaults; fixed-seed repeated timings; QUBO=-cut; quality then median time',
            solver_parameters={name: BQMSolverFactory.create(kind)._getParameters(parameters(name, args.runs[0], args.steps[0], args.seed)) for name, kind in SOLVERS.items()}))
    settings = [(runs, steps) for runs in args.runs for steps in args.steps]
    checkpoints = output / '.checkpoints'
    checkpoints.mkdir(exist_ok=True)
    records = {path.stem: json.loads(path.read_text()) for path in checkpoints.glob('*.json')}
    pending = []
    for runs, steps in settings:
        for index, entry in enumerate(entries):
            key = f"runs_{runs}_steps_{steps}_{entry['name']}"
            if key not in records:
                pending.append((entry, runs, steps, index + settings.index((runs, steps))))
    total = len(entries) * len(settings)
    overall = publish(output, records, settings, len(entries))
    start = time.monotonic()
    current = {}
    def status(state):
        atomicJson(output / 'status.json', dict(state=state, updated=stamp(), completed=len(records), total=total,
            percent=round(100 * len(records) / total, 2), solver_configurations=total * 3,
            timed_solves=total * 3 * args.repeats, elapsed_this_session_seconds=round(time.monotonic() - start, 1),
            active=current, provisional_leader=overall[0]['solver'] if records else None,
            errors=sum(any(v['status'] != 'ok' for v in r['solvers'].values()) for r in records.values())))
    print(f'[{stamp()}] START {len(entries)} instances x {len(settings)} settings x 3 solvers x {args.repeats} timed repeats; {len(args.devices)} workers; resumed={len(records)}', flush=True)
    status('running')
    context = mp.get_context('spawn')
    tasks, events = context.Queue(), context.Queue()
    workers = [context.Process(target=worker, args=(device, tasks, events, str(args.inputs.resolve()), args.repeats, args.seed)) for device in args.devices] if pending else []
    for process in workers:
        process.start()
    for task in pending:
        tasks.put(task)
    for _ in workers:
        tasks.put(None)
    last_heartbeat = time.monotonic()
    try:
        while len(records) < total:
            try:
                event = events.get(timeout=2)
            except queue.Empty:
                event = None
                if any(p.exitcode not in (None, 0) for p in workers) or all(not p.is_alive() for p in workers):
                    raise RuntimeError('A worker exited before completing the benchmark; completed checkpoints are retained')
            if event:
                kind, device = event['type'], event['device']
                if kind == 'fatal':
                    raise RuntimeError(event['error'])
                if kind == 'result':
                    key = f"runs_{event['runs']}_steps_{event['steps']}_{event['instance']['name']}"
                    atomicJson(checkpoints / (key + '.json'), event)
                    records[key] = event
                    overall = publish(output, records, settings, len(entries))
                    row = resultRow(event)
                    current.pop(device, None)
                    print(f"[{stamp()}] COMPLETE {len(records)}/{total} {key} winner={row['winner']} quality={row['quality_winners']} best_cut={row['best_cut']}", flush=True)
                else:
                    if kind in ('start', 'warmup'):
                        current[device] = dict(event, started=stamp())
                    print(f'[{stamp()}] {kind.upper()} {json.dumps(event)}', flush=True)
                status('running')
            if time.monotonic() - last_heartbeat >= 10:
                print(f'[{stamp()}] HEARTBEAT completed={len(records)}/{total} active={json.dumps(current)}', flush=True)
                status('running')
                last_heartbeat = time.monotonic()
        state = 'completed_with_errors' if any(row['solver_errors'] for row in overall) else 'completed'
        current.clear()
        status(state)
        print(f'[{stamp()}] FINISHED state={state} overall={json.dumps(overall)}', flush=True)
    except BaseException:
        status('failed')
        raise
    finally:
        for process in workers:
            if len(records) < total and process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--archive', type=Path, required=True)
    prep.add_argument('--references', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    bench = sub.add_parser('run')
    bench.add_argument('--inputs', type=Path, required=True)
    bench.add_argument('--output', type=Path, required=True)
    bench.add_argument('--devices', nargs='+', default=[f'cuda:{i}' for i in range(8)])
    bench.add_argument('--runs', nargs='+', type=int, default=[2, 4, 8, 16])
    bench.add_argument('--steps', nargs='+', type=int, default=[1000, 2000, 5000, 10000])
    bench.add_argument('--repeats', type=int, default=3)
    bench.add_argument('--seed', type=int, default=1)
    bench.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.archive, args.references, args.output)
    else:
        if min(*args.runs, *args.steps, args.repeats) < 1 or args.seed < 0 or (args.limit is not None and args.limit < 1):
            parser.error('Budgets must be positive and seed nonnegative')
        if len(set(args.runs)) != len(args.runs) or len(set(args.steps)) != len(args.steps):
            parser.error('Sweep values must be unique')
        run(args)


if __name__ == '__main__':
    main()

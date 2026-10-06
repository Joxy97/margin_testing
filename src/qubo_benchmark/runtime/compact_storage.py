"""Small checkpoint matrices with atomic seed commits and strict resume identity.

There is one float64 (seeds, checkpoints) array per solver/problem. A matrix
replacement is flushed before its row checksum/status is committed. An orphan
row after a crash is discarded and that entire seed remains pending. Committed
rows cannot be overwritten. Completion and aggregate checksums are durable
before optional raw removal, so completed runs resume without their arrays.
Per-seed checksums are removed with raw arrays; terminal statuses and independently
validated optimum capture times are retained in the same progress record.

A single runner owns the store; its normal directory lock protects mutations.
"""
from copy import deepcopy
import csv
import hashlib
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re

import numpy as np

from .common import atomic, digest, filehash, read
from .compact_metrics import AGGREGATE_COLUMNS, CHECKPOINTS, aggregate_energies


SCHEMA = 2
TERMINAL_STATUSES = frozenset(('complete', 'completed', 'failed', 'unavailable',
                              'no_in_budget_candidate', 'error', 'watchdog_timeout',
                              'invalid_output', 'validation_error', 'oom'))
SUCCESS_STATUSES = frozenset(('complete', 'completed'))


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or value in ('.', '..'):
        raise ValueError(f'Unsafe solver/problem identifier: {value!r}')
    return value


def _row_hash(row):
    # Normalize NaN payloads and byte order to a stable, platform-independent hash.
    values = np.asarray(row, dtype='<f8').copy()
    values[np.isnan(values)] = np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()


def _flush_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_array(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as handle:
            np.save(handle, np.asarray(matrix, dtype=np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _flush_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS, extrasaction='raise')
            writer.writeheader()
            for row in rows:
                formatted = {key: ('NaN' if math.isnan(value) else 'Inf' if math.isinf(value) else value)
                             for key, value in row.items()}
                writer.writerow(formatted)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _flush_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_checked(path):
    try:
        wrapper = read(path)
        payload = wrapper['payload']
        if wrapper['sha256'] != digest(payload):
            raise ValueError('checksum mismatch')
        return payload
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Corrupt compact state: {path}') from error


class CompactStore:
    """Persist terminal attempts, with no vectors, trajectories or telemetry.

    The complete supplied manifest is immutable and compared on every reopen,
    including source/configuration/input/reference/environment identities.
    Seeds and checkpoint arrays are saved once, in that manifest.
    """
    def __init__(self, root, manifest, seeds=range(100), checkpoints=CHECKPOINTS,
                 keep_raw_results=True):
        self.root = Path(root)
        supplied_seeds = tuple(seeds)
        if any(not isinstance(seed, Integral) or isinstance(seed, bool) for seed in supplied_seeds):
            raise ValueError('Seeds must be integers')
        self.seeds = tuple(int(seed) for seed in supplied_seeds)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Seeds must be nonempty and unique')
        self.checkpoints = tuple(float(value) for value in checkpoints)
        if len(self.checkpoints) != len(CHECKPOINTS) or self.checkpoints != CHECKPOINTS:
            raise ValueError(f'Compact checkpoint protocol requires exactly {CHECKPOINTS}')
        self.keep_raw_results = bool(keep_raw_results)
        self.manifest = deepcopy(manifest)
        self.manifest.setdefault('tolerance', 1e-9)
        if not np.isfinite(self.manifest['tolerance']) or self.manifest['tolerance'] < 0:
            raise ValueError('Floating objective tolerance must be finite and nonnegative')
        self.manifest.update(schema_version=SCHEMA, seeds=list(self.seeds),
                             checkpoints_seconds=list(self.checkpoints),
                             keep_raw_results=self.keep_raw_results)
        self.solvers = tuple(_identifier(solver) for solver in self.manifest['solvers'])
        if not self.solvers or len(set(self.solvers)) != len(self.solvers):
            raise ValueError('Solvers must be nonempty and unique')
        self.problems = {}
        for problem in self.manifest['problems']:
            key = _identifier(problem['id'])
            if key in self.problems:
                raise ValueError('Duplicate problem identifier')
            if problem['reference_type'] not in ('OPTIMUM', 'BKS'):
                raise ValueError('Reference type must be OPTIMUM or BKS')
            if not np.isfinite(problem['reference']):
                raise ValueError('Reference must be finite')
            self.problems[key] = problem
        if not self.problems:
            raise ValueError('Problems must be nonempty')
        self.identity = digest(self.manifest)
        self._seed_indices = {seed: index for index, seed in enumerate(self.seeds)}
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / 'manifest.json'
        state_path = self.root / 'state.json'
        if manifest_path.exists():
            if read(manifest_path) != self.manifest:
                raise ValueError('Compact resume identity mismatch (source/configuration/input/reference/environment/protocol)')
        else:
            if any(path.name != '.lock' for path in self.root.iterdir()):
                raise ValueError('Result directory has partial/stale files without a compact manifest')
            atomic(manifest_path, self.manifest)
        if state_path.exists():
            self.state = _read_checked(state_path)
            self.state['groups'] = {solver: {
                problem: _read_checked(self._progress_path(solver, problem))
                for problem in self.problems} for solver in self.solvers}
            self._validate_state()
        else:
            self.state = dict(schema_version=SCHEMA, manifest_sha256=self.identity,
                              complete=False, groups={solver: {
                                  problem: dict(status=[None] * len(self.seeds),
                                                row_sha256=[None] * len(self.seeds),
                                                time_to_optimum_s=[None] * len(self.seeds),
                                                aggregate_sha256=None, raw_discarded=False,
                                                manifest_sha256=self.identity,
                                                solver_id=solver, problem_id=problem)
                                  for problem in self.problems} for solver in self.solvers})
            # A crash during first creation can leave durable per-group progress.
            for solver in self.solvers:
                for problem in self.problems:
                    if self._progress_path(solver, problem).exists():
                        self.state['groups'][solver][problem] = _read_checked(self._progress_path(solver, problem))
                    else:
                        self._write_group(solver, problem)
            self._validate_state()
            self._write_state()
        self._recover()

    def _write_state(self):
        # Per-group progress is authoritative. These tiny display counters may
        # lag a crash and are rebuilt on resume, without rewriting the full plan.
        payload = dict(schema_version=SCHEMA, manifest_sha256=self.identity,
                       complete=self.state['complete'], planned=self.planned_count,
                       terminal=self.completed_count, failures=sum(
                           status is not None and status not in SUCCESS_STATUSES
                           for solver in self.solvers for problem in self.problems
                           for status in self._group(solver, problem)['status']))
        atomic(self.root / 'state.json', dict(sha256=digest(payload), payload=payload))

    def _progress_path(self, solver, problem):
        return self.root / 'progress' / solver / (problem + '.json')

    def _write_group(self, solver, problem):
        payload = self._group(solver, problem)
        atomic(self._progress_path(solver, problem), dict(sha256=digest(payload), payload=payload))

    def _validate_state(self):
        if self.state.get('schema_version') != SCHEMA or self.state.get('manifest_sha256') != self.identity:
            raise ValueError('Compact state identity mismatch')
        if set(self.state.get('groups', {})) != set(self.solvers):
            raise ValueError('Compact state solver identity mismatch')
        for solver in self.solvers:
            if set(self.state['groups'][solver]) != set(self.problems):
                raise ValueError('Compact state problem identity mismatch')
            for problem in self.problems:
                group = self.state['groups'][solver][problem]
                if (group.get('manifest_sha256') != self.identity or group.get('solver_id') != solver
                        or group.get('problem_id') != problem):
                    raise ValueError('Stale compact group progress identity')
                if len(group['status']) != len(self.seeds):
                    raise ValueError('Partial compact progress state')
                timings = group.get('time_to_optimum_s')
                if not isinstance(timings, list) or len(timings) != len(self.seeds):
                    raise ValueError('Partial/malformed compact optimum timing state')
                for status, timing in zip(group['status'], timings):
                    self._validate_timing(problem, status, timing)
                if group['row_sha256'] is None:
                    if (self.keep_raw_results or not group['raw_discarded']
                            or not (self.root / 'completion.json').exists()
                            or any(status not in TERMINAL_STATUSES for status in group['status'])
                            or not isinstance(group['aggregate_sha256'], str)):
                        raise ValueError('Compacted row checksums require completed raw-discarded results')
                    # Full completion and aggregate checksum validation follows
                    # before recovery may delete any remaining raw files.
                    continue
                if len(group['row_sha256']) != len(self.seeds):
                    raise ValueError('Partial compact progress checksums')
                for status, checksum in zip(group['status'], group['row_sha256']):
                    if status is None:
                        if checksum is not None:
                            raise ValueError('Uncommitted row has committed checksum')
                    elif status not in TERMINAL_STATUSES or not isinstance(checksum, str):
                        raise ValueError('Invalid compact terminal status/checksum')

    def raw_path(self, solver, problem):
        self._group(solver, problem)
        return self.root / 'raw' / solver / (problem + '.npy')

    def aggregate_path(self, solver, problem):
        self._group(solver, problem)
        return self.root / 'aggregated' / solver / (problem + '.csv')

    def _group(self, solver, problem):
        try:
            return self.state['groups'][solver][problem]
        except KeyError as error:
            raise ValueError('Unknown compact solver/problem group') from error

    def _load_array(self, solver, problem):
        path = self.raw_path(solver, problem)
        try:
            matrix = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise ValueError(f'Partial/corrupt compact energy matrix: {path}') from error
        if matrix.dtype != np.dtype(np.float64) or matrix.shape != (len(self.seeds), len(self.checkpoints)):
            raise ValueError(f'Stale compact matrix shape/dtype: {path}')
        if np.any(np.isinf(matrix)):
            raise ValueError(f'Invalid infinite energy in compact matrix: {path}')
        return matrix

    def _completion(self):
        path = self.root / 'completion.json'
        if not path.exists():
            return None
        completion = _read_checked(path)
        if completion.get('manifest_sha256') != self.identity:
            raise ValueError('Compact completion identity mismatch')
        expected = {solver: {problem: self._group(solver, problem)['aggregate_sha256']
                             for problem in self.problems} for solver in self.solvers}
        if completion.get('aggregates') != expected or self.pending():
            raise ValueError('Stale/partial compact completion record')
        for solver in self.solvers:
            for problem in self.problems:
                path = self.aggregate_path(solver, problem)
                if not path.exists() or filehash(path) != expected[solver][problem]:
                    raise ValueError(f'Missing/corrupt completed compact aggregate: {path}')
        return completion

    def _recover(self):
        completion = self._completion()
        for solver in self.solvers:
            for problem in self.problems:
                group = self._group(solver, problem)
                path = self.raw_path(solver, problem)
                path.with_name(path.name + '.tmp').unlink(missing_ok=True)
                if completion is not None and not self.keep_raw_results:
                    # Aggregates are the completed scientific record. A raw
                    # unlink interrupted by a crash may be safely repeated.
                    self._discard_raw(solver, problem)
                    continue
                if not path.exists():
                    if any(status is not None for status in group['status']):
                        raise ValueError(f'Missing committed compact matrix: {path}')
                    _write_array(path, np.full((len(self.seeds), len(self.checkpoints)), np.nan))
                matrix = self._load_array(solver, problem)
                orphan = False
                for index, status in enumerate(group['status']):
                    if status is None:
                        if not np.isnan(matrix[index]).all():
                            matrix[index] = np.nan
                            orphan = True
                    elif _row_hash(matrix[index]) != group['row_sha256'][index]:
                        raise ValueError(f'Stale/corrupt committed seed row: {solver}/{problem}/{self.seeds[index]}')
                    else:
                        self._validate_timing(problem, status, group['time_to_optimum_s'][index], matrix[index])
                if orphan:
                    _write_array(path, matrix)
                if group['raw_discarded'] and (completion is None or self.keep_raw_results):
                    raise ValueError('Stale raw-discard state with a retained matrix')
                aggregate = self.aggregate_path(solver, problem)
                if group['aggregate_sha256'] is not None and (
                        not aggregate.exists() or filehash(aggregate) != group['aggregate_sha256']):
                    raise ValueError(f'Partial/stale compact aggregate: {aggregate}')
        if completion is not None:
            self.state['complete'] = True
        elif self.state['complete']:
            raise ValueError('Compact state claims completion without durable completion record')
        # Also repair stale display counters after a group commit succeeded but
        # the following global progress replacement was interrupted.
        self._write_state()

    def _discard_raw(self, solver, problem):
        """Call only after validating/writing durable full completion proof."""
        path = self.raw_path(solver, problem)
        path.unlink(missing_ok=True)
        if path.parent.exists():
            _flush_directory(path.parent)
        group = self._group(solver, problem)
        group['raw_discarded'] = True
        group['row_sha256'] = None
        self._write_group(solver, problem)

    def pending(self, solver=None, problem=None):
        """Return full seeds needing a fresh solve, in deterministic plan order."""
        if solver is not None and solver not in self.solvers:
            raise ValueError('Unknown solver')
        if problem is not None and problem not in self.problems:
            raise ValueError('Unknown problem')
        return [(name, key, seed) for name in self.solvers if solver is None or solver == name
                for key in self.problems if problem is None or problem == key
                for seed, status in zip(self.seeds, self._group(name, key)['status']) if status is None]

    @property
    def planned_count(self):
        return len(self.solvers) * len(self.problems) * len(self.seeds)

    @property
    def completed_count(self):
        return self.planned_count - len(self.pending())

    @property
    def has_failures(self):
        return any(status is not None and status not in SUCCESS_STATUSES
                   for solver in self.solvers for problem in self.problems
                   for status in self._group(solver, problem)['status'])

    def _tolerance(self, problem):
        metadata = self.problems[problem]
        return 0.0 if metadata.get('integer_objective', True) else float(self.manifest.get('tolerance', 1e-9))

    def _validate_timing(self, problem, status, timing, row=None):
        if timing is None:
            return None
        if (not isinstance(timing, Real) or isinstance(timing, bool)
                or not math.isfinite(timing) or not 0 <= timing <= self.checkpoints[-1]):
            raise ValueError('Optimum timing must be a finite number in [0, 20] or None')
        metadata = self.problems[problem]
        if status not in SUCCESS_STATUSES or metadata['reference_type'] != 'OPTIMUM':
            raise ValueError('Optimum timing requires a successful OPTIMUM seed')
        if row is not None:
            hits = np.isfinite(row) & (np.abs(row-metadata['reference']) <= self._tolerance(problem))
            times = np.asarray(self.checkpoints)
            if not hits.any() or np.any(hits & (times < timing)) or not hits[times > timing].all():
                raise ValueError('Optimum timing is inconsistent with seed checkpoints')
        return float(timing)

    def commit_seed(self, solver, problem, seed, energies, status='complete', time_to_optimum_s=None):
        """Durably commit a full seed once. An interrupted seed stays pending.

        Terminal failures have unavailable energy (NaN) at every checkpoint.
        Their statuses remain visible and their attempts remain in denominators.
        The optional first optimum capture time commits with status and checksum.
        """
        group = self._group(solver, problem)
        if seed not in self._seed_indices:
            raise ValueError('Seed is outside the immutable seed schedule')
        index = self._seed_indices[seed]
        if group['status'][index] is not None:
            raise ValueError('Seed already committed; refusing overwrite')
        if status == 'interrupted':
            return False
        if status not in TERMINAL_STATUSES:
            raise ValueError(f'Unknown compact terminal status: {status}')
        row = np.asarray(energies, dtype=np.float64)
        if row.shape != (len(self.checkpoints),) or np.any(np.isinf(row)):
            raise ValueError('Seed energies must be nine float64 values, with NaN for unavailable values')
        metadata = self.problems[problem]
        aggregate_energies(row.reshape(1, -1), self.checkpoints, metadata['reference'],
                           metadata['reference_type'], self._tolerance(problem))
        if status not in SUCCESS_STATUSES:
            row = np.full(len(self.checkpoints), np.nan)
            time_to_optimum_s = None
        timing = self._validate_timing(problem, status, time_to_optimum_s, row)
        matrix = self._load_array(solver, problem)
        matrix[index] = row
        _write_array(self.raw_path(solver, problem), matrix)
        next_state = deepcopy(self.state)
        next_group = next_state['groups'][solver][problem]
        next_group['status'][index] = status
        next_group['row_sha256'][index] = _row_hash(row)
        next_group['time_to_optimum_s'][index] = timing
        previous_state = self.state
        self.state = next_state
        try:
            self._write_group(solver, problem)
        except BaseException:
            self.state = previous_state
            raise
        self._write_state()
        return True

    def aggregate(self, solver=None, problem=None):
        """Write only fully attempted groups; return their exact four-column rows."""
        if solver is None or problem is None:
            if solver is not None and solver not in self.solvers:
                raise ValueError('Unknown solver')
            if problem is not None and problem not in self.problems:
                raise ValueError('Unknown problem')
            return {(name, key): self.aggregate(name, key)
                    for name in self.solvers if solver is None or solver == name
                    for key in self.problems if problem is None or problem == key
                    if not self.pending(name, key)}
        group = self._group(solver, problem)
        if self.pending(solver, problem):
            return None
        path = self.aggregate_path(solver, problem)
        if group['raw_discarded']:
            if not path.exists() or filehash(path) != group['aggregate_sha256']:
                raise ValueError('Cannot recover corrupt aggregate after durable raw removal')
            with path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != AGGREGATE_COLUMNS:
                    raise ValueError('Aggregate CSV schema mismatch')
                return [{key: float(value) for key, value in row.items()} for row in reader]
        metadata = self.problems[problem]
        rows = aggregate_energies(self._load_array(solver, problem), self.checkpoints,
                                  metadata['reference'], metadata['reference_type'], self._tolerance(problem))
        _write_csv(path, rows)
        group['aggregate_sha256'] = filehash(path)
        self._write_group(solver, problem)
        self._write_state()
        return rows

    def finalize(self):
        """Commit aggregates and completion before optional removal of raw arrays."""
        if self.pending():
            self.aggregate()
            return False
        self.aggregate()
        completion = dict(schema_version=SCHEMA, manifest_sha256=self.identity,
                          aggregates={solver: {problem: self._group(solver, problem)['aggregate_sha256']
                                               for problem in self.problems} for solver in self.solvers})
        atomic(self.root / 'completion.json', dict(sha256=digest(completion), payload=completion))
        self.state['complete'] = True
        self._write_state()
        if not self.keep_raw_results:
            for solver in self.solvers:
                for problem in self.problems:
                    self._discard_raw(solver, problem)
            self._write_state()
        return True

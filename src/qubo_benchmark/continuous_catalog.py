"""Public inputs for the continuous suite, isolated from the historical catalog.

Density is measured on nonzero unordered QUBO interactions, excluding diagonals.
The suite calls a problem sparse below 20 percent and dense otherwise.  Missing
public inputs remain explicit slots; selection never substitutes another size.
"""
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
import json
from pathlib import Path
import re
import ssl
import struct
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

import certifi
import numpy as np

from .catalog import ROOT, DEFAULT_DATA, loadCatalog
from .formats import parseMatrix, parseWitness
from .model import Problem, binaryVector
from .pipeline import loadPrepared, readSource
from .storage import SourceCache, readJson, sha256, stamp, writeJson

CONTINUOUS_CATALOG = ROOT / 'configs' / 'continuous_catalog.json'
CONTINUOUS_DATA = ROOT / 'benchmark_data' / 'continuous12'
CONTINUOUS_SIZES = (100, 200, 500, 1000, 5000, 10000)
SPARSE_CUTOFF_PERCENT = 20.0
REQUIRED_GROUPS = {(n, d) for n in CONTINUOUS_SIZES for d in ('sparse', 'dense')}


class ContinuousInputUnavailable(ValueError):
    """A required public input or its independently checked reference is missing."""


def retrieve_public_zip_member(archive_url, member, directory=CONTINUOUS_DATA, offline=False):
    """Retrieve one publisher ZIP member with exact stable HTTP byte ranges.

    This caches public source data only. It never assigns an objective/reference
    or makes an unresolved catalog slot runnable. The bounded extraction avoids
    downloading a multi-gigabyte archive containing redundant permutations.
    """
    source_url = archive_url + '#' + member
    cache = SourceCache(directory)
    raw, metadata = cache.paths(source_url)
    if raw.exists() or metadata.exists():
        return cache.read(source_url)
    if offline:
        raise FileNotFoundError(f'Offline public ZIP member missing: {source_url}')
    context = ssl.create_default_context(cafile=certifi.where())

    def request_range(value, expected_etag=None):
        request = Request(archive_url, headers={'Range': value, 'User-Agent': 'QUBOBenchmark/1.0'})
        with urlopen(request, timeout=30, context=context) as response:
            content_range = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
            if response.status != 206 or not match:
                raise ValueError('Publisher did not honor the requested ZIP byte range')
            begin, end, total = map(int, match.groups())
            etag = response.headers.get('ETag')
            if not etag or (expected_etag is not None and etag != expected_etag):
                raise ValueError('Publisher archive identity changed during retrieval')
            if end - begin + 1 > 64 * 1024**2:
                raise ValueError('Selected archive range exceeds the 64 MiB input limit')
            payload = response.read(end - begin + 2)
            if len(payload) != end - begin + 1:
                raise ValueError('Truncated public archive byte range')
            if value.startswith('bytes=') and not value.startswith('bytes=-'):
                requested_begin, requested_end = map(int, value[6:].split('-'))
                if (begin, end) != (requested_begin, requested_end):
                    raise ValueError('Publisher returned a different ZIP byte range')
            return payload, etag, total

    tail, etag, archive_bytes = request_range('bytes=-65536')
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0:
        raise ValueError('ZIP central directory footer is missing')
    footer = struct.unpack_from('<4s4H2IH', tail, eocd)
    if footer[1] != 0 or footer[2] != 0 or footer[3] != footer[4]:
        raise ValueError('Unsupported split public ZIP archive')
    size, start = footer[5:7]
    directory_bytes, _, total = request_range(f'bytes={start}-{start+size-1}', etag)
    if total != archive_bytes:
        raise ValueError('Publisher ZIP length changed')
    position = 0
    selected = None
    while position < len(directory_bytes):
        header = struct.unpack_from('<4s6H3I5H2I', directory_bytes, position)
        if header[0] != b'PK\x01\x02':
            raise ValueError('Malformed public ZIP central directory')
        name = directory_bytes[position+46:position+46+header[10]].decode('utf-8')
        if name == member:
            if selected is not None:
                raise ValueError('Duplicate requested public ZIP member')
            selected = header
        position += 46 + header[10] + header[11] + header[12]
    if selected is None:
        raise ValueError('Requested publisher ZIP member does not exist')
    if selected[3] & 1 or selected[4] != 8 or selected[9] > 64 * 1024**2:
        raise ValueError('Unsupported encryption/compression or oversized public ZIP member')
    offset = selected[-1]
    local, _, _ = request_range(f'bytes={offset}-{offset+29}', etag)
    header = struct.unpack('<4s5H3I2H', local)
    if header[0] != b'PK\x03\x04':
        raise ValueError('Malformed public ZIP local header')
    begin = offset + 30 + header[-2] + header[-1]
    compressed, _, _ = request_range(f'bytes={begin}-{begin+selected[8]-1}', etag)
    inflater = zlib.decompressobj(-15)
    payload = inflater.decompress(compressed, 64 * 1024**2 + 1)
    if not inflater.eof or len(payload) != selected[9] or zlib.crc32(payload) != selected[7]:
        raise ValueError('Public ZIP member decompression/length/CRC integrity failure')
    record = dict(source_url=source_url, resolved_url=archive_url, retrieved_at=stamp(),
                  bytes=len(payload), sha256=sha256(payload), content_type='text/plain',
                  archive_member=member, archive_etag=etag, archive_bytes=archive_bytes,
                  compressed_bytes=selected[8], member_crc32=selected[7],
                  retrieval_note='Exact stable HTTP ranges; publisher ZIP directory, decompressed size and CRC32 checked; original vertex weights preserved.',
                  hash_meaning='local source fingerprint, not an independent publisher authenticity certificate')
    temporary = raw.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(raw)
    writeJson(metadata, record)
    return payload, record


def _local(path, root=ROOT):
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Continuous input paths must remain within the checkout')
    return resolved


def load_continuous_catalog(path=None):
    catalog = readJson(path or CONTINUOUS_CATALOG)
    entries = catalog['instances']
    groups = [(e['binary_variables'], e['density_category']) for e in entries]
    ids = [e['instance_id'] for e in entries]
    if len(entries) != 12 or set(groups) != REQUIRED_GROUPS or len(set(groups)) != 12:
        raise ValueError('Continuous catalog must preserve all twelve size/density slots')
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', i) for i in ids):
        raise ValueError('Continuous catalog requires twelve unique safe instance IDs')
    if catalog.get('sparse_cutoff_percent') != SPARSE_CUTOFF_PERCENT:
        raise ValueError('Unsupported continuous density rule')
    for entry in entries:
        if entry.get('preparation_status') == 'unavailable':
            if not entry.get('unavailable_reason'):
                raise ValueError('Unavailable input needs an explicit reason')
            continue
        if not isinstance(entry.get('normalized_reference_objective_min'), (int, float)):
            raise ValueError('Public input requires a numeric verified reference')
        if entry.get('reference_status') not in ('published_proven_optimum', 'published_best_known',
                                                'independently_verified_optimum'):
            raise ValueError('Reference must distinguish published optima, BKS and independently verified optima')
        if not entry.get('reference_sources') or not entry.get('reference_verification'):
            raise ValueError('Public input requires primary reference provenance')
        _local(entry['normalized_path'])
        measured = entry.get('actual_density_percent')
        if measured is not None:
            category = 'sparse' if measured < SPARSE_CUTOFF_PERCENT else 'dense'
            if not 0 <= measured <= 100 or category != entry['density_category']:
                raise ValueError('Measured QUBO density differs from its category')
    return catalog


def select_continuous_problem(n, density, catalog=None):
    if (n, density) not in REQUIRED_GROUPS:
        raise ValueError('Expected size 100, 200, 500, 1000, 5000 or 10000 and sparse/dense')
    entry = next(e for e in (catalog or load_continuous_catalog())['instances']
                 if (e['binary_variables'], e['density_category']) == (n, density))
    if entry.get('preparation_status') == 'unavailable':
        raise ContinuousInputUnavailable(f"{n} {density}: {entry['unavailable_reason']}")
    path = _local(entry['normalized_path'])
    if entry.get('preparation_status') != 'ready' or not path.is_file():
        raise ContinuousInputUnavailable(f"{n} {density}: public input has not been prepared")
    return entry, path


@dataclass
class WeightedCutSource:
    """Source edge-list scoring, independent of the normalized QUBO formula."""
    graph_n: int
    heads: np.ndarray
    tails: np.ndarray
    weights: np.ndarray
    fixed_last_vertex: bool = False

    @property
    def n(self):
        return self.graph_n - int(self.fixed_last_vertex)

    def score(self, sample):
        x = binaryVector(sample, self.n)
        if self.fixed_last_vertex:
            x = np.r_[x, 0]
        # A cut edge contributes its source weight exactly once. The canonical
        # objective minimizes negative cut weight; no QUBO coordinates are used.
        return -int(np.sum(self.weights * (x[self.heads] != x[self.tails]), dtype=np.int64))

    def canonical(self):
        degree = np.zeros(self.graph_n, dtype=np.int64)
        np.add.at(degree, self.heads, self.weights)
        np.add.at(degree, self.tails, self.weights)
        active = (self.heads < self.n) & (self.tails < self.n) & (self.weights != 0)
        diagonal = np.flatnonzero(degree[:self.n])
        rows = np.r_[self.heads[active], diagonal].astype(np.uint32)
        cols = np.r_[self.tails[active], diagonal].astype(np.uint32)
        values = np.r_[self.weights[active], -degree[diagonal]]
        return Problem(self.n, rows, cols, values)


@dataclass
class WeightedCliqueSource:
    """Original weighted DIMACS graph, retaining every published vertex weight."""
    n: int
    adjacency: np.ndarray
    heads: np.ndarray
    tails: np.ndarray
    weights: np.ndarray

    @property
    def penalty(self):
        return int(self.weights.max()) + 1

    def score(self, sample):
        x = binaryVector(sample, self.n)
        selected = int(x.sum())
        induced_edges = int(np.count_nonzero(x[self.heads] & x[self.tails]))
        missing_pairs = selected * (selected - 1) // 2 - induced_edges
        return -int(np.dot(self.weights, x)) + 2 * self.penalty * missing_pairs

    def canonical(self):
        # A nonedge costs 2P, strictly greater than any vertex reward. Removing
        # either endpoint of a violated pair strictly improves the objective;
        # therefore all minima are cliques and min E = -max clique weight.
        count = self.n * (self.n - 1) // 2 - len(self.heads) + self.n
        rows = np.empty(count, dtype=np.uint32)
        cols = np.empty(count, dtype=np.uint32)
        dtype = np.int32 if self.penalty <= np.iinfo(np.int32).max else np.int64
        values = np.empty(count, dtype=dtype)
        position = 0
        for i in range(self.n):
            missing = np.flatnonzero(~self.adjacency[i, i+1:]) + i + 1
            end = position + 1 + len(missing)
            rows[position:end] = i
            cols[position] = i
            cols[position+1:end] = missing
            values[position] = -self.weights[i]
            values[position+1:end] = self.penalty
            position = end
        return Problem(self.n, rows, cols, values)


def parse_weighted_clique(stream):
    """Strict, bounded parser for the authors' original weighted DIMACS input."""
    def records():
        for line in stream:
            words = line.split(maxsplit=1)
            if words and words[0] not in (b'c', b'#'):
                yield line
    lines = records()
    try:
        header = next(lines).split()
        if len(header) != 4 or header[:2] != [b'p', b'edge']:
            raise ValueError('Expected weighted DIMACS p edge header')
        n, m = map(int, header[2:])
        if not 0 < n <= 10000 or not 0 <= m <= n * (n - 1) // 2:
            raise ValueError('Invalid or oversized public weighted-clique header')
        weights = np.zeros(n, dtype=np.int64)
        for _ in range(n):
            record = next(lines).split()
            if len(record) != 3 or record[0] != b'n':
                raise ValueError('Expected all original vertex weights before edges')
            label, weight = map(int, record[1:])
            if not 1 <= label <= n or weight <= 0 or weights[label-1] != 0:
                raise ValueError('Invalid or duplicate original vertex weight')
            weights[label-1] = weight
        # Stock author arithmetic is binary64; its nonnegative sums must remain
        # exact integers. Also keep every canonical QUBO integer sum int64-safe.
        if sum(map(int, weights)) > 2**53 or 2 * (int(weights.max()) + 1) * n * (n-1)//2 > np.iinfo(np.int64).max:
            raise ValueError('Public weights exceed exact verification arithmetic')
        adjacency = np.zeros((n, n), dtype=bool)
        heads, tails = np.empty(m, dtype=np.uint32), np.empty(m, dtype=np.uint32)
        position = 0
        while position < m:
            block = list(islice(lines, min(100000, m-position)))
            if not block or any(line.split(maxsplit=1)[0] != b'e' for line in block):
                raise ValueError('Invalid or truncated original edge records')
            values = np.loadtxt([line.split(maxsplit=1)[1] for line in block], dtype=np.int64, ndmin=2)
            if values.shape != (len(block), 2):
                raise ValueError('Expected exactly two original edge labels')
            i, j = values.T
            if np.any(i < 1) or np.any(j < 1) or np.any(i > n) or np.any(j > n) or np.any(i == j):
                raise ValueError('Invalid original graph edge labels')
            i, j = np.minimum(i, j)-1, np.maximum(i, j)-1
            keys = i * n + j
            if len(np.unique(keys)) != len(keys) or np.any(adjacency[i, j]):
                raise ValueError('Duplicate original undirected edge')
            end = position + len(block)
            heads[position:end], tails[position:end] = i, j
            adjacency[i, j], adjacency[j, i] = True, True
            position = end
        if next(lines, None) is not None:
            raise ValueError('Trailing original graph records')
        return WeightedCliqueSource(n, adjacency, heads, tails, weights)
    except StopIteration as exc:
        raise ValueError('Truncated public weighted-clique input') from exc


def parse_weighted_cut(stream, fixed_last_vertex=False):
    """Bounded text parsing for the large public MQLib and Stanford inputs."""
    def records():
        for line in stream:
            words = line.split()
            if words and not words[0].startswith((b'#', b'%', b'c')):
                yield line
    lines = records()
    try:
        header = next(lines).split()
        if len(header) != 2:
            raise ValueError('Expected graph dimension and edge count')
        n, m = map(int, header)
        if n <= 0 or not 0 <= m <= n * (n - 1) // 2:
            raise ValueError('Invalid weighted graph header')
        heads = np.empty(m, dtype=np.uint32)
        tails = np.empty(m, dtype=np.uint32)
        weights = np.empty(m, dtype=np.int64)
        position = 0
        while position < m:
            block = list(islice(lines, min(100000, m - position)))
            if not block:
                raise ValueError('Truncated weighted graph')
            values = np.loadtxt(block, dtype=np.int64, ndmin=2)
            if values.shape != (len(block), 3):
                raise ValueError('Expected weighted graph triples')
            i, j, w = values.T
            if np.any(i < 1) or np.any(j < 1) or np.any(i > n) or np.any(j > n) or np.any(i == j):
                raise ValueError('Invalid graph source labels')
            end = position + len(block)
            heads[position:end] = np.minimum(i, j) - 1
            tails[position:end] = np.maximum(i, j) - 1
            weights[position:end] = w
            position = end
        if next(lines, None) is not None:
            raise ValueError('Trailing graph edge records')
        # Integer packed coordinates avoid a Python tuple/set per dense edge.
        keys = heads.astype(np.uint64) * np.uint64(n) + tails
        keys.sort()
        if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
            raise ValueError('Duplicate undirected source edge')
        return WeightedCutSource(n, heads, tails, weights, fixed_last_vertex)
    except StopIteration as exc:
        raise ValueError('Truncated weighted graph header') from exc


def _source(entry, root=ROOT):
    if entry.get('reuse_qubo37'):
        original = next(e for e in loadCatalog()['instances'] if e['instance_id'] == entry['reuse_qubo37'])
        return readSource(original, _local(entry['source_data_directory'], root))
    cache = SourceCache(_local(entry['source_data_directory'], root))
    payload, provenance = cache.read(entry['input_url'])
    kind = entry['source_kind']
    if kind == 'matrix':
        source = parseMatrix(payload, entry.get('position_in_bundle_1_based'),
                             -1 if entry['source_objective_sense'] == 'max' else 1)
    elif kind == 'weighted_maxcut':
        source = parse_weighted_cut(BytesIO(payload))
    elif kind == 'weighted_clique':
        source = parse_weighted_clique(BytesIO(payload))
    elif kind == 'mqlib_original_ubqp':
        with ZipFile(BytesIO(payload)) as archive:
            info = archive.getinfo(entry['archive_member'])
            if len(archive.infolist()) != 1 or info.file_size > 256 * 1024**2:
                raise ValueError('Unexpected MQLib archive members or size')
            with archive.open(info) as stream:
                source = parse_weighted_cut(stream, fixed_last_vertex=True)
    else:
        raise ValueError(f'Unknown continuous public source format: {kind}')
    if source.n != entry['binary_variables']:
        raise ValueError('Public source dimension differs from continuous catalog')
    return source, provenance


def _probes(n):
    rng = np.random.default_rng(20261006)
    yield np.zeros(n, dtype=np.int64)
    yield np.ones(n, dtype=np.int64)
    single = np.zeros(n, dtype=np.int64)
    single[-1] = 1
    yield single
    for _ in range(4):
        yield rng.integers(0, 2, n, dtype=np.int64)


def _checked_exact_proof(entry, source=None, problem=None, root=ROOT):
    """Check durable complete-search evidence, not a heuristic/BKS declaration."""
    specification = entry['reference_verification']
    path = _local(specification['proof_path'], root)
    if sha256(path.read_bytes()) != specification['proof_sha256']:
        raise ValueError('Independent exact reference proof checksum mismatch')
    proof = readJson(path)
    runs = proof.get('completed_exact_searches', [])
    target = entry['normalized_reference_objective_min']
    if (proof.get('status') != 'independently_verified_optimum' or
            proof.get('published_numeric_reference') is not False or
            proof.get('original_vertex_weights_preserved') is not True or
            proof.get('input_sha256') != entry['raw_sha256'] or
            proof.get('objective_min') != target or proof.get('objective_max') != -target or
            len(runs) != 2 or {r.get('method') for r in runs} != {'authors_MCQDW', 'independent_Cliquer'}):
        raise ValueError('Independent exact reference proof disagrees with the original input/catalog')
    archive_evidence = []
    scores = []
    for run in runs:
        if (run.get('exit_code') != 0 or run.get('completed') is not True or
                run.get('exact') is not True or run.get('timed_out') is not False or
                run.get('configured_time_limit') is not None or run.get('configured_step_limit') is not None or
                run.get('stderr', '').strip() or run.get('weight_integer') != -target):
            raise ValueError('Exact reference search failed, was cut short or differs from the target')
        labels = run.get('source_vertex_labels', [])
        if not labels or len(labels) != len(set(labels)) or any(type(i) is not int or not 1 <= i <= entry['binary_variables'] for i in labels):
            raise ValueError('Invalid exact reference source witness labels')
        if run['method'] == 'authors_MCQDW':
            result = json.loads(run['stdout'])
            if (result.get('completed') is not True or result.get('exact') is not True or
                    result.get('configured_time_limit') is not None or result.get('configured_step_limit') is not None or
                    result.get('weight_integer') != -target or result.get('source_vertex_labels') != labels):
                raise ValueError('Author exact-search completion output disagrees with its journal')
        else:
            match = re.fullmatch(r'\s*size=(\d+), weight=(\d+):\s*([\d\s]+)', run['stdout'])
            if (not match or int(match[1]) != len(labels) or int(match[2]) != -target or
                    list(map(int, match[3].split())) != labels or
                    run['command'][1:7] != ['-q', '-q', '-s', '-m', '0', '-M'] or run['command'][7] != '0'):
                raise ValueError('Cliquer output/configuration is not a completed global-maximum search')
        _, evidence = SourceCache(_local(entry['source_data_directory'], root), 'evidence').read(run['source_archive_url'])
        if evidence['sha256'] != run['source_archive_sha256']:
            raise ValueError('Independent exact solver source archive checksum differs')
        archive_evidence.append(evidence)
        x = np.zeros(entry['binary_variables'], dtype=np.int64)
        x[np.asarray(labels)-1] = 1
        if source is not None:
            selected = np.asarray(labels)-1
            if any(not source.adjacency[i, j] for k, i in enumerate(selected) for j in selected[k+1:]):
                raise ValueError('Independent exact reference witness is not a source clique')
            if -sum(int(source.weights[i]) for i in selected) != target or source.score(x) != target:
                raise ValueError('Independent original vertex-weight witness score differs')
        if problem is not None and problem.score(x) != target:
            raise ValueError('Independent exact witness canonical QUBO score differs')
        scores.append(dict(method=run['method'], source_vertex_labels=labels, objective_min=target,
                           canonical_score_checked=problem is not None,
                           original_weights_and_all_clique_pairs_checked=source is not None))
    return dict(reference_objective=target, reference_status='independently_verified_optimum',
                published_objective=None, published_status=None, published_numeric_reference=False,
                evidence_checked=True, checked_evidence=archive_evidence[0],
                additional_checked_evidence=archive_evidence[1:], proof_path=specification['proof_path'],
                proof_sha256=specification['proof_sha256'], reference_vector_available=True,
                reference_vector_evaluated=source is not None and problem is not None,
                witness_checks=scores, completed_independent_exact_searches=2,
                independent_optimality_certificate_verified=False,
                proof_note='Two unbounded, successfully completed independent exact algorithms agree. Both source-weight clique witnesses are checked independently; this is not a published value or a separately checkable formal certificate.')


def _reference(entry, source, problem, root=ROOT):
    if entry['reference_status'] == 'independently_verified_optimum':
        return _checked_exact_proof(entry, source, problem, root)
    record = dict(entry['reference_verification'])
    record.update(published_objective=entry['normalized_reference_objective_min'],
                  published_status=entry['reference_status'],
                  reference_vector_available=False, reference_vector_evaluated=False,
                  proof_note='A feasible witness verifies attainability, not an independent optimality proof.')
    kind = record.get('check_kind')
    if kind == 'biqmac_table':
        payload, evidence = SourceCache(DEFAULT_DATA, 'evidence').read('https://biqmac.aau.at/biqmaclib.tex')
        match = re.search(r'^' + re.escape(entry['instance_id']) + r'\s*&[^\n]*', payload.decode(), re.MULTILINE)
        if not match or str(entry['normalized_reference_objective_min']) not in match.group():
            raise ValueError('Primary Biq Mac table does not match the reference')
        bounded = r'\le' in match.group()
        if bounded != (entry['reference_status'] == 'published_best_known'):
            raise ValueError('Primary table bound/optimum status differs from catalog')
        record.update(evidence_checked=True, checked_evidence=evidence,
                      primary_table_row=match.group().strip())
    elif kind == 'angers_html_table':
        payload, evidence = SourceCache(_local(entry['source_data_directory'], root), 'evidence').read(record['primary_source_url'])
        pattern = r'P5000\.2.*?10836019'
        if not re.search(pattern, payload.decode(), re.DOTALL):
            raise ValueError('Primary Angers table differs from the published target')
        record.update(evidence_checked=True, checked_evidence=evidence)
    elif not record.get('evidence_checked'):
        raise ValueError('Published reference has not been checked against its primary table')
    else:
        _, evidence = SourceCache(_local(entry['source_data_directory'], root), 'evidence').read(record['primary_source_url'])
        record['checked_evidence'] = evidence
    witness_url = entry.get('reference_solution_vector_url')
    if witness_url:
        cache_directory = _local(entry['source_data_directory'], root)
        payload, witness = SourceCache(cache_directory, 'references').read(witness_url)
        if entry.get('reuse_qubo37'):
            x, details = parseWitness(payload, source, entry['normalized_reference_objective_min'])
        else:
            x = binaryVector(np.array(payload.split(), dtype=np.int64), problem.n)
            details = {'index_mapping': 'published vector position k -> binary index k-1'}
        actual = source.score(x)
        if actual != entry['normalized_reference_objective_min'] or problem.score(x) != actual:
            raise ValueError('Public reference witness does not reproduce the published target')
        record.update(reference_vector_available=True, reference_vector_evaluated=True,
                      witness_source=witness, witness_score=actual, witness=details)
    return record


def verify_continuous_problem(entry, normalized_path=None, root=ROOT, independent=True):
    if entry.get('preparation_status') == 'unavailable':
        raise ContinuousInputUnavailable(entry['unavailable_reason'])
    path = Path(normalized_path) if normalized_path else _local(entry['normalized_path'], root)
    if path.resolve() != _local(entry['normalized_path'], root):
        raise ValueError('Selected normalized path differs from catalog')
    if sha256(path.read_bytes()) != entry.get('normalized_sha256'):
        raise ValueError('Continuous normalized input checksum mismatch')
    problem = Problem.load(path)
    if problem.n != entry['binary_variables']:
        raise ValueError('Continuous normalized dimension mismatch')
    count = int(np.count_nonzero((problem.rows != problem.cols) & (problem.values != 0)))
    density = 100 * count / (problem.n * (problem.n - 1) // 2)
    if count != entry['nonzero_offdiagonal_pairs'] or density != entry['actual_density_percent']:
        raise ValueError('Measured QUBO interactions differ from catalog')
    if ('sparse' if density < SPARSE_CUTOFF_PERCENT else 'dense') != entry['density_category']:
        raise ValueError('Measured QUBO density differs from selection')
    _, raw = SourceCache(_local(entry['source_data_directory'], root)).read(entry['retrieval_url'])
    if raw['sha256'] != entry['raw_sha256']:
        raise ValueError('Continuous public source checksum mismatch')
    metadata_path = _local(entry['verification_path'], root)
    metadata = readJson(metadata_path)
    if metadata['normalized_sha256'] != entry['normalized_sha256'] or metadata['raw_source']['sha256'] != raw['sha256']:
        raise ValueError('Continuous verification provenance mismatch')
    if not metadata['reference']['evidence_checked']:
        raise ValueError('Continuous reference primary evidence is unverified')
    reference = metadata['reference']
    if (reference.get('reference_objective', reference.get('published_objective')) != entry['normalized_reference_objective_min'] or
            reference.get('reference_status', reference.get('published_status')) != entry['reference_status']):
        raise ValueError('Continuous reference differs from verified preparation metadata')
    evidence = metadata['reference']['checked_evidence']
    evidence_directory = (DEFAULT_DATA if entry['reference_verification']['check_kind'] == 'biqmac_table'
                          else _local(entry['source_data_directory'], root))
    _, checked_evidence = SourceCache(evidence_directory, 'evidence').read(evidence['source_url'])
    if checked_evidence['sha256'] != evidence['sha256']:
        raise ValueError('Continuous primary reference evidence checksum mismatch')
    if entry['reference_status'] == 'independently_verified_optimum':
        _checked_exact_proof(entry, problem=problem, root=root)
    checked = 0
    if independent:
        source, _ = _source(entry, root)
        for x in _probes(problem.n):
            if source.score(x) != problem.score(x):
                raise ValueError('Independent public-source score differs from canonical QUBO')
            checked += 1
        _reference(entry, source, problem, root)
    return dict(instance_id=entry['instance_id'], status='passed', binary_variables=problem.n,
                measured_density=density, nonzero_offdiagonal_pairs=count,
                normalized_sha256=entry['normalized_sha256'], raw_sha256=raw['sha256'],
                independently_scored_probes=checked, reference=metadata['reference'])


def prepare_continuous_inputs(offline=False, catalog_path=None, instance_ids=None):
    path = Path(catalog_path or CONTINUOUS_CATALOG)
    catalog = load_continuous_catalog(path)
    report = []
    for entry in catalog['instances']:
        if instance_ids is not None and entry['instance_id'] not in instance_ids:
            report.append(dict(instance_id=entry['instance_id'], status=entry['preparation_status'], skipped=True))
            continue
        if entry.get('preparation_status') == 'unavailable':
            report.append(dict(instance_id=entry['instance_id'], status='unavailable', error=entry['unavailable_reason']))
            continue
        try:
            directory = _local(entry['source_data_directory'])
            if not entry.get('reuse_qubo37'):
                if entry['source_kind'] == 'weighted_clique':
                    retrieve_public_zip_member(entry['archive_url'], entry['archive_member'], directory, offline)
                else:
                    SourceCache(directory).fetch(entry['input_url'], offline=offline,
                                                 allowDocument=entry['source_kind'] == 'mqlib_original_ubqp')
                for url in entry.get('preparation_evidence_urls', []):
                    SourceCache(directory, 'evidence').fetch(url, offline=offline, allowDocument=True)
                if entry.get('reference_solution_vector_url'):
                    SourceCache(directory, 'references').fetch(entry['reference_solution_vector_url'], offline=offline)
            source, provenance = _source(entry)
            entry['raw_sha256'] = provenance['sha256']
            npz = _local(entry['normalized_path'])
            if entry.get('reuse_qubo37'):
                original = next(e for e in loadCatalog()['instances'] if e['instance_id'] == entry['reuse_qubo37'])
                problem, _ = loadPrepared(original, directory)
            else:
                problem = source.canonical()
                problem.save(npz)
            for x in _probes(problem.n):
                if source.score(x) != problem.score(x):
                    raise ValueError('Independent source/canonical score mismatch during preparation')
            reference = _reference(entry, source, problem)
            count = int(np.count_nonzero((problem.rows != problem.cols) & (problem.values != 0)))
            density = 100 * count / (problem.n * (problem.n - 1) // 2)
            if ('sparse' if density < SPARSE_CUTOFF_PERCENT else 'dense') != entry['density_category']:
                raise ValueError('Actual QUBO density does not fit the requested slot')
            entry.update(preparation_status='ready', normalized_sha256=sha256(npz.read_bytes()),
                         raw_sha256=provenance['sha256'], retrieval_url=provenance['source_url'],
                         nonzero_offdiagonal_pairs=count, actual_density_percent=density)
            writeJson(_local(entry['verification_path']), dict(instance_id=entry['instance_id'],
                      normalized_sha256=entry['normalized_sha256'], raw_source=provenance,
                      reference=reference, independently_scored_probes=7, normalization=entry['normalization'],
                      objective='minimize offset + x.T @ symmetric_Q @ x', created_at=stamp()))
            report.append(dict(instance_id=entry['instance_id'], status='ready', n=problem.n,
                               actual_density_percent=density, reference_witness_verified=reference['reference_vector_evaluated']))
        except Exception as exc:
            entry['preparation_status'] = 'error'
            report.append(dict(instance_id=entry['instance_id'], status='error', error=str(exc)))
    writeJson(path, catalog)
    writeJson(CONTINUOUS_DATA / 'preparation_report.json', dict(created_at=stamp(), instances=report,
              required_count=12, ready_count=sum(r['status'] == 'ready' for r in report)))
    return report

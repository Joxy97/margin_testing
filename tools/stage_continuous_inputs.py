"""Stage exact public cache bytes server-side without changing the frozen catalog.

Upload cache .json metadata, the reference proof journal, frozen catalog, this
tool and a canonical fingerprint manifest. Existing isolated reference files
are reused after checksum validation; remaining cache payloads are fetched
directly from their recorded public URLs. No benchmark solver is invoked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from urllib.parse import urlsplit

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from qubo_benchmark.continuous_catalog import (
    ROOT, prepare_continuous_inputs, retrieve_public_zip_member,
    verify_continuous_problem,
)
from qubo_benchmark.storage import SourceCache, readJson, writeJson

DEFAULT_MANIFEST = 'benchmark_data/continuous12/canonical_fingerprints.json'
DEFAULT_STAGED_CATALOG = 'benchmark_data/continuous12/staged_catalog.json'


def safe(root, path):
    result = (root/Path(path)).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Staging target escapes the requested checkout: {path}')
    return result


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(path):
    """Stable numeric COO digest independent of NPZ compression and dtype width.

    COO order is deliberately included: equal digests prove equal numeric
    arrays as well as equal objective. Integers on this catalog are exactly
    representable as float64. No dense matrix or Python tuple set is allocated.
    """
    digest = hashlib.sha256(b'continuous-numeric-coo-v1\0')
    with np.load(path, allow_pickle=False) as archive:
        n = int(archive['n'])
        digest.update(struct.pack('<Qd', n, float(archive['offset'])))
        count = None
        for name, dtype in (('rows', '<u4'), ('cols', '<u4'), ('values', '<f8')):
            array = archive[name]
            if array.ndim != 1 or (count is not None and len(array) != count):
                raise ValueError('Invalid canonical coordinate arrays')
            count = len(array)
            digest.update(name.encode()+b'\0'+struct.pack('<Q', count))
            for position in range(0, count, 1_000_000):
                original = array[position:position+1_000_000]
                if name == 'values' and (np.any(original > 2**53) or np.any(original < -(2**53))):
                    raise ValueError('Catalog coefficient exceeds exact binary64 fingerprint arithmetic')
                block = np.asarray(original, dtype=dtype)
                if name == 'values' and (not np.all(np.isfinite(block)) or np.any(np.abs(block) > 2**53)):
                    raise ValueError('Catalog coefficient is not exactly fingerprintable as binary64')
                digest.update(block.tobytes(order='C'))
    return digest.hexdigest()


def make_manifest(root, catalog_path, manifest_path):
    catalog = readJson(catalog_path)
    entries = {}
    for entry in catalog['instances']:
        normalized = safe(root, entry['normalized_path'])
        actual = fingerprint(normalized)
        if actual != entry['normalized_sha256']:
            raise ValueError(f'Frozen normalized checksum mismatch: {entry["instance_id"]}')
        entries[entry['instance_id']] = dict(
            numeric_coo_sha256=canonical_fingerprint(normalized),
            normalized_sha256=actual, raw_sha256=entry['raw_sha256'],
            n=entry['binary_variables'],
            objective_min=entry['normalized_reference_objective_min'],
            reference_status=entry['reference_status'])
    manifest = dict(schema_version=1, digest_method='continuous-numeric-coo-v1',
                    frozen_catalog_sha256=fingerprint(catalog_path), instances=entries)
    writeJson(manifest_path, manifest)
    return dict(status='manifest_created', inputs=len(entries), path=str(manifest_path),
                sha256=fingerprint(manifest_path))


def cache_requirements(root, catalog):
    """Only dependencies of these twelve inputs, not all historical 37 caches."""
    requirements = {}
    for entry in catalog['instances']:
        directory = entry['source_data_directory']
        urls = [('raw', entry['retrieval_url'], directory)]
        urls += [('evidence', url, directory) for url in entry.get('preparation_evidence_urls', [])]
        reference = entry['reference_verification']
        evidence_directory = 'benchmark_data/qubo37' if reference['check_kind'] == 'biqmac_table' else directory
        urls.append(('evidence', reference['primary_source_url'], evidence_directory))
        if entry.get('reference_solution_vector_url'):
            urls.append(('references', entry['reference_solution_vector_url'], directory))
        for kind, url, relative in urls:
            directory_path = safe(root, Path(relative)/kind)
            key = hashlib.sha256(url.encode()).hexdigest()
            metadata_path = safe(root, directory_path/(key+'.json'))
            record = readJson(metadata_path)
            if (record.get('source_url') != url or
                    not isinstance(record.get('bytes'), int) or not 0 < record['bytes'] <= 64*1024**2 or
                    not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64):
                raise ValueError(f'Invalid frozen source-cache metadata: {metadata_path}')
            payload_path = safe(root, directory_path/(key+'.raw'))
            if kind == 'raw' and record['sha256'] != entry['raw_sha256']:
                raise ValueError('Raw source metadata differs from the frozen input fingerprint')
            requirements[str(payload_path)] = (payload_path, metadata_path, record, kind)
    return list(requirements.values())


def valid_payload(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and fingerprint(path) == record['sha256']


def install_payload(target, source=None, payload=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='continuous-source-', suffix='.tmp', dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            if source is not None:
                with Path(source).open('rb') as original:
                    shutil.copyfileobj(original, stream, length=1024**2)
            else:
                stream.write(payload)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_payload(root, requirement, reuse_dir, timeout, retries):
    target, metadata_path, record, kind = requirement
    frozen_metadata_sha = fingerprint(metadata_path)
    if valid_payload(target, record):
        return dict(path=str(target.relative_to(root)), method='existing_verified', bytes=record['bytes'])
    if target.exists():
        # Never overwrite a corrupt or partial existing payload silently.
        raise ValueError(f'Existing cache payload fails its frozen checksum: {target}')
    candidates = []
    if reuse_dir:
        candidates.append(reuse_dir/target.name)
        if record['source_url'] == 'https://users.aalto.fi/~pat/cliquer/cliquer-1.21.tar.gz':
            candidates.append(reuse_dir/'cliquer-1.21.tar.gz')
    for source in candidates:
        if valid_payload(source, record):
            install_payload(target, source=source)
            method = 'existing_reference_file_verified'
            break
    else:
        urls = list(dict.fromkeys([record['source_url'], record.get('resolved_url', record['source_url'])]))
        errors = []
        method = None
        with tempfile.TemporaryDirectory(prefix='continuous-download-', dir=safe(root, 'benchmark_data/continuous12')) as temporary:
            temporary = Path(temporary)
            if record.get('archive_member'):
                # The original archive is 2.78GB; fetch only its authenticated
                # member with the bounded existing HTTP-range/CRC helper.
                payload, _ = retrieve_public_zip_member(record['resolved_url'], record['archive_member'], temporary)
                if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                    raise ValueError('Publisher ZIP member differs from the frozen source fingerprint')
                install_payload(target, payload=payload)
                method = 'publisher_exact_zip_ranges'
            else:
                for position, url in enumerate(urls):
                    if urlsplit(url).scheme not in ('http', 'https'):
                        raise ValueError('Only recorded public HTTP(S) URLs may be downloaded')
                    try:
                        payload, _ = SourceCache(temporary/str(position), kind).fetch(
                            url, timeout=timeout, retries=retries, allowDocument=True)
                        if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                            raise ValueError('Public download differs from the frozen source fingerprint')
                        install_payload(target, payload=payload)
                        method = 'publisher_download_verified'
                        break
                    except Exception as error:
                        errors.append(f'{url}: {error}')
                if method is None:
                    raise RuntimeError('Unable to stage exact frozen bytes: '+'; '.join(errors))
    if fingerprint(metadata_path) != frozen_metadata_sha or not valid_payload(target, record):
        raise ValueError('Staging changed cache metadata or failed exact payload validation')
    return dict(path=str(target.relative_to(root)), method=method, bytes=record['bytes'])


def stage(root, catalog_path, manifest_path, staged_catalog_path, reuse_dir, timeout, retries, require_bytes):
    frozen_catalog_sha = fingerprint(catalog_path)
    catalog = readJson(catalog_path)
    manifest = readJson(manifest_path)
    if manifest.get('frozen_catalog_sha256') != frozen_catalog_sha:
        raise ValueError('Canonical fingerprint manifest does not match the frozen catalog')
    if staged_catalog_path == catalog_path:
        raise ValueError('Staged preparation must never overwrite the frozen catalog')
    requirements = cache_requirements(root, catalog)
    staged_payloads = []
    for requirement in requirements:
        result = stage_payload(root, requirement, reuse_dir, timeout, retries)
        staged_payloads.append(result)
        print(json.dumps(dict(event='cache_staged', **result)), flush=True)
    # Preserve all original proof journals. Only preparation metadata, normalized
    # files and this separate generated catalog/report are produced by this tool.
    writeJson(staged_catalog_path, catalog)
    preparation = prepare_continuous_inputs(offline=True, catalog_path=staged_catalog_path)
    if not all(item['status'] == 'ready' for item in preparation):
        raise RuntimeError(json.dumps(preparation))
    prepared = readJson(staged_catalog_path)
    comparisons = []
    for entry in prepared['instances']:
        original = manifest['instances'][entry['instance_id']]
        if (entry['raw_sha256'] != original['raw_sha256'] or entry['binary_variables'] != original['n'] or
                entry['normalized_reference_objective_min'] != original['objective_min'] or
                entry['reference_status'] != original['reference_status']):
            raise ValueError('Staged input changed its original source/dimension/reference')
        path = safe(root, entry['normalized_path'])
        if canonical_fingerprint(path) != original['numeric_coo_sha256']:
            raise ValueError(f'Staged canonical numerical input differs: {entry["instance_id"]}')
        verify_continuous_problem(entry, path, independent=False)
        comparisons.append(dict(instance_id=entry['instance_id'], canonical_arrays_identical=True,
                                npz_bytes_identical=entry['normalized_sha256'] == original['normalized_sha256']))
    if fingerprint(catalog_path) != frozen_catalog_sha:
        raise ValueError('Frozen catalog changed during staging')
    differences = [item['instance_id'] for item in comparisons if not item['npz_bytes_identical']]
    result = dict(status='staged_and_verified', frozen_catalog_unchanged=True,
                  frozen_catalog_sha256=frozen_catalog_sha, inputs=len(comparisons),
                  same_bytes_source_caches=len(requirements), canonical_arrays_identical=True,
                  byte_different_normalized_inputs=differences,
                  frozen_catalog_directly_runnable=not differences,
                  staged_catalog_path=str(staged_catalog_path),
                  note='If NPZ compression bytes differ, use the separately generated staged catalog, or transfer only the original differing NPZ. Never silently update the frozen catalog.',
                  inputs_verified=comparisons, cache_payloads=staged_payloads)
    writeJson(safe(root, 'benchmark_data/continuous12/staging_report.json'), result)
    print(json.dumps(result, indent=2), flush=True)
    return 3 if require_bytes and differences else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--make-manifest', action='store_true', help='Read frozen local NPZ files and emit only a tiny canonical fingerprint manifest')
    parser.add_argument('--catalog', default='configs/continuous_catalog.json')
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--staged-catalog', default=DEFAULT_STAGED_CATALOG)
    parser.add_argument('--reuse-dir', type=Path, default=Path('/workspace/continuous_reference_check'))
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--require-byte-identical', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path(ROOT).resolve():
        parser.error('--root must be this deployed checkout (source module ROOT is fixed)')
    if not 0 < args.timeout <= 300 or not 0 <= args.retries <= 5:
        parser.error('Timeout/retry values exceed the bounded staging limits')
    catalog_path, manifest_path = safe(root, args.catalog), safe(root, args.manifest)
    if args.make_manifest:
        print(json.dumps(make_manifest(root, catalog_path, manifest_path), indent=2))
        return 0
    return stage(root, catalog_path, manifest_path, safe(root, args.staged_catalog),
                 args.reuse_dir.resolve(), args.timeout, args.retries, args.require_byte_identical)


if __name__ == '__main__':
    raise SystemExit(main())

"""Public input identity, exact reductions and explicit incomplete-suite checks."""
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pytest

from qubo_benchmark.continuous_catalog import (
    CONTINUOUS_CATALOG, REQUIRED_GROUPS, ContinuousInputUnavailable,
    load_continuous_catalog, parse_weighted_cut, select_continuous_problem,
    verify_continuous_problem, retrieve_public_zip_member,
    parse_weighted_clique,
)


def test_catalog_keeps_every_required_slot():
    catalog = load_continuous_catalog()
    assert len(catalog['instances']) == 12
    assert {(e['binary_variables'], e['density_category']) for e in catalog['instances']} == REQUIRED_GROUPS
    assert catalog['sparse_cutoff_percent'] == 20


def test_missing_public_input_never_silently_substitutes():
    catalog = load_continuous_catalog()
    missing = [e for e in catalog['instances'] if e['preparation_status'] == 'unavailable']
    for entry in missing:
        with pytest.raises(ContinuousInputUnavailable, match=entry['unavailable_reason'][:35]):
            select_continuous_problem(entry['binary_variables'], entry['density_category'], catalog)


def test_catalog_rejects_narrowed_or_mislabeled_suite(tmp_path):
    catalog = load_continuous_catalog()
    catalog['instances'].pop()
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='twelve'):
        load_continuous_catalog(path)
    catalog = load_continuous_catalog()
    catalog['instances'][0]['actual_density_percent'] = 90
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='density'):
        load_continuous_catalog(path)


def test_weighted_cut_conversion_independent_scores_and_last_helper():
    # Original three-variable maximization matrix: diagonal [4,-3,2],
    # interactions q01=2,q02=-1,q12=3. Public MaxCut encoding adds last vertex.
    payload = b'# public format\n4 6\n1 2 -2\n1 3 1\n2 3 -3\n1 4 5\n2 4 2\n3 4 4\n'
    source = parse_weighted_cut(BytesIO(payload), fixed_last_vertex=True)
    problem = source.canonical()
    assert problem.n == 3
    for value in range(8):
        x = np.array([(value >> k) & 1 for k in range(3)])
        original = 4*x[0]-3*x[1]+2*x[2]+4*x[0]*x[1]-2*x[0]*x[2]+6*x[1]*x[2]
        assert source.score(x) == problem.score(x) == -original


def test_original_weighted_clique_conversion_is_exact_without_dropping_weights():
    payload = b'p edge 3 2\nn 1 17\nn 2 29\nn 3 31\ne 1 2\ne 2 3\n'
    source = parse_weighted_clique(BytesIO(payload))
    problem = source.canonical()
    assert source.penalty == 32
    values = []
    for value in range(8):
        x = np.array([(value >> k) & 1 for k in range(3)])
        expected = -17*x[0]-29*x[1]-31*x[2]+64*x[0]*x[2]
        assert source.score(x) == problem.score(x) == expected
        values.append(expected)
    # The weighted clique [2,3] is not the same objective as clique cardinality.
    assert min(values) == -60


@pytest.mark.parametrize('payload', [
    b'p edge 2 1\nn 1 17\nn 1 29\ne 1 2\n',
    b'p edge 2 1\nn 1 17\nn 2 -29\ne 1 2\n',
    b'p edge 2 1\nn 1 17\nn 2 29\ne 1 1\n',
    b'p edge 3 2\nn 1 17\nn 2 29\nn 3 31\ne 1 2\ne 2 1\n',
    b'p edge 2 1\nn 1 17\nn 2 29\ne 1 2 3\n',
    b'p edge 2 1\nn 1 17\nn 2 29\n',
    b'p edge 2 1\nn 1 17\nn 2 29\ne 1 2\ne 2 1\n',
])
def test_original_weighted_clique_rejects_changed_or_invalid_data(payload):
    with pytest.raises(ValueError):
        parse_weighted_clique(BytesIO(payload))


@pytest.mark.parametrize('payload', [
    b'3 2\n1 2 1\n2 1 1\n',
    b'3 2\n1 2 1\n',
    b'3 1\n1 4 1\n',
    b'3 1\n1 1 1\n',
    b'3 1\n1 2 1\n1 3 2\n',
    b'<html>login page</html>\n',
])
def test_weighted_cut_rejects_invalid_public_inputs(payload):
    with pytest.raises(ValueError):
        parse_weighted_cut(BytesIO(payload))


def test_prepared_public_inputs_are_hash_checked_without_dense_matrix_allocation():
    catalog = load_continuous_catalog()
    for entry in catalog['instances']:
        if entry['preparation_status'] == 'unavailable':
            continue
        selected, path = select_continuous_problem(entry['binary_variables'], entry['density_category'], catalog)
        assert selected == entry
        report = verify_continuous_problem(entry, path, independent=entry['binary_variables'] <= 500)
        assert report['status'] == 'passed'
        assert report['reference']['evidence_checked']
    entry, _ = select_continuous_problem(1000, 'dense', catalog)
    assert entry['actual_density_percent'] > 75
    assert entry['reference_status'] == 'published_proven_optimum'
    assert entry['normalized_path'].startswith('benchmark_data/qubo37/')


def test_refusing_changed_reference_and_changed_normalized_checksum():
    entry, path = select_continuous_problem(100, 'sparse')
    changed = dict(entry, normalized_sha256='0' * 64)
    with pytest.raises(ValueError, match='checksum'):
        verify_continuous_problem(changed, path)
    changed = dict(entry, normalized_reference_objective_min=entry['normalized_reference_objective_min']-1)
    with pytest.raises(ValueError, match='reference'):
        verify_continuous_problem(changed, path)


def test_public_zip_member_ranges_and_cache_preserve_original_bytes(tmp_path, monkeypatch):
    import qubo_benchmark.continuous_catalog as module
    original = b'p edge 3 2\nn 1 17\nn 2 29\nn 3 31\ne 1 2\ne 2 3\n'
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('data/public.clq', original)
        archive.writestr('data/another.clq', b'unselected member')
    encoded = buffer.getvalue()
    requests = []

    class Response:
        status = 206

        def __init__(self, request):
            value = request.get_header('Range')
            requests.append(value)
            if value.startswith('bytes=-'):
                begin, end = max(0, len(encoded)-int(value[7:])), len(encoded)-1
            else:
                begin, end = map(int, value[6:].split('-'))
            self.payload = encoded[begin:end+1]
            self.headers = {'Content-Range': f'bytes {begin}-{end}/{len(encoded)}', 'ETag': 'immutable-public-zip'}

        def read(self, count):
            return self.payload[:count]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(module, 'urlopen', lambda request, **kwargs: Response(request))
    payload, meta = retrieve_public_zip_member('https://example.org/published.zip', 'data/public.clq', tmp_path)
    assert payload == original
    assert meta['bytes'] == len(original)
    assert meta['archive_member'] == 'data/public.clq'
    assert len(requests) == 4
    cached, cached_meta = retrieve_public_zip_member('https://example.org/published.zip', 'data/public.clq', tmp_path, offline=True)
    assert cached == original and cached_meta == meta
    assert len(requests) == 4


@pytest.mark.parametrize('failure', ['timed_out', 'exit_code', 'configured_step_limit', 'completed'])
def test_independent_optimum_rejects_cutoff_or_failed_exact_search(tmp_path, monkeypatch, failure):
    import qubo_benchmark.continuous_catalog as module
    from qubo_benchmark.storage import readJson, sha256
    catalog = load_continuous_catalog()
    entry = next(e for e in catalog['instances'] if e['reference_status'] == 'independently_verified_optimum')
    entry = json.loads(json.dumps(entry))
    specification = entry['reference_verification']
    proof = readJson(module._local(specification['proof_path']))
    proof['completed_exact_searches'][0][failure] = {
        'timed_out': True, 'exit_code': 1, 'configured_step_limit': 100, 'completed': False,
    }[failure]
    changed = tmp_path/'changed_proof.json'
    changed.write_text(json.dumps(proof))
    specification['proof_sha256'] = sha256(changed.read_bytes())
    original_local = module._local
    monkeypatch.setattr(module, '_local', lambda path, root=module.ROOT:
                        changed if path == specification['proof_path'] else original_local(path, root))
    with pytest.raises(ValueError, match='failed|cut short'):
        module._checked_exact_proof(entry)


def test_dense10000_reference_is_not_misrepresented_as_published():
    catalog = load_continuous_catalog()
    entry = next(e for e in catalog['instances'] if (e['binary_variables'], e['density_category']) == (10000, 'dense'))
    assert entry['reference_status'] == 'independently_verified_optimum'
    assert entry['normalized_reference_objective_min'] == -8873650
    assert entry['reference_verification']['published_numeric_reference'] is False
    assert entry['reference_verification']['independent_optimality_certificate_verified'] is False


def _staging_tool():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'tools/stage_continuous_inputs.py'
    specification = importlib.util.spec_from_file_location('continuous_staging_test', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _staging_requirement(module, root, payload):
    from qubo_benchmark.storage import writeJson, sha256
    url = 'https://example.org/authentic.dat'
    key = sha256(url.encode())
    target = root/'benchmark_data/continuous12/raw'/(key+'.raw')
    metadata = target.with_suffix('.json')
    record = dict(source_url=url, resolved_url=url, bytes=len(payload), sha256=sha256(payload))
    writeJson(metadata, record)
    return target, metadata, record, 'raw'


def test_stager_reuses_exact_source_without_changing_metadata(tmp_path):
    module = _staging_tool()
    payload = b'authentic original input\n'
    requirement = _staging_requirement(module, tmp_path, payload)
    target, metadata, _, _ = requirement
    previous = metadata.read_bytes()
    reuse = tmp_path/'isolated_reference'
    reuse.mkdir()
    (reuse/target.name).write_bytes(payload)
    result = module.stage_payload(tmp_path, requirement, reuse, 1, 0)
    assert result['method'] == 'existing_reference_file_verified'
    assert target.read_bytes() == payload
    assert metadata.read_bytes() == previous


def test_stager_rejects_corrupt_existing_source_without_overwriting(tmp_path):
    module = _staging_tool()
    requirement = _staging_requirement(module, tmp_path, b'original')
    target = requirement[0]
    target.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Existing cache payload'):
        module.stage_payload(tmp_path, requirement, None, 1, 0)
    assert target.read_bytes() == b'corrupted'


def test_stager_download_requires_frozen_bytes_and_preserves_metadata(tmp_path, monkeypatch):
    module = _staging_tool()
    payload = b'authentic download\n'
    requirement = _staging_requirement(module, tmp_path, payload)
    previous = requirement[1].read_bytes()
    monkeypatch.setattr(module.SourceCache, 'fetch', lambda *args, **kwargs: (payload, {'discarded_new_record': True}))
    result = module.stage_payload(tmp_path, requirement, None, 1, 0)
    assert result['method'] == 'publisher_download_verified'
    assert requirement[0].read_bytes() == payload
    assert requirement[1].read_bytes() == previous


def test_stager_never_accepts_changed_publisher_payload(tmp_path, monkeypatch):
    module = _staging_tool()
    requirement = _staging_requirement(module, tmp_path, b'original')
    monkeypatch.setattr(module.SourceCache, 'fetch', lambda *args, **kwargs: (b'changed!', {}))
    with pytest.raises(RuntimeError, match='frozen source fingerprint'):
        module.stage_payload(tmp_path, requirement, None, 1, 0)
    assert not requirement[0].exists()


def test_stager_numeric_digest_ignores_storage_dtype_and_compression(tmp_path):
    module = _staging_tool()
    first, second = tmp_path/'first.npz', tmp_path/'second.npz'
    np.savez_compressed(first, n=3, offset=0., rows=np.array([0, 1], dtype=np.uint32),
                        cols=np.array([1, 1], dtype=np.uint32), values=np.array([7, -3], dtype=np.int32))
    np.savez(second, n=3, offset=0, rows=np.array([0, 1], dtype=np.int64),
             cols=np.array([1, 1], dtype=np.int64), values=np.array([7., -3.], dtype=np.float64))
    assert module.fingerprint(first) != module.fingerprint(second)
    assert module.canonical_fingerprint(first) == module.canonical_fingerprint(second)
    with pytest.raises(ValueError, match='escapes'):
        module.safe(tmp_path, '../outside')

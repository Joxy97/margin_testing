"""Read-only evidence audit, with and without raw energy retention."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qubo_benchmark.runtime.compact_storage import CompactStore


def validator():
    path = Path(__file__).resolve().parents[1]/'tools'/'validate_continuous_results.py'
    spec = importlib.util.spec_from_file_location('compact_output_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate


@pytest.mark.parametrize('keep', [True, False])
def test_audit_reproduces_exact_metrics_without_mutation(tmp_path, keep):
    manifest = dict(solvers={'example': {}}, problems=[dict(id='public', n=10000,
        density='dense', reference=-10, reference_type='BKS')])
    store = CompactStore(tmp_path, manifest, seeds=[0, 1], keep_raw_results=keep)
    store.commit_seed('example', 'public', 0, np.full(9, -10.))
    store.commit_seed('example', 'public', 1, np.full(9, -9.))
    store.finalize()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    result = validator()(tmp_path)
    assert result['status'] == 'passed' and not result['has_failures']
    assert result['numerical_payload_bytes'] == (144 if keep else 0)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after


def test_audit_rejects_unexpected_vector_artifact(tmp_path):
    manifest = dict(solvers={'example': {}}, problems=[dict(id='public', n=100,
        density='sparse', reference=-10, reference_type='BKS')])
    store = CompactStore(tmp_path, manifest, seeds=[0])
    store.commit_seed('example', 'public', 0, np.full(9, -10.))
    store.finalize()
    np.save(tmp_path/'solution_vector.npy', np.ones(100))
    with pytest.raises(ValueError, match='Unexpected result artifact'):
        validator()(tmp_path)

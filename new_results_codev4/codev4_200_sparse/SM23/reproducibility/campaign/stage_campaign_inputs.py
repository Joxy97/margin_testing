"""Verify the two supplied inputs and write a separate campaign catalog."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
repo = root / 'repo'
sys.path.insert(0, str(repo / 'src'))
from qubo_benchmark.continuous_catalog import verify_continuous_problem
from qubo_benchmark.storage import readJson, writeJson

spec = importlib.util.spec_from_file_location('stager', repo / 'tools/stage_continuous_inputs.py')
stager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stager)
catalog = readJson(repo / 'configs/continuous_catalog.windows.json')
fingerprints = readJson(repo / 'benchmark_data/continuous12/canonical_fingerprints.json')
evidence = []
for entry in catalog['instances']:
    if entry['instance_id'] not in ('gka1e', 'gka5e'):
        entry['preparation_status'] = 'unavailable'
        entry['unavailable_reason'] = 'This campaign prepares only the requested 200-variable pair.'
        continue
    npz = repo / entry['normalized_path']
    expected = fingerprints['instances'][entry['instance_id']]
    numeric = stager.canonical_fingerprint(npz)
    if numeric != expected['numeric_coo_sha256']:
        raise ValueError('Numerical input identity mismatch: ' + entry['instance_id'])
    actual = hashlib.sha256(npz.read_bytes()).hexdigest()
    if actual != entry['normalized_sha256']:
        raise ValueError('Supplied input differs from the original Windows catalog')
    # Linux and Windows NPZ containers can differ while COO values are identical.
    # Keep the published verification untouched and write a campaign-local copy.
    provenance = readJson(repo / entry['verification_path'])
    provenance['normalized_sha256'] = actual
    entry['verification_path'] = f"benchmark_data/continuous12/campaign_verification/{entry['instance_id']}.json"
    writeJson(repo / entry['verification_path'], provenance)
    result = verify_continuous_problem(entry, npz, independent=True)
    evidence.append(dict(instance_id=entry['instance_id'], numeric_coo_sha256=numeric,
                         normalized_sha256=actual, validation=result))
writeJson(root / 'catalog.json', catalog)
writeJson(root / 'input_proof.json', dict(inputs=evidence, source_commit='62b58c9227276ac132086b2bbcc8ebfb1f85a732'))
print(json.dumps(dict(status='verified', instances=[x['instance_id'] for x in evidence]), indent=2))

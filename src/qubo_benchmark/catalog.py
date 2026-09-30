"""Selection manifest and portable benchmark locations."""

from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'benchmark_data' / 'qubo37'
GROUPS = {(200, 'sparse'): 3, (200, 'dense'): 10,
          (500, 'sparse'): 10, (500, 'dense'): 2,
          (1000, 'sparse'): 10, (1000, 'dense'): 2}
SMOKE_IDS = ('gka1e', 'be200.8.1', 'bqp500-1', 'gka5f', 'bqp1000-1',
             'p_hat1000-1__clique_QUBO')
FORMAT_URLS = (
    'https://biqmac.aau.at/biqmaclib.html',
    'https://biqmac.aau.at/biqmaclib.tex',
    'https://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html',
    'https://www.dcs.gla.ac.uk/~pat/maxClique/distribution/readme.txt',
)


def loadCatalog(directory=DEFAULT_DATA):
    path = Path(directory) / 'catalog.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    entries = catalog['instances']
    ids = [entry['instance_id'] for entry in entries]
    if len(ids) != len(set(ids)) or len(ids) != 37 or catalog['instance_count'] != 37:
        raise ValueError('Catalog must contain exactly 37 unique instances')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', name) for name in ids):
        raise ValueError('Invalid instance ID')
    counts = Counter((e['binary_variables'], e['density_category']) for e in entries)
    if counts != GROUPS:
        raise ValueError(f'Incorrect catalog size-density group counts: {counts}')
    if sum(e['problem_family'] == 'native_QUBO' for e in entries) != 33:
        raise ValueError('Expected 33 native QUBOs and four clique-derived QUBOs')
    for entry in entries:
        if not isinstance(entry['normalized_reference_objective_min'], (int, float)):
            raise ValueError('Every catalog entry needs a numeric reference objective')
        if not entry['reference_status'] or not entry['reference_sources']:
            raise ValueError('Missing published reference metadata')
    return catalog


def selectEntries(catalog, ids=None):
    entries = catalog['instances']
    if ids is None or ids == 'all':
        return entries
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError('instances must be all or a list of unique IDs')
    known = {e['instance_id']: e for e in entries}
    if set(ids) - known.keys():
        raise ValueError(f'Unknown instance IDs: {sorted(set(ids)-known.keys())}')
    return [known[name] for name in ids]

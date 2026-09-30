"""Offline preparation, provenance and independent validation of the catalog."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import numpy as np
import json
import re
from .catalog import DEFAULT_DATA, FORMAT_URLS, GROUPS, loadCatalog
from .formats import PARSER_VERSION, GraphSource, parseGraph, parseMatrix, parseWitness
from .model import Problem
from .storage import SourceCache, readJson, writeJson, sha256, stamp


def download(directory=DEFAULT_DATA, offline=False):
    jobs = {}
    for entry in loadCatalog(directory)['instances']:
        jobs['raw',sourceUrl(entry,directory)] = False
        if entry['reference_solution_vector_url']:
            jobs['references',entry['reference_solution_vector_url']] = False
        for url in entry['reference_sources']: jobs['evidence',url] = True
    for url in FORMAT_URLS: jobs['evidence',url] = True
    def fetch(item):
        (kind,url),document = item
        try:
            payload,meta = SourceCache(directory,kind).fetch(url,offline=offline,allowDocument=document)
            if kind == 'raw':
                entry=next(e for e in loadCatalog(directory)['instances'] if sourceUrl(e,directory)==url)
                if entry['problem_family']=='native_QUBO':parseMatrix(payload,entry.get('position_in_bundle_1_based'))
                else:parseGraph(payload)
            return dict(kind=kind,url=url,status='cached' if offline else 'downloaded',sha256=meta['sha256'])
        except Exception as exc: return dict(kind=kind,url=url,status='error',error=str(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:
        report = [future.result() for future in as_completed([pool.submit(fetch,j) for j in jobs.items()])]
    report.sort(key=lambda r:(r['kind'],r['url']))
    writeJson(Path(directory)/'download_report.json',report)
    return report


def sourceUrl(entry,directory):
    overrides=Path(directory)/'source_overrides.json'
    record=readJson(overrides).get(entry['instance_id'],{}) if overrides.exists() else {}
    return record.get('retrieval_url',entry['input_url'])


def readSource(entry,directory):
    payload,provenance = SourceCache(directory).read(sourceUrl(entry,directory))
    if entry['problem_family'] == 'native_QUBO':
        source = parseMatrix(payload,entry.get('position_in_bundle_1_based'),
                             -1 if entry['source_objective_sense'] == 'max' else 1)
    else: source = parseGraph(payload)
    if source.n != entry['binary_variables']: raise ValueError('Declared dimension differs from catalog')
    return source,provenance


def paths(directory,name):
    base = Path(directory)/'normalized'/name
    return base.with_suffix(base.suffix+'.npz'), base.with_suffix(base.suffix+'.json')


def referenceRecord(entry,source,problem,directory):
    url = entry['reference_solution_vector_url']
    record = dict(published_objective=entry['normalized_reference_objective_min'],
                  published_status=entry['reference_status'],sources=entry['reference_sources'],
                  evidence_checked=False,evidence_note='Published status retained from catalog; downloading a document is not verification of its claim.',
                  reference_vector_available=url is not None,reference_vector_evaluated=False)
    # Check exact table rows, preserving the original catalog status separately.
    if entry['problem_family'] == 'native_QUBO':
        try:
            if entry.get('position_in_bundle_1_based') is None:
                urlEvidence='https://biqmac.aau.at/biqmaclib.tex'
                payload,evidence=SourceCache(directory,'evidence').read(urlEvidence)
                pattern=r'^'+re.escape(entry['instance_id'])+r'\s*&.*?'+str(entry['normalized_reference_objective_min'])+r'\b'
                matched=bool(re.search(pattern,payload.decode(),re.MULTILINE))
            else:
                urlEvidence='https://raw.githubusercontent.com/rliang/qubo-benchmark-instances/main/metadata.json'
                payload,evidence=SourceCache(directory,'evidence').read(urlEvidence)
                key=entry['instance_id'].replace('-','.')
                matched=json.loads(payload)[key]['best']==entry['normalized_reference_objective_min']
            record.update(evidence_checked=True,published_value_matches_checked_source=matched,
                          checked_evidence=evidence,
                          evidence_note='Exact objective table entry checked; no certificate independently verified. Original catalog status preserved.')
            if not matched:record['evidence_issue']='Catalog target differs from checked source; original target retained.'
        except (OSError,ValueError,KeyError) as exc:record['evidence_issue']=str(exc)
    if url:
        payload,provenance = SourceCache(directory,'references').read(url)
        x,details = parseWitness(payload,source,entry['normalized_reference_objective_min'])
        vectorPath = Path(directory)/'references'/(entry['instance_id']+'.npz')
        np.savez_compressed(vectorPath,x=x)
        record.update(reference_vector_evaluated=True,witness=details,witness_source=provenance,
                      vector_file=vectorPath.name,vector_sha256=sha256(vectorPath.read_bytes()),
                      proof_note='A feasible witness is not an independent optimality proof.')
    return record


def normalize(directory=DEFAULT_DATA):
    directory = Path(directory); rows=[]
    for entry in loadCatalog(directory)['instances']:
        name=entry['instance_id']
        try:
            source,provenance=readSource(entry,directory); problem=source.canonical()
            npz,metaPath=paths(directory,name);problem.save(npz)
            reference=referenceRecord(entry,source,problem,directory)
            count=int(np.count_nonzero((problem.rows != problem.cols)&(problem.values != 0)))
            if isinstance(source,GraphSource):
                if len(source.edges) != entry['original_graph_edges'] or count != entry['nonzero_offdiagonal_pairs']:
                    raise ValueError('Published graph edge/nonedge counts differ from catalog')
            meta=dict(instance_id=name,catalog_entry=entry,raw_source=provenance,
                      parser_version=PARSER_VERSION,normalized_sha256=sha256(npz.read_bytes()),
                      objective='minimize offset + x.T @ symmetric_Q @ x',offset=problem.offset,
                      source_labels=list(range(1,problem.n+1)),index_mapping='source label k -> index k-1',
                      normalization=('diagonal -1; nonedge offdiagonal +1; no auxiliary variables'
                                     if isinstance(source,GraphSource) else f'mirror missing reverse entries once; multiply source coefficients by {source.sign}'),
                      measured_nonzero_offdiagonal_pairs=count,
                      measured_interaction_density_percent=100*count/(problem.n*(problem.n-1)//2),reference=reference)
            writeJson(metaPath,meta);rows.append(dict(instance_id=name,status='ready'))
        except Exception as exc: rows.append(dict(instance_id=name,status='error',error=str(exc)))
    report=dict(created_at=stamp(),instances=rows,groups=readiness(rows,directory))
    writeJson(directory/'normalization_report.json',report);return report


def readiness(rows,directory):
    states={r['instance_id']:r['status'] for r in rows}; groups=[]
    entries=loadCatalog(directory)['instances']
    for (n,density),expected in GROUPS.items():
        ids=[e['instance_id'] for e in entries if (e['binary_variables'],e['density_category'])==(n,density)]
        ready=[name for name in ids if states.get(name) in ('ready','passed')]
        groups.append(dict(n=n,density=density,expected=expected,ready=len(ready),
                           missing=[name for name in ids if name not in ready]))
    return groups


def loadPrepared(entry,directory=DEFAULT_DATA):
    npz,metaPath=paths(directory,entry['instance_id']);meta=readJson(metaPath)
    if meta['catalog_entry'] != entry or meta['normalized_sha256'] != sha256(npz.read_bytes()):
        raise ValueError('Normalized metadata/catalog/hash mismatch')
    _,raw=SourceCache(directory).read(sourceUrl(entry,directory))
    if meta['raw_source']['sha256'] != raw['sha256']: raise ValueError('Raw/normalized provenance mismatch')
    problem=Problem.load(npz)
    if problem.n != entry['binary_variables']: raise ValueError('Normalized dimension mismatch')
    return problem,meta


def validate(directory=DEFAULT_DATA):
    directory=Path(directory);rows=[]
    for entry in loadCatalog(directory)['instances']:
        name=entry['instance_id']
        try:
            problem,meta=loadPrepared(entry,directory);source,_=readSource(entry,directory)
            rng=np.random.default_rng(20260929)
            vectors=[np.zeros(problem.n,dtype=int),np.ones(problem.n,dtype=int),
                     np.eye(1,problem.n,problem.n-1,dtype=int)[0]]
            vectors += [rng.integers(0,2,problem.n) for _ in range(4)]
            dense=problem.dense();sparse=problem.sparse()
            for x in vectors:
                expected=source.score(x)
                scores=[problem.score(x),problem.offset+x@dense@x,problem.offset+x@sparse@x]
                if any(v != expected for v in scores): raise ValueError(f'Independent objective mismatch: {scores} vs {expected}')
            reference=meta['reference']
            if entry['reference_solution_vector_url']:
                payload,_=SourceCache(directory,'references').read(entry['reference_solution_vector_url'])
                x,details=parseWitness(payload,source,entry['normalized_reference_objective_min'])
                vectorPath=directory/'references'/reference['vector_file']
                if sha256(vectorPath.read_bytes()) != reference['vector_sha256']: raise ValueError('Reference vector hash mismatch')
                with np.load(vectorPath,allow_pickle=False) as z:
                    if not np.array_equal(z['x'],x): raise ValueError('Stored witness differs from source')
                if problem.score(x) != entry['normalized_reference_objective_min']: raise ValueError('Witness objective mismatch')
            elif reference['reference_vector_available'] or reference['reference_vector_evaluated']:
                raise ValueError('Invented reference vector')
            rows.append(dict(instance_id=name,status='passed',vectors_checked=len(vectors),
                             reference_vector_evaluated=reference['reference_vector_evaluated']))
        except Exception as exc: rows.append(dict(instance_id=name,status='error',error=str(exc)))
    report=dict(created_at=stamp(),passed=sum(r['status']=='passed' for r in rows),expected=37,
                instances=rows,groups=readiness(rows,directory),float_tolerance='Integer source coefficients: exact equality; float64 helper objectives: rtol=1e-12, atol=1e-9.')
    writeJson(directory/'validation_report.json',report);return report

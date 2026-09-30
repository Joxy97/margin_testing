"""Offline benchmark contract tests. Missing catalog data is a failure, never a skip."""
from itertools import combinations, product
from pathlib import Path
import tempfile
import json
import time
import unittest
from unittest.mock import patch
import numpy as np
from qubo_benchmark.adapters import (Adapter, SOLVERS, GPU_SOLVERS, CLASSICAL_SOLVERS,
                                     UnsupportedProblem, BackendUnavailable, toSolverProblem)
from qubo_benchmark.catalog import DEFAULT_DATA, ROOT, GROUPS, loadCatalog
from qubo_benchmark.formats import GraphSource, parseGraph, parseMatrix, parseWitness
from qubo_benchmark.model import Problem, binaryVector
from qubo_benchmark.pipeline import loadPrepared, validate
from qubo_benchmark.runner import supervise, comparison, checkConfig, effectiveConfig, run
from qubo_benchmark.storage import SourceCache, rejectHtml, writeJson, sha256, readJson


def stuckWorker(connection,path,specification,seed,config):
    if specification.get('ready'):
        connection.send({'kind':'ready'});connection.recv()
    time.sleep(30)


class ParserTests(unittest.TestCase):
    def test_triangles_and_full_symmetric(self):
        variants=[b'4 3\n1 1 1\n1 2 2\n2 2 -3',
                  b'4 3\n1 1 1\n2 1 2\n2 2 -3',
                  b' c comment\r\n\n4 4\n 1 1 1 \n1 2 2\n2 1 2\n2 2 -3\n']
        for raw in variants:
            source=parseMatrix(raw);q=source.canonical()
            self.assertEqual(q.n,4)
            self.assertEqual(q.score([1,1,1,1]),2)
            self.assertEqual(source.score([1,1,1,1]),2)
            self.assertTrue(np.all(q.dense()[3]==0))

    def test_bundle_sign_and_index(self):
        raw=b'2\n2 1\n1 1 4\n3 2\n2 3 7\n3 3 -2\n'
        source=parseMatrix(raw,2,-1)
        self.assertEqual(source.n,3)
        self.assertEqual(source.canonical().score([0,1,1]),-12)
        self.assertEqual(source.score([0,1,1]),-12)
        with self.assertRaises(ValueError):parseMatrix(raw,3)

    def test_invalid_matrices(self):
        malformed=[b'2 2\n1 2 1\n1 2 1',b'2 2\n1 2 1\n2 1 2',
                   b'2 1\n0 1 2',b'2 1\n1 3 2',b'2 2\n1 1 2',
                   b'2 0\n1 1 2',b'2 1\n1 1 nan',b'0 0',b'<html>login</html>']
        for raw in malformed:
            with self.subTest(raw=raw),self.assertRaises(ValueError):parseMatrix(raw)

    def test_invalid_downloads_and_cache(self):
        for raw in (b'',b'<!doctype html>oops',b'<html>error',b'\x00binary'):
            with self.assertRaises(ValueError):rejectHtml(raw)
        with tempfile.TemporaryDirectory() as tmp:
            cache=SourceCache(tmp);url='https://example.invalid/data'
            with self.assertRaises(FileNotFoundError):cache.fetch(url,offline=True)
            raw,meta=cache.paths(url);raw.write_bytes(b'2 0\n')
            writeJson(meta,dict(source_url=url,bytes=4,sha256=sha256(b'2 0\n')))
            self.assertEqual(cache.fetch(url,offline=True)[0],b'2 0\n')
            raw.write_bytes(b'3 0\n')
            with self.assertRaises(ValueError):cache.fetch(url,offline=True)

    def test_graph_errors(self):
        bad=[b'p edge 2 1\ne 1 1',b'p edge 2 1\ne 0 2',
             b'p edge 2 1\ne 1 3',b'p edge 3 2\ne 1 2\ne 2 1',
             b'p edge 3 2\ne 1 2',b'e 1 2\np edge 2 1',
             b'p edge 2 0\np edge 2 0',b'p edge 2 0\nx 1']
        for raw in bad:
            with self.subTest(raw=raw),self.assertRaises(ValueError):parseGraph(raw)

    def test_witness_mapping_and_failures(self):
        graph=parseGraph(b'p edge 4 1\ne 2 4')
        x,details=parseWitness(b'2 5 10\n2 4',graph,-2)
        np.testing.assert_array_equal(x,[0,1,0,1])
        for raw in (b'2 5 10\n1 3',b'2 5 10\n2 2',b'2 5 10\n0 4',b'3 5 10\n2 4'):
            with self.assertRaises(ValueError):parseWitness(raw,graph,-2)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.q=Problem(2,np.array([0,0,1]),np.array([0,1,1]),np.array([1,2,-3]),7)

    def test_hand_objective_roundtrip_dense_sparse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'q.npz';self.q.save(path);loaded=Problem.load(path)
            for x,expected in [([0,0],7),([1,0],8),([0,1],4),([1,1],9)]:
                a=np.array(x)
                self.assertEqual(self.q.score(x),expected)
                self.assertEqual(loaded.score(x),expected)
                self.assertEqual(7+a@self.q.dense()@a,expected)
                self.assertEqual(7+a@self.q.sparse()@a,expected)

    def test_adapter_objective(self):
        bqm=toSolverProblem(self.q)
        self.assertEqual(bqm.variableCount,2)
        self.assertEqual(len(bqm.oneHotGroups),0)
        for x in product((0,1),repeat=2):
            a=np.array(x)
            value=bqm.offset+bqm.linear@a+np.sum(bqm.quadraticBiases*a[bqm.quadraticHeads]*a[bqm.quadraticTails])
            self.assertEqual(value,self.q.score(a))

    def test_overflow_and_float(self):
        q=Problem(1,np.array([0]),np.array([0]),np.array([2**63-1],dtype=np.int64),10)
        self.assertEqual(q.score([1]),2**63+9)
        q=Problem(1,np.array([0]),np.array([0]),np.array([.125]),.5)
        self.assertAlmostEqual(q.score([1]),.625)

    def test_binary_validation(self):
        for value in ([0,2],[0,.5],[0,float('nan')],[0,float('inf')],[[0,1]],[0],['0','1']):
            with self.subTest(value=value),self.assertRaises(ValueError):binaryVector(value,2)

    def test_graph_exhaustive(self):
        n=6
        for edges in (set(),set(combinations(range(n),2)),{(0,1),(0,2),(1,2),(2,3),(3,4)}):
            graph=GraphSource(n,edges);q=graph.canonical();best=0;cliqueSize=0
            for x in product((0,1),repeat=n):
                selected=[i for i,b in enumerate(x) if b]
                violations=sum((i,j) not in edges for i,j in combinations(selected,2))
                expected=-len(selected)+2*violations
                self.assertEqual(q.score(x),expected)
                self.assertEqual(graph.score(x),expected)
                self.assertEqual(graph.details(x)['valid_clique_size'],None if violations else len(selected))
                if not violations:cliqueSize=max(cliqueSize,len(selected))
                best=min(best,expected)
            self.assertEqual(best,-cliqueSize)


class RunnerTests(unittest.TestCase):
    def test_registry_and_all_configuration(self):
        from qubo_solvers import create_bqm_solver
        self.assertTrue(all(create_bqm_solver(name) is not None for name in SOLVERS))
        self.assertEqual(len(GPU_SOLVERS),23)
        self.assertEqual(len(SOLVERS),28)
        full=checkConfig(readJson(ROOT/'benchmark_configs/full.json'))
        self.assertEqual({s['solver'] for s in full['solvers']},set(GPU_SOLVERS))
        full=checkConfig(readJson(ROOT/'benchmark_configs/full_all.json'))
        self.assertEqual({s['solver'] for s in full['solvers']},set(SOLVERS))
        for spec in full['solvers']:
            config=effectiveConfig(full,spec)
            if spec['solver'] in CLASSICAL_SOLVERS:
                self.assertEqual(config['device'],'cpu');self.assertEqual(config['precision'],'float64')

    def test_classical_adapters_on_exact_tiny_objective(self):
        # Zero-field planar triangle, suitable for every classical backend.
        q=parseMatrix(b'3 6\n1 1 -2\n2 2 -2\n3 3 -2\n1 2 1\n1 3 1\n2 3 1').canonical()
        for name in CLASSICAL_SOLVERS:
            with self.subTest(solver=name):
                parameters={}
                if name=='lib_tree_decomposition_sampler':parameters['marginals']=False
                adapter=Adapter(q,name,0,'cpu','float64',parameters)
                json.dumps(adapter.describe(),allow_nan=False)
                sample,energy=adapter.solve()
                self.assertEqual(q.score(sample),energy)
                if name in ('lib_planar_graph','lib_tree_decomposition_solver'):self.assertEqual(energy,-2)
                self.assertEqual('seed' in adapter.parameters,name not in ('lib_planar_graph','lib_tree_decomposition_solver'))

    def test_unsupported_structure_and_native_dependency(self):
        q=parseMatrix(b'3 1\n1 1 -1').canonical()
        with self.assertRaises(UnsupportedProblem):Adapter(q,'lib_planar_graph',0,'cpu','float64',{})
        dense=GraphSource(27,set()).canonical()
        for name in ('lib_tree_decomposition_solver','lib_tree_decomposition_sampler'):
            with self.assertRaisesRegex(UnsupportedProblem,'minimum degree 26'):
                Adapter(dense,name,0,'cpu','float64',{})
        with self.assertRaises(ValueError):
            Adapter(q,'lib_simulated_bifurcation',0,'cpu','float32',{},dict(libraryPath='retired.so'))
        with self.assertRaises(ValueError):Adapter(q,'random',0,'cpu','float64',{})
        with self.assertRaises(ValueError):Adapter(q,'lib_planar_graph',0,'cuda:0','float64',{})
        with self.assertRaises(ValueError):Adapter(q,'lib_planar_graph',0,'cpu','float32',{})
        with self.assertRaises(ValueError):Adapter(q,'lib_random_search',0,'cpu','float64',{'misspelled_option':1})
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(BackendUnavailable):Adapter(q,'lib_random_search',0,'cuda:0','float32',{})

    def test_mixed_run_records_supported_and_structural_rejections(self):
        config=readJson(ROOT/'benchmark_configs/full_all.json')
        config.update(instances=['gka1e'],seeds=[0],device='cpu',setup_limit_s=30,time_limit_s=10)
        config['solvers']=[s for s in config['solvers'] if s['solver'] in
                           ('lib_random_search','lib_planar_graph','lib_categorical')]
        for spec in config['solvers']:
            spec['device']='cpu'
            if spec['solver']=='lib_random_search':spec['parameters'].update(steps=2,runs=1)
        with tempfile.TemporaryDirectory() as tmp:
            report=run(config,output=Path(tmp)/'results')
            self.assertEqual((report['completed'],report['unsupported'],report['unavailable']),(1,2,0))
            statuses={r['solver_id']:r['status'] for r in report['per_instance']}
            self.assertEqual(statuses,dict(lib_random_search='completed',lib_planar_graph='unsupported',lib_categorical='unsupported'))

    def test_hard_deadlines(self):
        for ready in (False,True):
            start=time.perf_counter()
            result=supervise('unused',{'ready':ready},0,dict(setup_limit_s=3,time_limit_s=.15),stuckWorker)
            self.assertEqual(result['status'],'timeout')
            self.assertEqual(result['timeout_phase'],'solve' if ready else 'setup')
            self.assertLess(time.perf_counter()-start,8)

    def test_signed_comparisons(self):
        entry=dict(normalized_reference_objective_min=-10,reference_status='published_proven_optimum')
        result=comparison(-12,entry)
        self.assertEqual(result['delta'],-2);self.assertEqual(result['relative_delta_percent'],-20)
        self.assertEqual(result['alarm'],'below_published_proven_optimum')
        self.assertIsNone(result['time_to_target_seconds'])
        entry['reference_status']='best_published';self.assertEqual(comparison(-12,entry)['alarm'],'candidate_improvement')

    def test_full_configuration_and_all_adapters(self):
        config=checkConfig(readJson(ROOT/'benchmark_configs'/'full.json'))
        self.assertEqual(config['seeds'],list(range(20)))
        q=parseMatrix(b'4 5\n1 1 -2\n2 2 -1\n3 3 1\n4 4 -1\n1 3 2').canonical()
        for spec in config['solvers']:
            with self.subTest(solver=spec['solver']):
                parameters=dict(spec['parameters'],steps=2,runs=1)
                adapter=Adapter(q,spec['solver'],0,'cpu','float32',parameters)
                json.dumps(adapter.describe(),allow_nan=False)
                x,energy=adapter.solve();self.assertEqual(len(x),4)
                self.assertAlmostEqual(q.score(x),energy)
        with self.assertRaises(ValueError):Adapter(q,'lib_categorical',0,'cpu','float32',{})


class PreparedDataTests(unittest.TestCase):
    def test_all_37_independent_source_scores_and_witnesses(self):
        with patch('qubo_benchmark.storage.urlopen',side_effect=AssertionError('Tests must be offline')):
            report=validate()
        failures=[r for r in report['instances'] if r['status']!='passed']
        self.assertEqual(failures,[])
        self.assertEqual(report['passed'],37)
        self.assertEqual(sum(r['reference_vector_evaluated'] for r in report['instances']),4)
        self.assertEqual({(g['n'],g['density']):g['ready'] for g in report['groups']},GROUPS)

    def test_missing_data_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            writeJson(Path(tmp)/'catalog.json',loadCatalog())
            report=validate(tmp)
            self.assertEqual(report['passed'],0)
            self.assertEqual(len([r for r in report['instances'] if r['status']=='error']),37)

    def test_catalog_counts(self):
        entries=loadCatalog()['instances']
        self.assertEqual(len(set(e['instance_id'] for e in entries)),37)
        self.assertEqual(sum(e['reference_solution_vector_url'] is not None for e in entries),4)


if __name__=='__main__':unittest.main()

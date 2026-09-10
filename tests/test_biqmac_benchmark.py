"""Independent MaxCut encoding and benchmark winner semantics."""

import importlib.util
from itertools import product
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('biqmac_benchmark', Path(__file__).parents[1] / 'tools/benchmark_biqmac.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BiqMacBenchmarkTest(unittest.TestCase):
    def test_signed_parallel_edges_and_loops_match_every_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph'
            path.write_text('# signed edges and ignored loop\n3 5\n1 2 3\n2 1 -1\n2 3 -4\n1 3 5\n2 2 100\n')
            q, heads, tails, weights = benchmark.readGraph(path)
            for bits in product((0, 1), repeat=3):
                cut = sum(w for h, t, w in zip(heads, tails, weights) if bits[h] != bits[t])
                self.assertEqual(q.energy(bits), -cut)
            path.write_text('3 2\n1 2 1\n')
            with self.assertRaises(ValueError):
                benchmark.readGraph(path)
            path.write_text('3 1\n0 2 1\n')
            with self.assertRaises(ValueError):
                benchmark.readGraph(path)

    def test_quality_ties_use_median_time_and_errors_do_not_win(self):
        def result(cut, seconds):
            return dict(status='ok', trials=[dict(cut=cut, seconds=t, sample='010', sample_sha256='hash') for t in seconds])
        record = dict(instance=dict(name='tiny', family='rudy', vertices=3, edges=2, reference_cut=10),
            runs=2, steps=1000, seed=1, device='cpu', solvers={
                'SBM': result(9, [3., 1., 2.]), 'SVL': result(10, [4., 3., 2.]), 'TRF': result(10, [1., 2., 3.])})
        row = benchmark.resultRow(record)
        self.assertEqual(row['quality_winners'], 'SVL|TRF')
        self.assertEqual(row['winner'], 'TRF')
        self.assertEqual(row['sbm_median_seconds'], 2.)
        with tempfile.TemporaryDirectory() as directory:
            overall = benchmark.publish(Path(directory), {'tiny': record}, [(2, 1000)], 1)
            self.assertEqual(overall[0]['solver'], 'TRF')
            self.assertEqual(overall[0]['quality_win_points'], .5)
        record['solvers']['SVL'] = dict(status='error', error='failed', trials=[])
        row = benchmark.resultRow(record)
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['winner'], '')

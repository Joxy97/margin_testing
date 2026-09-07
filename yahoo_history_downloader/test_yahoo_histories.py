import tempfile
import unittest
from pathlib import Path
from build_yahoo_histories import assignments, digest, read_portfolio


class Tests(unittest.TestCase):
    def test_original_files(self):
        for name, n in [('portfolio.csv', 200), ('portfolio(1).csv', 1000)]:
            path = Path(__file__).parent / name
            if not path.exists():
                path = Path(__file__).parent / 'upload' / name
            before = digest(path)
            rows = read_portfolio(path, n)
            pairs = assignments(rows, [{'symbol': str(i)} for i in range(n)])
            self.assertEqual(len(pairs), n)
            self.assertEqual(digest(path), before)
            self.assertEqual({r['ticker'] for r, _ in pairs}, {r['ticker'] for r in rows})

    def test_numeric_weight_sort_and_ties(self):
        rows = [{'ticker': '10', 'weight': '0.1'}, {'ticker': '2', 'weight': '0.1'},
                {'ticker': '0', 'weight': '0.9'}]
        pairs = assignments(rows, [{'symbol': s} for s in ['BIG', 'MID', 'SMALL']])
        self.assertEqual([(r['ticker'], s['symbol']) for r, s in pairs],
                         [('0', 'BIG'), ('2', 'MID'), ('10', 'SMALL')])

    def test_wrong_count(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'p.csv'
            p.write_text('client_id,ticker,weight\na,1,0.5\n')
            with self.assertRaises(ValueError):
                read_portfolio(p, 2)


if __name__ == '__main__':
    unittest.main()

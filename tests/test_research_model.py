"""Characterize standalone model semantics before sharing numerical kernels."""

import unittest
import numpy


class ResearchModelTest(unittest.TestCase):
    def test_standalone_pca_preserves_the_supplied_standardization(self):
        from market_to_qubo import exponentially_weighted_pca
        values = numpy.array([[1., 0.], [0., 2.], [-1., 0.], [0., -2.]])
        result = exponentially_weighted_pca(values, numpy.array([[0., 4.]]), 1, 1., 4)
        # Population covariance is diag(1/2, 2); there is no second standardization.
        numpy.testing.assert_allclose(result.eigenvalues, [2.])
        numpy.testing.assert_allclose(result.explained, [.8])
        numpy.testing.assert_allclose(result.backtest_factors, [[4.]])

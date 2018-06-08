import unittest

import numpy as np
import numpy.testing as npt

import matlab_alikes


class TestQuantiles(unittest.TestCase):
    def test_is_consistent_with_original(self):
        values = np.arange(11)
        quantiles = np.arange(0, 1.1, .1)
        quantile_values = matlab_alikes.quantile(values, quantiles)
        expected = np.array([0, .6, 1.7, 2.8, 3.9, 5., 6.1, 7.2, 8.3, 9.4, 10.])
        npt.assert_almost_equal(quantile_values, expected)


class TestNQuantiles(unittest.TestCase):
    def test_returns_correct_number_of_quantiles(self):
        values = np.arange(11)
        quantile_values = matlab_alikes.n_quantiles(values, 10)
        self.assertEqual(10, len(quantile_values))

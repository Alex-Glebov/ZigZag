import numpy as np
import pandas as pd

from zigzag.core import (
  identify_initial_pivot,
  peak_valley_pivots,
  max_drawdown,
  compute_segment_returns,
  compute_performance,
  pivots_to_modes
  )
from zigzag import   PEAK,  VALLEY,  SIDEMOVE

from unittest import TestCase
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy import double, array
from numpy import ndarray 

#from zigzag import PEAK, VALLEY
#from zigzag.core import *
#
# zigzag.identify_initial_pivot(data, 1.1,  0.9) has absolute values instead 
# zigzag.identify_initial_pivot(data, 0.1, -0.1)  
# as threshold 0.1 and -0.1 should be increased by 1.0 as it is used to measure relations. xn/x1 
# previous version has inner operation adding 1 inside this function. nowadays it is done once at
# caller function peak_valley_pivots_detailed where this increased value also in use 
#    
#


class TestIdentifyInitialPivot(TestCase):
    def test_strictly_increasing(self):
        data = np.linspace(1, 2., 10)
        self.assertEqual(identify_initial_pivot(data, 1.1, 0.9),
                         VALLEY)

    def test_increasing_kinked(self):
        data = np.array([1.0, 0.99, 1.1])
        self.assertEqual(identify_initial_pivot(data, 1.1, 0.9),
                         PEAK)

    def test_strictly_increasing_under_threshold(self):
        data = np.linspace(1, 1.01, 10)
        self.assertEqual(identify_initial_pivot(data, 1.1, 0.9),
                         VALLEY)

    def test_increasing_under_threshold_kinked(self):
        data = np.array([1.0, 0.99, 1.02])
        self.assertEqual(identify_initial_pivot(data, 1.1, 0.9),
                         VALLEY)

    def test_strictly_decreasing(self):
        data = np.linspace(1, 0.5, 10)
        self.assertEqual(identify_initial_pivot(data, 1.1, 0.9),
                         PEAK)

    def test_decreasing_kinked(self):
        data = np.array([1.0, 1.01, 0.9])
        self.assertEqual(identify_initial_pivot(data,1.1, 0.9),
                         VALLEY)

    def test_strictly_decreasing_under_threshold(self):
        data = np.linspace(1, 0.99, 10)
        self.assertEqual(identify_initial_pivot(data, 1.1, 0.9),
                         PEAK)

    def test_decreasing_under_threshold_kinked(self):
        data = np.array([1.0, 1.01, 0.99])

        self.assertEqual(identify_initial_pivot(data, 1.1, 0.9),
                         PEAK)


class TestPeakValleyPivots(TestCase):
    def test_guard_against_common_threshold_value_mistake(self):
        data = np.array([1.0, 2.0, 3.0])
        self.assertRaises(ValueError, peak_valley_pivots,
                          data, 0.1, 0.1)

    def test_strictly_increasing(self):
        data = np.linspace(1, 10, 10)
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.zeros_like(data)
        expected_result[0], expected_result[-1] = VALLEY, PEAK

        assert_array_equal(result, expected_result)

    def test_strictly_increasing_but_less_than_threshold(self):
        data = np.linspace(1.0, 1.05, 10)
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.zeros_like(data)
        expected_result[0], expected_result[-1] = VALLEY, PEAK

        self.assertTrue(data[0] < data[len(data)-1])
        assert_array_equal(result, expected_result)

    def test_strictly_decreasing(self):
        data = np.linspace(10, 0, 10)
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.zeros_like(data)
        expected_result[0], expected_result[-1] = PEAK, VALLEY

        assert_array_equal(result, expected_result)

    def test_strictly_decreasing_but_less_than_threshold(self):
        data = np.linspace(1.05, 1.0, 10)
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.zeros_like(data)
        expected_result[0], expected_result[-1] = PEAK, VALLEY

        assert_array_equal(result, expected_result)

    def test_single_peaked(self):
        data = np.array([1.0, 1.2, 1.05])
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.array([VALLEY, PEAK, VALLEY])

        assert_array_equal(result, expected_result)

    def test_single_valleyed(self):
        data = np.array([1.0, 0.9, 1.2])
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.array([PEAK, VALLEY, PEAK])

        assert_array_equal(result, expected_result)

    def test_increasing_kinked(self):
        data = np.array([1.0, 0.99, 1.1])
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.array([PEAK, VALLEY, PEAK])

        assert_array_equal(result, expected_result)

    def test_decreasing_kinked(self):
        data = np.array([1.0, 1.01, 0.9])
        result = peak_valley_pivots(data, 0.1, -0.1)
        expected_result = np.array([VALLEY, PEAK, VALLEY])

        assert_array_equal(result, expected_result)


class TestSegmentReturn(TestCase):
    def test_strictly_increasing(self):
        data = np.linspace(1.0, 100.0, 10)
        pivots = peak_valley_pivots(data, 0.1, -0.1)
        assert_array_almost_equal(compute_segment_returns(data, pivots),
                                  np.array([99.0]))

    def test_strictly_decreasing(self):
        data = np.linspace(100.0, 1.0, 10)
        pivots = peak_valley_pivots(data, 0.1, -0.1)
        assert_array_almost_equal(compute_segment_returns(data, pivots),
                                  np.array([-0.99]))

    def test_rise_fall_rise(self):
        data = np.array([1.0, 1.05, 1.1, 1.0, 0.9, 1.5])
        pivots = peak_valley_pivots(data, 0.1, -0.1)
        assert_array_almost_equal(compute_segment_returns(data, pivots),
                                  np.array([0.1, -0.181818, 0.6666666]))


class TestMaxDrawdown(TestCase):
    def test_strictly_increasing(self):
        data = np.linspace(1.0, 100.0, 10)
        self.assertEqual(max_drawdown(data), 0.0)

    def test_strictly_decreasing(self):
        data = np.linspace(100.0, 1.0, 10)
        self.assertEqual(max_drawdown(data), 0.99)

    def test_rise_fall_rise_drawdown(self):
        data = np.array([1.0, 1.05, 1.1, 1.0, 0.9, 1.5])
        self.assertAlmostEqual(max_drawdown(data), 0.18181818181818188)

class TestComputerPerformance(TestCase):
    def test_strictly_increasing(self):
        data = np.linspace(1.0, 100.0, 10)
        pivots = peak_valley_pivots(data, 0.1, -0.1)
        result = compute_performance(data,pivots)
        assert_array_almost_equal(
          result[0],      np.zeros(10, dtype=double)
          )
        assert_array_almost_equal(
          result[1],      
          [99.,  7.33333333,  3.34782609,  1.94117647,  1.22222222,  
           0.78571429,  0.49253731,  0.28205128,  0.12359551,  0. ]
          )
        assert_array_almost_equal(
          result[2],        np.linspace(9, 0, 10)
          )

    def test_strictly_decreasing(self):
        data = np.linspace(100.0, 1.0, 10)
        pivots = peak_valley_pivots(data, 0.1, -0.1)
        result = compute_performance(data,pivots)
        assert_array_almost_equal(
          result[1],      
          [-1., -1., -1., -1., -1., -1., -1., -1., -1., 0. ]
          )
        assert_array_almost_equal(
          result[0],      
          [-0.99, -0.98876404, -0.98717949, -0.98507463, -0.98214286, -0.97777778, -0.97058824, -0.95652174, -0.91666667,  0.]
          )
        assert_array_almost_equal(
          result[2],        np.linspace(9, 0, 10)
          )
 
    def test_rise_fall_rise(self):
        data = np.array([1.0, 1.05, 1.1, 1.0, 0.9, 1.5])
        pivots = peak_valley_pivots(data, 0.1, -0.1)
        result = compute_performance(data,pivots)
        assert_array_almost_equal(
          result[0],      
<<<<<<< HEAD
          [-0.1     , -0.142857,   -0.18181818, -0.1,         0.,          0.      ]
=======
          [0. ,         0.,         -0.18181818, -0.1,         0.,          0.      ]
>>>>>>> 7c9b47e2940b13544dbfeaf3d28f1a129522033f
          )
        assert_array_almost_equal(
          result[1],      
          [0.1,         0.04761905, -1.,         -1.,          0.66666667,  0.      ]
          )
        assert_array_almost_equal(
          result[2],
          [2,1, 2, 1, 1, 0 ]
          )

class TestPivotsToModes(TestCase):
    def test_pivots_to_modes(self):
        data = np.array([1, 0, 0, 0, -1, 0, 0, 1, -1, 0, 1])
        result = pivots_to_modes(data)
        expected_result = np.array([1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1])

        assert_array_equal(result, expected_result)


def test_peak_valley_pivots_pandas_compat():
    df = pd.DataFrame({'X': np.array([1, 2, 3, 4])})
    got = peak_valley_pivots(df.X, 0.2, -0.2)
    assert (got == np.array([-1, 0, 0, 1])).all()

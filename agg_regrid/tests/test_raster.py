# (C) British Crown Copyright 2015 - 2018, Met Office
#
# This file is part of agg-regrid.
#
# agg-regrid is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# agg-regrid is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with agg-regrid.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for the `agg_regrid._agg.raster` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, xrange, zip)

import numpy as np
from numpy.testing import assert_array_equal
import unittest

from agg_regrid._agg import raster


class TestDataType(unittest.TestCase):
    def setUp(self):
        self.emsg = 'Buffer dtype mismatch'
        self.xi = np.arange(4, dtype=np.float64).reshape(2, 2)
        self.yi = np.arange(4, dtype=np.float64).reshape(2, 2)
        ny, nx = 10, 8
        self.weights = np.zeros((ny, nx), dtype=np.uint8)

    def test_weights_bad_dtype(self):
        weights = np.zeros((5, 5))
        with self.assertRaisesRegexp(ValueError, self.emsg):
            raster(weights, self.xi, self.yi)

    def test_xi_bad_dtype(self):
        xi = np.arange(4).reshape(2, 2)
        with self.assertRaisesRegexp(ValueError, self.emsg):
            raster(self.weights, xi, self.yi)

    def test_yi_bad_dtype(self):
        yi = np.arange(4).reshape(2, 2)
        with self.assertRaisesRegexp(ValueError, self.emsg):
            raster(self.weights, self.xi, yi)


class TestShape(unittest.TestCase):
    def setUp(self):
        self.emsg = 'Buffer has wrong number of dimensions'
        self.xi = np.arange(4, dtype=np.float64).reshape(2, 2)
        self.yi = np.arange(4, dtype=np.float64).reshape(2, 2)
        ny, nx = 10, 8
        self.weights = np.zeros((ny, nx), dtype=np.uint8)

    def test_weights_bad_shape(self):
        weights = self.weights.flatten()
        with self.assertRaisesRegexp(ValueError, self.emsg):
            raster(weights, self.xi, self.yi)

    def test_xi_bad_shape(self):
        xi = self.xi.flatten()
        with self.assertRaisesRegexp(ValueError, self.emsg):
            raster(self.weights, xi, self.yi)

    def test_yi_bad_shape(self):
        yi = self.yi.flatten()
        with self.assertRaisesRegexp(ValueError, self.emsg):
            raster(self.weights, self.xi, yi)


class TestWeightsCoverage(unittest.TestCase):
    def setUp(self):
        self.ny, self.nx = self.shape = 6, 8
        self.weights = np.zeros(self.shape, dtype=np.uint8)

    def test_top_left_cell(self):
        xi = np.array([[0, 1],
                       [0, 1]], dtype=np.float64)
        yi = np.array([[0, 0],
                       [1, 1]], dtype=np.float64)
        raster(self.weights, xi, yi)
        self.assertEqual(self.weights.sum(), 255)
        self.assertEqual(self.weights[0, 0], 255)

    def test_top_right_cell(self):
        xi = np.array([[self.nx - 1, self.nx],
                       [self.nx - 1, self.nx]], dtype=np.float64)
        yi = np.array([[0, 0],
                       [1, 1]], dtype=np.float64)
        raster(self.weights, xi, yi)
        self.assertEqual(self.weights.sum(), 255)
        self.assertEqual(self.weights[0, self.nx - 1], 255)

    def test_bottom_left_cell(self):
        xi = np.array([[0, 1],
                       [0, 1]], dtype=np.float64)
        yi = np.array([[self.ny - 1, self.ny - 1],
                       [self.ny, self.ny]], dtype=np.float64)
        raster(self.weights, xi, yi)
        self.assertEqual(self.weights.sum(), 255)
        self.assertEqual(self.weights[self.ny - 1, 0], 255)

    def test_bottom_right_cell(self):
        xi = np.array([[self.nx - 1, self.nx],
                       [self.nx - 1, self.nx]], dtype=np.float64)
        yi = np.array([[self.ny - 1, self.ny - 1],
                       [self.ny, self.ny]], dtype=np.float64)
        raster(self.weights, xi, yi)
        self.assertEqual(self.weights.sum(), 255)
        self.assertEqual(self.weights[self.ny - 1, self.nx - 1], 255)

    def test_full_coverage(self):
        xi = np.array([[0, self.nx],
                       [0, self.nx]], dtype=np.float64)
        yi = np.array([[0, 0],
                       [self.ny, self.ny]], dtype=np.float64)
        raster(self.weights, xi, yi)
        expected = np.ones((self.ny, self.nx), dtype=np.float64) * 255
        assert_array_equal(self.weights, expected)

    def test_inset_by_one_cell(self):
        xi = np.array([[1, self.nx - 1],
                       [1, self.nx - 1]], dtype=np.float64)
        yi = np.array([[1, 1],
                       [self.ny - 1, self.ny - 1]], dtype=np.float64)
        raster(self.weights, xi, yi)
        expected = np.ones((self.ny, self.nx), dtype=np.float64) * 255
        expected[0, :] = expected[-1, :] = 0
        expected[:, 0] = expected[:, -1] = 0
        assert_array_equal(self.weights, expected)

    def test_inset_by_half_cell(self):
        xi = np.array([[0.5, self.nx - 0.5],
                       [0.5, self.nx - 0.5]], dtype=np.float64)
        yi = np.array([[0.5, 0.5],
                       [self.ny - 0.5, self.ny - 0.5]], dtype=np.float64)
        raster(self.weights, xi, yi)
        expected = np.ones((self.ny, self.nx), dtype=np.float64) * 255
        half, quarter = 255 // 2, 255 // 4
        expected[0, :] = expected[-1, :] = half
        expected[:, 0] = expected[:, -1] = half
        expected[0, 0] = expected[0, -1] = quarter
        expected[-1, 0] = expected[-1, -1] = quarter
        assert_array_equal(self.weights, expected)

    def test_rotated(self):
        xi = np.array([[1.5, 4.5],
                       [3.5, 6.5]], dtype=np.float64)
        yi = np.array([[3.5, 0.5],
                       [5.5, 2.5]], dtype=np.float64)
        raster(self.weights, xi, yi)
        expected = np.zeros((self.shape), dtype=np.uint8)
        full, half, quarter = 255, 255 // 2, 255 // 4
        # corners ...
        expected[3, 1] = expected[0, 4] = quarter
        expected[5, 3] = expected[2, 6] = quarter
        # edges ...
        expected[2, 2] = expected[1, 3] = half
        expected[4, 2] = half
        expected[1, 5] = half
        expected[4, 4] = expected[3, 5] = half
        # inner ...
        expected[1, 4] = full
        expected[2, 3] = expected[2, 4] = expected[2, 5] = full
        expected[3, 2] = expected[3, 3] = expected[3, 4] = full
        expected[4, 3] = full
        assert_array_equal(self.weights, expected)


if __name__ == '__main__':
    unittest.main()

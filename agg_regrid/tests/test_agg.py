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
"""Unit tests for the `agg_regrid.agg` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, xrange, zip)  # noqa
from six import assertRaisesRegex

import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import unittest

from agg_regrid import agg


class TestDimensionality(unittest.TestCase):
    def setUp(self):
        nz, ny, nx = shape = 5, 10, 20
        self.data = np.empty(shape)
        self.sx_points = np.arange(nx)
        self.sx_bounds = np.arange(nx + 1)
        self.sy_points = np.arange(ny)
        self.sy_bounds = np.arange(ny + 1)
        self.sy_dim, self.sx_dim = 1, 2
        self.gx_bounds = np.empty((5, 5))
        self.gy_bounds = np.empty((5, 5))

    def test_src_x_points(self):
        sx_points = self.sx_points.reshape(1, -1)
        emsg = 'Expected 1d src x-coordinate points'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_y_points(self):
        sy_points = self.sy_points.reshape(1, -1)
        emsg = 'Expected 1d src y-coordinate points'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_x_bounds(self):
        sx_bounds = self.sx_bounds.reshape(1, -1)
        emsg = 'Expected 1d contiguous src x-coordinate bounds'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_y_bounds(self):
        sy_bounds = self.sy_bounds.reshape(1, -1)
        emsg = 'Expected 1d contiguous src y-coordinate bounds'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_data(self):
        data = self.data.flatten()
        emsg = 'Expected at least 2d src data'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_x_dim(self):
        # Positive index beyond last dimension.
        sx_dim = self.data.ndim
        emsg = 'Invalid src x-coordinate dimension'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_y_dim(self):
        # Negative index before first dimension.
        sy_dim = -(self.data.ndim + 1)
        emsg = 'Invalid src y-coordinate dimension'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_grid_x_bounds(self):
        gx_bounds = self.gx_bounds.flatten()
        emsg = 'Expected 2d contiguous grid x-coordinate bounds'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                gx_bounds, self.gy_bounds)

    def test_grid_y_bounds(self):
        gy_bounds = self.gy_bounds.flatten()
        emsg = 'Expected 2d contiguous grid y-coordinate bounds'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, gy_bounds)


class TestShape(unittest.TestCase):
    def setUp(self):
        nz, self.ny, self.nx = self.shape = 5, 10, 20
        self.data = np.empty(self.shape)
        self.sx_points = np.arange(self.nx)
        self.sx_bounds = np.arange(self.nx + 1)
        self.sy_points = np.arange(self.ny)
        self.sy_bounds = np.arange(self.ny + 1)
        self.sy_dim, self.sx_dim = 1, 2
        self.gx_bounds = np.empty((5, 5))
        self.gy_bounds = np.empty((5, 5))

    def test_src_x_points_and_bounds(self):
        sx_points = np.arange(self.nx - 1)
        emsg = 'Invalid number of src x-coordinate bounds'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_y_points_and_bounds(self):
        sy_points = np.arange(self.ny - 1)
        emsg = 'Invalid number of src y-coordinate bounds'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_x_points_and_data(self):
        shape = list(self.shape)
        shape[self.sx_dim] = self.nx + 1
        data = np.empty(shape)
        emsg = 'src x-coordinate points .* do not align'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_src_y_points_and_data(self):
        shape = list(self.shape)
        shape[self.sy_dim] = self.ny - 1
        data = np.empty(shape)
        emsg = 'src y-coordinate points .* do not align'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_grid_x_bounds_and_y_bounds(self):
        shape = np.array(self.gy_bounds.shape) + 1
        gy_bounds = np.empty(shape)
        emsg = 'Misaligned grid'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, gy_bounds)


class TestRegridSingleLevel(unittest.TestCase):
    def setUp(self):
        # Source has points shape (y:6, x:8)
        self.ny, self.nx = self.shape = (6, 8)
        size = np.prod(self.shape)
        self.data = np.arange(size).reshape(self.shape)
        self.sx_points = np.arange(1, self.nx + 1) - 0.5
        self.sx_bounds = np.arange(self.nx + 1)
        self.sy_points = np.arange(1, self.ny + 1) - 0.5
        self.sy_bounds = np.arange(self.ny + 1)
        self.sy_dim, self.sx_dim = 0, 1
        # Target grid has points shape (y:2, x:2)
        gx_bounds = np.array([1.5, 4.0, 6.5], dtype=np.float64)
        gy_bounds = np.array([1.5, 3.0, 4.5], dtype=np.float64)
        self.gx_bounds, self.gy_bounds = np.meshgrid(gx_bounds, gy_bounds)

    def _expected(self, transpose=False):
        data = self.data
        if transpose:
            data = self.data.T
        # Expected raster weights per target grid cell.
        # This is the (fractional) source cell contribution
        # to each target cell (out of 255)
        weights = np.array([[[63, 127, 127],   # top left hand cell (tlhc)
                             [127, 255, 255]],
                            [[127, 127, 63],   # top right hand cell (trhc)
                             [255, 255, 127]],
                            [[127, 255, 255],  # bottom left hand cell (blhc)
                             [63, 127, 127]],
                            [[255, 255, 127],  # bottom right hand cell (brhc)
                             [127, 127, 63]]], dtype=np.uint8)
        weights = weights / 255
        # Expected source points per target grid cell.
        tmp = data[1:-1, 1:-1]
        shape = (-1, 2, 3)
        cells = [tmp[slice(0, 2), slice(0, 3)].reshape(shape),        # tlhc
                 tmp[slice(0, 2), slice(3, None)].reshape(shape),     # trhc
                 tmp[slice(2, None), slice(0, 3)].reshape(shape),     # blhc
                 tmp[slice(2, None), slice(3, None)].reshape(shape)]  # brhc
        cells = ma.vstack(cells)
        # Expected fractional weighted result.
        num = (cells * weights).sum(axis=(1, 2))
        dom = weights.sum(axis=(1, 2))
        expected = num / dom
        expected = ma.asarray(expected.reshape(2, 2))
        if transpose:
            expected = expected.T
        return expected

    def test_regrid_ok(self):
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        assert_array_equal(result, self._expected())

    def test_regrid_ok_transpose(self):
        self.data = self.data.T
        sy_dim, sx_dim = self.sx_dim, self.sy_dim
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     sx_dim, sy_dim,
                     self.gx_bounds, self.gy_bounds)
        expected = self._expected(transpose=True)
        assert_array_equal(result, expected)

    def test_regrid_ok_src_masked(self):
        self.data = ma.asarray(self.data)
        self.data[2, 3] = ma.masked  # tlhc 1x masked of 6x src cells
        self.data[2, 4] = ma.masked  # trhc 2x masked of 6x src cells
        self.data[2, 5] = ma.masked
        self.data[3, 2] = ma.masked  # blhc 3x masked of 6x src cells
        self.data[3, 3] = ma.masked
        self.data[4, 3] = ma.masked
        self.data[3, 4] = ma.masked  # brhc 4x masked of 6x src cells
        self.data[3, 5] = ma.masked
        self.data[4, 4] = ma.masked
        self.data[4, 5] = ma.masked
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        assert_array_equal(result, self._expected())

    def test_regrid_ok_src_x_points_cast(self):
        self.sx_points = np.asarray(self.sx_points, dtype=np.float32)
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        assert_array_equal(result, self._expected())

    def test_regrid_ok_src_y_points_cast(self):
        self.sy_points = np.asarray(self.sy_points, dtype=np.float32)
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        assert_array_equal(result, self._expected())

    def test_regrid_ok_negative_sx_dim(self):
        self.sx_dim = -(self.data.ndim - self.sx_dim)
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        assert_array_equal(result, self._expected())

    def test_regrid_ok_negative_sy_dim(self):
        self.sy_dim = -(self.data.ndim - self.sy_dim)
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        assert_array_equal(result, self._expected())

    def test_regrid_ok_grid_tlhc_out_of_bounds(self):
        self.gx_bounds[0, 0] = self.gx_bounds[0, 0] * -1
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        expected = self._expected()
        expected[0, 0] = ma.masked
        assert_array_equal(result, self._expected())

    def test_regrid_ok_grid_trhc_out_of_bounds(self):
        self.gx_bounds[0, -1] = self.gx_bounds[0, -1] * 2
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        expected = self._expected()
        expected[0, 1] = ma.masked
        assert_array_equal(result, self._expected())

    def test_regrid_ok_grid_blhc_out_of_bounds(self):
        self.gy_bounds[-1, 0] = self.gy_bounds[-1, 0] * 2
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        expected = self._expected()
        expected[1, 0] = ma.masked
        assert_array_equal(result, self._expected())

    def test_regrid_ok_grid_brhc_out_of_bounds(self):
        self.gy_bounds[-1, -1] = self.gy_bounds[-1, -1] * 2
        result = agg(self.data, self.sx_points, self.sx_bounds,
                     self.sy_points, self.sy_bounds,
                     self.sx_dim, self.sy_dim,
                     self.gx_bounds, self.gy_bounds)
        expected = self._expected()
        expected[1, 1] = ma.masked
        assert_array_equal(result, self._expected())

    def test_regrid_irregular_src_x_points(self):
        self.sx_points[-1] = self.sx_points[-1] * 1.1
        emsg = 'Expected src x-coordinate points to be regular'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)

    def test_regrid_irregular_src_y_points(self):
        self.sy_points[0] = self.sy_points[0] * 1.01
        emsg = 'Expected src y-coordinate points to be regular'
        with assertRaisesRegex(self, ValueError, emsg):
            agg(self.data, self.sx_points, self.sx_bounds,
                self.sy_points, self.sy_bounds,
                self.sx_dim, self.sy_dim,
                self.gx_bounds, self.gy_bounds)


if __name__ == '__main__':
    unittest.main()

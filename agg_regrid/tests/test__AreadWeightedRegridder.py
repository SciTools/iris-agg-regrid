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
"""Unit tests for the `agg_regrid._AreaWeightedRegridder` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, xrange, zip)  # noqa
from six import assertRaisesRegex

import unittest

import iris
try:
    from unittest import mock
except ImportError:
    import mock

from agg_regrid import (_AreaWeightedRegridder as Regridder,
                        DEFAULT_BUFFER_DEPTH)


class Test(unittest.TestCase):
    def setUp(self):
        self.src_cube = mock.Mock(spec=iris.cube.Cube)
        self.tgt_cube = mock.Mock(spec=iris.cube.Cube)
        scs = mock.sentinel.srs
        self.sx = mock.Mock(coord_system=scs)
        self.sy = mock.Mock(coord_system=scs)
        tcs = mock.sentinel.tcs
        self.gx = mock.Mock(coord_system=tcs)
        self.gy = mock.Mock(coord_system=tcs)
        self.src_grid = (self.sx, self.sy)
        self.tgt_grid = (self.gx, self.gy)
        self.side_effect = (self.src_grid, self.tgt_grid)
        self.snapshot_grid = 'agg_regrid.snapshot_grid'

    def test_bad_src_grid_cube(self):
        emsg = 'source grid must be a cube'
        with assertRaisesRegex(self, TypeError, emsg):
            Regridder('dummy', self.tgt_cube)

    def test_bad_tgt_grid_cube(self):
        emsg = 'target grid must be a cube'
        with assertRaisesRegex(self, TypeError, emsg):
            Regridder(self.src_cube, 'dummy')

    def test_src_grid_tgt_grid(self):
        with mock.patch(self.snapshot_grid, side_effect=self.side_effect):
            regridder = Regridder(self.src_cube, self.tgt_cube)

        self.assertEqual(regridder._sx, self.sx)
        self.assertEqual(regridder._sy, self.sy)
        self.assertEqual(regridder._gx, self.gx)
        self.assertEqual(regridder._gy, self.gy)
        self.assertIsNone(regridder._gx_bounds)
        self.assertIsNone(regridder._gy_bounds)
        self.assertIsNone(regridder._sx_bounds)
        self.assertIsNone(regridder._sy_bounds)

    def test_snapshot_grid__no_sx_coord_system(self):
        sx = mock.Mock(coord_system=None)
        src_grid = (sx, self.sy)
        side_effect = (src_grid, self.tgt_grid)

        with mock.patch(self.snapshot_grid, side_effect=side_effect):
            emsg = 'source grid cube requires a native coordinate system'
            with assertRaisesRegex(self, ValueError, emsg):
                Regridder(self.src_cube, self.tgt_cube)

    def test_snapshot_grid__no_sy_coord_system(self):
        sy = mock.Mock(coord_system=None)
        src_grid = (self.sx, sy)
        side_effect = (src_grid, self.tgt_grid)

        with mock.patch(self.snapshot_grid, side_effect=side_effect):
            emsg = 'source grid cube requires a native coordinate system'
            with assertRaisesRegex(self, ValueError, emsg):
                Regridder(self.src_cube, self.tgt_cube)

    def test_snapshot_grid__no_gx_coord_system(self):
        gx = mock.Mock(coord_system=None)
        tgt_grid = (gx, self.gy)
        side_effect = (self.src_grid, tgt_grid)

        with mock.patch(self.snapshot_grid, side_effect=side_effect):
            emsg = 'target grid cube requires a native coordinate system'
            with assertRaisesRegex(self, ValueError, emsg):
                Regridder(self.src_cube, self.tgt_cube)

    def test_snapshot_grid__no_gy_coord_system(self):
        gy = mock.Mock(coord_system=None)
        tgt_grid = (self.gx, gy)
        side_effect = (self.src_grid, tgt_grid)

        with mock.patch(self.snapshot_grid, side_effect=side_effect):
            emsg = 'target grid cube requires a native coordinate system'
            with assertRaisesRegex(self, ValueError, emsg):
                Regridder(self.src_cube, self.tgt_cube)


class Test___call__(unittest.TestCase):
    def setUp(self):
        self.src_cube = mock.Mock(spec=iris.cube.Cube)
        self.tgt_cube = mock.Mock(spec=iris.cube.Cube)
        self.sx_dim = mock.sentinel.sx_dim
        self.sy_dim = mock.sentinel.sy_dim
        coord_dims = mock.Mock(side_effect=([self.sx_dim], [self.sy_dim]))
        self.metadata = dict(standard_name='air_pressure',
                             long_name='long_name', var_name='var_name',
                             units='Pa', attributes={}, cell_methods=())
        self.data = mock.sentinel.data
        self.cube = mock.Mock(spec=iris.cube.Cube, coord_dims=coord_dims,
                              metadata=self.metadata, dim_coords=(),
                              aux_coords=(), data=self.data)
        scrs = mock.sentinel.scrs
        self.sxp = mock.sentinel.sx_points
        self.sxb = mock.sentinel.sx_contiguous_bounds
        self.syp = mock.sentinel.sy_points
        self.syb = mock.sentinel.sy_contiguous_bounds
        self.sx = mock.Mock(coord_system=scrs,
                            points=self.sxp,
                            contiguous_bounds=mock.Mock(return_value=self.sxb))
        self.sy = mock.Mock(coord_system=scrs,
                            points=self.syp,
                            contiguous_bounds=mock.Mock(return_value=self.syb))
        tcrs = mock.sentinel.tcrs
        gx = mock.Mock(coord_system=tcrs)
        gy = mock.Mock(coord_system=tcrs)
        self.src_grid = (self.sx, self.sy)
        tgt_grid = (gx, gy)
        self.side_effect = (self.src_grid, tgt_grid)
        self.snapshot_grid = 'agg_regrid.snapshot_grid'
        self.get_xy_dim_coords = 'agg_regrid.get_xy_dim_coords'
        self.depth = DEFAULT_BUFFER_DEPTH

    def test_bad_src_cube(self):
        with mock.patch(self.snapshot_grid, side_effect=self.side_effect):
            emsg = 'source must be a cube'
            with assertRaisesRegex(self, TypeError, emsg):
                regridder = Regridder(self.src_cube, self.tgt_cube)
                regridder('dummy')

    def test_bad_src_cube__sx_different_grid(self):
        with mock.patch(self.snapshot_grid, side_effect=self.side_effect):
            return_value = (False, self.sy)
            with mock.patch(self.get_xy_dim_coords, return_value=return_value):
                emsg = 'source cube is not defined on the same source grid'
                with assertRaisesRegex(self, ValueError, emsg):
                    regridder = Regridder(self.src_cube, self.tgt_cube)
                    regridder(self.cube)

    def test_bad_src_cube__sy_different_grid(self):
        with mock.patch(self.snapshot_grid, side_effect=self.side_effect):
            return_value = (self.sx, False)
            with mock.patch(self.get_xy_dim_coords, return_value=return_value):
                emsg = 'source cube is not defined on the same source grid'
                with assertRaisesRegex(self, ValueError, emsg):
                    regridder = Regridder(self.src_cube, self.tgt_cube)
                    regridder(self.cube)

    def test_same_crs(self):
        side_effect = (self.src_grid, self.src_grid)
        with mock.patch(self.snapshot_grid, side_effect=side_effect):
            with mock.patch(self.get_xy_dim_coords,
                            return_value=self.src_grid):
                gxx, gyy = (mock.sentinel.gxx, mock.sentinel.gyy)
                with mock.patch('numpy.meshgrid', return_value=(gxx, gyy)):
                    data = 1
                    with mock.patch('agg_regrid.agg',
                                    return_value=data) as magg:
                        regridder = Regridder(self.src_cube, self.tgt_cube)
                        result = regridder(self.cube)

        self.assertEqual(regridder._sx_bounds, self.sxb)
        self.assertEqual(regridder._sy_bounds, self.syb)
        self.assertEqual(regridder._gx_bounds, gxx)
        self.assertEqual(regridder._gy_bounds, gyy)
        expected = [mock.call(self.data, self.sxp, self.sxb, self.syp,
                              self.syb, self.sx_dim, self.sy_dim, gxx, gyy,
                              self.depth)]
        self.assertEqual(magg.call_args_list, expected)
        cube = iris.cube.Cube(data)
        cube.metadata = self.metadata
        self.assertEqual(result, cube)


if __name__ == '__main__':
    unittest.main()

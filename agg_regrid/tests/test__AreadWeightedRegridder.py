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

from agg_regrid import _AreaWeightedRegridder as Regridder


class Test(unittest.TestCase):
    def setUp(self):
        self.src_cube = mock.Mock(spec=iris.cube.Cube)
        self.tgt_cube = mock.Mock(spec=iris.cube.Cube)

    def test_bad_src_grid_cube(self):
        emsg = 'source grid must be a cube'
        with assertRaisesRegex(self, TypeError, emsg):
            Regridder('dummy', self.tgt_cube)

    def test_bad_tgt_grid_cube(self):
        emsg = 'target grid must be a cube'
        with assertRaisesRegex(self, TypeError, emsg):
            Regridder(self.src_cube, 'dummy')

    def test_snapshot_grid(self):
        snapshot = 'agg_regrid.snapshot_grid'
        src_grid = mock.sentinel.src_grid
        gx = mock.Mock(coord_system=True)
        gy = mock.Mock(coord_system=True)
        tgt_grid = (gx, gy)
        side_effect = [src_grid, tgt_grid]

        with mock.patch(snapshot, side_effect=side_effect):
            regridder = Regridder(self.src_cube, self.tgt_cube)

        self.assertEqual(regridder._src_grid, src_grid)
        self.assertEqual(regridder._gx, gx)
        self.assertEqual(regridder._gy, gy)
        self.assertIsNone(regridder._gx_bounds)
        self.assertIsNone(regridder._gy_bounds)
        self.assertIsNone(regridder._sx_bounds)
        self.assertIsNone(regridder._sy_bounds)

    def test_snapshot_grid__no_gx_coord_system(self):
        snapshot = 'agg_regrid.snapshot_grid'
        src_grid = mock.sentinel.src_grid
        gx = mock.Mock(coord_system=None)
        gy = mock.Mock(coord_system=True)
        tgt_grid = (gx, gy)
        side_effect = [src_grid, tgt_grid]

        with mock.patch(snapshot, side_effect=side_effect):
            emsg = 'grid cube requires a native coordinate system'
            with assertRaisesRegex(self, ValueError, emsg):
                Regridder(self.src_cube, self.tgt_cube)

    def test_snapshot_grid__no_gy_coord_system(self):
        snapshot = 'agg_regrid.snapshot_grid'
        src_grid = mock.sentinel.src_grid
        gx = mock.Mock(coord_system=True)
        gy = mock.Mock(coord_system=None)
        tgt_grid = (gx, gy)
        side_effect = [src_grid, tgt_grid]

        with mock.patch(snapshot, side_effect=side_effect):
            emsg = 'grid cube requires a native coordinate system'
            with assertRaisesRegex(self, ValueError, emsg):
                Regridder(self.src_cube, self.tgt_cube)


if __name__ == '__main__':
    unittest.main()

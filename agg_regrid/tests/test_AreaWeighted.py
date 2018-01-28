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
"""Unit tests for the `agg_regrid.AreaWeighted` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, xrange, zip)  # noqa

import unittest

import mock

from agg_regrid import AreaWeighted


class Test(unittest.TestCase):
    def setUp(self):
        self.src = mock.sentinel.src
        self.tgt = mock.sentinel.tgt
        self.regridder = mock.sentinel.regridder

    def test_regridder(self):
        regridder = 'agg_regrid._AreaWeightedRegridder'
        with mock.patch(regridder, autospec=True,
                        return_value=self.regridder) as mocker:
            scheme = AreaWeighted()
            result = scheme.regridder(self.src, self.tgt)
            self.assertEqual(result, self.regridder)
            expected = [mock.call(self.src, self.tgt)]
            self.assertEqual(mocker.mock_calls, expected)


if __name__ == '__main__':
    unittest.main()

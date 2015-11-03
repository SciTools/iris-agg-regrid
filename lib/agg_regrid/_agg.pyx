# (C) British Crown Copyright 2015, Met Office
#
# This file is part of iris-extras.
#
# iris-extras is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# iris-extras is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with iris-extras.  If not, see <http://www.gnu.org/licenses/>.
"""Anti-Grain Geometry (AGG) raster weight functionality."""

cimport numpy as np


cdef extern from "_agg_raster.h":
    void _raster(np.uint8_t *weights, const double *xi, const double *yi,
                 int nx, int ny)


def raster(np.ndarray[np.uint8_t, ndim=2] weights,
           np.ndarray[np.float64_t, ndim=2] xi,
           np.ndarray[np.float64_t, ndim=2] yi):
    """
    Utilises the sub-pixel accuracy and anti-aliasing capability of the
    Anti-Grain Geometry (AGG) to calculate rasterised weights.

    Renders a single grey-scale (0-255) target cell in the source grid buffer.
    Draws the 4 sided target cell as a straight edged polygon, in a clock-wise
    direction from the top-left-hand-corner, to the top-right-hand-corner, to
    the bottom-right-hand-corner to the bottom-left-hand-corner. 

    The origin of the buffer is the top-left-hand-corner.

    Args:

    * weights:
        The 2d weights buffer is updated in-place.
    * xi:
        The fractional x-coordinates of the target cell corners.
    * yi:
        The fractional y-coordinates of the target cell corners.

    """
    _raster(<np.uint8_t *>weights.data, <const double *>xi.data,
            <const double *>yi.data, weights.shape[1], weights.shape[0])

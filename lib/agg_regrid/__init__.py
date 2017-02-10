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
"""A package for experimental regridding functionality."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import copy
from math import ceil, floor
import operator

import numpy as np
import numpy.ma as ma

import iris
from iris.analysis._interpolation import snapshot_grid, get_xy_dim_coords

from ._agg import raster as agg_raster


__version__ = '0.1.0'


class AreaWeighted(object):
    """
    The Anti-Grain Geometry (AGG) regridding scheme for performing
    area-weighted conservative regridding.

    """
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def regridder(self, src_grid, tgt_grid):
        """
        Creates an Anti-Grain Geometry (AGG) regridder to perform
        area-weighted conservative regridding between the source
        grid and the target grid.

        Args:

        * src_grid:
            The :class:`~iris.cube.Cube` defining the source grid.
        * tgt_grid:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a :class:`iris.cube.Cube` with the same grid
            as the source grid defined for regridding to the target grid.

        """
        return _AreaWeightedRegridder(src_grid, tgt_grid)


class _AreaWeightedRegridder(object):
    """
    This class provides support for performing area-weighted regridding.

    """
    def __init__(self, src_grid_cube, tgt_grid_cube):
        """
        Creates a area-weighted regridder which uses an Anti-Grain
        Geometry (AGG) backend to rasterise the conversion between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * tgt_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.

        """
        if not isinstance(src_grid_cube, iris.cube.Cube):
            raise TypeError('This source grid must be a cube.')
        if not isinstance(tgt_grid_cube, iris.cube.Cube):
            raise TypeError('The target grid must be a cube.')

        # Snapshot the state of the grid cubes to ensure that the regridder
        # is impervious to external changes to the original cubes.
        self._src_grid = snapshot_grid(src_grid_cube)
        self._gx, self._gy = snapshot_grid(tgt_grid_cube)

        # Check the grid cube coordinate system.
        if self._gx.coord_system is None:
            msg = 'The grid cube requires a native coordinate system.'
            raise ValueError(msg)

        # Cache the grid bounds converted to the source crs.
        self._gx_bounds = None
        self._gy_bounds = None

        # Cache the source contiguous bounds.
        self._sx_bounds = None
        self._sy_bounds = None

    def __call__(self, src_cube):
        """
        Regrid the provided :class:`~iris.cube.Cube` on to the target grid
        of this :class:`AreaWeightedRegridder`.

        The supplied :class:`~iris.cube.Cube` must be defined with the same
        grid as the source grid used to create this
        :class:`AreaWeightedRegridder`.

        Args:

        * src_cube:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A :class:`~iris.cube.Cube` defined with the horizontal dimensions
            of the target and the other dimensions from the supplied source
            :class:`~iris.cube.Cube`. The data values of the supplied source
            :class:`~iris.cube.Cube` will be converted to values on the new
            grid using conservative area-weighted regridding.

        """
        # Sanity check the supplied source cube.
        if not isinstance(src_cube, iris.cube.Cube):
            raise TypeError('The source must be a cube.')

        # Get the source cube x and y coordinates.
        sx, sy = get_xy_dim_coords(src_cube)
        if (sx, sy) != self._src_grid:
            emsg = 'The source cube is not defined on the same source grid ' \
                'as this regridder.'
            raise ValueError(emsg)
        if sx.coord_system is None:
            msg = 'The source cube requires a native coordinate system.'
            raise ValueError(msg)

        # Convert the contiguous bounds of the grid to the source crs.
        gxx, gyy = np.meshgrid(self._gx.contiguous_bounds(),
                               self._gy.contiguous_bounds())

        # Now calculate and cache the grid bounds in the source crs.
        if self._gx_bounds is None or self._gy_bounds is None:
            if sx.coord_system == self._gx.coord_system:
                self._gx_bounds, self._gy_bounds = gxx, gyy
            else:
                from_crs = self._gx.coord_system.as_cartopy_crs()
                to_crs = sx.coord_system.as_cartopy_crs()
                xyz = to_crs.transform_points(from_crs, gxx, gyy)
                self._gx_bounds, self._gy_bounds = xyz[..., 0], xyz[..., 1]

        # Calculate and cache the source contiguous bounds.
        if self._sx_bounds is None or self._sy_bounds is None:
            self._sx_bounds = sx.contiguous_bounds()
            self._sy_bounds = sy.contiguous_bounds()

        sx_dim = src_cube.coord_dims(sx)[0]
        sy_dim = src_cube.coord_dims(sy)[0]

        # Perform the regrid.
        result = agg(src_cube.data, sx.points, self._sx_bounds,
                     sy.points, self._sy_bounds, sx_dim, sy_dim,
                     self._gx_bounds, self._gy_bounds)

        #
        # XXX: Need to deal the factories when constructing result cube.
        #
        result_cube = iris.cube.Cube(result)
        result_cube.metadata = copy.deepcopy(src_cube.metadata)
        coord_mapping = {}

        def copy_coords(coords, add_coord):
            for coord in coords:
                dims = src_cube.coord_dims(coord)
                if coord is sx:
                    coord = self._gx
                elif coord is sy:
                    coord = self._gy
                elif sx_dim in dims or sy_dim in dims:
                    continue
                result_coord = coord.copy()
                add_coord(result_coord, dims)
                coord_mapping[id(coord)] = result_coord

        copy_coords(src_cube.dim_coords, result_cube.add_dim_coord)
        copy_coords(src_cube.aux_coords, result_cube.add_aux_coord)

        return result_cube


def agg(data, sx_points, sx_bounds, sy_points, sy_bounds,
        sx_dim, sy_dim, gx_bounds, gy_bounds):
    """
    Perform a area-weighted regrid of the data using an Anti-Grain
    Geometry (AGG) backend to rasterise the conversion between the source
    and target grids.

    Args:

    * data:
        The source grid data, which must be at least 2d, that requires
        to be regridded to the target grid.
    * sx_points:
        The source grid x-coordinate points, which must be 1d, monotonic
        and regular.
    * sx_bounds:
        The source grid x-coordinate contiguous bounds, which must be 1d,
        monotonic and regular.
    * sy_points:
        The source grid y-coordinate points, which must be 1d, monotonic
        and regular.
    * sy_bounds:
        The source grid y-coordinate contiguous bounds, which must be 1d,
        monotonic and regular.
    * sx_dim:
        The data dimension of the x-coordinate.
    * sy_dim:
        The data dimensoin of the y-coordinate.
    * gx_bounds:
        The target grid x-coordinate contiguous bounds, which must be 2d.
        The dimensionality of the target grid is assumed to be in (y, x) order.
    * gy_bounds:
        The target grid y-coordinate contiguous bounds, which must be 2d.
        The dimensionality of the target grid is assumed to be in (y, x) order.

    Returns:
        The data with same horizontal dimensionality as the target grid. The
        data values are converted to the new grid using conservative
        area-weighted regridding.

    """
    #
    # Sanity check the arguments ...
    #
    if sx_points.ndim != 1:
        emsg = 'Expected 1d src x-coordinate points, got {}d.'
        raise ValueError(emsg.format(sx_points.ndim))

    if sy_points.ndim != 1:
        emsg = 'Expected 1d src y-coordinate points, got {}d.'
        raise ValueError(emsg.format(sy_points.ndim))

    if sx_bounds.ndim != 1:
        emsg = 'Expected 1d contiguous src x-coordinate bounds, got {}d.'
        raise ValueError(emsg.format(sx_bounds.ndim))

    if sy_bounds.ndim != 1:
        emsg = 'Expected 1d contiguous src y-coordinate bounds, got {}d.'
        raise ValueError(emsg.format(sy_bounds.ndim))

    if sx_bounds.size != sx_points.size + 1:
        emsg = 'Invalid number of src x-coordinate bounds, got {} expected {}.'
        raise ValueError(emsg.format(sx_bounds.size, sx_points.size + 1))

    if sy_bounds.size != sy_points.size + 1:
        emsg = 'Invalid number of src y-coordinate bounds, got {} expected {}.'
        raise ValueError(emsg.format(sy_bounds.size, sy_points.size + 1))

    # Determine the source data dimensionality ...
    ndim = data.ndim
    dims = list(range(ndim))

    if data.ndim < 2:
        emsg = 'Expected at least 2d src data, got {}d.'
        raise ValueError(emsg.format(data.ndim))

    # Account for negative indexing ...
    dim = sx_dim
    if sx_dim < 0:
        sx_dim = ndim + sx_dim

    if sx_dim not in dims:
        emsg = 'Invalid src x-coordinate dimension, got {} expected ' \
            'within range {}-{}.'
        raise ValueError(emsg.format(dim, dims[0], dims[-1]))

    # Account for negative indexing ...
    dim = sy_dim
    if sy_dim < 0:
        sy_dim = ndim + sy_dim

    if sy_dim not in dims:
        emsg = 'Invalid src y-coordinate dimension, got {} expected ' \
            'within range {}-{}.'
        raise ValueError(emsg.format(dim, dims[0], dims[-1]))

    # Determine the source data shape ...
    shape = data.shape

    if shape[sx_dim] != sx_points.size:
        emsg = 'The src x-coordinate points {} do not align with src data {}' \
            ' over dimension {}.'
        raise ValueError(emsg.format(sx_points.shape, shape, sx_dim))

    if shape[sy_dim] != sy_points.size:
        emsg = 'The src y-coordinate points {} do not align with src data {}' \
            ' over dimension {}.'
        raise ValueError(emsg.format(sy_points.shape, shape, sy_dim))

    if gx_bounds.ndim != 2:
        emsg = 'Expected 2d contiguous grid x-coordinate bounds, got {}d.'
        raise ValueError(emsg.format(gx_bounds.ndim))

    if gy_bounds.ndim != 2:
        emsg = 'Expected 2d contiguous grid y-coordinate bounds, got {}d.'
        raise ValueError(emsg.format(gy_bounds.ndim))

    if gx_bounds.shape != gy_bounds.shape:
        emsg = 'Misaligned grid x-coordinate bounds {} and ' \
            'y-coordinate bounds {}.'
        raise ValueError(emsg.format(gx_bounds.shape, gy_bounds.shape))

    # Ensure the grid bounds have the correct dtype ...
    gx_bounds = np.asarray(gx_bounds, dtype=np.float64)
    gy_bounds = np.asarray(gy_bounds, dtype=np.float64)

    #
    # XXX: Makes gross assumption that all coordinates are increasing.
    # Need to deal with this properly in a generic way.
    #

    def start_and_delta(points, bounds, kind):
        # Constrain to regular points only.
        delta = np.diff(points)
        mean_delta = np.mean(delta)
        rtol = 0.002
        try:
            atol = mean_delta * rtol
            np.testing.assert_allclose(delta, mean_delta, atol=atol)
        except AssertionError as e:
            emsg = 'Expected src {}-coordinate points to be regular{}'
            raise ValueError(emsg.format(kind, e.message))
        return bounds.min(), mean_delta

    sx0, sdx = start_and_delta(sx_points, sx_bounds, 'x')
    sy0, sdy = start_and_delta(sy_points, sy_bounds, 'y')

    #
    # Deal with generic source shape ...
    #
    snx, sny = sx_points.size, sy_points.size
    dr = [sy_dim, sx_dim]
    do = ndim - len(dr)
    ds = sorted(dims, key=lambda d: d in dr)
    dmap = {d: dr.index(d) + do if d in dr else ds.index(d) for d in dims}
    regrid_order, _ = zip(*sorted(dmap.items(), key=operator.itemgetter(1)))
    _, result_order = zip(*sorted(dmap.items(), key=operator.itemgetter(0)))

    if regrid_order != tuple(dims):
        data = np.transpose(data, regrid_order)

    # Reshape the source data into (-1, y, x)
    regrid_shape = data.shape
    data = data.reshape((-1,) + regrid_shape[-2:])

    #
    # Deal with generic grid shape ...
    #
    gnx, gny = gx_bounds.shape[1] - 1, gx_bounds.shape[0] - 1
    result = ma.empty((data.shape[0], gny, gnx))
    result_shape = list(regrid_shape)
    result_shape[-2:] = gny, gnx
    result_shape = tuple(result_shape)
    result.mask = True

    #
    # XXX: Cythonise this ...
    #
    dims = (1, 2)
    for yi in xrange(gny):
        for xi in xrange(gnx):
            yi_stop = yi + 2
            xi_stop = xi + 2
            # Get the bounding box of the grid cell in source coordinates.
            cell_x = gx_bounds[yi:yi_stop, xi:xi_stop]
            cell_y = gy_bounds[yi:yi_stop, xi:xi_stop]
            # Convert to fractional source indices.
            cell_xi = (cell_x - sx0) / sdx
            cell_yi = (cell_y - sy0) / sdy
            xi_min, xi_max = min(*cell_xi.flat), max(*cell_xi.flat)
            yi_min, yi_max = min(*cell_yi.flat), max(*cell_yi.flat)
            if xi_min < 0 or yi_min < 0 or xi_max > snx or yi_max > sny:
                # At least one vertex of the grid cell is out of bounds.
                continue
            # Snap fractional cell indices outwards to actual source indices.
            xi_min = int(floor(xi_min))
            xi_max = int(ceil(xi_max))
            yi_min = int(floor(yi_min))
            yi_max = int(ceil(yi_max))
            # Calculate the weights for the source region
            # overlapped by this grid cell.
            cell_xi -= xi_min
            cell_yi -= yi_min
            weights = np.zeros((yi_max - yi_min, xi_max - xi_min),
                               dtype=np.uint8)
            agg_raster(weights, cell_xi, cell_yi)
            weights = weights / 255
            # Now calculate the weighted result for this grid cell.
            tmp = data[:, yi_min:yi_max, xi_min:xi_max]
            result[:, yi, xi] = (tmp * weights).sum(axis=dims) / weights.sum()

    if result.shape != result_shape:
        result = result.reshape(result_shape)

    if result_order != tuple(dims):
        result = np.transpose(result, result_order)

    return result

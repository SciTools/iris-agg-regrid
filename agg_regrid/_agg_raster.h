/*
# (C) British Crown Copyright 2015 - 2017, Met Office
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
*/

#ifndef _AGG_RASTER_H
#define _AGG_RASTER_H

#include <stdint.h>


void _raster(uint8_t *weights, const double *xi, const double *yi,
             int nx, int ny);

#endif

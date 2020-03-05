/*
# (C) British Crown Copyright 2015 - 2020, Met Office
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

#include <string.h>

#include <agg_basics.h>
#include <agg_pixfmt_gray.h>
#include <agg_rasterizer_scanline_aa.h>
#include <agg_renderer_base.h>
#include <agg_renderer_scanline.h>
#include <agg_rendering_buffer.h>
//#include <agg_scanline_p.h>
#include <agg_scanline_u.h>

#include <_agg_raster.h>


void _raster(uint8_t *weights, const double *xi, const double *yi,
             int nx, int ny)
{
    typedef agg::renderer_base<agg::pixfmt_gray8> ren_base;

    agg::rendering_buffer rbuf(weights, nx, ny, nx);
    agg::pixfmt_gray8 pixf(rbuf);
    ren_base ren(pixf);
    //agg::scanline_p8 sl;
    agg::scanline_u8 sl;

    //ren.clear(agg::gray8(255));

    agg::rasterizer_scanline_aa<> ras;

    ras.reset();
    ras.move_to_d(xi[0], yi[0]);
    ras.line_to_d(xi[1], yi[1]);
    ras.line_to_d(xi[3], yi[3]);
    ras.line_to_d(xi[2], yi[2]);

    agg::render_scanlines_aa_solid(ras, sl, ren, agg::gray8(255));
}

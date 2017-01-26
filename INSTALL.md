Installation
------------

When installing iris-agg-regrid, you must point to the include directory:

      python setup.py build_ext -I/path/to/agg24/source/include
      python setup.py install

Dependencies
------------

Anti-Grain Geometry library v2.4 (http://www.antigrain.com/agg-2.4.zip)
    2D vector graphics library. Agg produces pixel images in memory from
    vectorial data.

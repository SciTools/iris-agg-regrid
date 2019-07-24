from __future__ import absolute_import, division, print_function

import os
from setuptools import Extension, find_packages, setup

from Cython.Build import cythonize
import numpy as np

NAME = 'agg_regrid'
DIR = os.path.abspath(os.path.dirname(__file__))

extensions = [Extension(name='{}._agg'.format(NAME),
                        sources=[os.path.join(DIR, NAME, '_agg.pyx'),
                                 os.path.join(DIR, NAME,
                                              '_agg_raster.cpp')],
                        include_dirs=[os.path.join(DIR, NAME),
                                      os.path.join(DIR, 'extern', 'agg-2.4',
                                                   'include'),
                                      np.get_include()],
                        language='c++')]


def extract_version():
    version = None
    fname = os.path.join(DIR, NAME, '__init__.py')
    with open(fname, 'r') as fi:
        for line in fi:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters
                break
    return version


args = dict(
    name=NAME,
    description=('{} is a cross coordinate system, conservative '
                 'area-weighted regridder, which uses the Anti-Grain Geometry '
                 '(AGG) to rasterise the conversion of a source Iris cube to '
                 'a target grid.'.format(NAME)),
    keywords=NAME,
    version=extract_version(),
    author='UK Met Office',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    classifiers=[
        'Development Status :: 3 - Alpha',
        ('License :: OSI Approved :: '
         'GNU Lesser General Public License v3 or later (LGPLv3+)'),
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        ],
    test_suite='{}.tests'.format(NAME)
)


if __name__ == '__main__':
    setup(**args)

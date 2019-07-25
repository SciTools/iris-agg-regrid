from __future__ import absolute_import, division, print_function

import os
from setuptools import Extension, setup
from setuptools.extension import have_pyrex  # alias for _have_cython
import sys

import numpy as np

BASEDIR = os.path.abspath(os.path.dirname(__file__))
CMDS_SKIP_CYTHONIZE = ['clean', 'sdist']
FLAG_COVERAGE = '--cython-coverage'  # custom flag enabling Cython line tracing
NAME = 'agg_regrid'
AGG_DIR = os.path.join(BASEDIR, NAME)


def extract_version():
    version = None
    fname = os.path.join(AGG_DIR, '__init__.py')
    with open(fname, 'r') as fi:
        for line in fi:
            if line.startswith('__version__'):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters
                break
    return version


compiler_directives = {}
extension_kwargs = {}

if FLAG_COVERAGE in sys.argv or os.environ.get('CYTHON_COVERAGE', None):
    if FLAG_COVERAGE in sys.argv:
        sys.argv.remove(FLAG_COVERAGE)
    if have_pyrex():
        compiler_directives.update({'linetrace': True})
        macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
        extension_kwargs.update({'define_macros': macros})
        print('enable : "linetrace" Cython compiler directive')

extensions = [Extension(name='{}._agg'.format(NAME),
                        sources=[os.path.join(AGG_DIR, '_agg.pyx'),
                                 os.path.join(AGG_DIR, '_agg_raster.cpp')],
                        include_dirs=[AGG_DIR,
                                      os.path.join(BASEDIR, 'extern',
                                                   'agg-2.4', 'include'),
                                      np.get_include()],
                        language='c++',
                        **extension_kwargs)]
ext_modules = extensions

if (not any([arg in CMDS_SKIP_CYTHONIZE for arg in sys.argv]) and
        have_pyrex()):
    from Cython.Build import cythonize

    ext_modules = cythonize(extensions,
                            compiler_directives=compiler_directives,
                            language_level=2)

args = dict(
    name=NAME,
    version=extract_version(),
    ext_modules=ext_modules,
    setup_requires=['setuptools>=18.0', 'numpy'],
    tests_require=['mock'],
    test_suite='{}.tests'.format(NAME)
)


if __name__ == '__main__':
    setup(**args)

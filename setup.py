from __future__ import absolute_import, division, print_function

from distutils.core import setup, Command
import os

import numpy as np
import setuptools


try:
    from Cython.Distutils import build_ext
except ImportError:
    emsg = 'Cython v0.22+ is required'
    raise ImportError(emsg)


def file_walk_relative(top, remove=''):
    """
    Returns a generator of files from the top of the tree, removing
    the given prefix from the root/file result.
    """
    top = top.replace('/', os.path.sep)
    remove = remove.replace('/', os.path.sep)
    for root, dirs, fnames in os.walk(top):
        for fname in fnames:
            yield os.path.join(root, fname).replace(remove, '')


class CleanSource(Command):
    """
    Removes orphaned pyc/pyo files from the sources.

    """
    description = 'clean orphaned pyc/pyo files from sources'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for root_path, dir_names, file_names in os.walk('lib'):
            for file_name in file_names:
                if file_name.endswith('pyc') or file_name.endswith('pyo'):
                    compiled_path = os.path.join(root_path, file_name)
                    source_path = compiled_path[:-1]
                    if not os.path.exists(source_path):
                        print('Cleaning', compiled_path)
                        os.remove(compiled_path)


def extract_version():
    version = None
    dname = os.path.dirname(__file__)
    fname = os.path.join(dname, 'lib', 'agg_regrid', '__init__.py')
    with open(fname, 'r') as fi:
        for line in fi:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters
                break
    return version


setup(
    name='iris-extras',
    version=extract_version(),
    author='UK Met Office',
    packages=['agg_regrid', 'agg_regrid.tests'],
    package_dir={'': 'lib'},
    ext_modules=[
        setuptools.Extension(
            'agg_regrid._agg',
            [os.path.join('lib', 'agg_regrid', '_agg.pyx'),
             os.path.join('lib', 'agg_regrid', '_agg_raster.cpp')],
            include_dirs=[os.path.join('lib', 'agg_regrid'),
                          np.get_include()],
            language='c++',
            ),
        ],
    cmdclass={'clean_source': CleanSource, 'build_ext': build_ext},
    classifiers=[
        'Development Status :: 3 - Alpha',
        ('License :: OSI Approved :: '
         'GNU Lesser General Public License v3 or later (LGPLv3+)'),
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: AIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        ],
    )


import os
import platform
import sys
from distutils import sysconfig
from distutils.command import build
from distutils.command.build_ext import build_ext
from distutils.spawn import spawn

from setuptools import Extension, find_packages, setup
import versioneer

_version_module = None

try:
    from packaging import version as _version_module
except ImportError:
    try:
        from setuptools._vendor.packaging import version as _version_module
    except ImportError:
        pass

min_python_version = "3.7"
max_python_version = "3.11"  # exclusive
min_numpy_build_version = "1.18"
min_numpy_run_version = "1.21"


if sys.platform.startswith('linux'):
    # Patch for #2555 to make wheels without libpython
    sysconfig.get_config_vars()['Py_ENABLE_SHARED'] = 0


def _guard_py_ver():
    if _version_module is None:
        return

    parse = _version_module.parse

    min_py = parse(min_python_version)
    max_py = parse(max_python_version)
    cur_py = parse('.'.join(map(str, sys.version_info[:3])))

    if not min_py <= cur_py < max_py:
        msg = ('Cannot install on Python version {}; only versions >={},<{} '
               'are supported.')
        raise RuntimeError(msg.format(cur_py, min_py, max_py))


_guard_py_ver()

packages = find_packages(include=["lyceanem", "lyceanem.*"])

class build_doc(build.build):
    description = "build documentation"

    def run(self):
        spawn(['make', '-C', 'docs', 'html'])

cmdclass = versioneer.get_cmdclass()
cmdclass['build_doc'] = build_doc

build_requires = ['numpy >={}'.format(min_numpy_build_version)]
install_requires = [
    'numpy >={}'.format(min_numpy_run_version),
    'matplotlib >=3.1.2',
    'numba==0.55.1',
    'open3d==0.9.0.0',
    'scipy==1.4.1',
    'solidpython==0.2.0',
    'setuptools',
    'importlib_metadata; python_version < "3.9"',
]

metadata = dict(
    name='LyceanEM',
    description="LyceanEM is a Python library for modelling electromagnetic propagation for sensors and communications. You can find the documentation at https://lyceanem-python.readthedocs.io/en/latest/",
    version=versioneer.get_version(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Electromagnetics",
    ],
    url="https://lyceanem-python.readthedocs.io/en/latest/index.html",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
    python_requires=">={}".format(min_python_version),
    license="GPL v3",
    cmdclass=cmdclass,
    author='Timothy Pelham',
    author_email='t.g.pelham@bristol.ac.uk',
)


setup(**metadata)

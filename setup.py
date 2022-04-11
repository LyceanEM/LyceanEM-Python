from setuptools import setup, find_packages

setup(
  name='LyceanEM',
  version='0.0.2',
  description='LyceanEM is a Python library for modelling electromagnetic propagation for sensors and communications. You can find the documentation at https://lyceanem-python.readthedocs.io/en/latest/',
  packages=find_packages(exclude=('docs', '*.tests')),
  python_requires='>=3.6',
  install_requires=[
    'numpy~=1.19.2',
    'open3d~=0.9.0.0',
    'cupy~=8.3.0',
    'matplotlib~=3.3.4',
    'numba~=0.52.0',
    'solidpython~=1.1.1',
    'scipy~=1.6.2',
  ],
  package_dir={'':'lyceanem'},
  url='https://lyceanem-python.readthedocs.io/en/latest/',
  author='Timothy Pelham',
  author_email='t.g.pelham@bristol.ac.uk',

)

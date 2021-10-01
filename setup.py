from setuptools import setup, find_packages

setup(
  name='LyceanEM',
  version='0.0.1',
  description='trial packaging of the LyceanEM model to allow for more organised development and eventual distribution',
  packages=find_packages(exclude=('docs', '*.tests')),
  python_requires='>=3.6',
  package_dir={'':'lyceanem'},
  url='https://lyceanem.github.io/',
  author='Timothy Pelham',
  author_email='t.g.pelham@bristol.ac.uk',

)

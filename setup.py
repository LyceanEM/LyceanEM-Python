from setuptools import setup

setup(
  name='LyceanEM',
  version='0.0.1',
  description='trial packaging of the LyceanEM model to allow for more organised development and eventual distribution',
  py_modules=['rayfunctions,empropagation,targets,utility'],
  package_dir={'':'src'},
  url='https://lyceanem.github.io/',
  author='Timothy Pelham',
  author_email='t.g.pelham@bristol.ac.uk',
  
)

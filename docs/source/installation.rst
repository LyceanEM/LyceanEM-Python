
.. _installation:
Installation
=====

LyceanEM can be install via pip, the reccommended method is to create a virtual environment using conda, install cudatoolkit and cupy using conda, and then lyceanem using pip

.. code-block:: console

   $ conda install -c conda-forge cudatoolkit
   $ conda install -c conda-forge cupy
   $ pip install lyceanem

Alternatively the codebase can be downloaded from git directly and built from source


.. code-block:: console

    $ git clone 'https://github.com/LyceanEM/LyceanEM-Python.git'
    $ cd LyceanEM
    $ pip install -e .



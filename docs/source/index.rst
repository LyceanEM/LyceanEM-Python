.. LyceanEM documentation master file, created by
   sphinx-quickstart on Tue Mar 22 09:31:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LyceanEM's documentation!
====================================
**LyceanEM** is a Python library for modelling electromagnetic propagation for sensors and communications.
:ref:`Frequency Domain <frequency_domain>` and :ref:`Time Domain <time_domain>` models are included that allow the
user to model a wide array of complex problems from antenna array architecture and assess beamforming algorithm
performance to channel modelling. The model is built upon a ray tracing approach, allowing for efficient modelling of
large, low density spaces.

.. note::
   This project is under active development, but can be installed by cloning the git repository, as described in the :ref:`install <installation>` documentation.

Installation
-------------
To install LyceanEM, clone the repository using git

.. code-block:: console

    $ git clone 'https://github.com/LyceanEM/LyceanEM-Python.git'

The module can then be installed using pip by navigating to the git folder.

.. code-block:: console

    $ pip3 install .

If you are interesting in developing LyceanEM, you should clone from Github and install with the editable flag set.

.. code-block:: console

    $ pip3 install -e .

Contents
========

.. toctree::
   :maxdepth: 2

   installation
   design
   Tutorial
   models
   electromagnetics
   raycasting
   contributing
   copyright


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
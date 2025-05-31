.. LyceanEM documentation master file, created by
Welcome to LyceanEM's documentation!
====================================

.. image:: _static/LY_logo_RGB_2000px.png
   :width: 300px
   :align: center
   :alt: LyceanEM Logo

**LyceanEM** is a Python library for modelling electromagnetic propagation for sensors and communications.
:ref:`Frequency Domain <frequency_domain>` and :ref:`Time Domain <time_domain>` models are included that allow the
user to model a wide array of complex problems from antenna array architecture and assess beamforming algorithm
performance to channel modelling. The model is built upon a ray tracing approach, allowing for efficient modelling of
large, low density spaces.

.. note::
   This project is under active development, the recommended installation method is to use conda for either Windows or Linux as per the :ref:`install <installation>` documentation.

Installation
-------------
To install LyceanEM using conda

.. code-block:: console

    $ conda install -c lyceanem lyceanem


Development
-----------
To download LyceanEM for development, clone the repository using git

.. code-block:: console

    $ git clone 'https://github.com/LyceanEM/LyceanEM-Python.git'



Contents
========

.. toctree::
   :maxdepth: 2

   installation
   design
   base
   models
   electromagnetics
   raycasting
   geometry
   utility
   auto_examples/index
   contributing
   copyright


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

# LyceanEM
[![PyPI version](https://badge.fury.io/py/lyceanem.svg)](https://pypi.python.org/pypi/metawards)
[![Downloads](https://static.pepy.tech/personalized-badge/lyceanem?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/lyceanem)
 [![status](https://joss.theoj.org/papers/618868c0e8d7e1f7ae6b9f0b4e1e5a2a/status.svg)](https://joss.theoj.org/papers/618868c0e8d7e1f7ae6b9f0b4e1e5a2a)
LyceanEM was conceived to enable rapid assesments of the suitability of difference antenna apertures for a wide range of platforms.
This is based upon the use of ray tracing to determine the field of view of points of interest on the platform, whether building, train, plane, or mobile phone handset. Allowing the application of Wheelers formulation of the gain of an aperture.

This has been developed further since that point to include a frequency domain propagation model, allowing for antenna arrays and aperture antennas to be simulated with environment scattering.

Further development is planned for time domain modelling, computational efficiency, and eventually a Finite-Difference Time-Domain algorithm may be implemented to allow for modelling of a wider range of situations, or possibly hybrid modelling. This would use the FDTD algorithm for near field calculations, while using the ray tracing for more sparse situations.

Further documentation can be found [here](https://lyceanem-python.readthedocs.io/en/latest/index.html).

If you use LyceanEM in an academic project, please cite our paper:

::

    @article{Pelham2021,
        author      = {Timothy G Pelham and Geoff Hilton and Evangelos Mellios and Rob Lewis},
        title       = {Conformal Antenna Array Design Using Aperture Synthesis and On-Platform Modeling},
        journal     = {IEEE Access},
        year        = {2021},
        doi         = {10.1109/ACCESS.2021.3074317}
    }

## Core Features

* 3D Visualization of Platform and Antenna Arrays
* Aperture Projection
* Raycasting
* Frequency Domain Electromagnetics Modelling for scattering, antennas, and antenna array patterns
* Time Domain Electromagnetics Modelling for scattering, antennas, and antenna array patterns
* GPU acceleration of core operations

## Supported Platforms


The package has been tested on:

* Ubuntu and Mint 18.04 and 20.04
* Windows 10 64-bit

With Python versions:

* 3.7

## Installation

LyceanEM uses CUDA for GPU acceleration. The advised installation method is to use Conda to setup a virtual
environment, and installing both cudatoolkit and cupy.

```

   $ conda install -c conda-forge cudatoolkit
   $ conda install -c conda-forge cupy
   $ conda install -c open3d-admin open3d==0.9.0
   $ pip install lyceanem

```
Assuming the cudatoolkit and cupy are already installed, then LyceanEM can also be installed via pip.

```
 pip install LyceanEM
```
## Development Roadmap

`LyceanEM` is electromagnetics simulation software that is used by researchers and engineers in a variety of fields. The software is currently under development, and the developers have outlined a roadmap for future changes. The roadmap includes three key areas:

* Computational efficiency and scalability: The developers plan to improve the computational efficiency of `LyceanEM` so that it can be used on a wider range of hardware platforms, including desktop computers and high-performance computing (HPC) clusters. This will make `LyceanEM` more accessible to a wider range of users. It is the intention of the developers to support antenna arrays with multiple billon antenna elements, `Giga-scale` antenna arrays.
* Core propagation engine: The developers plan to improve the core propagation engine of `LyceanEM` to include more realistic models of lossy propagation, atmospheric effects for each layer of the atmosphere, and dynamic environments. This will make `LyceanEM` more accurate and versatile for a wider range of applications.
* Modelling fidelity: The developers plan to add new features to `LyceanEM` that will allow users to model electromagnetic systems with greater fidelity. This includes support for importing antenna patterns and time domain sources, as well as the development of open source standards for antenna array designs, antenna patterns and field sources, and wireless power transfer.

Here are some specific ways that users can contribute to the development of `LyceanEM`:

* Report bugs: If you find a bug in `LyceanEM`, please report it to the developers so that they can fix it.
* Submit patches: If you know how to fix a bug or add a new feature, please submit a pull request to the developers.
* Donate: If you would like to support the development of `LyceanEM`, you can make a donation to the developers.

Your contributions will help to make `LyceanEM` the best possible electromagnetics simulation software for a wide range of users. Thank you for your support!

## Resources

* Code: [github.com/LyceanEM/LyceanEM-Python](https://github.com/LyceanEM/LyceanEM-Python)
* Documentation: [https://documentation.lyceanem.com/en/latest/index.html](https://documentation.lyceanem.com/en/latest/index.html)
* License: [github.com/LyceanEM/LyceanEM-Python/blob/master/LICENSE.txt](https://github.com/LyceanEM/LyceanEM-Python/blob/master/LICENSE.txt)

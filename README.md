# LyceanEM-Python
Python codebase for LyceanEM

## Background
LyceanEM was conceived to enable rapid assesments of the suitability of difference antenna apertures for a wide range of platforms.
This is based upon the use of ray tracing to determine the field of view of points of interest on the platform, whether building, train, plane, or mobile phone handset. Allowing the application of Wheelers formulation of the gain of an aperture.

This has been developed further since that point to include a frequency domain propagation model, allowing for antenna arrays and aperture antennas to be simulated with environment scattering.

Further development is planned for time domain modelling, computational efficiency, and eventually a Finite-Difference Time-Domain algorithm may be implemented to allow for modelling of a wider range of situations, or possibly hybrid modelling. This would use the FDTD algorithm for near field calculations, while using the ray tracing for more sparse situations.

Further documentation can be found [here](https://lyceanem-python.readthedocs.io/en/latest/index.html)
## Dependencies

The dependencies of this package are limited at present by those of BlueCrystal the University of Bristol's High Performance Computing Resource.
In order to maintain compatibility, Open3D is held at version 0.9.0. 

In addition to the module dependancies, this model is designed to use the features provided by CUDA, and so a compatible Nvidia graphics card is required to run these models.

## Installation

LyceanEM can be installed using conda or pip. Using conda the command is

```
conda install -c lyceanem lyceanem
```

While to install using pip the command is 

```
 pip install LyceanEM
```

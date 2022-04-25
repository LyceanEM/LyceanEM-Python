.. _time_domain:
Time Domain Models
==================

The Time Domain models consider propagation of broadband signals in the time domain, allowing for one model to be run and then evaluated at a range of frequncies. This approach allows for time domain channel models to be created, or numerous frequency domain antenna patterns to be produced by sampling the time domain response at different frequencies using fourier transforms. At present only the general scattering case is implemented, but farfield patterns can still be calculated this way by defining a spherical shell of sinks.

As a mesh approach is used to generate the source coordinates, the mesh resolution for these points should be defined based upon the highest frequency of interest.

.. automodule:: lyceanem.models.time_domain
    :members: calculate_scattering


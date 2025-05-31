.. _models:

Models
========
In order to organise the functions of LyceanEM in a convenient manner, the major uses are arranged as either `Frequency Domain` or `Time Domain` models. The frequency domain models are more developed, but only consider a single spot frequency, and are hence inherently narrowband. The time domain models are much more capable, but have not yet been developed to offer all the same functionality, or support the full capability of the existing functions. A grounding in fourier transforms and signals analysis is required to get the best out of time domain modelling.

In addition to these models there are two different acceleration structures which can be used to organise the ray tracing process. The first (:class:`lyceanem.electromagnetics.acceleration_structures.Tile_acceleration_structure`), is a tile based method which is optimised for scenarios where the source and sink are positioned with the majority of triangles in a plane, such as a satellite above a map tile.

The second (:class:`lyceanem.electromagnetics.acceleration_structures.Brute_Force_acceleration_structure`), is a brute force approach without any acceleration method.

.. toctree::
    accelerationstructures
    frequencydomainmodels
    timedomainmodels
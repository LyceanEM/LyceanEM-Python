.. _targets:
Targets
==================

A collection of useful primatives for scattering and propagation modelling, including a meshedHorn function for consistent definition of Horn antennas.

Targets is named so because it generates objects for the purpose of scattering rays and calculating their contribution to the channel.

The most general purpose function in Targets is :func:'source_cloud_from_shape' which can be used to mesh a shape with consistently spaced scattering or source points with normal vectors inherited from the surface normals of the solid.


.. automodule:: lyceanem.geometry.targets
    :members: source_cloud_from_shape, meshedHorn, meshedReflector


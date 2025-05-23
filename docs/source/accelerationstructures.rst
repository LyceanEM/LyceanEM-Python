.. _accelerationstructures:

Acceleration Structures
=======================

Two different acceleration structures are implemented in LyceanEM. The first is a tile based method which is optimised for scenarios where the source and sink are positioned with the majority of triangles in a plane, such as a satellite above a map tile.

The second is a brute force approach without any acceleration method.


.. automodule:: lyceanem.models.acceleration_structures
    :members: Tile_acceleration_structure, Brute_Force_acceleration_structure

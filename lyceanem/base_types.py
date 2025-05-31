import numpy as np
from numba import from_dtype

# A numpy record array (like a struct) to record triangle
point_data = np.dtype(
    [
        # conductivity, permittivity and permiability
        # free space values should be
        # permittivity of free space 8.8541878176e-12F/m
        # permeability of free space 1.25663706212e-6H/m
        ("permittivity", "c8"),
        ("permeability", "c8"),
        # electric or magnetic current sources? E if True
        ("Electric", "?"),
    ],
    align=True,
)
point_t = from_dtype(point_data)  # Create a type that numba can recognize!

# A numpy record array (like a struct) to record triangle
triangle_data = np.dtype(
    [
        # v0 data
        ("v0x", "f4"),
        ("v0y", "f4"),
        ("v0z", "f4"),
        # v1 data
        ("v1x", "f4"),
        ("v1y", "f4"),
        ("v1z", "f4"),
        # v2 data
        ("v2x", "f4"),
        ("v2y", "f4"),
        ("v2z", "f4"),
    ],
    align=True,
)
triangle_t = from_dtype(triangle_data)  # Create a type that numba can recognize!
"""
A Numba type representing a triangle in 3D space, based on a numpy record array.

Attributes
----------
v0x, v0y, v0z : float
    Coordinates of the first vertex of the triangle.
v1x, v1y, v1z : float
    Coordinates of the second vertex of the triangle.
v2x, v2y, v2z : float
    Coordinates of the third vertex of the triangle.    
"""

# ray class, to hold the ray origin, direction, and eventuall other data.
ray_data = np.dtype(
    [
        # origin data
        ("ox", "f4"),
        ("oy", "f4"),
        ("oz", "f4"),
        # direction vector
        ("dx", "f4"),
        ("dy", "f4"),
        ("dz", "f4"),
        # target
        # direction vector
        # ('tx','f4'),('ty','f4'),('tz','f4'),
        # distance traveled
        ("dist", "f4"),
        # intersection
        ("intersect", "?"),
    ],
    align=True,
)
ray_t = from_dtype(ray_data)
"""
A Numba type representing a ray in 3D space, based on a numpy record array.

Attributes
----------
ox, oy, oz : float
    Coordinates of the origin of the ray.
dx, dy, dz : float
    Direction vector of the ray.
dist : float
    Distance traveled by the ray.
intersect : bool
    Whether the ray has intersected with an object.
"""

scattering_point = np.dtype(
    [
        # position data
        ("px", "f4"),
        ("py", "f4"),
        ("pz", "f4"),
        # velocity
        ("vx", "f4"),
        ("vy", "f4"),
        ("vz", "f4"),
        # normal
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        # weights
        ("ex", "c8"),
        ("ey", "c8"),
        ("ez", "c8"),
        # conductivity, permittivity and permiability
        # free space values should be
        # permittivity of free space 8.8541878176e-12F/m
        # permeability of free space 1.25663706212e-6H/m
        ("permittivity", "c8"),
        ("permeability", "c8"),
        # electric or magnetic current sources? E if True
        ("Electric", "?"),
    ],
    align=True,
)

scattering_t = from_dtype(scattering_point)  # Create a type that numba can recognize!
""" 
A Numba type representing a scattering point in 3D space, based on a numpy record array.

Attributes
----------
px, py, pz : float
    Coordinates of the scattering point.
vx, vy, vz : float
    Velocity vector of the scattering point. Not used in the current implementation.
nx, ny, nz : float
    Normal vector at the scattering point.
ex, ey, ez : complex
    Electric field components at the scattering point.
permittivity : complex
    Permittivity of the medium at the scattering point.
permeability : complex
    Permeability of the medium at the scattering point.
Electric : bool
    Indicates whether the scattering point is an electric source (True) or a magnetic source (False).
"""

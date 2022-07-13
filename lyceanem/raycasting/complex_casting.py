# -*- coding: utf-8 -*-
import copy
import math
from timeit import default_timer as timer

import numba as nb
import numpy as np
import open3d as o3d
import scipy.stats
from matplotlib import cm
from numba import cuda, from_dtype, float32, jit, njit, guvectorize, prange
from numpy.linalg import norm
from scipy.spatial import distance

import lyceanem.base_classes as base_classes
import lyceanem.base_types as base_types
import lyceanem.electromagnetics.empropagation as EM

# A numpy record array (like a struct) to record triangle
complex_triangle = np.dtype(
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
        # material data
        # ('reflection', np.float64),
        # ('diffuse_c', np.float64),
        # ('specular_c', np.float64),
        # interaction type
        ("interaction", "i4"),
    ],
    align=True,
)
triangle_t = from_dtype(complex_triangle)  # Create a type that numba can recognize!

# ray class, to hold the ray origin, direction, and eventuall other data.
complex_ray = np.dtype(
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
        # efield vector (volts/m)
        ("ex", "f4"),
        ("ey", "f4"),
        ("ez", "f4"),
        # distance traveled
        ("dist", "f4"),
        # time taken
        ("time", "f4"),
        # reflections, if zero this must be cast at the sinks, nowhere else
        ("reflections", "i4"),
    ],
    align=True,
)
ray_t = from_dtype(complex_ray)  # Create a type that numba can recognize!


class environment:
    """
    cuda class for the environment, hosting the triangles making up the environment, and the different interaction
    programs.
    """

    def __init__(self, triangles):
        # triangles is a list of triangles in the complex triangle format, with different interaction switches
        self.triangles = []
        for item in triangles:
            self.triangles.append(item)
        # self.materials = []
        # for item in material_characteristics:
        #    self.materials.append(item)

    def ray_check(self, ray):
        """
        Check ray for interaction with stored triangles

        Parameters
        ----------
        ray

        Returns
        -------
        triangle_index : int
            index of closest triangle hit

        """

    def interaction(self, triangle_index, ray):
        """
        calculates the interaction between the ray and triangle indexed
        This will depend on the interaction type specified by the triangle, and could terminate the ray
        (no further action), or generate more rays, either just reflection, reflection & transmission, or straight to
        the sinks.

        Parameters
        ----------
        triangle_index
        ray

        Returns
        -------

        """

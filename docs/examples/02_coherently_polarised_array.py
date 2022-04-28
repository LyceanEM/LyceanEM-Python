#!/usr/bin/env python
# coding: utf-8
"""
Modelling a Coherently Polarised Aperture
======================================================

This example uses the frequency domain :func:`lyceanem.models.frequency_domain.calculate_farfield` function to predict
the farfield pattern for a linearly polarised aperture. This could represent an antenna array without any beamforming
weights.


"""
import numpy as np
import open3d as o3d
import copy

# %%
# Setting Farfield Resolution and Wavelength
# -------------------------------------------
# LyceanEM uses Elevation and Azimuth to record spherical coordinates, ranging from -180 to 180 degrees in azimuth,
# and from -90 to 90 degrees in elevation. In order to launch the aperture projection function, the resolution in
# both azimuth and elevation is requried.
# In order to ensure a fast example, 37 points have been used here for both, giving a total of 1369 farfield points.
#
# The wavelength of interest is also an important variable for antenna array analysis, so we set it now for 10GHz,
# an X band aperture.

az_res = 37
elev_res = 37
wavelength = 3e8 / 10e9

# %%
# Geometries
# ------------------------
# In order to make things easy to start, an example geometry has been included within LyceanEM for a UAV, and the
# open3d trianglemesh structures can be accessed by importing the data subpackage
import lyceanem.tests.reflectordata as data

body, array = data.exampleUAV()

# visualise UAV and Array
o3d.visualization.draw_geometries([body, array])

# %%
# .. image:: ../_static/open3d_structure.png

# crop the inner surface of the array trianglemesh (not strictly required, as the UAV main body provides blocking to
# the hidden surfaces, but correctly an aperture will only have an outer face.
surface_array = copy.deepcopy(array)
surface_array.triangles = o3d.utility.Vector3iVector(
    np.asarray(array.triangles)[: len(array.triangles) // 2, :]
)
surface_array.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(array.triangle_normals)[: len(array.triangle_normals) // 2, :]
)


from lyceanem.models.frequency_domain import calculate_farfield

from lyceanem.geometry.targets import source_cloud_from_shape

source_points = source_cloud_from_shape(surface_array, wavelength * 0.5)

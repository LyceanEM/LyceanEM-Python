#!/usr/bin/env python
# coding: utf-8
"""
Modelling a Coherently Polarised Aperture using the Antenna Structure Class
============================================================================

This example uses the frequency domain :func:`lyceanem.models.frequency_domain.calculate_farfield` function to predict
the farfield pattern for a linearly polarised aperture. This could represent an antenna array without any beamforming
weights. This example differs from 02 by using the antenna structure class as a container for both the antenna points and structure, and also by calling the calculate farfield function using the class.


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

az_res = 181
elev_res = 181
wavelength = 3e8 / 10e9

# %%
# Generating consistent Horn Antenna and
# -------------------------------------------

import lyceanem.geometry.targets as TL
from lyceanem.base_classes import points, structures, antenna_structures

horn_body, aperture_coords = TL.meshedHorn(
    58e-3, 58e-3, 128e-3, 2e-3, 0.21, 0.5 * wavelength
)

aperture = points([aperture_coords])
blockers = structures([horn_body])
horn_antenna = antenna_structures(blockers, aperture)

horn_antenna.visualise_antenna()

# %%
# Generate U directed electric current source
#

desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
desired_E_axis[0, 0] = 1.0
u_pattern = horn_antenna.calculate_farfield(desired_E_axis, wavelength)
u_pattern.display_pattern()
u_pattern.display_pattern(desired_pattern="Power")


# %%
# Generate V directed electric current source
#

desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
desired_E_axis[0, 1] = 1.0
v_pattern = horn_antenna.calculate_farfield(desired_E_axis, wavelength)
v_pattern.display_pattern(desired_pattern="Power")


# %%
# Generate N-normal directed electric current source
#

desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
desired_E_axis[0, 2] = 1.0
n_pattern = horn_antenna.calculate_farfield(desired_E_axis, wavelength)
n_pattern.display_pattern(desired_pattern="Power")


# %%
# Rotate point source and calculate new patterns. The important thing to understand here is that the polarisation is consitent with both the farfield and global axes, and the local antenna orientation, so that if you rotate the antenna and generate the pattern it is consistent with the way the polarisation would change if you rotated a physical antenna in this way. This has been written this way to make modelling antennas and antenna arrays on moving platforms easier, so the local axes and motion relative to the global reference frame can be accounted for in a consistent manner.
#

horn_antenna.rotate_antenna(
    o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.radians(np.asarray([0.0, 45.0, 0.0]))
    )
)
horn_antenna.visualise_antenna()

# %%
# Generate U directed electric current source
#

desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
desired_E_axis[0, 0] = 1.0
u_pattern = horn_antenna.calculate_farfield(desired_E_axis, wavelength)
u_pattern.display_pattern()
u_pattern.display_pattern(desired_pattern="Power")


# %%
# Generate V directed electric current source
#

desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
desired_E_axis[0, 1] = 1.0
v_pattern = horn_antenna.calculate_farfield(desired_E_axis, wavelength)
v_pattern.display_pattern(desired_pattern="Power")


# %%
# Generate N-normal directed electric current source
#

desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
desired_E_axis[0, 2] = 1.0
n_pattern = horn_antenna.calculate_farfield(desired_E_axis, wavelength)
n_pattern.display_pattern(desired_pattern="Power")

#!/usr/bin/env python
# coding: utf-8
"""
Modelling a Physical Channel in the Frequency Domain
======================================================

This example uses the frequency domain :func:`lyceanem.models.frequency_domain.calculate_scattering` function to
predict the scattering parameters for the frequency and environment included in the model.
This model allows for a very wide range of antennas and antenna arrays to be considered, but for simplicity only horn
antennas will be included in this example. The simplest case would be a single source point and single receive point,
rather than an aperture antenna such as a horn.



"""

import numpy as np
import open3d as o3d
import copy

# %%
# Frequency and Mesh Resolution
# ------------------------------
#
freq = np.asarray(15.0e9)
wavelength = 3e8 / freq
mesh_resolution = 0.5 * wavelength

# %%
# Setup transmitters and receivers
# -----------------------------------
#
import lyceanem.geometry.targets as TL
import lyceanem.geometry.geometryfunctions as GF

transmit_horn_structure, transmitting_antenna_surface_coords = TL.meshedHorn(
    58e-3, 58e-3, 128e-3, 2e-3, 0.21, mesh_resolution
)
receive_horn_structure, receiving_antenna_surface_coords = TL.meshedHorn(
    58e-3, 58e-3, 128e-3, 2e-3, 0.21, mesh_resolution
)

# %%
# Position Transmitter
# ----------------------
# rotate the transmitting antenna to the desired orientation, and then translate to final position.
# :func:`lyceanem.geometryfunctions.open3drotate` allows both the center of rotation to be defined, and ensures the
# right syntax is used for Open3d, as it was changed from 0.9.0 to 0.10.0 and onwards.
#
rotation_vector1 = np.radians(np.asarray([90.0, 0.0, 0.0]))
rotation_vector2 = np.radians(np.asarray([0.0, 0.0, -90.0]))
transmit_horn_structure = GF.open3drotate(
    transmit_horn_structure,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
)
transmit_horn_structure = GF.open3drotate(
    transmit_horn_structure,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2),
)
transmit_horn_structure.translate(np.asarray([2.695, 0, 0]), relative=True)
transmitting_antenna_surface_coords = GF.open3drotate(
    transmitting_antenna_surface_coords,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
)
transmitting_antenna_surface_coords = GF.open3drotate(
    transmitting_antenna_surface_coords,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2),
)
transmitting_antenna_surface_coords.translate(np.asarray([2.695, 0, 0]), relative=True)
# %%
# Position Receiver
# ------------------
# rotate the receiving horn to desired orientation and translate to final position.
receive_horn_structure = GF.open3drotate(
    receive_horn_structure,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
)
receive_horn_structure.translate(np.asarray([0, 1.427, 0]), relative=True)
receiving_antenna_surface_coords = GF.open3drotate(
    receiving_antenna_surface_coords,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
)
receiving_antenna_surface_coords.translate(np.asarray([0, 1.427, 0]), relative=True)

# %%
# Create Scattering Plate
# --------------------------
# Create a Scattering plate a source of multipath reflections

reflectorplate, scatter_points = TL.meshedReflector(
    0.3, 0.3, 6e-3, wavelength * 0.5, sides="front"
)
position_vector = np.asarray([29e-3, 0.0, 0])
rotation_vector1 = np.radians(np.asarray([0.0, 90.0, 0.0]))
scatter_points = GF.open3drotate(
    scatter_points,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
)
reflectorplate = GF.open3drotate(
    reflectorplate,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
)
reflectorplate.translate(position_vector, relative=True)
scatter_points.translate(position_vector, relative=True)

# %%
# Specify Reflection Angle
# --------------------------
# Rotate the scattering plate to the optimum angle for reflection from the transmitting to receiving horn

plate_orientation_angle = 45.0

rotation_vector = np.radians(np.asarray([0.0, 0.0, plate_orientation_angle]))
scatter_points = GF.open3drotate(
    scatter_points,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
)
reflectorplate = GF.open3drotate(
    reflectorplate,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
)

from lyceanem.base_classes import structures

blockers = structures([reflectorplate, receive_horn_structure, transmit_horn_structure])

# %%
# Visualise the Scene Geometry
# ------------------------------
# Use open3d function :func:`open3d.visualization.draw_geometries` to visualise the scene and ensure that all the
# relavent sources and scatter points are correct. Point normal vectors can be displayed by pressing 'n' while the
# window is open.
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)
o3d.visualization.draw_geometries(
    [
        transmitting_antenna_surface_coords,
        receiving_antenna_surface_coords,
        scatter_points,
        reflectorplate,
        mesh_frame,
        receive_horn_structure,
        transmit_horn_structure,
    ]
)
# %%
# .. image:: ../_static/03_frequency_domain_channel_model_picture_01.png
#

# %%
# Specify desired Transmit Polarisation
# --------------------------------------
# The transmit polarisation has a significant effect on the channel characteristics. In this example the transmit
# horn will be vertically polarised, (e-vector aligned with the y direction)

desired_E_axis = np.zeros((1, 3), dtype=np.float32)
desired_E_axis[0, 1] = 1.0

# %%
# Frequency Domain Scattering
# ----------------------------
# Once the arrangement of interest has been setup, :func:`lyceanem.models.frequency_domain.calculate_scattering` can
# be called, using raycasting to calculate the scattering parameters based upon the inputs. The scattering parameter
# determines how many reflections will be considered. A value of 0 would mean only line of sight contributions will be
# calculated, with 1 including single reflections, and 2 including double reflections as well.

import lyceanem.models.frequency_domain as FD

Ex, Ey, Ez = FD.calculate_scattering(
    aperture_coords=transmitting_antenna_surface_coords,
    sink_coords=receiving_antenna_surface_coords,
    antenna_solid=blockers,
    desired_E_axis=desired_E_axis,
    scatter_points=scatter_points,
    wavelength=wavelength,
    scattering=1,
)

# %%
# Examine Scattering
# ---------------------
# The resultant scattering is decomposed into the Ex,Ey,Ez components at the receiving antenna, by itself this is not
# that interesting, so for this example we will rotate the reflector back, and then create a loop to step the reflector
# through different angles from 0 to 90 degrees in 1 degree steps.


angle_values = np.linspace(0, 90, 91)
angle_increment = np.diff(angle_values)[0]
responsex = np.zeros((len(angle_values)), dtype="complex")
responsey = np.zeros((len(angle_values)), dtype="complex")
responsez = np.zeros((len(angle_values)), dtype="complex")

plate_orientation_angle = -45.0

rotation_vector = np.radians(
    np.asarray([0.0, 0.0, plate_orientation_angle + angle_increment])
)
scatter_points = GF.open3drotate(
    scatter_points,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
)
reflectorplate = GF.open3drotate(
    reflectorplate,
    o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
)

from tqdm import tqdm

for angle_inc in tqdm(range(len(angle_values))):
    rotation_vector = np.radians(np.asarray([0.0, 0.0, angle_increment]))
    scatter_points = GF.open3drotate(
        scatter_points,
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
    )
    reflectorplate = GF.open3drotate(
        reflectorplate,
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
    )
    Ex, Ey, Ez = FD.calculate_scattering(
        aperture_coords=transmitting_antenna_surface_coords,
        sink_coords=receiving_antenna_surface_coords,
        antenna_solid=blockers,
        desired_E_axis=desired_E_axis,
        scatter_points=scatter_points,
        wavelength=wavelength,
        scattering=1,
    )
    responsex[angle_inc] = Ex
    responsey[angle_inc] = Ey
    responsez[angle_inc] = Ez

# %%
# Plot Normalised Response
# ----------------------------
# Using matplotlib, plot the results

import matplotlib.pyplot as plt

normalised_max = np.max(
    np.array(
        [
            np.max(20 * np.log10(np.abs(responsex))),
            np.max(20 * np.log10(np.abs(responsey))),
            np.max(20 * np.log10(np.abs(responsez))),
        ]
    )
)
ExdB = 20 * np.log10(np.abs(responsex)) - normalised_max
EydB = 20 * np.log10(np.abs(responsey)) - normalised_max
EzdB = 20 * np.log10(np.abs(responsez)) - normalised_max

fig, ax = plt.subplots()
ax.plot(angle_values - 45, ExdB, label="Ex")
ax.plot(angle_values - 45, EydB, label="Ey")
ax.plot(angle_values - 45, EzdB, label="Ez")
plt.xlabel("$\\theta_{N}$ (degrees)")
plt.ylabel("Normalised Level (dB)")
ax.set_ylim(-60.0, 0)
ax.set_xlim(np.min(angle_values) - 45, np.max(angle_values) - 45)
ax.set_xticks(np.linspace(np.min(angle_values) - 45, np.max(angle_values) - 45, 19))
ax.set_yticks(np.linspace(-60, 0.0, 21))
legend = ax.legend(loc="upper right", shadow=True)
plt.grid()
plt.show()

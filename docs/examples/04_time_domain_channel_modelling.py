#!/usr/bin/env python
# coding: utf-8
"""
Modelling a Physical Channel in the Time Domain
======================================================

This example uses the frequency domain :func:`lyceanem.models.time_domain.calculate_scattering` function to
predict the time domain response for a given excitation signal and environment included in the model.
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
sampling_freq = 60e9
model_time = 1e-7
num_samples = int(model_time * (sampling_freq))

# simulate receiver noise
bandwidth = 8e9
kb = 1.38065e-23
receiver_impedence = 50
thermal_noise_power = 4 * kb * 293.15 * receiver_impedence * bandwidth
noise_power = -80  # dbw
mean_noise = 0

model_freq = 10e9
wavelength = 3e8 / model_freq

# %%
# Setup transmitters and receivers
# -----------------------------------
#
import lyceanem.geometry.targets as TL
import lyceanem.geometry.geometryfunctions as GF

transmit_horn_structure, transmitting_antenna_surface_coords = TL.meshedHorn(
    58e-3, 58e-3, 128e-3, 2e-3, 0.21, wavelength * 0.5
)
receive_horn_structure, receiving_antenna_surface_coords = TL.meshedHorn(
    58e-3, 58e-3, 128e-3, 2e-3, 0.21, wavelength * 0.5
)

# %%
# Position Transmitter
# ----------------------
# rotate the transmitting antenna to the desired orientation, and then translate to final position.
# :func:`lyceanem.geometry.geometryfunctions.open3drotate` allows both the center of rotation to be defined, and
# ensures the right syntax is used for Open3d, as it was changed from 0.9.0 to 0.10.0 and onwards.
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

from lyceanem.base import structures

blockers = structures([reflectorplate, receive_horn_structure, transmit_horn_structure])

# %%
# Visualise the Scene Geometry
# ------------------------------
# Use open3d function draw_geometries to visualise the scene and ensure that all the relavent sources and scatter points are correct. Point normal vectors can be displayed by pressing 'n' while the window is open.
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

# %%
# Specify desired Transmit Polarisation
# --------------------------------------
# The transmit polarisation has a significant effect on the channel characteristics. In this example the transmit horn will be vertically polarised, (e-vector aligned with the z direction)

desired_E_axis = np.zeros((1, 3), dtype=np.float32)
desired_E_axis[0, 1] = 1.0

# %%
# Time Domain Scattering
# ----------------------------
#
import scipy.signal as sig
import lyceanem.models.time_domain as TD
from lyceanem.base import structures


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

wake_times = np.zeros((len(angle_values)))
Ex = np.zeros((len(angle_values), num_samples))
Ey = np.zeros((len(angle_values), num_samples))
Ez = np.zeros((len(angle_values), num_samples))

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
    blockers = structures([reflectorplate,transmit_horn_structure,receive_horn_structure])
    pulse_time = 5e-9
    output_power = 0.01  # dBwatts
    powerdbm = 10 * np.log10(output_power) + 30
    v_transmit = ((10 ** (powerdbm / 20)) * receiver_impedence) ** 0.5
    output_amplitude_rms = v_transmit / (1 / np.sqrt(2))
    output_amplitude_peak = v_transmit

    desired_E_axis = np.zeros((3), dtype=np.float32)
    desired_E_axis[2] = 1.0
    noise_volts_peak = (10 ** (noise_power / 10) * receiver_impedence) * 0.5

    excitation_signal = output_amplitude_rms * sig.chirp(
        np.linspace(0, pulse_time, int(pulse_time * sampling_freq)),
        model_freq - bandwidth,
        pulse_time,
        model_freq,
        method="linear",
        phi=0,
        vertex_zero=True,
    ) + np.random.normal(mean_noise, noise_volts_peak, int(pulse_time * sampling_freq))
    (
        Ex[angle_inc, :],
        Ey[angle_inc, :],
        Ez[angle_inc, :],
        wake_times[angle_inc],
    ) = TD.calculate_scattering(
        transmitting_antenna_surface_coords,
        receiving_antenna_surface_coords,
        excitation_signal,
        blockers,
        desired_E_axis,
        scatter_points=scatter_points,
        wavelength=wavelength,
        scattering=1,
        elements=False,
        sampling_freq=sampling_freq,
        num_samples=num_samples,
    )

    noise_volts = np.random.normal(mean_noise, noise_volts_peak, num_samples)
    Ex[angle_inc, :] = Ex[angle_inc, :] + noise_volts
    Ey[angle_inc, :] = Ey[angle_inc, :] + noise_volts
    Ez[angle_inc, :] = Ez[angle_inc, :] + noise_volts


# %%
# Plot Normalised Response
# ----------------------------
# Using matplotlib, plot the results

time_index = np.linspace(0, model_time, num_samples)

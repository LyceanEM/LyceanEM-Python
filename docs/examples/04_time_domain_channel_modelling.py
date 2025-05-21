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

model_freq = 24e9
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
# :func:`lyceanem.geometry.geometryfunctions.translate_mesh`, :func:`lyceanem.geometry.geometryfunctions.mesh_rotate` and :func:`lyceanem.geometry.geometryfunctions.mesh_transform` are included, allowing translation, rotation, and transformation of the meshio objects as required.
#
rotation_vector1 = np.radians(np.asarray([90.0, 0.0, 0.0]))
rotation_vector2 = np.radians(np.asarray([0.0, 0.0, -90.0]))



transmit_horn_structure = GF.mesh_rotate(transmit_horn_structure, rotation_vector1)
transmit_horn_structure = GF.mesh_rotate(transmit_horn_structure, rotation_vector2)
transmit_horn_structure = GF.translate_mesh(transmit_horn_structure, np.asarray([2.695, 0, 0]))
transmitting_antenna_surface_coords = GF.mesh_rotate(transmitting_antenna_surface_coords, rotation_vector1)
transmitting_antenna_surface_coords = GF.mesh_rotate(transmitting_antenna_surface_coords, rotation_vector2)
transmitting_antenna_surface_coords = GF.translate_mesh(transmitting_antenna_surface_coords, np.asarray([2.695, 0, 0]))
# %%
# Position Receiver
# ------------------
# rotate the receiving horn to desired orientation and translate to final position.


receive_horn_structure = GF.mesh_rotate(receive_horn_structure, rotation_vector1)
receive_horn_structure = GF.translate_mesh(receive_horn_structure, np.asarray([0, 1.427, 0]))
receiving_antenna_surface_coords = GF.mesh_rotate(receiving_antenna_surface_coords, rotation_vector1)
receiving_antenna_surface_coords = GF.translate_mesh(receiving_antenna_surface_coords, np.asarray([0, 1.427, 0]))

# %%
# Create Scattering Plate
# --------------------------
# Create a Scattering plate a source of multipath reflections

reflectorplate, scatter_points = TL.meshedReflector(
    0.3, 0.3, 6e-3, wavelength * 0.5, sides="front"
)
position_vector = np.asarray([29e-3, 0.0, 0])
rotation_vector1 = np.radians(np.asarray([0.0, 90.0, 0.0]))
scatter_points = GF.mesh_rotate(scatter_points, rotation_vector1)
reflectorplate = GF.mesh_rotate(reflectorplate, rotation_vector1)
reflectorplate = GF.translate_mesh(reflectorplate, position_vector)
scatter_points = GF.translate_mesh(scatter_points, position_vector)


# %%
# Specify Reflection Angle
# --------------------------
# Rotate the scattering plate to the optimum angle for reflection from the transmitting to receiving horn

plate_orientation_angle = 45.0

rotation_vector = np.radians(np.asarray([0.0, 0.0, plate_orientation_angle]))
scatter_points = GF.mesh_rotate(scatter_points, rotation_vector)
reflectorplate = GF.mesh_rotate(reflectorplate, rotation_vector)
from lyceanem.base_classes import structures

blockers = structures([reflectorplate, receive_horn_structure, transmit_horn_structure])


# %%
# Visualise the Scene Geometry
# ------------------------------


import pyvista as pv

def structure_cells(array):
    ## add collumn of 3s to beggining of each row
    array = np.append(np.ones((array.shape[0], 1), dtype=np.int32) * 3, array, axis=1)
    return array
pyvista_mesh = pv.PolyData(reflectorplate.points, structure_cells(reflectorplate.cells[0].data))
pyvista_mesh2 = pv.PolyData(receive_horn_structure.points, structure_cells(receive_horn_structure.cells[0].data))
pyvista_mesh3 = pv.PolyData(transmit_horn_structure.points, structure_cells(transmit_horn_structure.cells[0].data))
## plot the mesh
plotter = pv.Plotter()
plotter.add_mesh(pyvista_mesh, color="white", show_edges=True)
plotter.add_mesh(pyvista_mesh2, color="blue", show_edges=True)
plotter.add_mesh(pyvista_mesh3, color="red", show_edges=True)
plotter.add_axes_at_origin()
plotter.show()

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
from lyceanem.base_classes import structures


angle_values = np.linspace(0, 90, 91)
angle_increment = np.diff(angle_values)[0]
responsex = np.zeros((len(angle_values)), dtype="complex")
responsey = np.zeros((len(angle_values)), dtype="complex")
responsez = np.zeros((len(angle_values)), dtype="complex")

plate_orientation_angle = -45.0

rotation_vector = np.radians(
    np.asarray([0.0, 0.0, plate_orientation_angle + angle_increment])
)
scatter_points = GF.mesh_rotate(scatter_points, rotation_vector)
reflectorplate = GF.mesh_rotate(reflectorplate, rotation_vector)

from tqdm import tqdm

wake_times = np.zeros((len(angle_values)))
Ex = np.zeros((len(angle_values), num_samples))
Ey = np.zeros((len(angle_values), num_samples))
Ez = np.zeros((len(angle_values), num_samples))

for angle_inc in tqdm(range(len(angle_values))):
    rotation_vector = np.radians(np.asarray([0.0, 0.0, angle_increment]))
    scatter_points = GF.mesh_rotate(scatter_points, rotation_vector)
    reflectorplate = GF.mesh_rotate(reflectorplate, rotation_vector)
    blockers = structures(
        [reflectorplate, transmit_horn_structure, receive_horn_structure]
    )
    pulse_time = 5e-9
    output_power = 0.01  # dBwatts
    powerdbm = 10 * np.log10(output_power) + 30
    v_transmit = ((10 ** (powerdbm / 20)) * receiver_impedence) ** 0.5
    output_amplitude_rms = v_transmit / (1 / np.sqrt(2))
    output_amplitude_peak = v_transmit

    desired_E_axis = np.zeros((3), dtype=np.float32)
    desired_E_axis[1] = 1.0
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
        beta=(2 * np.pi) / wavelength
    )

    noise_volts = np.random.normal(mean_noise, noise_volts_peak, num_samples)
    Ex[angle_inc, :] = Ex[angle_inc, :] + noise_volts
    Ey[angle_inc, :] = Ey[angle_inc, :] + noise_volts
    Ez[angle_inc, :] = Ez[angle_inc, :] + noise_volts


# %%
# Plot Normalised Response
# ----------------------------
# Using matplotlib, plot the results


import matplotlib.pyplot as plt

time_index = np.linspace(0, model_time * 1e9, num_samples)
time, anglegrid = np.meshgrid(time_index[:1801], angle_values - 45)
norm_max = np.nanmax(
    np.array(
        [
            np.nanmax(10 * np.log10((Ex ** 2) / receiver_impedence)),
            np.nanmax(10 * np.log10((Ey ** 2) / receiver_impedence)),
            np.nanmax(10 * np.log10((Ez ** 2) / receiver_impedence)),
        ]
    )
)

fig2, ax2 = plt.subplots(constrained_layout=True)
origin = "lower"
# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.

levels = np.linspace(-80, 0, 41)

CS = ax2.contourf(
    anglegrid,
    time,
    10 * np.log10((Ez[:, :1801] ** 2) / receiver_impedence) - norm_max,
    levels,
    origin=origin,
    extend="both",
)
cbar = fig2.colorbar(CS)
cbar.ax.set_ylabel("Received Power (dBm)")

ax2.set_ylim(0, 30)
ax2.set_xlim(-45, 45)

ax2.set_xticks(np.linspace(-45, 45, 7))
ax2.set_yticks(np.linspace(0, 30, 16))

ax2.set_xlabel("Rotation Angle (degrees)")
ax2.set_ylabel("Time of Flight (ns)")
ax2.set_title("Received Power vs Time for rotating Plate (24GHz)")

from scipy.fft import fft, fftfreq
import scipy

xf = fftfreq(len(time_index), 1 / sampling_freq)[: len(time_index) // 2]
input_signal = excitation_signal * (output_amplitude_peak)
inputfft = fft(input_signal)
input_freq = fftfreq(120, 1 / sampling_freq)[:60]
freqfuncabs = scipy.interpolate.interp1d(input_freq, np.abs(inputfft[:60]))
freqfuncangle = scipy.interpolate.interp1d(input_freq, np.angle(inputfft[:60]))
newinput = freqfuncabs(xf[1600]) * np.exp(freqfuncangle(xf[1600]))
Exf = fft(Ex)
Eyf = fft(Ey)
Ezf = fft(Ez)

# %%
# .. image:: ../_static/sphx_glr_04_time_domain_channel_modelling_001.png

# %%
# Frequency Specific Results
# -------------------------------
# The time of flight plot is useful to displaying the output of the model, giving a understanding about what is
# physically happening in the channel, but to get an idea of the behaviour in the frequency domain we need to use a
# fourier transform to move from time and voltages to frequency.

s21x = 20 * np.log10(np.abs(Exf[:, 1600] / newinput))
s21y = 20 * np.log10(np.abs(Eyf[:, 1600] / newinput))
s21z = 20 * np.log10(np.abs(Ezf[:, 1600] / newinput))
tdangles = np.linspace(-45, 45, 91)
fig, ax = plt.subplots()
ax.plot(tdangles, s21x - np.max(s21z), label="Ex")
ax.plot(tdangles, s21y - np.max(s21z), label="Ey")
ax.plot(tdangles, s21z - np.max(s21z), label="Ez")
plt.xlabel("$\\theta_{N}$ (degrees)")
plt.ylabel("Normalised Level (dB)")
ax.set_ylim(-60.0, 0)
ax.set_xlim(np.min(angle_values) - 45, np.max(angle_values) - 45)
ax.set_xticks(np.linspace(np.min(angle_values) - 45, np.max(angle_values) - 45, 19))
ax.set_yticks(np.linspace(-60, 0.0, 21))
legend = ax.legend(loc="upper right", shadow=True)
plt.grid()
plt.title("$S_{21}$ at 16GHz")
plt.show()

# %%
# .. image:: ../_static/sphx_glr_04_time_domain_channel_modelling_002.png

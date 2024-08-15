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

# %%
# Frequency and Mesh Resolution
# ------------------------------
#
freq = np.asarray(16.0e9)
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
# :func:`lyceanem.geometryfunctions.mesh_rotate` and :func:`lyceanem.geometryfunctions.translate_mesh` are used to achive this
#
rotation_vector1 = np.radians(np.asarray([90.0, 0.0, 0.0]))
rotation_vector2 = np.radians(np.asarray([0.0, 0.0, -90.0]))
rotation_vector3 = np.radians(np.asarray([0.0, 0.0, 90.0]))
transmit_horn_structure = GF.mesh_rotate(
    transmit_horn_structure,
    rotation_vector1
)
transmit_horn_structure = GF.mesh_rotate(transmit_horn_structure,rotation_vector2)

transmit_horn_structure = GF.translate_mesh(transmit_horn_structure,np.asarray([2.695, 0, 0]))

transmitting_antenna_surface_coords = GF.mesh_rotate(transmitting_antenna_surface_coords,rotation_vector1)

transmitting_antenna_surface_coords = GF.mesh_rotate(
    transmitting_antenna_surface_coords,rotation_vector2)

transmitting_antenna_surface_coords = GF.translate_mesh(transmitting_antenna_surface_coords,np.asarray([2.695, 0, 0]))
# %%
# Position Receiver
# ------------------
# rotate the receiving horn to desired orientation and translate to final position.

receive_horn_structure = GF.mesh_rotate(receive_horn_structure,rotation_vector1)
#receive_horn_structure = GF.mesh_rotate(receive_horn_structure,rotation_vector3)
receive_horn_structure = GF.translate_mesh(receive_horn_structure,np.asarray([0, 1.427, 0]))
receiving_antenna_surface_coords = GF.mesh_rotate(receiving_antenna_surface_coords,rotation_vector1)
#receiving_antenna_surface_coords = GF.mesh_rotate(receiving_antenna_surface_coords,rotation_vector3)
receiving_antenna_surface_coords = GF.translate_mesh(receiving_antenna_surface_coords,np.asarray([0, 1.427, 0]))



# %%
# Create Scattering Plate
# --------------------------
# Create a Scattering plate a source of multipath reflections

reflectorplate, scatter_points = TL.meshedReflector(
    0.3, 0.3, 6e-3, wavelength * 0.5, sides="front"
)

position_vector = np.asarray([29e-3, 0.0, 0])
rotation_vector1 = np.radians(np.asarray([0.0, 90.0, 0.0]))
scatter_points = GF.mesh_rotate(
    scatter_points,
   rotation_vector1
)
reflectorplate = GF.mesh_rotate(
    reflectorplate,
    rotation_vector1
)
reflectorplate = GF.translate_mesh(reflectorplate,position_vector)
scatter_points = GF.translate_mesh(scatter_points,position_vector)

# %%
# Specify Reflection Angle
# --------------------------
# Rotate the scattering plate to the optimum angle for reflection from the transmitting to receiving horn

plate_orientation_angle = 45.0

rotation_vector = np.radians(np.asarray([0.0, 0.0, plate_orientation_angle]))
scatter_points = GF.mesh_rotate(
    scatter_points,
    rotation_vector)
reflectorplate = GF.mesh_rotate(
    reflectorplate,
    rotation_vector
)

from lyceanem.base_classes import structures

blockers = structures([reflectorplate, receive_horn_structure, transmit_horn_structure])

# %%
# Visualise the Scene Geometry
# ------------------------------
#############################################NEED TO FIX THIS with pyvista
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
    project_vectors=False,
    beta=(2*np.pi)/wavelength
)


# %%
# Examine Scattering
# ---------------------
# The resultant scattering is decomposed into the Ex,Ey,Ez components at the receiving antenna, by itself this is not
# that interesting, so for this example we will rotate the reflector back, and then create a loop to step the reflector
# through different angles from 0 to 90 degrees in 1 degree steps.




angle_values = np.linspace(0, 90, 181)
angle_increment = np.diff(angle_values)[0]
responsex = np.zeros((len(angle_values)), dtype="complex")
responsey = np.zeros((len(angle_values)), dtype="complex")
responsez = np.zeros((len(angle_values)), dtype="complex")

plate_orientation_angle = -45.0

rotation_vector = np.radians(
    np.asarray([0.0, 0.0, plate_orientation_angle + 0.0])
)
scatter_points = GF.mesh_rotate(scatter_points,rotation_vector)
reflectorplate = GF.mesh_rotate(reflectorplate,rotation_vector)
import copy

from tqdm import tqdm

for angle_inc in tqdm(range(len(angle_values))):
    rotation_vector = np.radians(np.asarray([0.0, 0.0, angle_values[angle_inc]]))
    scatter_points_temp = GF.mesh_rotate(copy.deepcopy(scatter_points),rotation_vector)
    reflectorplate_temp = GF.mesh_rotate(copy.deepcopy(reflectorplate),rotation_vector)
    blockers = structures([reflectorplate_temp, receive_horn_structure, transmit_horn_structure])
    # pyvista_mesh = pv.PolyData(reflectorplate_temp.points, structure_cells(reflectorplate_temp.cells[0].data))
    # pyvista_mesh2 = pv.PolyData(receive_horn_structure.points, structure_cells(receive_horn_structure.cells[0].data))
    # pyvista_mesh3 = pv.PolyData(transmit_horn_structure.points, structure_cells(transmit_horn_structure.cells[0].data))
    # pyvista_mesh4 = pv.PolyData(scatter_points_temp.points)
    # ## plot the mesh
    # plotter = pv.Plotter()
    # plotter.add_mesh(pyvista_mesh, color="white", show_edges=True)
    # plotter.add_mesh(pyvista_mesh2, color="blue", show_edges=True)
    # plotter.add_mesh(pyvista_mesh3, color="red", show_edges=True)
    # plotter.add_mesh(pyvista_mesh4, color="green")
    # plotter.add_axes_at_origin()
    # plotter.show()
    Ex, Ey, Ez = FD.calculate_scattering(
        aperture_coords=transmitting_antenna_surface_coords,
        sink_coords=receiving_antenna_surface_coords,
        antenna_solid=blockers,
        desired_E_axis=desired_E_axis,
        scatter_points=scatter_points_temp,
        wavelength=wavelength,
        scattering=1,
        project_vectors=False
    )
    responsex[angle_inc] = np.sum(Ex)
    responsey[angle_inc] = np.sum(Ey)
    responsez[angle_inc] = np.sum(Ez)





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

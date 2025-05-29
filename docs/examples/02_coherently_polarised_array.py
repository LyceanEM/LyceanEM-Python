#!/usr/bin/env python
# coding: utf-8
"""
Modelling a Coherently Polarised Aperture
======================================================

This example uses the frequency domain :func:`lyceanem.models.frequency_domain.calculate_farfield` function to predict
the farfield pattern for a linearly polarised aperture. This could represent an antenna array without any beamforming
weights.


"""
import copy

import numpy as np

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
# Geometries
# ------------------------
# In order to make things easy to start, an example geometry has been included within LyceanEM for a UAV, and the
# triangle structures can be accessed by importing the data subpackage
import lyceanem.tests.reflectordata as data
body=data.UAV_Demo(wavelength*0.5)
array=data.UAV_Demo_Aperture(wavelength*0.5)



# %%


from lyceanem.base_classes import structures, points,antenna_structures

blockers = structures([body])
aperture=points([array])
array_on_platform=antenna_structures(blockers, aperture)
from lyceanem.models.frequency_domain import calculate_farfield




import pyvista as pv

pl=pv.Plotter()
pl.add_mesh(pv.from_meshio(body),color="green")
pl.add_mesh(pv.from_meshio(array))
pl.add_axes()
pl.show()




# %%
# .. image:: ../_static/sourcecloudfromshapeuav.png

# %%
# Drawbacks of :func:`lyceanem.geometry.geometryfunctions.sourcecloudfromshape`
# ------------------------------------------------------------------------------
# As can be seen by comparing the two source point sets, :func:`lyceanem.geometry.geometryfunctions.sourcecloudfromshape`
# has a significant drawback when used for complex sharply curved antenna arrays, as the poisson disk sampling method
# does not produce consistently spaced results.

desired_E_axis = np.zeros((1, 3), dtype=np.float32)
desired_E_axis[0, 1] = 1.0

Etheta, Ephi = calculate_farfield(
    array_on_platform.export_all_points(),
    array_on_platform.export_all_structures(),
    array_on_platform.excitation_function(desired_e_vector=desired_E_axis,wavelength=wavelength,transmit_power=1.0),
    az_range=np.linspace(-180, 180, az_res),
    el_range=np.linspace(-90, 90, elev_res),
    wavelength=wavelength,
    farfield_distance=20,
    project_vectors=True,
    beta=(2*np.pi)/wavelength
)

# %%
# Storing and Manipulating Antenna Patterns
# ---------------------------------------------
# The resultant antenna pattern can be stored in :class:`lyceanem.base.antenna_pattern` as it has been modelled as one
# distributed aperture, the advantage of this class is the integrated display, conversion and export functions. It is
# very simple to define, and save the pattern, and then display with a call
# to :func:`lyceanem.base.antenna_pattern.display_pattern`. This produces 3D polar plots which can be manipulated to
# give a better view of the whole pattern, but if contour plots are required, then this can also be produced by passing
# plottype='Contour' to the function.

from lyceanem.base_classes import antenna_pattern

UAV_Static_Pattern = antenna_pattern(
    azimuth_resolution=az_res, elevation_resolution=elev_res
)
UAV_Static_Pattern.pattern[:, :, 0] = Etheta.reshape(elev_res,az_res)
UAV_Static_Pattern.pattern[:, :, 1] = Ephi.reshape(elev_res,az_res)

UAV_Static_Pattern.display_pattern(desired_pattern='Power')

# %%
# .. image:: ../_static/sphx_glr_02_coherently_polarised_array_001.png
# .. image:: ../_static/sphx_glr_02_coherently_polarised_array_002.png

UAV_Static_Pattern.display_pattern(plottype="Contour")

# %%
# .. image:: ../_static/sphx_glr_02_coherently_polarised_array_003.png
# .. image:: ../_static/sphx_glr_02_coherently_polarised_array_004.png
pattern_mesh=UAV_Static_Pattern.pattern_mesh()

from lyceanem.electromagnetics.beamforming import create_display_mesh

display_mesh=create_display_mesh(pattern_mesh,label="D(Total)")
display_mesh.plot(scalars="D(Total)")



pl=pv.Plotter()
pl.add_mesh(pv.from_meshio(body),color="green")
pl.add_mesh(pv.from_meshio(array),color="aqua")
pl.add_mesh(display_mesh,scalars="D(Total)",style="points")
pl.add_axes()
pl.show()
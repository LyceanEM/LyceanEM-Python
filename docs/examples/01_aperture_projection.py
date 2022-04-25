#!/usr/bin/env python
# coding: utf-8

"""
Calculating Antenna Array Performance Envelope using Aperture Projection
==========================================================================
This is a demonstration using the aperture projection function in the context of a conformal antenna array mounted upon an unmanned aerial vehicle.

Aperture Projection as a technique is based upon Hannan's formulation of the gain of an aperture based upon its surface area and the freuqency of interest.
This is defined in terms of the maximum gain :math:`G_{max}`, the effective area of the aperture :math:`A_{e}`, and the wavelength of interest :math:`\lambda`.

.. math::
    G_{max}=\dfrac{4 \pi A_{e}}{\lambda^{2}}

While this has been in common use since the 70s, as a formula it is limited to planar surfaces, and only providing the maximum gain in the boresight direction for that surface.

Aperture projection as a function is based upon the rectilinear projection of the aperture into the farfield. This can then be used with Hannan's formula to predict the maximum achievable directivity for all farfield directions of interest :footcite:p:`Pelham2017`.

As this method is built into a raytracing environment, the maximum performance for an aperture on any platform can also be predicted using the :func:`lyceanem.models.frequency_domain.aperture_projection` function.

"""
import numpy as np
import lyceanem.tests.reflectordata as data
import open3d as o3d
import copy
from lyceanem.models.frequency_domain import aperture_projection
from lyceanem.base import structures
#set farfield resolution and wavelength
az_res=37
elev_res=37
wavelength=3e8/10e9
#import example UAV and nose mounted array from examples
body,array=data.exampleUAV()
#visualise UAV and Array
o3d.visualization.draw_geometries([body,array])
#crop the inner surface of the array trianglemesh (not strictly required, as the UAV main body provides blocking to the hidden surfaces, but correctly an aperture will only have an outer face.
surface_array=copy.deepcopy(array)
surface_array.triangles=o3d.utility.Vector3iVector(
        np.asarray(array.triangles)[:len(array.triangles) // 2, :])
surface_array.triangle_normals=o3d.utility.Vector3dVector(
        np.asarray(array.triangle_normals)[:len(array.triangle_normals) // 2, :])

#populate blocking structures
blockers=structures([body])

directivity_envelope,pcd=aperture_projection(surface_array,
                                            environment=blockers,
                                            wavelength=wavelength,
                                            az_range=np.linspace(-180.0, 180.0, az_res),
                                            elev_range=np.linspace(-90.0, 90.0, elev_res))
# %%
# Open3D Visualisation
# ------------------------
# The resultant maximum directivity envelope is provided as both a numpy array of directivities for each angle, but also as an open3d point cloud.
# This allows easy visualisation using the open3d draw_geometries function

o3d.visualization.draw_geometries([body,surface_array,pcd])

#Maximum Directivity
print('Maximum Directivity of {:3.1f} dBi'.format(np.max(10*np.log10(directivity_envelope))))

# %%
# Plotting the Output
# ------------------------
# While the open3d visualisation is very intuitive for examining the results of the

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

azmesh,elevmesh=np.meshgrid(np.linspace(-180.0,180.0,az_res),np.linspace(-90,90,elev_res))
fig, ax = plt.subplots(constrained_layout=True)
origin = 'lower'
# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.

levels = np.linspace(20-40,20,81)
CS = ax.contourf(azmesh,elevmesh, 10*np.log10(directivity_envelope), levels,
                     origin=origin,
                     extend='both')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Relative Power (dB)')
cbar.set_ticks(np.linspace(-20,20,9))
cbar.ax.set_yticklabels(np.linspace(-20,20,9).astype('str'))
levels2=np.linspace(np.nanmax(10*np.log10(directivity_envelope))-60,np.nanmax(10*np.log10(directivity_envelope)),21)
CS4 = ax.contour(azmesh, elevmesh, 10*np.log10(directivity_envelope), levels2,
                          colors=('k',),
                          linewidths=(2,),
                          origin=origin)
ax.set_ylim(-90,90)
ax.set_xlim(-180.0,180)
ax.set_xticks(np.linspace(-180, 180, 13))
ax.set_yticks(np.linspace(-90, 90, 13))
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Maximum Directivity Envelope')

# %%
# .. footbibliography::

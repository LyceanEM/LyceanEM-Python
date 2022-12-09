#!/usr/bin/env python
# coding: utf-8
"""
Modelling Different Farfield Polarisations for an Aperture
=============================================================

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
# Generating consistent aperture to explore farfield polarisations, and rotating the source
# ----------------------------------------------------------------------------------------------

from lyceanem.base_classes import points,structures,antenna_structures

from lyceanem.geometry.targets import meshedHorn
antenna_height=4*wavelength
antenna_width=4*wavelength
antenna_diameter=(((antenna_height*0.5)**2+(antenna_width*0.5)**2)**0.5)*2
structure,array_points=meshedHorn(antenna_height, antenna_width, 4*wavelength, 0.05*wavelength,np.radians(10),wavelength*0.5)

horn_antenna=antenna_structures(structures(solids=[structure]), points(points=[array_points]))
aperture_points=np.asarray(array_points.points).shape[0]

antenna_farfield_distance=((antenna_diameter)**2)/wavelength
step_num=100
sep_scale=np.logspace(0,4,step_num)*wavelength
from lyceanem.models.frequency_domain import calculate_scattering

receive_point_spread=0.1*wavelength
receive_aperture_height=wavelength*8
receive_aperture_width=wavelength*8
receive_diameter=(((receive_aperture_height*0.5)**2+(receive_aperture_width*0.5)**2)**0.5)*2
receive_farfield_distance=((receive_diameter)**2)/wavelength
width_point_num=np.ceil(receive_aperture_width/(receive_point_spread)).astype(int)
height_point_num=np.ceil(receive_aperture_height/(receive_point_spread)).astype(int)

Poynting_vector_distance0p1=np.zeros((width_point_num,height_point_num,step_num))
sample_angle_distance0p1=np.zeros((width_point_num,height_point_num,step_num))
for sep_index in range(len(sep_scale)):
    seperation_distance=sep_scale[sep_index]
    sink_points=o3d.geometry.PointCloud()
    sink_cords=np.zeros((width_point_num*height_point_num,3))
    sink_normals=np.zeros((width_point_num*height_point_num,3))
    grid_meshx,grid_meshy=np.meshgrid(np.linspace(-receive_aperture_width/2,receive_aperture_width/2,width_point_num),np.linspace(-receive_aperture_height/2,receive_aperture_height/2,height_point_num))
    sink_cords[:,0]=grid_meshx.ravel()
    sink_cords[:,1]=grid_meshy.ravel()
    sink_cords[:,2]=seperation_distance
    sink_normals[:,2]=-1.0
    sink_points.points=o3d.utility.Vector3dVector(sink_cords)
    sink_points.normals=o3d.utility.Vector3dVector(sink_normals)
    # %%
    # The first source polarisation is based upon the u-vector of the source point. When the excitation_function method of the antenna structure class is used, it will calculate the appropriate polarisation vectors based upon the local normal vectors.
    
    desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
    desired_E_axis[0, 0] = 1.0
    Ex,Ey,Ez = calculate_scattering(
        horn_antenna.export_all_points(),
        sink_points,
        horn_antenna,
        horn_antenna.excitation_function(desired_e_vector=desired_E_axis),
        elements=True,
        project_vectors=False,
    )
    
    Poynting_vector_distance0p1[:,:,sep_index]=np.abs((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))
    sample_angle_distance0p1[:,:,sep_index]=np.angle((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))

from scipy.stats import circstd
standard_deviation0p1=circstd(sample_angle_distance0p1.reshape(-1,step_num),axis=0)

receive_point_spread=0.25*wavelength
receive_diameter=(((receive_aperture_height*0.5)**2+(receive_aperture_width*0.5)**2)**0.5)*2
receive_farfield_distance=((receive_diameter)**2)/wavelength
transmit_farfield_distance=((antenna_diameter)**2)/wavelength
width_point_num=np.ceil(receive_aperture_width/(receive_point_spread)).astype(int)
height_point_num=np.ceil(receive_aperture_height/(receive_point_spread)).astype(int)

Poynting_vector_distance0p25=np.zeros((width_point_num,height_point_num,step_num))
sample_angle_distance0p25=np.zeros((width_point_num,height_point_num,step_num))
for sep_index in range(len(sep_scale)):
    seperation_distance=sep_scale[sep_index]
    sink_points=o3d.geometry.PointCloud()
    sink_cords=np.zeros((width_point_num*height_point_num,3))
    sink_normals=np.zeros((width_point_num*height_point_num,3))
    grid_meshx,grid_meshy=np.meshgrid(np.linspace(-receive_aperture_width/2,receive_aperture_width/2,width_point_num),np.linspace(-receive_aperture_height/2,receive_aperture_height/2,height_point_num))
    sink_cords[:,0]=grid_meshx.ravel()
    sink_cords[:,1]=grid_meshy.ravel()
    sink_cords[:,2]=seperation_distance
    sink_normals[:,2]=-1.0
    sink_points.points=o3d.utility.Vector3dVector(sink_cords)
    sink_points.normals=o3d.utility.Vector3dVector(sink_normals)
    # %%
    # The first source polarisation is based upon the u-vector of the source point. When the excitation_function method of the antenna structure class is used, it will calculate the appropriate polarisation vectors based upon the local normal vectors.
    
    desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
    desired_E_axis[0, 0] = 1.0
    Ex,Ey,Ez = calculate_scattering(
        horn_antenna.export_all_points(),
        sink_points,
        horn_antenna,
        horn_antenna.excitation_function(desired_e_vector=desired_E_axis),
        elements=True,
        project_vectors=False,
    )
    
    Poynting_vector_distance0p25[:,:,sep_index]=np.abs((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))
    sample_angle_distance0p25[:,:,sep_index]=np.angle((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))

from scipy.stats import circstd
standard_deviation0p25=circstd(sample_angle_distance0p25.reshape(-1,step_num),axis=0)

receive_point_spread=0.5*wavelength
width_point_num=np.ceil(receive_aperture_width/(receive_point_spread)).astype(int)
height_point_num=np.ceil(receive_aperture_height/(receive_point_spread)).astype(int)

Poynting_vector_distance0p5=np.zeros((width_point_num,height_point_num,step_num))
sample_angle_distance0p5=np.zeros((width_point_num,height_point_num,step_num))
for sep_index in range(len(sep_scale)):
    seperation_distance=sep_scale[sep_index]
    sink_points=o3d.geometry.PointCloud()
    sink_cords=np.zeros((width_point_num*height_point_num,3))
    sink_normals=np.zeros((width_point_num*height_point_num,3))
    grid_meshx,grid_meshy=np.meshgrid(np.linspace(-receive_aperture_width/2,receive_aperture_width/2,width_point_num),np.linspace(-receive_aperture_height/2,receive_aperture_height/2,height_point_num))
    sink_cords[:,0]=grid_meshx.ravel()
    sink_cords[:,1]=grid_meshy.ravel()
    sink_cords[:,2]=seperation_distance
    sink_normals[:,2]=-1.0
    sink_points.points=o3d.utility.Vector3dVector(sink_cords)
    sink_points.normals=o3d.utility.Vector3dVector(sink_normals)
    # %%
    # The first source polarisation is based upon the u-vector of the source point. When the excitation_function method of the antenna structure class is used, it will calculate the appropriate polarisation vectors based upon the local normal vectors.
    
    desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
    desired_E_axis[0, 0] = 1.0
    Ex,Ey,Ez = calculate_scattering(
        horn_antenna.export_all_points(),
        sink_points,
        horn_antenna,
        horn_antenna.excitation_function(desired_e_vector=desired_E_axis),
        elements=True,
        project_vectors=False,
    )
    
    Poynting_vector_distance0p5[:,:,sep_index]=np.abs((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))
    sample_angle_distance0p5[:,:,sep_index]=np.angle((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))

from scipy.stats import circstd
standard_deviation0p5=circstd(sample_angle_distance0p5.reshape(-1,step_num),axis=0)

receive_point_spread=1*wavelength
width_point_num=np.ceil(receive_aperture_width/(receive_point_spread)).astype(int)
height_point_num=np.ceil(receive_aperture_height/(receive_point_spread)).astype(int)

Poynting_vector_distance1=np.zeros((width_point_num,height_point_num,step_num))
sample_angle_distance1=np.zeros((width_point_num,height_point_num,step_num))
for sep_index in range(len(sep_scale)):
    seperation_distance=sep_scale[sep_index]
    sink_points=o3d.geometry.PointCloud()
    sink_cords=np.zeros((width_point_num*height_point_num,3))
    sink_normals=np.zeros((width_point_num*height_point_num,3))
    grid_meshx,grid_meshy=np.meshgrid(np.linspace(-receive_aperture_width/2,receive_aperture_width/2,width_point_num),np.linspace(-receive_aperture_height/2,receive_aperture_height/2,height_point_num))
    sink_cords[:,0]=grid_meshx.ravel()
    sink_cords[:,1]=grid_meshy.ravel()
    sink_cords[:,2]=seperation_distance
    sink_normals[:,2]=-1.0
    sink_points.points=o3d.utility.Vector3dVector(sink_cords)
    sink_points.normals=o3d.utility.Vector3dVector(sink_normals)
    # %%
    # The first source polarisation is based upon the u-vector of the source point. When the excitation_function method of the antenna structure class is used, it will calculate the appropriate polarisation vectors based upon the local normal vectors.
    
    desired_E_axis = np.zeros((1, 3), dtype=np.complex64)
    desired_E_axis[0, 0] = 1.0
    Ex,Ey,Ez = calculate_scattering(
        horn_antenna.export_all_points(),
        sink_points,
        horn_antenna,
        horn_antenna.excitation_function(desired_e_vector=desired_E_axis),
        elements=True,
        project_vectors=False,
    )
    
    Poynting_vector_distance1[:,:,sep_index]=np.abs((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))
    sample_angle_distance1[:,:,sep_index]=np.angle((np.sum(np.ones((aperture_points,1))*Ex,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ey,axis=0)**2+np.sum(np.ones((aperture_points,1))*Ez,axis=0)**2).reshape(width_point_num,height_point_num))

from scipy.stats import circstd
standard_deviation1=circstd(sample_angle_distance1.reshape(-1,step_num),axis=0)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(sep_scale/wavelength, np.rad2deg(standard_deviation0p1),label="$0.1\lambda$")
ax.plot(sep_scale/wavelength, np.rad2deg(standard_deviation0p25),label="$0.25\lambda$")
ax.plot(sep_scale/wavelength, np.rad2deg(standard_deviation0p5),label="$0.5\lambda$")
ax.plot(sep_scale/wavelength, np.rad2deg(standard_deviation1),label="$1\lambda$")
plt.axvline(x = receive_farfield_distance/wavelength, color = 'c', label = 'Receive Farfield')
plt.axvline(x = transmit_farfield_distance/wavelength, color = 'm', label = 'Transmit Farfield')
#ax.plot(angle_values-45.0, test2, label='$Small W$')
#ax.plot(angle_values-45.0, test3, label='$Medium$')
#ax.plot(angle_values-45.0, test4, label='$Large T$')
#ax.plot(angle_values-45.0, test5, label='$Large W$')
plt.xlabel('Seperation Distance ($\lambda$)')
plt.ylabel('Standard Deviation of Farfield Phase (degrees)')
plt.title('Effects of Aperture Mesh Resolution on Sampled Phase Variation')
ax.legend()
ax.set_ylim(0.0,30)
#ax.set_xlim(0.0,100)
plt.grid()
plt.show()
#graphname=filename+'Graph.pdf'
#fig.savefig(graphname)   # save the figure to file
#plt.close(fig)  

import matplotlib.animation as animation

def circular_dist(angle1,angle2):
    return np.pi-abs(np.pi-abs(angle1-angle2))

data=copy.deepcopy(np.angle(np.exp(-1j*sample_angle_distance0p1)))
normalised_values=np.mean(np.mean(circular_dist((data),0),axis=0),axis=0)
norm_data=circular_dist((data),0)-normalised_values
dimension1=np.linspace(-receive_aperture_width/2,receive_aperture_width/2,80)/wavelength
dimension2=np.linspace(-receive_aperture_height/2,receive_aperture_height/2,80)/wavelength
dimension3=sep_scale/wavelength
def animate(i):
    title_text='Sampled Phase {:.2f} Wavelengths from the Transmitting Antenna'.format(dimension3[i])
    ax.clear()
    ax.contourf(
        dimension1, dimension2, norm_data[:,:,i], levels, cmap="viridis", origin=origin
    )
    ax.contour(
            dimension1, dimension2, norm_data[:,:,i], levels, colors=("k",), origin=origin
        )
    ax.set_xlim([np.min(dimension1), np.max(dimension1)])
    ax.set_ylim([np.min(dimension2), np.max(dimension2)])
    #ax.set_xticks(np.linspace(-180, 180, 9))
    #ax.set_yticks(np.linspace(-90, 90.0, 13))
    ax.set_xlabel("x ($\lambda$)")
    ax.set_ylabel("y ($\lambda$)")
    ax.set_title(title_text)
    
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
ticknum=9
fig, ax = plt.subplots(constrained_layout=True)
origin = "lower"
pattern_min=-np.pi/2
pattern_max=np.pi/2
levels = np.linspace(pattern_min, pattern_max, 73)
CS = ax.contourf(dimension1, dimension2, norm_data[:,:,0], levels, cmap="viridis", origin=origin)
cbar = fig.colorbar(CS, ticks=np.linspace(pattern_min, pattern_max, ticknum))
bar_label="Sampled Phase Angle (Radians)"
cbar.ax.set_ylabel(bar_label)
c_label_values=np.linspace(pattern_min, pattern_max, ticknum)
c_labels = np.char.mod('%.2f', c_label_values)
cbar.set_ticklabels(c_labels.tolist())
ax.set_xlim([np.min(dimension1), np.max(dimension1)])
ax.set_ylim([np.min(dimension2), np.max(dimension2)])
#ax.set_xticks(np.linspace(-180, 180, 9))
#ax.set_yticks(np.linspace(-90, 90.0, 13))
ax.set_xlabel("x ($\lambda$)")
ax.set_ylabel("y ($\lambda$)")
# setup for 3dB contours
#contournum = np.ceil((pattern_max - pattern_min) / 3).astype(int)
#levels2 = np.linspace(-contournum * 3, plot_max, contournum + 1)
title_text=None
ax.grid()
if title_text != None:
    ax.set_title(title_text)

ani = animation.FuncAnimation(fig, animate, 100, interval=50, blit=False)
plt.show()

f = r"C:/Users/lycea/Documents/10-19 Research Projects/farfieldanimation.gif" 
writergif = animation.PillowWriter(fps=5) 
ani.save(f, writer=writergif)

#Rendering
receive_point_spread=0.1*wavelength
receive_aperture_height=wavelength*8
receive_aperture_width=wavelength*8
receive_diameter=(((receive_aperture_height*0.5)**2+(receive_aperture_width*0.5)**2)**0.5)*2
receive_farfield_distance=((receive_diameter)**2)/wavelength
width_point_num=np.ceil(receive_aperture_width/(receive_point_spread)).astype(int)
height_point_num=np.ceil(receive_aperture_height/(receive_point_spread)).astype(int)

Poynting_vector_distance0p1=np.zeros((width_point_num,height_point_num,step_num))
sample_angle_distance0p1=np.zeros((width_point_num,height_point_num,step_num))
sep_index=10
seperation_distance=sep_scale[sep_index]
sink_points=o3d.geometry.PointCloud()
sink_cords=np.zeros((width_point_num*height_point_num,3))
sink_normals=np.zeros((width_point_num*height_point_num,3))
grid_meshx,grid_meshy=np.meshgrid(np.linspace(-receive_aperture_width/2,receive_aperture_width/2,width_point_num),np.linspace(-receive_aperture_height/2,receive_aperture_height/2,height_point_num))
sink_cords[:,0]=grid_meshx.ravel()
sink_cords[:,1]=grid_meshy.ravel()
sink_cords[:,2]=seperation_distance
sink_normals[:,2]=-1.0
sink_points.points=o3d.utility.Vector3dVector(sink_cords)
sink_points.normals=o3d.utility.Vector3dVector(sink_normals)
o3d.visualization.draw([sink_points,horn_antenna.export_all_points()]+horn_antenna.export_all_structures())
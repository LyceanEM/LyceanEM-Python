#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:15:07 2020

@author: timtitan
"""
import logging
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.signal as sig
# import GPUScatteringTest as GU
import scipy.stats
from scipy.fft import fft, fftfreq
from tqdm import tqdm

import lyceanem.geometry.targets as TL
import lyceanem.models.time_domain as TD
import lyceanem.raycasting.rayfunctions as RF
import lyceanem.tests.reflectordata as reflectordata

nb_logger = logging.getLogger('numba')
nb_logger.setLevel(logging.ERROR)  # only show error
#angle_values=np.linspace(0,90,181)
max_scatter=2
sampling_freq=60e9
model_time=1e-7
num_samples=int(model_time*(sampling_freq))
#time_record=np.zeros((len(angle_values),num_samples,3),dtype='float')
#simulate receiver noise
bandwidth=8e9
kb=1.38065e-23
receiver_impedence=50
thermal_noise_power=4*kb*293.15*receiver_impedence*bandwidth
noise_power=-80#dbw
mean_noise=0

#noise_volts = np.random.normal(mean_noise, np.sqrt(thermal_noise_power*receiver_impedence), num_samples)
#time_ref=np.zeros((len(angle_values)),dtype='float')
model_freq=26e9
wavelength=3e8/model_freq
show_scene=True
transmit_horn_structure,transmitting_antenna_surface_coords=TL.meshedHorn(58e-3,58e-3,128e-3,2e-3,0.21,0.5*wavelength)
rotation_vector1=np.radians(np.asarray([90.0,0.0,0.0]))
rotation_vector2=np.radians(np.asarray([0.0,0.0,-90.0]))
transmit_horn_structure.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),center=False)
transmit_horn_structure.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2),center=False)
transmitting_antenna_surface_coords.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),center=False)
transmitting_antenna_surface_coords.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2),center=False)
transmit_horn_structure.translate(np.asarray([2.695,0,0]),relative=True)
transmitting_antenna_surface_coords.translate(np.asarray([2.695,0,0]),relative=True)
source_coords=np.asarray(transmitting_antenna_surface_coords.points)
source_normals=np.asarray(transmitting_antenna_surface_coords.normals)
receive_horn_structure,recieving_antenna_surface_coords=TL.meshedHorn(58e-3,58e-3,128e-3,2e-3,0.21,0.5*wavelength)
receive_horn_structure.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),center=False)
#rotation_vector3=np.radians(np.asarray([0.0,0.0,-90.0]))
#receive_horn_structure.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector3),center=False)
receive_horn_structure.translate(np.asarray([0,1.427,0]),relative=True)
recieving_antenna_surface_coords.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),center=False)
#recieving_antenna_surface_coords.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector3),center=False)
recieving_antenna_surface_coords.translate(np.asarray([0,1.427,0]),relative=True)

#source_normals[0,2]=1.0
#sinks=np.zeros((1,3),dtype=np.float32)    
#sinks[0,2]=1.0
#sink_normals=np.zeros((1,3),dtype=np.float32)  
#sink_normals[0,2]=-1.0
resolution=2
angles=np.linspace(0,90,91)
wake_times=np.zeros((len(angles)))
Ex=np.zeros((len(angles),num_samples))
Ey=np.zeros((len(angles),num_samples))
Ez=np.zeros((len(angles),num_samples))
time_points=np.zeros((len(angles)))
for angle in tqdm(range(len(angles))):
    start_angle=timer()
    rotation_vector3=np.radians(np.asarray([0.0,90.0,0.0]))
    rotation_vector4=np.radians(np.asarray([0.0,0.0,angles[angle]]))
    #reflectorplate,scatter_points=TL.meshedReflectorv2(0.3,0.3,6e-3,0.5*wavelength,sides='all')
    if(resolution==0):
        #reflectorplate=rectReflector(1.0,1.0,0.1)
        reflectorplate,scatter_points=TL.meshedReflector(0.3,0.15,6e-3,wavelength*0.5,sides='front')
        filename='SmallTallReflector'
        position_vector=np.asarray([29e-3,0.0,0])
    elif(resolution==1):
        reflectorplate,scatter_points=TL.meshedReflector(0.15,0.3,6e-3,wavelength*0.5,sides='front')
        filename='SmallWideReflector'
        position_vector=np.asarray([29e-3,0.0,0])
        #reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare1Meshed.stl")            
    elif(resolution==2):
        #point_area=(mesh_resolution)**2
        #reflectorplate=copy.deepcopy(reflectorplate_main)
        #scatter_points=copy.deepcopy(scatter_points_main)
        reflectorplate,scatter_points=TL.meshedReflector(0.3,0.3,6e-3,wavelength*0.5,sides='all')
        #scatter_points=TL.meshsolid(reflectorplate, point_area)
        filename='MediumSquareReflectorEMtest'
        position_vector=np.asarray([29e-3,0.0,0])
        #position_vector=np.asarray([0,0.0,0])
        #reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare0p5Meshed.stl")
    elif(resolution==3):
        reflectorplate,scatter_points=TL.meshedReflector(0.3,0.6,6e-3,wavelength*0.5)
        filename='LargeWideReflector'
        position_vector=np.asarray([29e-3,0.0,0])
        #reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare0p1Meshed.stl")
    elif(resolution==4):
        reflectorplate,scatter_points=TL.meshedReflector(0.6,0.3,6e-3,wavelength*0.5)
        filename='LargeTallReflector'
        position_vector=np.asarray([29e-3,0.0,0])
        #reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare0p1Meshed.stl")
        #scatter_points=o3d.geometry.PointCloud()
        #reflectorplate.compute_vertex_normals()
        #scatter_points.points=o3d.utility.Vector3dVector(np.asarray(reflectorplate.vertices)+np.asarray(reflectorplate.vertex_normals)*EPSILON)
    elif(resolution==5):
        #reflectorplate,scatter_points=RF.parabolic_reflector(0.6,0.147,6e-3,wavelength*5,sides='front')
        #reflectorplate,scatter_points=TL.parabolic_reflector(0.3,0.147,6e-3,wavelength*0.5,sides='front')
        reflectorplate,scatter_points=TL.parabola(0.6,0.147,6e-3,wavelength*0.5)
        position_vector=np.asarray([0.0,0.0,17e-2])
        filename='ParabaloidReflector'
    elif(resolution==6):
        reflectorplate,scatter_points=reflectordata.mediumreferencescan(wavelength*0.5)
        position_vector=np.asarray([0,0,29e-3])
        filename='LaserScannedMediumReference'
    elif(resolution==7):
        reflectorplate,scatter_points=reflectordata.prototypescan(wavelength*0.5)
        position_vector=np.asarray([29e-3,0.0,0])
        filename='LaserScannedPrototypeReflector'
    #reflectorplate,scatter_points=TL.meshedReflector(0.3,0.3,6e-3,wavelength*0.5,sides='front')
    #filename='SmallTallReflector'
    position_vector=np.asarray([29e-3,0.0,0])
    #reflectorplate.subdivide_midpoint(4)
    reflectorplate.compute_triangle_normals()
    #position_vector=np.asarray([29e-3,0.0,0])
    #position_vector=np.asarray([0,0.0,0])
    reflectorplate.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector3),center=False)
    reflectorplate.translate(position_vector)
    reflectorplate.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector4),center=False)
    scatter_points.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector3),center=False)
    scatter_points.translate(position_vector)
    scatter_points.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector4),center=False)
    #reflectorplate.translate([-angles[angle]/100,0,0])
    #scatter_points.translate([-angles[angle]/100,0,0])
    noise_volts = np.random.normal(mean_noise, np.sqrt(thermal_noise_power*receiver_impedence), num_samples)
    scattering_points=np.asarray(scatter_points.points)
    scattering_normals=np.asarray(scatter_points.normals)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    temp_environment=RF.convertTriangles(reflectorplate)
    #full_index=RF.workchunkingv2(source_coords,sinks,scattering_points,temp_environment,max_scatter)
    #o3d.visualization.draw_geometries([mesh_frame,recieving_antenna_surface_coords,transmitting_antenna_surface_coords,reflectorplate,scatter_points,transmit_horn_structure])
    #for excitation_index in range(3):
    excitation_index=1
    #process for scattering matrix
    #angle_response[:,:,incrementer,plate,angle_inc],impulse_response[:,:,incrementer,plate,angle_inc],max_time[incrementer,plate,angle_inc]=RF.CalculatePoyntingVectors(total_network,wavelength,full_index,az_bins=np.linspace(-np.pi,np.pi,361),impulse=False,aoa=True)
    #scatter_map_test=RF.VectorNetworkProcessv2(np.asarray(transmitting_antenna_surface_coords.points),np.asarray(recieving_antenna_surface_coords.points),scattering_points,full_index,wavelength)
        #scatter_map=RF.VectorNetworkProcessEM(source_coords.shape[0],sinks.shape[0],unified_model,unified_normals,unified_weights,point_informationv2,full_index,scattering_coefficient.astype(np.complex64),wavelength)
    #scatter_map=RF.VectorNetworkProcessTimeEM(source_coords.shape[0],sinks.shape[0],unified_model,unified_normals,unified_weights,point_informationv2,full_index,wavelength)
    #responsex2[angle_inc,excitation_index]=np.dot(np.dot(np.ones((transmitting_antenna_points)),scatter_map[:,:,0,1]),np.ones((receiving_antenna_points)))
    #responsey2[angle_inc,excitation_index]=np.dot(np.dot(np.ones((transmitting_antenna_points)),scatter_map[:,:,1,1]),np.ones((receiving_antenna_points)))
    #responsez2[angle_inc,excitation_index]=np.dot(np.dot(np.ones((transmitting_antenna_points)),scatter_map[:,:,2,1]),np.ones((receiving_antenna_points)))
    #responsex[angle_inc,excitation_index]=np.dot(np.dot(np.ones((transmitting_antenna_points)),np.sum(scatter_map[:,:,0,:],axis=2)),np.ones((receiving_antenna_points)))
    #responsey[angle_inc,excitation_index]=np.dot(np.dot(np.ones((transmitting_antenna_points)),np.sum(scatter_map[:,:,1,:],axis=2)),np.ones((receiving_antenna_points)))
    #responsez[angle_inc,excitation_index]=np.dot(np.dot(np.ones((transmitting_antenna_points)),np.sum(scatter_map[:,:,2,:],axis=2)),np.ones((receiving_antenna_points)))
    #paths=EM.EMGPUPathLengths(source_coords.shape[0],sinks.shape[0],full_index,point_informationv2)
    #polar_coefficients=EM.EMGPUPolarMixing(source_coords.shape[0],sinks.shape[0],full_index,point_informationv2)
    #ray_components=EM.EMGPUWrapper(num_sources,num_sinks,full_index,point_informationv2,wavelength)
    #loss=EM.pathloss(paths,wavelength)
    #full_rays=loss.reshape(loss.shape[0],1)*polar_coefficients
    #depth_slicelos=full_index[np.equal(full_index[:,2],0),:][:,[0,1]]
    #depth_slicebounce=full_index[~np.equal(full_index[:,2],0),:][:,[0,2]]
    #scatter_iso_map=RF.scatter_net_sorttest(source_coords.shape[0],sinks.shape[0],np.zeros((source_coords.shape[0],sinks.shape[0],2),dtype=np.complex64),depth_slicelos,loss[np.equal(full_index[:,2],0)],0)
    #scatter_iso_map=RF.scatter_net_sorttest(source_coords.shape[0],sinks.shape[0],scatter_iso_map,depth_slicebounce,loss[~np.equal(full_index[:,2],0)],1)
    #responseiso2[angle_inc]=np.dot(np.dot(np.ones((source_coords.shape[0])),np.sum(scatter_iso_map[:,:,:],axis=2)),np.ones((sinks.shape[0])))
    #scatter_map2=RF.scatter_net_sortEM(source_coords.shape[0],sinks.shape[0],np.zeros((source_coords.shape[0],sinks.shape[0],3,2),dtype=np.complex64),depth_slicelos,full_rays[np.equal(full_index[:,2],0),:],0)
    #scatter_map2=RF.scatter_net_sortEM(source_coords.shape[0],sinks.shape[0],scatter_map2,depth_slicebounce,full_rays[~np.equal(full_index[:,2],0),:],1)
    #time_index=np.linspace(0,model_time,num_samples)
    pulse_time=5e-9
    output_power=0.01 #dBwatts
    powerdbm=10*np.log10(output_power)+30
    v_transmit=((10**(powerdbm/20))*receiver_impedence)**0.5
    output_amplitude_rms=v_transmit/(1/np.sqrt(2))
    output_amplitude_peak=v_transmit
    #excitation_signal=sig.unit_impulse(100)
   
    desired_E_axis=np.zeros((3),dtype=np.float32)
    desired_E_axis[2]=1.0
    noise_volts_peak=(10**(noise_power/10)*receiver_impedence)*0.5
    #o3d.visualization.draw_geometries([mesh_frame,reflectorplate,scatter_hit_points,transmit_horn_structure])
    #print(angles[angle],len(full_index[full_index[:,2]!=0,:]))
    excitation_signal=output_amplitude_rms*sig.chirp(np.linspace(0,
                                                                 pulse_time,
                                                                 int(pulse_time*sampling_freq)),
                                                                 model_freq-bandwidth, 
                                                                 pulse_time, 
                                                                 model_freq, 
                                                                 method='linear', 
                                                                 phi=0, 
                                                                 vertex_zero=True)+np.random.normal(mean_noise, noise_volts_peak, int(pulse_time*sampling_freq))
    Ex[angle,:],Ey[angle,:],Ez[angle,:],wake_times[angle]=TD.calculate_scattering(transmitting_antenna_surface_coords,
                                                                recieving_antenna_surface_coords,
                                                                excitation_signal,
                                                                reflectorplate,
                                                                desired_E_axis,
                                                                scatter_points=scatter_points,
                                                                wavelength=wavelength,
                                                                scattering=1,
                                                                elements=False,
                                                                sampling_freq=sampling_freq,
                                                                num_samples=num_samples)
    
    noise_volts=np.random.normal(mean_noise, noise_volts_peak, num_samples)
    Ex[angle,:]=Ex[angle,:]+noise_volts
    Ey[angle,:]=Ey[angle,:]+noise_volts
    Ez[angle,:]=Ez[angle,:]+noise_volts
    time_points[angle]=timer()-start_angle
    #est_remaining=np.mean(time_points[:(angle+1)])*(len(angles)-(angle+1))
    #print("Processing Time:  {:3.1f} s, Estimated Remaining time: {:3.1f} minutes".format(time_points[angle],est_remaining/60) )
    #np.savez('timedomainsmallplatechirp',Ex=Ex,Ey=Ey,Ez=Ez,excitation_signal=excitation_signal,time_index=time_index)
    
norm_max=np.nanmax(np.array([np.nanmax(10*np.log10((Ex**2)/receiver_impedence)),np.nanmax(10*np.log10((Ey**2)/receiver_impedence)),np.nanmax(10*np.log10((Ez**2)/receiver_impedence))]))
np.savez(filename,Ex=Ex,Ey=Ey,Ez=Ez,excitation_signal=excitation_signal,time_index=time_index)
#ax.plot(time_index,time_map[0,0,:])
#ax.plot(time_index,20*np.log10(Ex)-norm_max)
#ax.plot(time_index,20*np.log10(Ey)-norm_max)
#ax.plot(time_index,20*np.log10(Ez[0,:])-norm_max)
#ax.plot(time_index,20*np.log10(Ez[1,:]**2)-norm_max)
#ax.plot(time_index,20*np.log10(Ez[2,:]**2)-norm_max)
#ax.plot(time_index,20*np.log10(Ez[91,:]**2)-norm_max)
fig,ax=plt.subplots()
ax.plot(angles,10*np.log10(np.sum((Ex**2/receiver_impedence),axis=1)))
ax.plot(angles,10*np.log10(np.sum((Ey**2/receiver_impedence),axis=1)))
ax.plot(angles,10*np.log10(np.sum((Ez**2/receiver_impedence),axis=1)))
time_index=np.linspace(0,model_time*1e9,num_samples)
time,anglegrid=np.meshgrid(time_index[:1801],angles-45)
#fig2 = plt.figure()
#ax2 = plt.axes(projection='3d')
fig2, ax2 = plt.subplots(constrained_layout=True)
origin = 'lower'
# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.

levels = np.linspace(-80,0,41)
#CS = ax2.plot_surface(anglegrid,time, 10*np.log10((Ez[:,:1801]**2)/receiver_impedence)-norm_max,levels,
#                    origin=origin,
#                    extend='both')
CS = ax2.contourf(anglegrid,time, 10*np.log10((Ez[:,:1801]**2)/receiver_impedence)-norm_max, levels,
                  origin=origin,
                  extend='both')
cbar = fig2.colorbar(CS)
cbar.ax.set_ylabel('Received Power (dBm)')
#cbar.set_clim(-60,0)
#cbar.set_ticks([norm_max-60,norm_max])
#cbar.ax.set_yticklabels([norm_max-60,norm_max])
ax2.set_ylim(0,30)
ax2.set_xlim(-45,45)
#ax2.set_zlim(-60,0)
ax2.set_xticks(np.linspace(-45, 45, 7))
ax2.set_yticks(np.linspace(0, 30, 16))
#ax2.set_yticklabels(np.linspace(0, np.max(time_index)/1e-9, 11).astype(str))
ax2.set_xlabel('Rotation Angle (degrees)')
ax2.set_ylabel('Time of Flight (ns)')
ax2.set_title('Received Power vs Time for rotating Plate (24GHz)')
#ax2.grid()
#ax.plot(time_index,time_map[0,2,:])
#graphname=filename+'Graph.pdf'
#fig.savefig(graphname)   # save the figure to file
#plt.close(fig)    # close the figure window
# testx2=20*np.log10(np.abs(responsex2[:,2]))
# testy2=20*np.log10(np.abs(responsey2[:,2]))
# testz2=20*np.log10(np.abs(responsez2[:,2]))
# testiso=20*np.log10(np.abs(responseiso))
# #test1=20*np.log10(np.abs(response[:,0]))
# #test2=20*np.log10(np.abs(response[:,1]))
# #test3=20*np.log10(np.abs(response[:,2]))
# #test2=20*np.log10(np.abs(response[:,1]))-np.max(np.max(20*np.log10(np.abs(response[:,1]))))
# #test3=20*np.log10(np.abs(response[:,2]))-np.max(np.max(20*np.log10(np.abs(response[:,2]))))
# #test4=20*np.log10(np.abs(response[:,3]))-np.max(np.max(20*np.log10(np.abs(response[:,3]))))
# #test5=20*np.log10(np.abs(response[:,4]))-np.max(np.max(20*np.log10(np.abs(response[:,4]))))
# #test5=20*np.log10(np.abs(response[:,0])-np.max(np.max(np.abs(response[:,4]))))
# #test4=20*np.log10(np.abs(response[:,3]))
# #test5=20*np.log10(np.abs(response[:,4]))
# #test2=20*np.log10(np.abs(response[:,1]))-np.max(np.max(20*np.log10(np.abs(response[:,1]))))
# #test5=20*np.log10(np.abs(response[:,4]))-np.max(np.max(20*np.log10(np.abs(response[:,4]))))
# #plt.plot(angle_values-45.0,test4,angle_values-45.0,test5)
# # Create plots with pre-defined labels.
# normalised_max=np.max(np.array([np.max(20*np.log10(np.abs(responsex2[:,0]))),np.max(20*np.log10(np.abs(responsex2[:,1]))),np.max(20*np.log10(np.abs(responsex2[:,2]))),np.max(20*np.log10(np.abs(responsex2[:,2]))),np.max(20*np.log10(np.abs(responseiso)))]))
#fig, ax = plt.subplots()
#ax.plot(time_record[0,time_arrive:time_arrive+3000,0],label='Ex')
#ax.plot(time_record[0,time_arrive:time_arrive+3000,1],label='Ey')
#ax.plot(time_record[0,time_arrive:time_arrive+3000,2],label='Ez')
# ax.plot(angle_values-45, testy2-normalised_max,label='Ey')
# ax.plot(angle_values-45, testz2-normalised_max,label='Ez')
#fig, ax = plt.subplots()
#ax.plot(time_record[45,time_arrive:time_arrive+3000,0],label='Ex')
#ax.plot(time_record[45,time_arrive:time_arrive+3000,1],label='Ey')
#ax.plot(time_record[45,time_arrive:time_arrive+3000,2],label='Ez2')
# ax.plot(angle_values-45, testiso-normalised_max, linestyle=':', label='No Polariztion')
# #ax.plot(angle_values-45.0, test3, label='$Medium$')
# #ax.plot(angle_values-45.0, test4, label='$Large T$')
# #ax.plot(angle_values-45.0, test5, label='$Large W$')
# plt.xlabel('$\\theta_{N}$ (degrees)')
# plt.ylabel('Normalised Level (dB)')
# plt.title('Ex Excitation')
# ax.set_ylim(-100.0,0)
# ax.set_xlim(-45.0,45)
# ax.set_xticks(np.linspace(-45, 90-45, 19))
# ax.set_yticks(np.linspace(-100, 0., 21))
# legend = ax.legend(loc='upper right', shadow=True)
# plt.grid()
# plt.show()
freq,psd=sig.welch(Ez,fs=sampling_freq)
freqgrid,anglegrid=np.meshgrid(freq,angles-45)
fig2, ax2 = plt.subplots(constrained_layout=True)
origin = 'lower'
# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.

levels = np.linspace(0,1,61)
CS = ax2.contourf(anglegrid,freqgrid/1e9, psd/np.max(psd), 
                  levels, 
                  origin=origin,
                  extend='both')
cbar = fig2.colorbar(CS)
cbar.ax.set_ylabel('Power Spectral Density')
#cbar.set_ticks([norm_max-60,norm_max])
#cbar.ax.set_yticklabels([norm_max-60,norm_max])
ax2.set_ylim(2,30)
ax2.set_xlim(-45,45)
ax2.set_xticks(np.linspace(-45, 45, 7))
ax2.set_yticks(np.linspace(2, 30, 15))
#ax2.set_yticklabels(np.linspace(0, np.max(time_index)/1e-9, 11).astype(str))
ax2.set_xlabel('Rotation Angle (degrees)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Power Spectral Density for rotating Plate (24GHz)')
xf = fftfreq(len(time_index), 1/sampling_freq)[:len(time_index)//2]
yf=fft(Ez)
w = sig.blackman(6000)
ywf = fft(Ez*w)
freqgrid,anglegrid=np.meshgrid(xf[1:len(time_index)//2],angles-45)
fig2, ax2 = plt.subplots(constrained_layout=True)
origin = 'lower'
# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.

levels = np.linspace(-60,0,61)
CS = ax2.contourf(anglegrid,freqgrid/1e9, 10*np.log10(np.abs(ywf[:,1:len(time_index)//2]))-np.nanmax(10*np.log10(np.abs(ywf[:,1:len(time_index)//2]))), 
                  levels, 
                  origin=origin,
                  extend='both')
cbar = fig2.colorbar(CS)
cbar.ax.set_ylabel('Power (dBm)')
#cbar.set_ticks([norm_max-60,norm_max])
#cbar.ax.set_yticklabels([norm_max-60,norm_max])
ax2.set_ylim(2,30)
ax2.set_xlim(-45,45)
ax2.set_xticks(np.linspace(-45, 45, 7))
ax2.set_yticks(np.linspace(2, 30, 15))
#ax2.set_yticklabels(np.linspace(0, np.max(time_index)/1e-9, 11).astype(str))
ax2.set_xlabel('Rotation Angle (degrees)')
ax2.set_ylabel('Frequency (GHz)')
ax2.set_title('Power Spectral Density for rotating Plate (24GHz)')

#setfreq_psd=20*np.log10(np.abs(ywf[:,1500]))
setfreq_psdalt=20*np.log10(np.abs(yf[:,1500]))
fig,ax=plt.subplots()
#ax.plot(angles,setfreq_psd)
ax.plot(angles,setfreq_psdalt)

input_signal=excitation_signal*(output_amplitude_peak)
inputfft=fft(input_signal)
input_freq=fftfreq(120,1/sampling_freq)[:60]
freqfuncabs=scipy.interpolate.interp1d(input_freq,np.abs(inputfft[:60]))
freqfuncangle=scipy.interpolate.interp1d(input_freq,np.angle(inputfft[:60]))
newinput=freqfuncabs(xf[:2950])*np.exp(freqfuncangle(xf[:2950]))
#calculate s21 at 24GHz over angles
s21=20*np.log10(np.abs(yf[:,:2950]/newinput))
tdangles=np.linspace(-45,45,91)
fig,ax=plt.subplots()
ax.plot(tdangles,s21[:,2400]-np.max(s21[:,2400]),label='24GHz')
freqcroppeds21=s21[:,2200:2601]
freqgrid,anglegrid=np.meshgrid(xf[2200:2601],angles-45)
fig2, ax2 = plt.subplots(constrained_layout=True)
origin = 'lower'
# Now make a contour plot with the levels specified,
# and with the colormap generated automatically from a list
# of colors.

levels = np.linspace(np.nanmax(20*np.log10(np.abs(freqcroppeds21[np.isfinite(freqcroppeds21)])))-10,np.nanmax(20*np.log10(np.abs(freqcroppeds21[np.isfinite(freqcroppeds21)]))),61)
CS = ax2.contourf(anglegrid,freqgrid/1e9, 20*np.log10(np.abs(freqcroppeds21)), 
                  levels, 
                  origin=origin,
                  extend='both')
cbar = fig2.colorbar(CS)
cbar.ax.set_ylabel('$S_{2,1}$ (dB)')
#cbar.set_ticks([norm_max-60,norm_max])
#cbar.ax.set_yticklabels([norm_max-60,norm_max])
ax2.set_ylim(18,30)
ax2.set_xlim(-45,45)
ax2.set_xticks(np.linspace(-45, 45, 10))
ax2.set_yticks(np.linspace(18, 30, 10))
#ax2.set_yticklabels(np.linspace(0, np.max(time_index)/1e-9, 11).astype(str))
ax2.set_xlabel('Rotation Angle (degrees)')
ax2.set_ylabel('Frequency (GHz)')
ax2.set_title('$S_{2,1}$ for rotating Plate (24GHz sine wave)')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cmath
import copy
import math
import pathlib

import cupy as cp
import numpy as np
import open3d as o3d
import scipy.stats
from matplotlib import cm
from numba import cuda, float32, float64, complex64, njit, guvectorize, prange
from numpy.linalg import norm
from scipy.spatial import distance

import lyceanem.base as base
import lyceanem.raycasting.rayfunctions as RF
import lyceanem.geometry.geometryfunctions as GF

@cuda.jit(device=True)
def dot(ax1,ay1,az1,ax2,ay2,az2):
    result=ax1*ax2+ay1*ay2+az1*az2
    return result

@cuda.jit(device=True)
def cross(ax1,ay1,az1,ax2,ay2,az2):
    rx=ay1*az2-az1*ay2
    ry=az1*ax2-ax1*az2
    rz=ax1*ay2-ay1*ax2
    return rx,ry,rz

@cuda.jit(device=True)
def cross_vec(a,b,resultant_vector):
    resultant_vector[0]=a[1]*b[2]-a[2]*b[1]
    resultant_vector[1]=a[2]*b[0]-a[0]*b[2]
    resultant_vector[2]=a[0]*b[1]-a[1]*b[0]
    return resultant_vector

@cuda.jit(device=True)
def cross_vec_norm(a,b,resultant_vector):
    resultant_vector[0]=a[1]*b[2]-a[2]*b[1]
    resultant_vector[1]=a[2]*b[0]-a[0]*b[2]
    resultant_vector[2]=a[0]*b[1]-a[1]*b[0]
    norm=abs(cmath.sqrt(resultant_vector[0]**2.0+resultant_vector[1]**2.0+resultant_vector[2]**2.0))
    resultant_vector[0]=resultant_vector[0]/norm
    resultant_vector[1]=resultant_vector[1]/norm
    resultant_vector[2]=resultant_vector[2]/norm
    return resultant_vector

@cuda.jit(device=True)
def dot_vec(a,b):
    return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])

@cuda.jit(device=True)
def cross_norm(vector_a,vector_b,norm):
    # noinspection PyTypeChecker
    temp_vector = cuda.local.array(shape=(3), dtype=complex64)
    temp_vector[:]=0.0
    cross_vec(vector_a,vector_b,temp_vector)
    norm=abs(cmath.sqrt(temp_vector[0]**2+temp_vector[1]**2+temp_vector[2]**2))
    return norm

@cuda.jit(device=True)
def calc_sep(source,target,dist):
    dist=abs(cmath.sqrt((target['px']-source['px'])**2+(target['py']-source['py'])**2+(target['pz']-source['pz'])**2))+dist
    return dist

@cuda.jit(device=True)
def calc_dv(source,target,vector):
    vector[0]=(target['px']-source['px'])
    vector[1]=(target['py']-source['py'])
    vector[2]=(target['pz']-source['pz'])
    norm=abs(cmath.sqrt(vector[0]**2+vector[1]**2+vector[2]**2))
    vector[0]=vector[0]/norm
    vector[1]=vector[1]/norm
    vector[2]=vector[2]/norm
    return vector

@cuda.jit(device=True)
def transmit(ray_component,starting_point,end_point,wavelength):
    # noinspection PyTypeChecker
    loss1 = cuda.local.array(shape=(1), dtype=complex64)
    # noinspection PyTypeChecker
    lengths = cuda.local.array(shape=(1), dtype=float32)
    wave_vector=(2.0*scipy.constants.pi)/wavelength[0]
    calc_sep(starting_point,end_point,lengths)
    # noinspection PyTypeChecker
    outgoing_dir = cuda.local.array(shape=(3), dtype=complex64)
    # noinspection PyTypeChecker
    local_E_vector = cuda.local.array(shape=(3), dtype=complex64)
    calc_dv(starting_point,end_point,lengths,outgoing_dir)
    if lengths==0:
        loss1=1.0
    else:
        loss1=(cmath.exp(1j*wave_vector*lengths[0]))*(wavelength[0]/(4*(scipy.constants.pi)*(lengths[0])))
        #loss1=(wavelength[0]/(4*(3.1415926535)*(lengths)))

    sourcelaunchtransformGPU(ray_component,starting_point,outgoing_dir)

    ray_component[0]=ray_component[0]*loss1
    ray_component[1]=ray_component[1]*loss1
    ray_component[2]=ray_component[2]*loss1

@cuda.jit(device=True)
def transmitone(ray_component,starting_point,end_point,lengths,wavelength):
    lengths[0]=calc_sep(starting_point,end_point,lengths[0])
    # noinspection PyTypeChecker
    outgoing_dir = cuda.local.array(shape=(3), dtype=complex64)
    calc_dv(starting_point,end_point,lengths,outgoing_dir)
    sourcelaunchtransformGPU(ray_component,starting_point,outgoing_dir)
    ray_component[0]=ray_component[0]*end_point['ex']
    ray_component[1]=ray_component[1]*end_point['ey']
    ray_component[2]=ray_component[2]*end_point['ez']
    #return ray_component,lengths

@cuda.jit(device=True)
def transmitmulti(ray_component,starting_point,end_point,lengths,wavelength):
    lengths[0]=calc_sep(starting_point,end_point,lengths[0])
    # noinspection PyTypeChecker
    outgoing_dir = cuda.local.array(shape=(3), dtype=complex64)
    calc_dv(starting_point,end_point,outgoing_dir)
    ray_component=sourcelaunchtransformGPU(ray_component,starting_point,outgoing_dir)
    ray_component[0]=ray_component[0]*end_point['ex']
    ray_component[1]=ray_component[1]*end_point['ey']
    ray_component[2]=ray_component[2]*end_point['ez']
    return ray_component,lengths

@cuda.jit(device=True)
def pointlink(ray_component,starting_point,end_point,lengths,wavelength):
    #assuming we start with a `static' electric field vector, convert to `ray',
    #then propagate
    lengths=calc_sep(starting_point,end_point,lengths)
    # noinspection PyTypeChecker
    propagation_dir = cuda.local.array(shape=(3), dtype=complex64)
    calc_dv(starting_point,end_point,propagation_dir)
    ray_component=sourcelaunchtransformGPU(ray_component,starting_point,propagation_dir)
    return ray_component,lengths


@cuda.jit(device=True)
def sourcelaunchtransformGPU(ray_field,outgoing_dir):
    """
    Field Transform to create either travelling ray fields or tangential surface currents,
    in each case no electric field can propagate parallel to the normal vector.
    For a travelling ray, the outgoing_dir represents the direction of propagation, while for the surface currents
    the outgoing_dir represents the surface normal vector
    This should be called for each `illumination' when the surface is `hit', and when the outgoing ray is calculated
    """

    # noinspection PyTypeChecker
    temp_E_vector= cuda.local.array(shape=(2), dtype=ray_field.dtype)
    temp_E_vector[:]=0.0
    # noinspection PyTypeChecker
    ray_u= cuda.local.array(shape=(3), dtype=ray_field.dtype)
    # noinspection PyTypeChecker
    ray_v= cuda.local.array(shape=(3), dtype=ray_field.dtype)
    # noinspection PyTypeChecker
    x_vec= cuda.local.array(shape=(3), dtype=float32)
    # noinspection PyTypeChecker
    y_vec= cuda.local.array(shape=(3), dtype=float32)
    # noinspection PyTypeChecker
    z_vec= cuda.local.array(shape=(3), dtype=float32)
    x_orth=float(0)
    y_orth=float(0)
    z_orth=float(0)
    x_vec[:]=0.0
    y_vec[:]=0.0
    z_vec[:]=0.0
    x_vec[0]=1.0
    y_vec[1]=1.0
    z_vec[2]=1.0
    #make sure ray vectors are locked on appropriate global axes
    x_orth=cross_norm(x_vec,outgoing_dir,x_orth)
    y_orth=cross_norm(y_vec,outgoing_dir,y_orth)
    z_orth=cross_norm(z_vec,outgoing_dir,z_orth)

    #print(y_orth)
    #print(z_orth)
    if (abs(x_orth)>abs(y_orth)) and (abs(x_orth)>abs(z_orth)):
        #use x-axis to establish ray uv axes
        cross_vec_norm(x_vec,outgoing_dir,ray_u)

    elif (abs(y_orth)>=abs(x_orth)) and (abs(y_orth)>abs(z_orth)):
        #use y-axis to establish ray uv axes
        cross_vec_norm(y_vec,outgoing_dir,ray_u)

    elif (abs(z_orth)>=abs(x_orth)) and (abs(z_orth)>=abs(y_orth)):
        #use z-axis
        cross_vec_norm(z_vec,outgoing_dir,ray_u)


    cross_vec_norm(outgoing_dir,ray_u,ray_v)
    # #the ray fields must be contained in ray_v and ray_u, as there can be no E field in the direction of propagation
    # #so if the depature vector is 0,0,1, then Ez must be 0.
    temp_E_vector[0]=dot_vec(ray_field,ray_u)
    temp_E_vector[1]=dot_vec(ray_field,ray_v)
    ray_field[:]=0.0
    # #map ray axes onto global coordinate axes to keep everything neat
    # ray_field=np.array([temp_E_vector[0]*np.dot(x_vec,ray_u)+temp_E_vector[1]*np.dot(x_vec,ray_v),
    #                             temp_E_vector[0]*np.dot(y_vec,ray_u)+temp_E_vector[1]*np.dot(y_vec,ray_v),
    #                             temp_E_vector[0]*np.dot(z_vec,ray_u)+temp_E_vector[1]*np.dot(z_vec,ray_v)])
    ray_field[0]=temp_E_vector[0]*dot_vec(x_vec,ray_u)+temp_E_vector[1]*dot_vec(x_vec,ray_v)
    ray_field[1]=temp_E_vector[0]*dot_vec(y_vec,ray_u)+temp_E_vector[1]*dot_vec(y_vec,ray_v)
    ray_field[2]=temp_E_vector[0]*dot_vec(z_vec,ray_u)+temp_E_vector[1]*dot_vec(z_vec,ray_v)

    return ray_field

@cuda.jit(device=True)
def sourcelaunchtransformreal(ray_field,launch_point,outgoing_dir):
    """
    Field Transform to create either travelling ray fields or tangential surface currents,
    in each case no electric field can propagate parallel to the normal vector.
    For a travelling ray, the outgoing_dir represents the direction of propagation, while for the surface currents
    the outgoing_dir represents the surface normal vector
    This should be called for each `illumination' when the surface is `hit', and when the outgoing ray is calculated
    """

    # noinspection PyTypeChecker
    temp_E_vector= cuda.local.array(shape=(2), dtype=float64)
    temp_E_vector[:]=0.0
    # noinspection PyTypeChecker
    ray_u= cuda.local.array(shape=(3), dtype=float32)
    # noinspection PyTypeChecker
    ray_v= cuda.local.array(shape=(3), dtype=float32)
    # noinspection PyTypeChecker
    x_vec= cuda.local.array(shape=(3), dtype=float32)
    # noinspection PyTypeChecker
    y_vec= cuda.local.array(shape=(3), dtype=float32)
    # noinspection PyTypeChecker
    z_vec= cuda.local.array(shape=(3), dtype=float32)
    x_orth=float(0)
    y_orth=float(0)
    z_orth=float(0)
    x_vec[:]=0.0
    y_vec[:]=0.0
    z_vec[:]=0.0
    x_vec[0]=1.0
    y_vec[1]=1.0
    z_vec[2]=1.0
    #make sure ray vectors are locked on appropriate global axes
    x_orth=cross_norm(x_vec,outgoing_dir,x_orth)
    y_orth=cross_norm(y_vec,outgoing_dir,y_orth)
    z_orth=cross_norm(z_vec,outgoing_dir,z_orth)

    #print(y_orth)
    #print(z_orth)
    if (abs(x_orth)>abs(y_orth)) and (abs(x_orth)>abs(z_orth)):
        #use x-axis to establish ray uv axes
        cross_vec_norm(x_vec,outgoing_dir,ray_u)

        #ray_u[0]=ray_u/x_orth
    elif (abs(y_orth)>=abs(x_orth)) and (abs(y_orth)>abs(z_orth)):
    #     #use y-axis to establish ray uv axes
        cross_vec_norm(y_vec,outgoing_dir,ray_u)

    elif (abs(z_orth)>=abs(x_orth)) and (abs(z_orth)>=abs(y_orth)):
    #     #use z-axis
        cross_vec_norm(z_vec,outgoing_dir,ray_u)


    cross_vec_norm(outgoing_dir,ray_u,ray_v)
    # #the ray fields must be contained in ray_v and ray_u, as there can be no E field in the direction of propagation
    # #so if the depature vector is 0,0,1, then Ez must be 0.
    temp_E_vector[0]=dot_vec(ray_field,ray_u)
    temp_E_vector[1]=dot_vec(ray_field,ray_v)
    ray_field[:]=0.0
    # #map ray axes onto global coordinate axes to keep everything neat
    # ray_field=np.array([temp_E_vector[0]*np.dot(x_vec,ray_u)+temp_E_vector[1]*np.dot(x_vec,ray_v),
    #                             temp_E_vector[0]*np.dot(y_vec,ray_u)+temp_E_vector[1]*np.dot(y_vec,ray_v),
    #                             temp_E_vector[0]*np.dot(z_vec,ray_u)+temp_E_vector[1]*np.dot(z_vec,ray_v)])
    ray_field[0]=temp_E_vector[0]*dot_vec(x_vec,ray_u)+temp_E_vector[1]*dot_vec(x_vec,ray_v)
    ray_field[1]=temp_E_vector[0]*dot_vec(y_vec,ray_u)+temp_E_vector[1]*dot_vec(y_vec,ray_v)
    ray_field[2]=temp_E_vector[0]*dot_vec(z_vec,ray_u)+temp_E_vector[1]*dot_vec(z_vec,ray_v)

    return ray_field


@cuda.jit
def scatteringkernal(network_index,point_information,ray_components,wavelength):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    # noinspection PyTypeChecker
    lengths=cuda.local.array(shape=(1),dtype=np.float64)
    while (i < (network_index.shape[1]-1)):
        if i==0:
            lengths[0]=0.0
            ray_components[cu_ray_num,0]=point_information[network_index[cu_ray_num,i]-1]['ex']
            ray_components[cu_ray_num,1]=point_information[network_index[cu_ray_num,i]-1]['ey']
            ray_components[cu_ray_num,2]=point_information[network_index[cu_ray_num,i]-1]['ez']

        if (network_index[cu_ray_num,-1]==0):
            transmitmulti(ray_components[cu_ray_num,:],
                          point_information[network_index[cu_ray_num,i]-1],
                          point_information[network_index[cu_ray_num,i+1]-1],
                          lengths,
                          wavelength)
            i=network_index.shape[1]-1
        else:
            transmitmulti(ray_components[cu_ray_num,:],
                          point_information[network_index[cu_ray_num,i]-1],
                          point_information[network_index[cu_ray_num,i+1]-1],
                          lengths,
                          wavelength)
        #else:
        #    ray_components[cu_ray_num],lengths=transmitmulti(ray_components[cu_ray_num,:],
        #                point_information[network_index[cu_ray_num,i]-1],
        #                point_information[network_index[cu_ray_num,i+1]-1],
        #                wavelength)
        i+=1

    #print(cu_ray_num,lengths[0],abs(ray_components[cu_ray_num,2]))
    wave_vector=(2.0*scipy.constants.pi)/wavelength[0]
    #loss1=cuda.local.array(shape=(1),dtype=complex64)
    if lengths[0]==0.0:
        loss1=1.0
    else:
        loss1=(cmath.exp(1j*wave_vector*lengths[0]))*(wavelength[0]/(4*(scipy.constants.pi)*(lengths[0])))

    #print(cu_ray_num,lengths[0],abs(loss1[0]),abs(ray_components[cu_ray_num,2]))
    ray_components[cu_ray_num,0]=ray_components[cu_ray_num,0]*loss1
    ray_components[cu_ray_num,1]=ray_components[cu_ray_num,1]*loss1
    ray_components[cu_ray_num,2]=ray_components[cu_ray_num,2]*loss1

@cuda.jit
def scatteringkernalv2(problem_size,network_index,point_information,scattering_matrix,wavelength):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    # noinspection PyTypeChecker
    lengths=cuda.local.array(shape=(1),dtype=np.float64)
    # noinspection PyTypeChecker
    ray_component=cuda.local.array(shape=(3),dtype=np.complex64)
    sink_index=network_index[cu_ray_num,-1]-1-problem_size[0]
    #print(cu_ray_num,sink_index)
    while (i < (network_index.shape[1]-1)):
        if i==0:
            lengths[0]=0.0
            if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']
                ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']
                ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']
            else:
                source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability']/point_information[network_index[cu_ray_num,i]-1]['permittivity'])
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=complex64)
                calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                ray_component[0]=ray_component[0]/source_impedance
                ray_component[1]=ray_component[1]/source_impedance
                ray_component[2]=ray_component[2]/source_impedance

        if (network_index[cu_ray_num,i+1]==0):
            sink_index=network_index[cu_ray_num,i]
            i=network_index.shape[1]-1
            #transmitmulti(ray_component,
            #              point_information[network_index[cu_ray_num,i]-1],
            #              point_information[network_index[cu_ray_num,i+1]-1],
            #              lengths,
            #              wavelength)

        else:
            transmitmulti(ray_component,
                          point_information[network_index[cu_ray_num,i]-1],
                          point_information[network_index[cu_ray_num,i+1]-1],
                          lengths,
                          wavelength)
        #else:
        #    ray_components[cu_ray_num],lengths=transmitmulti(ray_components[cu_ray_num,:],
        #                point_information[network_index[cu_ray_num,i]-1],
        #                point_information[network_index[cu_ray_num,i+1]-1],
        #                wavelength)
        i+=1


    wave_vector=(2.0*cmath.pi)/wavelength[0]
    #loss1=cuda.local.array(shape=(1),dtype=complex64)
    if (lengths[0]!=0.0) or (lengths[0]!=cmath.inf):
        loss1=(cmath.exp(1j*wave_vector*lengths[0]))*(wavelength[0]/(4*(cmath.pi)*(lengths[0])))
        scattering_matrix[sink_index,0]+=ray_component[0]*loss1
        scattering_matrix[sink_index,1]+=ray_component[1]*loss1
        scattering_matrix[sink_index,2]+=ray_component[2]*loss1

@cuda.jit
def scatteringkernaltest(problem_size,network_index,point_information,scattering_matrix,wavelength):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    # noinspection PyTypeChecker
    ray_component=cuda.local.array(shape=(3),dtype=np.complex64)
    sink_index=network_index[cu_ray_num,-1]-1-problem_size[0]
    #print(cu_ray_num,sink_index)
    while (i < (network_index.shape[1]-1)):
        if i==0:
            lengths=float(0)
            if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']
                ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']
                ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']

            else:
                source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability']/point_information[network_index[cu_ray_num,i]-1]['permittivity'])
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=complex64)
                calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                ray_component[0]=ray_component[0]/source_impedance
                ray_component[1]=ray_component[1]/source_impedance
                ray_component[2]=ray_component[2]/source_impedance

        if (network_index[cu_ray_num,i+1]!=0):
            #sink_index=network_index[cu_ray_num,i]-1-problem_size[0]
            temp=lengths
            ray_component,lengths=pointlink(ray_component,
                          point_information[network_index[cu_ray_num,i]-1],
                          point_information[network_index[cu_ray_num,i+1]-1],
                          lengths,
                          wavelength)
            if (temp==lengths):
                print('error',network_index[cu_ray_num,i],lengths)


            #convert field amplitudes to tangential surface currents
            if (i<network_index.shape[1]-1) and (network_index[cu_ray_num,i+2]!=0):
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=complex64)
                outgoing_dir[0]=point_information[network_index[cu_ray_num,i+1]]['nx']
                outgoing_dir[1]=point_information[network_index[cu_ray_num,i+1]]['ny']
                outgoing_dir[2]=point_information[network_index[cu_ray_num,i+1]]['nz']
                ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i+1]],outgoing_dir)
                ray_component[0]=ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex']
                ray_component[1]=ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey']
                ray_component[2]=ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez']
            else:
                ray_component[0]=ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex']
                ray_component[1]=ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey']
                ray_component[2]=ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez']
        else:
            break

        i+=1

    wave_vector=(2.0*np.pi)/wavelength[0]
    #loss1=cuda.local.array(shape=(1),dtype=complex64)
    if (lengths!=0.0) or (lengths!=cmath.inf):
        #print(cu_ray_num,lengths[0],abs(ray_component[2]))
        loss1=complex(0,0)
        loss1=cmath.exp(1j*wave_vector*lengths)
        loss1=loss1*(wavelength[0]/(4*(cmath.pi)*(lengths)))
        scattering_matrix[cu_ray_num,0]=ray_component[0]*loss1
        scattering_matrix[cu_ray_num,1]=ray_component[1]*loss1
        scattering_matrix[cu_ray_num,2]=ray_component[2]*loss1

@cuda.jit
def scatteringkernalv3(problem_size,network_index,point_information,scattering_matrix,scattering_coefficient,wavelength):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    flag=0
    # noinspection PyTypeChecker
    ray_component=cuda.local.array(shape=(3),dtype=np.complex128)
    for sink_test in range(1,network_index.shape[1]):
        if (network_index[cu_ray_num,sink_test]==0):
            if (flag==0):
                flag=1
                sink_index=network_index[cu_ray_num,sink_test-1]-1-problem_size[0]
        elif (sink_test==network_index.shape[1]-1):
            if (flag==0):
                flag=1
                sink_index=network_index[cu_ray_num,sink_test]-1-problem_size[0]

    if flag==0:
        print('error',cu_ray_num,sink_index)
          #  print(cu_ray_num,sink_index)
    #else:
    #    sink_index=network_index[cu_ray_num,-1]-1-problem_size[0]
    #print(cu_ray_num,network_index[cu_ray_num,1]-1-problem_size[0],network_index[cu_ray_num,2]-1-problem_size[0],sink_index)
    while (i < (network_index.shape[1]-1)):
        #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
        if i==0:
            lengths=float(0)
            #lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
            if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']*scattering_coefficient[0]
                ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']*scattering_coefficient[0]
                ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']*scattering_coefficient[0]

            else:
                source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability'].real/point_information[network_index[cu_ray_num,i]-1]['permittivity'].real).real
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex128)
                outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                ray_component[0]=(ray_component[0]/source_impedance)*scattering_coefficient[0]
                ray_component[1]=(ray_component[1]/source_impedance)*scattering_coefficient[0]
                ray_component[2]=(ray_component[2]/source_impedance)*scattering_coefficient[0]
        elif i!=0:
            # noinspection PyTypeChecker
            normal = cuda.local.array(shape=(3), dtype=np.complex128)
            normal[0]=point_information[network_index[cu_ray_num,i]-1]['nx']
            normal[1]=point_information[network_index[cu_ray_num,i]-1]['ny']
            normal[2]=point_information[network_index[cu_ray_num,i]-1]['nz']
            ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]-1],normal)

        if (network_index[cu_ray_num,i+1]!=0):
            #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
            #     #convert source point field to ray
            # noinspection PyTypeChecker
            outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex128)
            outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
            ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]-1],outgoing_dir)
            ray_component[0]=(ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex'])*scattering_coefficient[0]
            ray_component[1]=(ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey'])*scattering_coefficient[0]
            ray_component[2]=(ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez'])*scattering_coefficient[0]
            lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)

        i=i+1

    wave_vector=(2.0*cmath.pi)/wavelength[0]
    #loss1=cuda.local.array(shape=(1),dtype=complex64)
    if (lengths!=0.0) or (lengths!=cmath.inf):
        #print(cu_ray_num,lengths[0])
        loss1=complex(0,0)
        loss1=cmath.exp(1j*wave_vector*lengths)
        loss1=loss1*(wavelength[0]/(4*(cmath.pi)*(lengths)))
        scattering_matrix[sink_index,0]=scattering_matrix[sink_index,0]+(ray_component[0]*loss1)
        scattering_matrix[sink_index,1]=scattering_matrix[sink_index,1]+(ray_component[1]*loss1)
        scattering_matrix[sink_index,2]=scattering_matrix[sink_index,2]+(ray_component[2]*loss1)

@cuda.jit
def scatteringkernalv4(problem_size,network_index,point_information,scattering_matrix,scattering_coefficient,wavelength,target_index):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    flag=0
    source_index=target_index[cu_ray_num,0]-1
    sink_index=target_index[cu_ray_num,1]-1-problem_size[0]
    # noinspection PyTypeChecker
    ray_component=cuda.local.array(shape=(3),dtype=np.complex128)
    # for sink_test in range(1,network_index.shape[1]):
    #     if (network_index[cu_ray_num,sink_test]==0):
    #         if (flag==0):
    #             flag=1
    #             sink_index=network_index[cu_ray_num,sink_test-1]-1-problem_size[0]
    #     elif (sink_test==network_index.shape[1]-1):
    #         if (flag==0):
    #             flag=1
    #             sink_index=network_index[cu_ray_num,sink_test]-1-problem_size[0]

    # if flag==0:
    #     print('error',cu_ray_num,sink_index)
    #       #  print(cu_ray_num,sink_index)
    #else:
    #    sink_index=network_index[cu_ray_num,-1]-1-problem_size[0]
    #print(cu_ray_num,network_index[cu_ray_num,1]-1-problem_size[0],network_index[cu_ray_num,2]-1-problem_size[0],sink_index)
    while (i < (network_index.shape[1]-1)):
        #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
        if i==0:
            lengths=float(0)
            #lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
            if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']*scattering_coefficient[0]
                ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']*scattering_coefficient[0]
                ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']*scattering_coefficient[0]

            else:
                source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability'].real/point_information[network_index[cu_ray_num,i]-1]['permittivity'].real).real
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex128)
                outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                ray_component[0]=(ray_component[0]/source_impedance)*scattering_coefficient[0]
                ray_component[1]=(ray_component[1]/source_impedance)*scattering_coefficient[0]
                ray_component[2]=(ray_component[2]/source_impedance)*scattering_coefficient[0]
        #elif i!=0:
            #normal = cuda.local.array(shape=(3), dtype=np.complex128)
            #normal[0]=point_information[network_index[cu_ray_num,i]-1]['nx']
            #normal[1]=point_information[network_index[cu_ray_num,i]-1]['ny']
            #normal[2]=point_information[network_index[cu_ray_num,i]-1]['nz']
            #ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]-1],normal)

        if (network_index[cu_ray_num,i+1]!=0):
        #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
        #     #convert source point field to ray
              #outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex128)
              #outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
              #ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]-1],outgoing_dir)
              #ray_component[0]=(ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex'])*scattering_coefficient[0]
              #ray_component[1]=(ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey'])*scattering_coefficient[0]
              #ray_component[2]=(ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez'])*scattering_coefficient[0]
              lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)

        i=i+1

    wave_vector=(2.0*cmath.pi)/wavelength[0]
    #loss1=cuda.local.array(shape=(1),dtype=complex64)
    if (lengths!=0.0) or (lengths!=cmath.inf):
        #print(cu_ray_num,lengths[0])
        loss1=complex(0,0)
        loss1=cmath.exp(1j*wave_vector*lengths)
        loss1=loss1*(wavelength[0]/(4*(cmath.pi)*(lengths)))
        scattering_matrix[sink_index,0]=scattering_matrix[sink_index,0]+(ray_component[0]*loss1)
        scattering_matrix[sink_index,1]=scattering_matrix[sink_index,1]+(ray_component[1]*loss1)
        scattering_matrix[sink_index,2]=scattering_matrix[sink_index,2]+(ray_component[2]*loss1)

    cuda.syncthreads()

@cuda.jit
def scatteringkernaltest(problem_size,network_index,point_information,scattering_matrix,scattering_coefficient,wavelength):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    flag=0
    # noinspection PyTypeChecker
    ray_component=cuda.local.array(shape=(3),dtype=np.complex128)
    for sink_test in range(1,network_index.shape[1]):
        if (network_index[cu_ray_num,sink_test]==0):
            if (flag==0):
                flag=1
                sink_index=network_index[cu_ray_num,sink_test-1]-1-problem_size[0]
        elif (sink_test==network_index.shape[1]-1):
            if (flag==0):
                flag=1
                sink_index=network_index[cu_ray_num,sink_test]-1-problem_size[0]

    if flag==0:
        print('error',cu_ray_num,sink_index)

    scattering_matrix[cu_ray_num]=complex(sink_index)


@cuda.jit
def polaranddistance(network_index,point_information,polar_coefficients,distances):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    if (cu_ray_num<network_index.shape[0]):
        # noinspection PyTypeChecker
        ray_component=cuda.local.array(shape=(3),dtype=np.complex64)
        i = 0 # emulate a C-style for-loop, exposing the idx increment logic
        #ray_components[cu_ray_num,:]=0.0
        #print(scattering_matrix.shape[0],scattering_matrix.shape[1])
        while (i < (network_index.shape[1]-1)):
            #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
            if i==0:
                lengths=float(0)
                lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
                if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                    ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']
                    ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']
                    ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']

                else:
                    source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability'].real/point_information[network_index[cu_ray_num,i]-1]['permittivity'].real).real
                    # noinspection PyTypeChecker
                    outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                    outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                    ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                    ray_component[0]=ray_component[0]/source_impedance
                    ray_component[1]=ray_component[1]/source_impedance
                    ray_component[2]=ray_component[2]/source_impedance
            elif i!=0:
                # noinspection PyTypeChecker
                normal = cuda.local.array(shape=(3), dtype=np.complex64)
                normal[0]=point_information[network_index[cu_ray_num,i]-1]['nx']
                normal[1]=point_information[network_index[cu_ray_num,i]-1]['ny']
                normal[2]=point_information[network_index[cu_ray_num,i]-1]['nz']
                ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]],normal)
                lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)

            if (network_index[cu_ray_num,i+1]!=0):
                #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
                #     #convert source point field to ray
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]-1],outgoing_dir)

                ray_component[0]=ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex']
                ray_component[1]=ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey']
                ray_component[2]=ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez']


            i=i+1

        polar_coefficients[cu_ray_num,0]=ray_component[0]
        polar_coefficients[cu_ray_num,1]=ray_component[1]
        polar_coefficients[cu_ray_num,2]=ray_component[2]
        distances[cu_ray_num]=lengths

@cuda.jit
def freqdomainkernal(network_index,point_information,source_sink_index,wavelength,scattering_network):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    if (cu_ray_num<network_index.shape[0]):
        # noinspection PyTypeChecker
        ray_component=cuda.local.array(shape=(3),dtype=np.complex128)
        #ray_components[cu_ray_num,:]=0.0
        #print(scattering_matrix.shape[0],scattering_matrix.shape[1])
        for i in range(network_index.shape[1]-1):
            #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
            if i==0:
                lengths=float(0)
                lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
                if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                    ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']
                    ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']
                    ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']

                else:
                    source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability'].real/point_information[network_index[cu_ray_num,i]-1]['permittivity'].real).real
                    # noinspection PyTypeChecker
                    outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                    outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                    ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                    ray_component[0]=ray_component[0]/source_impedance
                    ray_component[1]=ray_component[1]/source_impedance
                    ray_component[2]=ray_component[2]/source_impedance

            elif i!=0:
                # noinspection PyTypeChecker
                normal = cuda.local.array(shape=(3), dtype=np.complex64)
                normal[0]=point_information[network_index[cu_ray_num,i]-1]['nx']
                normal[1]=point_information[network_index[cu_ray_num,i]-1]['ny']
                normal[2]=point_information[network_index[cu_ray_num,i]-1]['nz']
                ray_component=sourcelaunchtransformGPU(ray_component,normal)
                lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)

            if (network_index[cu_ray_num,i+1]!=0):
                #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
                #     #convert source point field to ray
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component=sourcelaunchtransformGPU(ray_component,outgoing_dir)

                ray_component[0]=ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex']
                ray_component[1]=ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey']
                ray_component[2]=ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez']
                scatter_index=i




        # print(cu_ray_num,source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1])
        wave_vector=(2.0*cmath.pi)/wavelength
        #scatter_coefficient=(1/(4*cmath.pi))**(complex(scatter_index))
        if (scatter_index==0):
            loss=cmath.exp(lengths*wave_vector*1j)*(wavelength/(4*cmath.pi*lengths))
        elif(scatter_index==1):
            loss=cmath.exp(lengths*wave_vector*1j)*(wavelength/(4*cmath.pi*lengths))
        elif(scatter_index==2):
            loss=cmath.exp(lengths*wave_vector*1j)*(wavelength/(4*cmath.pi*lengths))

        ray_component[0]*=loss
        ray_component[1]*=loss
        ray_component[2]*=loss
        #print(ray_component[0].real,ray_component[1].real,ray_component[2].real)
        #add real components
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],0,:],0,ray_component[0].real)
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],1,:],0,ray_component[1].real)
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],2,:],0,ray_component[2].real)
        #add imaginary components
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],0,:],1,ray_component[0].imag)
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],1,:],1,ray_component[1].imag)
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],2,:],1,ray_component[2].imag)

@cuda.jit
def freqdomainisokernal(network_index,point_information,source_sink_index,wavelength,scattering_network):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    if (cu_ray_num<network_index.shape[0]):
        # noinspection PyTypeChecker
        ray_component=cuda.local.array(shape=(3),dtype=np.complex128)
        for i in range(network_index.shape[1]-1):

            #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
            if i==0:
                lengths=float(0)
                lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)

            elif i!=0:
                if (network_index[cu_ray_num,i+1]!=0):
                    lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)



        #print(network_index[cu_ray_num,0],network_index[cu_ray_num,1],network_index[cu_ray_num,2],lengths)
        # print(cu_ray_num,source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1])
        wave_vector=(2.0*cmath.pi)/wavelength
        #scatter_coefficient=(1/(4*cmath.pi))**(complex(scatter_index))
        loss=cmath.exp(lengths*wave_vector*1j)*(wavelength/(4*cmath.pi*lengths))
        ray_component[0]=loss
        #ray_component[1]=loss
        #ray_component[2]=loss
        #print(ray_component[0].real,ray_component[1].real,ray_component[2].real)
        #print(len(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:]))
        #add real components
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],0,:],0,ray_component[0].real)
        #cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],1,:],0,ray_component[1].real)
        #cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],2,:],0,ray_component[2].real)
        #scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],0]+=ray_component[0]
        #add imaginary components
        cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],0,:],1,ray_component[0].imag)
        #cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],1,:],1,ray_component[1].imag)
        #cuda.atomic.add(scattering_network[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],2,:],1,ray_component[2].imag)

@cuda.jit
def timedomainkernal(full_index,point_information,source_sink_index,wavelength,excitation,sampling_freq,arrival_time,wake_time,time_map):
    #this kernal is planned to calculate the time domain response for a given input signal
    #for flexibility this should probably start out as smn port pairs
     cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
     if (cu_ray_num<full_index.shape[0]):
        # noinspection PyTypeChecker
        ray_component=cuda.local.array(shape=(3),dtype=np.float64)
        i = 0 # emulate a C-style for-loop, exposing the idx increment logic
        #ray_components[cu_ray_num,:]=0.0
        #print(scattering_matrix.shape[0],scattering_matrix.shape[1])
        while (i < (full_index.shape[1]-1)):
            #print(i,full_index[cu_ray_num,i],full_index[cu_ray_num,i+1])
            if i==0:
                lengths=float(0)
                time_delay=float(0)
                lengths=calc_sep(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],lengths)
                #print(cu_ray_num,'start ',full_index[cu_ray_num,i]-1,' end ',full_index[cu_ray_num,i+1]-1,lengths,'m')
                time_delay+=(point_information[full_index[cu_ray_num,i]-1]['ex'].imag)
                if point_information[full_index[cu_ray_num,i]-1]['Electric']:
                    ray_component[0]=point_information[full_index[cu_ray_num,i]-1]['ex'].real
                    ray_component[1]=point_information[full_index[cu_ray_num,i]-1]['ey'].real
                    ray_component[2]=point_information[full_index[cu_ray_num,i]-1]['ez'].real

                else:
                    source_impedance=cmath.sqrt(point_information[full_index[cu_ray_num,i]-1]['permeability'].real/point_information[full_index[cu_ray_num,i]-1]['permittivity'].real).real
                    # noinspection PyTypeChecker
                    outgoing_dir = cuda.local.array(shape=(3), dtype=np.float64)
                    outgoing_dir=calc_dv(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],outgoing_dir)
                    ray_component[0],ray_component[1],ray_component[2]=cross(point_information[full_index[cu_ray_num,i]-1]['ex'].real,point_information[full_index[cu_ray_num,i]-1]['ey'].real,point_information[full_index[cu_ray_num,i]-1]['ez'].real,outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                    ray_component[0]=ray_component[0]/source_impedance
                    ray_component[1]=ray_component[1]/source_impedance
                    ray_component[2]=ray_component[2]/source_impedance
            elif i!=0:
                # noinspection PyTypeChecker
                normal = cuda.local.array(shape=(3), dtype=np.float64)
                normal[0]=point_information[full_index[cu_ray_num,i]-1]['nx']
                normal[1]=point_information[full_index[cu_ray_num,i]-1]['ny']
                normal[2]=point_information[full_index[cu_ray_num,i]-1]['nz']
                ray_component=sourcelaunchtransformGPU(ray_component,normal)
                lengths=calc_sep(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],lengths)
                #print(cu_ray_num,lengths,'m')

            if (full_index[cu_ray_num,i+1]!=0):
                #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
                #     #convert source point field to ray
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.float64)
                outgoing_dir=calc_dv(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component=sourcelaunchtransformGPU(ray_component,outgoing_dir)
                #in time domain, the real part is the magnitude, and the imaginary part is the time delay

                ray_component[0]=ray_component[0]*point_information[full_index[cu_ray_num,i+1]-1]['ex'].real
                ray_component[1]=ray_component[1]*point_information[full_index[cu_ray_num,i+1]-1]['ey'].real
                ray_component[2]=ray_component[2]*point_information[full_index[cu_ray_num,i+1]-1]['ez'].real
                time_delay+=(point_information[full_index[cu_ray_num,i+1]-1]['ex'].imag)
                scatter_index=i


            i=i+1

        # print(cu_ray_num,source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1])
        wave_vector=(2.0*cmath.pi)/wavelength[0]
        #scatter_coefficient=(1/(4*cmath.pi))**(complex(scatter_index))
        loss=(wavelength[0]/(4*cmath.pi*lengths))

        ray_component[0]*=loss
        ray_component[1]*=loss
        ray_component[2]*=loss
        arrival_time[cu_ray_num]=(lengths/scipy.constants.c)+time_delay
        #print(arrival_time[cu_ray_num] * 1e9)

        cuda.atomic.min(wake_time,0,arrival_time[cu_ray_num])
        #pause here to sync threads to make sure wake_time is populated
        cuda.syncthreads()

        #wake_time=min(arrival_time) the obvious idea didnt work, so will need to think of a way to access the minimum value
        #print(wake_time[0]*1e9)
        #wake_time=0.0


        #print(ray_component[0].real,ray_component[1].real,ray_component[2].real)
        time_offset=arrival_time[cu_ray_num]-wake_time[0]
        #calculate begin index, then add the excitation signal
        time_index=int(0)
        time_sep=(1.0/sampling_freq[0])
        time_index=(time_offset//time_sep)
        #print(cu_ray_num,time_index)
        if (time_index+excitation.shape[0])<=time_map.shape[2]:
            index=0
            #print(cu_ray_num,source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1])
            while index < excitation.shape[0]:
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,0],time_index+index,excitation[index]*ray_component[0])
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,1],time_index+index,excitation[index]*ray_component[1])
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,2],time_index+index,excitation[index]*ray_component[2])
                index+=1
        else:
            index=0
            while index < (time_map.shape[2]-time_index):
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,0],time_index+index,excitation[index]*ray_component[0])
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,1],time_index+index,excitation[index]*ray_component[1])
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,2],time_index+index,excitation[index]*ray_component[2])
                print(index)
                index+=1


@cuda.jit(device=True)
def xyztothetaphivectors(ray_component,point_information):
    #assuming a prime vector along the z axis
    # noinspection PyTypeChecker
    thetaphi=cuda.local.array(shape=(2),dtype=np.complex128)
    # noinspection PyTypeChecker
    prime = cuda.local.array(shape=(3), dtype=np.complex128)
    # noinspection PyTypeChecker
    theta_vector = cuda.local.array(shape=(3), dtype=np.complex128)
    # noinspection PyTypeChecker
    phi_vector=cuda.local.array(shape=(3), dtype=np.complex128)
    # noinspection PyTypeChecker
    normal = cuda.local.array(shape=(3), dtype=np.complex128)
    normal[0]=point_information['nx']
    normal[1]=point_information['ny']
    normal[2]=point_information['nz']
    prime[:]=0.0
    prime[2]=1.0
    theta_vector=dot_vec(prime,normal)
    phi_vector=cross_norm(theta_vector,normal,phi_vector)
    thetaphi[0]=ray_component[0]*theta_vector[0]+ray_component[1]*theta_vector[1]+ray_component[2]*theta_vector[2]
    thetaphi[1]=ray_component[0]*phi_vector[0]+ray_component[1]*phi_vector[1]+ray_component[2]*phi_vector[2]
    return thetaphi

@cuda.jit
def timedomainthetaphi(full_index,point_information,source_sink_index,wavelength,excitation,sampling_freq,arrival_time,wake_time,time_map):
    #this kernal is planned to calculate the time domain response for a given input signal
    #for flexibility this should probably start out as smn port pairs
     cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
     if (cu_ray_num<full_index.shape[0]):
        # noinspection PyTypeChecker
        ray_component=cuda.local.array(shape=(3),dtype=np.complex128)
        i = 0 # emulate a C-style for-loop, exposing the idx increment logic
        #ray_components[cu_ray_num,:]=0.0
        #print(scattering_matrix.shape[0],scattering_matrix.shape[1])
        while (i < (full_index.shape[1]-1)):
            #print(i,full_index[cu_ray_num,i],full_index[cu_ray_num,i+1])
            if i==0:
                lengths=float(0)
                time_delay=float(0)
                lengths=calc_sep(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],lengths)
                #print(cu_ray_num,'start ',full_index[cu_ray_num,i]-1,' end ',full_index[cu_ray_num,i+1]-1,lengths,'m')
                time_delay+=(point_information[full_index[cu_ray_num,i]-1]['ex'].imag)
                if point_information[full_index[cu_ray_num,i]-1]['Electric']:
                    ray_component[0]=point_information[full_index[cu_ray_num,i]-1]['ex']
                    ray_component[1]=point_information[full_index[cu_ray_num,i]-1]['ey']
                    ray_component[2]=point_information[full_index[cu_ray_num,i]-1]['ez']

                else:
                    source_impedance=cmath.sqrt(point_information[full_index[cu_ray_num,i]-1]['permeability'].real/point_information[full_index[cu_ray_num,i]-1]['permittivity'].real).real
                    # noinspection PyTypeChecker
                    outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                    outgoing_dir=calc_dv(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],outgoing_dir)
                    ray_component[0],ray_component[1],ray_component[2]=cross(point_information[full_index[cu_ray_num,i]-1]['ex'],point_information[full_index[cu_ray_num,i]-1]['ey'],point_information[full_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                    ray_component[0]=ray_component[0]/source_impedance
                    ray_component[1]=ray_component[1]/source_impedance
                    ray_component[2]=ray_component[2]/source_impedance
            elif i!=0:
                # noinspection PyTypeChecker
                normal = cuda.local.array(shape=(3), dtype=np.complex64)
                normal[0]=point_information[full_index[cu_ray_num,i]-1]['nx']
                normal[1]=point_information[full_index[cu_ray_num,i]-1]['ny']
                normal[2]=point_information[full_index[cu_ray_num,i]-1]['nz']
                ray_component=sourcelaunchtransformGPU(ray_component,point_information[full_index[cu_ray_num,i]],normal)
                lengths=calc_sep(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],lengths)
                #print(cu_ray_num,lengths,'m')

            if (full_index[cu_ray_num,i+1]!=0):
                #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
                #     #convert source point field to ray
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                outgoing_dir=calc_dv(point_information[full_index[cu_ray_num,i]-1],point_information[full_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component=sourcelaunchtransformGPU(ray_component,point_information[full_index[cu_ray_num,i]-1],outgoing_dir)
                #in time domain, the real part is the magnitude, and the imaginary part is the time delay

                ray_component[0]=ray_component[0]*point_information[full_index[cu_ray_num,i+1]-1]['ex'].real
                ray_component[1]=ray_component[1]*point_information[full_index[cu_ray_num,i+1]-1]['ey'].real
                ray_component[2]=ray_component[2]*point_information[full_index[cu_ray_num,i+1]-1]['ez'].real
                time_delay+=(point_information[full_index[cu_ray_num,i+1]-1]['ex'].imag)
                scatter_index=i


            i=i+1

        # print(cu_ray_num,source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1])
        wave_vector=(2.0*cmath.pi)/wavelength[0]
        #scatter_coefficient=(1/(4*cmath.pi))**(complex(scatter_index))
        loss=(wavelength[0]/(4*cmath.pi*lengths))

        ray_component[0]*=loss
        ray_component[1]*=loss
        ray_component[2]*=loss
        arrival_time[cu_ray_num]=(lengths/scipy.constants.c)+time_delay
        #print(arrival_time[cu_ray_num] * 1e9)

        cuda.atomic.min(wake_time,0,arrival_time[cu_ray_num])
        #pause here to sync threads to make sure wake_time is populated
        cuda.syncthreads()

        #wake_time=min(arrival_time) the obvious idea didnt work, so will need to think of a way to access the minimum value
        #print(wake_time[0]*1e9)
        #wake_time=0.0
        thetaphi=xyztothetaphivectors(ray_component,point_information[full_index[cu_ray_num,i]-1])

        #print(ray_component[0].real,ray_component[1].real,ray_component[2].real)
        time_offset=arrival_time[cu_ray_num]-wake_time[0]
        #calculate begin index, then add the excitation signal
        time_index=int(0)
        time_sep=(1.0/sampling_freq[0])
        time_index=(time_offset//time_sep)
        #print(cu_ray_num,time_index)
        if (time_index+excitation.shape[0])<=time_map.shape[2]:
            index=0
            #print(cu_ray_num,source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1])
            while index < excitation.shape[0]:
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,0],time_index+index,excitation[index]*abs(ray_component[0]))
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,1],time_index+index,excitation[index]*abs(ray_component[1]))
                #cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,2],time_index+index,excitation[index]*abs(ray_component[2]))
                index+=1
        else:
            index=0
            while index < (time_map.shape[2]-time_index):
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,0],time_index+index,excitation[index]*abs(ray_component[0]))
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,1],time_index+index,excitation[index]*abs(ray_component[1]))
                cuda.atomic.add(time_map[source_sink_index[cu_ray_num,0],source_sink_index[cu_ray_num,1],:,2],time_index+index,excitation[index]*abs(ray_component[2]))
                print(index)
                index+=1

@cuda.jit
def pathlength(network_index,point_information,distances):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    #print(cu_ray_num,sink_index)
    while (i < (network_index.shape[1]-1)):
        if i==0:
            lengths=float(0)
            lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
        elif (i!=0) and (network_index[cu_ray_num,i+1]!=0) and (i < (network_index.shape[1]-1)):
            temp=lengths
            lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
            if (temp==lengths):
                print('error',network_index[cu_ray_num,i],lengths)


        i+=1

    distances[cu_ray_num]=lengths

@cuda.jit
def polarmixing(network_index,point_information,polar_coefficients):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    # noinspection PyTypeChecker
    ray_component=cuda.local.array(shape=(3),dtype=np.complex64)
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    #print(scattering_matrix.shape[0],scattering_matrix.shape[1])
    while (i < (network_index.shape[1]-1)):
        #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
        if i==0:
            #lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
            if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']
                ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']
                ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']

            else:
                source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability'].real/point_information[network_index[cu_ray_num,i]-1]['permittivity'].real).real
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                ray_component[0]=ray_component[0]/source_impedance
                ray_component[1]=ray_component[1]/source_impedance
                ray_component[2]=ray_component[2]/source_impedance

        elif i!=0:
            # noinspection PyTypeChecker
            normal = cuda.local.array(shape=(3), dtype=np.complex64)
            normal[0]=point_information[network_index[cu_ray_num,i]-1]['nx']
            normal[1]=point_information[network_index[cu_ray_num,i]-1]['ny']
            normal[2]=point_information[network_index[cu_ray_num,i]-1]['nz']
            ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]],normal)

        if (network_index[cu_ray_num,i+1]!=0):
            #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
            #     #convert source point field to ray
            # noinspection PyTypeChecker
            outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
            outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
            ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]-1],outgoing_dir)

            #     #project ray field onto scatter surface if it is a scatter point
            #     #if (i!=0) and (network_index[cu_ray_num,i+1]!=0) and (i < (network_index.shape[1]-1)):
            #         #print('scatter point',i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1],abs(ray_component[0]),abs(ray_component[1]),abs(ray_component[2]))
            #     #    outgoing_dir = cuda.local.array(shape=(3), dtype=complex64)
            #     #    outgoing_dir[0]=point_information[network_index[cu_ray_num,i+1]-1]['nx']
            #     #    outgoing_dir[1]=point_information[network_index[cu_ray_num,i+1]-1]['ny']
            #     #    outgoing_dir[2]=point_information[network_index[cu_ray_num,i+1]-1]['nz']
            #     #    ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i+1]],outgoing_dir)
            #     #else:
            #         #print('end point',network_index[cu_ray_num,i],network_index[cu_ray_num,i+1],abs(ray_component[0]),abs(ray_component[1]),abs(ray_component[2]))

            ray_component[0]=ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex']
            ray_component[1]=ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey']
            ray_component[2]=ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez']

        i=i+1

    polar_coefficients[cu_ray_num,0]=ray_component[0]
    polar_coefficients[cu_ray_num,1]=ray_component[1]
    polar_coefficients[cu_ray_num,2]=ray_component[2]


@cuda.jit
def pp(network_index,point_information,polar_coefficients,paths):

    cu_ray_num = cuda.grid(1) # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
                        #           threadIdx.y + ( blockIdx.y * blockDim.y )
     #margin=1e-5
    # noinspection PyTypeChecker
    ray_component=cuda.local.array(shape=(3),dtype=np.complex64)
    i = 0 # emulate a C-style for-loop, exposing the idx increment logic
    #ray_components[cu_ray_num,:]=0.0
    #print(scattering_matrix.shape[0],scattering_matrix.shape[1])
    while (i < (network_index.shape[1]-1)):
        #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
        if i==0:
            #lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
            if point_information[network_index[cu_ray_num,i]-1]['Electric']:
                ray_component[0]=point_information[network_index[cu_ray_num,i]-1]['ex']
                ray_component[1]=point_information[network_index[cu_ray_num,i]-1]['ey']
                ray_component[2]=point_information[network_index[cu_ray_num,i]-1]['ez']

            else:
                source_impedance=cmath.sqrt(point_information[network_index[cu_ray_num,i]-1]['permeability'].real/point_information[network_index[cu_ray_num,i]-1]['permittivity'].real).real
                # noinspection PyTypeChecker
                outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
                outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
                ray_component[0],ray_component[1],ray_component[2]=cross(point_information[network_index[cu_ray_num,i]-1]['ex'],point_information[network_index[cu_ray_num,i]-1]['ey'],point_information[network_index[cu_ray_num,i]-1]['ez'],outgoing_dir[0],outgoing_dir[1],outgoing_dir[2])
                ray_component[0]=ray_component[0]/source_impedance
                ray_component[1]=ray_component[1]/source_impedance
                ray_component[2]=ray_component[2]/source_impedance
            lengths=float(0)
            lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)
        elif i!=0:
            # noinspection PyTypeChecker
            normal = cuda.local.array(shape=(3), dtype=np.complex64)
            normal[0]=point_information[network_index[cu_ray_num,i]-1]['nx']
            normal[1]=point_information[network_index[cu_ray_num,i]-1]['ny']
            normal[2]=point_information[network_index[cu_ray_num,i]-1]['nz']
            ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]],normal)
            #temp=lengths
            lengths=calc_sep(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],lengths)

        if (network_index[cu_ray_num,i+1]!=0):
            #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
            #     #convert source point field to ray
            # noinspection PyTypeChecker
            outgoing_dir = cuda.local.array(shape=(3), dtype=np.complex64)
            outgoing_dir=calc_dv(point_information[network_index[cu_ray_num,i]-1],point_information[network_index[cu_ray_num,i+1]-1],outgoing_dir)
            ray_component=sourcelaunchtransformGPU(ray_component,point_information[network_index[cu_ray_num,i]-1],outgoing_dir)

            ray_component[0]=ray_component[0]*point_information[network_index[cu_ray_num,i+1]-1]['ex']
            ray_component[1]=ray_component[1]*point_information[network_index[cu_ray_num,i+1]-1]['ey']
            ray_component[2]=ray_component[2]*point_information[network_index[cu_ray_num,i+1]-1]['ez']

        i=i+1

    polar_coefficients[cu_ray_num,0]=ray_component[0]
    polar_coefficients[cu_ray_num,1]=ray_component[1]
    polar_coefficients[cu_ray_num,2]=ray_component[2]
    paths[cu_ray_num]=lengths

@cuda.jit
def EMGPUSummation(scattering_matrix,full_rays,depth_slice,source_index):
    source_num=source_index[1]
    sink_index = cuda.grid(1)
    for array_index in range(full_rays.shape[0]):
        if (depth_slice[array_index,0]-1==source_index[0]):
            if (depth_slice[array_index,1]-source_num-1==sink_index):
                scattering_matrix[sink_index,0]=scattering_matrix[sink_index,0]+full_rays[array_index,0]
                scattering_matrix[sink_index,1]=scattering_matrix[sink_index,1]+full_rays[array_index,1]
                scattering_matrix[sink_index,2]=scattering_matrix[sink_index,2]+full_rays[array_index,2]


#def soundingTDZC(N,R):
#    """
#    Generates a time domain Zadoff-Chu sequence of length N, and parameter R for use in channel sounding.
#    The main benefit is that when shifted by a symbol, it has a very low cross corellation.
#    Parameters
#    ----------
#    N : TYPE
#        DESCRIPTION.
#    R : TYPE
#        DESCRIPTION.
#
#    Returns
#    -------
#    TYPE
#        DESCRIPTION.#
#
#    """
#    #complex frequency domain sequence
#    y=np.zeros((N),dtype=np.complex64)
#    y=np.exp(-1j*R*np.pi())*(0:N-1).*((0:N-1)+bitand(N,1)+2*Q)/N)
#    return

def EMGPUJointPathLengthandPolar(source_num,sink_num,full_index,point_information):
    """
    wrapper for the GPU EM processer, outputting the resultant ray components as lengths, allowing for the whole thing to be sorted again.
    At present, the indexing only supports processing the rays for line of sight and single bounce, but that will be sorted quite quickly
    Parameters
    ----------
    full_index : int array
        index of all successful rays
    point_information : TYPE
        DESCRIPTION.
    wavelength : TYPE
        DESCRIPTION.

    Returns
    -------
    resultant_rays : TYPE
        DESCRIPTION.

    """
    #cuda.select_device(0)
    #network_index,point_information,ray_components
    ray_num=full_index.shape[0]
    threads_in_block=256
    max_blocks=65535
    maximum_chunk_size=2**8
    path_lengths=np.zeros((ray_num),dtype=np.float32)
    polar_coefficients=np.ones((ray_num,3),dtype=np.complex64)
    d_point_information = cuda.device_array(point_information.shape[0], dtype=base.scattering_t)
    d_point_information=cuda.to_device(point_information)
    #divide in terms of a block for each source, then

    d_full_index = cuda.device_array((full_index.shape[0],full_index.shape[1]), dtype=np.int64)
    #d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
    #d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
    paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
    polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
    d_full_index = cuda.to_device(full_index)
    # Here, we choose the granularity of the threading on our device. We want
    # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
    # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

    grids=(math.ceil(full_index.shape[0]/threads_in_block))
    threads = threads_in_block
    #print(grids,' blocks, ',threads,' threads')
    # Execute the kernel
    #cuda.profile_start()
    polaranddistance[grids, threads](d_full_index,d_point_information,polar_c,paths)
    #cuda.profile_stop()
    #ray_components[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
    #distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
    #first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
    #resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()
    #chunks=np.linspace(0,path_lengths.shape[0],math.ceil(path_lengths.shape[0]/maximum_chunk_size)+1,dtype=np.int32)
    #for n in range(chunks.shape[0]-1):
    #polar_coefficients=d_polar_c.copy_to_host()
    polar_coefficients=cp.asnumpy(polar_c)
    path_lengths=cp.asnumpy(paths)
    #path_lengths=d_paths.copy_to_host()
    #print('Polar Mixing Progress {:3.0f}%'.format((source_index/source_num)*100))

    return path_lengths,polar_coefficients


def EMGPUFreqDomain(source_num,sink_num,full_index,point_information,wavelength):
    """
    Wrapper for the GPU EM processer
    At present, the indexing only supports processing the rays for line of sight and single or double bounces

    Parameters
    -----------
    source_num : (int)
        the number of source points
    sink_num : (int)
        the number of sink points
    full_index : (2D numpy array of ints)
        index of all successful rays
    point_information : (custom point data type)
        the point information contains the amplitude exciation for the sources, and the positions and normal vectors for all points, together with electromangetic properties,
        however, the general assumption of this model is that there is only freespace and metal interacting.
    wavelength : (float)
        the wavelength of interest

    Returns
    -------
    scattering_network_comp : (2D numpy array, complex)
        the resultant scattering network for the provided ray paths

    """
    #cuda.select_device(0)
    #network_index,point_information,ray_components
    ray_num=full_index.shape[0]
    threads_in_block=256
    max_blocks=65535
    maximum_chunk_size=2**8
    path_lengths=np.zeros((ray_num),dtype=np.float32)
    scattering_network=np.zeros((source_num,sink_num,3,2),dtype=np.float64)
    d_scattering_network=cuda.device_array((scattering_network.shape[0],scattering_network.shape[1],scattering_network.shape[2]),dtype=np.complex64)
    d_scattering_network=cuda.to_device(scattering_network)
    d_point_information = cuda.device_array(point_information.shape[0], dtype=base.scattering_t)
    d_point_information=cuda.to_device(point_information)
    d_wavelength=cuda.device_array((1),dtype=np.complex64)
    d_wavelength=cuda.to_device(np.csingle(np.ones((1),dtype=np.complex64)*wavelength))
    #divide in terms of a block for each source, then
    depthslice,_=targettingindex(copy.deepcopy(full_index))
    depthslice[:,0]-=1
    depthslice[:,1]-=(source_num+1)
    d_target_index = cuda.device_array((depthslice.shape[0],depthslice.shape[1]), dtype=np.int64)
    d_target_index=cuda.to_device(depthslice)
    d_full_index = cuda.device_array((full_index.shape[0],full_index.shape[1]), dtype=np.int64)
    #d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
    #d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
    #paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
    #polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
    d_full_index = cuda.to_device(full_index)
    # Here, we choose the granularity of the threading on our device. We want
    # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
    # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

    grids=(math.ceil(full_index.shape[0]/threads_in_block))
    threads = threads_in_block
    #print(grids,' blocks, ',threads,' threads')
    # Execute the kernel
    #cuda.profile_start()
    freqdomainkernal[grids, threads](d_full_index,point_information,d_target_index,wavelength,d_scattering_network)
    #polaranddistance(d_full_index,d_point_information,polar_c,paths)
    #cuda.profile_stop()
    #ray_components[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
    #distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
    #first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
    #resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()
    #chunks=np.linspace(0,path_lengths.shape[0],math.ceil(path_lengths.shape[0]/maximum_chunk_size)+1,dtype=np.int32)
    #for n in range(chunks.shape[0]-1):
    #polar_coefficients=d_polar_c.copy_to_host()
    scattering_network=cp.asnumpy(d_scattering_network)
    scattering_network_comp = scattering_network.view(dtype=np.complex128)[...,0]
    #scattering_network_comp = scattering_network[:, :, :, 0] + scattering_network[:, :, :, 1] * 1j
    #path_lengths=d_paths.copy_to_host()
    #print('Polar Mixing Progress {:3.0f}%'.format((source_index/source_num)*100))

    return scattering_network_comp

def IsoGPUFreqDomain(source_num,sink_num,full_index,point_information,wavelength):
    """
    wrapper for the GPU EM processer, outputting the resultant ray components as lengths, allowing for the whole thing to be sorted again.
    At present, the indexing only supports processing the rays for line of sight and single bounce, but that will be sorted quite quickly
    Parameters
    ----------
    full_index : int array
        index of all successful rays
    point_information : TYPE
        DESCRIPTION.
    wavelength : TYPE
        DESCRIPTION.

    Returns
    -------
    resultant_rays : TYPE
        DESCRIPTION.

    """
    #cuda.select_device(0)
    #network_index,point_information,ray_components
    ray_num=full_index.shape[0]
    threads_in_block=256
    max_blocks=65535
    maximum_chunk_size=2**8
    path_lengths=np.zeros((ray_num),dtype=np.float32)
    test_d=cuda.device_array((path_lengths.shape[0]),dtype=np.float64)
    scattering_network=np.zeros((source_num,sink_num,3,2),dtype=np.float64)
    d_scattering_network=cuda.device_array((scattering_network.shape[0],scattering_network.shape[1],scattering_network.shape[2],scattering_network.shape[3]),dtype=np.float64)
    d_scattering_network=cuda.to_device(scattering_network)
    d_point_information = cuda.device_array(point_information.shape[0], dtype=base.scattering_t)
    d_point_information=cuda.to_device(point_information)
    d_wavelength=cuda.device_array((1),dtype=np.complex64)
    d_wavelength=cuda.to_device(np.csingle(np.ones((1),dtype=np.complex64)*wavelength))
    #divide in terms of a block for each source, then
    depthslice,_=targettingindex(copy.deepcopy(full_index))
    depthslice[:,0]-=1
    depthslice[:,1]-=(source_num+1)
    d_target_index = cuda.device_array((depthslice.shape[0],depthslice.shape[1]), dtype=np.int64)
    d_target_index=cuda.to_device(depthslice)
    d_full_index = cuda.device_array((full_index.shape[0],full_index.shape[1]), dtype=np.int64)
    #d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
    #d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
    #paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
    #polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
    d_full_index = cuda.to_device(full_index)
    # Here, we choose the granularity of the threading on our device. We want
    # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
    # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

    grids=(math.ceil(full_index.shape[0]/threads_in_block))
    threads = threads_in_block
    #print(grids,' blocks, ',threads,' threads')
    # Execute the kernel
    #cuda.profile_start()
    freqdomainisokernal[grids, threads](d_full_index,point_information,d_target_index,wavelength,d_scattering_network)
    #polaranddistance(d_full_index,d_point_information,polar_c,paths)
    #cuda.profile_stop()
    #ray_components[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
    #distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
    #first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
    #resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()
    #chunks=np.linspace(0,path_lengths.shape[0],math.ceil(path_lengths.shape[0]/maximum_chunk_size)+1,dtype=np.int32)
    #for n in range(chunks.shape[0]-1):
    #polar_coefficients=d_polar_c.copy_to_host()
    scattering_network=cp.asnumpy(d_scattering_network)
    scattering_network_comp = scattering_network[:,:,:,0]+scattering_network[:,:,:,1]*1j
    #path_lengths=d_paths.copy_to_host()
    #print('Polar Mixing Progress {:3.0f}%'.format((source_index/source_num)*100))

    return scattering_network_comp


def EMWrapperMerged(source_num,sink_num,point_informationv2,full_index,scattering_coefficient,wavelength):
    #diff_sources=np.size(np.unique(full_index[:,0]))
    paths,polar_coefficients=EMGPUJointPathLengthandPolar(source_num,sink_num,full_index,point_informationv2)
    #polar_coefficients=EMGPUPolarMixing(source_num,sink_num,full_index,point_informationv2)
    #ray_components=EM.EMGPUWrapper(num_sources,num_sinks,full_index,point_informationv2,wavelength)
    depthslice,scatter_index=targettingindex(full_index)
    loss=pathloss(paths,wavelength)*(np.abs(np.power(scattering_coefficient,scatter_index-1))*np.exp(-1j*np.pi))
    full_rays=loss.reshape(loss.shape[0],1)*polar_coefficients
    if full_index.shape[1]==2:
        depth_slicelos=full_index
        scatter_map2=RF.scatter_net_sortEM(source_num,sink_num,np.zeros((source_num,sink_num,3,1),dtype=np.complex64),depth_slicelos,full_rays,0)
    elif full_index.shape[1]==3:
        depth_slicelos=depthslice[scatter_index==1,:]
        depth_slicebounce=depthslice[scatter_index==2,:]
        scatter_map2=RF.scatter_net_sortEM(source_num,sink_num,np.zeros((source_num,sink_num,3,2),dtype=np.complex64),depth_slicelos,full_rays[scatter_index==1,:],0)
        scatter_map2=RF.scatter_net_sortEM(source_num,sink_num,scatter_map2,depth_slicebounce,full_rays[scatter_index==2,:],1)
    elif full_index.shape[1]==4:
        depth_slicelos=depthslice[scatter_index==1,:]
        depth_slicebounce1=depthslice[scatter_index==2,:]
        depth_slicebounce2=depthslice[scatter_index==3,:]
        scatter_map2=RF.scatter_net_sortEM(source_num,sink_num,np.zeros((source_num,sink_num,3,3),dtype=np.complex64),depth_slicelos,full_rays[scatter_index==1,:],0)
        scatter_map2=RF.scatter_net_sortEM(source_num,sink_num,scatter_map2,depth_slicebounce1,full_rays[scatter_index==2,:],1)
        scatter_map2=RF.scatter_net_sortEM(source_num,sink_num,scatter_map2,depth_slicebounce2,full_rays[scatter_index==3,:],1)

    #deallocate memory on gpu
    ctx = cuda.current_context()
    deallocs = ctx.deallocations
    deallocs.clear()
    return scatter_map2

#@njit
def time_indexing(arr, starting_num, arr_length,fill_value=0.0):
    if arr_length<arr.shape[0]:
        return np.pad(arr,(starting_num,0),'constant',constant_values=(fill_value,fill_value))[0:arr_length]
    else:
        return np.pad(arr,(starting_num,arr_length-arr.shape[0]),'constant',constant_values=(fill_value,fill_value))[0:arr_length]

@njit(cache=True, nogil=True)
def time_sortingv2(source_num,sink_num,time_steps,depthslice,loss,polar,time_map,excitation_signal):
    for sink_index in range(sink_num):
        sink_slice=time_steps[depthslice[:,1]==sink_index+source_num+1]
        temp_loss=loss[depthslice[:,1]==sink_index+source_num+1]
        temp_polar=polar[depthslice[:,1]==sink_index+source_num+1,:]
        for slice_index in range(len(sink_slice)):
            if (sink_slice[slice_index]+excitation_signal.shape[0]>time_map.shape[2]):
                end_point=time_map.shape[2]-sink_slice[slice_index]
                time_map[sink_index,0,sink_slice[slice_index]:]=time_map[sink_index,0,sink_slice[slice_index]:]+temp_polar[slice_index,0]*temp_loss[slice_index]*excitation_signal[0:end_point]
                time_map[sink_index,1,sink_slice[slice_index]:]=time_map[sink_index,1,sink_slice[slice_index]:]+temp_polar[slice_index,1]*temp_loss[slice_index]*excitation_signal[0:end_point]
                time_map[sink_index,2,sink_slice[slice_index]:]=time_map[sink_index,2,sink_slice[slice_index]:]+temp_polar[slice_index,2]*temp_loss[slice_index]*excitation_signal[0:end_point]
            else:
                time_map[sink_index,0,sink_slice[slice_index]:sink_slice[slice_index]+excitation_signal.shape[0]]=time_map[sink_index,0,sink_slice[slice_index]:sink_slice[slice_index]+excitation_signal.shape[0]]+temp_polar[slice_index,0]*temp_loss[slice_index]*excitation_signal
                time_map[sink_index,1,sink_slice[slice_index]:sink_slice[slice_index]+excitation_signal.shape[0]]=time_map[sink_index,1,sink_slice[slice_index]:sink_slice[slice_index]+excitation_signal.shape[0]]+temp_polar[slice_index,1]*temp_loss[slice_index]*excitation_signal
                time_map[sink_index,2,sink_slice[slice_index]:sink_slice[slice_index]+excitation_signal.shape[0]]=time_map[sink_index,2,sink_slice[slice_index]:sink_slice[slice_index]+excitation_signal.shape[0]]+temp_polar[slice_index,2]*temp_loss[slice_index]*excitation_signal
                #time_map[sink_index,0,:]=time_map[sink_index,0,:]+temp_polar[slice_index,0]*temp*temp_loss[slice_index]
            #time_map[sink_index,1,:]=time_map[sink_index,1,:]+temp_polar[slice_index,1]*temp*temp_loss[slice_index]
            #time_map[sink_index,2,:]=time_map[sink_index,2,:]+temp_polar[slice_index,2]*temp*temp_loss[slice_index]

    return time_map

def time_sorting(source_num,sink_num,time_steps,depthslice,loss,polar,time_map,excitation_signal):
    for sink_index in range(sink_num):
        sink_slice=time_steps[depthslice[:,1]==sink_index+source_num+1]
        temp_loss=loss[depthslice[:,1]==sink_index+source_num+1]
        temp_polar=polar[depthslice[:,1]==sink_index+source_num+1,:]
        for slice_index in range(len(sink_slice)):
            temp=time_indexing(excitation_signal,sink_slice[slice_index],time_map.shape[2])
            time_map[sink_index,0,:]=time_map[sink_index,0,:]+temp_polar[slice_index,0]*temp*temp_loss[slice_index]
            time_map[sink_index,1,:]=time_map[sink_index,1,:]+temp_polar[slice_index,1]*temp*temp_loss[slice_index]
            time_map[sink_index,2,:]=time_map[sink_index,2,:]+temp_polar[slice_index,2]*temp*temp_loss[slice_index]

    return time_map

def TimeDomain(source_num,sink_num,point_informationv2,full_index,scattering_coefficient,wavelength,excitation_signal,sampling_freq,num_samples):
    #use the path networks to generate a time domain polarimetric plot, representing voltage received in each polarisation
    time_map=np.zeros((sink_num,3,num_samples),dtype=np.float32)
    time_index=np.linspace(0,(1/sampling_freq)*num_samples,num_samples)
    #paths=EMGPUPathLengths(source_num,sink_num,full_index,point_informationv2)
    #polar_coefficients=np.abs(EMGPUPolarMixing(source_num,sink_num,full_index,point_informationv2))
    #paths,polar_coefficients=EMGPUPathandPolarMixing(source_num,sink_num,full_index,point_informationv2)
    paths,polar_coefficients=EMGPUJointPathLengthandPolar(source_num,sink_num,full_index,point_informationv2)
    arrival_times=paths/scipy.constants.c
    time_ref=np.min(arrival_times)
    time_spread=(np.max(paths)-np.min(paths))/scipy.constants.c
    #print(time_spread,' seconds')
    #print((np.max(paths)-np.min(paths)/3e8),' seconds')
    #time_ref=0.0
    time_steps=np.digitize(arrival_times,time_index)
    #print(np.max(time_steps)-np.min(time_steps))
    depthslice,scatter_index=targettingindex(full_index)
    loss=np.abs(pathlossv2(paths,wavelength)*(np.abs(np.power(scattering_coefficient,scatter_index-1))*np.exp(-1j*np.pi)))
    # for sink_index in range(sink_num):
    #     sink_slice=time_steps[depthslice[:,1]==sink_index+source_num+1]
    #     temp_loss=loss[depthslice[:,1]==sink_index+source_num+1]
    #     for slice_index in range(len(sink_slice)):
    #         time_map[sink_index,0,:]=time_map[sink_index,0,:]+time_indexing(excitation_signal,sink_slice[0],num_samples)
    time_map=time_sortingv2(source_num,sink_num,time_steps,depthslice,loss,np.abs(polar_coefficients),time_map,excitation_signal)

    return time_map,time_ref

def TimeDomainv2(source_num,
                 sink_num,
                 point_informationv2,
                 full_index,
                 scattering_coefficient,
                 wavelength,
                 excitation_signal,
                 sampling_freq,
                 num_samples):
    """
    New wrapper to run time domain propagation on the GPU, allowing for faster simulations.
    This model is static, as the points do not currently allow for motion vectors
    Parameters
    ----------
    source_num : int
        DESCRIPTION.
    sink_num : int
        DESCRIPTION.
    point_informationv2 : point data type
        currently has position, normal vector, and electric weighting in each axis, to allow for polarised scattering.
    full_index : int array
        index of all sucesful ray paths from source (1 to source_num+1), to sink (source_num+1 to sink_num+source_num+1), with all intermediate steps
    scattering_coefficient : float
        allows for exploration of different spreading factors
    wavelength : float
        wavelength of the central frequency
    excitation_signal : float array
        the excitation signal, sampled at sampling_freq rate.
    sampling_freq : float
        the sampling frequency in Hz
    num_samples : TYPE
        the number of samples required from the first incoming wave

    Returns
    -------
    time_map : array of floats
        an array of floats of size source_num * sink_num * num_samples * 3 to contain the 3D polarised information in the time domain

    wake_time : float
        the time in seconds that the earliest return arrived at the sinks
    """
    ray_num=full_index.shape[0]
    threads_in_block=256
    max_blocks=65535
    maximum_chunk_size=2**8
    path_lengths=np.zeros((ray_num),dtype=np.float32)
    time_map=np.zeros((source_num,sink_num,num_samples,3),dtype=np.float64)
    print(time_map.nbytes)
    d_time_map=cuda.device_array((time_map.shape[0],time_map.shape[1],time_map.shape[2],time_map.shape[3]),dtype=np.float64)
    d_time_map=cuda.to_device(time_map)
    d_point_information = cuda.device_array(point_informationv2.shape[0], dtype=base.scattering_t)
    d_point_information=cuda.to_device(point_informationv2)
    d_excitation=cuda.device_array(excitation_signal.shape[0],dtype=np.float64)
    d_excitation=cuda.to_device(excitation_signal)
    d_wavelength=cuda.device_array((1),dtype=np.complex64)
    d_wavelength=cuda.to_device(np.ones((1),dtype=np.float64)*wavelength)
    d_sampling_freq=cuda.device_array((1),dtype=np.float64)
    d_sampling_freq=cuda.to_device(np.ones((1),dtype=np.float64)*sampling_freq)
    d_wake_time=cuda.device_array((1),dtype=np.float64)
    d_wake_time=cuda.to_device(np.ones((1),dtype=np.float64))
    d_arrival_times=cuda.device_array(full_index.shape[0],dtype=np.float64)
    d_arrival_times=cuda.to_device(np.zeros(full_index.shape[0],dtype=np.float64))
    #divide in terms of a block for each source, then
    depthslice,_=targettingindex(copy.deepcopy(full_index))
    depthslice[:,0]-=1
    depthslice[:,1]-=(source_num+1)
    d_target_index = cuda.device_array((depthslice.shape[0],depthslice.shape[1]), dtype=np.int64)
    d_target_index=cuda.to_device(depthslice)
    d_full_index = cuda.device_array((full_index.shape[0],full_index.shape[1]), dtype=np.int64)
    #d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
    #d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
    #paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
    #polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
    d_full_index = cuda.to_device(full_index)
    # Here, we choose the granularity of the threading on our device. We want
    # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
    # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

    grids=(math.ceil(full_index.shape[0]/threads_in_block))
    threads = threads_in_block
    #print(grids,' blocks, ',threads,' threads')
    # Execute the kernel
    #cuda.profile_start()
    timedomainkernal[grids, threads](d_full_index,d_point_information,d_target_index,d_wavelength,d_excitation,d_sampling_freq,d_arrival_times,d_wake_time,d_time_map)
    #polaranddistance(d_full_index,d_point_information,polar_c,paths)
    #cuda.profile_stop()
    #ray_components[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
    #distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
    #first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
    #resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()
    #chunks=np.linspace(0,path_lengths.shape[0],math.ceil(path_lengths.shape[0]/maximum_chunk_size)+1,dtype=np.int32)
    #for n in range(chunks.shape[0]-1):
    #polar_coefficients=d_polar_c.copy_to_host()
    time_map=cp.asnumpy(d_time_map)
    wake_time=cp.asnumpy(d_wake_time)
    return time_map, wake_time

def TimeDomainv3(source_num,
                 sink_num,
                 point_informationv2,
                 full_index,
                 scattering_coefficient,
                 wavelength,
                 excitation_signal,
                 sampling_freq,
                 num_samples):
    """
    New wrapper to run time domain propagation on the GPU, allowing for faster simulations.
    This model is static, as the points do not currently allow for motion vectors
    Parameters
    ----------
    source_num : int
        DESCRIPTION.
    sink_num : int
        DESCRIPTION.
    point_informationv2 : point data type
        currently has position, normal vector, and electric weighting in each axis, to allow for polarised scattering.
    full_index : int array
        index of all sucesful ray paths from source (1 to source_num+1), to sink (source_num+1 to sink_num+source_num+1), with all intermediate steps
    scattering_coefficient : float
        allows for exploration of different spreading factors
    wavelength : float
        wavelength of the central frequency
    excitation_signal : float array
        the excitation signal, sampled at sampling_freq rate.
    sampling_freq : float
        the sampling frequency in Hz
    num_samples : TYPE
        the number of samples required from the first incoming wave

    Returns
    -------
    time_map : array of floats
        an array of floats of size source_num * sink_num * num_samples * 3 to contain the 3D polarised information in the time domain

    wake_time : float
        the time in seconds that the earliest return arrived at the sinks
    """
    ray_num=full_index.shape[0]
    threads_in_block=256
    max_blocks=65535
    maximum_chunk_size=2**8
    path_lengths=np.zeros((ray_num),dtype=np.float32)
    time_map=np.zeros((source_num,sink_num,num_samples,3),dtype=np.float64)
    time_step=1.0/sampling_freq
    flag=True
    if np.ceil(time_map.nbytes/1e9)>1:
        #setup time_map chunking
        print('source chunking ',time_map.nbytes/1e9,'Gb')
        num_chunks=np.ceil(time_map.nbytes/1e9).astype(np.int32)
        source_chunking=np.linspace(0,source_num,num_chunks+1).astype(np.int32)
        #setup wake time as a second
        wake_time=np.ones((1),dtype=np.float64)
        wake_times = np.full((len(source_chunking)-1), 1, dtype=np.float64)
        for n in range(len(source_chunking) - 1):
            #print(n,time_map[source_chunking[n]:source_chunking[n+1],:,:,:].shape)
            d_temp_map=cuda.device_array((time_map[source_chunking[n]:source_chunking[n+1],:,:,:].shape[0],time_map.shape[1],time_map.shape[2],time_map.shape[3]),dtype=np.float64)
            d_temp_map = cuda.to_device(time_map[source_chunking[n]:source_chunking[n+1],:,:,:])
            d_point_information = cuda.device_array(point_informationv2.shape[0], dtype=base.scattering_t)
            d_point_information = cuda.to_device(point_informationv2)
            d_excitation = cuda.device_array(excitation_signal.shape[0], dtype=np.float64)
            d_excitation = cuda.to_device(excitation_signal)
            d_wavelength = cuda.device_array((1), dtype=np.complex64)
            d_wavelength = cuda.to_device(np.ones((1), dtype=np.float64) * wavelength)
            d_sampling_freq = cuda.device_array((1), dtype=np.float64)
            d_sampling_freq = cuda.to_device(np.ones((1), dtype=np.float64) * sampling_freq)
            d_wake_time = cuda.device_array((1), dtype=np.float64)
            d_wake_time = cuda.to_device(wake_times)
            sources=np.linspace(source_chunking[n]+1,source_chunking[n+1],source_chunking[n+1]-source_chunking[n]).astype(np.int32)
            temp_index=full_index[np.isin(full_index[:, 0], sources), :]
            d_arrival_times = cuda.device_array(temp_index.shape[0], dtype=np.float64)
            d_arrival_times = cuda.to_device(np.zeros(temp_index.shape[0], dtype=np.float64))
            depthslice, _ = targettingindex(copy.deepcopy(temp_index))
            depthslice[:, 0] -= (1+source_chunking[n])
            depthslice[:, 1] -= (source_num + 1)
            d_target_index = cuda.device_array((depthslice.shape[0], depthslice.shape[1]), dtype=np.int64)
            d_target_index = cuda.to_device(depthslice)
            d_full_index = cuda.device_array((temp_index.shape[0], full_index.shape[1]), dtype=np.int64)
            # d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
            # d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
            # paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
            # polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
            d_full_index = cuda.to_device(temp_index)
            # Here, we choose the granularity of the threading on our device. We want
            # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
            # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

            grids = (math.ceil(temp_index.shape[0] / threads_in_block))
            threads = threads_in_block
            # print(grids,' blocks, ',threads,' threads')
            # Execute the kernel
            # cuda.profile_start()
            timedomainkernal[grids, threads](d_full_index, d_point_information, d_target_index, d_wavelength,
                                             d_excitation, d_sampling_freq, d_arrival_times, d_wake_time, d_temp_map)
            #print(source_chunking[n],source_chunking[n+1])
            
            time_map[source_chunking[n]:source_chunking[n+1],:,:,:] = cp.asnumpy(d_temp_map)
            wake_times[n] = cp.asnumpy(d_wake_time)[0]
            #calculate seperation and shift former time_map values to ensure sync
            wake_time=wake_times[n]


        print(wake_times)
    else:
        d_time_map=cuda.device_array((time_map.shape[0],time_map.shape[1],time_map.shape[2],time_map.shape[3]),dtype=np.float64)
        d_time_map=cuda.to_device(time_map)
        d_point_information = cuda.device_array(point_informationv2.shape[0], dtype=base.scattering_t)
        d_point_information=cuda.to_device(point_informationv2)
        d_excitation=cuda.device_array(excitation_signal.shape[0],dtype=np.float64)
        d_excitation=cuda.to_device(excitation_signal)
        d_wavelength=cuda.device_array((1),dtype=np.complex64)
        d_wavelength=cuda.to_device(np.ones((1),dtype=np.float64)*wavelength)
        d_sampling_freq=cuda.device_array((1),dtype=np.float64)
        d_sampling_freq=cuda.to_device(np.ones((1),dtype=np.float64)*sampling_freq)
        d_wake_time=cuda.device_array((1),dtype=np.float64)
        d_wake_time=cuda.to_device(np.ones((1),dtype=np.float64))
        d_arrival_times=cuda.device_array(full_index.shape[0],dtype=np.float64)
        d_arrival_times=cuda.to_device(np.zeros(full_index.shape[0],dtype=np.float64))
        #divide in terms of a block for each source, then
        depthslice,_=targettingindex(copy.deepcopy(full_index))
        depthslice[:,0]-=1
        depthslice[:,1]-=(source_num+1)
        d_target_index = cuda.device_array((depthslice.shape[0],depthslice.shape[1]), dtype=np.int64)
        d_target_index=cuda.to_device(depthslice)
        d_full_index = cuda.device_array((full_index.shape[0],full_index.shape[1]), dtype=np.int64)
        #d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
        #d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
        #paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
        #polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
        d_full_index = cuda.to_device(full_index)
        # Here, we choose the granularity of the threading on our device. We want
        # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
        # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

        grids=(math.ceil(full_index.shape[0]/threads_in_block))
        threads = threads_in_block
        #print(grids,' blocks, ',threads,' threads')
        # Execute the kernel
        #cuda.profile_start()
        timedomainkernal[grids, threads](d_full_index,d_point_information,d_target_index,d_wavelength,d_excitation,d_sampling_freq,d_arrival_times,d_wake_time,d_time_map)
        #polaranddistance(d_full_index,d_point_information,polar_c,paths)
        #cuda.profile_stop()
        #ray_components[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
        #distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
        #first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
        #resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()
        #chunks=np.linspace(0,path_lengths.shape[0],math.ceil(path_lengths.shape[0]/maximum_chunk_size)+1,dtype=np.int32)
        #for n in range(chunks.shape[0]-1):
        #polar_coefficients=d_polar_c.copy_to_host()
        time_map=cp.asnumpy(d_time_map)
        wake_times=cp.asnumpy(d_wake_time)

    if wake_times.size>1:
        if (np.max((wake_times-np.min(wake_times))/time_step)>=1.0):
            corrected_time_map=np.empty_like(time_map)
            timeadjustments=np.round((wake_times-np.min(wake_times))/time_step).astype(int)
            for n in range(len(source_chunking) - 1):
                corrected_time_map[source_chunking[n]:source_chunking[n + 1], :, :timeadjustments[n], :]=0
                corrected_time_map[source_chunking[n]:source_chunking[n + 1], :, timeadjustments[n]:, :] = time_map[source_chunking[n]:source_chunking[n + 1], :,:-timeadjustments[n],:]

            time_map=corrected_time_map

    return time_map, wake_times[0]


def TimeDomainThetaPhi(source_num,
                 sink_num,
                 point_informationv2,
                 full_index,
                 scattering_coefficient,
                 wavelength,
                 excitation_signal,
                 sampling_freq,
                 num_samples):
    """
    New wrapper to run time domain propagation on the GPU, allowing for faster simulations.
    This model is static, as the points do not currently allow for motion vectors
    Parameters
    ----------
    source_num : int
        DESCRIPTION.
    sink_num : int
        DESCRIPTION.
    point_informationv2 : point data type
        currently has position, normal vector, and electric weighting in each axis, to allow for polarised scattering.
    full_index : int array
        index of all sucesful ray paths from source (1 to source_num+1), to sink (source_num+1 to sink_num+source_num+1), with all intermediate steps
    scattering_coefficient : float
        allows for exploration of different spreading factors
    wavelength : float
        wavelength of the central frequency
    excitation_signal : float array
        the excitation signal, sampled at sampling_freq rate.
    sampling_freq : float
        the sampling frequency in Hz
    num_samples : TYPE
        the number of samples required from the first incoming wave

    Returns
    -------
    time_map : array of floats
        an array of floats of size source_num * sink_num * num_samples * 2 to contain the 3D polarised information in the time domain

    wake_time : float
        the time in seconds that the earliest return arrived at the sinks
    """
    ray_num = full_index.shape[0]
    threads_in_block = 256
    max_blocks = 65535
    maximum_chunk_size = 2 ** 8
    path_lengths = np.zeros((ray_num), dtype=np.float32)
    time_map = np.zeros((source_num, sink_num, num_samples, 2), dtype=np.float64)
    flag = True
    if np.ceil(time_map.nbytes / 1e9) > 1:
        # setup time_map chunking
        print('source chunking ', time_map.nbytes / 1e9, 'Gb')
        num_chunks = np.ceil(time_map.nbytes / 1e9).astype(np.int32)
        source_chunking = np.linspace(0, source_num, num_chunks + 1).astype(np.int32)
        # setup wake time as a second
        wake_time = np.ones((1), dtype=np.float64)
        wake_times = np.full((len(source_chunking) - 1), 1, dtype=np.float64)
        for n in range(len(source_chunking) - 1):
            # print(n,time_map[source_chunking[n]:source_chunking[n+1],:,:,:].shape)
            d_temp_map = cuda.device_array((time_map[source_chunking[n]:source_chunking[n + 1], :, :, :].shape[0],
                                            time_map.shape[1], time_map.shape[2], time_map.shape[3]), dtype=np.float64)
            d_temp_map = cuda.to_device(time_map[source_chunking[n]:source_chunking[n + 1], :, :, :])
            d_point_information = cuda.device_array(point_informationv2.shape[0], dtype=base.scattering_t)
            d_point_information = cuda.to_device(point_informationv2)
            d_excitation = cuda.device_array(excitation_signal.shape[0], dtype=np.float64)
            d_excitation = cuda.to_device(excitation_signal)
            d_wavelength = cuda.device_array((1), dtype=np.complex64)
            d_wavelength = cuda.to_device(np.ones((1), dtype=np.float64) * wavelength)
            d_sampling_freq = cuda.device_array((1), dtype=np.float64)
            d_sampling_freq = cuda.to_device(np.ones((1), dtype=np.float64) * sampling_freq)
            d_wake_time = cuda.device_array((1), dtype=np.float64)
            d_wake_time = cuda.to_device(wake_times)
            sources = np.linspace(source_chunking[n] + 1, source_chunking[n + 1],
                                  source_chunking[n + 1] - source_chunking[n]).astype(np.int32)
            temp_index = full_index[np.isin(full_index[:, 0], sources), :]
            d_arrival_times = cuda.device_array(temp_index.shape[0], dtype=np.float64)
            d_arrival_times = cuda.to_device(np.zeros(temp_index.shape[0], dtype=np.float64))
            depthslice, _ = targettingindex(copy.deepcopy(temp_index))
            depthslice[:, 0] -= (1 + source_chunking[n])
            depthslice[:, 1] -= (source_num + 1)
            d_target_index = cuda.device_array((depthslice.shape[0], depthslice.shape[1]), dtype=np.int64)
            d_target_index = cuda.to_device(depthslice)
            d_full_index = cuda.device_array((temp_index.shape[0], full_index.shape[1]), dtype=np.int64)
            # d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
            # d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
            # paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
            # polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
            d_full_index = cuda.to_device(temp_index)
            # Here, we choose the granularity of the threading on our device. We want
            # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
            # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

            grids = (math.ceil(temp_index.shape[0] / threads_in_block))
            threads = threads_in_block
            # print(grids,' blocks, ',threads,' threads')
            # Execute the kernel
            # cuda.profile_start()
            timedomainthetaphi[grids, threads](d_full_index, d_point_information, d_target_index, d_wavelength,
                                             d_excitation, d_sampling_freq, d_arrival_times, d_wake_time, d_temp_map)
            # print(source_chunking[n],source_chunking[n+1])

            time_map[source_chunking[n]:source_chunking[n + 1], :, :, :] = cp.asnumpy(d_temp_map)
            wake_times[n] = cp.asnumpy(d_wake_time)[0]
            wake_time = wake_times[n]

        print(wake_times)
    else:
        d_time_map = cuda.device_array((time_map.shape[0], time_map.shape[1], time_map.shape[2], time_map.shape[3]),
                                       dtype=np.float64)
        d_time_map = cuda.to_device(time_map)
        d_point_information = cuda.device_array(point_informationv2.shape[0], dtype=base.scattering_t)
        d_point_information = cuda.to_device(point_informationv2)
        d_excitation = cuda.device_array(excitation_signal.shape[0], dtype=np.float64)
        d_excitation = cuda.to_device(excitation_signal)
        d_wavelength = cuda.device_array((1), dtype=np.complex64)
        d_wavelength = cuda.to_device(np.ones((1), dtype=np.float64) * wavelength)
        d_sampling_freq = cuda.device_array((1), dtype=np.float64)
        d_sampling_freq = cuda.to_device(np.ones((1), dtype=np.float64) * sampling_freq)
        d_wake_time = cuda.device_array((1), dtype=np.float64)
        d_wake_time = cuda.to_device(np.ones((1), dtype=np.float64))
        d_arrival_times = cuda.device_array(full_index.shape[0], dtype=np.float64)
        d_arrival_times = cuda.to_device(np.zeros(full_index.shape[0], dtype=np.float64))
        # divide in terms of a block for each source, then
        depthslice, _ = targettingindex(copy.deepcopy(full_index))
        depthslice[:, 0] -= 1
        depthslice[:, 1] -= (source_num + 1)
        d_target_index = cuda.device_array((depthslice.shape[0], depthslice.shape[1]), dtype=np.int64)
        d_target_index = cuda.to_device(depthslice)
        d_full_index = cuda.device_array((full_index.shape[0], full_index.shape[1]), dtype=np.int64)
        # d_paths=cuda.device_array((path_lengths.shape[0]),dtype=np.float32)
        # d_polar_c=cuda.device_array((polar_coefficients.shape),dtype=np.complex64)
        # paths=cp.zeros((path_lengths.shape[0]),dtype=np.float32)
        # polar_c=cp.zeros((polar_coefficients.shape),dtype=np.complex64)
        d_full_index = cuda.to_device(full_index)
        # Here, we choose the granularity of the threading on our device. We want
        # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
        # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads

        grids = (math.ceil(full_index.shape[0] / threads_in_block))
        threads = threads_in_block
        # print(grids,' blocks, ',threads,' threads')
        # Execute the kernel
        # cuda.profile_start()
        timedomainthetaphi[grids, threads](d_full_index, d_point_information, d_target_index, d_wavelength, d_excitation,
                                         d_sampling_freq, d_arrival_times, d_wake_time, d_time_map)
        # polaranddistance(d_full_index,d_point_information,polar_c,paths)
        # cuda.profile_stop()
        # ray_components[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
        # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
        # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
        # resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()
        # chunks=np.linspace(0,path_lengths.shape[0],math.ceil(path_lengths.shape[0]/maximum_chunk_size)+1,dtype=np.int32)
        # for n in range(chunks.shape[0]-1):
        # polar_coefficients=d_polar_c.copy_to_host()
        time_map = cp.asnumpy(d_time_map)
        wake_times = cp.asnumpy(d_wake_time)
    return time_map, wake_times
#@njit(cache=True, nogil=True)
def targettingindex(full_index):
    #slice the full index to produce the source and sink index for each ray
    scatter_index=np.ones((full_index.shape[0]),dtype=np.int64)
    if (full_index.shape[1]==2):
        depth_slice=full_index

    if (full_index.shape[1]==3):
        depth_slice=full_index[np.all(full_index[:,2:]==0,axis=1),:][:,[0,1]]
        depth_slice=np.append(depth_slice,full_index[np.all(full_index[:,2:]!=0,axis=1),:][:,[0,2]],axis=0)
        scatter_index[np.all(full_index[:,2:]!=0,axis=1)]=2

    if (full_index.shape[1]==4):
        depth_slice=full_index[np.all(full_index[:,2:]==0,axis=1),:][:,[0,1]]
        remainder=full_index[~np.all(full_index[:,2:]==0,axis=1),:]
        depth_slice=np.append(depth_slice,remainder[np.all(remainder[:,3:]==0,axis=1),:][:,[0,2]],axis=0)
        remainder2=remainder[~np.all(remainder[:,3:]==0,axis=1),:]
        depth_slice=np.append(depth_slice,remainder2[np.all(remainder2[:,3:]!=0,axis=1),:][:,[0,3]],axis=0)
        scatter_index[np.all(full_index[:,2:]!=0,axis=1)]=3
        scatter_index[np.all(np.array([full_index[:,2]!=0,full_index[:,3]==0]),axis=0)]=2


    return depth_slice,scatter_index

def TimeDomainEthetaEphiTransform(Ex,Ey,Ez,point_normals,prime_vector=np.array([[0,0,1]],dtype=np.float32)):
    """
    Convert the time domain electric field vectors from a cartesian basis (Ex,Ey,Ez) at each point into a Etheta,Ephi
    polarisations.
    Parameters
    ----------
    Ex: time domain electric field in the x plane
        float array of shape num_points * num_samples
    Ey: time domain electric field in the y plane
        float array of shape num_points * num_samples
    Ez: time domain electric field in the z plane
        float array of shape num_points * num_samples
    point_normals: the normal vectors of each point of interest (xyz)
        float array of shape num_points * 3

    Returns
    ----------
    Etheta : time domain electric field (Etheta polarisation)
        float array of shape num_points * num_samples
    Ephi : time domain electric field (Ephi polarisation)
        float array of shape num_points * num_samples
    """
    if len(Ex.shape)==1:
        num_points=Ex.shape[0]
        num_samples=1
    else:
        num_points,num_samples=Ex.shape
    Etheta = np.zeros((num_points, num_samples), dtype=np.float32)
    Ephi = np.zeros((num_points, num_samples), dtype=np.float32)
    Etheta_vectors=calculate_conformalVectors(prime_vector,point_normals)
    Ephi_vectors=np.cross(Etheta_vectors,point_normals)
    Etheta=Etheta_vectors[:,0].reshape(num_points,1)*Ex.reshape(num_points,num_samples)+Etheta_vectors[:,1].reshape(num_points,1)*Ey.reshape(num_points,num_samples)+Etheta_vectors[:,2].reshape(num_points,1)*Ez.reshape(num_points,num_samples)
    Ephi = Ephi_vectors[:, 0].reshape(num_points,1) * Ex.reshape(num_points,num_samples) + Ephi_vectors[:, 1].reshape(num_points,1) * Ey.reshape(num_points,num_samples) + Ephi_vectors[:, 2].reshape(num_points,1) * Ez.reshape(num_points,num_samples)

    return Etheta,Ephi
# def EMGPUCompressedScatteringtest(source_num,sink_num,unified_model,unified_normals,unified_weights,full_index,point_information,wavelength):
#     """
#     wrapper for the GPU EM processer, outputting the resultant ray components as complex values, allowing for the whole thing to be sorted again.
#     At present, the indexing only supports processing the rays for line of sight and single bounce, but that will be sorted quite quickly
#     Parameters
#     ----------
#     full_index : int array
#         index of all successful rays
#     point_information : TYPE
#         DESCRIPTION.
#     wavelength : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     resultant_rays : TYPE
#         DESCRIPTION.

#     """
#     #cuda.select_device(0)
#     #network_index,point_information,ray_components
#     scattering_matrix=np.zeros((source_num,sink_num,3,2),dtype=np.complex64)
#     ray_tempslosref=EMGPUScatteringWrapper(source_num,sink_num,full_index[full_index[:,-1]==0,:],point_information,wavelength)
#     ray_tempslos=RF.ScatteringNetworkGenerator(full_index[full_index[:,-1]==0,:],unified_model,unified_normals,unified_weights,point_information,wavelength,np.zeros((len(full_index[full_index[:,-1]==0,:]),3),dtype=np.complex64))
#     ray_tempsbounceref=EMGPUScatteringWrapper(source_num,sink_num,full_index[full_index[:,-1]!=0,:],point_information,wavelength)
#     ray_tempsbounce=RF.ScatteringNetworkGenerator(full_index[full_index[:,-1]!=0,:],unified_model,unified_normals,unified_weights,point_information,wavelength,np.zeros((len(full_index[full_index[:,-1]!=0,:]),3),dtype=np.complex64))
#     if np.max(np.abs(ray_tempslos-ray_tempslosref))>0.0:
#         print('los error',np.max(np.abs(ray_tempslos-ray_tempslosref)))
#     if np.max(np.abs(ray_tempsbounce-ray_tempsbounceref))>0.0:
#         print('bounce error',np.max(np.abs(ray_tempsbounce-ray_tempsbounceref)))
#     depth_slice=full_index[full_index[:,-1]==0,:][:,[0,1]]
#     for sink_index in range(sink_num):
#         for source_index in range(source_num):
#             scattering_matrix[source_index,sink_index,0,0]=np.sum(ray_tempslos[(depth_slice[:,0]-1==source_index) & (depth_slice[:,1]-source_num-1==sink_index),0])
#             scattering_matrix[source_index,sink_index,1,0]=np.sum(ray_tempslos[(depth_slice[:,0]-1==source_index) & (depth_slice[:,1]-source_num-1==sink_index),1])
#             scattering_matrix[source_index,sink_index,2,0]=np.sum(ray_tempslos[(depth_slice[:,0]-1==source_index) & (depth_slice[:,1]-source_num-1==sink_index),2])

#     depth_slice=full_index[full_index[:,-1]!=0,:][:,[0,2]]
#     for sink_index in range(sink_num):
#         for source_index in range(source_num):
#             scattering_matrix[source_index,sink_index,0,1]=np.sum(ray_tempsbounce[(depth_slice[:,0]-1==source_index) & (depth_slice[:,1]-source_num-1==sink_index),0])
#             scattering_matrix[source_index,sink_index,1,1]=np.sum(ray_tempsbounce[(depth_slice[:,0]-1==source_index) & (depth_slice[:,1]-source_num-1==sink_index),1])
#             scattering_matrix[source_index,sink_index,2,1]=np.sum(ray_tempsbounce[(depth_slice[:,0]-1==source_index) & (depth_slice[:,1]-source_num-1==sink_index),2])


#         #scattering_matrix[source_index,:,:]=temp_matrix
#     #scattering_matrix=d_scatter_matrix.copy_to_host()
#     #cuda.close()
#     return scattering_matrix

def EMGPUScatteringWrapper(source_num,sink_num,full_index,point_information,wavelength):
    """
    wrapper for the GPU EM processer, outputting the resultant ray components as complex values, allowing for the whole thing to be sorted again.
    At present, the indexing only supports processing the rays for line of sight and single bounce, but that will be sorted quite quickly
    Parameters
    ----------
    full_index : int array
        index of all successful rays
    point_information : TYPE
        DESCRIPTION.
    wavelength : TYPE
        DESCRIPTION.

    Returns
    -------
    resultant_rays : TYPE
        DESCRIPTION.

    """
    #cuda.select_device(0)
    #network_index,point_information,ray_components
    ray_num=full_index.shape[0]
    maximum_chunk_size=2**8
    threads_in_block=1024
    maximum_sources=1
    scattering_matrix=np.zeros((source_num,sink_num,3),dtype=np.complex64)
    ray_components=np.zeros((full_index.shape[0],3),dtype=np.complex64)
    depth_slice=np.append(full_index[full_index[:,-1]==0,0:2],full_index[full_index[:,-1]!=0,0:2],axis=0)
    #source_chunks=np.linspace(0,source_num-1,math.ceil(source_num/maximum_sources)+1,dtype=np.int32)
    #distance_temp=np.empty((chunk_ray_size),dtype=np.float32)
    #distance_temp[:]=math.inf
    #dist_list=cuda.to_device(distance_temp)
    d_problem_size=cuda.device_array((2),dtype=np.int32)
    d_problem_size=cuda.to_device(np.asarray([source_num,sink_num],dtype=np.int32))
    d_wavelength=cuda.device_array((1),dtype=np.float64)
    d_wavelength=cuda.to_device(np.ones((1),dtype=np.float32)*wavelength)
    d_point_information = cuda.device_array(point_information.shape[0], dtype=base.scattering_t)
    d_point_information=cuda.to_device(point_information)
    for source_index in range(source_num):
        temp_payload=full_index[full_index[:,0]==(source_index+1),:]
        temp_rays=np.zeros((temp_payload.shape[0],3),dtype=np.complex64)
        ray_chunks=np.linspace(0,temp_payload.shape[0],math.ceil(temp_payload.shape[0]/maximum_chunk_size)+1,dtype=np.int32)
        for n in range(ray_chunks.shape[0]-1):
            chunk_payload=temp_payload[ray_chunks[n]:ray_chunks[n+1],:]
            d_full_index = cuda.device_array((chunk_payload.shape[0],chunk_payload.shape[1]), dtype=np.int64)
            temp_matrix=np.zeros((chunk_payload.shape[0],3),dtype=np.complex64)
            d_scatter_matrix=cuda.device_array((temp_matrix.shape),dtype=np.complex64)
            d_full_index = cuda.to_device(chunk_payload)
            # Here, we choose the granularity of the threading on our device. We want
            # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
            # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
            grids=(math.ceil(chunk_payload.shape[0]/threads_in_block))
            threads = (min(chunk_payload.shape[0],threads_in_block))
            # Execute the kernel
            #cuda.profile_start()
            scatteringkernaltest[grids, threads](d_problem_size,d_full_index,d_point_information,d_scatter_matrix,d_wavelength)
            #cuda.profile_stop()
            temp_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
            #ray_components[ray_chunks[n]:ray_chunks[n+1],:]=d_scatter_matrix.copy_to_host()
            #distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
            #first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
            #resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()

        ray_components[full_index[:,0]==(source_index+1),:]=temp_rays

    return ray_components

def EMGPUWrapper(source_num,sink_num,full_index,point_information,wavelength):
    """
    wrapper for the GPU EM processer, outputting the resultant ray components as complex values, allowing for the whole thing to be sorted again.
    At present, the indexing only supports processing the rays for line of sight and single bounce, but that will be sorted quite quickly
    Parameters
    ----------
    full_index : int array
        index of all successful rays
    point_information : TYPE
        DESCRIPTION.
    wavelength : TYPE
        DESCRIPTION.

    Returns
    -------
    resultant_rays : TYPE
        DESCRIPTION.

    """
    #network_index,point_information,ray_components
    ray_num=full_index.shape[0]
    maximum_chunk_size=2**8
    threads_in_block=1024
    resultant_rays=np.zeros((ray_num,3),dtype=np.complex64)
    ray_chunks=np.linspace(0,ray_num,math.ceil(ray_num/maximum_chunk_size)+1,dtype=np.int32)
    #distance_temp=np.empty((chunk_ray_size),dtype=np.float32)
    #distance_temp[:]=math.inf
    #dist_list=cuda.to_device(distance_temp)
    d_problem_size=cuda.device_array((2),dtype=np.int32)
    d_problem_size=cuda.to_device(np.asarray([source_num,sink_num],dtype=np.int32))
    d_wavelength=cuda.device_array((1),dtype=np.float64)
    d_wavelength=cuda.to_device(np.ones((1),dtype=np.float32)*wavelength)
    d_point_information = cuda.device_array(point_information.shape[0], dtype=base.scattering_t)
    d_point_information=cuda.to_device(point_information)
    for n in range(ray_chunks.shape[0]-1):
        chunk_payload=full_index[ray_chunks[n]:ray_chunks[n+1],:]
        d_full_index = cuda.device_array((chunk_payload.shape[0],chunk_payload.shape[1]), dtype=np.int64)
        d_full_index = cuda.to_device(chunk_payload)
        d_channels = cuda.device_array((chunk_payload.shape[0],3), dtype=np.complex64)
        # Here, we choose the granularity of the threading on our device. We want
        # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
        # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
        grids=(math.ceil(chunk_payload.shape[0]/threads_in_block))
        threads = (min(chunk_payload.shape[0],threads_in_block))
        # Execute the kernel
        #cuda.profile_start()
        #scatteringkernal[grids, threads](d_full_index,d_point_information,d_channels,d_wavelength)
        scatteringkernaltest[grids,threads](d_problem_size,d_full_index,d_point_information,d_channels,d_wavelength)
        #cuda.profile_stop()
        #distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
        #first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
        resultant_rays[ray_chunks[n]:ray_chunks[n+1],:]=d_channels.copy_to_host()


    return resultant_rays


def DisplayESources(source_display_coords,E_vectors,source_type='E',arrow_length=0.1):
    #create a set of arrows, coloured depending on the type of source
    E_colour=np.array([0.08,0,1])
    M_colour=np.array([1,0.04,0.04])
    if source_type=='E':
        arrow_color=E_colour
    else:
        arrow_color=M_colour

    quiver_set = o3d.geometry.TriangleMesh.create_arrow(cone_height= 0.2 * arrow_length,
                                                            cone_radius= 0.06 * arrow_length,
                                                            cylinder_height= 0.8 * arrow_length,
                                                            cylinder_radius=  0.04 * arrow_length
                                                            )
    rot_mat = caculate_align_mat(E_vectors[0,:])
    quiver_set=GF.open3drotate(quiver_set,rot_mat)
    quiver_set.translate(source_display_coords[0,:]+E_vectors[0,:]*(-0.5*arrow_length))
    quiver_set.paint_uniform_color(arrow_color)
    quiver_set.compute_vertex_normals()

    for arrow_num in range(1,source_display_coords.shape[0]):
        mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(cone_height= 0.2 * arrow_length,
                                                            cone_radius= 0.06 * arrow_length,
                                                            cylinder_height= 0.8 * arrow_length,
                                                            cylinder_radius=  0.04 * arrow_length
                                                            )
        rot_mat = caculate_align_mat(E_vectors[arrow_num,:])
        mesh_arrow=GF.open3drotate(mesh_arrow,rot_mat)
        mesh_arrow.translate(source_display_coords[arrow_num]+E_vectors[arrow_num,:]*(-0.5*arrow_length))
        mesh_arrow.paint_uniform_color(arrow_color)
        mesh_arrow.compute_vertex_normals()
        quiver_set=quiver_set+mesh_arrow


    return quiver_set

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    qTrans_Mat = np.eye(3,3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat

@njit(cache=True, nogil=True)
def orthconformalvector(desired_axis,point_normal):
    ray_u=np.zeros((3),dtype=np.float32)
    ray_v=np.zeros((3),dtype=np.float32)
    x_vec=np.zeros((3),dtype=np.float32)
    y_vec=np.zeros((3),dtype=np.float32)
    z_vec=np.zeros((3),dtype=np.float32)
    x_vec[0]=1
    y_vec[1]=1
    z_vec[2]=1
    #make sure ray vectors are locked on appropriate global axes
    x_orth=np.linalg.norm(np.cross(x_vec,desired_axis))
    y_orth=np.linalg.norm(np.cross(y_vec,desired_axis))
    z_orth=np.linalg.norm(np.cross(z_vec,desired_axis))
    if (abs(x_orth)>abs(y_orth)) and (abs(x_orth)>abs(z_orth)):
        #use x-axis to establish ray uv axes
        ray_u=np.cross(x_vec,desired_axis)/np.linalg.norm(np.cross(x_vec,desired_axis))

    elif (abs(y_orth)>=abs(x_orth)) and (abs(y_orth)>abs(z_orth)):
        #use y-axis to establish ray uv axes
        ray_u=np.cross(y_vec,desired_axis)/np.linalg.norm(np.cross(y_vec,desired_axis))

    elif (abs(z_orth)>=abs(x_orth)) and (abs(z_orth)>=abs(y_orth)):
        #use z-axis
        ray_u=np.cross(z_vec,desired_axis)/np.linalg.norm(np.cross(z_vec,desired_axis))

    #ray_v=np.cross(desired_axis,ray_u)
    conformal_vector=ray_u
    return conformal_vector

#@njit(cache=True, nogil=True)
def calculate_conformalVectors(desired_E_axis,source_normals):
    #based upon the provided source normal vectors and the desired polrization axis, calculate the conformal E vectors required for conformal current sources.
    conformal_E_vectors=np.zeros((source_normals.shape[0],3),dtype=np.float32)
    temp_axis=np.zeros((source_normals.shape[0],3),dtype=np.float32)
    for normal_inc in range(len(source_normals)):
        #generate a normalised orthogonal vector

        if (np.linalg.norm(np.cross(desired_E_axis,source_normals[normal_inc,:])))==0:
            conformal_E_vectors[normal_inc,:]=orthconformalvector(desired_E_axis,source_normals[normal_inc,:])
        else:
            temp_axis[normal_inc,:]=np.cross(desired_E_axis,source_normals[normal_inc,:])/np.linalg.norm(np.cross(desired_E_axis,source_normals[normal_inc,:]))
            conformal_E_vectors[normal_inc,:]=np.cross(-1*temp_axis[normal_inc,:],source_normals[normal_inc,:])/np.linalg.norm(np.cross(-1*temp_axis[normal_inc,:],source_normals[normal_inc,:]))

    return conformal_E_vectors

def face_centric_E_vectors(sink_normals,major_axis,scatter_map):
    #expectation is that the scatter map is source_num*sink_num*3 (xyz)
    #in order to make everything easier, the major axis should be defined each time, just in case I want 'V' to be aligned with an axis other than the z direction
    #This must be converted from xyz E vectors to V H N vectors for each point, which can then be summed appropriately and returned as a new scatter_map
    new_scatter_map=np.zeros(scatter_map.shape,dtype=np.complex64)

    #V Section (defined relative to major axis)
    V_alignment=calculate_conformalVectors(major_axis, sink_normals)
    minor_axis=np.zeros((sink_normals.shape))
    H_alignment=np.zeros((sink_normals.shape))
    for normal_inc in range(minor_axis.shape[0]):
        minor_axis[normal_inc,:]=np.cross(major_axis,sink_normals[normal_inc,:])
        H_alignment[normal_inc,:]=calculate_conformalVectors(minor_axis[normal_inc,:], sink_normals[normal_inc,:].reshape(1,3))

    N_alignment=copy.deepcopy(sink_normals)

    #temporaily do via a loop, I can test vector operation later
    for source_inc in range(new_scatter_map.shape[0]):
        for sink_inc in range(new_scatter_map.shape[1]):
            new_scatter_map[source_inc,sink_inc,0]=np.dot(scatter_map[source_inc,sink_inc,:],V_alignment[sink_inc,:])
            new_scatter_map[source_inc,sink_inc,1]=np.dot(scatter_map[source_inc,sink_inc,:],H_alignment[sink_inc,:])
            new_scatter_map[source_inc,sink_inc,2]=np.dot(scatter_map[source_inc,sink_inc,:],N_alignment[sink_inc,:])

    return new_scatter_map


def definePatch(wavelength,width,length,substrate_dielectric=1,mode='Single'):
    #define a patch antenna
    #try sources, mapped a tenth of a wavelength along, and given appropriate weights
    #x_mesh=np.linspace(-length/2,length/2,np.int(np.ceil(length/(wavelength*0.1))+1))
    if mode=='Single':
        sources=np.zeros((1,3),dtype=np.float32)
        patch_normals=np.zeros((1,3),dtype=np.float32)
        patch_normals[:,2]=1
        patch_sources=o3d.geometry.PointCloud()
        patch_sources.points=o3d.utility.Vector3dVector(sources)
        patch_sources.normals=o3d.utility.Vector3dVector(patch_normals)


    patch_structure=o3d.geometry.TriangleMesh.create_box(length,
                                                      width,
                                                      1e-4)
    translate_dist=np.array([-length/2.0,-width/2.0,-(1e-4)])
    #fine_mesh=reflector1.subdivide_midpoint(3)
    patch_structure.compute_vertex_normals()
    patch_structure.paint_uniform_color([184/256,115/256,51/256])
    patch_structure.translate(translate_dist,relative=True)
    patch_weights=np.ones((sources.shape[0]),dtype=np.complex64)


    return patch_sources,patch_weights,patch_structure

def importDat(fileaddress):
    datafile=pathlib.Path(fileaddress)
    # noinspection PyTypeChecker
    temp=np.loadtxt(datafile,delimiter=',')
    freq=temp[0,4]*1e6#Hz
    planes=temp[0,0]
    phi_lower=temp[0,1]
    phi_upper=temp[0,2]
    norm=10**(temp[0,3]/20)
    phi_values=np.linspace(phi_lower,phi_upper,int(planes))
    theta_values=temp[1:int((temp.shape[0]-1)/planes)+1,0]
    Ea=np.asarray((10**(temp[1:,1]/20))*np.exp(-1j*np.radians(temp[1:,2]))).reshape(len(phi_values),len(theta_values))
    Eb=np.asarray((10**(temp[1:,3]/20))*np.exp(-1j*np.radians(temp[1:,4]))).reshape(len(phi_values),len(theta_values))
    return Ea,Eb,freq,norm,theta_values,phi_values

def Steering_Efficiency(Dtheta,Dphi,Dtot,first_dimension_angle,second_dimension_angle,angular_coverage):
    """
    Calculate Steering Efficiency for the provided pattern, in radians

    Parameters
    ----------
    Dtheta : TYPE
        DESCRIPTION.
    Dphi : TYPE
        DESCRIPTION.
    Dtot : TYPE
        DESCRIPTION.
    angular coverage : TYPE
        DESCRIPTION.

    Returns
    -------
    setheta : TYPE
        DESCRIPTION.
    sephi : TYPE
        DESCRIPTION.
    setot : TYPE
        DESCRIPTION.

    """
    with np.errstate(divide='ignore'):
        a_index=10*np.log10(np.abs(Dtheta))>=(10*np.log10(np.nanmax(np.abs(Dtheta)))-3)
        b_index=10*np.log10(np.abs(Dphi))>=(10*np.log10(np.nanmax(np.abs(Dphi)))-3)
        tot_index=10*np.log10(np.abs(Dtot))>=(10*np.log10(np.nanmax(np.abs(Dtot)))-3)
        setheta=(np.sum(a_index)/(Dtheta.shape[0]*Dtheta.shape[1]))*100
        sephi=(np.sum(b_index)/(Dphi.shape[0]*Dphi.shape[1]))*100
        setot=(np.sum(tot_index)/(Dtot.shape[0]*Dtot.shape[1]))*100
        #setheta=((np.sum(a_index)*(first_dimension_angle*second_dimension_angle))/angular_coverage)*100
        #sephi=((np.sum(b_index)*(first_dimension_angle*second_dimension_angle))/angular_coverage)*100
        #setot=((np.sum(tot_index)*(first_dimension_angle*second_dimension_angle))/angular_coverage)*100

    return setheta,sephi,setot

@njit(cache=True, nogil=True)
def WavefrontWeights(source_coords,steering_vector,wavelength):
    #calculate the weights for a given set of element coordinates, wavelength, and steering vector (cartesian)
    weights=np.zeros((source_coords.shape[0]),dtype=np.float64)
    #calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    #dist=distance.cdist(source_coords,(steering_vector*1e9))
    #_,_,_,dist=calc_dv(source_coords,(steering_vector*1e9))
    dist=np.zeros((source_coords.shape[0]),dtype=np.float32)
    dist=np.sqrt(np.abs((source_coords[:,0]-steering_vector.ravel()[0]*1e9)**2+(source_coords[:,1]-steering_vector.ravel()[1]*1e9)**2+(source_coords[:,2]-steering_vector.ravel()[2]*1e9)**2))
    #RF.fast_calc_dv(source_coords,target,dv,dist)
    dist=dist-np.min(dist)
    #calculate required time delays, and then convert to phase delays
    delays=dist/scipy.constants.speed_of_light
    weights=np.exp(-1j*2*np.pi*(scipy.constants.speed_of_light/wavelength)*delays)
    return weights

def ArbitaryCoherenceWeights(source_coords,target_coord,wavelength):
    """
    Generate Wavefront coherence weights based upon the desired wavelength and the coordinates of the target point
    """
    weights=np.zeros((len(source_coords)),dtype=np.float64)
    #calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    dist=distance.cdist(source_coords,target_coord)
    dist=dist-np.min(dist)
    #calculate required time delays, and then convert to phase delays
    delays=dist/scipy.constants.speed_of_light
    weights=np.exp(-1j*2*np.pi*(scipy.constants.speed_of_light/wavelength)*delays)
    return weights

def TimeDelayWeights(source_coords,target_coord,magnitude=1.0):
    """
    Generate time delay weights to focus on a target coordinate
    """
    weights=np.zeros((len(source_coords)),dtype=np.complex64)
    #calculate distances of coords from steering_vector by using it to calculate arbitarily distant point
    dist=distance.cdist(source_coords,target_coord)
    dist=dist-np.min(dist)
    #calculate required time delays, and then convert to phase delays
    delays=dist/scipy.constants.speed_of_light
    weights=magnitude+delays*1j
    return weights

@njit(cache=True, nogil=True)
def MRCWeights(Etheta,Ephi,command_angles,polarization_switch='Etheta',az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19)):
    #calculate the maximal ratio combining weights for a given set of element coordinates, wavelength, and command angles (az,elev)
    weights=np.zeros((Etheta.shape[0]),dtype=np.complex64)
    az_index=np.argmin(np.abs(az_range-command_angles[0]))
    elev_index=np.argmin(np.abs(elev_range-command_angles[1]))
    if polarization_switch=='Etheta':
        angle_vector=np.angle(Etheta[:,elev_index,az_index].astype(np.complex64))
    else:
        angle_vector=np.angle(Ephi[:,elev_index,az_index].astype(np.complex64))

    weights=np.exp(-1j*angle_vector)
    return weights

def OAMWeights(x,y,mode):
    #generate OAM mode weights, based upon the radial angle of each element.
    #assumed array is x directed
    angles=np.arctan2(x,y)
    weights=np.zeros((len(x)),dtype=np.complex64)
    weights=np.exp(-1j*mode*angles)
    return weights

def OAMFourier(Ex,Ey,Ez,coordinates,prime_vector,mode_limit,first_dimension,second_dimension,coord_format='AzEl'):
    #producing mode index, mode coefficiencts, and mode probabilities with the co and crosspolar (Etheta,Ephi), but can probably be done the same for Ex,Ey,Ez
    #establish coordinate set, in this case theta and phi, but would work just as well with elevation and azimuth, assume that array is propagating in the z+ direction.
    if coord_format=='xyz':
        mode_index,mode_coefficients,mode_probabilites=OAMFourierCartesian(Ex,Ey,Ez,coordinates,mode_limit,first_dimension,second_dimension)
    elif coord_format=='AzEl':
        mode_index,mode_coefficients,mode_probabilites=OAMFourierSpherical(Ex,Ey,Ez,coordinates,mode_limit,first_dimension,second_dimension)

    return mode_index,mode_coefficients,mode_probabilites

def OAMFourierCartesian(Ex,Ey,Ez,coordinates,mode_limit):
    #assume probagation is in the Ez dimension
    mode_index=np.linspace(-mode_limit,mode_limit,mode_limit*2+1)
    mode_coefficients=np.zeros((mode_index.shape[0],3),dtype=np.complex64)
    az,el,r=RF.cart2sph(coordinates[:,0], coordinates[:,1], coordinates[:,2])
    #a coefficient of mode m, at angle theta is defined in terms of the
    #integral of the phi dimension in the range 0 to 2pi.
    for oam_m in range(len(mode_index)):
        mode_coefficients[oam_m,0]=(1/(2*np.pi))*np.sum(Ex.ravel()*np.exp(1j*mode_index[oam_m]*az),axis=0)
        mode_coefficients[oam_m,1]=(1/(2*np.pi))*np.sum(Ey.ravel()*np.exp(1j*mode_index[oam_m]*az),axis=0)
        mode_coefficients[oam_m,2]=(1/(2*np.pi))*np.sum(Ez.ravel()*np.exp(1j*mode_index[oam_m]*az),axis=0)

    powers=np.sum(np.abs(mode_coefficients**2),axis=0)


    mode_probabilities=np.zeros((mode_index.shape[0],3),dtype=np.float32)
    mode_probabilities[:,0]=np.abs((1/np.sum(powers))*(mode_coefficients[:,0]**2))
    mode_probabilities[:,1]=np.abs((1/np.sum(powers))*(mode_coefficients[:,1]**2))
    mode_probabilities[:,2]=np.abs((1/np.sum(powers))*(mode_coefficients[:,2]**2))

    return mode_index,mode_coefficients,mode_probabilities

def OAMFourierSpherical(Ex,Ey,Ez,coordinates,mode_limit,az_range,elev_range):
    #assume probagation is in the Ez dimension
    mode_index=np.linspace(-mode_limit,mode_limit,mode_limit*2+1)
    mode_coefficients=np.zeros((mode_index.shape[0],len(elev_range),3))
    #a coefficient of mode m, at angle theta is defined in terms of the
    #integral of the azimuth dimension in the range -pi to pi.
    for oam_m in range(mode_index):
        #mode_coefficients(oam_m,:,1)=(1/(2*pi))*sum(CoPolar(:,1:theta_limit).*exp(-sqrt(-1)*mode_index(oam_m)*p'*ones(1,theta_limit)));
        #mode_coefficients(oam_m,:,2)=(1/(2*pi))*sum(CrossPolar(:,1:theta_limit).*exp(-sqrt(-1)*mode_index(oam_m)*p'*ones(1,theta_limit)));
        mode_coefficients[oam_m,:,0]=(1/(2*np.pi))*np.sum(Ex[:,:]*np.exp(-1j*mode_index[oam_m]),axis=0)

     #copolar_power=sum(sum(abs(mode_coefficients(:,:,1)).^2));
     #crosspolar_power=sum(sum(abs(mode_coefficients(:,:,2)).^2));

    mode_probabilities=np.zeros((mode_index.shape[0],2),dtype=np.float32)
    #mode_prob(:,1)=(1/sum([copolar_power,crosspolar_power]))*sum(abs(mode_coefficients(:,:,1)').^2);
    #mode_prob(:,2)=(1/sum([copolar_power,crosspolar_power]))*sum(abs(mode_coefficients(:,:,2)').^2);

    return mode_index,mode_coefficients,mode_probabilities

#@njit(parallel=True,cache=True, nogil=True)
def MaximumDirectivityMap(Etheta,Ephi,source_coords,wavelength,az_res,elev_res,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-180.0,180.0,19),forming='Total',total_solid_angle=(4*np.pi),phase_resolution=24):
    directivity_map=np.zeros((elev_res,az_res,3))
    command_angles=np.zeros((2),dtype=np.float32)
    for az_inc in range(az_res):
        for elev_inc in range(elev_res):
            command_angles[0]=az_range[az_inc]
            command_angles[1]=elev_range[elev_inc]
            steering_vector=np.zeros((1,3))
            steering_vector[0,0],steering_vector[0,1],steering_vector[0,2]=RF.sph2cart(np.radians(command_angles[0]), np.radians(command_angles[1]), 1)
            WS_weights=WavefrontWeights(source_coords,steering_vector,wavelength)
            MRC_weights=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range)
            MRC_weights2=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range,polarization_switch='Ephi')
            if phase_resolution<=12:
                WS_weights=WeightTruncation(WS_weights,phase_resolution)
                MRC_weights=WeightTruncation(MRC_weights,phase_resolution)
                MRC_weights2=WeightTruncation(MRC_weights2,phase_resolution)

            Ethetabeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
            Ephibeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
            Ethetabeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
            Ephibeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
            Ethetabeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
            Ephibeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
            Dtheta,Dphi,Dtot,_=directivity_transform(Ethetabeamformed,Ephibeamformed,az_range=az_range,elev_range=elev_range)
            Dtheta2,Dphi2,Dtot2,_=directivity_transform(Ethetabeamformed2,Ephibeamformed2,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
            Dtheta3,Dphi3,Dtot3,_=directivity_transform(Ethetabeamformed3,Ephibeamformed3,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
            if forming=='Total':
                comparitor=np.asarray([Dtot[elev_inc,az_inc],Dtot2[elev_inc,az_inc],Dtot3[elev_inc,az_inc]])
            elif forming=='Etheta':
                comparitor=np.asarray([Dtheta[elev_inc,az_inc],Dtheta2[elev_inc,az_inc],Dtheta3[elev_inc,az_inc]])
            elif forming=='Ephi':
                comparitor=np.asarray([Dphi[elev_inc,az_inc],Dphi2[elev_inc,az_inc],Dphi3[elev_inc,az_inc]])

            if np.any(np.isnan(comparitor)):
                print('error')

            best_index=np.where(comparitor==np.max(comparitor))[0][0]
            if best_index==0:
                directivity_map[elev_inc,az_inc,0]=Dtheta[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,1]=Dphi[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,2]=Dtot[elev_inc,az_inc]
            elif best_index==1:
                directivity_map[elev_inc,az_inc,0]=Dtheta2[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,1]=Dphi2[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,2]=Dtot2[elev_inc,az_inc]
            elif best_index==2:
                directivity_map[elev_inc,az_inc,0]=Dtheta3[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,1]=Dphi3[elev_inc,az_inc]
                directivity_map[elev_inc,az_inc,2]=Dtot3[elev_inc,az_inc]

    return directivity_map

@njit(parallel=False,cache=True, nogil=True)
def MaximumDirectivityMapDiscrete(Etheta,Ephi,source_coords,wavelength,az_res,elev_res,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-180.0,180.0,19),forming='Total',total_solid_angle=(4*np.pi),phase_resolution=np.asarray([24])):
    directivity_map=np.zeros((elev_res,az_res,3,phase_resolution.shape[0]),dtype=np.float32)
    command_angles=np.zeros((2),dtype=np.float32)
    for az_inc in range(az_res):
        for elev_inc in range(elev_res):
            inc_res=0
            for res_inc in range(phase_resolution.shape[0]):
                resolution=phase_resolution[res_inc]
                command_angles[0]=az_range[az_inc]
                command_angles[1]=elev_range[elev_inc]
                steering_vector=np.zeros((1,3))
                steering_vector[0,0],steering_vector[0,1],steering_vector[0,2]=RF.sph2cart(np.radians(command_angles[0]), np.radians(command_angles[1]), 1)
                WS_weights=WavefrontWeights(source_coords,steering_vector,wavelength)
                MRC_weights=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range)
                MRC_weights2=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range,polarization_switch='Ephi')

                WS_weights=WeightTruncation(WS_weights,resolution)
                MRC_weights=WeightTruncation(MRC_weights,resolution)
                MRC_weights2=WeightTruncation(MRC_weights2,resolution)

                Ethetabeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Dtheta,Dphi,Dtot,_=directivity_transform(Ethetabeamformed,Ephibeamformed,az_range=az_range,elev_range=elev_range)
                Dtheta2,Dphi2,Dtot2,_=directivity_transform(Ethetabeamformed2,Ephibeamformed2,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                Dtheta3,Dphi3,Dtot3,_=directivity_transform(Ethetabeamformed3,Ephibeamformed3,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                if forming=='Total':
                    comparitor=np.asarray([Dtot[elev_inc,az_inc],Dtot2[elev_inc,az_inc],Dtot3[elev_inc,az_inc]])
                elif forming=='Etheta':
                    comparitor=np.asarray([Dtheta[elev_inc,az_inc],Dtheta2[elev_inc,az_inc],Dtheta3[elev_inc,az_inc]])
                elif forming=='Ephi':
                    comparitor=np.asarray([Dphi[elev_inc,az_inc],Dphi2[elev_inc,az_inc],Dphi3[elev_inc,az_inc]])

                if np.any(np.isnan(comparitor)):
                    print('error')

                best_index=np.where(comparitor==np.max(comparitor))[0][0]
                if best_index==0:
                    directivity_map[elev_inc,az_inc,0,inc_res]=Dtheta[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,1,inc_res]=Dphi[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,2,inc_res]=Dtot[elev_inc,az_inc]
                elif best_index==1:
                    directivity_map[elev_inc,az_inc,0,inc_res]=Dtheta2[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,1,inc_res]=Dphi2[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,2,inc_res]=Dtot2[elev_inc,az_inc]
                elif best_index==2:
                    directivity_map[elev_inc,az_inc,0,inc_res]=Dtheta3[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,1,inc_res]=Dphi3[elev_inc,az_inc]
                    directivity_map[elev_inc,az_inc,2,inc_res]=Dtot3[elev_inc,az_inc]

                inc_res+=1

    return directivity_map

@njit(cache=True, nogil=True)
def MaximumfieldMapDiscrete(Etheta,Ephi,source_coords,wavelength,az_res,elev_res,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-180.0,180.0,19),forming='Total',total_solid_angle=(4*np.pi),phase_resolution=[24]):
    efield_map=np.zeros((elev_res,az_res,3,len(phase_resolution)),dtype=np.complex64)
    command_angles=np.zeros((2),dtype=np.float32)
    for az_inc in prange(az_res):
        for elev_inc in range(elev_res):
            inc_res=0
            for resolution in phase_resolution:
                command_angles[0]=az_range[az_inc]
                command_angles[1]=elev_range[elev_inc]
                steering_vector=np.zeros((1,3))
                steering_vector[0,0],steering_vector[0,1],steering_vector[0,2]=RF.sph2cart(np.radians(command_angles[0]), np.radians(command_angles[1]), 1)
                WS_weights=WavefrontWeights(source_coords,steering_vector,wavelength)
                MRC_weights=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range)
                MRC_weights2=MRCWeights(Etheta, Ephi, command_angles,az_range=az_range,elev_range=elev_range,polarization_switch='Ephi')

                WS_weights=WeightTruncation(WS_weights,resolution)
                MRC_weights=WeightTruncation(MRC_weights,resolution)
                MRC_weights2=WeightTruncation(MRC_weights2,resolution)

                Ethetabeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed=np.sum(MRC_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed2=np.sum(MRC_weights2.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Ethetabeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Etheta,axis=0)
                Ephibeamformed3=np.sum(WS_weights.reshape(Etheta.shape[0],1,1)*Ephi,axis=0)
                Etotal=Ethetabeamformed**2+Ephibeamformed**2
                Etotal2=Ethetabeamformed2**2+Ephibeamformed2**2
                Etotal3=Ethetabeamformed3**2+Ephibeamformed3**2
                #Dtheta,Dphi,Dtot,_=directivity_transform(Ethetabeamformed,Ephibeamformed,az_range=az_range,elev_range=elev_range)
                #Dtheta2,Dphi2,Dtot2,_=directivity_transform(Ethetabeamformed2,Ephibeamformed2,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                #Dtheta3,Dphi3,Dtot3,_=directivity_transform(Ethetabeamformed3,Ephibeamformed3,az_range=az_range,elev_range=elev_range,total_solid_angle=total_solid_angle)
                if forming=='Total':
                    comparitor=np.asarray([Etotal[elev_inc,az_inc],Etotal2[elev_inc,az_inc],Etotal3[elev_inc,az_inc]])
                if forming=='Etheta':
                    comparitor=np.asarray([Ethetabeamformed[elev_inc,az_inc],Ethetabeamformed2[elev_inc,az_inc],Ethetabeamformed3[elev_inc,az_inc]])
                elif forming=='Ephi':
                    comparitor=np.asarray([Ephibeamformed[elev_inc,az_inc],Ephibeamformed2[elev_inc,az_inc],Ephibeamformed3[elev_inc,az_inc]])

                if np.any(np.isnan(comparitor)):
                    print('error')

                best_index=np.where(comparitor==np.max(comparitor))[0][0]
                if best_index==0:
                    efield_map[elev_inc,az_inc,0,inc_res]=Ethetabeamformed[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,1,inc_res]=Ephibeamformed[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,2,inc_res]=Etotal[elev_inc,az_inc]
                elif best_index==1:
                    efield_map[elev_inc,az_inc,0,inc_res]=Ethetabeamformed2[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,1,inc_res]=Ephibeamformed2[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,2,inc_res]=Etotal2[elev_inc,az_inc]
                elif best_index==2:
                    efield_map[elev_inc,az_inc,0,inc_res]=Ethetabeamformed3[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,1,inc_res]=Ephibeamformed3[elev_inc,az_inc]
                    efield_map[elev_inc,az_inc,2,inc_res]=Etotal3[elev_inc,az_inc]

                inc_res+=1

    return efield_map

def PatternTransform3D(norm_magnitudes,min_level=-40,linearswitch='log',logswitch='amplitude',az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19),shell_range=1.0,centre=np.zeros((3),dtype=np.float32)):
    #create point cloud, coloured and radius scaled with the magnitudes
    if logswitch=='amplitude':
        log_multiplier=20.0
    elif logswitch=='power':
        log_multiplier=10.0


    azaz,elel=np.meshgrid(az_range,elev_range)
    sinks=np.zeros((len(np.ravel(azaz)),3),dtype=np.float32)
    if linearswitch=='log':
        norm_ranges=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
        norm_ranges[norm_ranges<min_level]=min_level
        norm_ranges=((norm_ranges-min_level)/np.abs(min_level))*shell_range
    elif linearswitch=='linear':
        norm_ranges=np.ravel(norm_magnitudes)*shell_range

    sinks[:,0],sinks[:,1],sinks[:,2]=RF.azeltocart(np.ravel(azaz),np.ravel(elel),np.ravel(norm_ranges))
    point_cloud=RF.points2pointcloud(sinks+centre)

    #normdata
    logdata=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
    viridis = cm.get_cmap('viridis', 40)
    np_colors=viridis(np.ravel(norm_magnitudes))
    point_cloud.colors = o3d.utility.Vector3dVector(np_colors[:,0:3])

    return point_cloud

def PatternTransformArbitary(norm_magnitudes,coordinates,min_level=-40,linearswitch='log',logswitch='amplitude'):
    #create point cloud, coloured and radius scaled with the magnitudes
    if logswitch=='amplitude':
        log_multiplier=20.0
    elif logswitch=='power':
        log_multiplier=10.0


    point_cloud=coordinates
    if linearswitch=='log':
        norm_ranges=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
        norm_ranges[norm_ranges<min_level]=min_level
        norm_ranges=((norm_ranges-min_level)/np.abs(min_level))*1.0
    elif linearswitch=='linear':
        norm_ranges=np.ravel(norm_magnitudes)*1.0

    #normdata
    logdata=log_multiplier*np.log10(np.ravel(norm_magnitudes)+1e-24)
    viridis = cm.get_cmap('viridis', 1000)
    np_colors=viridis(np.ravel(norm_magnitudes))
    point_cloud.colors = o3d.utility.Vector3dVector(np_colors[:,0:3])

    return point_cloud

def PatternTransformPhase3D(norm_magnitudes,phases,min_level=-40,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19),shell_range=1.0):
    #create point cloud, radius scaled with the magnitudes, and coloured by the phases
    azaz,elel=np.meshgrid(az_range,elev_range)
    sinks=np.zeros((len(np.ravel(azaz)),3),dtype=np.float32)
    norm_ranges=20*np.log10(np.ravel(norm_magnitudes)+1e-24)
    norm_ranges[norm_ranges<min_level]=min_level
    norm_ranges=((norm_ranges-min_level)/np.abs(min_level))*shell_range
    sinks[:,0],sinks[:,1],sinks[:,2]=RF.azeltocart(np.ravel(azaz),np.ravel(elel),np.ravel(norm_ranges))
    point_cloud=RF.points2pointcloud(sinks)

    #normdata in interval - pi to pi, and convert to interval from 0 to 1
    logdata=(np.ravel(phases)+np.pi)/(np.pi*2)
    viridis = cm.get_cmap('viridis', 181)
    np_colors=viridis(np.ravel(logdata))
    point_cloud.colors = o3d.utility.Vector3dVector(np_colors[:,0:3])

    return point_cloud

@njit(cache=True, nogil=True)
def directivity_transform(Etheta,Ephi,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19),total_solid_angle=(4*np.pi)):
    #transform Etheta and Ephi data into antenna directivity
    #directivity is defined in terms of the power radiated in a specific direction, over the average radiated power
    #power per unit solid angle
    Dmax=np.zeros((3),dtype=np.float32)
    Umax=np.zeros((3),dtype=np.float32)
    Utheta=np.abs(Etheta**2)
    Uphi=np.abs(Ephi**2)
    Utotal=np.abs(Etheta**2)+np.abs(Ephi**2)
    power_sum=0.0
    temp_vector=np.zeros((4),dtype=np.float32)
    for elinc in range(len(elev_range)-1):
        for azinc in range(len(az_range)-1):
            temp_vector[0]=Utheta[elinc,azinc]
            temp_vector[1]=Utheta[elinc+1,azinc]
            temp_vector[2]=Utheta[elinc,azinc+1]
            temp_vector[3]=Utheta[elinc+1,azinc+1]
            r=np.mean(temp_vector)
            power_sum=power_sum+r*np.abs(np.sin(np.radians(az_range[azinc]+az_range[azinc+1])/2.0))

    etheta_power_sum=power_sum
    for elinc in range(len(elev_range)-1):
        for azinc in range(len(az_range)-1):
            temp_vector[0]=Uphi[elinc,azinc]
            temp_vector[1]=Uphi[elinc+1,azinc]
            temp_vector[2]=Uphi[elinc,azinc+1]
            temp_vector[3]=Uphi[elinc+1,azinc+1]
            r=np.mean(temp_vector)
            power_sum=power_sum+r*np.abs(np.sin(np.radians(az_range[azinc]+az_range[azinc+1])/2.0))

    ephi_power_sum=power_sum-etheta_power_sum
    Umax[0]=np.nanmax(Utheta)
    Umax[1]=np.nanmax(Uphi)
    Umax[2]=np.nanmax(Utotal)
    Uav = power_sum*((np.radians(np.abs(az_range[1]-az_range[0])))*(np.radians(np.abs(elev_range[1]-elev_range[0]))))/(total_solid_angle)
    Dmax=Umax/Uav
    Dtheta=Utheta/Uav
    Dphi=Uphi/Uav
    Dtot=Utotal/Uav

    return Dtheta,Dphi,Dtot,Dmax

@njit(cache=True, nogil=True)
def directivity_transformv2(Etheta,Ephi,az_range=np.linspace(-180.0,180.0,19),elev_range=np.linspace(-90.0,90.0,19),total_solid_angle=(4*np.pi)):
    #transform Etheta and Ephi data into antenna directivity
    #directivity is defined in terms of the power radiated in a specific direction, over the average radiated power
    #power per unit solid angle
    Dmax=np.zeros((3),dtype=np.float32)
    Umax=np.zeros((3),dtype=np.float32)
    Utheta=np.abs(Etheta**2)
    Uphi=np.abs(Ephi**2)
    Utotal=np.abs(Etheta**2)+np.abs(Ephi**2)
    power_sum=0.0
    temp_vector=np.zeros((4),dtype=np.float32)
    temp_vector2 = np.zeros((4), dtype=np.float32)
    for elinc in range(len(elev_range)-1):
        for azinc in range(len(az_range)-1):
            temp_vector[0]=Utheta[elinc,azinc]
            temp_vector[1]=Utheta[elinc+1,azinc]
            temp_vector[2]=Utheta[elinc,azinc+1]
            temp_vector[3]=Utheta[elinc+1,azinc+1]
            temp_vector2[0] = Uphi[elinc, azinc]
            temp_vector2[1] = Uphi[elinc + 1, azinc]
            temp_vector2[2] = Uphi[elinc, azinc + 1]
            temp_vector2[3] = Uphi[elinc + 1, azinc + 1]
            r1=np.mean(temp_vector)
            r2 = np.mean(temp_vector2)
            power_sum=power_sum+r1*np.abs(np.sin(np.radians(az_range[azinc]+az_range[azinc+1])/2.0))+r2*np.abs(np.sin(np.radians(az_range[azinc]+az_range[azinc+1])/2.0))

    Umax[0]=np.nanmax(Utheta)
    Umax[1]=np.nanmax(Uphi)
    Umax[2]=np.nanmax(Utotal)
    Uav = power_sum*((np.radians(az_range[1]-az_range[0]))*(np.radians(elev_range[1]-elev_range[0])))/total_solid_angle
    Dmax=Umax/Uav
    Dtheta=Utheta/Uav
    Dphi=Uphi/Uav
    Dtot=Utotal/Uav

    return Dtheta,Dphi,Dtot,Dmax

@njit(cache=True, nogil=True)
def WeightTruncation(weights,resolution):
    quant=2**resolution
    levels=np.zeros((weights.shape),dtype=np.float32)
    #shift numpy complex angles from -pi to pi, into 0 to 2pi, then quantise for `quant' levels
    levels=np.round((quant-1)*((np.angle(weights)+np.pi)/(2*np.pi)),0,levels)/(quant-1)
    new_weights=np.abs(weights)*np.exp(1j*((levels*2*np.pi)-np.pi))
    return new_weights

@njit(cache=True, nogil=True)
def pathloss(lengths,wavelength):
    #convert length to loss and phase terms
    channel=np.zeros((lengths.shape[0]),dtype=np.complex64)
    wave_vector=(2*np.pi)/wavelength
    channel=(np.exp(lengths*wave_vector*1j))*(wavelength/(4*np.pi*lengths))
    return channel

@njit(cache=True, nogil=True)
def pathlossv2(lengths,wavelength):
    #convert length to loss and phase terms
    channel=np.zeros((lengths.shape[0]),dtype=np.complex64)
    wave_vector=(2*np.pi)/wavelength
    channel[lengths!=0]=(np.exp(lengths[lengths!=0]*wave_vector*1j))*(wavelength/(4*np.pi*(lengths[lengths!=0]**2)))
    channel[lengths==0]=1
    return channel

#@njit(cache=True, nogil=True)
def losChannel(source_point,source_normal,source_weight,source_information,sink_point,sink_normal,sink_weight,sink_information,scattering_coefficient,wavelength):
    """

    Parameters
    ----------
    source_point : 1*3 float
        source coordinates (xyz)
    source_normal : 1*3 float
        source normal (xyz)
    sink_point : 1*3 float
        sink coordinates (xyz)
    sink_normal : 1*3 float
        sink normal (xyz)
    channel : 1*3 complex
        excitation function in global reference frame Ex,Ey,Ez
    wavelength : 1 float
        wavelength of interest

    Need to define stokes parameters of each vector, thus turning all weights into stokes vectors
    Returns
    -------
    channel : as defined

    """
    wave_vector=(2*np.pi)/wavelength
    channel=np.zeros((3),dtype=np.complex64)
    outgoing_dir=np.zeros((3),dtype=np.float32)
    lengths=np.zeros((1),dtype=np.float32)
    outgoing_dir[0],outgoing_dir[1],outgoing_dir[2],lengths=RF.calc_dv(source_point,sink_point)
    if lengths==0:
        loss1=1.0
    else:
        loss1=(np.exp(lengths*wave_vector*1j))*(wavelength/(4*np.pi*(lengths)))
    ray_field=launchtransform(source_normal,outgoing_dir,source_weight,source_information)*scattering_coefficient
    channel=ray_field*sink_weight*loss1

    return channel

@njit(cache=True, nogil=True)
def losChannelv2(source_point,source_normal,source_weight,source_information,sink_point,sink_normal,sink_weight,sink_information,wavelength):
    """

    Parameters
    ----------
    source_point : 1*3 float
        source coordinates (xyz)
    source_normal : 1*3 float
        source normal (xyz)
    sink_point : 1*3 float
        sink coordinates (xyz)
    sink_normal : 1*3 float
        sink normal (xyz)
    channel : 1*3 complex
        excitation function in global reference frame Ex,Ey,Ez
    wavelength : 1 float
        wavelength of interest

    Need to define stokes parameters of each vector, thus turning all weights into stokes vectors
    Returns
    -------
    channel : as defined

    """
    wave_vector=(2*np.pi)/wavelength
    channel=np.zeros((3),dtype=np.complex64)
    outgoing_dir=np.zeros((3),dtype=np.float32)
    lengths=np.zeros((1),dtype=np.float32)
    outgoing_dir[0],outgoing_dir[1],outgoing_dir[2],lengths=RF.calc_dv(source_point,sink_point)
    if lengths==0:
        loss1=1.0
    else:
        loss1=(np.exp(lengths*wave_vector*1j))*(wavelength/(4*np.pi*(lengths)))
    ray_field=launchtransform(source_normal,outgoing_dir,source_weight,source_information)#*ray_phase
    channel=ray_field*sink_weight*loss1

    return channel, lengths/scipy.constants.c

@njit(cache=True, nogil=True)
def losplus1Channel(source_point,source_normal,source_weight,source_information,scatter_point,scatter_normal,scatter_weight,scatter_information,sink_point,sink_normal,sink_weight,sink_information,scattering_coefficient,wavelength):
    """

    Parameters
    ----------
    source_point : 1*3 float
        source coordinates (xyz)
    source_normal : 1*3 float
        source normal (xyz)
    sink_point : 1*3 float
        sink coordinates (xyz)
    sink_normal : 1*3 float
        sink normal (xyz)
    channel : 1*2 complex
        channel propagation coefficients for local h, local v from the source to the sink
    wavelength : 1 float
        wavelength of interest

    Returns
    -------
    channel : as defined

    """
    wave_vector=(2*np.pi)/wavelength
    channel=np.zeros((3),dtype=np.complex64)
    outgoing_dir=np.zeros((3),dtype=np.float32)
    scatter_outgoing_dir=np.zeros((3),dtype=np.float32)
    lengths=np.zeros((1),dtype=np.float32)
    lengths2=np.zeros((1),dtype=np.float32)
    outgoing_dir[0],outgoing_dir[1],outgoing_dir[2],lengths=RF.calc_dv(source_point,scatter_point)
    ray_field=launchtransform(source_normal,outgoing_dir,source_weight,source_information)*scattering_coefficient
    scatter_outgoing_dir[0],scatter_outgoing_dir[1],scatter_outgoing_dir[2],lengths2=RF.calc_dv(scatter_point,sink_point)
    if lengths==0 or lengths2==0:
        loss2=1.0
    else:
        loss2=(np.exp((lengths+lengths2)*wave_vector*1j))*(wavelength/(4*np.pi*(lengths+lengths2)))

    channel=launchtransform(scatter_normal,scatter_outgoing_dir,ray_field*sink_weight,scatter_information,source=False,arrival_vector=outgoing_dir*-1)*loss2*scattering_coefficient

    return channel

@njit(cache=True, nogil=True)
def losplus1Channelv2(source_point,source_normal,source_weight,source_information,scatter_point,scatter_normal,scatter_weight,scatter_information,sink_point,sink_normal,sink_weight,sink_information,wavelength):
    """

    Parameters
    ----------
    source_point : 1*3 float
        source coordinates (xyz)
    source_normal : 1*3 float
        source normal (xyz)
    sink_point : 1*3 float
        sink coordinates (xyz)
    sink_normal : 1*3 float
        sink normal (xyz)
    channel : 1*2 complex
        channel propagation coefficients for local h, local v from the source to the sink
    wavelength : 1 float
        wavelength of interest

    Returns
    -------
    channel : as defined

    """
    wave_vector=(2*np.pi)/wavelength
    channel=np.zeros((3),dtype=np.complex64)
    outgoing_dir=np.zeros((3),dtype=np.float32)
    scatter_outgoing_dir=np.zeros((3),dtype=np.float32)
    lengths=np.zeros((1),dtype=np.float32)
    lengths2=np.zeros((1),dtype=np.float32)
    outgoing_dir[0],outgoing_dir[1],outgoing_dir[2],lengths=RF.calc_dv(source_point,scatter_point)
    ray_field=launchtransform(source_normal,outgoing_dir,source_weight,source_information)#*ray_phase
    scatter_outgoing_dir[0],scatter_outgoing_dir[1],scatter_outgoing_dir[2],lengths2=RF.calc_dv(scatter_point,sink_point)
    if lengths==0 or lengths2==0:
        loss2=1.0
    else:
        loss2=(np.exp((lengths+lengths2)*wave_vector*1j))*(wavelength/(4*np.pi*(lengths+lengths2)))

    channel=launchtransform(scatter_normal,scatter_outgoing_dir,ray_field*sink_weight,scatter_information,source=False,arrival_vector=outgoing_dir*-1)*loss2

    return channel, (lengths+lengths2)/scipy.constants.c

@njit(cache=True, nogil=True)
def losplus2Channel(source_point,source_normal,source_weight,source_information,scatter_point,scatter_normal,scatter_weight,scatter_information,scatter_point2,scatter_normal2,scatter_weight2,scatter_information2,sink_point,sink_normal,sink_weight,sink_information,scattering_coefficient,wavelength):
    """

    Parameters
    ----------
    source_point : 1*3 float
        source coordinates (xyz)
    source_normal : 1*3 float
        source normal (xyz)
    sink_point : 1*3 float
        sink coordinates (xyz)
    sink_normal : 1*3 float
        sink normal (xyz)
    channel : 1*2 complex
        channel propagation coefficients for local h, local v from the source to the sink
    wavelength : 1 float
        wavelength of interest

    Returns
    -------
    channel : as defined

    """
    wave_vector=(2*np.pi)/wavelength
    channel=np.zeros((3),dtype=np.complex64)
    outgoing_dir=np.zeros((3),dtype=np.float32)
    scatter_outgoing_dir=np.zeros((3),dtype=np.float32)
    scatter_outgoing_dir2=np.zeros((3),dtype=np.float32)
    lengths=np.zeros((1),dtype=np.float32)
    lengths2=np.zeros((1),dtype=np.float32)
    lengths3=np.zeros((1),dtype=np.float32)
    #scatter from source to point 1
    outgoing_dir[0],outgoing_dir[1],outgoing_dir[2],lengths=RF.calc_dv(source_point,scatter_point)
    ray_field1=launchtransform(source_normal,outgoing_dir,source_weight,source_information)*scattering_coefficient
    #scatter from point 1 to point 2
    scatter_outgoing_dir[0],scatter_outgoing_dir[1],scatter_outgoing_dir[2],lengths2=RF.calc_dv(scatter_point,scatter_point2)
    ray_field2=launchtransform(scatter_normal,scatter_outgoing_dir,ray_field1*scatter_weight,scatter_information,source=False,arrival_vector=outgoing_dir*-1)*scattering_coefficient
    scatter_outgoing_dir2[0],scatter_outgoing_dir2[1],scatter_outgoing_dir2[2],lengths3=RF.calc_dv(scatter_point2,sink_point)
    if lengths==0 or lengths2==0 or lengths3==0:
        loss3=1.0
    else:
        loss3=(np.exp((lengths+lengths2+lengths3)*wave_vector*1j))*(wavelength/(4*np.pi*(lengths+lengths2+lengths3)))

    channel=launchtransform(scatter_normal2,scatter_outgoing_dir,ray_field2*sink_weight,scatter_information2,source=False,arrival_vector=scatter_outgoing_dir*-1)*loss3*scattering_coefficient

    return channel

@njit(cache=True, nogil=True)
def losplus2Channelv2(source_point,source_normal,source_weight,source_information,scatter_point,scatter_normal,scatter_weight,scatter_information,scatter_point2,scatter_normal2,scatter_weight2,scatter_information2,sink_point,sink_normal,sink_weight,sink_information,wavelength):
    """

    Parameters
    ----------
    source_point : 1*3 float
        source coordinates (xyz)
    source_normal : 1*3 float
        source normal (xyz)
    sink_point : 1*3 float
        sink coordinates (xyz)
    sink_normal : 1*3 float
        sink normal (xyz)
    channel : 1*2 complex
        channel propagation coefficients for local h, local v from the source to the sink
    wavelength : 1 float
        wavelength of interest

    Returns
    -------
    channel : as defined

    """
    wave_vector=(2*np.pi)/wavelength
    channel=np.zeros((3),dtype=np.complex64)
    outgoing_dir=np.zeros((3),dtype=np.float32)
    scatter_outgoing_dir=np.zeros((3),dtype=np.float32)
    scatter_outgoing_dir2=np.zeros((3),dtype=np.float32)
    lengths=np.zeros((1),dtype=np.float32)
    lengths2=np.zeros((1),dtype=np.float32)
    lengths3=np.zeros((1),dtype=np.float32)
    #scatter from source to point 1
    outgoing_dir[0],outgoing_dir[1],outgoing_dir[2],lengths=RF.calc_dv(source_point,scatter_point)
    ray_field1=launchtransform(source_normal,outgoing_dir,source_weight,source_information)#*ray_phase
    #scatter from point 1 to point 2
    scatter_outgoing_dir[0],scatter_outgoing_dir[1],scatter_outgoing_dir[2],lengths2=RF.calc_dv(scatter_point,scatter_point2)
    ray_field2=launchtransform(scatter_normal,scatter_outgoing_dir,ray_field1*scatter_weight,scatter_information,source=False,arrival_vector=outgoing_dir*-1)
    scatter_outgoing_dir2[0],scatter_outgoing_dir2[1],scatter_outgoing_dir2[2],lengths3=RF.calc_dv(scatter_point2,sink_point)
    if lengths==0 or lengths2==0 or lengths3==0:
        loss3=1.0
    else:
        loss3=(np.exp((lengths+lengths2+lengths3)*wave_vector*1j))*(wavelength/(4*np.pi*(lengths+lengths2+lengths3)))

    channel=launchtransform(scatter_normal2,scatter_outgoing_dir,ray_field2*sink_weight,scatter_information2,source=False,arrival_vector=scatter_outgoing_dir*-1)*loss3

    return channel,(lengths+lengths2+lengths3)/scipy.constants.c

@njit(cache=True, nogil=True)
def sourcedefinition(departure_vector,local_E_vector,local_information):
     #establish source correctly, including E or M vector
    final_E_vector=np.zeros((3),dtype=np.complex64)
    if local_information['Electric']:
        #source is electric current sources, so local_E_vector is correct
        final_E_vector=local_E_vector
    else:
        source_impedance=np.complex64((local_information['permeability']/local_information['permittivity'])**0.5)
        final_E_vector=np.cross(departure_vector.astype(np.complex64),local_E_vector.astype(np.complex64))/source_impedance

    return final_E_vector

@njit(cache=True, nogil=True)
def launchtransform(source_normal,departure_vector,local_E_vector,local_information,source=True,arrival_vector=np.zeros((3),dtype=np.float32)):
    """
    Launch transform maps the local E vector onto the outgoing ray. The local coordinate sysem of uv-normal axes defining the e_vector,
    which is then mapped onto the h and v transverse components of the outgoing ray
    Parameters
    ------
    source_normal : (numpy 1*3 array)
        a normalised direction vector recording the orientation of the source with respect to the global axes
    departure vector : (numpy 1*3 array)
        normalised direction vector of the departing ray
    local_E_vector : (numpy 1*3 array)
        electric field vector in uv-normal axes, representing the local `illumination function', this is not normalised, as it should be recording the field amplitude as well as directions.

    Returns
    -------
    outgoing_E_vector : (numpy 1*3 complex array)
        electric field in global axes
    """
    if not(source):
        #transform the arrived E vector to local axes for correct polarization mixing
        local_E_vector=illuminationtransform(source_normal,arrival_vector,local_E_vector,local_information)
    else:
        local_E_vector=sourcedefinition(departure_vector,local_E_vector,local_information)

    temp_E_vector=np.zeros((2),dtype=np.complex64)
    outgoing_E_vector=np.zeros((3),dtype=np.complex64)
    ray_u=np.zeros((3),dtype=np.float32)
    ray_v=np.zeros((3),dtype=np.float32)
    x_vec=np.zeros((3),dtype=np.float32)
    y_vec=np.zeros((3),dtype=np.float32)
    z_vec=np.zeros((3),dtype=np.float32)
    x_vec[0]=1
    y_vec[1]=1
    z_vec[2]=1
    #make sure ray vectors are locked on appropriate global axes
    x_orth=np.linalg.norm(np.cross(x_vec,departure_vector))
    y_orth=np.linalg.norm(np.cross(y_vec,departure_vector))
    z_orth=np.linalg.norm(np.cross(z_vec,departure_vector))
    if (abs(x_orth)>abs(y_orth)) and (abs(x_orth)>abs(z_orth)):
        #use x-axis to establish ray uv axes
        ray_u=np.cross(x_vec,departure_vector)/np.linalg.norm(np.cross(x_vec,departure_vector))

    elif (abs(y_orth)>=abs(x_orth)) and (abs(y_orth)>abs(z_orth)):
        #use y-axis to establish ray uv axes
        ray_u=np.cross(y_vec,departure_vector)/np.linalg.norm(np.cross(y_vec,departure_vector))

    elif (abs(z_orth)>=abs(x_orth)) and (abs(z_orth)>=abs(y_orth)):
        #use z-axis
        ray_u=np.cross(z_vec,departure_vector)/np.linalg.norm(np.cross(z_vec,departure_vector))


    ray_v=np.cross(departure_vector,ray_u)
    #the ray fields must be contained in ray_v and ray_u, as there can be no E field in the direction of propagation
    #so if the depature vector is 0,0,1, then Ez must be 0.
    temp_E_vector[0]=np.dot(local_E_vector.astype(np.complex64),ray_u.astype(np.complex64))
    temp_E_vector[1]=np.dot(local_E_vector.astype(np.complex64),ray_v.astype(np.complex64))
    #map ray axes onto global coordinate axes to keep everything neat
    outgoing_E_vector=np.array([temp_E_vector[0]*np.dot(x_vec,ray_u)+temp_E_vector[1]*np.dot(x_vec,ray_v),
                                temp_E_vector[0]*np.dot(y_vec,ray_u)+temp_E_vector[1]*np.dot(y_vec,ray_v),
                                temp_E_vector[0]*np.dot(z_vec,ray_u)+temp_E_vector[1]*np.dot(z_vec,ray_v)])
    return outgoing_E_vector

@njit(cache=True, nogil=True)
def illuminationtransform(source_normal,arrival_vector,arriving_E_vector,local_information):
    """
    Illuminating transform maps the local E vector onto the surface. The local coordinate sysem of uv-normal axes defining the surface and avaliable axes,
    which is then mapped back into global E vector axes
    Parameters
    ------
    source_normal : (numpy 1*3 array)
        a normalised direction vector recording the orientation of the source with respect to the global axes
    departure vector : (numpy 1*3 array)
        normalised direction vector of the departing ray
    local_E_vector : (numpy 1*3 array, complex)
        electric field vector in uv-normal axes, representing the local `illumination function', this is not normalised, as it should be recording the field amplitude as well as directions.

    Returns
    -------
    outgoing_E_vector : ((numpy 1*3 array, complex)
        electric field in the global axes direction
    """
    point_E_vector=np.zeros((2),dtype=np.complex64)
    outgoing_E_vector=np.zeros((3),dtype=np.complex64)
    point_u=np.zeros((3),dtype=np.float32)
    point_v=np.zeros((3),dtype=np.float32)
    x_vec=np.zeros((3),dtype=np.float32)
    y_vec=np.zeros((3),dtype=np.float32)
    z_vec=np.zeros((3),dtype=np.float32)
    x_vec[0]=1
    y_vec[1]=1
    z_vec[2]=1
    #make sure illuminated point vectors are locked on appropriate global axes
    x_orth=np.linalg.norm(np.cross(x_vec,source_normal))
    y_orth=np.linalg.norm(np.cross(y_vec,source_normal))
    z_orth=np.linalg.norm(np.cross(z_vec,source_normal))
    if (abs(x_orth)>abs(y_orth)) and (abs(x_orth)>abs(z_orth)):
        #use x-axis to establish ray uv axes
        point_u=np.cross(x_vec,source_normal)/np.linalg.norm(np.cross(x_vec,source_normal))

    elif (abs(y_orth)>=abs(x_orth)) and (abs(y_orth)>abs(z_orth)):
        #use y-axis to establish ray uv axes
        point_u=np.cross(y_vec,source_normal)/np.linalg.norm(np.cross(y_vec,source_normal))

    elif (abs(z_orth)>=abs(x_orth)) and (abs(z_orth)>=abs(y_orth)):
        #use z-axis
        point_u=np.cross(z_vec,source_normal)/np.linalg.norm(np.cross(z_vec,source_normal))


    point_v=np.cross(source_normal,point_u)
    #the ray fields must be contained in ray_v and ray_u, as there can be no E field in the direction of propagation
    #so if the depature vector is 0,0,1, then Ez must be 0.
    point_E_vector[0]=np.dot(arriving_E_vector.astype(np.complex64),point_u.astype(np.complex64))
    point_E_vector[1]=np.dot(arriving_E_vector.astype(np.complex64),point_v.astype(np.complex64))
    #point_E_vector[2]=np.dot(arriving_E_vector.astype(np.complex64),source_normal.astype(np.complex64))
    #map ray axes onto global coordinate axes to keep everything neat
    outgoing_E_vector=np.array([point_E_vector[0]*np.dot(x_vec,point_u)+point_E_vector[1]*np.dot(x_vec,point_v),
                                point_E_vector[0]*np.dot(y_vec,point_u)+point_E_vector[1]*np.dot(y_vec,point_v),
                                point_E_vector[0]*np.dot(z_vec,point_u)+point_E_vector[1]*np.dot(z_vec,point_v)])
    return outgoing_E_vector

#@njit(cache=True, nogil=True)
@guvectorize([(float32[:],complex64[:],complex64[:])],'(m),(n)->(m)')
def source_transform2to3(departure_vector,thetaphi_E_vector,xyz_E_vector):
    """
    This function is intended to allow for transformation of measured or
    modelled antenna patterns into the model cartesian space by linking each
    Etheta/Ephi point to it's corresponding local coordinate which is then
    rotated and the local U/V vectors for each point and associated normal
    vectors can be used to calculate the resultant Ex,Ey,Ez fields.

    Inputs
    ------
    departure vector : numpy m*3 array,
        normalised direction vector of the departing ray

    local_E_vector : numpy m*2 array,
        electric field vector in Etheta/Ephi

    local_to_global : numpy m*3

    Returns
    -------
    xyz_E_vector : numpy 1*3 complex array in Ex,Ey,Ez format

    """

    ray_u = np.zeros((3), dtype=np.float32)
    ray_v = np.zeros((3), dtype=np.float32)
    x_vec = np.zeros((3), dtype=np.float32)
    y_vec = np.zeros((3), dtype=np.float32)
    z_vec = np.zeros((3), dtype=np.float32)
    x_vec[0] = 1
    y_vec[1] = 1
    z_vec[2] = 1
    # make sure ray vectors are locked on appropriate global axes
    x_orth = np.linalg.norm(np.cross(x_vec, departure_vector))
    y_orth = np.linalg.norm(np.cross(y_vec, departure_vector))
    z_orth = np.linalg.norm(np.cross(z_vec, departure_vector))
    if (abs(x_orth) > abs(y_orth)) and (abs(x_orth) > abs(z_orth)):
        # use x-axis to establish ray uv axes
        ray_u = np.cross(x_vec, departure_vector) / np.linalg.norm(np.cross(x_vec, departure_vector))

    elif (abs(y_orth) >= abs(x_orth)) and (abs(y_orth) > abs(z_orth)):
        # use y-axis to establish ray uv axes
        ray_u = np.cross(y_vec, departure_vector) / np.linalg.norm(np.cross(y_vec, departure_vector))

    elif (abs(z_orth) >= abs(x_orth)) and (abs(z_orth) >= abs(y_orth)):
        # use z-axis
        ray_u = np.cross(z_vec, departure_vector) / np.linalg.norm(np.cross(z_vec, departure_vector))

    ray_v = np.cross(departure_vector, ray_u)
    # the ray fields must be contained in ray_v and ray_u, as there can be no E field in the direction of propagation
    # map ray axes onto global coordinate axes
    xyz_E_vector[0]=thetaphi_E_vector[0] * np.dot(x_vec, ray_u) + thetaphi_E_vector[1] * np.dot(x_vec, ray_v)
    xyz_E_vector[1]=thetaphi_E_vector[0] * np.dot(y_vec, ray_u) + thetaphi_E_vector[1] * np.dot(y_vec, ray_v)
    xyz_E_vector[2]=thetaphi_E_vector[0] * np.dot(z_vec, ray_u) + thetaphi_E_vector[1] * np.dot(z_vec, ray_v)

    #return xyz_E_vector

@guvectorize(['(float32[:],complex64[:],complex64[:],complex64[:])','(float64[:],complex128[:],complex128[:],complex128[:])'],'(n),(n),(m)->(m)',target='parallel')
def source_transform3to2(departure_vector,xyz_E_vector,thetaphi_E_vector_dummy,thetaphi_E_vector):
    """
    This function is intended to allow for transformation of measured or
    modelled antenna patterns into the model cartesian space by linking each
    Etheta/Ephi point to it's corresponding local coordinate which is then
    rotated and the local U/V vectors for each point and associated normal
    vectors can be used to calculate the resultant Ex,Ey,Ez fields.

    Inputs
    ------
    departure vector : numpy m*3 array,
        normalised direction vector of the departing ray

    local_E_vector : numpy m*2 array,
        electric field vector in Etheta/Ephi

    local_to_global : numpy m*3

    Returns
    -------
    xyz_E_vector : numpy 1*3 complex array in Ex,Ey,Ez format

    """

    ray_u = np.zeros((3), dtype=departure_vector.dtype)
    ray_v = np.zeros((3), dtype=departure_vector.dtype)
    x_vec = np.zeros((3), dtype=departure_vector.dtype)
    y_vec = np.zeros((3), dtype=departure_vector.dtype)
    z_vec = np.zeros((3), dtype=departure_vector.dtype)
    x_vec[0] = 1
    y_vec[1] = 1
    z_vec[2] = 1
    # make sure ray vectors are locked on appropriate global axes
    x_orth = np.linalg.norm(np.cross(x_vec, departure_vector))
    y_orth = np.linalg.norm(np.cross(y_vec, departure_vector))
    z_orth = np.linalg.norm(np.cross(z_vec, departure_vector))
    if (abs(x_orth) > abs(y_orth)) and (abs(x_orth) > abs(z_orth)):
        # use x-axis to establish ray uv axes
        ray_u = np.cross(x_vec, departure_vector) / np.linalg.norm(np.cross(x_vec, departure_vector))

    elif (abs(y_orth) >= abs(x_orth)) and (abs(y_orth) > abs(z_orth)):
        # use y-axis to establish ray uv axes
        ray_u = np.cross(y_vec, departure_vector) / np.linalg.norm(np.cross(y_vec, departure_vector))

    elif (abs(z_orth) >= abs(x_orth)) and (abs(z_orth) >= abs(y_orth)):
        # use z-axis
        ray_u = np.cross(z_vec, departure_vector) / np.linalg.norm(np.cross(z_vec, departure_vector))

    ray_v = np.cross(departure_vector, ray_u)
    # the ray fields must be contained in ray_v and ray_u, as there can be no E field in the direction of propagation
    # map ray axes onto global coordinate axes
    thetaphi_E_vector[0] = np.dot(xyz_E_vector.astype(np.complex64), ray_u.astype(np.complex64))
    thetaphi_E_vector[1] = np.dot(xyz_E_vector.astype(np.complex64), ray_v.astype(np.complex64))

    #return thetaphi_E_vector
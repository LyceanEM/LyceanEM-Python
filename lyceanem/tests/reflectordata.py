#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np
import open3d as o3d
import scipy.io as io
from importlib_resources import files

import lyceanem.tests.data
import lyceanem.geometry.targets as tl
import lyceanem.geometry.geometryfunctions as GF


#freq=24e9
#wavelength=3e8/freq
#measurementaddress=Path("./")
#units are mm in the laser scans
#medium_reference='Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt'
#matfile='plate_copper.mat'
#datafile='Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt'
#stream=pkg_resources.resource_stream(__name__,'data/Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt')
#temp=np.loadtxt(stream,delimiter=';',skiprows=2)

#medium_reference_reflector = o3d.geometry.PointCloud()
#medium_reference_reflector.points = o3d.utility.Vector3dVector(temp/1000)
#medium_reference_reflector.estimate_normals()
#reference_point=np.asarray([[np.min(temp[:,0]),np.min(temp[:,1]),np.min(temp[:,2])]])/1000

#downsampled_reflector=medium_reference_reflector.voxel_down_sample(voxel_size=wavelength*0.5)
#downsampled_reflector.translate(-reference_point.ravel(),relative=True)
#downsampled_reflector.translate([-0.15,-0.15,0],relative=True)
def exampleUAV():
    bodystream = files(lyceanem.tests.data).joinpath('UAV.stl')
    arraystream = files(lyceanem.tests.data).joinpath('UAVarray.stl')
    body = o3d.io.read_triangle_mesh(str(bodystream))
    array = o3d.io.read_triangle_mesh(str(arraystream))
    rotation_vector1 = np.asarray([0.0, np.deg2rad(90), 0.0])
    rotation_vector2 = np.asarray([np.deg2rad(90), 0.0, 0.0])
    body=GF.open3drotate(body,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))
    body=GF.open3drotate(body,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2))
    array=GF.open3drotate(array,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))
    array=GF.open3drotate(array,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2))
    body.translate(np.asarray([0.25, 0, 0]), relative=True)
    array.translate(np.asarray([0.25, 0, 0]), relative=True)
    array.translate(np.array([-0.18, 0, 0.0125]), relative=True)
    body.translate(np.array([-0.18, 0, 0.0125]), relative=True)
    array.compute_vertex_normals()
    array.paint_uniform_color(np.array([1, 0.259, 0.145]))
    body.compute_vertex_normals()
    body.paint_uniform_color(np.array([0, 0.259, 0.145]))
    return body,array

def prototypescan(mesh_size,wavelength=3e8/24e9,importpcd=True):
    if importpcd:
        if mesh_size<(wavelength*0.5):
            #stream = pkg_resources.resource_stream(__name__,
            #                                       'data/prototypescanpoints24GHztenthwavelength.ply')
            stream=files('lyceanem.tests.data').joinpath('prototypescanpoints24GHztenthwavelength.ply')
            prototype_points=o3d.io.read_point_cloud(str(stream))
        else:
            #stream = pkg_resources.resource_stream(__name__,
            #                                       'data/prototypescanpoints24GHztenthwavelength.ply')
            stream = files(lyceanem.tests.data).joinpath('prototypescanpoints24GHztenthwavelength.ply')
            prototype_points = o3d.io.read_point_cloud(str(stream))
        prototype_mesh,_=tl.meshedReflector(0.3,0.3,6e-3,mesh_size)
    else:
        #stream = pkg_resources.resource_stream(__name__,
        #                                       'data/plate_copper.mat')
        stream = files('lyceanem.tests.data').joinpath('plate_copper.mat')
        matdata=io.loadmat(str(stream))['Flat_prototype']
        copper_reference_point=np.asarray([[np.min(matdata[:,0]),np.min(matdata[:,1]),np.min(matdata[:,2])]])/1000
        prototype= o3d.geometry.PointCloud()
        #convert from mm to si units
        prototype.points= o3d.utility.Vector3dVector(matdata/1000)
        prototype.estimate_normals()
        prototype.orient_normals_to_align_with_direction(orientation_reference=[0., 0., 1.])
        downsampled_prototype=prototype.voxel_down_sample(voxel_size=mesh_size*1.1)
        downsampled_prototype.translate(-copper_reference_point.ravel(),relative=True)
        prototype.translate(-copper_reference_point.ravel(),relative=True)
        downsampled_prototype.translate([-0.15,-0.15,0],relative=True)
        prototype.translate([-0.15,-0.15,0],relative=True)
        reflectorplate,scatter_points=tl.meshedReflector(0.3,0.3,6e-3,mesh_size,sides='all')
        temp_points=np.asarray(scatter_points.points)
        temp_points_back=temp_points[temp_points[:,2]<-1e-3,:]
        temp_points_top=temp_points[temp_points[:,1]>1.49e-1,:]
        top_normals=np.zeros((temp_points_top.shape[0],3))
        top_normals[:,1]=1
        toppoints=o3d.geometry.PointCloud()
        toppoints.points= o3d.utility.Vector3dVector(temp_points_top)
        toppoints.normals= o3d.utility.Vector3dVector(top_normals)
        sideapoints=copy.deepcopy(toppoints)
        rotation_vector1=np.radians(np.array([0,0,90]))
        sideapoints=GF.open3drotate(sideapoints,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))
        bottompoints=copy.deepcopy(sideapoints)
        bottompoints=GF.open3drotate(bottompoints,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))
        sidebpoints=copy.deepcopy(bottompoints)
        sidebpoints=GF.open3drotate(sidebpoints,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))

        back_normals=np.zeros((temp_points_back.shape[0],3))
        back_normals[:,2]=-1
        backpoints=o3d.geometry.PointCloud()
        backpoints.points= o3d.utility.Vector3dVector(temp_points_back)
        backpoints.normals= o3d.utility.Vector3dVector(back_normals)
        #for now only interested in points on the front surface
        temp=np.asarray(downsampled_prototype.points)
        interest_reference=np.min(temp[:,2])
        test_prototype=downsampled_prototype+backpoints+toppoints+bottompoints+sideapoints+sidebpoints
        prototype_mesh,output=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(test_prototype,linear_fit=True)
        prototype_mesh.compute_triangle_normals()
        prototype_points=o3d.geometry.PointCloud()
        vertices=np.asarray(prototype_mesh.vertices)
        normals=np.asarray(prototype_mesh.vertex_normals)
        prototype_points.points=o3d.utility.Vector3dVector(vertices[vertices[:,2]>=interest_reference,:])
        prototype_points.normals=o3d.utility.Vector3dVector(normals[vertices[:,2]>=interest_reference,:])
    return prototype_mesh,prototype_points

def mediumreferencescan(mesh_size,wavelength=3e8/24e9,importpcd=True):
    if importpcd:
        if mesh_size<(wavelength*0.5):
            #stream = pkg_resources.resource_stream(__name__,
            #                                       'data/referencescanpoints24GHztenthwavelength.ply')
            stream = files('lyceanem.tests.data').joinpath('referencescanpoints24GHztenthwavelength.ply')
            reflector_points=o3d.io.read_point_cloud(str(stream))
        else:
            #stream = pkg_resources.resource_stream(__name__,
            #                                       'data/referencescanpoints24GHztenthwavelength.ply')
            stream = files('lyceanem.tests.data').joinpath('referencescanpoints24GHztenthwavelength.ply')
            reflector_points = o3d.io.read_point_cloud(str(stream))
        reference_mesh,_=tl.meshedReflector(0.3,0.3,6e-3,mesh_size)
    else:
        #stream=pkg_resources.resource_stream(__name__,'data/Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt')
        stream = files('lyceanem.tests.data').joinpath('Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt')
        temp=np.loadtxt(str(stream),delimiter=';',skiprows=2)
        medium_reference_reflector = o3d.geometry.PointCloud()
        medium_reference_reflector.points = o3d.utility.Vector3dVector(temp/1000)
        medium_reference_reflector.estimate_normals()
        medium_reference_reflector.orient_normals_to_align_with_direction(orientation_reference=[0., 0., 1.])
        reference_point=np.asarray([[np.min(temp[:,0]),np.min(temp[:,1]),np.min(temp[:,2])]])/1000
    
        downsampled_reflector=medium_reference_reflector.voxel_down_sample(voxel_size=mesh_size*1.1)
        downsampled_reflector.translate(-reference_point.ravel(),relative=True)
        medium_reference_reflector.translate(-reference_point.ravel(),relative=True)
        downsampled_reflector.translate([-0.15,-0.15,0],relative=True)
        medium_reference_reflector.translate([-0.15,-0.15,0],relative=True)
        reflectorplate,scatter_points=tl.meshedReflector(0.3,0.3,6e-3,mesh_size,sides='all')
        temp_points=np.asarray(scatter_points.points)
        temp_points_back=temp_points[temp_points[:,2]<-1e-3,:]
        temp_points_top=temp_points[temp_points[:,1]>1.49e-1,:]
        top_normals=np.zeros((temp_points_top.shape[0],3))
        top_normals[:,1]=1
        toppoints=o3d.geometry.PointCloud()
        toppoints.points= o3d.utility.Vector3dVector(temp_points_top)
        toppoints.normals= o3d.utility.Vector3dVector(top_normals)
        sideapoints=copy.deepcopy(toppoints)
        rotation_vector1=np.radians(np.array([0,0,90]))
        sideapoints=GF.open3drotate(sideapoints,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))

        bottompoints=copy.deepcopy(sideapoints)
        bottompoints=GF.open3drotate(bottompoints,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))

        sidebpoints=copy.deepcopy(bottompoints)
        sidebpoints=GF.open3drotate(sidebpoints,o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1))

        back_normals=np.zeros((temp_points_back.shape[0],3))
        back_normals[:,2]=-1
        backpoints=o3d.geometry.PointCloud()
        backpoints.points= o3d.utility.Vector3dVector(temp_points_back)
        backpoints.normals= o3d.utility.Vector3dVector(back_normals)
        temp=np.asarray(downsampled_reflector.points)
        interest_reference=np.min(temp[:,2])
        test_prototype=downsampled_reflector+backpoints+toppoints+bottompoints+sideapoints+sidebpoints
        reference_mesh,output=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(test_prototype,linear_fit=True)
        reference_mesh.compute_triangle_normals()
        reflector_points=o3d.geometry.PointCloud()
        vertices=np.asarray(reference_mesh.vertices)
        normals=np.asarray(reference_mesh.vertex_normals)
        reflector_points.points=o3d.utility.Vector3dVector(vertices[vertices[:,2]>=interest_reference,:])
        reflector_points.normals=o3d.utility.Vector3dVector(normals[vertices[:,2]>=interest_reference,:])
    return reference_mesh,reflector_points

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
import numpy as np
import open3d as o3d
import scipy.io as io
from importlib_resources import files

import lyceanem.tests.data
import lyceanem.geometry.targets as tl
import lyceanem.geometry.geometryfunctions as GF
import lyceanem.raycasting.rayfunctions as RF


# freq=24e9
# wavelength=3e8/freq
# measurementaddress=Path("./")
# units are mm in the laser scans
# medium_reference='Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt'
# matfile='plate_copper.mat'
# datafile='Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt'
# stream=pkg_resources.resource_stream(__name__,'data/Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt')
# temp=np.loadtxt(stream,delimiter=';',skiprows=2)

# medium_reference_reflector = o3d.geometry.PointCloud()
# medium_reference_reflector.points = o3d.utility.Vector3dVector(temp/1000)
# medium_reference_reflector.estimate_normals()
# reference_point=np.asarray([[np.min(temp[:,0]),np.min(temp[:,1]),np.min(temp[:,2])]])/1000

# downsampled_reflector=medium_reference_reflector.voxel_down_sample(voxel_size=wavelength*0.5)
# downsampled_reflector.translate(-reference_point.ravel(),relative=True)
# downsampled_reflector.translate([-0.15,-0.15,0],relative=True)
def exampleUAV(frequency):
    bodystream = files(lyceanem.tests.data).joinpath("UAV.stl")
    arraystream = files(lyceanem.tests.data).joinpath("UAVarray.stl")
    body = o3d.io.read_triangle_mesh(str(bodystream))
    array = o3d.io.read_triangle_mesh(str(arraystream))
    rotation_vector1 = np.asarray([0.0, np.deg2rad(90), 0.0])
    rotation_vector2 = np.asarray([np.deg2rad(90), 0.0, 0.0])
    body = GF.open3drotate(
        body, o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1)
    )
    body = GF.open3drotate(
        body, o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2)
    )
    array = GF.open3drotate(
        array, o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1)
    )
    array = GF.open3drotate(
        array, o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2)
    )
    body.translate(np.asarray([0.25, 0, 0]), relative=True)
    array.translate(np.asarray([0.25, 0, 0]), relative=True)
    array.translate(np.array([-0.18, 0, 0.0125]), relative=True)
    body.translate(np.array([-0.18, 0, 0.0125]), relative=True)
    array.compute_vertex_normals()
    array.paint_uniform_color(np.array([0, 1.0, 1.0]))
    body.compute_vertex_normals()
    body.paint_uniform_color(np.array([0, 0.259, 0.145]))

    wavelength = 3e8 / frequency
    mesh_sep = wavelength * 0.5

    # define UAV array, a line, a circle, then a line with reference to the array solid
    array_vertices = np.asarray(array.vertices)
    array_height = np.max(array_vertices[:, 2]) - np.min(array_vertices[:, 2])
    top_index = (
        array_vertices[:, 2] >= np.max(array_vertices[:, 2]) - 1e-4
    )  # pick only vertices on the top of the array
    lower_index = (
        array_vertices[:, 2] <= np.min(array_vertices[:, 2]) + 1e-4
    )  # pick only vertices on the bottom of the surface
    array_top_points = array_vertices[top_index, :]
    array_bottom_points = array_vertices[lower_index, :]
    r1 = np.max(array_bottom_points[:, 1])
    r2 = np.mean(
        np.asarray([np.max(array_vertices[:, 1]), np.abs(np.min(array_vertices[:, 1]))])
    )
    top_length = np.abs(np.min(array_top_points[:, 0]) - np.max(array_top_points[:, 0]))
    bottom_length = np.abs(
        np.min(array_bottom_points[:, 0]) - np.max(array_bottom_points[:, 0])
    )
    slant_adjustment = (top_length - r2) - (bottom_length - r1)
    # r2 center is 25mm further forward than r1
    l = bottom_length - r1

    slant_angle = np.arctan2(r2 - r1, array_height)
    slant_h = (array_height ** 2 + (r2 - r1) ** 2) ** 0.5
    rows = math.ceil((slant_h) / (mesh_sep) + 1)
    heights = np.linspace(0, slant_h, rows)
    centers = np.linspace(l + 0, l + slant_adjustment, rows)
    rad_sep = np.linspace(r1, r2, rows)
    nose_distances = np.linspace(
        0, rad_sep[0] * np.pi, math.ceil((rad_sep[0] * np.pi) / (mesh_sep) + 1)
    )
    nose_side1 = np.empty((0, 3), dtype=np.float32)
    nose_side2 = np.empty((0, 3), dtype=np.float32)
    nose_circ = np.empty((0, 3), dtype=np.float32)
    nose_circ_normals = np.empty((0, 3), dtype=np.float32)
    for nose_inc in range(rows):
        nose_x = np.linspace(
            l, centers[nose_inc], math.ceil(((centers[nose_inc] - l) / mesh_sep + 1))
        )[1:]
        nose_y = (
            np.zeros((len(nose_x.ravel())), dtype=np.float32)
            + heights[nose_inc] * np.sin(slant_angle)
            + r1
        )
        nose_z = np.zeros((len(nose_x.ravel())), dtype=np.float32) + heights[
            nose_inc
        ] * np.cos(slant_angle)
        nose_side1 = np.append(
            nose_side1, np.array([nose_x, nose_y, nose_z]).transpose(), axis=0
        )
        nose_side2 = np.append(
            nose_side2, np.array([nose_x, -nose_y, nose_z]).transpose(), axis=0
        )
        angle_inc = np.pi
        nose_angles = np.linspace(
            0, np.pi, math.ceil((rad_sep[nose_inc] * np.pi) / (mesh_sep) + 1)
        )
        tempx = rad_sep[nose_inc] * np.sin(nose_angles)[1:-1] + centers[nose_inc]
        tempy = rad_sep[nose_inc] * np.cos(nose_angles)[1:-1]
        tempz = np.zeros((len(tempx.ravel())), dtype=np.float32) + heights[
            nose_inc
        ] * np.cos(slant_angle)
        testnose = np.array([tempx, tempy, tempz]).transpose()
        nose_circ = np.append(nose_circ, testnose, axis=0)
        tempnormx = np.cos(nose_angles[1:-1] - np.pi / 2)
        tempnormy = -np.sin(nose_angles[1:-1] - np.pi / 2)
        tempnormz = np.ones((len(nose_angles[1:-1])), dtype=np.float32) * np.sin(
            slant_angle + np.pi / 2 + np.pi / 2
        )
        nose_circ_normals = np.append(
            nose_circ_normals,
            np.array([tempnormx, tempnormy, tempnormz]).transpose(),
            axis=0,
        )

    nose_side1_normals = np.zeros((len(nose_side1), 3), dtype=np.float32)
    nose_side2_normals = np.zeros((len(nose_side2), 3), dtype=np.float32)
    nose_side2_normals[:, 1] = np.cos(slant_angle + np.pi / 2 + np.pi / 2)
    nose_side2_normals[:, 2] = np.sin(slant_angle + np.pi / 2 + np.pi / 2)
    nose_side1_normals[:, 1] = np.cos(np.pi / 2 - slant_angle - np.pi / 2)
    nose_side1_normals[:, 2] = np.sin(np.pi / 2 - slant_angle - np.pi / 2)

    side1x = np.linspace(0, l, math.ceil((l) / (mesh_sep) + 1))
    side1y = heights * np.sin(-slant_angle)
    side1z = heights * np.cos(slant_angle)
    xx, yy = np.meshgrid(side1x, heights)
    side1 = np.zeros(((len(np.ravel(xx)), 3)), dtype=np.float32)
    side2_normals = np.zeros(((len(np.ravel(xx)), 3)), dtype=np.float32)
    side2_normals[:, 1] = np.cos(slant_angle + np.pi / 2 + np.pi / 2)
    side2_normals[:, 2] = np.sin(slant_angle + np.pi / 2 + np.pi / 2)
    side1_normals = np.zeros(((len(np.ravel(xx)), 3)), dtype=np.float32)
    side1_normals[:, 1] = np.cos(np.pi / 2 - slant_angle - np.pi / 2)
    side1_normals[:, 2] = np.sin(np.pi / 2 - slant_angle - np.pi / 2)
    side1[:, 0] = xx.ravel()
    side1[:, 1] = yy.ravel() * np.sin(slant_angle) + r1
    side1[:, 2] = yy.ravel() * np.cos(slant_angle)
    side2 = np.empty(((len(np.ravel(xx)), 3)), dtype=np.float32)
    side2[:, 0] = xx.ravel()
    side2[:, 1] = yy.ravel() * np.sin(-slant_angle) - r1
    side2[:, 2] = yy.ravel() * np.cos(-slant_angle)

    total_array = np.append(
        np.append(
            np.append(np.append(side1, side2, axis=0), nose_circ, axis=0),
            nose_side1,
            axis=0,
        ),
        nose_side2,
        axis=0,
    ) + np.array([0.1025, 0, -0.025])
    total_array_normals = np.append(
        np.append(
            np.append(
                np.append(side1_normals, side2_normals, axis=0),
                nose_circ_normals,
                axis=0,
            ),
            nose_side1_normals,
            axis=0,
        ),
        nose_side2_normals,
        axis=0,
    ) + np.array([0.1025, 0, -0.025])
    source_pcd = RF.points2pointcloud(total_array)
    source_pcd.normals = o3d.utility.Vector3dVector(total_array_normals)
    source_pcd.translate(np.array([-0.18, 0, 0.0125]), relative=True)

    return body, array, source_pcd


def prototypescan(mesh_size, wavelength=3e8 / 24e9, importpcd=True):
    if importpcd:
        if mesh_size < (wavelength * 0.5):
            # stream = pkg_resources.resource_stream(__name__,
            #                                       'data/prototypescanpoints24GHztenthwavelength.ply')
            stream = files("lyceanem.tests.data").joinpath(
                "prototypescanpoints24GHztenthwavelength.ply"
            )
            prototype_points = o3d.io.read_point_cloud(str(stream))
        else:
            # stream = pkg_resources.resource_stream(__name__,
            #                                       'data/prototypescanpoints24GHztenthwavelength.ply')
            stream = files(lyceanem.tests.data).joinpath(
                "prototypescanpoints24GHztenthwavelength.ply"
            )
            prototype_points = o3d.io.read_point_cloud(str(stream))
        prototype_mesh, _ = tl.meshedReflector(0.3, 0.3, 6e-3, mesh_size)
    else:
        # stream = pkg_resources.resource_stream(__name__,
        #                                       'data/plate_copper.mat')
        stream = files("lyceanem.tests.data").joinpath("plate_copper.mat")
        matdata = io.loadmat(str(stream))["Flat_prototype"]
        copper_reference_point = (
            np.asarray(
                [[np.min(matdata[:, 0]), np.min(matdata[:, 1]), np.min(matdata[:, 2])]]
            )
            / 1000
        )
        prototype = o3d.geometry.PointCloud()
        # convert from mm to si units
        prototype.points = o3d.utility.Vector3dVector(matdata / 1000)
        prototype.estimate_normals()
        prototype.orient_normals_to_align_with_direction(
            orientation_reference=[0.0, 0.0, 1.0]
        )
        downsampled_prototype = prototype.voxel_down_sample(voxel_size=mesh_size * 1.1)
        downsampled_prototype.translate(-copper_reference_point.ravel(), relative=True)
        prototype.translate(-copper_reference_point.ravel(), relative=True)
        downsampled_prototype.translate([-0.15, -0.15, 0], relative=True)
        prototype.translate([-0.15, -0.15, 0], relative=True)
        reflectorplate, scatter_points = tl.meshedReflector(
            0.3, 0.3, 6e-3, mesh_size, sides="all"
        )
        temp_points = np.asarray(scatter_points.points)
        temp_points_back = temp_points[temp_points[:, 2] < -1e-3, :]
        temp_points_top = temp_points[temp_points[:, 1] > 1.49e-1, :]
        top_normals = np.zeros((temp_points_top.shape[0], 3))
        top_normals[:, 1] = 1
        toppoints = o3d.geometry.PointCloud()
        toppoints.points = o3d.utility.Vector3dVector(temp_points_top)
        toppoints.normals = o3d.utility.Vector3dVector(top_normals)
        sideapoints = copy.deepcopy(toppoints)
        rotation_vector1 = np.radians(np.array([0, 0, 90]))
        sideapoints = GF.open3drotate(
            sideapoints,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        )
        bottompoints = copy.deepcopy(sideapoints)
        bottompoints = GF.open3drotate(
            bottompoints,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        )
        sidebpoints = copy.deepcopy(bottompoints)
        sidebpoints = GF.open3drotate(
            sidebpoints,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        )

        back_normals = np.zeros((temp_points_back.shape[0], 3))
        back_normals[:, 2] = -1
        backpoints = o3d.geometry.PointCloud()
        backpoints.points = o3d.utility.Vector3dVector(temp_points_back)
        backpoints.normals = o3d.utility.Vector3dVector(back_normals)
        # for now only interested in points on the front surface
        temp = np.asarray(downsampled_prototype.points)
        interest_reference = np.min(temp[:, 2])
        test_prototype = (
            downsampled_prototype
            + backpoints
            + toppoints
            + bottompoints
            + sideapoints
            + sidebpoints
        )
        (
            prototype_mesh,
            output,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            test_prototype, linear_fit=True
        )
        prototype_mesh.compute_triangle_normals()
        prototype_points = o3d.geometry.PointCloud()
        vertices = np.asarray(prototype_mesh.vertices)
        normals = np.asarray(prototype_mesh.vertex_normals)
        prototype_points.points = o3d.utility.Vector3dVector(
            vertices[vertices[:, 2] >= interest_reference, :]
        )
        prototype_points.normals = o3d.utility.Vector3dVector(
            normals[vertices[:, 2] >= interest_reference, :]
        )
    return prototype_mesh, prototype_points


def mediumreferencescan(mesh_size, wavelength=3e8 / 24e9, importpcd=True):
    if importpcd:
        if mesh_size < (wavelength * 0.5):
            # stream = pkg_resources.resource_stream(__name__,
            #                                       'data/referencescanpoints24GHztenthwavelength.ply')
            stream = files("lyceanem.tests.data").joinpath(
                "referencescanpoints24GHztenthwavelength.ply"
            )
            reflector_points = o3d.io.read_point_cloud(str(stream))
        else:
            # stream = pkg_resources.resource_stream(__name__,
            #                                       'data/referencescanpoints24GHztenthwavelength.ply')
            stream = files("lyceanem.tests.data").joinpath(
                "referencescanpoints24GHztenthwavelength.ply"
            )
            reflector_points = o3d.io.read_point_cloud(str(stream))
        reference_mesh, _ = tl.meshedReflector(0.3, 0.3, 6e-3, mesh_size)
    else:
        # stream=pkg_resources.resource_stream(__name__,'data/Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt')
        stream = files("lyceanem.tests.data").joinpath(
            "Medium_Reference_Plate_Covered_Normal_1_plan_grid_0p5mm.txt"
        )
        temp = np.loadtxt(str(stream), delimiter=";", skiprows=2)
        medium_reference_reflector = o3d.geometry.PointCloud()
        medium_reference_reflector.points = o3d.utility.Vector3dVector(temp / 1000)
        medium_reference_reflector.estimate_normals()
        medium_reference_reflector.orient_normals_to_align_with_direction(
            orientation_reference=[0.0, 0.0, 1.0]
        )
        reference_point = (
            np.asarray([[np.min(temp[:, 0]), np.min(temp[:, 1]), np.min(temp[:, 2])]])
            / 1000
        )

        downsampled_reflector = medium_reference_reflector.voxel_down_sample(
            voxel_size=mesh_size * 1.1
        )
        downsampled_reflector.translate(-reference_point.ravel(), relative=True)
        medium_reference_reflector.translate(-reference_point.ravel(), relative=True)
        downsampled_reflector.translate([-0.15, -0.15, 0], relative=True)
        medium_reference_reflector.translate([-0.15, -0.15, 0], relative=True)
        reflectorplate, scatter_points = tl.meshedReflector(
            0.3, 0.3, 6e-3, mesh_size, sides="all"
        )
        temp_points = np.asarray(scatter_points.points)
        temp_points_back = temp_points[temp_points[:, 2] < -1e-3, :]
        temp_points_top = temp_points[temp_points[:, 1] > 1.49e-1, :]
        top_normals = np.zeros((temp_points_top.shape[0], 3))
        top_normals[:, 1] = 1
        toppoints = o3d.geometry.PointCloud()
        toppoints.points = o3d.utility.Vector3dVector(temp_points_top)
        toppoints.normals = o3d.utility.Vector3dVector(top_normals)
        sideapoints = copy.deepcopy(toppoints)
        rotation_vector1 = np.radians(np.array([0, 0, 90]))
        sideapoints = GF.open3drotate(
            sideapoints,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        )

        bottompoints = copy.deepcopy(sideapoints)
        bottompoints = GF.open3drotate(
            bottompoints,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        )

        sidebpoints = copy.deepcopy(bottompoints)
        sidebpoints = GF.open3drotate(
            sidebpoints,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        )

        back_normals = np.zeros((temp_points_back.shape[0], 3))
        back_normals[:, 2] = -1
        backpoints = o3d.geometry.PointCloud()
        backpoints.points = o3d.utility.Vector3dVector(temp_points_back)
        backpoints.normals = o3d.utility.Vector3dVector(back_normals)
        temp = np.asarray(downsampled_reflector.points)
        interest_reference = np.min(temp[:, 2])
        test_prototype = (
            downsampled_reflector
            + backpoints
            + toppoints
            + bottompoints
            + sideapoints
            + sidebpoints
        )
        (
            reference_mesh,
            output,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            test_prototype, linear_fit=True
        )
        reference_mesh.compute_triangle_normals()
        reflector_points = o3d.geometry.PointCloud()
        vertices = np.asarray(reference_mesh.vertices)
        normals = np.asarray(reference_mesh.vertex_normals)
        reflector_points.points = o3d.utility.Vector3dVector(
            vertices[vertices[:, 2] >= interest_reference, :]
        )
        reflector_points.normals = o3d.utility.Vector3dVector(
            normals[vertices[:, 2] >= interest_reference, :]
        )
    return reference_mesh, reflector_points

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from subprocess import run

import numpy as np
import open3d as o3d
import scipy.stats
import solid as sd
from importlib_resources import files
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

from ..geometry import geometryfunctions as GF
from ..raycasting import rayfunctions as RF
from ..base_classes import antenna_structures, structures, points
from ..utility import math_functions as math_functions
EPSILON = 1e-6  # how close to zero do we consider zero?


def NasaAlmond(resolution="quarter"):
    """
    NASA Almond is imported and then converted into the appropriate types for modelling
    This depends on the appropriate stl files in the working folder
    Parameters
    ----------
    resolution : TYPE, optional
        DESCRIPTION. The default is 'quarter', right now limited to half,quarter,tenth

    Returns
    -------
    NasaAlmond : o3d triangle mesh
        the physical structure of the nasa almond
    NasaAlmond_points : o3d points
        the mesh points for scattering

    """
    if resolution == "half":
        stream = files("lyceanem.geometry.data").joinpath(
            "NasaAlmondHalfWavelengthv2.stl"
        )
        NasaAlmond = o3d.io.read_triangle_mesh(str(stream))
    elif resolution == "quarter":
        stream = files("lyceanem.geometry.data").joinpath(
            "NasaAlmondQuarterWavelengthv2.stl"
        )
        NasaAlmond = o3d.io.read_triangle_mesh(str(stream))
    elif resolution == "tenth":
        stream = files("lyceanem.geometry.data").joinpath(
            "NasaAlmondTenthWavelengthv2.stl"
        )
        NasaAlmond = o3d.io.read_triangle_mesh(str(stream))

    NasaAlmond.compute_vertex_normals()
    # points=np.asarray(NasaAlmond.vertices)
    # normals=np.asarray(NasaAlmond.vertex_normals)
    _, scatter_cloud = GF.tri_centroids(NasaAlmond)
    return NasaAlmond, scatter_cloud


def source_cloud_from_shape(o3dshape, ideal_point_sep, maxdeviation=0.01):
    """sample surface or mesh and return surface algined points on that mesh

    Parameters
    ----------
    o3dshape : :class:`open3d.geometry.TriangleMesh`
        the surface or shape of interest
    ideal_point_sep : float
        the desired spacing between each point and it's nearest neighbours, for apertures, this is usually half a
        wavelength at the highest frequency of interest
    maxdeviation : float
        the maximum allowable deviation between the ideal point seperation and the average point seperation, as a
        fraction

    Returns
    ------
    source_cloud : :class:`open3d.geometry.PointCloud`
        the sampled points on the surface, with normal vectors aligned with the surface normal vectors
    areas : array of float32
        the area of each triangle in the :class:`open3d.geometry.TriangleMesh` in world units, as long as the surface
        is specified in metres, this will be sqm, alinged with the triangle index in the surface.
    """
    source_cloud = o3d.geometry.PointCloud()
    o3dshape.compute_triangle_normals()
    vertex_points = np.asarray(o3dshape.vertices)
    norms = np.asarray(o3dshape.triangle_normals)
    area_triangles = np.asarray(o3dshape.triangles)
    # total_area=o3dshape.get_surface_area()
    areas = np.zeros((area_triangles.shape[0]), dtype=np.float32)
    centroids = np.zeros((area_triangles.shape[0], 3), dtype=np.float32)
    for tri_index in range(area_triangles.shape[0]):
        areas[tri_index] = (
            np.linalg.norm(
                np.cross(
                    vertex_points[area_triangles[tri_index, 0], :]
                    - vertex_points[area_triangles[tri_index, 1], 0],
                    vertex_points[area_triangles[tri_index, 0], :]
                    - vertex_points[area_triangles[tri_index, 2], 0],
                )
            )
            / 2.0
        )
        centroids[tri_index, :] = np.asarray(
            [
                (
                    vertex_points[area_triangles[tri_index, 0], 0]
                    + vertex_points[area_triangles[tri_index, 1], 0]
                    + vertex_points[area_triangles[tri_index, 2], 0]
                )
                / 3,
                (
                    vertex_points[area_triangles[tri_index, 0], 1]
                    + vertex_points[area_triangles[tri_index, 1], 1]
                    + vertex_points[area_triangles[tri_index, 2], 1]
                )
                / 3,
                (
                    vertex_points[area_triangles[tri_index, 0], 2]
                    + vertex_points[area_triangles[tri_index, 1], 2]
                    + vertex_points[area_triangles[tri_index, 2], 2]
                )
                / 3,
            ]
        )

    # source_cloud.points=o3d.utility.Vector3dVector(centroids+np.array(o3dshape.triangle_normals)*offset)
    # source_cloud.normals=o3dshape.triangle_normals
    # print(np.sum(areas))
    area_per_point = (ideal_point_sep ** 2) * 0.5
    num_points = np.ceil(np.sum(areas) / area_per_point).astype(int)
    source_cloud = o3dshape.sample_points_poisson_disk(num_points)
    error = (
        ideal_point_sep - np.mean(source_cloud.compute_nearest_neighbor_distance())
    ) / ideal_point_sep
    loopcount = 0
    maxloops = 20
    errorlog = np.full((maxloops + 1, 2), np.nan)
    while np.abs(error) > maxdeviation:
        # point sampling is to high, put in control logic for allowable variation in terms of the maxdeviation (fraction of ideal_point_sep)
        print(error)
        loopcount += 1
        errorlog[loopcount, 0] = error
        errorlog[loopcount, 1] = np.ceil(num_points).astype(int)
        if loopcount >= maxloops:
            print("ran out counter, aborting")
            break
        if error < 0:
            # points are too far apart, add more points
            num_points *= 1 + np.abs(error)
            source_cloud = o3dshape.sample_points_poisson_disk(
                np.ceil(num_points).astype(int)
            )
            error = (
                ideal_point_sep
                - np.mean(source_cloud.compute_nearest_neighbor_distance())
            ) / ideal_point_sep
        elif error > 0:
            # points are to close together, take away points
            num_points *= 1 - np.abs(error)
            source_cloud = o3dshape.sample_points_poisson_disk(
                np.ceil(num_points).astype(int)
            )
            error = (
                ideal_point_sep
                - np.mean(source_cloud.compute_nearest_neighbor_distance())
            ) / ideal_point_sep

    errorlog[loopcount + 1, 0] = error
    errorlog[loopcount + 1, 1] = np.ceil(num_points).astype(int)

    # ensure that if the while loop cycled past the minimum it still finds it
    optimum_num = errorlog[
        np.where(np.abs(errorlog[:, 0]) == np.nanmin(np.abs(errorlog[:, 0])))[0][0], 1
    ]
    source_cloud = o3dshape.sample_points_poisson_disk(optimum_num.astype(int))
    # breakpoint()
    source_cloud.estimate_normals()
    return source_cloud, areas


def parabola(radius, focal_length, thickness, mesh_length, mesh="all"):
    """
    function to generate parabola of set focal length and radius using solids for more consistent meshing

    Parameters
    ----------
    focal_length : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    thickness : TYPE
        DESCRIPTION.
    mesh_length : TYPE
        DESCRIPTION.
    mesh : TYPE, optional
        DESCRIPTION. The default is 'top'.

    Returns
    -------
    parabola_mesh : TYPE
        DESCRIPTION.
    parabola_scattering_points : TYPE
        DESCRIPTION.

    """
    # stream = pkg_resources.resource_stream(__name__, 'parabolas.scad')
    stream = files("lyceanem.geometry").joinpath("parabolas.scad")
    parabolas = sd.import_scad(str(stream))
    height = (1 / (4 * focal_length)) * radius ** 2
    height_external = (1 / (4 * focal_length)) * (radius + thickness) ** 2
    # keep with a focal point with zero radius.
    focal_radius = 0
    # keep internal for now, gemetry will always be centred with focus a (0,0)
    geometry_centred = 0
    # better to be overmeshed than under, so based on circumference
    detail_level = np.ceil((2 * scipy.constants.pi * radius) / mesh_length).astype(int)
    parabola_desired = parabolas.openscad_paraboloid(
        y=height,
        f=focal_length,
        rfa=focal_radius,
        fc=geometry_centred,
        detail=detail_level,
    )
    parabola_external = parabolas.openscad_paraboloid(
        y=height_external,
        f=focal_length,
        rfa=focal_radius,
        fc=geometry_centred,
        detail=detail_level,
    )

    final_parabola = sd.difference()(
        sd.translate([0, 0, -thickness])(parabola_external),
        sd.union()(
            parabola_desired,
            sd.translate([0, 0, height])(
                sd.cylinder(r=radius, h=height_external, segments=detail_level)
            ),
        ),
    )

    sd.scad_render_to_file(final_parabola, "temp.scad")
    # run openscad and export to stl
    converttostl()

    parabola_mesh = o3d.io.read_triangle_mesh("temp.stl")
    parabola_mesh.compute_vertex_normals()
    parabola_mesh.compute_triangle_normals()
    _, parabola_scatter_cloud = GF.tri_centroids(parabola_mesh)

    parabola_structures=structures([parabola_mesh])
    parabola_points=points([parabola_scatter_cloud])
    parabola=antenna_structures(parabola_structures,parabola_points)
    return parabola

def converttostl():
    """
    This function is a convinence to allow the repeatable used of openscad to generate watertight stl files for use within LyceanEM. This function assumes that all openscad files are saved as temp.scad in the current working directory, and the output will be saved as temp.stl, ready for the home function to import it.

    Returns
    -------
    Nothing

    """
    import subprocess, os
    try:
        #print(os.getcwd())
        working_directory="."
        #run([, ])
        p = subprocess.Popen(["openscad-nightly", "-o", "temp.stl", "temp.scad", "--export-format=binstl"], cwd=working_directory)
        p.wait()
    except:
        #run(["openscad", "-o", "temp.stl", "temp.scad", "--export-format=binstl"])
        p = subprocess.Popen(["openscad", "-o", "temp.stl", "temp.scad", "--export-format=binstl"],cwd=working_directory)
        p.wait()

def meshed_pipe(
    eradius1, eradius2, iradius1, iradius2, height, mesh_length, mesh="centres"
):
    """
    creates a cylinder
    Parameters
    ----------
    eradius1 : float
        external bottom radius
    eradius2 : float
        external top radius
    height : float
        DESCRIPTION.
    mesh_length : float
        the maximum length of a facet, and maximum distance between mesh points in fully meshed mode
    mesh : string, optional
        DESCRIPTION. The default is 'centres'. Indicating only the centre of each face should be meshed. If `full' is chosen instead then a fully gridded surface will be generated.'

    Returns
    -------
    shape : open3d triangle mesh of the solid

    mesh : open3d point cloud of the mesh of points with normal vectors

    """

    segment_nums = np.max(
        [
            10,
            np.ceil((2 * np.pi * np.max([iradius1, iradius2])) / mesh_length).astype(
                "int"
            ),
        ]
    )
    centre = sd.difference()(
        sd.cylinder(r1=eradius1, r2=eradius2, h=height, segments=segment_nums),
        sd.cylinder(r1=iradius1, r2=iradius2, h=height * 1.1, segments=segment_nums),
    )
    if mesh == "centres":
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        mid_radius = np.min([iradius1, iradius2]) + np.abs(iradius1 - iradius2) / 2
        saggita = mid_radius * (1 - np.cos((2 * np.pi / segment_nums) * 0.5))
        tilt_angle = np.arctan2(iradius1 - iradius2, height)
        test_points = np.asarray(
            [
                (mid_radius - saggita) * np.cos(angles),
                (mid_radius - saggita) * np.sin(angles),
                (height * 0.5) * np.ones((segment_nums)),
            ]
        ).transpose()
        test_normals = np.asarray(
            [np.cos(angles), np.sin(angles), np.zeros((segment_nums))]
        ).transpose()
        topbottom = np.asarray([[0, 0], [0, 0], [0, height]]).transpose()
        tb_normals = np.asarray([[0, 0], [0, 0], [-1, 1]]).transpose()
        test_faces = np.append(test_points, topbottom, axis=0)
        face_normals = np.append(test_normals, -1 * tb_normals, axis=0)
    elif mesh == "tops":
        # create a mesh for all sides with maximum spacing of mesh_length
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(iradius1 - iradius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(iradius1 - iradius2, height)
        adjust_radius = np.linspace(iradius1, iradius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, -1 * col_normals, axis=0)

        # minimum of 1 top point
        top_segments = np.max([2, np.ceil(iradius2 / mesh_length).astype("int")])
        top_radials = np.linspace(0, iradius2, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * top_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = top_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (top_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    height * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, -1 * col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, -1 * r.apply(test_normals), axis=0)

        # add top centre
        test_faces = np.append(
            test_faces, np.asarray([0, 0, height]).reshape(1, 3), axis=0
        )
        face_normals = np.append(
            face_normals, -1 * np.asarray([0, 0, 1]).reshape(1, 3), axis=0
        )
    elif mesh == "faces":
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(iradius1 - iradius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(iradius1 - iradius2, height)
        adjust_radius = np.linspace(iradius1, iradius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, -1 * col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, r.apply(test_normals), axis=0)

    else:
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(iradius1 - iradius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(iradius1 - iradius2, height)
        adjust_radius = np.linspace(iradius1, iradius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, -1 * col_normals, axis=0)

        # minimum of 1 top point
        top_segments = np.max([2, np.ceil(iradius2 / mesh_length).astype("int")])
        top_radials = np.linspace(0, iradius2, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * top_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = top_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (top_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    height * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, -1 * col_normals, axis=0)

        # minimum of 1 bottom point
        bottom_segments = np.max([2, np.ceil(iradius1 / mesh_length).astype("int")])
        bottom_radials = np.linspace(0, iradius1, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * bottom_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = bottom_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (bottom_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    np.zeros((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    -1 * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, -1 * col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, r.apply(test_normals), axis=0)

        # add top centre
        test_faces = np.append(
            test_faces, np.asarray([0, 0, height]).reshape(1, 3), axis=0
        )
        face_normals = np.append(
            face_normals, -1 * np.asarray([0, 0, 1]).reshape(1, 3), axis=0
        )
        test_faces = np.append(test_faces, np.asarray([0, 0, 0]).reshape(1, 3), axis=0)
        face_normals = np.append(
            face_normals, -1 * np.asarray([0, 0, -1]).reshape(1, 3), axis=0
        )

    # generate valid openscad code and store it in file
    sd.scad_render_to_file(centre, "temp.scad")
    # run openscad and export to stl
    #run(["openscad-nightly", "-o", "temp.stl", "temp.scad", "--export-format=binstl"])
    converttostl()

    structure = o3d.io.read_triangle_mesh("temp.stl")
    structure.compute_vertex_normals()
    scatter_cloud = RF.points2pointcloud(np.copy(test_faces))
    scatter_cloud.normals = o3d.utility.Vector3dVector(np.copy(face_normals))

    return structure, scatter_cloud


def meshed_cylinder(radius1, radius2, height, mesh_length, mesh="centres",segment_nums=0):
    """
    creates a cylinder
    Parameters
    ----------
    radius1 : float
        bottom radius
    radius2 : float
        top radius
    height : float
        DESCRIPTION.
    mesh_length : float
        the maximum length of a facet, and maximum distance between mesh points in fully meshed mode
    mesh : string, optional
        DESCRIPTION. The default is 'centres'. Indicating only the centre of each face should be meshed. If `full' is chosen instead then a fully gridded surface will be generated.'

    Returns
    -------
    shape : open3d triangle mesh of the solid

    mesh : open3d point cloud of the mesh of points with normal vectors

    """
    if segment_nums<3:
        segment_nums = np.max(
            [
                3,
                np.ceil((2 * np.pi * np.max([radius1, radius2])) / mesh_length).astype(
                    "int"
                ),
            ]
        )

    centre = sd.cylinder(r1=radius1, r2=radius2, h=height, segments=segment_nums)
    if mesh == "centres":
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        mid_radius = np.min([radius1, radius2]) + np.abs(radius1 - radius2) / 2
        saggita = mid_radius * (1 - np.cos((2 * np.pi / segment_nums) * 0.5))
        tilt_angle = np.arctan2(radius1 - radius2, height)
        test_points = np.asarray(
            [
                (mid_radius - saggita) * np.cos(angles),
                (mid_radius - saggita) * np.sin(angles),
                (height * 0.5) * np.ones((segment_nums)),
            ]
        ).transpose()
        test_normals = np.asarray(
            [np.cos(angles), np.sin(angles), np.zeros((segment_nums))]
        ).transpose()
        topbottom = np.asarray([[0, 0], [0, 0], [0, height]]).transpose()
        tb_normals = np.asarray([[0, 0], [0, 0], [-1, 1]]).transpose()
        test_faces = np.append(test_points, topbottom, axis=0)
        face_normals = np.append(test_normals, tb_normals, axis=0)
    elif mesh == "tops":
        # create a mesh for all sides with maximum spacing of mesh_length
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(radius1 - radius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(radius1 - radius2, height)
        adjust_radius = np.linspace(radius1, radius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        # minimum of 1 top point
        top_segments = np.max([2, np.ceil(radius2 / mesh_length).astype("int")])
        top_radials = np.linspace(0, radius2, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * top_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = top_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (top_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    height * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, r.apply(test_normals), axis=0)

        # add top centre
        test_faces = np.append(
            test_faces, np.asarray([0, 0, height]).reshape(1, 3), axis=0
        )
        face_normals = np.append(
            face_normals, np.asarray([0, 0, 1]).reshape(1, 3), axis=0
        )
    elif mesh == "sides":
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(radius1 - radius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(radius1 - radius2, height)
        adjust_radius = np.linspace(radius1, radius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, r.apply(test_normals), axis=0)

    else:
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(radius1 - radius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(radius1 - radius2, height)
        adjust_radius = np.linspace(radius1, radius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        # minimum of 1 top point
        top_segments = np.max([2, np.ceil(radius2 / mesh_length).astype("int")])
        top_radials = np.linspace(0, radius2, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * top_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = top_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (top_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    height * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        # minimum of 1 bottom point
        bottom_segments = np.max([2, np.ceil(radius1 / mesh_length).astype("int")])
        bottom_radials = np.linspace(0, radius1, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * bottom_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = bottom_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (bottom_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    np.zeros((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    -1 * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, r.apply(test_normals), axis=0)

        # add top centre
        test_faces = np.append(
            test_faces, np.asarray([0, 0, height]).reshape(1, 3), axis=0
        )
        face_normals = np.append(
            face_normals, np.asarray([0, 0, 1]).reshape(1, 3), axis=0
        )
        test_faces = np.append(test_faces, np.asarray([0, 0, 0]).reshape(1, 3), axis=0)
        face_normals = np.append(
            face_normals, np.asarray([0, 0, -1]).reshape(1, 3), axis=0
        )

    # generate valid openscad code and store it in file
    sd.scad_render_to_file(centre, "temp.scad")
    # run openscad and export to stl
    #run(["openscad-nightly", "-o", "temp.stl", "temp.scad", "--export-format=binstl"])
    converttostl()

    #structure = o3d.io.read_triangle_mesh("temp.stl")
    temp_mesh=o3d.io.read_triangle_mesh("temp.stl")
    temp_mesh.compute_vertex_normals()
    scatter_cloud = RF.points2pointcloud(np.copy(test_faces))
    scatter_cloud.normals = o3d.utility.Vector3dVector(np.copy(face_normals))
    structure = structures([temp_mesh])
    total_points = points([scatter_cloud])
    return structure, total_points


def meshed_trapazoid(radius1, radius2, height, mesh_length, mesh="centres"):
    """
    creates a trapazoid
    Parameters
    ----------
    radius1 : float
        bottom radius
    radius2 : float
        top radius
    height : float
        DESCRIPTION.
    mesh_length : float
        the maximum length of a facet, and maximum distance between mesh points in fully meshed mode
    mesh : string, optional
        DESCRIPTION. The default is 'centres'. Indicating only the centre of each face should be meshed. If `full' is chosen instead then a fully gridded surface will be generated.'

    Returns
    -------
    shape : open3d triangle mesh of the solid

    mesh : open3d point cloud of the mesh of points with normal vectors

    """

    # segment_nums=np.max([10,np.ceil((2*np.pi*radius)/mesh_length).astype('int')])
    segment_nums = 4
    centre = sd.cylinder(r1=radius1, r2=radius2, h=height, segments=segment_nums)
    if mesh == "centres":
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        mid_radius = np.min([radius1, radius2]) + np.abs(radius1 - radius2) / 2
        saggita = mid_radius * (1 - np.cos((2 * np.pi / segment_nums) * 0.5))
        tilt_angle = np.arctan2(radius1 - radius2, height)
        test_points = np.asarray(
            [
                (mid_radius - saggita) * np.cos(angles),
                (mid_radius - saggita) * np.sin(angles),
                (height * 0.5) * np.ones((segment_nums)),
            ]
        ).transpose()
        test_normals = np.asarray(
            [np.cos(angles), np.sin(angles), np.zeros((segment_nums))]
        ).transpose()
        topbottom = np.asarray([[0, 0], [0, 0], [0, height]]).transpose()
        tb_normals = np.asarray([[0, 0], [0, 0], [-1, 1]]).transpose()
        test_faces = np.append(test_points, topbottom, axis=0)
        face_normals = np.append(test_normals, tb_normals, axis=0)
    elif mesh == "tops":
        # create a mesh for all sides with maximum spacing of mesh_length
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(radius1 - radius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(radius1 - radius2, height)
        adjust_radius = np.linspace(radius1, radius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        # minimum of 1 top point
        top_segments = np.max([2, np.ceil(radius2 / mesh_length).astype("int")])
        top_radials = np.linspace(0, radius2, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * top_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = top_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (top_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    height * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, r.apply(test_normals), axis=0)

        # add top centre
        test_faces = np.append(
            test_faces, np.asarray([0, 0, height]).reshape(1, 3), axis=0
        )
        face_normals = np.append(
            face_normals, np.asarray([0, 0, 1]).reshape(1, 3), axis=0
        )
    else:
        angles = np.linspace(0, np.pi * 2, segment_nums + 1)[0:-1] + (np.pi) / (
            segment_nums
        )
        face_length = (np.abs(radius1 - radius2) ** 2 + height ** 2) ** 0.5
        face_segments = np.max([3, np.ceil(face_length / mesh_length).astype("int")])
        face_heights = np.linspace(0, height, face_segments)
        test_points = np.empty((0, 3), dtype=np.float32)
        test_normals = np.empty((0, 3), dtype=np.float32)
        tilt_angle = np.arctan2(radius1 - radius2, height)
        adjust_radius = np.linspace(radius1, radius2, face_segments)
        for row_num in range(face_segments):
            segment_chord = (
                2 * adjust_radius[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = adjust_radius[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (adjust_radius[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    face_heights[row_num] * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    np.cos(tilt_angle) * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.sin(tilt_angle) * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        # minimum of 1 top point
        top_segments = np.max([2, np.ceil(radius2 / mesh_length).astype("int")])
        top_radials = np.linspace(0, radius2, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * top_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = top_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (top_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    height * np.ones((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        # minimum of 1 bottom point
        bottom_segments = np.max([2, np.ceil(radius1 / mesh_length).astype("int")])
        bottom_radials = np.linspace(0, radius1, top_segments)[1:-1]
        for row_num in range(top_segments - 2):
            segment_chord = (
                2 * bottom_radials[row_num] * np.sin((2 * np.pi / segment_nums) * 0.5)
            )
            saggita = bottom_radials[row_num] * (
                1 - np.cos((2 * np.pi / segment_nums) * 0.5)
            )
            col_num = np.max([2, np.ceil(segment_chord / mesh_length).astype("int")])
            col_coords = np.asarray(
                [
                    (bottom_radials[row_num] - saggita) * np.ones((col_num - 1)),
                    np.linspace(0, segment_chord, col_num)[0:-1] - segment_chord * 0.5,
                    np.zeros((col_num - 1)),
                ]
            ).transpose()
            col_normals = np.asarray(
                [
                    0 * np.ones((col_num - 1)),
                    np.zeros((col_num - 1)),
                    -1 * np.ones((col_num - 1)),
                ]
            ).transpose()
            test_points = np.append(test_points, col_coords, axis=0)
            test_normals = np.append(test_normals, col_normals, axis=0)

        test_faces = np.empty((0, 3), dtype=np.float32)
        face_normals = np.empty((0, 3), dtype=np.float32)
        for faces in range(segment_nums):
            # quaternion rotation around z axis
            r = R.from_quat(
                [0, 0, np.sin(angles[faces] / 2), np.cos(angles[faces] / 2)]
            )
            test_faces = np.append(test_faces, r.apply(test_points), axis=0)
            face_normals = np.append(face_normals, r.apply(test_normals), axis=0)

        # add top centre
        test_faces = np.append(
            test_faces, np.asarray([0, 0, height]).reshape(1, 3), axis=0
        )
        face_normals = np.append(
            face_normals, np.asarray([0, 0, 1]).reshape(1, 3), axis=0
        )
        test_faces = np.append(test_faces, np.asarray([0, 0, 0]).reshape(1, 3), axis=0)
        face_normals = np.append(
            face_normals, np.asarray([0, 0, -1]).reshape(1, 3), axis=0
        )

    # generate valid openscad code and store it in file
    sd.scad_render_to_file(centre, "temp.scad")
    # run openscad and export to stl
    #run(["openscad-nightly", "-o", "temp.stl", "temp.scad", "--export-format=binstl"])
    converttostl()

    structure = o3d.io.read_triangle_mesh("temp.stl")
    structure.compute_vertex_normals()
    scatter_cloud = RF.points2pointcloud(np.copy(test_faces))
    scatter_cloud.normals = o3d.utility.Vector3dVector(np.copy(face_normals))

    return structure, scatter_cloud


def rectReflector(majorsize, minorsize, thickness):
    """
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,0)
    """

    reflector1 = o3d.geometry.TriangleMesh.create_box(majorsize, minorsize, thickness)
    translate_dist = np.array(
        [-majorsize / 2.0, -minorsize / 2.0, -(thickness + EPSILON)]
    )
    # fine_mesh=reflector1.subdivide_midpoint(3)
    fine_mesh = reflector1
    fine_mesh.compute_vertex_normals()
    fine_mesh.paint_uniform_color([0.79, 0.50, 0.24])
    fine_mesh.translate(translate_dist, relative=True)

    return fine_mesh


def shapeTrapezoid(x_size, y_size, length, flare_angle):
    """
    create a trapazoid to represent a simple horn of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,0)
    """
    # create trapazoid vertices
    mesh_vertices = np.zeros((8, 3), dtype=np.float32)
    mesh_vertices[0, :] = [-x_size / 2.0, -y_size / 2, -1e-6]
    mesh_vertices[1, :] = [x_size / 2.0, -y_size / 2, -1e-6]
    mesh_vertices[2, :] = [x_size / 2.0, y_size / 2, -1e-6]
    mesh_vertices[3, :] = [-x_size / 2.0, y_size / 2, -1e-6]
    mesh_vertices[4, :] = mesh_vertices[0, :] - np.array(
        [
            mesh_vertices[0, 0] * np.sin(flare_angle),
            mesh_vertices[0, 1] * np.sin(flare_angle),
            length,
        ]
    )
    mesh_vertices[5, :] = mesh_vertices[1, :] - np.array(
        [
            mesh_vertices[1, 0] * np.sin(flare_angle),
            mesh_vertices[1, 1] * np.sin(flare_angle),
            length,
        ]
    )
    mesh_vertices[6, :] = mesh_vertices[2, :] - np.array(
        [
            mesh_vertices[2, 0] * np.sin(flare_angle),
            mesh_vertices[2, 1] * np.sin(flare_angle),
            length,
        ]
    )
    mesh_vertices[7, :] = mesh_vertices[3, :] - np.array(
        [
            mesh_vertices[3, 0] * np.sin(flare_angle),
            mesh_vertices[3, 1] * np.sin(flare_angle),
            length,
        ]
    )
    triangle_list = np.zeros((12, 3), dtype=np.int32)
    triangle_list[0, :] = [0, 1, 3]
    triangle_list[1, :] = [2, 3, 1]
    triangle_list[2, :] = [0, 3, 4]
    triangle_list[3, :] = [3, 7, 4]
    triangle_list[4, :] = [3, 2, 7]
    triangle_list[5, :] = [2, 6, 7]
    triangle_list[6, :] = [0, 4, 1]
    triangle_list[7, :] = [5, 1, 4]
    triangle_list[8, :] = [2, 1, 6]
    triangle_list[9, :] = [5, 6, 1]
    triangle_list[10, :] = [4, 7, 5]
    triangle_list[11, :] = [6, 5, 7]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle_list)
    mesh.compute_triangle_normals()
    mesh.paint_uniform_color(np.array([0, 0.259, 0.145]))

    return mesh


def meshedReflector(majorsize, minorsize, thickness, grid_resolution, sides="all"):
    """

    A helper function which creates a meshed cuboid with the `front` face center at the origin, and with the boresight aligned with the positive z direction

    Parameters
    ----------
    majorsize : float
        the width of the cuboid in the x direction
    minorsize : float
        the width of the cuboid in the y direction
    thickness : float
        the thickness of the cuboid structure
    grid_resolution : float
        the spacing between the scattering points, should be half a wavelength at the frequency of interest
    sides : str
        command for the mesh, default is 'all', creating a mesh of surface points on all sides of the cuboid, other
        options are 'front' which only creates points for the side aligned with the positive z direction, or 'centres',
         which creates a point for the centre of each side.

    Returns
    -------
    reflector : :class:`open3d.geometry.TriangleMesh`
        the defined cuboid
    mesh_points : :class:`open3d.geometry.PointCloud`
        the scattering points, spaced at grid_resolution seperation between each point, and with normal vectors from
        the populating surfaces

    """
    reflector = rectReflector(majorsize, minorsize, thickness)
    mesh_points = gridedReflectorPoints(
        majorsize, minorsize, thickness, grid_resolution, sides
    )
    return reflector, mesh_points


def OTAEllipsoid(
    reflector_index,
    major_axis_size,
    minor_axis_size,
    max_grid,
    reflector_shape="rectangle",
):
    # standard defintion of the OTA Prototpe
    # antenna 1 should be at (-0.98107,0,0)
    # antenna 2 should be at (0.98107,0,0)
    # major axis of 2.05m, minor axis of 1.8m

    #
    gridded_sides = "front"
    semi_major = 2.05
    semi_minor = 1.8
    freq = 24e9
    wavelength = 3e8 / freq
    local_coords = np.array(
        [
            [-1.9998, 0.2781, 0],
            [-1.8041, 0.8071, 0],
            [-1.4317, 1.2571, 0],
            [-0.9192, 1.5841, 0],
            [-0.3167, 1.7560, 0],
        ]
    )
    full_coords = np.append(local_coords, np.flipud(local_coords), axis=0)
    full_coords[5:, 0] = full_coords[5:, 0] * -1
    #
    origins = np.zeros((10, 3), dtype=np.float32)
    directions = np.zeros((10, 3), dtype=np.float32)
    norm_length = np.zeros((10, 1), dtype=np.float32)
    directions, norm_length = math_functions.calc_dv_norm(
        full_coords, origins, directions, norm_length
    )
    angle_values = np.arctan2(directions[:, 1], directions[:, 0])
    reflector_pointers = (
        np.arctan((semi_major ** 2) / (semi_minor ** 2) * np.tan(angle_values))
        - np.pi / 2.0
    )
    if reflector_index == 0 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            s1_plate, s1_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            s1_plate, s1_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        s1_points = GF.open3drotate(
            s1_points,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
        )
        s1_points = GF.open3drotate(
            s1_points,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[0]]
            ),
        )

        s1_points.translate(full_coords[0, :].transpose(), relative=True)
        s1_plate = GF.open3drotate(
            s1_plate,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
        )
        s1_plate = GF.open3drotate(
            s1_plate,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[0]]
            ),
        )
        s1_plate.translate(full_coords[0, :].transpose(), relative=True)

    if reflector_index == 1 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            u1_plate, u1_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            u1_plate, u1_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        u1_plate = GF.open3drotate(
            u1_plate,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
        )
        u1_plate = GF.open3drotate(
            u1_plate,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[1]]
            ),
        )
        u1_plate.translate(full_coords[1, :].transpose(), relative=True)
        u1_points = GF.open3drotate(
            u1_points,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
        )
        u1_points = GF.open3drotate(
            u1_points,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[1]]
            ),
        )
        u1_points.translate(full_coords[1, :].transpose(), relative=True)

    if reflector_index == 2 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            v1_plate, v1_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            v1_plate, v1_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        v1_plate = GF.open3drotate(
            v1_plate,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        v1_plate = GF.open3drotate(
            v1_plate,
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[2]]
            ),
            center=False,
        )
        v1_plate.translate(full_coords[2, :].transpose(), relative=True)
        v1_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        v1_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[2]]
            ),
            center=False,
        )
        v1_points.translate(full_coords[2, :].transpose(), relative=True)

    if reflector_index == 3 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            w1_plate, w1_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            w1_plate, w1_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        w1_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        w1_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[3]]
            ),
            center=False,
        )
        w1_plate.translate(full_coords[3, :].transpose(), relative=True)
        w1_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        w1_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[3]]
            ),
            center=False,
        )
        w1_points.translate(full_coords[3, :].transpose(), relative=True)

    if reflector_index == 4 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            z1_plate, z1_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            z1_plate, z1_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        z1_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        z1_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[4]]
            ),
            center=False,
        )
        z1_plate.translate(full_coords[4, :].transpose(), relative=True)
        z1_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        z1_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[4]]
            ),
            center=False,
        )
        z1_points.translate(full_coords[4, :].transpose(), relative=True)

    if reflector_index == 5 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            z2_plate, z2_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            z2_plate, z2_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        z2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        z2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[5] + np.pi]
            ),
            center=False,
        )
        z2_plate.translate(full_coords[5, :].transpose(), relative=True)
        z2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        z2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[5] + np.pi]
            ),
            center=False,
        )
        z2_points.translate(full_coords[5, :].transpose(), relative=True)

    if reflector_index == 6 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            w2_plate, w2_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            w2_plate, w2_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        w2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        w2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[6] + np.pi]
            ),
            center=False,
        )
        w2_plate.translate(full_coords[6, :].transpose(), relative=True)
        w2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        w2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[6] + np.pi]
            ),
            center=False,
        )
        w2_points.translate(full_coords[6, :].transpose(), relative=True)

    if reflector_index == 7 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            v2_plate, v2_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            v2_plate, v2_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        v2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        v2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[7] + np.pi]
            ),
            center=False,
        )
        v2_plate.translate(full_coords[7, :].transpose(), relative=True)
        v2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        v2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[7] + np.pi]
            ),
            center=False,
        )
        v2_points.translate(full_coords[7, :].transpose(), relative=True)

    if reflector_index == 8 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            u2_plate, u2_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            u2_plate, u2_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        u2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        u2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[8] + np.pi]
            ),
            center=False,
        )
        u2_plate.translate(full_coords[8, :].transpose(), relative=True)
        u2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        u2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[8] + np.pi]
            ),
            center=False,
        )
        u2_points.translate(full_coords[8, :].transpose(), relative=True)

    if reflector_index == 9 or reflector_index >= 10:
        if reflector_shape == "rectangle":
            s2_plate, s2_points = meshedReflector(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )
        elif reflector_shape == "circle":
            s2_plate, s2_points = meshedCircle(
                major_axis_size, minor_axis_size, 6e-3, max_grid, sides=gridded_sides
            )

        s2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        s2_plate.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[9] + np.pi]
            ),
            center=False,
        )
        s2_plate.translate(full_coords[9, :].transpose(), relative=True)
        s2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                np.asarray([np.arcsin(-1.0), 0.0, 0.0])
            ),
            center=False,
        )
        s2_points.rotate(
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
                [0.0, 0.0, reflector_pointers[9] + np.pi]
            ),
            center=False,
        )
        s2_points.translate(full_coords[9, :].transpose(), relative=True)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([s1_plate,u1_plate,v1_plate,w1_plate,z1_plate,s1_points,u1_points,v1_points,w1_points,z1_points,mesh_frame])
    if reflector_index == 0:
        # total_scatter_points=points2pointcloud(np.asarray(s1_points.points))
        total_scatter_points = s1_points
        environment = RF.convertTriangles(s1_plate)
        all_plates = s1_plate

    if reflector_index == 1:
        # total_scatter_points=points2pointcloud(np.asarray(u1_points.points))
        total_scatter_points = u1_points
        environment = RF.convertTriangles(u1_plate)
        all_plates = u1_plate

    if reflector_index == 2:
        # total_scatter_points=points2pointcloud(np.asarray(v1_points.points))
        total_scatter_points = v1_points
        environment = RF.convertTriangles(v1_plate)
        all_plates = v1_plate

    if reflector_index == 3:
        # total_scatter_points=points2pointcloud(np.asarray(w1_points.points))
        total_scatter_points = w1_points
        environment = RF.convertTriangles(w1_plate)
        all_plates = w1_plate

    if reflector_index == 4:
        # total_scatter_points=points2pointcloud(np.asarray(z1_points.points))
        total_scatter_points = z1_points
        environment = RF.convertTriangles(z1_plate)
        all_plates = z1_plate

    if reflector_index == 5:
        # total_scatter_points=points2pointcloud(np.asarray(z2_points.points))
        total_scatter_points = z2_points
        environment = RF.convertTriangles(z2_plate)
        all_plates = z2_plate

    if reflector_index == 6:
        # total_scatter_points=points2pointcloud(np.asarray(w2_points.points))
        total_scatter_points = w2_points
        environment = RF.convertTriangles(w2_plate)
        all_plates = w2_plate

    if reflector_index == 7:
        # total_scatter_points=points2pointcloud(np.asarray(v2_points.points))
        total_scatter_points = v2_points
        environment = RF.convertTriangles(v2_plate)
        all_plates = v2_plate

    if reflector_index == 8:
        # total_scatter_points=points2pointcloud(np.asarray(u2_points.points))
        total_scatter_points = u2_points
        environment = RF.convertTriangles(u2_plate)
        all_plates = u2_plate

    if reflector_index == 9:
        # total_scatter_points=points2pointcloud(np.asarray(s2_points.points))
        total_scatter_points = s2_points
        environment = RF.convertTriangles(s2_plate)
        all_plates = s2_plate

    if reflector_index >= 10:
        total_scatter_temp = RF.points2pointcloud(
            np.append(
                np.append(
                    np.append(
                        np.append(
                            np.append(
                                np.append(
                                    np.append(
                                        np.append(
                                            np.append(
                                                np.asarray(s1_points.points),
                                                np.asarray(u1_points.points),
                                                axis=0,
                                            ),
                                            np.asarray(v1_points.points),
                                            axis=0,
                                        ),
                                        np.asarray(w1_points.points),
                                        axis=0,
                                    ),
                                    np.asarray(z1_points.points),
                                    axis=0,
                                ),
                                np.asarray(z2_points.points),
                                axis=0,
                            ),
                            np.asarray(w2_points.points),
                            axis=0,
                        ),
                        np.asarray(v2_points.points),
                        axis=0,
                    ),
                    np.asarray(u2_points.points),
                    axis=0,
                ),
                np.asarray(s2_points.points),
                axis=0,
            )
        )
        total_scatter_normals = RF.points2pointcloud(
            np.append(
                np.append(
                    np.append(
                        np.append(
                            np.append(
                                np.append(
                                    np.append(
                                        np.append(
                                            np.append(
                                                np.asarray(s1_points.normals),
                                                np.asarray(u1_points.normals),
                                                axis=0,
                                            ),
                                            np.asarray(v1_points.normals),
                                            axis=0,
                                        ),
                                        np.asarray(w1_points.normals),
                                        axis=0,
                                    ),
                                    np.asarray(z1_points.normals),
                                    axis=0,
                                ),
                                np.asarray(z2_points.normals),
                                axis=0,
                            ),
                            np.asarray(w2_points.normals),
                            axis=0,
                        ),
                        np.asarray(v2_points.normals),
                        axis=0,
                    ),
                    np.asarray(u2_points.normals),
                    axis=0,
                ),
                np.asarray(s2_points.normals),
                axis=0,
            )
        )
        total_scatter_points = o3d.geometry.PointCloud()
        total_scatter_points.points = o3d.utility.Vector3dVector(total_scatter_temp)
        total_scatter_points.normals = o3d.utility.Vector3dVector(total_scatter_normals)
        environment = np.append(
            np.append(
                np.append(
                    np.append(
                        np.append(
                            np.append(
                                np.append(
                                    np.append(
                                        RF.convertTriangles(s1_plate),
                                        RF.convertTriangles(u1_plate),
                                    ),
                                    RF.convertTriangles(v1_plate),
                                ),
                                RF.convertTriangles(w1_plate),
                            ),
                            RF.convertTriangles(z1_plate),
                        ),
                        RF.convertTriangles(z2_plate),
                    ),
                    RF.convertTriangles(w2_plate),
                ),
                RF.convertTriangles(v2_plate),
            ),
            RF.convertTriangles(u2_plate),
        )
        all_plates = [
            s1_plate,
            u1_plate,
            v1_plate,
            w1_plate,
            z1_plate,
            s2_plate,
            u2_plate,
            v2_plate,
            w2_plate,
            z2_plate,
        ]

    return environment, total_scatter_points, all_plates


def circleReflector(majorsize, minorsize, thickness, grid_resolution):
    # create a circle
    circ_num = np.ceil((np.pi * majorsize) / grid_resolution).astype(int)
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=majorsize / 2, height=thickness, resolution=circ_num, split=1
    )
    translate_vector = np.asarray([0, 0, -(thickness / 2 + 1e-6)])
    mesh.translate(translate_vector, relative=True)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.79, 0.50, 0.24])
    return mesh


def meshedCircle(majorsize, minorsize, thickness, grid_resolution, sides="all"):
    # define a circle, and grid the front and back
    reflector = circleReflector(majorsize, minorsize, thickness, grid_resolution)
    mesh_points = gridedCirclePoints(
        majorsize, minorsize, thickness, grid_resolution, sides="front"
    )
    return reflector, mesh_points


def meshedHorn(
    majorsize,
    minorsize,
    length,
    edge_width,
    flare_angle,
    grid_resolution,
    sides="front",
):
    """
    A basic horn antenna, providing the aperture points, and the basic physical structure
    The horn is orientated with the centre of the aperture at the origin, and the boresight aligned with the positive z direction.

    Parameters
    ----------
    majorsize : float
        the width of the horn aperture in the x direction
    minorsize : float
        the width of the horn aperture in the y direction
    length : float
        the length of the horn structure
    edge_width : float
        the width of the physical structure around the horn
    flare_angle :
        the taper angle of the horn
    grid_resolution : float
        the spacing between the aperture points, should be half a wavelength at the frequency of interest
    sides : str
        command for the mesh, default is 'front', and for a horn, this should not be changed.

    Returns
    -------
    structure : :class:`open3d.geometry.TriangleMesh`
        the physical structure of the horn
    mesh_points : :class:`open3d.geometry.PointCloud`
        the source points for the horn aperture
    """
    structure = shapeTrapezoid(
        majorsize + (edge_width * 2), minorsize + (edge_width * 2), length, flare_angle
    )
    mesh_points = gridedReflectorPoints(
        majorsize, minorsize, 1e-6, grid_resolution, sides
    )
    mesh_points.translate(np.asarray([0, 0, 1e-6]))
    return structure, mesh_points


def coneReflector(radius, height):
    """
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,1)
    """
    resolution = 200000
    split = 1
    reflector1 = o3d.geometry.TriangleMesh.create_cone(
        radius, height, resolution, split
    )
    translate_dist = np.array([0, 0, 3])
    reflector1 = GF.open3drotate(
        reflector1,
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle(
            np.array([0.0, np.radians(180), 0.0])
        ),
        center=True,
    )
    reflector1.compute_vertex_normals()
    reflector1.paint_uniform_color([0.79, 0.50, 0.24])
    reflector1.translate(translate_dist, relative=True)

    return reflector1


def ringarray(
    ring_rad, elements, angle_offset, backing_rad, thickness, grid_resolution=20
):
    # define a single ring array, backed by a circular reflector
    reflector = circleReflector(backing_rad, backing_rad, thickness, grid_resolution)
    # define ring array, with element 0 on x axis
    element_angles = np.linspace(angle_offset, 2 * np.pi - angle_offset, elements)
    element_locations = np.zeros((elements, 3), dtype=np.float32)
    element_normals = np.zeros((elements, 3), dtype=np.float32)
    element_normals[:, 2] = 1.0
    element_locations[:, 0] = ring_rad * np.cos(element_angles)
    element_locations[:, 1] = ring_rad * np.sin(element_angles)
    mesh_points = o3d.geometry.PointCloud()
    mesh_points.points = o3d.utility.Vector3dVector(np.copy(element_locations))
    mesh_points.normals = o3d.utility.Vector3dVector(np.copy(element_normals))
    return reflector, mesh_points


def gridedCirclePoints(majorsize, minorsize, thickness, grid_resolution, sides="front"):
    """
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,1)
    """
    if sides == "all":
        r_space = np.linspace(
            grid_resolution,
            (majorsize / 2),
            int(
                np.max(np.asarray([2, np.ceil((majorsize * 0.5) / (grid_resolution))]))
            ),
        )
        c_space = np.ceil((2 * np.pi * r_space) / grid_resolution).astype(int)
        source_coords = np.empty(((1, 3)), dtype=np.float32)
        source_coords[0, 0] = 0
        source_coords[0, 1] = 0
        source_coords[0, 2] = 0
        for r_index in range(r_space.shape[0]):
            source_coords = np.append(
                source_coords,
                np.array(
                    [
                        r_space[r_index]
                        * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        r_space[r_index]
                        * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        np.zeros(c_space[r_index] - 1),
                    ]
                ).transpose(),
                axis=0,
            )

        source_normals = np.empty(((source_coords.shape[0], 3)), dtype=np.float32)
        source_normals[:] = 0.0
        source_normals[:, 2] = 1.0
        mesh_vertices = source_coords
        # temp removed sides
        # mesh_vertices=np.append(source_coords,back_coords,axis=0)
        mesh_normals = source_normals
    elif sides == "front":
        r_space = np.linspace(
            grid_resolution,
            (majorsize / 2),
            int(
                np.max(np.asarray([2, np.ceil((majorsize * 0.5) / (grid_resolution))]))
            ),
        )
        c_space = np.ceil((2 * np.pi * r_space) / grid_resolution).astype(int)
        source_coords = np.empty(((1, 3)), dtype=np.float32)
        source_coords[0, 0] = 0
        source_coords[0, 1] = 0
        source_coords[0, 2] = 0
        for r_index in range(r_space.shape[0]):
            source_coords = np.append(
                source_coords,
                np.array(
                    [
                        r_space[r_index]
                        * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        r_space[r_index]
                        * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        np.zeros(c_space[r_index] - 1),
                    ]
                ).transpose(),
                axis=0,
            )

        source_normals = np.empty(((source_coords.shape[0], 3)), dtype=np.float32)
        source_normals[:] = 0.0
        source_normals[:, 2] = 1.0
        mesh_vertices = source_coords
        # temp removed sides
        # mesh_vertices=np.append(source_coords,back_coords,axis=0)
        mesh_normals = source_normals

    mesh_points = o3d.geometry.PointCloud()
    mesh_points.points = o3d.utility.Vector3dVector(np.copy(mesh_vertices))
    mesh_points.normals = o3d.utility.Vector3dVector(np.copy(mesh_normals))
    return mesh_points


def gridedBullsEyePoints(
    diameter,
    ring_start,
    ring_num,
    ring_wavelength,
    ring_amplitude,
    thickness,
    grid_resolution,
    sides="all",
):
    """
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,1)
    """
    if sides == "front":
        sine_start = 5e-3
        # ring_num=2
        # sine_wavelength=5e-3
        # sine_amplitude=(0.91/2)*1e-3
        r_space = np.linspace(
            grid_resolution,
            (diameter / 2),
            int(np.max(np.asarray([2, np.ceil((diameter * 0.5) / (grid_resolution))]))),
        )
        c_space = np.ceil((2 * np.pi * r_space) / grid_resolution).astype(int)
        source_coords = np.empty(((1, 3)), dtype=np.float32)
        source_coords[0, 0] = 0
        source_coords[0, 1] = 0
        source_coords[0, 2] = 0
        for r_index in range(r_space.shape[0]):
            if (
                ring_start
                <= r_space[r_index]
                <= (ring_start + ring_num * ring_wavelength)
            ):
                source_coords = np.append(
                    source_coords,
                    np.array(
                        [
                            r_space[r_index]
                            * np.cos(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            r_space[r_index]
                            * np.sin(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            ring_amplitude
                            * np.sin(
                                (r_space[r_index] - ring_start)
                                * ((np.pi * 2) / ring_wavelength)
                                + np.pi / 2
                            )
                            * np.ones(c_space[r_index] - 1)
                            - (ring_amplitude),
                        ]
                    ).transpose(),
                    axis=0,
                )
            else:
                source_coords = np.append(
                    source_coords,
                    np.array(
                        [
                            r_space[r_index]
                            * np.cos(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            r_space[r_index]
                            * np.sin(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            np.zeros(c_space[r_index] - 1),
                        ]
                    ).transpose(),
                    axis=0,
                )

        # source_normals=np.empty(((source_coords.shape[0],3)),dtype=np.float32)
        # source_normals[:]=0.
        # source_normals[:,2]=1.
        mesh_vertices = source_coords
        # temp removed sides
        # mesh_vertices=np.append(source_coords,back_coords,axis=0)
        # mesh_normals=source_normals

        mesh_points = o3d.geometry.PointCloud()
        mesh_points.points = o3d.utility.Vector3dVector(np.copy(mesh_vertices))
        mesh_points.estimate_normals()
    elif sides == "all":
        r_space = np.linspace(
            grid_resolution,
            (diameter / 2),
            int(np.max(np.asarray([2, np.ceil((diameter * 0.5) / (grid_resolution))]))),
        )
        c_space = np.ceil((2 * np.pi * r_space) / grid_resolution).astype(int)
        source_coords = np.empty(((1, 3)), dtype=np.float32)
        source_coords[0, 0] = 0
        source_coords[0, 1] = 0
        source_coords[0, 2] = 0
        for r_index in range(r_space.shape[0]):
            if (
                ring_start
                <= r_space[r_index]
                <= (ring_start + ring_num * ring_wavelength)
            ):
                source_coords = np.append(
                    source_coords,
                    np.array(
                        [
                            r_space[r_index]
                            * np.cos(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            r_space[r_index]
                            * np.sin(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            ring_amplitude
                            * np.sin(
                                (r_space[r_index] - ring_start)
                                * ((np.pi * 2) / ring_wavelength)
                                + np.pi / 2
                            )
                            * np.ones(c_space[r_index] - 1)
                            - (ring_amplitude),
                        ]
                    ).transpose(),
                    axis=0,
                )
            else:
                source_coords = np.append(
                    source_coords,
                    np.array(
                        [
                            r_space[r_index]
                            * np.cos(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            r_space[r_index]
                            * np.sin(
                                np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]
                            ),
                            np.zeros(c_space[r_index] - 1),
                        ]
                    ).transpose(),
                    axis=0,
                )

            source_coords = np.append(
                source_coords,
                np.array(
                    [
                        r_space[r_index]
                        * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        r_space[r_index]
                        * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        np.zeros(c_space[r_index] - 1) - thickness,
                    ]
                ).transpose(),
                axis=0,
            )

        source_coords = np.append(
            source_coords,
            np.array(
                [
                    r_space[-1]
                    * np.cos(np.linspace(0, (2 * np.pi), c_space[-1])[0:-1]),
                    r_space[-1]
                    * np.sin(np.linspace(0, (2 * np.pi), c_space[-1])[0:-1]),
                    np.zeros(c_space[-1] - 1) - thickness / 2,
                ]
            ).transpose(),
            axis=0,
        )
        source_coords = np.append(
            source_coords,
            np.array(
                [
                    r_space[-1]
                    * np.cos(np.linspace(0, (2 * np.pi), c_space[-1])[0:-1]),
                    r_space[-1]
                    * np.sin(np.linspace(0, (2 * np.pi), c_space[-1])[0:-1]),
                    np.zeros(c_space[-1] - 1) - thickness,
                ]
            ).transpose(),
            axis=0,
        )
        source_coords = np.append(
            source_coords, np.array([0.0, 0.0, -thickness]).reshape(1, 3), axis=0
        )
        mesh_vertices = source_coords

        mesh_points = o3d.geometry.PointCloud()
        mesh_points.points = o3d.utility.Vector3dVector(np.copy(mesh_vertices))
        mesh_points.estimate_normals()

    return mesh_points


def gridedReflectorPoints(
    majorsize, minorsize, thickness, grid_resolution, sides="all"
):
    """
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,1)
    """
    if sides == "all":
        x = np.linspace(
            -(majorsize / 2),
            (majorsize / 2),
            int(np.max(np.asarray([2, np.ceil(majorsize / (grid_resolution))]))),
        )
        y = np.linspace(
            -(minorsize / 2),
            (minorsize / 2),
            int(np.max(np.asarray([2, np.ceil(minorsize / (grid_resolution))]))),
        )
        xback = np.linspace(
            -(majorsize / 2),
            (majorsize / 2),
            int(np.max(np.asarray([2, np.ceil(majorsize / (grid_resolution))]))),
        )
        yback = np.linspace(
            -(minorsize / 2),
            (minorsize / 2),
            int(np.max(np.asarray([2, np.ceil(minorsize / (grid_resolution))]))),
        )
        majorsidey = np.linspace(
            -(minorsize / 2),
            (minorsize / 2),
            int(np.max(np.asarray([2, np.ceil(minorsize / grid_resolution)]))),
        )
        majorsidez = np.linspace(
            -thickness,
            0,
            int(np.max(np.asarray([2, np.ceil(thickness / grid_resolution)]))),
        )
        minorsidex = np.linspace(
            -(majorsize / 2),
            (majorsize / 2),
            int(np.max(np.asarray([2, np.ceil(majorsize / grid_resolution)]))),
        )
        minorsidez = np.linspace(
            -thickness,
            0,
            int(np.max(np.asarray([2, np.ceil(thickness / grid_resolution)]))),
        )
        xx, yy = np.meshgrid(x, y)
        my, mz = np.meshgrid(majorsidey, majorsidez)
        mx = np.zeros((len(np.ravel(my))))
        majorside1 = np.empty(((len(np.ravel(mx)), 3)), dtype=np.float32)
        majorside2 = np.empty(((len(np.ravel(mx)), 3)), dtype=np.float32)
        majorside1[:, 0] = mx + majorsize / 2.0  # +EPSILON
        majorside1[:, 1] = np.ravel(my)
        majorside1[:, 2] = np.ravel(mz)
        majorside2[:, 0] = mx - majorsize / 2.0  # -EPSILON
        majorside2[:, 1] = np.ravel(my)
        majorside2[:, 2] = np.ravel(mz)
        majorside1normals = np.empty(((len(np.ravel(mx)), 3)), dtype=np.float32)
        majorside2normals = np.empty(((len(np.ravel(mx)), 3)), dtype=np.float32)
        majorside1normals[:] = 0
        majorside2normals[:] = 0
        majorside1normals[:, 0] = 1.0
        majorside2normals[:, 0] = -1.0
        max1, maz1 = np.meshgrid(minorsidex, minorsidez)
        may = np.zeros((len(np.ravel(max1))))
        minorside1 = np.empty(((len(np.ravel(max1)), 3)), dtype=np.float32)
        minorside2 = np.empty(((len(np.ravel(max1)), 3)), dtype=np.float32)
        minorside1[:, 0] = np.ravel(max1)
        minorside1[:, 1] = np.ravel(may) + minorsize / 2.0  # +EPSILON
        minorside1[:, 2] = np.ravel(maz1)
        minorside2[:, 0] = np.ravel(max1)
        minorside2[:, 1] = np.ravel(may) - minorsize / 2.0  # -EPSILON
        minorside2[:, 2] = np.ravel(maz1)
        minorside1normals = np.empty(((len(np.ravel(max1)), 3)), dtype=np.float32)
        minorside2normals = np.empty(((len(np.ravel(max1)), 3)), dtype=np.float32)
        minorside1normals[:] = 0
        minorside2normals[:] = 0
        minorside1normals[:, 1] = 1.0
        minorside2normals[:, 1] = -1.0
        zz = np.zeros((len(np.ravel(xx)), 1))  # +0.00025
        source_coords = np.empty(((len(np.ravel(xx)), 3)), dtype=np.float32)
        source_normals = np.empty(((len(np.ravel(xx)), 3)), dtype=np.float32)
        source_normals[:] = 0.0
        source_normals[:, 2] = 1.0
        source_coords[:, 0] = np.ravel(xx)
        source_coords[:, 1] = np.ravel(yy)
        source_coords[:, 2] = np.ravel(zz)
        backx, backy = np.meshgrid(xback, yback)
        backz = np.empty((len(np.ravel(backx)), 1), dtype=np.float32)
        backz[:] = -thickness
        back_coords = np.empty(((len(np.ravel(backx)), 3)), dtype=np.float32)
        back_normals = np.empty(((len(np.ravel(backx)), 3)), dtype=np.float32)
        back_coords[:, 0] = np.ravel(backx)
        back_coords[:, 1] = np.ravel(backy)
        back_coords[:, 2] = np.ravel(backz)
        back_normals[:] = 0.0
        back_normals[:, 2] = -1.0
        top_bottom = np.append(majorside1, majorside2, axis=0)
        top_bottom_normals = np.append(majorside1normals, majorside2normals, axis=0)
        left_right = np.append(minorside1, minorside2, axis=0)
        left_right_normals = np.append(minorside1normals, minorside2normals, axis=0)
        sides = np.append(top_bottom, left_right, axis=0)
        side_normals = np.append(top_bottom_normals, left_right_normals, axis=0)
        mesh_vertices = np.append(
            np.append(source_coords, back_coords, axis=0), sides, axis=0
        )
        # temp removed sides
        # mesh_vertices=np.append(source_coords,back_coords,axis=0)
        mesh_normals = np.append(
            np.append(source_normals, back_normals, axis=0), side_normals, axis=0
        )
        # mesh_normals=np.append(source_normals,back_normals,axis=0)
        mesh_points = o3d.geometry.PointCloud()
        mesh_points.points = o3d.utility.Vector3dVector(np.copy(mesh_vertices))
        mesh_points.normals = o3d.utility.Vector3dVector(np.copy(mesh_normals))
    elif sides == "front":
        x = np.linspace(
            -(majorsize / 2),
            (majorsize / 2),
            int(np.max(np.asarray([2, np.ceil(majorsize / (grid_resolution))]))),
        )
        y = np.linspace(
            -(minorsize / 2),
            (minorsize / 2),
            int(np.max(np.asarray([2, np.ceil(minorsize / (grid_resolution))]))),
        )
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros((len(np.ravel(xx)), 1))  # +0.00025
        source_coords = np.empty(((len(np.ravel(xx)), 3)), dtype=np.float32)
        source_normals = np.empty(((len(np.ravel(xx)), 3)), dtype=np.float32)
        source_normals[:] = 0.0
        source_normals[:, 2] = 1.0
        source_coords[:, 0] = np.ravel(xx)
        source_coords[:, 1] = np.ravel(yy)
        source_coords[:, 2] = np.ravel(zz)
        mesh_vertices = source_coords
        # temp removed sides
        # mesh_vertices=np.append(source_coords,back_coords,axis=0)
        mesh_normals = source_normals
    elif sides == "centres":
        source_coords = np.zeros((1, 3), dtype=np.float32)
        source_normals = np.zeros((1, 3), dtype=np.float32)
        source_normals[0, 2] = 1
        mesh_vertices = source_coords
        mesh_normals = source_normals

    mesh_points = o3d.geometry.PointCloud()
    mesh_points.points = o3d.utility.Vector3dVector(np.copy(mesh_vertices))
    mesh_points.normals = o3d.utility.Vector3dVector(np.copy(mesh_normals))
    return mesh_points


def gridedParabola(diameter, focal_length, thickness, grid_resolution, sides="all"):
    """
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,1)
    equation of parabola is y=(1/(4*focal_length))*x**2, this will then be rotated in the Az domain.
    Need to ensure that the euclidean distance between each point is grid_resolution or less
    """
    if sides == "all":
        x_space = np.linspace(
            grid_resolution,
            (diameter / 2),
            int(np.max(np.asarray([2, np.ceil((diameter * 0.5) / (grid_resolution))]))),
        )
        z_space = (1 / (4 * focal_length)) * x_space ** 2
        c_space = np.ceil((2 * np.pi * x_space) / grid_resolution).astype(int)
        normal_gradiant_vector = np.array(
            [
                np.ones((len(x_space))),
                np.zeros((len(x_space))),
                -1 / (1 / (2 * focal_length) * x_space),
            ]
        )
        source = np.array([x_space, np.zeros((len(x_space))), z_space]).transpose()
        target = (normal_gradiant_vector * -x_space).transpose()
        base_directions = np.zeros((x_space.shape[0], 3), dtype=np.float32)
        norm_length = np.zeros((x_space.shape[0], 1), dtype=np.float32)
        base_directions, norm_length = math_functions.calc_dv_norm(
            source, target, base_directions, norm_length
        )
        base_directions = np.append(base_directions, np.array([[1, 0, 0]]), axis=0)
        source_coords = np.empty(((1, 3)), dtype=np.float32)
        source_coords[0, :] = 0
        source_normals = np.empty(((1, 3)), dtype=np.float32)
        source_normals[0, :] = 0
        source_normals[0, 2] = 1
        for r_index in range(x_space.shape[0]):
            source_coords = np.append(
                source_coords,
                np.array(
                    [
                        x_space[r_index]
                        * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        x_space[r_index]
                        * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        z_space[r_index] * np.ones(c_space[r_index] - 1),
                    ]
                ).transpose(),
                axis=0,
            )
            source_normals = np.append(
                source_normals,
                np.array(
                    [
                        base_directions[r_index, 0]
                        * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        base_directions[r_index, 0]
                        * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        base_directions[r_index, 2] * np.ones(c_space[r_index] - 1),
                    ]
                ).transpose(),
                axis=0,
            )

        mesh_vertices = np.append(
            np.append(
                source_coords - np.array([0, 0, np.max(source_coords[:, 2])]),
                source_coords
                - np.array([0, 0, (np.max(source_coords[:, 2]) + thickness)]),
                axis=0,
            ),
            np.array(
                [
                    x_space[-1]
                    * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                    x_space[-1]
                    * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                    z_space[-1] * np.ones(c_space[-1] - 1),
                ]
            ).transpose()
            - np.array([0, 0, (np.max(source_coords[:, 2]) + thickness / 2)]),
            axis=0,
        )
        mesh_normals = np.append(
            np.append(source_normals, source_normals * -1, axis=0),
            np.array(
                [
                    base_directions[-1, 0]
                    * np.cos(np.linspace(0, (2 * np.pi), c_space[-1])[0:-1]),
                    base_directions[-1, 0]
                    * np.sin(np.linspace(0, (2 * np.pi), c_space[-1])[0:-1]),
                    np.zeros(c_space[-1] - 1),
                ]
            ).transpose(),
            axis=0,
        )

    elif sides == "front":
        x_space = np.linspace(
            grid_resolution,
            (diameter / 2),
            int(np.max(np.asarray([2, np.ceil((diameter * 0.5) / (grid_resolution))]))),
        )
        z_space = (1 / (4 * focal_length)) * x_space ** 2
        c_space = np.ceil((2 * np.pi * x_space) / grid_resolution).astype(int)
        normal_gradiant_vector = np.array(
            [
                np.ones((len(x_space))),
                np.zeros((len(x_space))),
                -1 / (1 / (2 * focal_length) * x_space),
            ]
        )
        source = np.array([x_space, np.zeros((len(x_space))), z_space]).transpose()
        target = (normal_gradiant_vector * -x_space).transpose()
        base_directions = np.zeros((x_space.shape[0], 3), dtype=np.float32)
        norm_length = np.zeros((x_space.shape[0], 1), dtype=np.float32)
        base_directions, norm_length = math_functions.calc_dv_norm(
            source, target, base_directions, norm_length
        )
        source_coords = np.empty(((1, 3)), dtype=np.float32)
        source_coords[0, :] = 0
        source_normals = np.empty(((1, 3)), dtype=np.float32)
        source_normals[0, :] = 0
        source_normals[0, 2] = 1
        for r_index in range(x_space.shape[0]):
            source_coords = np.append(
                source_coords,
                np.array(
                    [
                        x_space[r_index]
                        * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        x_space[r_index]
                        * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        z_space[r_index] * np.ones(c_space[r_index] - 1),
                    ]
                ).transpose(),
                axis=0,
            )
            source_normals = np.append(
                source_normals,
                np.array(
                    [
                        base_directions[r_index, 0]
                        * np.cos(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        base_directions[r_index, 0]
                        * np.sin(np.linspace(0, (2 * np.pi), c_space[r_index])[0:-1]),
                        base_directions[r_index, 2] * np.ones(c_space[r_index] - 1),
                    ]
                ).transpose(),
                axis=0,
            )

        mesh_vertices = source_coords - np.array([0, 0, np.max(source_coords[:, 2])])
        mesh_normals = source_normals

    mesh_points = o3d.geometry.PointCloud()
    mesh_points.points = o3d.utility.Vector3dVector(np.copy(mesh_vertices))
    mesh_points.normals = o3d.utility.Vector3dVector(np.copy(mesh_normals))
    return mesh_points


def BullsEye(O2, n_rings, innerdia, period, ht, basethick, grid_resolution):
    # create bullseye lens structure on a reflector, this is a temp setup
    rotate_nums = 31
    # make 5 rings with a preiod of 5mm between them.
    # period=4.53e-3# period of the rings
    # n_rings=2# number of rings to make
    # ht=(0.91e-3)/2# amplitude  of ring waves
    m_rings = 0  # ring to start at
    # innerdia=7.12e-3# diameter of inner ring
    # basethick=3e-3# thickness of the base below the bottom of the slots5
    # O2=2e-3
    # generate appropriatly paced sine curve for face of array, then convert to point list with .tolist()
    # convert with grid resolution between each points, from innerdia to (innerdia+period*n_rings+)
    points2d = np.zeros()
    # face_curve=points2d.tolist()

    centre = sd.translate([0, 0, -basethick])(
        sd.cylinder(
            r=innerdia / 2 + m_rings * period + period * 0.01,
            h=basethick,
            segments=rotate_nums,
        )
    )
    for p in range(m_rings, n_rings + 1):
        centre += sd.rotate_extrude(angle=360, segments=rotate_nums)(
            sd.translate([innerdia / 2 + p * period, 0, 0])(
                sd.scale([period, ht, 1])(
                    sd.polygon(
                        points=[
                            [0.000000, 0.000000],
                            [0.025000, -0.012312],
                            [0.050000, -0.048943],
                            [0.075000, -0.108993],
                            [0.100000, -0.190983],
                            [0.125000, -0.292893],
                            [0.150000, -0.412215],
                            [0.175000, -0.546010],
                            [0.200000, -0.690983],
                            [0.225000, -0.843566],
                            [0.250000, -1.000000],
                            [0.275000, -1.156434],
                            [0.300000, -1.309017],
                            [0.325000, -1.453990],
                            [0.350000, -1.587785],
                            [0.375000, -1.707107],
                            [0.400000, -1.809017],
                            [0.425000, -1.891007],
                            [0.450000, -1.951057],
                            [0.475000, -1.987688],
                            [0.500000, -2.000000],
                            [0.525000, -1.987688],
                            [0.550000, -1.951057],
                            [0.575000, -1.891007],
                            [0.600000, -1.809017],
                            [0.625000, -1.707107],
                            [0.650000, -1.587785],
                            [0.675000, -1.453990],
                            [0.700000, -1.309017],
                            [0.725000, -1.156434],
                            [0.750000, -1.000000],
                            [0.775000, -0.843566],
                            [0.800000, -0.690983],
                            [0.825000, -0.546010],
                            [0.850000, -0.412215],
                            [0.875000, -0.292893],
                            [0.900000, -0.190983],
                            [0.925000, -0.108993],
                            [0.950000, -0.048943],
                            [0.975000, -0.012312],
                            [1.000000, 0.000000],
                            [1, -basethick / ht],
                            [0, -basethick / ht],
                        ]
                    )
                )
            )
        )

    centre += sd.translate([0, 0, -basethick / 2])(
        sd.cylinder(
            r=innerdia / 2 + (n_rings + 1) * period + O2,
            h=basethick,
            center=True,
            segments=rotate_nums,
        )
        - sd.cylinder(
            r=innerdia / 2 + (n_rings + 1) * period - period * 0.01,
            h=2 * basethick,
            center=True,
            segments=rotate_nums,
        )
    )

    # generate valid openscad code and store it in file
    sd.scad_render_to_file(centre, "d.scad")

    # run openscad and export to stl
    #run(["openscad-nightly", "-o", "d.stl", "d.scad", "--export-format=binstl"])
    converttostl()

    solid = o3d.io.read_triangle_mesh("d.stl")
    solid.compute_vertex_normals()
    # o3d.visualization.draw_geometries([test_structure])

    test_points = np.asarray(solid.vertices)
    test_normals = np.asarray(solid.vertex_normals)
    image_points = RF.points2pointcloud(
        np.copy(test_points[test_points[:, 2] >= -ht * 2, :])
    )
    image_points.normals = o3d.utility.Vector3dVector(
        np.copy(test_normals[test_points[:, 2] >= -ht * 2, :])
    )
    return solid, image_points


def defineParabola(diameter, focal_length, thickness, grid_resolution):
    # create shape from points
    mesh_points = gridedParabola(
        diameter, focal_length, thickness, grid_resolution, sides="all"
    )
    distances = mesh_points.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        mesh_points, o3d.utility.DoubleVector([radius, radius * 2])
    )
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #       mesh_points,
    #       o3d.utility.DoubleVector([radius, radius * 2]))

    mesh.compute_triangle_normals()
    return mesh


def parabolic_reflector_segment():
    """
    This function is intended to generate a segment parabolic reflector, such as used for some radars
    """
    reflector = o3d.geometry.TriangleMesh()
    mesh_points = o3d.geometry.PointCloud()
    return reflector, mesh_points


def CASSIOPeiA_triplet(
    wavelength, triplet_number, triplet_spacing=1.0, offset_angle=0.0
):
    """
    convinience function to create the CASSIOPeiA triplets, three dipole antennas with centres space a quarter
    wavelength apart. The efield vectors are arranged for z direction polarisation.

    Parameters
    ----------
    wavelength : float
        the wavelength of interest
    triplet_number: int
        the number of triplets required, to be generated with center at the origin, and extending symmetrically
        in the x axis. If the number is even then the central clusters will be spaced symmetrically either side of
        the origin. If odd, then there will be a central cluster.
    triplet_spacing : float
        the spacing between the centres of the triplets in the xy plane (m)
    offset_angle:float
        offset angle of the primary antenna element from the x axis. Default is zero, but the clusters can be rotated
        around by altering this parameter.

    Returns
    -------
    source_points:
        the positions of the antennas with normal vectors, the normals are defined arbitarily to aid in the helical
        arrangement of the layers
    efield_vectors:
        the polarisation vectors of the triplets.
    """
    source_points = o3d.geometry.PointCloud()
    efield_vectors = np.zeros((triplet_number * 3, 3), dtype=np.complex64)
    efield_vectors[:, 2] = 1.0
    triplet_points = np.zeros((3, 3), dtype=np.float32)
    # spacing is equivalent to an equilateral triangle
    antenna_spacing = wavelength * 0.25
    triplet_points[0, 0] = (3 ** 0.5 / 3) * antenna_spacing
    # rotate as required for primary element offset
    offsetangle = np.radians(offset_angle)
    rot_mat = np.asarray(
        [
            [np.cos(offsetangle), -np.sin(offsetangle), 0],
            [np.sin(offsetangle), np.cos(offsetangle), 0],
            [0, 0, 1],
        ]
    )
    triplet_points[0, :] = (triplet_points[0, :] * rot_mat)[:, 0]
    angle = np.radians(120)
    rot_mat = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    triplet_points[1, :] = (triplet_points[0, :] * rot_mat)[:, 0]
    angle = np.radians(240)
    rot_mat = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    triplet_points[2, :] = (triplet_points[0, :] * rot_mat)[:, 0]
    # now lay out the clusters at the required spacing along the x axis.
    outer_cluster = ((triplet_number - 1) / 2) * triplet_spacing
    cluster_centres = np.linspace(-outer_cluster, outer_cluster, triplet_number)
    cluster_points = np.tile(triplet_points, [triplet_number, 1])
    for cluster in range(triplet_number):
        cluster_points[cluster * 3 : (cluster + 1) * 3, 0] += cluster_centres[cluster]
    normals = np.zeros((triplet_number * 3, 3), dtype=np.float32)
    normals[:, 1] = 1
    source_points.points = o3d.utility.Vector3dVector(cluster_points)
    source_points.normals = o3d.utility.Vector3dVector(normals)
    # create board and center
    plane = o3d.geometry.TriangleMesh.create_box(
        width=2 * (outer_cluster + triplet_spacing * 0.5),
        height=wavelength * 0.5,
        depth=wavelength * 0.05,
    )
    plane.translate(
        [
            -(outer_cluster + triplet_spacing * 0.5),
            -wavelength * 0.25,
            -wavelength * 0.025,
        ],
        relative=True,
    )

    return source_points, plane, efield_vectors


def CASSIOPeiA_Array(
    wavelength,
    num_rows,
    num_col,
    row_spacing=1.0,
    triplet_spacing=1.0,
    offset_angle=0.0,
    total_twist_angle=180,
):
    adjusted_twist_angle=(total_twist_angle/(num_rows+1))*num_rows
    outer_row = ((num_rows - 1) / 2) * row_spacing
    row_centres = np.linspace(-outer_row, outer_row, num_rows)
    angle_offsets = np.linspace(-adjusted_twist_angle / 2, adjusted_twist_angle / 2, num_rows)
    array_points = o3d.geometry.PointCloud()
    array_structure = o3d.geometry.TriangleMesh()
    # create row of triplet clusters
    source_points, plane, efield_vectors = CASSIOPeiA_triplet(
        wavelength, num_col, triplet_spacing=triplet_spacing, offset_angle=offset_angle
    )
    for row in range(num_rows):
        temp_row = copy.deepcopy(source_points)
        temp_plane = copy.deepcopy(plane)
        temp_row.translate([0, 0, row_centres[row]], relative=True)
        temp_plane.translate([0, 0, row_centres[row]], relative=True)
        rotation_vector = np.radians(np.array([0, 0, angle_offsets[row]]))
        rotation_matrix = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
            rotation_vector
        )
        temp_row = GF.open3drotate(temp_row, rotation_matrix)
        temp_plane = GF.open3drotate(temp_plane, rotation_matrix)
        array_points += temp_row
        array_structure += temp_plane

    array_efield_vectors = np.tile(efield_vectors, [num_rows, 1])
    return array_points, array_structure, array_efield_vectors

def chain_home_transmitter():
    """
    This function generates the required antenna geometry to model the Chain Home transmitter, which operated at 20MHz.
    This model is based upon the information at http://www.johnhearfield.com/Radar/Magnetron.htm
    Returns
    -------
    chain_home_transmit

    """
    wavelength=3e8/20e6
    #eight dipoles vertically stacked half wavelength spacing, with reflectors behind each, seperated by 0.18 wavelengths horizontally, a stack for each pair of towers? So three stacks for the early quad arrangements, and two stacks for the later triple towers?
    #mean height of the array is 215 feet, and claimed main lobe at 2.6 degrees in elevation due to ground reflection, first null at 5.2 degrees, and a horizontal beamwidth of 100 degrees
    chain_home_transmit=antenna_structures()
    return chain_home_transmit

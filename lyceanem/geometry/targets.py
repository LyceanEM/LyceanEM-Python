#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import numpy as np
import meshio
import pyvista as pv
import pygmsh
from importlib_resources import files
from scipy.spatial.transform import Rotation as R

from ..base_classes import antenna_structures, structures, points
from ..geometry import geometryfunctions as GF
from ..raycasting import rayfunctions as RF
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
    NasaAlmond : meshio triangle mesh
        the physical structure of the nasa almond
    NasaAlmond_points : o3d points
        the mesh points for scattering

    """
    if resolution == "half":
        stream = files("lyceanem.geometry.data").joinpath(
            "NasaAlmondHalfWavelengthv2.stl"
        )
        NasaAlmond = meshio.read(str(stream))
        nasa = pv.read(str(stream))

    elif resolution == "quarter":
        stream = files("lyceanem.geometry.data").joinpath(
            "NasaAlmondQuarterWavelengthv2.stl"
        )
        NasaAlmond = meshio.read(str(stream))
        nasa = pv.read(str(stream))
    elif resolution == "tenth":
        stream = files("lyceanem.geometry.data").joinpath(
            "NasaAlmondTenthWavelengthv2.stl"
        )
        NasaAlmond = meshio.read(str(stream))
        nasa = pv.read(str(stream))

    nasa.compute_normals(inplace=True)

    NasaAlmond.point_data["Normals"] = nasa.point_normals

    # points=np.asarray(NasaAlmond.vertices)
    # normals=np.asarray(NasaAlmond.vertex_normals)
    _, scatter_cloud = GF.tri_centroids(NasaAlmond)
    return NasaAlmond, scatter_cloud


def converttostl():
    """
    This function is a convinence to allow the repeatable used of openscad to generate watertight stl files for use within LyceanEM. This function assumes that all openscad files are saved as temp.scad in the current working directory, and the output will be saved as temp.stl, ready for the home function to import it.

    Returns
    -------
    Nothing

    """
    import subprocess

    try:
        # print(os.getcwd())
        working_directory = "."
        # run([, ])
        p = subprocess.Popen(
            [
                "openscad-nightly",
                "-o",
                "temp.stl",
                "temp.scad",
                "--export-format=binstl",
            ],
            cwd=working_directory,
        )
        p.wait()
    except:
        # run(["openscad", "-o", "temp.stl", "temp.scad", "--export-format=binstl"])
        p = subprocess.Popen(
            ["openscad", "-o", "temp.stl", "temp.scad", "--export-format=binstl"],
            cwd=working_directory,
        )
        p.wait()





def rectReflector(majorsize, minorsize, thickness):
    """
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,0)
    """
    print("majorsize",majorsize)
    print("minorsize",minorsize)
    print("thickness",thickness)

    halfMajor = majorsize / 2.0
    halfMinor = minorsize / 2.0
    pv_mesh = pv.Box((-halfMajor, halfMajor, -halfMinor, halfMinor, -(thickness+EPSILON),- EPSILON))
    pv_mesh = pv_mesh.triangulate()
    pv_mesh.compute_normals(inplace=True,consistent_normals=False)
    triangles = np.reshape(np.array(pv_mesh.faces),(12,4))
    triangles = triangles[:,1:]

    mesh = meshio.Mesh(pv_mesh.points, {"triangle": triangles})



    mesh.point_data["Normals"] = pv_mesh.point_normals
    mesh.cell_data["Normals"] = pv_mesh.cell_normals
    red = np.zeros((8, 1), dtype=np.float32) 
    green = np.ones((8, 1), dtype=np.float32) * 0.259
    blue = np.ones((8, 1), dtype=np.float32) * 0.145

    mesh.point_data["red"] = red
    mesh.point_data["green"] = green
    mesh.point_data["blue"] = blue
    return mesh


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
    triangle_list = np.zeros(((12, 3)), dtype=int)
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
    mesh = meshio.Mesh(
        points=mesh_vertices,
        cells=[("triangle", triangle_list)],)
    print(mesh)
    triangle_list = np.insert(triangle_list, 0, 3, axis=1)
    
    pv_mesh = pv.PolyData( mesh_vertices, faces = triangle_list)
    pv_mesh.compute_normals(inplace=True,consistent_normals=False)

    mesh.point_data["Normals"] = np.asarray(pv_mesh.point_normals)
    mesh.cell_data["Normals"] = np.asarray(pv_mesh.cell_normals)

    red = np.zeros((8, 1), dtype=np.float32) 
    green = np.ones((8, 1), dtype=np.float32) * 0.259
    blue = np.ones((8, 1), dtype=np.float32) * 0.145

    mesh.point_data["red"] = red
    mesh.point_data["green"] = green
    mesh.point_data["blue"] = blue
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
    print("meshing reflector")
    print("args", majorsize, minorsize, thickness)
    reflector = rectReflector(majorsize, minorsize, thickness)
    mesh_points = gridedReflectorPoints(
        majorsize, minorsize, thickness, grid_resolution, sides
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
    print("HIHIH")
    structure = shapeTrapezoid(
        majorsize + (edge_width * 2), minorsize + (edge_width * 2), length, flare_angle
    )
    mesh_points = gridedReflectorPoints(
        majorsize, minorsize, 1e-6, grid_resolution, sides
    )
    mesh_points = GF.translate_mesh(mesh_points, [0, 0, 1e-6])

    mesh_points = GF.translate_mesh(mesh_points, [0, 0, 1e-6])

    return structure, mesh_points


def parabolic_aperture(diameter, focal_length, thickness, mesh_size, sides='front'):
    # Define function for parabola equation (y^2 = 4*focal_length*x)
    def parabola(x):
        return (1 / (4 * focal_length)) * x ** 2

    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_max = mesh_size
        # Define points
        cp1 = geom.add_point([0, 0, 0])  # Center point
        cp2 = geom.add_point([diameter * 0.5 * (1 / 6), 0, parabola(diameter * 0.5 * (1 / 6))])
        cp3 = geom.add_point([diameter * 0.5 * (2 / 6), 0, parabola(diameter * 0.5 * (2 / 6))])
        cp4 = geom.add_point([diameter * 0.5 * (3 / 6), 0, parabola(diameter * 0.5 * (3 / 6))])
        cp5 = geom.add_point([diameter * 0.5 * (4 / 6), 0, parabola(diameter * 0.5 * (4 / 6))])
        cp6 = geom.add_point([diameter * 0.5 * (5 / 6), 0, parabola(diameter * 0.5 * (5 / 6))])
        cp7 = geom.add_point([diameter * 0.5 * (6 / 6), 0, parabola(diameter * 0.5 * (6 / 6))])

        # Define top line based on points
        line = geom.add_bspline([cp1, cp2, cp3, cp4, cp5, cp6, cp7])

        _, surface, _ = geom.extrude(line, translation_axis=[0.0, 0.0, -thickness])

        # Revolve line to create revolution surface
        volume_list = []
        _, b, _ = geom.revolve(surface, rotation_axis=[0.0, 0.0, 1.0], point_on_axis=[0.0, 0.0, 0.0],
                               angle=0.25 * np.pi)
        volume_list.append(b)
        for inc in range(7):

            geom.rotate(surface, point=[0.0, 0.0, 0.0], angle=(1 / 4) * np.pi, axis=[0.0, 0.0, 1.0])
            _, b2, _ = geom.revolve(surface, rotation_axis=[0.0, 0.0, 1.0], point_on_axis=[0.0, 0.0, 0.0],
                                    angle=0.25 * np.pi)
            volume_list.append(b2)

        full_reflector = geom.boolean_union(volume_list)

        mesh_temp = geom.generate_mesh(dim=2)
    for inc, cell in enumerate(mesh_temp.cells):
        if cell.type == 'triangle':
            triangle_index = inc

    import meshio
    triangle_cells = [("triangle", mesh_temp.cells[triangle_index].data)]
    mesh = meshio.Mesh(mesh_temp.points, triangle_cells)

    x_space = np.linspace(
        mesh_size,
        (diameter / 2),
        int(np.max(np.asarray([2, np.ceil((diameter * 0.5) / (mesh_size))]))),
    )
    z_space = (1 / (4 * focal_length)) * x_space ** 2
    c_space = np.ceil((2 * np.pi * x_space) / mesh_size).astype(int)
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

    mesh_vertices = source_coords + np.array([0, 0, 1e-6])
    mesh_normals = source_normals
    aperture_points = meshio.Mesh(points=mesh_vertices,
                                  cells=[("vertex", np.array([[i, ] for i in range(len(mesh_vertices))]))],
                                  point_data={'Normals': mesh_normals})

    return mesh, aperture_points






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

    mesh_points = meshio.Mesh(points=mesh_vertices, cells=[], point_data={"Normals": mesh_normals})

    return mesh_points

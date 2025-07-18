#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from importlib.resources import files

import meshio
import sys
import numpy as np
import gmsh
import pyvista as pv


EPSILON = 1e-6  # how close to zero do we consider zero?


def rectReflector(majorsize, minorsize, thickness):
    """
    :meta private:
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,0)
    """
    print("majorsize", majorsize)
    print("minorsize", minorsize)
    print("thickness", thickness)

    halfMajor = majorsize / 2.0
    halfMinor = minorsize / 2.0
    pv_mesh = pv.Box(
        (-halfMajor, halfMajor, -halfMinor, halfMinor, -(thickness + EPSILON), -EPSILON)
    )
    pv_mesh = pv_mesh.triangulate()
    pv_mesh.compute_normals(inplace=True, consistent_normals=False)
    from ..utility.mesh_functions import pyvista_to_meshio

    triangles = np.reshape(np.array(pv_mesh.faces), (12, 4))
    triangles = triangles[:, 1:]

    # mesh = meshio.Mesh(pv_mesh.points, {"triangle": triangles})
    mesh = pyvista_to_meshio(pv_mesh)

    from .geometryfunctions import compute_normals, compute_areas

    mesh = compute_areas(compute_normals(mesh))
    red = np.zeros((8, 1), dtype=np.float32)
    green = np.ones((8, 1), dtype=np.float32) * 0.259
    blue = np.ones((8, 1), dtype=np.float32) * 0.145

    mesh.point_data["red"] = red
    mesh.point_data["green"] = green
    mesh.point_data["blue"] = blue
    return mesh


def shapeTrapezoid(x_size, y_size, length, flare_angle):
    """
    :meta private:
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
        cells=[("triangle", triangle_list)],
    )
    # print(mesh)
    triangle_list = np.insert(triangle_list, 0, 3, axis=1)

    pv_mesh = pv.PolyData(mesh_vertices, faces=triangle_list)
    pv_mesh.compute_normals(inplace=True, consistent_normals=False)

    mesh.point_data["Normals"] = np.asarray(pv_mesh.point_normals)
    mesh.cell_data["Normals"] = [np.asarray(pv_mesh.cell_normals)]
    from .geometryfunctions import compute_areas, compute_normals

    mesh = compute_normals(compute_areas(mesh))
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
    reflector : :type:`meshio.Mesh`
        the defined cuboid
    mesh_points : :type:`meshio.Mesh`
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
    flare_angle : float
        the taper angle of the horn
    grid_resolution : float
        the spacing between the aperture points, should be half a wavelength at the frequency of interest
    sides : str
        command for the mesh, default is 'front', and for a horn, this should not be changed.

    Returns
    -------
    structure : :type:`meshio.Mesh`
        the physical structure of the horn
    mesh_points : :type:`meshio.Mesh`
        the source points for the horn aperture
    """
    # print("HIHIH")
    structure = shapeTrapezoid(
        majorsize + (edge_width * 2), minorsize + (edge_width * 2), length, flare_angle
    )
    mesh_points = gridedReflectorPoints(
        majorsize, minorsize, 1e-6, grid_resolution, sides
    )
    from .geometryfunctions import mesh_translate

    mesh_points = mesh_translate(mesh_points, [0, 0, EPSILON * 2])

    return structure, mesh_points


def parabolic_reflector(
    diameter,
    focal_length,
    thickness,
    mesh_size,
    lip=False,
    lip_height=1e-3,
    lip_width=1e-3,
    file_name="Parabolic_Reflector.stl",
    ONELAB=False,
):
    """
    Create a parabolic reflector with a specified diameter and focal length, and generate a mesh of points on the surface. If only the points on the front surface are required, then :func:`parabolic_surface` can be used instead.
    Alternatively if scattering points on all sides are desired, then the points from the reflector mesh may be used.

    Parameters
    ----------
    diameter : float
        Diameter of the parabolic reflector.
    focal_length : float
        Focal length of the parabolic reflector.
    thickness : float
        Thickness of the reflector.
    mesh_size : float
        Desired separation of the mesh points.
    lip : bool, optional
        If True, adds a flat lip to the reflector. Default is False.
    lip_height : float, optional
        Height of the reflector lip if `lip` is True. Default is 1e-3.
    lip_width : float, optional
        Width of the reflector lip if `lip` is True. Default is 1e-3.
    file_name : str, optional
        Name of the file to save the mesh. Default is "Parabolic_Reflector.stl".
    ONELAB : bool, optional
        If True, enables ONELAB for interactive geometry manipulation. Default is False.

    Returns
    -------
    mesh : :type:`meshio.Mesh`
        A mesh object containing the parabolic reflector.
    aperture_points : :type:`meshio.Mesh`
        A mesh object containing the points on the front surface of the parabolic reflector.

    """

    def parabola(x):
        return (1 / (4 * focal_length)) * x**2

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    # gmsh.option.setNumber("Geometry.OCCImportLabels", 1) # import colors from STEP
    # Create a new geometry object
    geo = gmsh.model.add("Parabolic Reflector")
    # Define points
    point_num = 15
    x_pos = np.linspace(0, 0.5 * diameter, point_num)
    z_pos = parabola(x_pos)
    coords = np.array([x_pos.ravel(), np.zeros((point_num)), z_pos.ravel()]).transpose()
    points_list = []
    for inc in range(point_num):
        points_list.append(gmsh.model.occ.add_point(*coords[inc, :].tolist()))

    # Define top line based on points
    line = gmsh.model.occ.add_bspline(points_list)

    temp = gmsh.model.occ.extrude([(1, line)], 0.0, 0.0, -thickness)
    surface = temp[1]

    # Revolve line to create revolution surface
    volume_list = []

    temp2 = gmsh.model.occ.revolve(
        [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
    )
    volume_list.append(temp2[1])
    for inc in range(7):
        gmsh.model.occ.rotate([surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, (1 / 4) * np.pi)

        temp3 = gmsh.model.occ.revolve(
            [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
        )
        volume_list.append(temp3[1])

    if lip:
        axis1 = np.array([0.0, 0.0, -lip_height])

        start_point = np.array([0.0, 0.0, parabola(diameter * 0.5)])
        cylinder1 = gmsh.model.occ.addCylinder(
            *start_point.ravel(), *axis1, diameter / 2
        )
        cylinder2 = gmsh.model.occ.addCylinder(
            *start_point.ravel(), *axis1, diameter / 2 + lip_width
        )
        final = gmsh.model.occ.cut([(3, cylinder2)], [(3, cylinder1)])[0]
        volume_list.append(final[0])

    full_reflector = gmsh.model.occ.fuse(volume_list[0:-1], [volume_list[-1]])

    gmsh.model.occ.synchronize()
    if "-nopopup" not in sys.argv and ONELAB:
        gmsh.fltk.run()

    gmsh.model.mesh.generate(dim=2)
    gmsh.write(file_name)
    gmsh.finalize()
    mesh = meshio.read(file_name)
    from .geometryfunctions import compute_normals, compute_areas

    mesh = compute_areas(compute_normals(mesh))
    aperture_points = parabolic_surface(
        diameter,
        focal_length,
        mesh_size,
        lip=lip,
        lip_width=lip_width,
        file_name="Parabolic_Surface.stl",
        ONELAB=ONELAB,
    )
    return mesh, aperture_points


def parabolic_surface(
    diameter,
    focal_length,
    mesh_size,
    lip=False,
    lip_width=1e-3,
    file_name="Parabolic_Surface.stl",
    ONELAB=False,
):
    """
    Create a parabolic surface with a specified diameter and focal length, and generate a mesh of points on the surface. This function is useful for generating the front surface of a parabolic reflector.

    Parameters
    ----------
    diameter : float
        Diameter of the parabolic surface.
    focal_length : float
        Focal length of the parabolic surface.
    mesh_size : float
        Desired separation of the mesh points.
    lip : bool, optional
        If True, adds a flat lip to the surface. Default is False.
    lip_width : float, optional
        Width of the surface lip if `lip` is True. Default is 1e-3.
    file_name : str, optional
        Name of the file to save the mesh. Default is "Parabolic_Surface.stl".
    ONELAB : bool, optional
        If True, enables ONELAB for interactive geometry manipulation. Default is False.

    Returns
    -------
    mesh : :type:`meshio.Mesh`
        A mesh object containing the parabolic surface.


    """

    def parabola(x):
        return (1 / (4 * focal_length)) * x**2

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    # gmsh.option.setNumber("Geometry.OCCImportLabels", 1) # import colors from STEP
    # Create a new geometry object
    geo = gmsh.model.add("Parabolic Surface")
    # Define points
    point_num = 15
    x_pos = np.linspace(0, 0.5 * diameter, point_num)
    z_pos = parabola(x_pos)
    coords = np.array([x_pos.ravel(), np.zeros((point_num)), z_pos.ravel()]).transpose()
    points_list = []
    for inc in range(point_num):
        points_list.append(gmsh.model.occ.add_point(*coords[inc, :].tolist()))

    # Define top line based on points
    line = gmsh.model.occ.add_bspline(points_list)

    # temp = gmsh.model.occ.extrude([(1,line)], 0.0, 0.0, -thickness)
    surface = (1, line)

    # Revolve line to create revolution surface
    volume_list = []

    temp2 = gmsh.model.occ.revolve(
        [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
    )
    volume_list.append(temp2[1])
    for inc in range(7):
        gmsh.model.occ.rotate([surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, (1 / 4) * np.pi)

        temp3 = gmsh.model.occ.revolve(
            [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
        )
        volume_list.append(temp3[1])

    if lip:

        start_point = np.array([0.0, 0.0, parabola(diameter * 0.5)])
        cylinder1 = gmsh.model.occ.addDisk(
            *start_point.ravel(), diameter / 2, diameter / 2
        )
        cylinder2 = gmsh.model.occ.addDisk(
            *start_point.ravel(), diameter / 2 + lip_width, diameter / 2 + lip_width
        )
        final = gmsh.model.occ.cut([(2, cylinder2)], [(2, cylinder1)])[0]
        volume_list.append(final[0])

    full_reflector = gmsh.model.occ.fuse(volume_list[0:-1], [volume_list[-1]])

    gmsh.model.occ.synchronize()
    if "-nopopup" not in sys.argv and ONELAB:
        gmsh.fltk.run()

    gmsh.model.mesh.generate(dim=2)
    gmsh.write(file_name)
    gmsh.finalize()
    mesh = meshio.read(file_name)
    from .geometryfunctions import compute_normals, compute_areas

    mesh = compute_areas(compute_normals(mesh))
    return mesh


def spherical_field(az_range, elev_range, outward_normals=False, field_radius=1.0):
    """
    Create a spherical field of points, with normals, a triangle mesh, and areas calculated for each triangle and adjusted for each field point.

    Parameters
    ----------
    az_range : numpy.ndarray of float
        Azimuthal angle in degrees ``[0, 360]``.
    elev_range : numpy.ndarray of float
        Elevation angle in degrees ``[-90, 90]``.
    outward_normals : bool, optional
        If outward pointing normals are required, set as True
    field_radius : float, optional
        The radius of the field, default is 1.0 m

    Returns
    -------
    mesh : :type:`meshio.Mesh`
        spherical field of points at specified azimuth and elevation angles, with meshed triangles
    """
    # vista_pattern = pv.Sphere(
    #    radius=field_radius,
    #    theta_resolution=az_range.shape[0],
    #    phi_resolution=elev_range.shape[0],
    #    start_theta=az_range[0],
    #    end_theta=az_range[-1],
    #    start_phi=elev_range[0],
    #    end_phi=elev_range[-1],
    # ).extract_surface()
    vista_pattern = pv.grid_from_sph_coords(
        az_range, (90 - elev_range), field_radius
    ).extract_surface()
    if outward_normals:
        vista_pattern.point_data["Normals"] = vista_pattern.points / (
            np.linalg.norm(vista_pattern.points, axis=1).reshape(-1, 1)
        )
    else:
        vista_pattern.point_data["Normals"] = (
            vista_pattern.points
            / (np.linalg.norm(vista_pattern.points, axis=1).reshape(-1, 1))
        ) * -1.0

    from ..utility.mesh_functions import pyvista_to_meshio

    mesh = pyvista_to_meshio(vista_pattern.triangulate())
    from ..geometry.geometryfunctions import compute_areas, theta_phi_r

    mesh = theta_phi_r(compute_areas(mesh))

    return mesh


def linear_parabolic_surface(
    diameter,
    focal_length,
    height,
    mesh_size,
    lip=False,
    lip_width=1e-3,
    file_name="Linear_Parabolic_Surface.stl",
    ONELAB=True,
):
    """
    Create a linear parabolic surface with a specified diameter, focal length and height, and generate a mesh of points on the surface. This function is useful for generating the front surface of a parabolic reflector.

    Parameters
    ----------
    diameter : float
        Diameter of the parabolic surface.
    focal_length : float
        Focal length of the parabolic surface.
    height : float
        Height of the parabolic surface. This creates a rectangular section of the full parabolic surface.
    mesh_size : float
        Desired separation of the mesh points.
    lip : bool, optional
        If True, adds a flat lip to the surface. Default is False.
    lip_width : float, optional
        Width of the surface lip if `lip` is True. Default is 1e-3.
    file_name : str, optional
        Name of the file to save the mesh. Default is "Parabolic_Surface.stl".
    ONELAB : bool, optional
        If True, enables ONELAB for interactive geometry manipulation. Default is False.

    Returns
    -------
    mesh : :type:`meshio.Mesh`
        A mesh object containing the parabolic surface.


    """

    def parabola(x):
        return (1 / (4 * focal_length)) * x**2

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    # gmsh.option.setNumber("Geometry.OCCImportLabels", 1) # import colors from STEP
    # Create a new geometry object
    geo = gmsh.model.add("Parabolic Reflector")
    # Define points
    point_num = 15
    x_pos = np.linspace(0, 0.5 * diameter, point_num)
    z_pos = parabola(x_pos)
    coords = np.array([x_pos.ravel(), np.zeros((point_num)), z_pos.ravel()]).transpose()
    points_list = []
    for inc in range(point_num):
        points_list.append(gmsh.model.occ.add_point(*coords[inc, :].tolist()))

    # Define top line based on points
    line = gmsh.model.occ.add_bspline(points_list)

    # temp = gmsh.model.occ.extrude([(1,line)], 0.0, 0.0, -thickness)
    surface = (1, line)

    # Revolve line to create revolution surface
    volume_list = []

    temp2 = gmsh.model.occ.revolve(
        [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
    )
    volume_list.append(temp2[1])
    for inc in range(7):
        gmsh.model.occ.rotate([surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, (1 / 4) * np.pi)

        temp3 = gmsh.model.occ.revolve(
            [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
        )
        volume_list.append(temp3[1])

    if lip:

        start_point = np.array([0.0, 0.0, parabola(diameter * 0.5)])
        cylinder1 = gmsh.model.occ.addDisk(
            *start_point.ravel(), diameter / 2, diameter / 2
        )
        cylinder2 = gmsh.model.occ.addDisk(
            *start_point.ravel(), diameter / 2 + lip_width, diameter / 2 + lip_width
        )
        final = gmsh.model.occ.cut([(2, cylinder2)], [(2, cylinder1)])[0]
        volume_list.append(final[0])

    full_reflector = gmsh.model.occ.fuse(volume_list[0:-1], [volume_list[-1]])

    gmsh.model.occ.synchronize()
    # define box for intersection
    box1 = gmsh.model.occ.addBox(
        -(diameter / 2 + lip_width * 2),
        -height / 2,
        parabola(diameter / 2) * 1.5,
        diameter + lip_width * 4,
        height,
        -parabola(diameter / 2) * 2.0,
    )

    truncated_reflector = gmsh.model.occ.intersect(full_reflector[0], [(3, box1)])
    gmsh.model.occ.synchronize()
    if "-nopopup" not in sys.argv and ONELAB:
        gmsh.fltk.run()

    gmsh.model.mesh.generate(dim=2)
    gmsh.write(file_name)
    gmsh.finalize()
    mesh = meshio.read(file_name)
    from .geometryfunctions import compute_normals, compute_areas

    mesh = compute_areas(compute_normals(mesh))
    return mesh


def linear_parabolic_reflector(
    diameter,
    focal_length,
    height,
    thickness,
    mesh_size,
    lip=False,
    lip_height=1e-3,
    lip_width=1e-3,
    file_name="Linear_Parabolic_Reflector.stl",
    ONELAB=True,
):
    """
    Create a parabolic reflector with a specified diameter and focal length, and generate a mesh of points on the surface. If only the points on the front surface are required, then :func:`parabolic_surface` can be used instead.
    Alternatively if scattering points on all sides are desired, then the points from the reflector mesh may be used.

    Parameters
    ----------
    diameter : float
        Diameter of the parabolic reflector.
    focal_length : float
        Focal length of the parabolic reflector.
    height : float
        Height of the parabolic reflector. This is the height of the rectangular section which is used to intersect the parabolic reflector and create the final shape.
    thickness : float
        Thickness of the reflector.
    mesh_size : float
        Desired separation of the mesh points.
    lip : bool, optional
        If True, adds a flat lip to the reflector. Default is False.
    lip_height : float, optional
        Height of the reflector lip if `lip` is True. Default is 1e-3.
    lip_width : float, optional
        Width of the reflector lip if `lip` is True. Default is 1e-3.
    file_name : str, optional
        Name of the file to save the mesh. Default is "Parabolic_Reflector.stl".
    ONELAB : bool, optional
        If True, enables ONELAB for interactive geometry manipulation. Default is False.

    Returns
    -------
    mesh : :type:`meshio.Mesh`
        A mesh object containing the parabolic reflector.
    aperture_points : :type:`meshio.Mesh`
        A mesh object containing the points on the front surface of the parabolic reflector.

    """

    def parabola(x):
        return (1 / (4 * focal_length)) * x**2

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    # gmsh.option.setNumber("Geometry.OCCImportLabels", 1) # import colors from STEP
    # Create a new geometry object
    geo = gmsh.model.add("Parabolic Reflector")
    # Define points
    point_num = 15
    x_pos = np.linspace(0, 0.5 * diameter, point_num)
    z_pos = parabola(x_pos)
    coords = np.array([x_pos.ravel(), np.zeros((point_num)), z_pos.ravel()]).transpose()
    points_list = []
    for inc in range(point_num):
        points_list.append(gmsh.model.occ.add_point(*coords[inc, :].tolist()))

    # Define top line based on points
    line = gmsh.model.occ.add_bspline(points_list)

    temp = gmsh.model.occ.extrude([(1, line)], 0.0, 0.0, -thickness)
    surface = temp[1]

    # Revolve line to create revolution surface
    volume_list = []

    temp2 = gmsh.model.occ.revolve(
        [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
    )
    volume_list.append(temp2[1])
    for inc in range(7):
        gmsh.model.occ.rotate([surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, (1 / 4) * np.pi)

        temp3 = gmsh.model.occ.revolve(
            [surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25 * np.pi
        )
        volume_list.append(temp3[1])

    if lip:
        axis1 = np.array([0.0, 0.0, -lip_height])

        start_point = np.array([0.0, 0.0, parabola(diameter * 0.5)])
        cylinder1 = gmsh.model.occ.addCylinder(
            *start_point.ravel(), *axis1, diameter / 2
        )
        cylinder2 = gmsh.model.occ.addCylinder(
            *start_point.ravel(), *axis1, diameter / 2 + lip_width
        )
        final = gmsh.model.occ.cut([(3, cylinder2)], [(3, cylinder1)])[0]
        volume_list.append(final[0])

    full_reflector = gmsh.model.occ.fuse(volume_list[0:-1], [volume_list[-1]])

    gmsh.model.occ.synchronize()

    # define box for intersection
    box1 = gmsh.model.occ.addBox(
        -(diameter / 2 + lip_width * 2),
        -height / 2,
        parabola(diameter / 2) * 1.5,
        diameter + lip_width * 4,
        height,
        -parabola(diameter / 2) * 2.0,
    )

    truncated_reflector = gmsh.model.occ.intersect(full_reflector[0], [(3, box1)])
    gmsh.model.occ.synchronize()
    if "-nopopup" not in sys.argv and ONELAB:
        gmsh.fltk.run()

    gmsh.model.mesh.generate(dim=2)
    gmsh.write(file_name)
    gmsh.finalize()
    mesh = meshio.read(file_name)
    from .geometryfunctions import compute_normals, compute_areas

    mesh = compute_areas(compute_normals(mesh))
    aperture_points = linear_parabolic_surface(
        diameter,
        focal_length,
        height,
        mesh_size,
        lip=lip,
        lip_width=lip_width,
        file_name="Linear_Parabolic_Surface.stl",
        ONELAB=ONELAB,
    )
    return mesh, aperture_points


def gridedReflectorPoints(
    majorsize, minorsize, thickness, grid_resolution, sides="all"
):
    """
    :meta private:
    create a primative of the right size, assuming always orientated
    with normal aligned with zenith, and major axis with x,
    adjust position so the face is centred on (0,0,1)

    Parameters
    ----------
    majorsize : float
        size in the x direction of the reflector
    minorsize : float
        size in the y direction of the reflector
    thickness : float
        thickness of the reflector
    grid_resolution : float
        Desired spacing between the points on the reflector surface, should be half a wavelength at the frequency of interest
    sides : str, optional
        Specifies which sides to mesh. Default is 'all', which creates a mesh of surface points on all sides of the cuboid.
        Other options are 'front', which only creates points for the side aligned with the positive z direction, or 'centres',
        which creates a point for the centre of each side.
    Returns
    -------
    mesh_points : :type:`meshio.Mesh`
        the source points for the reflector surface, with normals

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

    mesh_points = meshio.Mesh(
        points=mesh_vertices,
        cells=[
            (
                "vertex",
                np.array(
                    [
                        [
                            i,
                        ]
                        for i in range(len(mesh_vertices))
                    ]
                ),
            )
        ],
        point_data={"Normals": mesh_normals},
    )
    from ..utility.mesh_functions import pyvista_to_meshio
    from .geometryfunctions import compute_areas

    mesh_points = compute_areas(
        pyvista_to_meshio(pv.from_meshio(mesh_points).delaunay_2d())
    )

    return mesh_points

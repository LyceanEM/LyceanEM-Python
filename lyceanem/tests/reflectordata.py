#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
from importlib.resources import files

import meshio
import numpy as np


import lyceanem.tests.data


def UAV_Demo(mesh_resolution, file_name="DemoUAV.stl"):
    """
    Creates a meshed UAV as a demonstrator for the examples, with the mesh resolution set by the user.
    Parameters
    -----------
    mesh_resolution : float
        The target separation between mesh points in the final mesh.
    file_name : str
        The file name of the exported mesh file.

    Returns
    --------
    mesh : :class:`meshio.Mesh`
        The resultant mesh with areas and normal vectors as point data.

    """
    import gmsh
    import meshio

    uav_path = files(lyceanem.tests.data).joinpath("Demo UAV.step")
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_resolution)
    gmsh.model.add("Demo UAV")

    gmsh.merge(uav_path.as_posix())
    # scale from mm to m
    all_entities = gmsh.model.occ.getEntities()
    gmsh.model.occ.dilate(all_entities, 0, 0, 0, 1 / 1000, 1 / 1000, 1 / 1000)
    gmsh.model.occ.synchronize()
    aperture_surfaces = [(2, 6), (2, 7), (2, 13)]
    gmsh.model.occ.remove(aperture_surfaces)
    gmsh.model.occ.fuse([(3, 1)], [(3, 2)])

    gmsh.model.mesh.generate(dim=2)
    gmsh.write(file_name)
    gmsh.finalize()
    mesh = meshio.read(file_name)
    from ..geometry.geometryfunctions import compute_areas, compute_normals, mesh_rotate

    mesh = compute_normals(compute_areas(mesh))
    rotation_vector1 = np.asarray([0.0, np.deg2rad(90), 0.0])
    rotation_vector2 = np.asarray([np.deg2rad(90), 0.0, 0.0])
    mesh = mesh_rotate(mesh, rotation_vector1)
    mesh = mesh_rotate(mesh, rotation_vector2)
    return mesh


def UAV_Demo_Aperture(mesh_resolution, file_name="DemoAperture.stl"):
    """
    Creates a meshed conformal antenna array as a demonstrator for the examples, with the mesh resolution set by the user.
    Parameters
    -----------
    mesh_resolution : float
        The target separation between mesh points in the final mesh.
    file_name : str
        The file name of the exported mesh file.

    Returns
    --------
    mesh : :class:`meshio.Mesh`
        The resultant mesh with areas and normal vectors as point data.

    """
    import gmsh
    import meshio

    uav_path = files(lyceanem.tests.data).joinpath("Demo UAV.step")
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_resolution)
    gmsh.model.add("UAV Conformal Array")

    gmsh.merge(uav_path.as_posix())
    # scale from mm to m
    all_entities = gmsh.model.occ.getEntities()
    gmsh.model.occ.dilate(all_entities, 0, 0, 0, 1 / 1000, 1 / 1000, 1 / 1000)
    gmsh.model.occ.synchronize()
    surfaces = all_entities[370:455]
    volumes = all_entities[455:]

    # Only want surfaces 6,7,13 for the array face.
    wanted = [6 - 1, 7 - 2, 13 - 3]  # adjust index for removal of items in list.
    for i in wanted:
        surfaces.pop(i)

    # delete all volumes
    gmsh.model.occ.remove(volumes)
    gmsh.model.occ.synchronize()
    # delete all but wanted surfaces
    gmsh.model.occ.remove(surfaces, recursive=True)
    gmsh.model.occ.synchronize()
    remaining_entities = gmsh.model.occ.getEntities()

    gmsh.model.occ.fuse([(2, 6)], [(2, 7), (2, 13)])

    gmsh.model.mesh.generate(dim=2)
    gmsh.write(file_name)
    gmsh.finalize()
    mesh = meshio.read(file_name)
    from ..geometry.geometryfunctions import compute_areas, compute_normals, mesh_rotate

    mesh = compute_normals(compute_areas(mesh))
    rotation_vector1 = np.asarray([0.0, np.deg2rad(90), 0.0])
    rotation_vector2 = np.asarray([np.deg2rad(90), 0.0, 0.0])
    mesh = mesh_rotate(mesh, rotation_vector1)
    mesh = mesh_rotate(mesh, rotation_vector2)
    return mesh

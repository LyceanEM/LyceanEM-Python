import copy

import meshio
import numpy as np

from ..base_types import scattering_t
from ..electromagnetics import empropagation as EM
from ..geometry import geometryfunctions as GF
from ..geometry import targets as TL
from ..raycasting import rayfunctions as RF


def aperture_projection(
    aperture,
    environment=None,
    wavelength=1.0,
    az_range=None,
    elev_range=None,
    farfield_distance=2.0,
):
    """

    Using the aperture provided and any blocking structures, predict the maximum directivity envelope of the aperture.
    This will initially just cover all the triangles of a provided solid, but eventually I will include a filter list.

    Parameters
    ---------
    aperture : :class:`meshio.Mesh`
        triangle mesh of the desired aperture
    environment : :class:`lyceanem.base_classes.structures`
        the :class:`lyceanem.base_classes.structures` class should contain all the environment for scattering, providing the blocking for the rays
    wavelength : float
        the wavelength of interest in metres
    az_range : numpy.ndarray of float
        the azimuth range desired for the farfield pattern in degrees
    elev_range : numpy.ndarray of float
        the elevation range desired for the farfield pattern in degrees

    Returns
    ----------
    directivity_envelope : numpy.ndarray of float
        The predicted maximum directivity envelope for the provided aperture at the wavelength of interest
    pcd : :class:`meshio.Mesh`
        a point cloud colored according to the projected area, normalised to the total projected area of the aperture.
    """
    if az_range is None:
        az_range = np.linspace(-180.0, 180.0, 19)

    if elev_range is None:
        elev_range = np.linspace(-90.0, 90.0, 19)
    if environment is None:
        blocking_triangles = GF.mesh_conversion(aperture)
    else:
        blocking_triangles = GF.mesh_conversion(environment)

    directivity_envelope = np.zeros(
        (elev_range.shape[0], az_range.shape[0]), dtype=np.float32
    )
    triangle_centroids = GF.cell_centroids(aperture)
    aperture = GF.compute_areas(aperture)
    aperture = GF.compute_normals(aperture)
    triangle_cell_index = GF.locate_cell_index(aperture)
    triangle_normals = aperture.cell_data["Normals"][triangle_cell_index]
    # Ensure no clash with triangles in raycaster
    triangle_centroids.points += 1e-6 * triangle_normals
    import pyvista as pv

    # pl = pv.Plotter()
    # pl.add_mesh(pv.from_meshio(aperture), scalars="Area")
    # pl.add_mesh(pv.from_meshio(environment.solids[0]), color="red")
    # pl.add_mesh(pv.from_meshio(triangle_centroids), color="green")
    # pl.show()
    visible_patterns, pcd = RF.visiblespace(
        triangle_centroids,
        triangle_normals,
        blocking_triangles,
        vertex_area=triangle_centroids.point_data["Area"],
        az_range=az_range,
        elev_range=elev_range,
        shell_range=farfield_distance,
    )
    directivity_envelope[:, :] = (4 * np.pi * visible_patterns) / (wavelength**2)
    pcd.point_data["Directivity Envelope"] = (4 * np.pi * pcd.point_data["Area"]) / (
        wavelength**2
    )
    return directivity_envelope, pcd


def calculate_farfield(
    aperture_coords,
    antenna_solid,
    desired_E_axis,
    az_range=None,
    el_range=None,
    scatter_points=None,
    wavelength=1.0,
    farfield_distance=2.0,
    scattering=0,
    elements=False,
    los=True,
    project_vectors=False,
    antenna_axes=np.eye(3),
    alpha=0.0,
    beta=(np.pi * 2) / 1.0,
):
    """
    Based upon the aperture coordinates and solids, predict the farfield for the antenna.

    Parameters
    ---------
    aperture_coords : :type:`meshio.Mesh`
        aperture coordinates, from a single point to a mesh sampling across and aperture or surface
    antenna_solid : :class:`lyceanem.base_classes.structures`
        the class should contain all the environment for scattering, providing the blocking for the rays
    desired_E_axis : numpy.ndarray of floats
        1*3 numpy array of the desired excitation vector
    az_range : numpy.ndarray of floats
        the desired azimuth planes in degrees
    el_range : numpy.ndarray of floats
        the desired elevation planes in degrees
    scatter_points : :type:`meshio.Mesh`
        the environment scattering points, defaults to [None]
    wavelength : float
        wavelength of interest in meters, defaults to [1]
    farfield_distance : float
        the distance to evaluate the antenna pattern, defaults to [2]
    scattering: int
     the number of scatters required, if this is set to 0, then only line of sight propagation is considered, defaults to [0]
    elements : bool
        whether the sources and sinks should be considered as elements of a phased array, or a fixed phase aperture like a horn or reflector
    los : bool
        The line of sight component can be ignored by setting los to [False], defaults to [True]
    project_vectors : bool
        should the excitation vector/vectors be projected to be conformal with the surface of the source coordinates

    Returns
    ---------
    etheta : numpy.ndarray of complex
        The Etheta farfield component
    ephi : numpy.ndarray of complex
        The EPhi farfield component
    """
    if az_range is None:
        az_range = np.linspace(-180.0, 180.0, 19)

    if el_range is None:
        el_range = np.linspace(-90.0, 90.0, 19)
    # create sink points for the model
    from ..geometry.targets import spherical_field

    sink_coords = spherical_field(az_range, el_range, farfield_distance)

    Ex, Ey, Ez = calculate_scattering(
        aperture_coords,
        sink_coords,
        antenna_solid,
        desired_E_axis,
        scatter_points=scatter_points,
        wavelength=wavelength,
        scattering=scattering,
        elements=elements,
        los=los,
        project_vectors=project_vectors,
        antenna_axes=antenna_axes,
        multiE=False,
        alpha=alpha,
        beta=beta,
    )
    # convert to etheta,ephi
    etheta = (
        Ex
        * np.cos(sink_coords.point_data["phi_(Radians)"])
        * np.cos(sink_coords.point_data["theta_(Radians)"])
        + Ey
        * np.sin(sink_coords.point_data["phi_(Radians)"])
        * np.cos(sink_coords.point_data["theta_(Radians)"])
        - Ez * np.sin(sink_coords.point_data["theta_(Radians)"])
    )
    ephi = -Ex * np.sin(sink_coords.point_data["phi_(Radians)"]) + Ey * np.cos(
        sink_coords.point_data["phi_(Radians)"]
    )

    return etheta, ephi


def calculate_scattering_cuda(
    alpha,
    beta,
    wavelength,
    acceleration_structure,
    scatter_depth,
    source_mesh,
    sink_mesh,
    scatter_mesh=None,
    chunks=1,
):

    scatter_source_sink = acceleration_structure.calculate_scattering(
        source_mesh,
        sink_mesh,
        alpha,
        beta,
        wavelength,
        chunk_count=chunks,
        self_to_self=False,
    )
    print("scatter_source_sink")
    total_scatter = scatter_source_sink
    if scatter_depth == 0:
        return scatter_source_sink
    source_scatter = acceleration_structure.calculate_scattering(
        source_mesh,
        scatter_mesh,
        alpha,
        beta,
        wavelength,
        chunk_count=chunks,
        self_to_self=False,
    )
    print("source_scatter done")
    scatter_mesh.point_data["ex"] = np.dot(
        np.ones((source_mesh.points.shape[0])), source_scatter[:, :, 0]
    )
    scatter_mesh.point_data["ey"] = np.dot(
        np.ones((source_mesh.points.shape[0])), source_scatter[:, :, 1]
    )
    scatter_mesh.point_data["ez"] = np.dot(
        np.ones((source_mesh.points.shape[0])), source_scatter[:, :, 2]
    )
    print("scatter_points.shape", scatter_mesh.points.shape)
    print("scatter_points", scatter_mesh.point_data["ex"].shape)
    scatter_sink = acceleration_structure.calculate_scattering(
        scatter_mesh,
        sink_mesh,
        alpha,
        beta,
        wavelength,
        chunk_count=chunks,
        self_to_self=False,
    )
    print("scatter_sink done")
    if scatter_depth == 1:
        print(
            "scatter shapes",
            source_scatter.shape,
            scatter_sink.shape,
            scatter_source_sink.shape,
        )
        total_scatter[:, :, 0] += np.dot(source_scatter[:, :, 0], scatter_sink[:, :, 0])
        total_scatter[:, :, 1] += np.dot(source_scatter[:, :, 1], scatter_sink[:, :, 1])
        total_scatter[:, :, 2] += np.dot(source_scatter[:, :, 2], scatter_sink[:, :, 2])
        return total_scatter
    elif scatter_depth > 1:
        print("not yet implemented")
        return total_scatter


def calculate_scattering(
    aperture_coords,
    sink_coords,
    antenna_solid,
    desired_E_axis,
    scatter_points=None,
    wavelength=1.0,
    scattering=0,
    elements=False,
    los=True,
    project_vectors=False,
    antenna_axes=np.eye(3),
    multiE=False,
    alpha=0.0,
    beta=(np.pi * 2) / 1.0,
    cuda=False,
    acceleration_structure=None,
    chunks=1,
    permittivity=8.8541878176e-12,
    permeability=1.25663706212e-6,
):
    """

    calculating the scattering from the provided source coordinates, to the provided sink coordinates in the environment.
    This can be used to generate point to point scattering parameters or full scattering networks.

    Parameters
    ----------
    aperture_coords : :type:`meshio.Mesh`
        source coordinates
    sink_coords : :type:`meshio.Mesh`
        sink coordinates
    antenna_solid : :class:`lyceanem.base_classes.structures`
        the class should contain all the environment for scattering, providing the blocking for the rays
    desired_E_axis : numpy.ndarray of float
        the desired excitation vector, can be a 1*3 array or a n*3 array if multiple different exciations are desired in one lauch
    scatter_points : :type:`meshio.Mesh`
        the scattering points in the environment. Defaults to [None], in which case scattering points will be generated from the antenna_solid. If no scattering should be considered then set scattering to [0].
    wavelength : float
        the wavelength of interest in metres
    scattering : int
        the number of reflections to be considered, defaults to [0], but up to 2 can be considered. The higher this number to greater to computational effort, and for most situations 1 should be ample.
    elements : bool
        whether the sources and sinks should be considered as elements of a phased array, or a fixed phase aperture like a horn or reflector
    los : bool
        The line of sight component can be ignored by setting los to [False], defaults to [True]
    project_vectors : bool
    cuda : bool
        Choice of Cuda or Numba engine, will use Cuda if True
    acceleration_structure : None
        if no acceleration structure is provided, then the function will default to the tiled raycasting method. To pass acceleration structure construct one prior to calling this function
    chunks : int
        chunks is the number of chunks to split the raycasting into, defaults to 1, if gpu is going to run out of memory an error will be reported, and this number should be increased before retrying.

    Returns
    -------
    Ex : numpy.ndarray of complex
       The x-directed electric field components
    Ey : numpy.ndarray of complex
        The y-directed electric field components
    Ez : numpy.ndarray of complex
        The z-directed electric field components

    """
    from . import acceleration_structures

    if not cuda:
        num_sources = len(np.asarray(aperture_coords.points))
        num_sinks = len(np.asarray(sink_coords.points))

        environment_triangles = GF.mesh_conversion(antenna_solid)

        if not multiE:
            if project_vectors:
                conformal_E_vectors = EM.calculate_conformalVectors(
                    desired_E_axis,
                    np.asarray(aperture_coords.point_data["Normals"]),
                    antenna_axes,
                )
            else:
                # print("hi from here", aperture_coords.cell_data)
                if (
                    desired_E_axis.shape[0]
                    == np.asarray(aperture_coords.point_data["Normals"]).shape[0]
                ):
                    conformal_E_vectors = copy.deepcopy(desired_E_axis)
                else:
                    conformal_E_vectors = np.repeat(
                        desired_E_axis.reshape(1, 3).astype(np.complex64),
                        num_sources,
                        axis=0,
                    )
        else:
            if project_vectors:
                conformal_E_vectors = EM.calculate_conformalVectors(
                    desired_E_axis,
                    np.asarray(aperture_coords.point_data["Normals"]),
                    antenna_axes,
                )
            else:
                if (
                    desired_E_axis.shape[0]
                    == np.asarray(aperture_coords.point_data["Normals"]).shape[0]
                ):
                    conformal_E_vectors = copy.deepcopy(desired_E_axis)
                else:
                    conformal_E_vectors = np.repeat(
                        desired_E_axis.reshape(1, 3).astype(np.complex64),
                        num_sources,
                        axis=0,
                    )
        conformal_E_vectors = conformal_E_vectors * aperture_coords.point_data[
            "Area"
        ].reshape(-1, 1)
        if scattering == 0:
            # only use the aperture point cloud, no scattering required.
            scatter_points = meshio.Mesh(points=np.empty((0, 3)), cells=[])
            unified_model = np.append(
                np.asarray(aperture_coords.points).astype(np.float32),
                np.asarray(sink_coords.points).astype(np.float32),
                axis=0,
            )
            unified_normals = np.append(
                np.asarray(aperture_coords.point_data["Normals"]).astype(np.float32),
                np.asarray(sink_coords.point_data["Normals"]).astype(np.float32),
                axis=0,
            )
            unified_weights = np.ones((unified_model.shape[0], 3), dtype=np.complex64)
            unified_weights[0:num_sources, :] = (
                conformal_E_vectors  # / num_sources  # set total amplitude to 1 for the aperture
            )
            unified_weights[num_sources : num_sources + num_sinks, :] = (
                1  # / num_sinks  # set total amplitude to 1 for the aperture
            )
            point_informationv2 = np.empty((len(unified_model)), dtype=scattering_t)
            # set all sources as magnetic current sources, and permittivity and permeability as free space
            point_informationv2[:]["Electric"] = True
            point_informationv2[:]["permittivity"] = permittivity
            point_informationv2[:]["permeability"] = permeability
            # set position, velocity, normal, and weight of sources
            point_informationv2[0:num_sources]["px"] = np.asarray(
                aperture_coords.points
            ).astype(np.float32)[:, 0]
            point_informationv2[0:num_sources]["py"] = np.asarray(
                aperture_coords.points
            ).astype(np.float32)[:, 1]
            point_informationv2[0:num_sources]["pz"] = np.asarray(
                aperture_coords.points
            ).astype(np.float32)[:, 2]
            point_informationv2[0:num_sources]["nx"] = np.asarray(
                aperture_coords.point_data["Normals"]
            ).astype(np.float32)[:, 0]
            point_informationv2[0:num_sources]["ny"] = np.asarray(
                aperture_coords.point_data["Normals"]
            ).astype(np.float32)[:, 1]
            point_informationv2[0:num_sources]["nz"] = np.asarray(
                aperture_coords.point_data["Normals"]
            ).astype(np.float32)[:, 2]
            # set position and velocity of sinks
            point_informationv2[num_sources : (num_sources + num_sinks)]["px"] = (
                np.asarray(sink_coords.points).astype(np.float32)[:, 0]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["py"] = (
                np.asarray(sink_coords.points).astype(np.float32)[:, 1]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["pz"] = (
                np.asarray(sink_coords.points).astype(np.float32)[:, 2]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["nx"] = (
                np.asarray(sink_coords.point_data["Normals"]).astype(np.float32)[:, 0]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["ny"] = (
                np.asarray(sink_coords.point_data["Normals"]).astype(np.float32)[:, 1]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["nz"] = (
                np.asarray(sink_coords.point_data["Normals"]).astype(np.float32)[:, 2]
            )

            point_informationv2[:]["ex"] = unified_weights[:, 0]
            point_informationv2[:]["ey"] = unified_weights[:, 1]
            point_informationv2[:]["ez"] = unified_weights[:, 2]
            scatter_mask = np.zeros((point_informationv2.shape[0]), dtype=np.int32)
            scatter_mask[0:num_sources] = 0
            scatter_mask[(num_sources + num_sinks) :] = 0

        else:

            if not multiE:
                if project_vectors:
                    conformal_E_vectors = EM.calculate_conformalVectors(
                        desired_E_axis[0, :].reshape(1, 3),
                        np.asarray(aperture_coords.point_data["Normals"]).astype(
                            np.float32
                        ),
                    )
                else:
                    conformal_E_vectors = np.repeat(
                        desired_E_axis[0, :].astype(np.float32).reshape(1, 3),
                        num_sources,
                        axis=0,
                    )
            else:
                if project_vectors:
                    conformal_E_vectors = EM.calculate_conformalVectors(
                        desired_E_axis[0, :].reshape(1, 3),
                        np.asarray(aperture_coords.point_data["Normals"]).astype(
                            np.float32
                        ),
                    )
                else:
                    if desired_E_axis.size == 3:
                        conformal_E_vectors = np.repeat(
                            desired_E_axis[0, :].astype(np.float32).reshape(1, 3),
                            num_sources,
                            axis=0,
                        )
                    else:
                        conformal_E_vectors = desired_E_axis.reshape(num_sources, 3)

            unified_model = np.append(
                np.append(
                    np.asarray(aperture_coords.points).astype(np.float32),
                    np.asarray(sink_coords.points).astype(np.float32),
                    axis=0,
                ),
                np.asarray(scatter_points.points).astype(np.float32),
                axis=0,
            )
            unified_normals = np.append(
                np.append(
                    np.asarray(aperture_coords.point_data["Normals"]).astype(
                        np.float32
                    ),
                    np.asarray(sink_coords.point_data["Normals"]).astype(np.float32),
                    axis=0,
                ),
                np.asarray(scatter_points.point_data["Normals"]).astype(np.float32),
                axis=0,
            )
            unified_weights = np.ones((unified_model.shape[0], 3), dtype=np.complex64)
            unified_weights[0:num_sources, :] = (
                conformal_E_vectors  # / num_sources  # set total amplitude to 1 for the aperture
            )
            unified_weights[num_sources : num_sources + num_sinks, :] = (
                1  # / num_sinks  # set total amplitude to 1 for the aperture
            )
            unified_weights[num_sources + num_sinks :, :] = (
                1  # / len(np.asarray(scatter_points.points))  # set total amplitude to 1 for the aperture
            )
            point_informationv2 = np.empty((len(unified_model)), dtype=scattering_t)
            # set all sources as magnetic current sources, and permittivity and permeability as free space
            point_informationv2[:]["Electric"] = True
            point_informationv2[:]["permittivity"] = permittivity
            point_informationv2[:]["permeability"] = permeability
            # set position, velocity, normal, and weight of sources
            point_informationv2[0:num_sources]["px"] = np.asarray(
                aperture_coords.points
            ).astype(np.float32)[:, 0]
            point_informationv2[0:num_sources]["py"] = np.asarray(
                aperture_coords.points
            ).astype(np.float32)[:, 1]
            point_informationv2[0:num_sources]["pz"] = np.asarray(
                aperture_coords.points
            ).astype(np.float32)[:, 2]
            point_informationv2[0:num_sources]["nx"] = np.asarray(
                aperture_coords.point_data["Normals"]
            ).astype(np.float32)[:, 0]
            point_informationv2[0:num_sources]["ny"] = np.asarray(
                aperture_coords.point_data["Normals"]
            ).astype(np.float32)[:, 1]
            point_informationv2[0:num_sources]["nz"] = np.asarray(
                aperture_coords.point_data["Normals"]
            ).astype(np.float32)[:, 2]
            # point_informationv2[0:num_sources]['ex']=unified_weights[0:num_sources,0]
            # point_informationv2[0:num_sources]['ey']=unified_weights[0:num_sources,1]
            # point_informationv2[0:num_sources]['ez']=unified_weights[0:num_sources,2]
            # set position and velocity of sinks
            point_informationv2[num_sources : (num_sources + num_sinks)]["px"] = (
                np.asarray(sink_coords.points).astype(np.float32)[:, 0]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["py"] = (
                np.asarray(sink_coords.points).astype(np.float32)[:, 1]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["pz"] = (
                np.asarray(sink_coords.points).astype(np.float32)[:, 2]
            )
            # point_informationv2[num_sources:(num_sources+num_sinks)]['vx']=0.0
            # point_informationv2[num_sources:(num_sources+num_sinks)]['vy']=0.0
            # point_informationv2[num_sources:(num_sources+num_sinks)]['vz']=0.0
            point_informationv2[num_sources : (num_sources + num_sinks)]["nx"] = (
                np.asarray(sink_coords.point_data["Normals"]).astype(np.float32)[:, 0]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["ny"] = (
                np.asarray(sink_coords.point_data["Normals"]).astype(np.float32)[:, 1]
            )
            point_informationv2[num_sources : (num_sources + num_sinks)]["nz"] = (
                np.asarray(sink_coords.point_data["Normals"]).astype(np.float32)[:, 2]
            )
            point_informationv2[(num_sources + num_sinks) :]["px"] = np.asarray(
                scatter_points.points
            ).astype(np.float32)[:, 0]
            point_informationv2[(num_sources + num_sinks) :]["py"] = np.asarray(
                scatter_points.points
            ).astype(np.float32)[:, 1]
            point_informationv2[(num_sources + num_sinks) :]["pz"] = np.asarray(
                scatter_points.points
            ).astype(np.float32)[:, 2]
            point_informationv2[(num_sources + num_sinks) :]["nx"] = np.asarray(
                scatter_points.point_data["Normals"]
            ).astype(np.float32)[:, 0]
            point_informationv2[(num_sources + num_sinks) :]["ny"] = np.asarray(
                scatter_points.point_data["Normals"]
            ).astype(np.float32)[:, 1]
            point_informationv2[(num_sources + num_sinks) :]["nz"] = np.asarray(
                scatter_points.point_data["Normals"]
            ).astype(np.float32)[:, 2]
            point_informationv2[:]["ex"] = unified_weights[:, 0]
            point_informationv2[:]["ey"] = unified_weights[:, 1]
            point_informationv2[:]["ez"] = unified_weights[:, 2]
            # scatter_mask = np.zeros((point_informationv2.shape[0]), dtype=np.int32)
            # scatter_mask[0:num_sources] = 0
            # scatter_mask[(num_sources + num_sinks):] = scattering

        # full_index, initial_index = RF.integratedraycastersetup(num_sources,
        #                                                        num_sinks,
        #                                                        point_informationv2,
        #                                                        RF.convertTriangles(antenna_solid),
        #                                                        scatter_mask)

        full_index, rays = RF.workchunkingv2(
            np.asarray(aperture_coords.points).astype(np.float32),
            np.asarray(sink_coords.points).astype(np.float32),
            np.asarray(scatter_points.points).astype(np.float32),
            environment_triangles,
            scattering + 1,
            line_of_sight=los,
        )

        if not elements:
            # create efiles for model
            if multiE:
                Ex = np.zeros((desired_E_axis.shape[0], num_sinks), dtype=np.complex64)
                Ey = np.zeros((desired_E_axis.shape[0], num_sinks), dtype=np.complex64)
                Ez = np.zeros((desired_E_axis.shape[0], num_sinks), dtype=np.complex64)
                for e_inc in range(desired_E_axis.shape[0]):
                    conformal_E_vectors = EM.calculate_conformalVectors(
                        desired_E_axis[e_inc, :],
                        np.asarray(aperture_coords.point_data["Normals"]).astype(
                            np.float32
                        ),
                    )
                    unified_weights[0:num_sources, :] = (
                        conformal_E_vectors  # / num_sources
                    )
                    point_informationv2[:]["ex"] = unified_weights[:, 0]
                    point_informationv2[:]["ey"] = unified_weights[:, 1]
                    point_informationv2[:]["ez"] = unified_weights[:, 2]
                    scatter_map = EM.EMGPUFreqDomain(
                        num_sources,
                        num_sinks,
                        full_index,
                        point_informationv2,
                        wavelength,
                        alpha,
                        beta,
                    )
                    Ex[e_inc] = np.dot(np.ones((num_sources)), scatter_map[:, :, 0])
                    Ey[e_inc] = np.dot(np.ones((num_sources)), scatter_map[:, :, 1])
                    Ez[e_inc] = np.dot(np.ones((num_sources)), scatter_map[:, :, 2])
            else:
                scatter_map = EM.EMGPUFreqDomain(
                    num_sources,
                    num_sinks,
                    full_index,
                    point_informationv2,
                    wavelength,
                    alpha,
                    beta,
                )

                Ex = np.dot(np.ones((num_sources)), scatter_map[:, :, 0])
                Ey = np.dot(np.ones((num_sources)), scatter_map[:, :, 1])
                Ez = np.dot(np.ones((num_sources)), scatter_map[:, :, 2])

            # convert to etheta,ephi

        else:
            # create efiles for model
            if multiE:
                Ex = np.zeros((desired_E_axis.shape[1], num_sinks), dtype=np.complex64)
                Ey = np.zeros((desired_E_axis.shape[1], num_sinks), dtype=np.complex64)
                Ez = np.zeros((desired_E_axis.shape[1], num_sinks), dtype=np.complex64)
                for e_inc in range(desired_E_axis.shape[1]):
                    conformal_E_vectors = EM.calculate_conformalVectors(
                        desired_E_axis[e_inc, :],
                        np.asarray(aperture_coords.point_data["Normals"]).astype(
                            np.float32
                        ),
                    )
                    for element in range(num_sources):
                        point_informationv2[0:num_sources]["ex"] = 0.0
                        point_informationv2[0:num_sources]["ey"] = 0.0
                        point_informationv2[0:num_sources]["ez"] = 0.0
                        point_informationv2[element]["ex"] = np.ascontiguousarray(
                            conformal_E_vectors[element, 0]
                        )  # / num_sources
                        point_informationv2[element]["ey"] = conformal_E_vectors[
                            element, 1
                        ]  # / num_sources
                        point_informationv2[element]["ez"] = conformal_E_vectors[
                            element, 2
                        ]  # / num_sources
                        unified_weights[0:num_sources, :] = 0.0
                        unified_weights[element, :] = conformal_E_vectors[
                            element, :
                        ]  # / num_sources
                        scatter_map = EM.EMGPUFreqDomain(
                            num_sources,
                            num_sinks,
                            full_index,
                            point_informationv2,
                            wavelength,
                            alpha,
                            beta,
                        )
                        Ex[element, :, e_inc] = np.dot(
                            np.ones((num_sources)), scatter_map[:, :, 0]
                        )
                        Ey[element, :, e_inc] = np.dot(
                            np.ones((num_sources)), scatter_map[:, :, 1]
                        )
                        Ez[element, :, e_inc] = np.dot(
                            np.ones((num_sources)), scatter_map[:, :, 2]
                        )
            else:
                Ex = np.zeros((num_sources, num_sinks), dtype=np.complex64)
                Ey = np.zeros((num_sources, num_sinks), dtype=np.complex64)
                Ez = np.zeros((num_sources, num_sinks), dtype=np.complex64)
                scatter_map = EM.EMGPUFreqDomain(
                    num_sources,
                    num_sinks,
                    full_index,
                    point_informationv2,
                    wavelength,
                    alpha,
                    beta,
                )
                Ex = scatter_map[:, :, 0]
                Ey = scatter_map[:, :, 1]
                Ez = scatter_map[:, :, 2]

        return Ex, Ey, Ez
    else:
        num_sources = len(np.asarray(aperture_coords.points))
        num_sinks = len(np.asarray(sink_coords.points))
        num_scatters = 0
        if acceleration_structure is None:
            environment_mesh = GF.mesh_conversion_to_meshio(antenna_solid)
            tile_acceleration_structure = (
                acceleration_structures.Tile_acceleration_structure(environment_mesh, 1)
            )
        else:
            tile_acceleration_structure = acceleration_structure

        if not multiE:
            if project_vectors:
                conformal_E_vectors = EM.calculate_conformalVectors(
                    desired_E_axis,
                    np.asarray(aperture_coords.point_data["Normals"]),
                    antenna_axes,
                )
            else:
                if (
                    desired_E_axis.shape[0]
                    == np.asarray(aperture_coords.point_data["Normals"]).shape[0]
                ):
                    conformal_E_vectors = copy.deepcopy(desired_E_axis)
                else:
                    conformal_E_vectors = np.repeat(
                        desired_E_axis.reshape(1, 3).astype(np.complex64),
                        num_sources,
                        axis=0,
                    )
        else:
            if project_vectors:
                conformal_E_vectors = EM.calculate_conformalVectors(
                    desired_E_axis,
                    np.asarray(aperture_coords.point_data["Normals"]),
                    antenna_axes,
                )
            else:
                if (
                    desired_E_axis.shape[0]
                    == np.asarray(aperture_coords.point_data["Normals"]).shape[0]
                ):
                    conformal_E_vectors = copy.deepcopy(desired_E_axis)
                else:
                    conformal_E_vectors = np.repeat(
                        desired_E_axis.reshape(1, 3).astype(np.complex64),
                        num_sources,
                        axis=0,
                    )

            # only use the aperture point cloud, no scattering required.
        conformal_E_vectors = conformal_E_vectors * aperture_coords.point_data[
            "Area"
        ].reshape(-1, 1)
        # set all sources as magnetic current sources, and permittivity and permeability as free space
        aperture_coords.point_data["is_electric"] = np.ones(
            (num_sources), dtype=np.bool
        )
        aperture_coords.point_data["permittivity"] = (
            np.ones((num_sources), dtype=np.complex64) * permittivity
        )
        aperture_coords.point_data["permeability"] = (
            np.ones((num_sources), dtype=np.complex64) * permeability
        )
        # set e fields
        aperture_coords.point_data["ex"] = np.ascontiguousarray(
            conformal_E_vectors[:, 0]
        )
        aperture_coords.point_data["ey"] = np.ascontiguousarray(
            conformal_E_vectors[:, 1]
        )
        aperture_coords.point_data["ez"] = np.ascontiguousarray(
            conformal_E_vectors[:, 2]
        )
        if scattering > 0:
            num_scatters = len(np.asarray(scatter_points.points))
            scatter_points.point_data["is_electric"] = np.ones(
                (num_scatters), dtype=np.bool
            )
            scatter_points.point_data["permittivity"] = (
                np.ones((num_scatters), dtype=np.complex64) * permittivity
            )
            scatter_points.point_data["permeability"] = (
                np.ones((num_scatters), dtype=np.complex64) * permeability
            )

        if not elements:
            # create efiles for model
            if multiE:
                Ex = np.zeros((desired_E_axis.shape[0], num_sinks), dtype=np.complex64)
                Ey = np.zeros((desired_E_axis.shape[0], num_sinks), dtype=np.complex64)
                Ez = np.zeros((desired_E_axis.shape[0], num_sinks), dtype=np.complex64)
                for e_inc in range(desired_E_axis.shape[0]):
                    conformal_E_vectors = EM.calculate_conformalVectors(
                        desired_E_axis[e_inc, :],
                        np.asarray(aperture_coords.point_data["Normals"]).astype(
                            np.float32
                        ),
                    )
                    aperture_coords.point_data["ex"] = conformal_E_vectors[:, 0]
                    aperture_coords.point_data["ey"] = conformal_E_vectors[:, 1]
                    aperture_coords.point_data["ez"] = conformal_E_vectors[:, 2]
                    scatter_map = calculate_scattering_cuda(
                        alpha,
                        beta,
                        wavelength=wavelength,
                        acceleration_structure=tile_acceleration_structure,
                        scatter_depth=scattering,
                        source_mesh=aperture_coords,
                        sink_mesh=sink_coords,
                        scatter_mesh=scatter_points,
                        chunks=chunks,
                    )
                    Ex[e_inc] = np.dot(np.ones((num_sources)), scatter_map[:, :, 0])
                    Ey[e_inc] = np.dot(np.ones((num_sources)), scatter_map[:, :, 1])
                    Ez[e_inc] = np.dot(np.ones((num_sources)), scatter_map[:, :, 2])
            else:
                scatter_map = calculate_scattering_cuda(
                    alpha,
                    beta,
                    wavelength=wavelength,
                    acceleration_structure=tile_acceleration_structure,
                    scatter_depth=scattering,
                    source_mesh=aperture_coords,
                    sink_mesh=sink_coords,
                    scatter_mesh=scatter_points,
                    chunks=chunks,
                )

                Ex = np.dot(np.ones((num_sources)), scatter_map[:, :, 0])
                Ey = np.dot(np.ones((num_sources)), scatter_map[:, :, 1])
                Ez = np.dot(np.ones((num_sources)), scatter_map[:, :, 2])

            # convert to etheta,ephi

        else:
            # create efiles for model
            if multiE:
                Ex = np.zeros((desired_E_axis.shape[1], num_sinks), dtype=np.complex64)
                Ey = np.zeros((desired_E_axis.shape[1], num_sinks), dtype=np.complex64)
                Ez = np.zeros((desired_E_axis.shape[1], num_sinks), dtype=np.complex64)
                for e_inc in range(desired_E_axis.shape[1]):
                    conformal_E_vectors = EM.calculate_conformalVectors(
                        desired_E_axis[e_inc, :],
                        np.asarray(aperture_coords.point_data["Normals"]).astype(
                            np.float32
                        ),
                    )
                    for element in range(num_sources):
                        aperture_coords.point_data["ex"] = np.zeros(
                            (num_sources), dtype=np.complex64
                        )
                        aperture_coords.point_data["ey"] = np.zeros(
                            (num_sources), dtype=np.complex64
                        )
                        aperture_coords.point_data["ez"] = np.zeros(
                            (num_sources), dtype=np.complex64
                        )
                        aperture_coords.point_data["ex"][element] = conformal_E_vectors[
                            element, 0
                        ]
                        aperture_coords.point_data["ey"][element] = conformal_E_vectors[
                            element, 1
                        ]
                        aperture_coords.point_data["ez"][element] = conformal_E_vectors[
                            element, 2
                        ]

                        scatter_map = calculate_scattering_cuda(
                            alpha,
                            beta,
                            wavelength=wavelength,
                            acceleration_structure=tile_acceleration_structure,
                            scatter_depth=scattering,
                            source_mesh=aperture_coords,
                            sink_mesh=sink_coords,
                            scatter_mesh=scatter_points,
                            chunks=chunks,
                        )
                        Ex[element, :, e_inc] = np.dot(
                            np.ones((num_sources)), scatter_map[:, :, 0]
                        )
                        Ey[element, :, e_inc] = np.dot(
                            np.ones((num_sources)), scatter_map[:, :, 1]
                        )
                        Ez[element, :, e_inc] = np.dot(
                            np.ones((num_sources)), scatter_map[:, :, 2]
                        )
            else:
                Ex = np.zeros((num_sources, num_sinks), dtype=np.complex64)
                Ey = np.zeros((num_sources, num_sinks), dtype=np.complex64)
                Ez = np.zeros((num_sources, num_sinks), dtype=np.complex64)
                scatter_map = calculate_scattering_cuda(
                    alpha,
                    beta,
                    wavelength=wavelength,
                    acceleration_structure=tile_acceleration_structure,
                    scatter_depth=scattering,
                    source_mesh=aperture_coords,
                    sink_mesh=sink_coords,
                    scatter_mesh=scatter_points,
                    chunks=chunks,
                )

                Ex = scatter_map[:, :, 0]
                Ey = scatter_map[:, :, 1]
                Ez = scatter_map[:, :, 2]

        return Ex, Ey, Ez

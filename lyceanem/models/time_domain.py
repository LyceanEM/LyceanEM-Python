import numpy as np
import open3d as o3d
import scipy.constants
import copy

from ..base_types import scattering_t
from ..electromagnetics import empropagation as EM
from ..geometry import targets as TL
from ..raycasting import rayfunctions as RF
from ..geometry import geometryfunctions as GF

def calculate_scattering(
    aperture_coords,
    sink_coords,
    excitation_function,
    antenna_solid,
    desired_E_axis,
    scatter_points=None,
    wavelength=1.0,
    scattering=0,
    elements=False,
    sampling_freq=1e9,
    los=False,
    num_samples=10000,
    mesh_resolution=0.5,
    antenna_axes=np.eye(3),
    project_vectors=True
):
    """
    Based upon the parameters given, calculate the time domain scattering for the apertures and sinks.
    The defaults for sampling time are based upon a sampling rate of 1GHz, and a sample time of 1ms.

    Parameters
    ----------
    aperture_coords : :class:`open3d.geometry.TriangleMesh`
        source coordinates
    sink_coords : :class:`open3d.geometry.TriangleMesh`
        sink coordinates
    antenna_solid : :class:`lyceanem.base_classes.structures`
        the class should contain all the environment for scattering, providing the blocking for the rays
    desired_E_axis : 1D numpy array of floats
        the desired excitation vector, can be a 1*3 array or a n*3 array if multiple different exciations are desired in one lauch
    scatter_points : :class:`open3d.geometry.TriangleMesh`
        the scattering points in the environment. Defaults to [None], in which case scattering points will be generated from the antenna_solid. If no scattering should be considered then set scattering to [0].
    wavelength : float
        the wavelength of interest in metres
    scattering : int
        the number of reflections to be considered, defaults to [0], but up to 2 can be considered. The higher this number to greater to computational effort, and for most situations 1 should be ample.
    elements : boolean
        whether the sources and sinks should be considered as elements of a phased array, or a fixed phase aperture like a horn or reflector
    sampling_freq : float
        the desired sampling frequency, should be twice the highest frequency of interest to avoid undersampling artifacts
    num_samples : int
        the length of the desired sampling, can be calculated from the desired model time
    mesh_resolution : float
        the desired mesh resolution in terms of wavelengths if scattering points are not provided. A scattering mesh is generated on the surfaces of all provided trianglemesh structures.

    Return
    -------
    Ex : numpy array of float
        the x directed voltage at the sink coordinates in the time domain, if elements=True, then it will be an array of num_sinks * num_samples, otherwise it will be a 1D array
    Ey : numpy array of float
        the y directed voltage at the sink coordinates in the time domain, if elements=True, then it will be an array of num_sinks * num_samples, otherwise it will be a 1D array
    Ez : numpy array of float
        the z directed voltage at the sink coordinates in the time domain, if elements=True, then it will be an array of num_sinks * num_samples, otherwise it will be a 1D array
    Waketimes : numpy array of float
        the shortest time required for a ray to reach any sink from any source, as long as elements=False. If elements=True, then this is not implemented in the same way, and will return the shortest time required for the final source.
    """

    time_index = np.linspace(0, num_samples / sampling_freq, num_samples)
    multiE = False

    num_sources = len(np.asarray(aperture_coords.points))
    num_sinks = len(np.asarray(sink_coords.points))
    environment_triangles=GF.mesh_conversion(antenna_solid)

    if not multiE:
        if project_vectors:
            conformal_E_vectors = EM.calculate_conformalVectors(
                desired_E_axis, np.asarray(aperture_coords.normals), antenna_axes
            )
        else:
            if desired_E_axis.shape[0] == np.asarray(aperture_coords.normals).shape[0]:
                conformal_E_vectors = copy.deepcopy(desired_E_axis)
            else:
                conformal_E_vectors = np.repeat(
                    desired_E_axis.reshape(1, 3).astype(np.complex64), num_sources, axis=0
                )
    else:
        if project_vectors:
            conformal_E_vectors = EM.calculate_conformalVectors(
                desired_E_axis, np.asarray(aperture_coords.normals), antenna_axes
            )
        else:
            if desired_E_axis.size == 3:
                conformal_E_vectors = np.repeat(
                    desired_E_axis[0, :].astype(np.float32), num_sources, axis=0
                ).reshape(num_sources, 3)
            else:
                conformal_E_vectors = desired_E_axis.reshape(num_sources, 3)

    if scattering == 0:
        # only use the aperture point cloud, no scattering required.
        scatter_points = o3d.geometry.PointCloud()

        unified_model = np.append(
            np.asarray(aperture_coords.points).astype(np.float32),
            np.asarray(sink_coords.points).astype(np.float32),
            axis=0,
        )
        unified_normals = np.append(
            np.asarray(aperture_coords.normals).astype(np.float32),
            np.asarray(sink_coords.normals).astype(np.float32),
            axis=0,
        )
        unified_weights = np.ones((unified_model.shape[0], 3), dtype=np.complex64)
        unified_weights[0:num_sources, :] = (
            conformal_E_vectors #/ num_sources
        )  # set total amplitude to 1 for the aperture
        unified_weights[num_sources : num_sources + num_sinks, :] = (
            1 #/ num_sinks
        )  # set total amplitude to 1 for the aperture
        point_informationv2 = np.empty((len(unified_model)), dtype=scattering_t)
        # set all sources as magnetic current sources, and permittivity and permeability as free space
        point_informationv2[:]["Electric"] = True
        point_informationv2[:]["permittivity"] = 8.8541878176e-12
        point_informationv2[:]["permeability"] = 1.25663706212e-6
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
            aperture_coords.normals
        ).astype(np.float32)[:, 0]
        point_informationv2[0:num_sources]["ny"] = np.asarray(
            aperture_coords.normals
        ).astype(np.float32)[:, 1]
        point_informationv2[0:num_sources]["nz"] = np.asarray(
            aperture_coords.normals
        ).astype(np.float32)[:, 2]
        # set position and velocity of sinks
        point_informationv2[num_sources : (num_sources + num_sinks)]["px"] = np.asarray(
            sink_coords.points
        ).astype(np.float32)[:, 0]
        point_informationv2[num_sources : (num_sources + num_sinks)]["py"] = np.asarray(
            sink_coords.points
        ).astype(np.float32)[:, 1]
        point_informationv2[num_sources : (num_sources + num_sinks)]["pz"] = np.asarray(
            sink_coords.points
        ).astype(np.float32)[:, 2]
        point_informationv2[num_sources : (num_sources + num_sinks)]["nx"] = np.asarray(
            sink_coords.normals
        ).astype(np.float32)[:, 0]
        point_informationv2[num_sources : (num_sources + num_sinks)]["ny"] = np.asarray(
            sink_coords.normals
        ).astype(np.float32)[:, 1]
        point_informationv2[num_sources : (num_sources + num_sinks)]["nz"] = np.asarray(
            sink_coords.normals
        ).astype(np.float32)[:, 2]

        point_informationv2[:]["ex"] = unified_weights[:, 0]
        point_informationv2[:]["ey"] = unified_weights[:, 1]
        point_informationv2[:]["ez"] = unified_weights[:, 2]
        scatter_mask = np.zeros((point_informationv2.shape[0]), dtype=np.int32)
        scatter_mask[0:num_sources] = 0
        scatter_mask[(num_sources + num_sinks) :] = 0

    else:
        # create scatter points on antenna solids based upon a half wavelength square
        if scatter_points is None:
            scatter_points, areas = TL.source_cloud_from_shape(
                antenna_solid, 1e-6, (wavelength * mesh_resolution) ** 2
            )

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
                np.asarray(aperture_coords.normals).astype(np.float32),
                np.asarray(sink_coords.normals).astype(np.float32),
                axis=0,
            ),
            np.asarray(scatter_points.normals).astype(np.float32),
            axis=0,
        )
        unified_weights = np.ones((unified_model.shape[0], 3), dtype=np.complex64)
        unified_weights[0:num_sources, :] = (
            conformal_E_vectors #/ num_sources
        )  # set total amplitude to 1 for the aperture
        unified_weights[num_sources : num_sources + num_sinks, :] = (
            1 #/ num_sinks
        )  # set total amplitude to 1 for the aperture
        unified_weights[num_sources + num_sinks :, :] = 1 #/ len(
          #  np.asarray(scatter_points.points)
        #)  # set total amplitude to 1 for the aperture
        point_informationv2 = np.empty((len(unified_model)), dtype=scattering_t)
        # set all sources as magnetic current sources, and permittivity and permeability as free space
        point_informationv2[:]["Electric"] = True
        point_informationv2[:]["permittivity"] = 8.8541878176e-12
        point_informationv2[:]["permeability"] = 1.25663706212e-6
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
            aperture_coords.normals
        ).astype(np.float32)[:, 0]
        point_informationv2[0:num_sources]["ny"] = np.asarray(
            aperture_coords.normals
        ).astype(np.float32)[:, 1]
        point_informationv2[0:num_sources]["nz"] = np.asarray(
            aperture_coords.normals
        ).astype(np.float32)[:, 2]
        # point_informationv2[0:num_sources]['ex']=unified_weights[0:num_sources,0]
        # point_informationv2[0:num_sources]['ey']=unified_weights[0:num_sources,1]
        # point_informationv2[0:num_sources]['ez']=unified_weights[0:num_sources,2]
        # set position and velocity of sinks
        point_informationv2[num_sources : (num_sources + num_sinks)]["px"] = np.asarray(
            sink_coords.points
        ).astype(np.float32)[:, 0]
        point_informationv2[num_sources : (num_sources + num_sinks)]["py"] = np.asarray(
            sink_coords.points
        ).astype(np.float32)[:, 1]
        point_informationv2[num_sources : (num_sources + num_sinks)]["pz"] = np.asarray(
            sink_coords.points
        ).astype(np.float32)[:, 2]
        # point_informationv2[num_sources:(num_sources+num_sinks)]['vx']=0.0
        # point_informationv2[num_sources:(num_sources+num_sinks)]['vy']=0.0
        # point_informationv2[num_sources:(num_sources+num_sinks)]['vz']=0.0
        point_informationv2[num_sources : (num_sources + num_sinks)]["nx"] = np.asarray(
            sink_coords.normals
        ).astype(np.float32)[:, 0]
        point_informationv2[num_sources : (num_sources + num_sinks)]["ny"] = np.asarray(
            sink_coords.normals
        ).astype(np.float32)[:, 1]
        point_informationv2[num_sources : (num_sources + num_sinks)]["nz"] = np.asarray(
            sink_coords.normals
        ).astype(np.float32)[:, 2]
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
            scatter_points.normals
        ).astype(np.float32)[:, 0]
        point_informationv2[(num_sources + num_sinks) :]["ny"] = np.asarray(
            scatter_points.normals
        ).astype(np.float32)[:, 1]
        point_informationv2[(num_sources + num_sinks) :]["nz"] = np.asarray(
            scatter_points.normals
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
            Ex = np.zeros((desired_E_axis.shape[1], num_samples), dtype=np.float64)
            Ey = np.zeros((desired_E_axis.shape[1], num_samples), dtype=np.float64)
            Ez = np.zeros((desired_E_axis.shape[1], num_samples), dtype=np.float64)
            for e_inc in range(desired_E_axis.shape[0]):
                conformal_E_vectors = EM.calculate_conformalVectors(
                    desired_E_axis[e_inc, :],
                    np.asarray(aperture_coords.normals).astype(np.float32),
                )
                unified_weights[0:num_sources, :] = conformal_E_vectors #/ num_sources
                point_informationv2[:]["ex"] = unified_weights[:, 0]
                point_informationv2[:]["ey"] = unified_weights[:, 1]
                point_informationv2[:]["ez"] = unified_weights[:, 2]
                scattering_coefficient = 1 / 4 * scipy.constants.pi
                TimeMap, WakeTimes = EM.TimeDomainv3(
                    num_sources,
                    num_sinks,
                    point_informationv2,
                    full_index,
                    scattering_coefficient,
                    wavelength,
                    excitation_function,
                    sampling_freq,
                    num_samples,
                )
                wake_index = np.digitize(WakeTimes, time_index)
                Ex[e_inc, wake_index:] = np.dot(
                    np.ones((num_sources)),
                    np.dot(
                        np.ones((num_sinks)),
                        TimeMap[:, :, : (num_samples - wake_index), 0],
                    ),
                )
                Ey[e_inc, wake_index:] = np.dot(
                    np.ones((num_sources)),
                    np.dot(
                        np.ones((num_sinks)),
                        TimeMap[:, :, : (num_samples - wake_index), 1],
                    ),
                )
                Ez[e_inc, wake_index:] = np.dot(
                    np.ones((num_sources)),
                    np.dot(
                        np.ones((num_sinks)),
                        TimeMap[:, :, : (num_samples - wake_index), 2],
                    ),
                )

        else:
            Ex = np.zeros((num_samples), dtype=np.float64)
            Ey = np.zeros((num_samples), dtype=np.float64)
            Ez = np.zeros((num_samples), dtype=np.float64)
            scattering_coefficient = 1 / 4 * scipy.constants.pi
            TimeMap, WakeTimes = EM.TimeDomainv3(
                num_sources,
                num_sinks,
                point_informationv2,
                full_index,
                scattering_coefficient,
                wavelength,
                excitation_function,
                sampling_freq,
                num_samples,
            )
            wake_index = np.digitize(WakeTimes, time_index)
            Ex[wake_index:] = np.dot(
                np.ones((num_sources)),
                np.dot(
                    np.ones((num_sinks)), TimeMap[:, :, : (num_samples - wake_index), 0]
                ),
            )
            Ey[wake_index:] = np.dot(
                np.ones((num_sources)),
                np.dot(
                    np.ones((num_sinks)), TimeMap[:, :, : (num_samples - wake_index), 1]
                ),
            )
            Ez[wake_index:] = np.dot(
                np.ones((num_sources)),
                np.dot(
                    np.ones((num_sinks)), TimeMap[:, :, : (num_samples - wake_index), 2]
                ),
            )

        # convert to etheta,ephi

    else:
        # create efiles for model
        if multiE:
            Ex = np.zeros(
                (desired_E_axis.shape[1], num_sinks, num_samples), dtype=np.float64
            )
            Ey = np.zeros(
                (desired_E_axis.shape[1], num_sinks, num_samples), dtype=np.float64
            )
            Ez = np.zeros(
                (desired_E_axis.shape[1], num_sinks, num_samples), dtype=np.float64
            )
            for e_inc in range(desired_E_axis.shape[1]):
                conformal_E_vectors = EM.calculate_conformalVectors(
                    desired_E_axis[e_inc, :],
                    np.asarray(aperture_coords.normals).astype(np.float32),antenna_axes
                )
                for element in range(num_sources):
                    point_informationv2[0:num_sources]["ex"] = 0.0
                    point_informationv2[0:num_sources]["ey"] = 0.0
                    point_informationv2[0:num_sources]["ez"] = 0.0
                    point_informationv2[element]["ex"] = (
                        conformal_E_vectors[element, 0] / num_sources
                    )
                    point_informationv2[element]["ey"] = (
                        conformal_E_vectors[element, 1] / num_sources
                    )
                    point_informationv2[element]["ez"] = (
                        conformal_E_vectors[element, 2] / num_sources
                    )
                    unified_weights[0:num_sources, :] = 0.0
                    unified_weights[element, :] = (
                        conformal_E_vectors[element, :] / num_sources
                    )
                    scattering_coefficient = 1 / 4 * scipy.constants.pi
                    TimeMap, WakeTimes = EM.TimeDomainv3(
                        num_sources,
                        num_sinks,
                        point_informationv2,
                        full_index,
                        scattering_coefficient,
                        wavelength,
                        excitation_function,
                        sampling_freq,
                        num_samples,
                    )
                    wake_index = np.digitize(WakeTimes, time_index)
                    Ex[element, wake_index:] = np.dot(
                        np.ones((num_sources)),
                        np.dot(
                            np.ones((num_sinks)),
                            TimeMap[:, :, : (num_samples - wake_index), 0],
                        ),
                    )
                    Ey[element, wake_index:] = np.dot(
                        np.ones((num_sources)),
                        np.dot(
                            np.ones((num_sinks)),
                            TimeMap[:, :, : (num_samples - wake_index), 1],
                        ),
                    )
                    Ez[element, wake_index:] = np.dot(
                        np.ones((num_sources)),
                        np.dot(
                            np.ones((num_sinks)),
                            TimeMap[:, :, : (num_samples - wake_index), 2],
                        ),
                    )

        else:
            Ex = np.zeros((num_sources, num_sinks, num_samples), dtype=np.float64)
            Ey = np.zeros((num_sources, num_sinks, num_samples), dtype=np.float64)
            Ez = np.zeros((num_sources, num_sinks, num_samples), dtype=np.float64)
            for element in range(num_sources):
                point_informationv2[0:num_sources]["ex"] = 0.0
                point_informationv2[0:num_sources]["ey"] = 0.0
                point_informationv2[0:num_sources]["ez"] = 0.0
                point_informationv2[element]["ex"] = (
                    conformal_E_vectors[element, 0] / num_sources
                )
                point_informationv2[element]["ey"] = (
                    conformal_E_vectors[element, 1] / num_sources
                )
                point_informationv2[element]["ez"] = (
                    conformal_E_vectors[element, 2] / num_sources
                )
                unified_weights[0:num_sources, :] = 0.0
                unified_weights[element, :] = (
                    conformal_E_vectors[element, :] / num_sources
                )
                scattering_coefficient = 1 / 4 * scipy.constants.pi
                TimeMap, WakeTimes = EM.TimeDomainv3(
                    num_sources,
                    num_sinks,
                    point_informationv2,
                    full_index,
                    scattering_coefficient,
                    wavelength,
                    excitation_function,
                    sampling_freq,
                    num_samples,
                )
                Ex[element, :, :] = np.dot(
                    np.ones((num_sources)),
                    np.dot(np.ones((num_sinks)), TimeMap[:, :, :, 0]),
                )
                Ey[element, :, :] = np.dot(
                    np.ones((num_sources)),
                    np.dot(np.ones((num_sinks)), TimeMap[:, :, :, 1]),
                )
                Ez[element, :, :] = np.dot(
                    np.ones((num_sources)),
                    np.dot(np.ones((num_sinks)), TimeMap[:, :, :, 2]),
                )

    return Ex, Ey, Ez, WakeTimes

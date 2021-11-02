import numpy as np
import scipy.constants
from tqdm import tqdm
from ..raycasting import rayfunctions as RF
from ..electromagnetics import empropagation as EM
from ..geometry import targets as TL
from ..geometry import geometryfunctions as GF
import open3d as o3d
from ..base import scattering_t

def calculate_scattering(aperture_coords,
                         sink_coords,
                         excitation_function,
                         antenna_solid,
                         desired_E_axis,
                         scatter_points=None,
                         wavelength=1.0,
                         scattering=0,
                         elements=False,
                         sampling_freq=1e9,
                         num_samples=10000,
                         mesh_resolution=0.5):
    """
    Based upon the parameters giving, calculate the time domain scattering for the apertures and sinks
    The defaults for sampling time are based upon a sampling rate of 1GHz, and a sample time of 1ms.

    """
    if desired_E_axis.size > 3:
        # multiple excitations requried
        multiE = True
    else:
        multiE = False

    num_sources = len(np.asarray(aperture_coords.points))
    num_sinks = len(np.asarray(sink_coords.points))

    if scattering == 0:
        # only use the aperture point cloud, no scattering required.
        scatter_points = o3d.geometry.PointCloud()

        if ~multiE:
            conformal_E_vectors = EM.calculate_conformalVectors(desired_E_axis,
                                                                np.asarray(aperture_coords.normals).astype(np.float32))
        else:
            conformal_E_vectors = EM.calculate_conformalVectors(desired_E_axis[0, :],
                                                                np.asarray(aperture_coords.normals).astype(np.float32))

        unified_model = np.append(np.asarray(aperture_coords.points).astype(np.float32),
                                  np.asarray(sink_coords.points).astype(np.float32), axis=0)
        unified_normals = np.append(np.asarray(aperture_coords.normals).astype(np.float32),
                                    np.asarray(sink_coords.normals).astype(np.float32), axis=0)
        unified_weights = np.ones((unified_model.shape[0], 3), dtype=np.complex64)
        unified_weights[0:num_sources,
        :] = conformal_E_vectors / num_sources  # set total amplitude to 1 for the aperture
        unified_weights[num_sources:num_sources + num_sinks,
        :] = 1 / num_sinks  # set total amplitude to 1 for the aperture
        point_informationv2 = np.empty((len(unified_model)), dtype=scattering_t)
        # set all sources as magnetic current sources, and permittivity and permeability as free space
        point_informationv2[:]['Electric'] = True
        point_informationv2[:]['permittivity'] = 8.8541878176e-12
        point_informationv2[:]['permeability'] = 1.25663706212e-6
        # set position, velocity, normal, and weight of sources
        point_informationv2[0:num_sources]['px'] = np.asarray(aperture_coords.points).astype(np.float32)[:, 0]
        point_informationv2[0:num_sources]['py'] = np.asarray(aperture_coords.points).astype(np.float32)[:, 1]
        point_informationv2[0:num_sources]['pz'] = np.asarray(aperture_coords.points).astype(np.float32)[:, 2]
        point_informationv2[0:num_sources]['nx'] = np.asarray(aperture_coords.normals).astype(np.float32)[:, 0]
        point_informationv2[0:num_sources]['ny'] = np.asarray(aperture_coords.normals).astype(np.float32)[:, 1]
        point_informationv2[0:num_sources]['nz'] = np.asarray(aperture_coords.normals).astype(np.float32)[:, 2]
        # set position and velocity of sinks
        point_informationv2[num_sources:(num_sources + num_sinks)]['px'] = np.asarray(sink_coords.points).astype(
            np.float32)[:, 0]
        point_informationv2[num_sources:(num_sources + num_sinks)]['py'] = np.asarray(sink_coords.points).astype(
            np.float32)[:, 1]
        point_informationv2[num_sources:(num_sources + num_sinks)]['pz'] = np.asarray(sink_coords.points).astype(
            np.float32)[:, 2]
        point_informationv2[num_sources:(num_sources + num_sinks)]['nx'] = np.asarray(sink_coords.normals).astype(
            np.float32)[:, 0]
        point_informationv2[num_sources:(num_sources + num_sinks)]['ny'] = np.asarray(sink_coords.normals).astype(
            np.float32)[:, 1]
        point_informationv2[num_sources:(num_sources + num_sinks)]['nz'] = np.asarray(sink_coords.normals).astype(
            np.float32)[:, 2]

        point_informationv2[:]['ex'] = unified_weights[:, 0]
        point_informationv2[:]['ey'] = unified_weights[:, 1]
        point_informationv2[:]['ez'] = unified_weights[:, 2]
        scatter_mask = np.zeros((point_informationv2.shape[0]), dtype=np.int32)
        scatter_mask[0:num_sources] = 0
        scatter_mask[(num_sources + num_sinks):] = 0

    else:
        # create scatter points on antenna solids based upon a half wavelength square
        if scatter_points is None:
            scatter_points, areas = TL.source_cloud_from_shape(antenna_solid, 1e-6, (wavelength * mesh_resolution) ** 2)

        if ~multiE:
            conformal_E_vectors = EM.calculate_conformalVectors(desired_E_axis,
                                                                np.asarray(aperture_coords.normals).astype(np.float32))
        else:
            conformal_E_vectors = EM.calculate_conformalVectors(desired_E_axis[0, :],
                                                                np.asarray(aperture_coords.normals).astype(np.float32))

        unified_model = np.append(np.append(np.asarray(aperture_coords.points).astype(np.float32),
                                            np.asarray(sink_coords.points).astype(np.float32), axis=0),
                                  np.asarray(scatter_points.points).astype(np.float32), axis=0)
        unified_normals = np.append(np.append(np.asarray(aperture_coords.normals).astype(np.float32),
                                              np.asarray(sink_coords.normals).astype(np.float32), axis=0),
                                    np.asarray(scatter_points.normals).astype(np.float32), axis=0)
        unified_weights = np.ones((unified_model.shape[0], 3), dtype=np.complex64)
        unified_weights[0:num_sources,
        :] = conformal_E_vectors / num_sources  # set total amplitude to 1 for the aperture
        unified_weights[num_sources:num_sources + num_sinks,
        :] = 1 / num_sinks  # set total amplitude to 1 for the aperture
        unified_weights[num_sources + num_sinks:, :] = 1 / len(
            np.asarray(scatter_points.points))  # set total amplitude to 1 for the aperture
        point_informationv2 = np.empty((len(unified_model)), dtype=scattering_t)
        # set all sources as magnetic current sources, and permittivity and permeability as free space
        point_informationv2[:]['Electric'] = True
        point_informationv2[:]['permittivity'] = 8.8541878176e-12
        point_informationv2[:]['permeability'] = 1.25663706212e-6
        # set position, velocity, normal, and weight of sources
        point_informationv2[0:num_sources]['px'] = np.asarray(aperture_coords.points).astype(np.float32)[:, 0]
        point_informationv2[0:num_sources]['py'] = np.asarray(aperture_coords.points).astype(np.float32)[:, 1]
        point_informationv2[0:num_sources]['pz'] = np.asarray(aperture_coords.points).astype(np.float32)[:, 2]
        point_informationv2[0:num_sources]['nx'] = np.asarray(aperture_coords.normals).astype(np.float32)[:, 0]
        point_informationv2[0:num_sources]['ny'] = np.asarray(aperture_coords.normals).astype(np.float32)[:, 1]
        point_informationv2[0:num_sources]['nz'] = np.asarray(aperture_coords.normals).astype(np.float32)[:, 2]
        # point_informationv2[0:num_sources]['ex']=unified_weights[0:num_sources,0]
        # point_informationv2[0:num_sources]['ey']=unified_weights[0:num_sources,1]
        # point_informationv2[0:num_sources]['ez']=unified_weights[0:num_sources,2]
        # set position and velocity of sinks
        point_informationv2[num_sources:(num_sources + num_sinks)]['px'] = np.asarray(sink_coords.points).astype(
            np.float32)[:, 0]
        point_informationv2[num_sources:(num_sources + num_sinks)]['py'] = np.asarray(sink_coords.points).astype(
            np.float32)[:, 1]
        point_informationv2[num_sources:(num_sources + num_sinks)]['pz'] = np.asarray(sink_coords.points).astype(
            np.float32)[:, 2]
        # point_informationv2[num_sources:(num_sources+num_sinks)]['vx']=0.0
        # point_informationv2[num_sources:(num_sources+num_sinks)]['vy']=0.0
        # point_informationv2[num_sources:(num_sources+num_sinks)]['vz']=0.0
        point_informationv2[num_sources:(num_sources + num_sinks)]['nx'] = np.asarray(sink_coords.normals).astype(
            np.float32)[:, 0]
        point_informationv2[num_sources:(num_sources + num_sinks)]['ny'] = np.asarray(sink_coords.normals).astype(
            np.float32)[:, 1]
        point_informationv2[num_sources:(num_sources + num_sinks)]['nz'] = np.asarray(sink_coords.normals).astype(
            np.float32)[:, 2]
        point_informationv2[(num_sources + num_sinks):]['px'] = np.asarray(scatter_points.points).astype(np.float32)[:,
                                                                0]
        point_informationv2[(num_sources + num_sinks):]['py'] = np.asarray(scatter_points.points).astype(np.float32)[:,
                                                                1]
        point_informationv2[(num_sources + num_sinks):]['pz'] = np.asarray(scatter_points.points).astype(np.float32)[:,
                                                                2]
        point_informationv2[(num_sources + num_sinks):]['nx'] = np.asarray(scatter_points.normals).astype(np.float32)[:,
                                                                0]
        point_informationv2[(num_sources + num_sinks):]['ny'] = np.asarray(scatter_points.normals).astype(np.float32)[:,
                                                                1]
        point_informationv2[(num_sources + num_sinks):]['nz'] = np.asarray(scatter_points.normals).astype(np.float32)[:,
                                                                2]
        point_informationv2[:]['ex'] = unified_weights[:, 0]
        point_informationv2[:]['ey'] = unified_weights[:, 1]
        point_informationv2[:]['ez'] = unified_weights[:, 2]
        scatter_mask = np.zeros((point_informationv2.shape[0]), dtype=np.int32)
        scatter_mask[0:num_sources] = 0
        scatter_mask[(num_sources + num_sinks):] = scattering

    # full_index, initial_index = RF.integratedraycastersetup(num_sources,
    #                                                        num_sinks,
    #                                                        point_informationv2,
    #                                                        RF.convertTriangles(antenna_solid),
    #                                                        scatter_mask)
    full_index, rays = RF.workchunkingv2(np.asarray(aperture_coords.points).astype(np.float32),
                                         np.asarray(sink_coords.points).astype(np.float32),
                                         np.asarray(scatter_points.points).astype(np.float32),
                                         RF.convertTriangles(antenna_solid), scattering + 1)

    if not elements:
        # create efiles for model
        if multiE:
            Ex = np.zeros((desired_E_axis.shape[1],num_samples), dtype=np.float64)
            Ey = np.zeros((desired_E_axis.shape[1],num_samples), dtype=np.float64)
            Ez = np.zeros((desired_E_axis.shape[1],num_samples), dtype=np.float64)
            for e_inc in tqdm(range(desired_E_axis.shape[1])):
                conformal_E_vectors = EM.calculate_conformalVectors(desired_E_axis[e_inc, :],
                                                                    np.asarray(aperture_coords.normals).astype(
                                                                        np.float32))
                unified_weights[0:num_sources, :] = conformal_E_vectors / num_sources
                point_informationv2[:]['ex'] = unified_weights[:, 0]
                point_informationv2[:]['ey'] = unified_weights[:, 1]
                point_informationv2[:]['ez'] = unified_weights[:, 2]
                TimeMap,WakeTimes=EM.TimeDomainv3()

        else:
            TimeMap,WakeTimes=EM.TimeDomainv3()

        # convert to etheta,ephi

    else:
        # create efiles for model
        if multiE:
            Ex = np.zeros((desired_E_axis.shape[1], num_sinks,num_samples), dtype=np.float64)
            Ey = np.zeros((desired_E_axis.shape[1], num_sinks,num_samples), dtype=np.float64)
            Ez = np.zeros((desired_E_axis.shape[1], num_sinks,num_samples), dtype=np.float64)
            for e_inc in tqdm(range(desired_E_axis.shape[1])):
                conformal_E_vectors = EM.calculate_conformalVectors(desired_E_axis[e_inc, :],
                                                                    np.asarray(aperture_coords.normals).astype(
                                                                        np.float32))
                for element in tqdm(range(num_sources)):
                    point_informationv2[0:num_sources]['ex'] = 0.0
                    point_informationv2[0:num_sources]['ey'] = 0.0
                    point_informationv2[0:num_sources]['ez'] = 0.0
                    point_informationv2[element]['ex'] = conformal_E_vectors[element, 0] / num_sources
                    point_informationv2[element]['ey'] = conformal_E_vectors[element, 1] / num_sources
                    point_informationv2[element]['ez'] = conformal_E_vectors[element, 2] / num_sources
                    unified_weights[0:num_sources, :] = 0.0
                    unified_weights[element, :] = conformal_E_vectors[element, :] / num_sources
                    TimeMap,WakeTimes=EM.TimeDomainv3()
        else:
            Ex = np.zeros((num_sources, num_sinks,num_samples), dtype=np.float64)
            Ey = np.zeros((num_sources, num_sinks,num_samples), dtype=np.float64)
            Ez = np.zeros((num_sources, num_sinks,num_samples), dtype=np.float64)
            for element in tqdm(range(num_sources)):
                point_informationv2[0:num_sources]['ex'] = 0.0
                point_informationv2[0:num_sources]['ey'] = 0.0
                point_informationv2[0:num_sources]['ez'] = 0.0
                point_informationv2[element]['ex'] = conformal_E_vectors[element, 0] / num_sources
                point_informationv2[element]['ey'] = conformal_E_vectors[element, 1] / num_sources
                point_informationv2[element]['ez'] = conformal_E_vectors[element, 2] / num_sources
                unified_weights[0:num_sources, :] = 0.0
                unified_weights[element, :] = conformal_E_vectors[element, :] / num_sources
                scattering_coefficient=(1/4*scipy.constants.pi)
                TimeMap,WakeTimes=EM.TimeDomainv3(num_sources,
                                                  num_sinks,
                                                  point_informationv2,
                                                  full_index,
                                                  scattering_coefficient,
                                                  wavelength,
                                                  excitation_function,
                                                  sampling_freq,
                                                  num_samples)
                Ex[element, :, :] = np.dot(np.ones((num_sources)), np.dot(np.ones((num_sinks)),TimeMap[:, :, :,0]))
                Ey[element, :, :] = np.dot(np.ones((num_sources)), np.dot(np.ones((num_sinks)),TimeMap[:, :, :,1]))
                Ez[element, :, :] = np.dot(np.ones((num_sources)), np.dot(np.ones((num_sinks)),TimeMap[:, :, :,2]))

    return Ex, Ey, Ez
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:15:07 2020

@author: timtitan
"""
import logging
from timeit import default_timer as timer

import numpy as np
import open3d as o3d
from tqdm import tqdm

import lyceanem.geometry.targets as TL
import lyceanem.models.frequency_domain as FD
import lyceanem.raycasting.rayfunctions as RF

# import GPUScatteringTest as GU
import lyceanem.tests.reflectordata as RD

nb_logger = logging.getLogger("numba")
nb_logger.setLevel(logging.ERROR)  # only show error
angle_values = np.linspace(0, 90, 91)
max_scatter = 2
responsex2 = np.zeros((len(angle_values), 3), dtype="complex")
responsey2 = np.zeros((len(angle_values), 3), dtype="complex")
responsez2 = np.zeros((len(angle_values), 3), dtype="complex")
isoresponse = np.zeros((len(angle_values)), dtype="complex")
# responsey3=np.zeros((len(angle_values),3),dtype='complex')
# responsez3=np.zeros((len(angle_values),3),dtype='complex')


freq = np.asarray(6.0e9)
wavelength = 3e8 / freq
mesh_resolution = 0.5 * wavelength
# point_area=(mesh_resolution)**2
# reflectorplate_main,_=TL.meshedReflector(0.3,0.3,6e-3,wavelength*0.5,sides='front')
# scatter_points_main=TL.meshsolid(reflectorplate_main, point_area)

resolution = 5
show_scene = False
# response2=np.zeros((len(angle_values),3),dtype='complex')
for angle_inc in tqdm(range(len(angle_values))):
    overalltime = timer()
    # define antenna surface
    # freq=np.asarray(24.0e9)
    wavelength = 3e8 / freq
    # mesh_resolution=0.5*wavelength
    aperture_resolution = 0.5 * wavelength
    single_source = False
    single_sink = False
    if single_source:
        x = np.full((1, 1), 0.0, dtype=np.float32)
        z = np.full((1, 1), 0.0, dtype=np.float32)
        xx, yy = np.meshgrid(x, z)
        zz = np.zeros((len(np.ravel(xx)), 1))
        transmitting_antenna_surface_coords = RF.points2pointcloud(np.zeros((1, 3)))
        normals = np.zeros((1, 3))
        normals[0, 2] = 1.0
        transmitting_antenna_surface_coords.normals = o3d.utility.Vector3dVector(
            normals
        )
        transmit_horn_structure, _ = TL.meshedHorn(
            58e-3, 58e-3, 128e-3, 2e-3, 0.21, aperture_resolution
        )
    else:
        # horn spec with x_width, y_width, length, edge width, flare angle, and aperture resolution
        transmit_horn_structure, transmitting_antenna_surface_coords = TL.meshedHorn(
            58e-3, 58e-3, 128e-3, 2e-3, 0.21, aperture_resolution
        )

    if single_sink:
        x = np.full((1, 1), 0.0, dtype=np.float32)
        z = np.full((1, 1), 0.0, dtype=np.float32)
        xx, yy = np.meshgrid(x, z)
        zz = np.zeros((len(np.ravel(xx)), 1))
        recieving_antenna_surface_coords = RF.points2pointcloud(np.zeros((1, 3)))
        normals = np.zeros((1, 3))
        normals[0, 2] = 1.0
        recieving_antenna_surface_coords.normals = o3d.utility.Vector3dVector(normals)
        # recieving_antenna_surface_coords=np.zeros(((len(np.ravel(xx)),3)),dtype=np.float32)
        # recieving_antenna_surface_coords[:,0]=np.ravel(xx)
        # recieving_antenna_surface_coords[:,1]=np.ravel(yy)
        # recieving_antenna_surface_coords[:,2]=np.ravel(zz)
        receive_horn_structure, _ = TL.meshedHorn(
            58e-3, 58e-3, 128e-3, 2e-3, 0.21, aperture_resolution
        )
    else:
        receive_horn_structure, recieving_antenna_surface_coords = TL.meshedHorn(
            58e-3, 58e-3, 128e-3, 2e-3, 0.21, aperture_resolution
        )

    # rx90=R.from_euler('x',90,degrees=True)
    # rz90=R.from_euler('z',90,degrees=True)
    # sources=rz90.apply(rx90.apply(transmitting_antenna_surface_coords))+np.array([[2.695,0,0]])
    rotation_vector1 = np.radians(np.asarray([90.0, 0.0, 0.0]))
    rotation_vector2 = np.radians(np.asarray([0.0, 0.0, -90.0]))
    transmit_horn_structure.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        center=False,
    )
    transmit_horn_structure.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2),
        center=False,
    )
    transmit_horn_structure.translate(np.asarray([2.695, 0, 0]), relative=True)
    transmitting_antenna_surface_coords.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        center=False,
    )
    transmitting_antenna_surface_coords.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector2),
        center=False,
    )
    transmitting_antenna_surface_coords.translate(
        np.asarray([2.695, 0, 0]), relative=True
    )
    # position receiver
    receive_horn_structure.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        center=False,
    )
    # rotation_vector3=np.radians(np.asarray([0.0,0.0,-90.0]))
    # receive_horn_structure.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector3),center=False)
    receive_horn_structure.translate(np.asarray([0, 1.427, 0]), relative=True)
    recieving_antenna_surface_coords.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        center=False,
    )
    # recieving_antenna_surface_coords.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector3),center=False)
    recieving_antenna_surface_coords.translate(np.asarray([0, 1.427, 0]), relative=True)
    # sinks=rx90.apply(recieving_antenna_surface_coords)+np.array([[0.0,1.427,0]])
    source_points = transmitting_antenna_surface_coords
    sink_points = recieving_antenna_surface_coords
    receiving_antenna_points = len(np.asarray(recieving_antenna_surface_coords.points))
    transmitting_antenna_points = len(
        np.asarray(transmitting_antenna_surface_coords.points)
    )
    angle_d = angle_values[angle_inc]
    # angle_d=45
    # resolution=4
    # create environment triangle data
    # scatter_points=testtargets()
    # temp_environment=testbox()
    # reflectorplate=rectReflector(0.3,0.3,0.01)
    if resolution == 0:
        # reflectorplate=rectReflector(1.0,1.0,0.1)
        reflectorplate, scatter_points = TL.meshedReflector(
            0.3, 0.15, 6e-3, wavelength * 0.5, sides="front"
        )
        filename = "SmallTallReflector"
        position_vector = np.asarray([29e-3, 0.0, 0])
    elif resolution == 1:
        reflectorplate, scatter_points = TL.meshedReflector(
            0.15, 0.3, 6e-3, wavelength * 0.5, sides="front"
        )
        filename = "SmallWideReflector"
        position_vector = np.asarray([29e-3, 0.0, 0])
        # reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare1Meshed.stl")
    elif resolution == 2:
        # point_area=(mesh_resolution)**2
        # reflectorplate=copy.deepcopy(reflectorplate_main)
        # scatter_points=copy.deepcopy(scatter_points_main)
        reflectorplate, scatter_points = TL.meshedReflector(
            0.3, 0.3, 6e-3, wavelength * 0.5, sides="front"
        )
        # scatter_points=TL.meshsolid(reflectorplate, point_area)
        filename = "MediumSquareReflectorEMtest"
        position_vector = np.asarray([29e-3, 0.0, 0])
        # position_vector=np.asarray([0,0.0,0])
        # reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare0p5Meshed.stl")
    elif resolution == 3:
        reflectorplate, scatter_points = TL.meshedReflector(
            0.3, 0.6, 6e-3, wavelength * 0.5
        )
        filename = "LargeWideReflector"
        position_vector = np.asarray([29e-3, 0.0, 0])
        # reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare0p1Meshed.stl")
    elif resolution == 4:
        reflectorplate, scatter_points = TL.meshedReflector(
            0.6, 0.3, 6e-3, wavelength * 0.5
        )
        filename = "LargeTallReflector"
        position_vector = np.asarray([29e-3, 0.0, 0])
        # reflectorplate=o3d.io.read_triangle_mesh("ReferenceSquare0p1Meshed.stl")
        # scatter_points=o3d.geometry.PointCloud()
        # reflectorplate.compute_vertex_normals()
        # scatter_points.points=o3d.utility.Vector3dVector(np.asarray(reflectorplate.vertices)+np.asarray(reflectorplate.vertex_normals)*EPSILON)
    elif resolution == 5:
        # reflectorplate,scatter_points=RF.parabolic_reflector(0.6,0.147,6e-3,wavelength*5,sides='front')
        # reflectorplate,scatter_points=TL.parabolic_reflector(0.3,0.147,6e-3,wavelength*0.5,sides='front')
        reflectorplate, scatter_points = TL.parabola(0.6, 0.147, 6e-3, mesh_resolution)
        position_vector = np.asarray([0.0, 0.0, 17e-2])
        filename = "ParabaloidReflector"
    elif resolution == 6:
        reflectorplate, scatter_points = RD.mediumreferencescan(mesh_resolution)
        position_vector = np.asarray([0, 0, 29e-3])
        filename = "LaserScannedMediumReference"
    elif resolution == 7:
        reflectorplate, scatter_points = RD.prototypescan(mesh_resolution)
        position_vector = np.asarray([29e-3, 0.0, 0])
        filename = "LaserScannedPrototypeReflector"

    rotation_vector1 = np.radians(np.asarray([0.0, 90.0, 0.0]))
    scatter_cloud = scatter_points.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        center=False,
    )
    reflectorplate.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector1),
        center=False,
    )
    reflectorplate.translate(position_vector, relative=True)
    scatter_points.translate(position_vector, relative=True)

    # positionedreflector1=transformO3dobject(reflectorplate,position_vector,rotation_vector1)
    # scatter_cloud_positioned=transformO3dobject(scatter_points,position_vector,rotation_vector1)

    rotation_vector = np.radians(np.asarray([0.0, 0.0, angle_d]))
    scatter_cloud.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
        center=False,
    )
    reflectorplate.rotate(
        o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector),
        center=False,
    )

    # testobject=coneReflector(2,2)
    reflectorplate.compute_vertex_normals()
    reflectorplate.compute_triangle_normals()
    solids = reflectorplate + transmit_horn_structure + receive_horn_structure
    temp_environment = np.append(
        np.append(
            RF.convertTriangles(reflectorplate),
            RF.convertTriangles(transmit_horn_structure),
            axis=0,
        ),
        RF.convertTriangles(receive_horn_structure),
        axis=0,
    )
    ENV_TRI_NUM = len(temp_environment)
    # sources=np.zeros((1,3),dtype=np.float32)
    # sinks=np.zeros((1,3),dtype=np.float32)

    # scatter_cloud=TL.meshsolid(reflectorplate, point_area)
    scattering_points = np.asarray(scatter_cloud.points)
    # scatter_cloud=RF.points2pointcloud(scattering_points)

    if show_scene:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries(
            [
                source_points,
                sink_points,
                scatter_cloud,
                reflectorplate,
                mesh_frame,
                receive_horn_structure,
                transmit_horn_structure,
            ]
        )
        # o3d.visualization.draw_geometries([source_points,sink_points,scatter_cloud,mesh_frame])
    ray_start = timer()
    # desired_E_axis=np.zeros((3,3),dtype=np.float32)
    # desired_E_axis[0,0]=1.0
    # desired_E_axis[1,1]=1.0
    # desired_E_axis[2,2]=1.0
    desired_E_axis = np.zeros((1, 3), dtype=np.float32)
    desired_E_axis[0, 2] = 1.0
    # responsex2[angle_inc,:],responsey2[angle_inc,:],responsez2[angle_inc,:]=ml.calculate_scattering(transmitting_antenna_surface_coords,
    #                                                                                                recieving_antenna_surface_coords,
    #                                                                                                solids,
    #                                                                                                desired_E_axis,
    #                                                                                                scatter_points=scatter_cloud,
    #                                                                                                wavelength=wavelength,
    #                                                                                                scattering=1)
    Ex, Ey, Ez = FD.calculate_scattering(
        aperture_coords=transmitting_antenna_surface_coords,
        sink_coords=recieving_antenna_surface_coords,
        antenna_solid=solids,
        desired_E_axis=desired_E_axis,
        scatter_points=scatter_cloud,
        wavelength=wavelength,
        scattering=1,
    )
    isoresponse[angle_inc] = FD.calculate_scattering_isotropic(
        aperture_coords=transmitting_antenna_surface_coords,
        sink_coords=recieving_antenna_surface_coords,
        antenna_solid=solids,
        scatter_points=scatter_cloud,
        wavelength=wavelength,
        scattering=1,
    )
    responsex2[angle_inc, :] = Ex
    responsey2[angle_inc, :] = Ey
    responsez2[angle_inc, :] = Ez

    # print((timer2-timer1)/(timer3-timer2))
    # print("{:7.0f} Rays Processed in {:3.1f}s by version 1".format(total_rays,(start-ray_start)+(timer2-timer1)) )
    # print("{:7.0f} Rays Processed in {:3.1f}s by version 2".format(total_rays,(start-ray_start)+(timer3-timer2)) )
    # print("Progress:  {:3.1f}%".format(100*(angle_inc+1)/len(angle_values)) )
    # print("Final Processing Time:  {:3.1f} s, Total time: {:3.1f} s".format(kernel_dt,total_time) )


# normalised_max=np.max(np.array([np.max(20*np.log10(np.abs(responsex3))),np.max(20*np.log10(np.abs(responsey3))),np.max(20*np.log10(np.abs(responsez3)))]))
# np.savez(filename, responsex=responsex2,responsey=responsey2,responsez=responsez2,angle_values=angle_values)
excitaiton = 2

# normalised_max2=np.max(np.array([np.max(10*np.log10(np.abs(responsex3))),np.max(10*np.log10(np.abs(responsey3))),np.max(10*np.log10(np.abs(responsez3)))]))
normalised_max3 = np.max(
    np.array(
        [
            np.max(20 * np.log10(np.abs(responsex2))),
            np.max(20 * np.log10(np.abs(responsey2))),
            np.max(20 * np.log10(np.abs(responsez2))),
        ]
    )
)
testx = 20 * np.log10(np.abs(responsex2[:])) - normalised_max3
testy = 20 * np.log10(np.abs(responsey2[:])) - normalised_max3
testz = 20 * np.log10(np.abs(responsez2[:])) - normalised_max3
# testxtd=20*np.log10(np.abs(np.sum(Ex[:,excitation,:],axis=1)))
# testytd=20*np.log10(np.abs(np.sum(Ey[:,excitation,:],axis=1)))
# testztd=20*np.log10(np.abs(np.sum(Ez[:,excitation,:],axis=1)))
testiso = 20 * np.log10(np.abs(isoresponse))
# testiso2=20*np.log10(np.abs(responseiso2))
# test1=20*np.log10(np.abs(response[:,0]))
# test2=20*np.log10(np.abs(response[:,1]))
# test3=20*np.log10(np.abs(response[:,2]))
# test2=20*np.log10(np.abs(response[:,1]))-np.max(np.max(20*np.log10(np.abs(response[:,1]))))
# test3=20*np.log10(np.abs(response[:,2]))-np.max(np.max(20*np.log10(np.abs(response[:,2]))))
# test4=20*np.log10(np.abs(response[:,3]))-np.max(np.max(20*np.log10(np.abs(response[:,3]))))
# test5=20*np.log10(np.abs(response[:,4]))-np.max(np.max(20*np.log10(np.abs(response[:,4]))))
# test5=20*np.log10(np.abs(response[:,0])-np.max(np.max(np.abs(response[:,4]))))
# test4=20*np.log10(np.abs(response[:,3]))
# test5=20*np.log10(np.abs(response[:,4]))
# test2=20*np.log10(np.abs(response[:,1]))-np.max(np.max(20*np.log10(np.abs(response[:,1]))))
# test5=20*np.log10(np.abs(response[:,4]))-np.max(np.max(20*np.log10(np.abs(response[:,4]))))
# plt.plot(angle_values-45.0,test4,angle_values-45.0,test5)
# Create plots with pre-defined labels.
# fig, ax = plt.subplots()
# ax.plot(angle_values-45, 20*np.log10(np.abs(responsez2[:,1]))-np.max(20*np.log10(np.abs(responsez2[:,1]))),label='Ez')
# ax.plot(angle_values-45, 20*np.log10(np.abs(responsez[:,1]))-np.max(20*np.log10(np.abs(responsez[:,1]))),label='Ez')
# ax.plot(angle_values-45, testx[:,2],label='Ex')
# ax.plot(angle_values-45, testy[:,2],label='Ey')
# ax.plot(angle_values-46, testz[:,2],label='FD')
# ax.plot(angle_values-45, testx,label='Extd')
# ax.plot(angle_values-45, testy,label='Eytd')
# ax.plot(angle_values-45, testz,label='Eztd')
# ax.plot(angle_values-45, testiso-np.max(testiso), linestyle=':', label='No Polarization')
# ax.plot(angle_values-45, testiso2-np.max(testiso2), linestyle=':', label='No Polariztion')
# ax.plot(angle_values-45.0, test3, label='$Medium$')
# ax.plot(angle_values-45.0, test4, label='$Large T$')
# ax.plot(angle_values-45.0, test5, label='$Large W$')
# plt.xlabel('$\\theta_{N}$ (degrees)')
# plt.ylabel('Normalised Level (dB)')
# plt.title('Scattering=fraction')
# ax.set_ylim(-60.0,0)
# ax.set_xlim(np.min(angle_values)-45,np.max(angle_values)-45)
# ax.set_xticks(np.linspace(np.min(angle_values)-45, np.max(angle_values)-45, 19))
# ax.set_xticks(np.linspace(0, 360, 19))
# ax.set_yticks(np.linspace(-60, 0., 21))
# legend = ax.legend(loc='upper right', shadow=True)
# plt.grid()
# plt.show()

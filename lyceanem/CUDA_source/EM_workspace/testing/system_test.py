import numpy as np
import open3d as o3d
import pyvista as pv
import tempfile
import os
import pytest
import meshio
from em import calculate_scattering, bin_counts_to_numpy, bin_triangles_to_numpy

#from em_MVP import em_propagation_03  # Replace 'your_module' with the actual module name



    

def test_raycasttiles():
    """
    End to end test for example 3 in the documentation of old code. Test passes if the result of the new method matches
    the result of the old one for the range of angles specified in the old code.
    This comparison is made to a reference saved .npy file found in the reference_files folder.
    The test will fail if any step is wrong, including
    - the generation of the input data
    - the EM propagation
    - the conversion of the result to a numpy array of the same shape as the reference result
    """

    # Run EM propagation
    # load points
    source = meshio.read("/home/tf17270/lycean_em/LyceanEM-Python/LyceanEM-Python-master/docs/examples/transmitting_antenna17280.ply")
    end = meshio.read("/home/tf17270/lycean_em/LyceanEM-Python/LyceanEM-Python-master/docs/examples/receiving_antenna17280.ply")
    scatter = meshio.read("/home/tf17270/lycean_em/LyceanEM-Python/LyceanEM-Python-master/docs/examples/scatter_points17280.ply")
 
    # stack point data
    source_normal = np.ascontiguousarray(np.column_stack((source.point_data['nx'], source.point_data['ny'], source.point_data['nz'])))
    assert source_normal.shape[0] == source.points.shape[0]
    assert source_normal.shape[1] == 3
    end_normal = np.ascontiguousarray(np.column_stack((end.point_data['nx'], end.point_data['ny'], end.point_data['nz'])))
    assert end_normal.shape[0] == end.points.shape[0]
    assert end_normal.shape[1] == 3
    scatter_normal = np.ascontiguousarray(np.column_stack((scatter.point_data['nx'], scatter.point_data['ny'], scatter.point_data['nz'])))
    assert scatter_normal.shape[1] == 3
    assert scatter_normal.shape[0] == scatter.points.shape[0]


    
    normals = np.ascontiguousarray((np.append(np.append(source_normal,end_normal,axis=0), scatter_normal, axis=0)))
 
    assert normals.shape[0] == source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0]
    assert normals.shape[1] == 3
    print("normals, ",normals[72])
 
    ex_real = np.zeros(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    ex_imag = np.zeros(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    ey_real = np.zeros(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    ey_imag = np.zeros(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    ez_real = np.zeros(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    ez_imag = np.zeros(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    
 
    is_electric = np.ones(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    print(source)
    permittivity_real = np.ones(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    permittivity_imag =  np.zeros(permittivity_real.shape)
    permeability_real = np.ones(source.points.shape[0] + end.points.shape[0] + scatter.points.shape[0])
    permeability_imag = np.zeros(permeability_real.shape)

 
 
    
 
 
    # load blocking meshes
    transmit_horn = meshio.read("/home/tf17270/lycean_em/LyceanEM-Python/LyceanEM-Python-master/docs/examples/transmit_horn17280.ply")
    receive_horn = meshio.read("/home/tf17270/lycean_em/LyceanEM-Python/LyceanEM-Python-master/docs/examples/receive_horn17280.ply")
    reflector = meshio.read("/home/tf17270/lycean_em/LyceanEM-Python/LyceanEM-Python-master/docs/examples/reflector17280.ply")

    triangles = np.ascontiguousarray(np.vstack((reflector.cells[0].data, transmit_horn.cells[0].data, receive_horn.cells[0].data)))
    n = np.ascontiguousarray(np.vstack((reflector.points, transmit_horn.points, receive_horn.points)))

    ## add the number of points in the previous mesh to the triangles array to be able to identify the different meshes
    triangles[reflector.cells[0].data.shape[0]:reflector.cells[0].data.shape[0]+transmit_horn.cells[0].data.shape[0]] += reflector.points.shape[0]
    triangles[reflector.cells[0].data.shape[0]+transmit_horn.cells[0].data.shape[0]:] += transmit_horn.points.shape[0] + reflector.points.shape[0]
    
 
    
 
    freq = np.asarray(15.0e9)
    wavelength = 3e8 / freq    
    wave_vector = 2 * np.pi / wavelength

    max_x = np.max(n[:,0])
    min_x = np.min(n[:,0])
    max_y = np.max(n[:,1])
    min_y = np.min(n[:,1])
    max_z = np.max(n[:,2])
    min_z = np.min(n[:,2])
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    diff_z = max_z - min_z
    diff = min( diff_y, diff_z)
    diff/=1
    ncellsy = int(diff_y/diff) +1
    ncellsz = int(diff_z/diff) +1
    print("ncellsy, ncellsz", ncellsy, ncellsz)
    print("multiplying by 100", ncellsy*ncellsz)
    ## time with timeit
    bin_counts = bin_counts_to_numpy(np.array(n), np.array(triangles), ncellsy, ncellsz, min_y, diff, min_z)
    print("bin_counts", bin_counts.shape)
    print("sum", np.sum(bin_counts))
    print("mean", np.mean(bin_counts))
    print("max", np.max(bin_counts))
    print("min", np.min(bin_counts))
    print("count above 0", np.count_nonzero(bin_counts))

    binned_triangles = bin_triangles_to_numpy(n, triangles, ncellsy, ncellsz, min_y, diff, min_z, bin_counts, np.sum(bin_counts))
    print("min")
    print("binned_triangles", binned_triangles.shape)
    source_POINTS = np.array(source.points)
    end_POINTS = np.array(end.points)
    scatter_POINTS = np.array(scatter.points)




    print("max_x, min_x, max_y, min_y, max_z, min_z", max_x, min_x, max_y, min_y, max_z, min_z)
    print("triangle_vertices", n.shape)

    result = calculate_scattering(np.array(source_POINTS), np.array(end_POINTS), scatter_POINTS,
                                   triangles, n, wavelength,0, 
                                   ex_real, ex_imag, ey_real, ey_imag, ez_real, ez_imag, is_electric, permittivity_real, permittivity_imag,
                                     permeability_real, permeability_imag, normals, min_x, max_x, min_y, max_y, min_z, max_z, diff,bin_counts, binned_triangles, ncellsy, ncellsz, np.sum(bin_counts))


    
    # Check if the result matches the reference result

    ## assert almost equal
    assert False

def test_rays_intersection():
    """
    Similar to the first test but focuses on checking the consistency the set up of the problem
    The comparison is made to a reference saved .npy file found in the reference_files folder.
    The test will fail if any input is wrong, including
    - the rays casted

    as the raycaster returns a list of int2s with 0,0 at unsuccesful rays we need to create a boolean array to compare with the reference
    """
    antenna = meshio.read(rf"./reference_files/transmitting_antenna.ply")
    rectenna = meshio.read(rf"./reference_files/receiving_antenna.ply")
    transmit_horn = meshio.read(rf"./reference_files/transmit_horn.ply")
    receive_horn = meshio.read(rf"./reference_files/receive_horn.ply")
    reflector = meshio.read(rf"./reference_files/reflector.ply")
    triangles = np.ascontiguousarray(np.vstack((reflector.cells[0].data, transmit_horn.cells[0].data, receive_horn.cells[0].data)))
    triangle_vertices = np.ascontiguousarray(np.vstack((reflector.points, transmit_horn.points, receive_horn.points)))

    ## add the number of points in the previous mesh to the triangles array to be able to identify the different meshes
    triangles[reflector.cells[0].data.shape[0]:reflector.cells[0].data.shape[0]+transmit_horn.cells[0].data.shape[0]] += reflector.points.shape[0]
    triangles[reflector.cells[0].data.shape[0]+transmit_horn.cells[0].data.shape[0]:] += transmit_horn.points.shape[0] + reflector.points.shape[0]

    result = ray_cast_test(triangles, triangle_vertices, antenna.points, rectenna.points)
    bool_arr = np.zeros(antenna.points.shape[0]*rectenna.points.shape[0])
        #    as the raycaster returns a list of int2s with 0,0 at unsuccesful rays we need to create a boolean array to compare with the reference

    

    refrernce = np.load(rf"./reference_files/rayhits.npy")

    np.testing.assert_array_equal(bool_arr, refrernce)

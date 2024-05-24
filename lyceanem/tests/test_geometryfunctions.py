import numpy as np
import lyceanem
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R
import meshio
from importlib_resources import files
from  ..geometry import geometryfunctions as GF
import lyceanem.tests.data



@pytest.fixture
def standard_cube():
    cube_location= files(lyceanem.tests.data).joinpath("cube.ply")
    cube = meshio.read(cube_location)
    return cube


def are_meshes_equal(mesh1, mesh2, rtol=1e-5):
    # Check vertices
    meshes_are_equal = True
    meshes_are_equal =  np.allclose(mesh1.points, mesh2.points, rtol=rtol)


    # Check cells
    meshes_are_equal = len(mesh1.cells) == len(mesh2.cells)
    for i in range(len(mesh1.cells)):
        meshes_are_equal  *= np.allclose(mesh1.cells[i].data, mesh2.cells[i].data, rtol=rtol)
    ## normals
    print("checking 'normffffals3")

    if 'nx' in mesh1.point_data and 'ny' in mesh1.point_data and 'nz' in mesh1.point_data:
        if 'nx' in mesh2.point_data and 'ny' in mesh2.point_data and 'nz' in mesh2.point_data:
            meshes_are_equal  *= np.allclose(mesh1.point_data['nx'], mesh2.point_data['nx'], rtol=rtol)
            meshes_are_equal  *= np.allclose(mesh1.point_data['ny'], mesh2.point_data['ny'], rtol=rtol)
            meshes_are_equal  *= np.allclose(mesh1.point_data['nz'], mesh2.point_data['nz'], rtol=rtol)
        elif 'Normals' in mesh2.point_data:
            meshes_are_equal  *= np.allclose(mesh1.point_data['nx'], mesh2.point_data['Normals'][:,0], rtol=rtol)
            meshes_are_equal  *= np.allclose(mesh1.point_data['ny'], mesh2.point_data['Normals'][:,1], rtol=rtol)
            meshes_are_equal  *= np.allclose(mesh1.point_data['nz'], mesh2.point_data['Normals'][:,2], rtol=rtol)
        else:
            meshes_are_equal  *= False
    elif 'Normals' in mesh1.point_data:
        if 'nx' and 'ny' and 'nz' in mesh2.point_data:
            meshes_are_equal  *= np.allclose(mesh1.point_data['Normals'][:,0], mesh2.point_data['nx'], rtol=rtol)
            meshes_are_equal  *= np.allclose(mesh1.point_data['Normals'][:,1], mesh2.point_data['ny'], rtol=rtol)
            meshes_are_equal  *= np.allclose(mesh1.point_data['Normals'][:,2], mesh2.point_data['nz'], rtol=rtol)
            print("checking 'normals3")
        elif 'Normals' in mesh2.point_data:
            meshes_are_equal  *= np.allclose(mesh1.point_data['Normals'], mesh2.point_data['Normals'], rtol=rtol)
        else:
            meshes_are_equal  *= False
    else:
        meshes_are_equal  *= True
    ##normals cell data
    for i in range(len(mesh1.cells)):
        if 'nx' and 'ny' and 'nz' in mesh1.cell_data:
            if 'nx' and 'ny' and 'nz' in mesh2.cell_data:
                meshes_are_equal  *= np.allclose(mesh1.cell_data['nx'], mesh2.cell_data['nx'], rtol=rtol)
                meshes_are_equal  *= np.allclose(mesh1.cell_data['ny'], mesh2.cell_data['ny'], rtol=rtol)
                meshes_are_equal  *= np.allclose(mesh1.cell_data['nz'], mesh2.cell_data['nz'], rtol=rtol)
            elif 'Normals' in mesh2.cell_data:
                meshes_are_equal  *= np.allclose(mesh1.cell_data['nx'], mesh2.cell_data['Normals'][:,0], rtol=rtol)
                meshes_are_equal  *= np.allclose(mesh1.cell_data['ny'], mesh2.cell_data['Normals'][:,1], rtol=rtol)
                meshes_are_equal  *= np.allclose(mesh1.cell_data['nz'], mesh2.cell_data['Normals'][:,2], rtol=rtol)
            else:
                meshes_are_equal  *= False
        elif 'Normals' in mesh1.cell_data:
            if 'nx' and 'ny' and 'nz' in mesh2.cell_data:
                meshes_are_equal  *= np.allclose(mesh1.cell_data['Normals'][:,0], mesh2.cell_data['nx'], rtol=rtol)
                meshes_are_equal  *= np.allclose(mesh1.cell_data['Normals'][:,1], mesh2.cell_data['ny'], rtol=rtol)
                meshes_are_equal  *= np.allclose(mesh1.cell_data['Normals'][:,2], mesh2.cell_data['nz'], rtol=rtol)
            elif 'Normals' in mesh2.cell_data:
                meshes_are_equal  *= np.allclose(mesh1.cell_data['Normals'], mesh2.cell_data['Normals'], rtol=rtol)
            else:
                meshes_are_equal  *= False
        else:
            meshes_are_equal  *= True
    return meshes_are_equal
        

        


def test_tri_areas_2():
    refrence = np.load(files(lyceanem.tests.data).joinpath("areas.npy"))
    input = meshio.read(files(lyceanem.tests.data).joinpath("receive_horn.ply"))
    result = GF.tri_areas(input)
    assert np.allclose(result, refrence)

def test_tri_centroids():
    refrence = np.load(files(lyceanem.tests.data).joinpath('centroid_numpy.npy'))
    refrenece_mesh = meshio.read(files(lyceanem.tests.data).joinpath('centroid_mesh.ply'))
    input = meshio.read(files(lyceanem.tests.data).joinpath('receive_horn.ply'))
    result_numpy, result_mesh = GF.tri_centroids(input)
    assert np.allclose(result_numpy, refrence)
    assert are_meshes_equal(result_mesh, refrenece_mesh)
def test_mesh_rotate():
    refrence = meshio.read(files(lyceanem.tests.data).joinpath('rotated_recieve.ply'))
    input = meshio.read(files(lyceanem.tests.data).joinpath('receive_horn.ply'))
    rotation_vector1 = np.radians(np.asarray([90.0, 0.0, 0.0]))
    centre = np.array([1.0, 1.0, 1.0])
 

    result = GF.mesh_rotate(input, rotation_vector1, centre)
    
    assert are_meshes_equal(result, refrence)

def test_tri_areas(standard_cube):
    #define a cube with side length 1m, and test that the triangle areas are correct
    result=np.full((12),0.5)
    assert np.all(GF.tri_areas(standard_cube)==result)

def test_axes_from_normal_x_x():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[0] = 1
    rotation_matrix = R.from_euler('z',0,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_x_y():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[1] = 1
    rotation_matrix = R.from_euler('z',90,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_x_z():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = 1
    rotation_matrix = R.from_euler('y',-90,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_x_nz():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = -1
    rotation_matrix = R.from_euler('y',90,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)


def test_axes_from_normal_y_x():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[0] = 1
    rotation_matrix = R.from_euler('z',-90,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='y'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_y_y():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[1] = 1
    rotation_matrix = R.from_euler('z',0,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='y'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_y_z():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = 1
    rotation_matrix = R.from_euler('x',90,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='y'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_z_x():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[0] = 1
    rotation_matrix = np.array([[ 0.,  0.,  1.],
       [-1.,  0.,  0.],
       [ 0., -1.,  0.]])
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='z'),rotation_matrix,atol=1e-12)


def test_axes_from_normal_z_y():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[1] = 1
    rotation_matrix = R.from_euler('x',-90,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='z'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_z_z():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = 1
    rotation_matrix = R.from_euler('x',0,degrees=True).as_matrix()
    assert_allclose(GF.axes_from_normal(boresight,boresight_along='z'),rotation_matrix,atol=1e-12)

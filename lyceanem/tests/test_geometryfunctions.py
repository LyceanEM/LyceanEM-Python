import pytest
import numpy as np
import open3d as o3d
from numpy.testing import assert_equal, assert_allclose
from scipy.spatial.transform import Rotation as R
from ..geometry.geometryfunctions import tri_areas, axes_from_normal


def cube():
    # a cube centered at the origin, with side lengths of 1m (default)
    cube = o3d.geometry.TriangleMesh.create_box()
    cube.translate(np.array([-0.5,-0.5,-0.5]))
    return cube

@pytest.fixture
def standard_cube():
    return cube()

def test_tri_areas(standard_cube):
    #define a cube with side length 1m, and test that the triangle areas are correct
    result=np.full((12),0.5)
    assert np.all(tri_areas(standard_cube)==result)

def test_axes_from_normal_x_x():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[0] = 1
    rotation_matrix = R.from_euler('z',0,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_x_y():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[1] = 1
    rotation_matrix = R.from_euler('z',90,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_x_z():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = 1
    rotation_matrix = R.from_euler('y',-90,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_x_nz():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = -1
    rotation_matrix = R.from_euler('y',90,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='x'),rotation_matrix,atol=1e-12)


def test_axes_from_normal_y_x():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[0] = 1
    rotation_matrix = R.from_euler('z',-90,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='y'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_y_y():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[1] = 1
    rotation_matrix = R.from_euler('z',0,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='y'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_y_z():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = 1
    rotation_matrix = R.from_euler('x',90,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='y'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_z_x():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[0] = 1
    rotation_matrix = np.array([[ 0.,  0.,  1.],
       [-1.,  0.,  0.],
       [ 0., -1.,  0.]])
    assert_allclose(axes_from_normal(boresight,boresight_along='z'),rotation_matrix,atol=1e-12)


def test_axes_from_normal_z_y():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[1] = 1
    rotation_matrix = R.from_euler('x',-90,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='z'),rotation_matrix,atol=1e-12)

def test_axes_from_normal_z_z():
    boresight = np.zeros((3), dtype=np.float32)
    boresight[2] = 1
    rotation_matrix = R.from_euler('x',0,degrees=True).as_matrix()
    assert_allclose(axes_from_normal(boresight,boresight_along='z'),rotation_matrix,atol=1e-12)
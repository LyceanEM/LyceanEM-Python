import pytest
import numpy as np
import open3d as o3d
from numpy.testing import assert_equal, assert_allclose
from scipy.spatial.transform import Rotation as R
from ..electromagnetics.empropagation import vector_mapping

def test_vector_mapping_x_u():
    #initially test in global coordinate set
    rotation_matrix=R.from_euler('z',0,degrees=True).as_matrix()
    normal_vector=np.array([1.0, 0, 0])
    desired_E_vector=np.array([1.0, 0, 0],dtype=np.complex64) #uvn axes with n being the normal vector
    final_E_vector=np.array([0.0, 1.0, 0],dtype=np.complex64)
    assert_allclose(vector_mapping(desired_E_vector, normal_vector, rotation_matrix), final_E_vector,
                    atol=1e-12)

def test_vector_mapping_x_v():
    #initially test in global coordinate set
    rotation_matrix=R.from_euler('z',0,degrees=True).as_matrix()
    normal_vector=np.array([1.0, 0, 0])
    desired_E_vector=np.array([0.0, 1.0, 0],dtype=np.complex64) #uvn axes with n being the normal vector
    final_E_vector=np.array([0.0, 0.0, 1.0],dtype=np.complex64)
    assert_allclose(vector_mapping(desired_E_vector, normal_vector, rotation_matrix), final_E_vector,
                    atol=1e-12)

def test_vector_mapping_x_n():
    #initially test in global coordinate set
    rotation_matrix=R.from_euler('z',0,degrees=True).as_matrix()
    normal_vector=np.array([1.0, 0, 0])
    desired_E_vector=np.array([0, 0, 1.0],dtype=np.complex64) #uvn axes with n being the normal vector
    final_E_vector=np.array([1.0, 0, 0],dtype=np.complex64)
    assert_allclose(vector_mapping(desired_E_vector, normal_vector, rotation_matrix), final_E_vector,
                    atol=1e-12)

def test_vector_mapping_x_y_u():
    #initially test in global coordinate set
    rotation_matrix=R.from_euler('z',-90,degrees=True).as_matrix()
    normal_vector=np.array([1.0, 0, 0])
    desired_E_vector=np.array([1.0, 0, 0],dtype=np.complex64) #uvn axes with n being the normal vector
    final_E_vector=np.array([-1.0, 0.0, 0],dtype=np.complex64)
    assert_allclose(vector_mapping(desired_E_vector, normal_vector, rotation_matrix), final_E_vector,
                    atol=1e-12)

def test_vector_mapping_x_y_v():
    #initially test in global coordinate set
    rotation_matrix=R.from_euler('z',-90,degrees=True).as_matrix()
    normal_vector=np.array([1.0, 0, 0])
    desired_E_vector=np.array([0.0, 1.0, 0],dtype=np.complex64) #uvn axes with n being the normal vector
    final_E_vector=np.array([0.0, 0.0, 1.0],dtype=np.complex64)
    assert_allclose(vector_mapping(desired_E_vector, normal_vector, rotation_matrix), final_E_vector,
                    atol=1e-12)

def test_vector_mapping_x_y_n():
    #initially test in global coordinate set
    rotation_matrix=R.from_euler('z',-90,degrees=True).as_matrix()
    normal_vector=np.array([1.0, 0, 0])
    desired_E_vector=np.array([0, 0, 1.0],dtype=np.complex64) #uvn axes with n being the normal vector
    final_E_vector=np.array([0.0, 1.0, 0],dtype=np.complex64)
    assert_allclose(vector_mapping(desired_E_vector,normal_vector,rotation_matrix), final_E_vector,
                    atol=1e-12)
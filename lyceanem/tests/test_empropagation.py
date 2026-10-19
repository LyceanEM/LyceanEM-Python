import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R

from ..electromagnetics.empropagation import (
    calculate_conformalVectors,
    vector_mapping,
)

try:  # numpy >= 1.25
    ComplexWarning = np.exceptions.ComplexWarning
except AttributeError:  # older numpy
    ComplexWarning = np.ComplexWarning


def test_vector_mapping_x_u():
    # initially test in global coordinate set
    rotation_matrix = R.from_euler("z", 0, degrees=True).as_matrix()
    normal_vector = np.array([1.0, 0, 0])
    desired_E_vector = np.array(
        [1.0, 0, 0], dtype=np.complex64
    )  # uvn axes with n being the normal vector
    final_E_vector = np.array([0.0, 1.0, 0], dtype=np.complex64)
    assert_allclose(
        vector_mapping(desired_E_vector, normal_vector, rotation_matrix),
        final_E_vector,
        atol=1e-12,
    )


def test_vector_mapping_x_v():
    # initially test in global coordinate set
    rotation_matrix = R.from_euler("z", 0, degrees=True).as_matrix()
    normal_vector = np.array([1.0, 0, 0])
    desired_E_vector = np.array(
        [0.0, 1.0, 0], dtype=np.complex64
    )  # uvn axes with n being the normal vector
    final_E_vector = np.array([0.0, 0.0, 1.0], dtype=np.complex64)
    assert_allclose(
        vector_mapping(desired_E_vector, normal_vector, rotation_matrix),
        final_E_vector,
        atol=1e-12,
    )


def test_vector_mapping_x_n():
    # initially test in global coordinate set
    rotation_matrix = R.from_euler("z", 0, degrees=True).as_matrix()
    normal_vector = np.array([1.0, 0, 0])
    desired_E_vector = np.array(
        [0, 0, 1.0], dtype=np.complex64
    )  # uvn axes with n being the normal vector
    final_E_vector = np.array([1.0, 0, 0], dtype=np.complex64)
    assert_allclose(
        vector_mapping(desired_E_vector, normal_vector, rotation_matrix),
        final_E_vector,
        atol=1e-12,
    )


def test_vector_mapping_x_y_u():
    # initially test in global coordinate set
    rotation_matrix = R.from_euler("z", -90, degrees=True).as_matrix()
    normal_vector = np.array([1.0, 0, 0])
    desired_E_vector = np.array(
        [1.0, 0, 0], dtype=np.complex64
    )  # uvn axes with n being the normal vector
    final_E_vector = np.array([-1.0, 0.0, 0], dtype=np.complex64)
    assert_allclose(
        vector_mapping(desired_E_vector, normal_vector, rotation_matrix),
        final_E_vector,
        atol=1e-12,
    )


def test_vector_mapping_x_y_v():
    # initially test in global coordinate set
    rotation_matrix = R.from_euler("z", -90, degrees=True).as_matrix()
    normal_vector = np.array([1.0, 0, 0])
    desired_E_vector = np.array(
        [0.0, 1.0, 0], dtype=np.complex64
    )  # uvn axes with n being the normal vector
    final_E_vector = np.array([0.0, 0.0, 1.0], dtype=np.complex64)
    assert_allclose(
        vector_mapping(desired_E_vector, normal_vector, rotation_matrix),
        final_E_vector,
        atol=1e-12,
    )


def test_vector_mapping_x_y_n():
    # initially test in global coordinate set
    rotation_matrix = R.from_euler("z", -90, degrees=True).as_matrix()
    normal_vector = np.array([1.0, 0, 0])
    desired_E_vector = np.array(
        [0, 0, 1.0], dtype=np.complex64
    )  # uvn axes with n being the normal vector
    final_E_vector = np.array([0.0, 1.0, 0], dtype=np.complex64)
    assert_allclose(
        vector_mapping(desired_E_vector, normal_vector, rotation_matrix),
        final_E_vector,
        atol=1e-12,
    )


# --- ComplexWarning regression tests for vector_mapping / calculate_conformalVectors ---


def _vector_mapping_reference(local_E_vector, point_normal, rotation_matrix):
    """Pre-fix implementation of vector_mapping (LyceanEM Development f0c1cc5),
    kept to check that the real-basis fix is numerically unchanged."""
    point_vector = np.matmul(point_normal.astype(local_E_vector.dtype), rotation_matrix)
    local_axes = np.eye(3)
    uvn_axes = np.zeros((3, 3), dtype=local_E_vector.dtype)
    uvn_axes[2, :] = point_vector
    z_orth = np.linalg.norm(np.cross(local_axes[:, 2], point_vector))
    if abs(z_orth) == 0:
        uvn_axes[0, :] = np.cross(local_axes[0, :], point_vector) / np.linalg.norm(
            np.cross(local_axes[0, :], point_vector)
        )
    else:
        uvn_axes[0, :] = np.cross(local_axes[2, :], point_vector) / np.linalg.norm(
            np.cross(local_axes[2, :], point_vector)
        )
    uvn_axes[1, :] = np.cross(point_vector, uvn_axes[0, :]) / np.linalg.norm(
        np.cross(uvn_axes[0, :], point_vector)
    )
    return np.matmul(local_E_vector, uvn_axes)


def _calculate_conformalVectors_reference(desired_E_vector, source_normals, antenna_axes):
    conformal_E_vectors = np.zeros((source_normals.shape[0], 3), dtype=np.complex64)
    for normal_inc in range(len(source_normals)):
        if desired_E_vector.shape[0] == source_normals.shape[0]:
            local = desired_E_vector[normal_inc, :]
        else:
            local = desired_E_vector.ravel()
        conformal_E_vectors[normal_inc, :] = _vector_mapping_reference(
            local, source_normals[normal_inc, :], antenna_axes.astype(np.complex64)
        )
    return conformal_E_vectors


def _test_normals():
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -2.0, 0.5],
            [-0.3, 0.4, 0.866],
        ],
        dtype=np.float32,
    )
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def test_calculate_conformalVectors_real_E_no_complexwarning():
    normals = _test_normals()
    antenna_axes = np.eye(3)
    desired_E = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        result = calculate_conformalVectors(desired_E, normals, antenna_axes)
    assert result.dtype == np.complex64
    assert result.shape == (normals.shape[0], 3)
    # u vector is a unit vector orthogonal to the normal
    assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
    assert_allclose(np.sum(result.real * normals, axis=1), 0.0, atol=1e-6)


def test_vector_mapping_real_E_no_complexwarning_rotated():
    rotation_matrix = R.from_euler("xyz", [10, -35, 70], degrees=True).as_matrix()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        for normal in _test_normals():
            out = vector_mapping(
                np.array([0.0, 1.0, 0.0], dtype=np.float32), normal, rotation_matrix
            )
            assert np.isrealobj(out)


def test_calculate_conformalVectors_circular_polarisation_preserves_phase():
    normals = _test_normals()
    antenna_axes = np.eye(3)
    # RHCP-style local excitation: u + j v (unit power)
    desired_E = np.array([[1.0, 1.0j, 0.0]], dtype=np.complex64) / np.sqrt(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        result = calculate_conformalVectors(desired_E, normals, antenna_axes)
    assert result.dtype == np.complex64
    # real part is the u axis, imaginary part is the v axis: both unit-length/sqrt(2),
    # mutually orthogonal, and orthogonal to the normal, i.e. 90 degree phase offset kept
    u = result.real * np.sqrt(2)
    v = result.imag * np.sqrt(2)
    assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-6)
    assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    assert_allclose(np.sum(u * v, axis=1), 0.0, atol=1e-6)
    assert_allclose(np.sum(u * normals, axis=1), 0.0, atol=1e-6)
    assert_allclose(np.sum(v * normals, axis=1), 0.0, atol=1e-6)
    # right handed: u x v = n
    assert_allclose(np.cross(u, v), normals, atol=1e-6)
    # total power preserved
    assert_allclose(np.sum(np.abs(result) ** 2, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "desired_E",
    [
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[0.3, -0.7, 0.2]], dtype=np.float32),
        np.array([[1.0, 1.0j, 0.0]], dtype=np.complex64) / np.sqrt(2),
        np.array([[0.5 - 0.2j, 0.1 + 0.9j, 0.0]], dtype=np.complex64),
    ],
)
@pytest.mark.parametrize(
    "euler",
    [[0, 0, 0], [0, 0, -90], [10, -35, 70]],
)
def test_calculate_conformalVectors_matches_previous_implementation(desired_E, euler):
    normals = _test_normals()
    antenna_axes = R.from_euler("xyz", euler, degrees=True).as_matrix()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        expected = _calculate_conformalVectors_reference(desired_E, normals, antenna_axes)
    result = calculate_conformalVectors(desired_E, normals, antenna_axes)
    assert result.dtype == expected.dtype
    assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_calculate_conformalVectors_per_point_E_matches_previous_implementation():
    normals = _test_normals()
    antenna_axes = np.eye(3)
    rng = np.random.default_rng(0)
    desired_E = (
        rng.standard_normal((normals.shape[0], 3))
        + 1j * rng.standard_normal((normals.shape[0], 3))
    ).astype(np.complex64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        expected = _calculate_conformalVectors_reference(desired_E, normals, antenna_axes)
    result = calculate_conformalVectors(desired_E, normals, antenna_axes)
    assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_reference_implementation_did_raise_complexwarning():
    """Documents the original defect: the old code warns for a real float32 E vector."""
    normals = _test_normals()
    desired_E = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.warns(ComplexWarning):
        _calculate_conformalVectors_reference(desired_E, normals, np.eye(3))

import meshio
import numpy as np
from importlib_resources import files

import lyceanem.tests.data
from ..raycasting import rayfunctions as RF


def test_convertTriangles():
    reference = np.load(
        files(lyceanem.tests.data).joinpath("triangle_type_array_ref.npy")
    )
    mesh = meshio.read(files(lyceanem.tests.data).joinpath("receive_horn.ply"))
    triangles = RF.convertTriangles(mesh)
    for i in range(triangles.size):
        assert np.allclose(triangles[i]["v0x"], reference[i]["v0x"])
        assert np.allclose(triangles[i]["v0y"], reference[i]["v0y"])
        assert np.allclose(triangles[i]["v0z"], reference[i]["v0z"])
        assert np.allclose(triangles[i]["v1x"], reference[i]["v1x"])
        assert np.allclose(triangles[i]["v1y"], reference[i]["v1y"])
        assert np.allclose(triangles[i]["v1z"], reference[i]["v1z"])
        assert np.allclose(triangles[i]["v2x"], reference[i]["v2x"])
        assert np.allclose(triangles[i]["v2y"], reference[i]["v2y"])
        assert np.allclose(triangles[i]["v2z"], reference[i]["v2z"])

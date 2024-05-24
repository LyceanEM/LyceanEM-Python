import numpy as np
import pytest

from ray_tests import hit_test
import scipy.linalg as la




def test_hit():
    ## array of rays
    rays = np.array([[0.00001,	0,	1],[-1,	0,	1],[1,	0,	1],[1,	0,	1],[1,	0,	1]])
    ## array of origin
    ray_origin = np.array([[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-4,0,-1],[-2,0,-1]])
    ## array of triangles with bellow data
    triangles = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,0,1,1,0,1,1,0,1,1,0,1,1],[0,1,0,0,1,0,0,1,0,0,1,0,0,1,0]])
    triangle_edge_1 = np.ascontiguousarray(triangles[1,:]-triangles[0,:])
    triangle_edge_2 = np.ascontiguousarray(triangles[2,:]-triangles[0,:])
    triangle_origin = np.ascontiguousarray(triangles[0,:])
    result = hit_test(rays, ray_origin, triangle_edge_1, triangle_edge_2, triangle_origin)

    assert np.allclose(result, [0, 0, 0, 0, 0], atol=1e-6)






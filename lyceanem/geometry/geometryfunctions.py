import copy

import numpy as np
import open3d as o3d
from numba import vectorize
from packaging import version
from scipy.spatial.transform import Rotation as R
from .. import base_types as base_types
from .. import base_classes as base_classes
from ..raycasting import rayfunctions as RF


def tri_areas(solid):
    """
    Takes as an import an open3D triangle mesh, then calculates the area of all triangles in the global units, and
    returns as an array of areas
    """
    triangle_vertices = np.asarray(solid.vertices)
    triangleidx = np.asarray(solid.triangles)
    a_vectors = (
        triangle_vertices[triangleidx[:, 1], :]
        - triangle_vertices[triangleidx[:, 0], :]
    )
    b_vectors = (
        triangle_vertices[triangleidx[:, 2], :]
        - triangle_vertices[triangleidx[:, 0], :]
    )
    u = np.cross(a_vectors, b_vectors)
    areas = 0.5 * ((u[:, 0] ** 2 + u[:, 1] ** 2 + u[:, 2] ** 2) ** 0.5)

    return areas


def tri_centroids(solid):
    """
    In order to calculate the centroid of the triangle, take vertices from open3d triangle mesh, then put each triangle
    into origin space, creating oa,ob,oc vectors, then the centroid is a third of the sum of oa+ob+oc, converted back
    into global coordinates
    """
    triangle_vertices = np.asarray(solid.vertices)
    triangleidx = np.asarray(solid.triangles)
    oa = triangle_vertices[triangleidx[:, 0], :]
    ob = triangle_vertices[triangleidx[:, 1], :]
    oc = triangle_vertices[triangleidx[:, 2], :]
    centroids = (1 / 3) * (oa + ob + oc)
    centroid_cloud = o3d.geometry.PointCloud()
    centroid_cloud.points = o3d.utility.Vector3dVector(centroids)
    centroid_cloud.normals = solid.triangle_normals
    return centroids, centroid_cloud


def decimate_mesh(solid, mesh_sep):
    """
    In order to calculate the scattering appropriately the triangle mesh should be decimated so that the vertices
    are spaced mesh_sep apart.
    inputs are the :class:`open3d.geometry.TriangleMesh` solid, and the mesh_sep, and the output is a new :class:`open3d.geometry.TriangleMesh`. This is only required for
    the discrete scattering model, using the centroids or vertices
    """
    new_solid = o3d.geometry.TriangleMesh()
    # lineset=o3d.geometry.LineSet.create_from_triangle_mesh(solid)
    # identify triangles which are too large via areas greater than mesh_sep**2
    area_limit = mesh_sep ** 2
    areas = tri_areas(solid)
    large_tri_index = np.where(areas > area_limit)[0]
    while np.max(areas)>area_limit:
        solid=solid.subdivide_midpoint()
        areas = tri_areas(solid)

    new_solid=copy.deepcopy(solid)
    return new_solid


@vectorize(["(float32(float32))", "(float64(float64))"])
def elevationtotheta(el):
    # converting elevation in degrees to theta in degrees
    # elevation is in range -90 to 90 degrees
    # theta is in range 0 to 180 degrees
    if el >= 0.0:
        theta = 90.0 - el
    else:
        theta = np.abs(el) + 90.0

    return theta


@vectorize(["(float32(float32))", "(float64(float64))"])
def thetatoelevation(theta):
    # converting theta in degrees to elevation in degrees
    # elevation is in range -90 to 90 degrees
    # theta is in range 0 to 180 degrees
    if theta <= 90.0:
        # theta=(90.0-el)
        el = 90 - theta
    else:
        # theta=np.abs(el)+90.0
        el = -(theta - 90.0)

    return el


def open3drotate(
    item, rotation_matrix, rotation_centre=np.zeros((3, 1), dtype=np.float32)
):
    """

    Parameters
    ----------
    item : open3d object
        :class:`open3d.geometry.PointCloud`, :class:`open3d.geometry.TriangleMesh` or other open3d objects with .rotate function
    rotation_matrix : open3d rotation matrix
        rotation matrix for the desired transformation
    rotation_centre : numpy float (3,1)
        desired rotation centre, defaults to the origin

    Returns
    -------
    item : open3d object
        rotated item
    """
    if version.parse(o3d.__version__) >= version.parse("0.10.0"):
        # new syntax for rotations
        item.rotate(rotation_matrix, center=rotation_centre)
    else:

        item.translate(-1 * rotation_centre)
        item.rotate(rotation_matrix, center=False)
        item.translate(rotation_centre)

    return item


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array(
        [
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ]
    )
    return qCross_prod_mat


def calculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    qTrans_Mat = (
        np.eye(3, 3)
        + z_c_vec_mat
        + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
    )
    qTrans_Mat *= scale
    return qTrans_Mat


def axes_from_normal(boresight_vector, boresight_along="x"):
    """
    Calculates the required local axes within the global coordinate frame based upon the desired boresight, the standard
    is that the x-axis will be pointed along the desired boresight_vector, while the z-axis will be aligned as closely
    as possible to the global z-axis.

    Parameters
    ----------
    boresight_vector

    Returns
    -------
    rotation_matrix

    """
    # initially define the rotation matrix based inline with the global coordinate frame.
    #for more consistency, need to calculate additional vectors to ensure rotation mapping is correct.

    replacement_vector = boresight_vector / np.linalg.norm(boresight_vector)

    if boresight_along == "x":
        alignment_vector=np.array([[1.0,0,0]])
        u_vector=np.array([[0.0,1.0,0]])
        v_vector=np.array([[0,0,1.0]])
    elif boresight_along == "y":
        alignment_vector=np.array([[0.0,1.0,0]])
        u_vector = np.array([[1.0, 0.0, 0]])
        v_vector = np.array([[0, 0, 1.0]])
    elif boresight_along == "z":
        alignment_vector=np.array([[0.0,0,1.0]])
        u_vector = np.array([[1.0, 0.0, 0]])
        v_vector = np.array([[0, 1.0, 0]])
    rotation,_=R.align_vectors(replacement_vector.reshape(1,3),alignment_vector)
    # if boresight_along == "x":
    #     # then u should be based on y, and v on z
    #     rotation_matrix[1, :] = np.cross(
    #         rotation_matrix[2, :], rotation_matrix[0, :]
    #     ) / np.linalg.norm(np.cross(rotation_matrix[2, :], rotation_matrix[0, :]))
    #     rotation_matrix[2, :] = np.cross(
    #         rotation_matrix[0, :], rotation_matrix[1, :]
    #     ) / np.linalg.norm(np.cross(rotation_matrix[0, :], rotation_matrix[1, :]))
    #     if np.any(np.isnan(rotation_matrix)):
    #         # nan in rotation_matrix, start again with different basis vector
    #         rotation_matrix = np.eye(3)
    #         rotation_matrix[0, :] = replacement_vector
    #         rotation_matrix[2, :] = np.cross(
    #             rotation_matrix[0, :], rotation_matrix[1, :]
    #         ) / np.linalg.norm(np.cross(rotation_matrix[0, :], rotation_matrix[1, :]))
    #         rotation_matrix[1, :] = np.cross(
    #             rotation_matrix[2, :], rotation_matrix[0, :]
    #         ) / np.linalg.norm(np.cross(rotation_matrix[2, :], rotation_matrix[0, :]))
    # elif boresight_along == "y":
    #     # then u should be based on x, and v on z
    #     rotation_matrix[0, :] = np.cross(
    #         rotation_matrix[1, :], rotation_matrix[2, :]
    #     ) / np.linalg.norm(np.cross(rotation_matrix[1, :], rotation_matrix[2, :]))
    #     rotation_matrix[2, :] = np.cross(
    #         rotation_matrix[0, :], rotation_matrix[1, :]
    #     ) / np.linalg.norm(np.cross(rotation_matrix[0, :], rotation_matrix[1, :]))
    #     if np.any(np.isnan(rotation_matrix)):
    #         # nan in rotation_matrix, start again with different basis vector
    #         rotation_matrix = np.eye(3)
    #         rotation_matrix[1, :] = replacement_vector
    #         rotation_matrix[2, :] = np.cross(
    #             rotation_matrix[0, :], rotation_matrix[1, :]
    #         ) / np.linalg.norm(np.cross(rotation_matrix[0, :], rotation_matrix[1, :]))
    #         rotation_matrix[0, :] = np.cross(
    #             rotation_matrix[1, :], rotation_matrix[2, :]
    #         ) / np.linalg.norm(np.cross(rotation_matrix[1, :], rotation_matrix[2, :]))
    # elif boresight_along == "z":
    #     # then u should be based on x, and v on y
    #     rotation_matrix[0, :] = np.cross(
    #         rotation_matrix[1, :], rotation_matrix[2, :]
    #     ) / np.linalg.norm(np.cross(rotation_matrix[1, :], rotation_matrix[2, :]))
    #     rotation_matrix[1, :] = np.cross(
    #         rotation_matrix[2, :], rotation_matrix[0, :]
    #     ) / np.linalg.norm(np.cross(rotation_matrix[2, :], rotation_matrix[0, :]))
    #     if np.any(np.isnan(rotation_matrix)):
    #         # nan in rotation_matrix, start again with different basis vector
    #         print('nan')
    #         rotation_matrix = np.eye(3)
    #         rotation_matrix[2, :] = replacement_vector
    #         rotation_matrix[1, :] = np.cross(
    #             rotation_matrix[2, :], rotation_matrix[0, :]
    #         ) / np.linalg.norm(np.cross(rotation_matrix[2, :], rotation_matrix[0, :]))
    #         rotation_matrix[0, :] = np.cross(
    #             rotation_matrix[1, :], rotation_matrix[2, :]
    #         ) / np.linalg.norm(np.cross(rotation_matrix[1, :], rotation_matrix[2, :]))

    return rotation.as_matrix()

def mesh_conversion(conversion_object):
    """
    Convert the provide file object into triangle_t format

    Parameters
    ----------
    conversion_object : solid object to be converted into triangle_t format, could be open3d trianglemesh, solid, or antenna structure

    Returns
    -------
    triangles : numpy array of type triangle_t
    """
    if isinstance(conversion_object,base_classes.structures):
        triangles = object.triangles_base_raycaster()
    elif isinstance(conversion_object,base_classes.antenna_structures):
        exported_structure=base_classes.structures(solids=object.export_all_structures())
        triangles=exported_structure.triangles_base_raycaster()
    elif isinstance(conversion_object,type(o3d.geometry.TriangleMesh())):
        triangles=RF.convertTriangles(object)
    elif isinstance(conversion_object,list):
        triangles = np.empty((0), dtype=base_types.triangle_t)
        #print("Detected List")
        for item in conversion_object:
            #print(type(item))
            #print(isinstance(item,type(o3d.geometry.TriangleMesh())))
            if isinstance(item, type(o3d.geometry.TriangleMesh())):
                triangles = np.append(triangles,RF.convertTriangles(item),axis=0)
                #print(len(triangles))
            elif isinstance(item, base_classes.structures):
                triangles=np.append(triangles,item.triangles_base_raycaster(),axis=0)
                #print(len("B ",triangles))

    else:
        print("no structures")
        print(type(object))
        triangles = np.empty((0), dtype=base_types.triangle_t)

    return triangles

import copy

import meshio
import numpy as np
from numba import vectorize
from scipy.spatial.transform import Rotation as R

from .. import base_classes as base_classes
from .. import base_types as base_types
from ..electromagnetics.emfunctions import transform_em
from ..raycasting import rayfunctions as RF


def mesh_conversion_to_meshio(conversion_object):
    """
    Convert the provided file object into meshio format

    Parameters
    ----------
    conversion_object : solid object to be converted into meshio format, could be meshio.Mesh trianglemesh, solid, or antenna structure

    Returns
    -------
    meshio_mesh : meshio.Mesh
        The converted meshio mesh object
    """
    if isinstance(conversion_object, base_classes.structures):
        meshio_mesh = conversion_object.export_combined_meshio()
    elif isinstance(conversion_object, base_classes.antenna_structures):
        exported_structure = base_classes.structures(
            solids=conversion_object.export_all_structures()
        )
        meshio_mesh = exported_structure.export_combined_meshio()
    elif isinstance(conversion_object, meshio.Mesh):
        meshio_mesh = conversion_object
    elif isinstance(conversion_object, list):
        assert False
    else:
        print("no structures")
        print(type(conversion_object))
        meshio_mesh = None
    return meshio_mesh

def cell_centroids(field_data):
    """
    In order to calculate the centroid of the triangle, take vertices from meshio triangle mesh, then put each triangle
    into origin space, creating oa,ob,oc vectors, then the centroid is a third of the sum of oa+ob+oc, converted back
    into global coordinates

    Parameters
    ----------

    field_data : meshio.Mesh

    Returns
    --------
    centroid_cloud : meshio.Mesh
        The centroid cloud, with the centroid of each triangle as a vertex, and the cell data from the original mesh
        as point data
    """
    for inc, cell in enumerate(field_data.cells):

        if cell.type == "triangle":
            v0 = field_data.points[cell.data[:, 0], :]
            v1 = field_data.points[cell.data[:, 1], :]
            v2 = field_data.points[cell.data[:, 2], :]
            centroids = (1 / 3) * (v0 + v1 + v2)
            triangle_inc = inc

    centroid_cloud = meshio.Mesh(
        points=centroids,
        cells=[
            (
                "vertex",
                np.array(
                    [
                        [
                            i,
                        ]
                        for i in range(len(centroids))
                    ]
                ),
            )
        ],
    )
    for key in field_data.cell_data.keys():
        centroid_cloud.point_data[key] = field_data.cell_data[key][triangle_inc]

    return centroid_cloud


def mesh_rotate(mesh, rotation, rotation_centre=np.zeros((1, 3), dtype=np.float32)):
    """
    Rotate the provided meshio mesh about the defined center using either rotation vector or rotation matrix.
    This function will also rotate the cartesian electric field vectors if present, or convert spherical vectors to cartesian then rotate.

    Parameters
    ----------
    mesh : meshio.Mesh

    rotation : numpy.ndarray
        the rotation vector or rotation matrix for the desired rotation

    rotation_centre : numpy.ndarray
        Centre of rotation

    Returns
    -------
    mesh : meshio.Mesh
        The rotated mesh, with cartesian electric field vectors 

    """
    if rotation.shape == (3,):
        r = R.from_rotvec(rotation)
    elif rotation.shape == (3, 3):
        r = R.from_matrix(rotation)
    else:
        raise ValueError("Rotation  must be a 3x1 or 3x3 array")
    if rotation_centre.shape == (3,) or rotation_centre.shape == (3, 1):
        rotation_centre = rotation_centre.reshape(1, 3)
    rotated_points = r.apply(mesh.points - rotation_centre) + rotation_centre
    cell_data = mesh.cell_data
    point_data = mesh.point_data

    if "Normals" in mesh.point_data:
        # rotate normals cloud
        normals = mesh.point_data["Normals"]
        rotated_normals = r.apply(normals)
        point_data["Normals"] = rotated_normals
    if "Normals" in mesh.cell_data:
        # rotate normals cloud
        for i, rotated_normals in enumerate(mesh.cell_data["Normals"]):
            rotated_normals = r.apply(rotated_normals)
            cell_data["Normals"][i] = rotated_normals

    mesh_return = meshio.Mesh(points=rotated_points, cells=mesh.cells)
    mesh_return.point_data = point_data
    mesh_return.cell_data = cell_data

    # if field data is present, rotate fields
    if all(k in mesh.point_data.keys() for k in ("Ex-Real", "Ey-Real", "Ez-Real")):
        # rotate field data
        fields = transform_em(copy.deepcopy(mesh), r)
        for key in (
            "Ex-Real",
            "Ex-Imag",
            "Ey-Real",
            "Ey-Imag",
            "Ez-Real",
            "Ez-Imag",
        ):
            mesh_return.point_data[key] = fields.point_data[key]

    elif all(k in mesh.point_data.keys() for k in ("E(theta)", "E(phi)")):
        fields = transform_em(copy.deepcopy(mesh), r)
        for key in (
            "Ex-Real",
            "Ex-Imag",
            "Ey-Real",
            "Ey-Imag",
            "Ez-Real",
            "Ez-Imag",
        ):
            mesh_return.point_data[key] = fields.point_data[key]

    return mesh_return


def mesh_transform(mesh, transform_matrix, rotate_only):
    """
    Transform the provided meshio mesh using the provided transformation matrix. The transformation matrix should be a 4x4 matrix. This function should be used with care, as currently the electric field vectors are not transformed.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh to be transformed.

    transform_matrix : numpy.ndarray
        The transformation matrix to be applied to the mesh. Should be a 4x4 matrix.

    rotate_only : bool
        If True, only the rotation part of the transformation matrix will be applied. If False, the full transformation (including translation) will be applied.

    Returns
    -------
    mesh : meshio.Mesh
        The transformed mesh.
    """
    return_mesh = mesh
    if rotate_only:
        for i in range(mesh.points.shape[0]):
            return_mesh.points[i] = np.dot(
                transform_matrix, np.append(mesh.points[i], 0)
            )[:3]
            return_mesh.point_data["Normals"][i] = np.dot(
                transform_matrix, np.append(mesh.point_data["Normals"][i], 0)
            )[:3]

    else:
        for i in range(mesh.points.shape[0]):
            return_mesh.points[i] = np.dot(
                transform_matrix, np.append(mesh.points[i], 1)
            )[:3]
            return_mesh.point_data["Normals"][i] = np.dot(
                transform_matrix, np.append(mesh.point_data["Normals"][i], 0)
            )[:3]
        if "Normals" in mesh.cell_data:
            if isinstance(mesh.cell_data["Normals"], list):
                # multiple cell types
                for i in range(len(mesh.cell_data["Normals"])):
                    for j in range(mesh.cell_data["Normals"][i].shape[0]):
                        return_mesh.cell_data["Normals"][i][j] = np.dot(
                            transform_matrix,
                            np.append(mesh.cell_data["Normals"][i][j], 0),
                        )[:3]
            else:
                # single cell type
                for i in range(len(mesh.cell_data["Normals"])):
                    return_mesh.cell_data["Normals"][i] = np.dot(
                        transform_matrix, np.append(mesh.cell_data["Normals"][i], 0)
                    )[:3]

    return return_mesh


def locate_cell_index(field_data, cell_type="triangle"):
    """
    Locate the index of the first cell of the specified type in the field data.

    Parameters
    ----------
    field_data : meshio.Mesh
        The meshio mesh object containing the cells.
    cell_type : str, optional
        The type of cell to locate. The default is "triangle".

    Returns
    -------
    int
        The index of the first cell of the specified type.

    :private
    """
    for inc, cell in enumerate(field_data.cells):
        if cell.type == cell_type:
            desired_index = inc

    return desired_index


def compute_areas(field_data):
    """
    Computes the area of each cell in the mesh and assigns it to the cell data. The area is calculated using Heron's formula for triangles and quads.
    The area is also calculated for each point in the mesh, based on the area of the cells it is associated with.

    Parameters
    ----------
    field_data : meshio.Mesh
        The meshio mesh object containing the cells and points.

    Returns
    -------
    field_data : meshio.Mesh
        The meshio mesh object with the computed areas added to the cell data and point data.
    """
    cell_areas = []
    for inc, cell in enumerate(field_data.cells):

        if cell.type == "triangle":
            # Heron's Formula
            edge1 = np.linalg.norm(
                field_data.points[cell.data[:, 0], :]
                - field_data.points[cell.data[:, 1], :],
                axis=1,
            )
            edge2 = np.linalg.norm(
                field_data.points[cell.data[:, 1], :]
                - field_data.points[cell.data[:, 2], :],
                axis=1,
            )
            edge3 = np.linalg.norm(
                field_data.points[cell.data[:, 2], :]
                - field_data.points[cell.data[:, 0], :],
                axis=1,
            )
            s = (edge1 + edge2 + edge3) / 2.0
            areas = (s * (s - edge1) * (s - edge2) * (s - edge3)) ** 0.5
            cell_areas.append(areas)
        if cell.type == "quad":
            # Heron's Formula twice
            edge1 = np.linalg.norm(
                field_data.points[cell.data[:, 0], :]
                - field_data.points[cell.data[:, 1], :],
                axis=1,
            )
            edge2 = np.linalg.norm(
                field_data.points[cell.data[:, 1], :]
                - field_data.points[cell.data[:, 2], :],
                axis=1,
            )
            edge3 = np.linalg.norm(
                field_data.points[cell.data[:, 2], :]
                - field_data.points[cell.data[:, 0], :],
                axis=1,
            )
            edge4 = np.linalg.norm(
                field_data.points[cell.data[:, 2], :]
                - field_data.points[cell.data[:, 3], :],
                axis=1,
            )
            edge5 = np.linalg.norm(
                field_data.points[cell.data[:, 3], :]
                - field_data.points[cell.data[:, 0], :],
                axis=1,
            )

            s1 = (edge1 + edge2 + edge3) / 2.0
            s2 = (edge3 + edge4 + edge5) / 2.0
            areas = (s1 * ((s1 - edge1) * (s1 - edge2) * (s1 - edge3))) ** 0.5 + (
                s2 * (s2 - edge3) * (s2 - edge4) * (s2 - edge5)
            ) ** 0.5
            cell_areas.append(areas)

    field_data.cell_data["Area"] = cell_areas
    field_data.point_data["Area"] = np.zeros((field_data.points.shape[0]))
    for inc, cell in enumerate(field_data.cells):
        if field_data.cells[inc].type == "triangle":
            for point_inc in range(field_data.points.shape[0]):
                field_data.point_data["Area"][point_inc] = np.sum(
                    field_data.cell_data["Area"][inc][
                        np.where(field_data.cells[inc].data == point_inc)[0]
                    ] / 3
                )

    return field_data


def compute_normals(mesh):
    """
    Computes the Cell Normals for meshio mesh objects, the point normals will also be calculated for all points which are connected to a triangle, quad, or vertex, isolated points will be propulated with a normal vector of nan.
    If the only cells are vertex cells, then the normals are defined from the mesh origin to each vertex.

    Parameters
    ----------
    mesh

    Returns
    -------
    mesh : meshio.Mesh
        The meshio mesh object with the computed normals added to the cell data and point data.

    """
    cell_normal_list = []
    for inc, cell in enumerate(mesh.cells):
        # print(cell.type, cell.data.shape[0])
        if cell.type == "vertex":
            # assume outward pointing normals from centroid
            vertex_normals = mesh.points[cell.data[:, 0], :] / np.linalg.norm(
                mesh.points[cell.data[:, 0], :], axis=1
            ).reshape(-1, 1)
            cell_normal_list.append(vertex_normals)
        if cell.type == "line":
            line_normals = np.zeros((cell.data.shape[0], 3))
            cell_normal_list.append(line_normals)
        if cell.type == "triangle":
            # print(inc)
            edge1 = mesh.points[cell.data[:, 0], :] - mesh.points[cell.data[:, 1], :]
            edge2 = mesh.points[cell.data[:, 0], :] - mesh.points[cell.data[:, 2], :]
            tri_cell_normals = np.cross(edge1, edge2)
            tri_cell_normals *= (1 / np.linalg.norm(tri_cell_normals, axis=1)).reshape(
                -1, 1
            )
            cell_normal_list.append(tri_cell_normals)
        if cell.type == "tetra":
            # print(inc)
            edge1 = mesh.points[cell.data[:, 0], :] - mesh.points[cell.data[:, 1], :]
            edge2 = mesh.points[cell.data[:, 0], :] - mesh.points[cell.data[:, 2], :]
            tetra_cell_normals = np.cross(edge1, edge2)
            tetra_cell_normals *= (
                1 / np.linalg.norm(tetra_cell_normals, axis=1)
            ).reshape(-1, 1)
            cell_normal_list.append(tetra_cell_normals)

    mesh.cell_data["Normals"] = cell_normal_list

    # calculate vertex normals
    point_normals = np.empty((0, 3))
    for inc, cell in enumerate(mesh.cells):
        if cell.type == "triangle":
            for inc_b in range(mesh.points.shape[0]):
                associated_cells = np.where(inc_b == cell.data)[0]
                # print(associated_cells)
                point_normals = np.append(
                    point_normals,
                    np.mean(
                        mesh.cell_data["Normals"][0][associated_cells, :], axis=0
                    ).reshape(1, 3),
                    axis=0,
                )
        if cell.type == "vertex":
            for inc_b in range(mesh.points.shape[0]):
                associated_cells = np.where(inc_b == cell.data)[0]
                # print(associated_cells)
                point_normals = np.append(
                    point_normals,
                    np.mean(
                        mesh.cell_data["Normals"][0][associated_cells, :], axis=0
                    ).reshape(1, 3),
                    axis=0,
                )

    mesh.point_data["Normals"] = point_normals / np.linalg.norm(
        point_normals, axis=1
    ).reshape(-1, 1)

    return mesh


def theta_phi_r(field_data):
    """
    Calculate the spherical coordinates for the provided field data relative to the origin

    Parameters
    ----------
    field_data

    Returns
    -------
    field_data : meshio.Mesh
        The meshio mesh object with the computed spherical coordinates added to the point data.

    """
    field_data.point_data["Radial_Distance_(m)"] = np.linalg.norm(
        field_data.points - np.zeros((1, 3)), axis=1
    )
    field_data.point_data["theta_(Radians)"] = np.arccos(
        field_data.points[:, 2] / field_data.point_data["Radial_Distance_(m)"]
    )
    field_data.point_data["phi_(Radians)"] = np.arctan2(
        field_data.points[:, 1], field_data.points[:, 0]
    )
    return field_data


def mesh_conversion(conversion_object):
    """
    Convert the provide file object into triangle_t format for raycasting

    Parameters
    ----------
    conversion_object : solid object to be converted into triangle_t format, could be meshio.Mesh trianglemesh, solid, or antenna structure

    Returns
    -------
    triangles : numpy array of type triangle_t
    """
    if isinstance(conversion_object, base_classes.structures):
        triangles = conversion_object.triangles_base_raycaster()
    elif isinstance(conversion_object, base_classes.antenna_structures):
        exported_structure = base_classes.structures(
            solids=conversion_object.export_all_structures()
        )
        triangles = exported_structure.triangles_base_raycaster()
    elif isinstance(conversion_object, meshio.Mesh):
        triangles = RF.convertTriangles(conversion_object)
    elif isinstance(conversion_object, list):
        triangles = np.empty((0), dtype=base_types.triangle_t)
        for item in conversion_object:
            if isinstance(item, meshio.Mesh):
                triangles = np.append(triangles, RF.convertTriangles(item), axis=0)
            elif isinstance(item, base_classes.structures):
                triangles = np.append(
                    triangles, item.triangles_base_raycaster(), axis=0
                )

    else:
        print("no structures")
        print(type(conversion_object))
        triangles = np.empty((0), dtype=base_types.triangle_t)

    return triangles


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
    # for more consistency, need to calculate additional vectors to ensure rotation mapping is correct.

    replacement_vector = boresight_vector / np.linalg.norm(boresight_vector)

    if boresight_along == "x":
        alignment_vector = np.array([[1.0, 0, 0]])

    elif boresight_along == "y":
        alignment_vector = np.array([[0.0, 1.0, 0]])

    elif boresight_along == "z":
        alignment_vector = np.array([[0.0, 0, 1.0]])

    rotation, _ = R.align_vectors(replacement_vector.reshape(1, 3), alignment_vector)

    return rotation.as_matrix()


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


def mesh_translate(mesh, translation_vector):
    """
    Translate a mesh by a given translation vector.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh to be translated.
    translation_vector : numpy.ndarray
        The translation vector to be applied to the mesh. Should be a 1x3 array.

    Returns
    -------
    mesh : meshio.Mesh
        The translated mesh.
    """
    translated_points = mesh.points + translation_vector
    cell_data = mesh.cell_data
    point_data = mesh.point_data
    mesh_return = meshio.Mesh(points=translated_points, cells=mesh.cells)
    mesh_return.point_data = point_data
    mesh_return.cell_data = cell_data
    return mesh_return


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

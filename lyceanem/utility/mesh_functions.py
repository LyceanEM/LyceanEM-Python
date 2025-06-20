import meshio
import pyvista as pv


def pyvista_to_meshio(polydata_object):
    """
    Convert a pyvista object to a meshio object, if pyvista >=0.45, use to_meshio method.

    Parameters
    ----------
    polydata_object : pyvista.PolyData or pyvista.UnstructuredGrid
        The pyvista object to convert.

    Returns
    -------
    meshio_object : :type:`meshio.Mesh`
        The converted meshio object.
    """
    # from packaging.version import parse as parse_version
    # if parse_version(pv.__version__)>=parse_version("0.45.0"):
    #    meshio_object=pv.from_meshio(polydata_object)
    # else:
    if type(polydata_object) == pv.core.pointset.UnstructuredGrid:
        cells = id_cells(polydata_object.cells)
    else:
        cells = id_cells(polydata_object.faces)
    meshio_object = meshio.Mesh(
        points=polydata_object.points,
        cells=cells,
        point_data=polydata_object.point_data,
    )
    return meshio_object


def meshio_to_pyvista(meshio_object):
    """
    Convert a meshio object to a pyvista object.

    Parameters
    ----------
    meshio_object : :type:`meshio.Mesh`
        The meshio object to convert.

    Returns
    -------
    pyvista_object : :type:`pyvista.PolyData` or :type:`pyvista.UnstructuredGrid`
        The converted pyvista object.
    """
    import numpy as np

    for cell in meshio_object.cells:
        if cell.type == "triangle":
            tris = cell.data
            faces = np.hstack((np.ones((tris.shape[0], 1), dtype=int) * 3, tris))

    polydata_object = pv.PolyData(meshio_object.points, faces=faces)
    for key in meshio_object.point_data.keys():
        polydata_object.point_data[key] = meshio_object.point_data[key]

    for key in meshio_object.cell_data.keys():
        polydata_object.cell_data[key] = meshio_object.cell_data[key][0]
    return polydata_object


def id_cells(faces):
    """
    Identify cell ids

    Parameters
    ----------
    faces : numpy.ndarray of int
        The faces of the mesh.

    Returns
    -------
    meshio_cells : list
        A list of tuples containing the cell type and the cell data.
    """
    # temp_faces=copy.deepcopy(faces)

    cell_types = {1: "vertex", 2: "line", 3: "triangle", 4: "quad"}
    cells = {"vertex": [], "line": [], "triangle": [], "quad": []}

    moving_index = 0
    while moving_index <= faces.shape[0] - 1:
        face_num = faces[moving_index]
        temp_array = faces[moving_index + 1 : moving_index + face_num + 1]
        cells[cell_types[face_num]].append(temp_array.tolist())
        moving_index += face_num + 1

    # while (temp_faces.shape[0]>=1):
    #     trim_num =temp_faces[0]
    #     if trim_num>3:
    #         print("Cell Face Error")
    #     temp_array =temp_faces[1:trim_num +1]
    #     cells[cell_types[trim_num]].append(temp_array.tolist())
    #     temp_faces =np.delete(temp_faces ,np.arange(0 ,trim_num +1))

    meshio_cells = []
    for key in cells:
        if len(cells[key]) > 0:
            meshio_cells.append((key, cells[key]))

    return meshio_cells


def points2pointcloud(xyz):
    """
    turns numpy array of xyz data into a meshio format point cloud

    Parameters
    ----------
    xyz : numpy.ndarray of float
        The xyz data to convert. The shape of the array should be (n, 3) or (n, 1, 3). If the shape is (n, 1, 3), it will be reshaped to (n, 3).

    Returns
    -------
    new_point_cloud : :type:`meshio.Mesh`
        The converted meshio point cloud.

    """
    import numpy as np

    if xyz.shape[1] == 3:
        reshaped = xyz

    else:
        reshaped = xyz.reshape((int(len(xyz.ravel()) / 3), 3))

    # assume mean of xyz is centre, and normals are outwards facing from center.
    center = np.mean(xyz, axis=0)
    normals = reshaped - center
    mesh_points = meshio.Mesh(
        points=reshaped,
        cells=[
            (
                "vertex",
                np.array(
                    [
                        [
                            i,
                        ]
                        for i in range(reshaped.shape[0])
                    ]
                ),
            )
        ],
        point_data={
            "Normals": normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)
        },
    )
    return mesh_points

import meshio
import pyvista as pv


def pyvista_to_meshio(polydata_object):
    """
    Convert a pyvista object to a meshio object
    """
    # extract only the triangles
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


def id_cells(faces):
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

import pyvista as pv
import meshio

def pyvista_to_meshio(polydata_object):
    """
    Convert a pyvista object to a meshio object
    """
    #extract only the triangles
    if type(polydata_object )==pv.core.pointset.UnstructuredGrid:
        cells =id_cells(polydata_object.cells)
    else:
        cells =id_cells(polydata_object.faces)
    meshio_object = meshio.Mesh(
        points=polydata_object.points,
        cells=cells,
        point_data=polydata_object.point_data,
    )
    return meshio_object

def id_cells(faces):
    import numpy as np
    cell_types ={1 :"vertex",
                2 :"line",
                3 :"triangle",
                4 :"quad"
                }
    cells ={"vertex" :[],
           "line" :[],
           "triangle" :[],
           "quad" :[]
           }

    while (faces.shape[0 ]>=1):
        trim_num =faces[0]
        temp_array =faces[1:trim_num +1]
        cells[cell_types[trim_num]].append(temp_array.tolist())
        faces =np.delete(faces ,np.arange(0 ,trim_num +1))

    meshio_cells =[]
    for key in cells:
        if len(cells[key] ) >0:
            meshio_cells.append((key ,cells[key]))

    return meshio_cells
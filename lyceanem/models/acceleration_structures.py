import numpy as np

from lyceanem.em import bin_counts_to_numpy, bin_triangles_to_numpy
from lyceanem.em import calculate_scattering_brute_force
from lyceanem.em import calculate_scattering_tiles


class Tile_acceleration_structure:
    """
    This class is used to create a tile acceleration structure for the scattering calculation.

    Parameters
    ----------
    blocking_mesh : meshio.Mesh
        The mesh to be used for the blocking structure.
    n_cells : int
        The number of cells to be used for the tile acceleration structure.

    Attributes
    ----------
    triangle_verticies : numpy.ndarray
        The vertices of the triangles in the blocking mesh.
    max_x : float
        The maximum x coordinate of the blocking mesh.
    min_x : float
        The minimum x coordinate of the blocking mesh.
    max_y : float
        The maximum y coordinate of the blocking mesh.
    min_y : float
        The minimum y coordinate of the blocking mesh.
    max_z : float
        The maximum z coordinate of the blocking mesh.
    min_z : float
        The minimum z coordinate of the blocking mesh.
    tile_size : float
        The size of the tiles used for the acceleration structure.
    y_cells_count : int
        The number of cells in the y direction.
    z_cells_count : int
        The number of cells in the z direction.
    bin_counts : numpy.ndarray
        The counts of triangles in each bin.
    binned_triangles_count : int
        The total number of triangles in the bins
    binned_triangles : numpy.ndarray
        The triangles in the bins.

    Methods
    -------
    calculate_scattering(source_mesh, sink_mesh, alpha, beta, wavelength, self_to_self, chunk_count=1)
        Calculate the scattering from the source mesh to the sink mesh.


    """
    def __init__(self, blocking_mesh, n_cells):
        self.triangle_verticies = np.ascontiguousarray(blocking_mesh.points)
        self.max_x = np.max(blocking_mesh.points[:, 0]) + abs(np.max(blocking_mesh.points[:, 0]) * 0.001)
        self.min_x = np.min(blocking_mesh.points[:, 0]) -abs(np.min(blocking_mesh.points[:, 0]) * 0.001)
        self.max_y = np.max(blocking_mesh.points[:, 1]) + abs(np.max(blocking_mesh.points[:, 1]) * 0.001)
        self.min_y = np.min(blocking_mesh.points[:, 1]) - abs(np.min(blocking_mesh.points[:, 1]) * 0.001)
        self.max_z = np.max(blocking_mesh.points[:, 2]) + abs(np.max(blocking_mesh.points[:, 2]) * 0.001)
        self.min_z = np.min(blocking_mesh.points[:, 2]) - abs(np.min(blocking_mesh.points[:, 2]) * 0.001)
        diff_y = self.max_y - self.min_y
        diff_z = self.max_z - self.min_z
        diff = min( diff_y, diff_z)
        self.tile_size = diff / n_cells
        self.y_cells_count = int(np.ceil(diff_y/self.tile_size)) 
        self.z_cells_count = int(np.ceil(diff_z/self.tile_size)) 
        self.bin_counts= bin_counts_to_numpy(np.array(blocking_mesh.points), np.array(blocking_mesh.cells[0].data), self.y_cells_count, self.z_cells_count, self.min_y, self.tile_size, self.min_z)
        self.binned_triangles_count = np.sum(self.bin_counts)
        self.binned_triangles = bin_triangles_to_numpy(np.array(blocking_mesh.points), np.array(blocking_mesh.cells[0].data), self.y_cells_count, self.z_cells_count, self.min_y, self.tile_size, self.min_z, self.bin_counts, self.binned_triangles_count)
    def calculate_scattering(self, source_mesh, sink_mesh, alpha, beta,wavelength,self_to_self, chunk_count = 1):
        assert chunk_count > 0, "chunk_count must be greater than 0"
        source_start = 0
        source_end = int(np.floor(source_mesh.points.shape[0]/chunk_count))
        return_array = np.zeros((source_mesh.points.shape[0], sink_mesh.points.shape[0],3), dtype=np.complex64)
        for i in range(chunk_count):

            array =  calculate_scattering_tiles(source_mesh.points[source_start:source_end,:],
                        sink_mesh.points,
                        self.triangle_verticies,
                        wavelength,
                        source_mesh.point_data["ex"].real[source_start:source_end],
                        source_mesh.point_data["ex"].imag[source_start:source_end],
                        source_mesh.point_data["ey"].real[source_start:source_end],
                        source_mesh.point_data["ey"].imag[source_start:source_end],
                        source_mesh.point_data["ez"].real[source_start:source_end],
                        source_mesh.point_data["ez"].imag[source_start:source_end],
                        np.append(source_mesh.point_data["Normals"][source_start:source_end,:],sink_mesh.point_data["Normals"],axis=0),
                        self.min_x,
                        self.max_x,
                        self.min_y,
                        self.max_y,
                        self.min_z,
                        self.max_z,
                        self.tile_size,
                        self.bin_counts,
                        self.binned_triangles,
                        self.y_cells_count,
                        self.z_cells_count,
                        self.binned_triangles_count,
                        alpha,
                        beta,
                        self_to_self)
            array = np.ascontiguousarray(array)
            array = array.view(np.complex64)
            array = array.reshape((source_end-source_start, sink_mesh.points.shape[0],3))
            return_array[source_start:source_end,:] = array
            source_start = source_end
            source_end = int(source_end + np.floor(source_mesh.points.shape[0]/chunk_count))
            ## the abpve might miss points on the last chunk
            if i == chunk_count - 2:
                source_end = source_mesh.points.shape[0]

        return return_array


class Brute_Force_acceleration_structure:
    """
    This class is used to create a brute force acceleration structure for the scattering calculation.
    Parameters
    ----------
    blocking_mesh : meshio.Mesh
        The mesh to be used for the blocking structure.

    Attributes
    ----------
    triangle_verticies : numpy.ndarray
        The vertices of the triangles in the blocking mesh.
    triangles : numpy.ndarray
        The triangles in the blocking mesh.

    Methods
    -------
    calculate_scattering(source_mesh, sink_mesh, alpha, beta, wavelength, self_to_self, chunk_count=1)
        Calculate the scattering from the source mesh to the sink mesh.

    """
    def __init__(self, blocking_mesh):
        self.triangle_verticies = np.ascontiguousarray(blocking_mesh.points)
        self.triangles =blocking_mesh.cells[0].data
    def calculate_scattering(self, source_mesh, sink_mesh, alpha, beta,wavelength,self_to_self,chunk_count=1):
        assert chunk_count > 0, "chunk_count must be greater than 0"
        source_start = 0
        source_end = source_mesh.points.shape[0]/chunk_count
        return_array = np.zeros((source_mesh.points.shape[0], sink_mesh.points.shape[0],3))
        for i in range(chunk_count):

            array =  calculate_scattering_brute_force(source_mesh.points[source_start:source_end,:],
                sink_mesh.points,
                self.triangle_verticies,
                wavelength,
                source_mesh.point_data["ex"].real,
                source_mesh.point_data["ex"].imag,
                source_mesh.point_data["ey"].real,
                source_mesh.point_data["ey"].imag,
                source_mesh.point_data["ez"].real,
                source_mesh.point_data["ez"].imag,
                source_mesh.point_data["Normals"],
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y,
                self.min_z,
                self.max_z,
                self.tile_size,
                self.bin_counts,
                self.binned_triangles,
                self.y_cells_count,
                self.z_cells_count,
                self.binned_triangles_count,
                alpha,
                beta,
                self_to_self)
            array = np.ascontiguousarray(array)
            array = array.view(np.complex64)
            array = array.reshape((source_end-source_start, sink_mesh.points.shape[0],3))
            source_start = source_end
            source_end = source_end + source_mesh.points.shape[0]/chunk_count
            ## the abpve might miss points on the last chunk
            if i == chunk_count - 2:
                source_end = source_mesh.points.shape[0]
        return return_array
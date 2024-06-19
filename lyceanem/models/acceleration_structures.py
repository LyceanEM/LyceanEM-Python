import numpy as np
from lyceanem.em import bin_counts_to_numpy, bin_triangles_to_numpy
from lyceanem.em import calculate_scattering as calculate_scattering
import meshio

class Tile_acceleration_structure:
    def __init__(self, blocking_mesh, n_cells):
        print("Creating acceleration structure")
        print("blocking mesh points", blocking_mesh)
        self.max_x = np.max(blocking_mesh.points[:, 0])
        self.min_x = np.min(blocking_mesh.points[:, 0])
        self.max_y = np.max(blocking_mesh.points[:, 1])
        self.min_y = np.min(blocking_mesh.points[:, 1])
        self.max_z = np.max(blocking_mesh.points[:, 2])
        self.min_z = np.min(blocking_mesh.points[:, 2])
        print("maxminx", self.max_x, self.min_x)
        print("maxminy", self.max_y, self.min_y)
        print("maxminz", self.max_z, self.min_z)

        diff_y = self.max_y - self.min_y
        diff_z = self.max_z - self.min_z
        diff = min( diff_y, diff_z)
        self.tile_size = diff / n_cells
        self.triangle_verticies = np.array(blocking_mesh.points)
        self.y_cells_count = int(np.floor(diff_y/self.tile_size)) 
        self.z_cells_count = int(np.floor(diff_z/self.tile_size)) 
        print("ncellsy, ncellsz", self.y_cells_count, self.z_cells_count)
        print("diff", diff)
        self.bin_counts= bin_counts_to_numpy(np.array(blocking_mesh.points), np.array(blocking_mesh.cells[0].data), self.y_cells_count, self.z_cells_count, self.min_y, self.tile_size, self.min_z)
        self.binned_triangles_count = np.sum(self.bin_counts)
        self.binned_triangles = bin_triangles_to_numpy(np.array(blocking_mesh.points), np.array(blocking_mesh.cells[0].data), self.y_cells_count, self.z_cells_count, self.min_y, self.tile_size, self.min_z, self.bin_counts, self.binned_triangles_count)
    def calculate_scattering(self, source_mesh, sink_mesh, alpha, beta,wavelength):
        array =  calculate_scattering(source_mesh.points,
            sink_mesh.points,
            self.triangle_verticies,
            wavelength,
            source_mesh.point_data["ex"].real,
            source_mesh.point_data["ex"].imag,
            source_mesh.point_data["ey"].real,
            source_mesh.point_data["ey"].imag,
            source_mesh.point_data["ez"].real,
            source_mesh.point_data["ez"].imag,
            source_mesh.point_data["is_electric"],
            source_mesh.point_data["permittivity"].real,
            source_mesh.point_data["permittivity"].imag,
            source_mesh.point_data["permeability"].real,
            source_mesh.point_data["permeability"].imag,
            source_mesh.point_data["Normals"],
            self.min_x,
            self.min_y,
            self.min_z,
            self.max_x,
            self.max_y,
            self.max_z,
            self.tile_size,
            self.bin_counts,
            self.binned_triangles,
            self.y_cells_count,
            self.z_cells_count,
            self.binned_triangles_count,
            alpha,
            beta)
        return array.reshape((source_mesh.points.shape[0],sink_mesh.points.shape[0],3))
    
class brute_force_acceleration_structure:
    def __init__(self, blocking_mesh):
        self.triangle_verticies = np.array(blocking_mesh.points)
        self.triangles = np.array(blocking_mesh.cells[0].data)
import copy
import numpy as np
from scipy import interpolate as sp
from scipy.spatial.transform import Rotation as R

import open3d as o3d
from . import base_types as base_types
from .electromagnetics import beamforming as BM
from .electromagnetics import empropagation as EM
from .geometry import geometryfunctions as GF
from .raycasting import rayfunctions as RF
from .utility import math_functions as MF
#from .models.frequency_domain import calculate_farfield as farfield_generator
#from .models.frequency_domain import calculate_scattering as frequency_domain_channel

class object3d:
    def __init__(self):
        #define object pose as rotation matrix from world frame using Denavit-Hartenbeg convention
        self.pose=np.eye(4)
    def rotate_euler(self,x,y,z,degrees=True,replace=True):
        rot=R.from_euler('xyz',[x,y,z],degrees=degrees)
        transform=np.eye(4)
        transform[:3,:3]=rot.as_matrix()
        if replace==True:
            self.pose=transform
        else:
            self.pose=np.matmul(self.pose,transform)
        return self.pose

    def rotate_matrix(self,new_axes,replace=True):
        rot=R.from_matrix(new_axes)
        transform=np.eye(4)
        transform[:3,:3]=rot.as_matrix()
        if replace==True:
            self.pose=transform
        else:
            self.pose=np.matmul(self.pose,transform)
        return self.pose


class points(object3d):
    """
    Structure class to store information about the geometry and materials in the environment, holding the seperate
    shapes as :class:`open3d.geometry.TriangleMesh` data structures. Everything in the class will be considered an integrated unit, rotating and moving together.
    This class will be developed to include material parameters to enable more complex modelling.

    Units should be SI, metres

    This is the default class for passing structures to the different models.
    """

    def __init__(self, points=None):
        super().__init__()
        # solids is a list of open3D :class:`open3d.geometry.TriangleMesh` structures
        if points==None:
            #no points provided at creation,
            print("Empty Object Created, please add points")
            self.points=[]
        else:
            self.points = []
            for item in points:
                self.points.append(item)
            # self.materials = []
            # for item in material_characteristics:
            #    self.materials.append(item)

    def remove_points(self, deletion_index):
        """
        removes a component or components from the class

        Parameters
        -----------
        deletion_index : list
            list of integers or numpy array of integers to the solids to be removed

        Returns
        --------
        None
        """
        for entry in range(len(deletion_index)):
            self.points.pop(deletion_index[entry])
            self.materials.pop(deletion_index[entry])

    def add_points(self, new_points):
        """
        adds a component or components from the structure

        Parameters
        -----------
        new_points : :class:`open3d.geometry.PointCloud`
            the point cloud to be added to the point cloud collection

        Returns
        --------
        None

        """
        self.points.append(new_points)
        # self.materials.append(new_materials)
    def create_points(self, points,normals):
        """
        create points within the class based upon the provided numpy arrays of floats in local coordinates

        Parameters
        ----------
        points : numpy 2d array
            the coordinates of all the poitns
        normals : numpy 2d array
            the normal vectors of each point

        Returns
        -------
        None
        """
        new_point_cloud=o3d.geometry.PointCloud()
        new_point_cloud.points=o3d.utility.Vector3dVector(points.reshape(-1,3))
        new_point_cloud.normals = o3d.utility.Vector3dVector(normals.reshape(-1, 3))
        self.add_points(new_point_cloud)

    def rotate_points(
        self, rotation_matrix, rotation_centre=np.zeros((3, 1), dtype=np.float32)
    ):
        """
        rotates the components of the structure around a common point, default is the origin

        Parameters
        ----------
        rotation_matrix : open3d rotation matrix
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector)
        rotation_centre : 1*3 numpy float array
            centre of rotation for the structures

        Returns
        --------
        None
        """
        # warning, current commond just rotates around the origin, and until Open3D can be brought up to the
        # latest version without breaking BlueCrystal reqruiements, this will require additional code.
        for item in range(len(self.points)):
            self.point[item] = GF.open3drotate(
                self.point[item], rotation_matrix, rotation_centre
            )

    def translate_points(self, vector):
        """
        translates the point clouds in the class by the given cartesian vector (x,y,z)

        Parameters
        -----------
        vector : 1*3 numpy array of floats
            The desired translation vector for the structures

        Returns
        --------
        None
        """
        for item in range(len(self.points)):
            self.points[item].translate(vector)

    def export_points(self,point_index=None):
        """
        combines all the points in the collection as a combined point cloud for modelling

        Returns
        -------
        combined points
        """
        if point_index==None:
            combined_points = o3d.geometry.PointCloud()
            for item in range(len(self.points)):
                combined_points = combined_points + self.points[item]

            combined_points.transform(self.pose)

        else:
            combined_points = o3d.geometry.PointCloud()
            for item in point_index:
                combined_points = combined_points + self.points[item]

            combined_points.transform(self.pose)

        return combined_points


class structures(object3d):
    """
    Structure class to store information about the geometry and materials in the environment, holding the seperate
    shapes as :class:`open3d.geometry.TriangleMesh` data structures. Everything in the class will be considered an integrated unit, rotating and moving together.
    This class will be developed to include material parameters to enable more complex modelling.

    Units should be SI, metres

    This is the default class for passing structures to the different models.
    """

    def __init__(self, solids=None):
        super().__init__()
        # solids is a list of open3D :class:`open3d.geometry.TriangleMesh` structures
        if solids == None:
            # no points provided at creation,
            print("Empty Object Created, please add solids")
            self.solids = []
        else:
            self.solids = []
            for item in range(len(solids)):
                self.solids.append(solids[item])
            # self.materials = []
            # for item in material_characteristics:
            #    self.materials.append(item)

    def remove_structure(self, deletion_index):
        """
        removes a component or components from the class

        Parameters
        -----------
        deletion_index : list
            list of integers or numpy array of integers to the solids to be removed

        Returns
        --------
        None
        """
        for entry in range(len(deletion_index)):
            self.solids.pop(deletion_index[entry])
            self.materials.pop(deletion_index[entry])

    def add_structure(self, new_solids):
        """
        adds a component or components from the structure

        Parameters
        -----------
        new_solids : :class:`open3d.geometry.TriangleMesh`
            the solid to be added to the structure

        Returns
        --------
        None

        """
        self.solids.append(new_solids)
        # self.materials.append(new_materials)

    def rotate_structures(
        self, rotation_matrix, rotation_centre=np.zeros((3, 1), dtype=np.float32)
    ):
        """
        rotates the components of the structure around a common point, default is the origin

        Parameters
        ----------
        rotation_matrix : open3d rotation matrix
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector)
        rotation_centre : 1*3 numpy float array
            centre of rotation for the structures

        Returns
        --------
        None
        """

        for item in range(len(self.solids)):
            self.solids[item] = GF.open3drotate(
                self.solids[item], rotation_matrix, rotation_centre
            )

    def translate_structures(self, vector):
        """
        translates the structures in the class by the given cartesian vector (x,y,z)

        Parameters
        -----------
        vector : 1*3 numpy array of floats
            The desired translation vector for the structures

        Returns
        --------
        None
        """
        for item in range(len(self.solids)):
            self.solids[item].translate(vector,relative=True)

    def export_vertices(self, structure_index=None):
        """
        Exports the vertices for either all or the indexed point clouds, transformed to the global coordinate frame.

        Parameters
        ----------
        structure_index : list
            list of structures of interest for vertices export

        Returns
        -------
        point_cloud :
        """
        point_cloud = o3d.geometry.PointCloud()
        if structure_index == None:
            # if no structure index is provided, then generate an index including all items in the class
            structure_index = []
            for item in range(len(self.solids)):
                structure_index.append(item)

        for item in structure_index:
            if self.solids[item] == None:
                # do nothing
                temp_cloud = o3d.geometry.PointCloud()
            else:
                temp_cloud = o3d.geometry.PointCloud()
                temp_cloud.points = self.solids[item].vertices
                if self.solids[item].has_vertex_normals():
                    temp_cloud.normals = self.solids[item].vertex_normals
                else:
                    self.solids[item].compute_vertex_normals()
                    temp_cloud.normals = self.solids[item].vertex_normals

                point_cloud = point_cloud + temp_cloud

        point_cloud.transform(self.pose)
        return point_cloud

    def import_cad(self,posixpath,scale=1.0,center=np.zeros((3,1))):
        """
        Import trianglemesh from cad format, with scalling factor is requried. Open3d supports .ply, .stl, .obj, .off, and .gltf/.glb files
        Parameters
        ----------
        posixpath
            file address to cad file to import
        scale
            scalling factor to account for units which are not SI.

        Returns
        -------

        """
        new_solid=o3d.io.read_triangle_mesh(posixpath.as_posix())
        new_solid.compute_triangle_normals()
        new_solid.scale(scale,center=center)
        self.add_structure(new_solid)
    def triangles_base_raycaster(self):
        """
        generates the triangles for all the :class:`open3d.geometry.TriangleMesh` objects in the structure, and outputs them as a continuous array of
        triangle_t format triangles

        Parameters
        -----------
        None

        Returns
        --------
        triangles : N by 1 numpy array of triangle_t triangles
            a continuous array of all the triangles in the structure
        """
        triangles = np.empty((0), dtype=base_types.triangle_t)
        for item in range(len(self.solids)):
            temp_object = copy.deepcopy(self.solids[item])
            temp_object.transform(self.pose)
            triangles = np.append(triangles, RF.convertTriangles(temp_object))

        return triangles


class antenna_structures(object3d):
    """
    Dedicated class to store information on a specific antenna, including aperture points
    as :class:`open3d.geometry.PointCloud` data structures, and structure shapes
    as :class:`open3d.geometry.TriangleMesh` data structures. Everything in the class will be considered an integrated
    unit, rotating and moving together. This inherits functions from the structures and points classes.

    This class will be developed to include material parameters to enable more complex modelling.

    Units should be SI, metres

    """

    def __init__(self, structures, points):
        super().__init__()
        self.structures = structures
        self.points = points
        self.antenna_xyz=o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1, origin=self.pose[:3,3]
)


    def rotate_antenna(
        self, rotation_matrix, rotation_centre=np.zeros((3, 1), dtype=np.float32)
    ):
        """
        rotates the components of the structure around a common point, default is the origin within the local coordinate system

        Parameters
        ----------
        rotation_matrix : open3d rotation matrix
            o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rotation_vector)
        rotation_centre : 1*3 numpy float array
            centre of rotation for the structures

        Returns
        --------
        None
        """
        self.antenna_xyz=GF.open3drotate(self.antenna_xyz,rotation_matrix, rotation_centre
                )
        #translate to the rotation centre, then rotate around origin
        temp_origin=self.pose[:3,3].ravel()-rotation_centre.ravel()
        temp_origin=np.matmul(rotation_matrix,temp_origin)
        self.pose[:3,3]=temp_origin.ravel()+rotation_centre.ravel()
        self.pose[:3,:3]=np.matmul(rotation_matrix,self.pose[:3,:3])
        for item in range(len(self.structures.solids)):
            if (self.structures.solids[item] is not None):
                self.structures.solids[item] = GF.open3drotate(
                    self.structures.solids[item], rotation_matrix, rotation_centre
                )
        for item in range(len(self.points.points)):
            if (self.points.points[item] is not None):
                self.points.points[item] = GF.open3drotate(
                    self.points.points[item], rotation_matrix, rotation_centre
                )

    def translate_antenna(self, vector):
        """
        translates the structures and points in the class by the given cartesian vector (x,y,z) within the local coordinate system

        Parameters
        -----------
        vector : 1*3 numpy array of floats
            The desired translation vector for the antenna

        Returns
        --------
        None
        """

        self.pose[:3,3]=self.pose[:3,3].ravel()+vector.ravel()
        #for item in range(len(self.structures.solids)):
        #    self.structures.solids[item].translate(vector)
        #for item in range(len(self.points.points)):
        #    self.points.points[item].translate(vector)

    def export_all_points(self,point_index=None):
        if point_index==None:
            point_cloud = self.points.export_points()
            point_cloud.transform(self.pose) ##+ self.structures.export_vertices()
        else:
            point_cloud=self.points.export_points(point_index=point_index)
            point_cloud.transform(self.pose)

        return point_cloud

    def excitation_function(self,desired_e_vector,transmit_amplitude=1,point_index=None,phase_shift="none",wavelength=1.0,steering_vector = np.zeros((1, 3))):
        #generate the local excitation function and then convert into the global coordinate frame.
        if point_index==None:
            aperture_points=self.export_all_points()
        else:
            aperture_points = self.export_all_points(point_index=point_index)
        aperture_weights = EM.calculate_conformalVectors(
            desired_e_vector, np.asarray(aperture_points.normals), self.pose[:3,:3]
        )
        if phase_shift=="wavefront":
            source_points = np.asarray(aperture_points.points)
            phase_weights= BM.WavefrontWeights(source_points, steering_vector, wavelength)
            aperture_weights=aperture_weights*phase_weights.reshape(-1,1)

        return aperture_weights*transmit_amplitude

    def receive_transform(self,aperture_polarisation,excitation_function,beamforming_weights):
        #combine local aperture polarisation, received excitation function, and beamforming weights to calculate the received signal
        #convert global polarisation functions to local ones
        transform_R = R.from_matrix(self.pose[:3, :3])
        transform_global_to_local = transform_R.inv()
        local_signal = np.matmul(excitation_function, transform_global_to_local.as_matrix())
        local_received_signal = np.zeros((excitation_function.shape[1], 3))
        received_signal = np.zeros((excitation_function.shape[1], 3))
        local_received_signal[:, 0] = np.dot(beamforming_weights, local_signal[:, :, 0])
        local_received_signal[:, 1] = np.dot(beamforming_weights, local_signal[:, :, 1])
        local_received_signal[:, 2] = np.dot(beamforming_weights, local_signal[:, :, 2])
        received_signal = np.dot(local_received_signal, aperture_polarisation.ravel())

        return received_signal

    def export_all_structures(self):

        objects = copy.deepcopy(self.structures.solids)
        for item in range(len(objects)):
            if objects[item] is None:
                print("Structure does not exist")
            else:
                objects[item]=objects[item].transform(self.pose)

        return objects
    def farfield_distance(self, freq):
        # calculate farfield distance for the antenna, based upon the Fraunhofer distance
        total_points = self.export_all_points()
        # calculate bounding box
        bounding_box = total_points.get_oriented_bounding_box()
        max_points = bounding_box.get_max_bound()
        min_points = bounding_box.get_min_bound()
        center = bounding_box.get_center()
        max_dist = np.sqrt(
            (max_points[0] - center[0]) ** 2
            + (max_points[1] - center[1]) ** 2
            + (max_points[2] - center[2]) ** 2
        )
        min_dist = np.sqrt(
            (center[0] - min_points[0]) ** 2
            + (center[1] - min_points[1]) ** 2
            + (center[2] - min_points[2]) ** 2
        )
        a = np.mean([max_dist, min_dist])
        wavelength = 3e8 / freq
        farfield_distance = (2 * (2 * a) ** 2) / wavelength

        return farfield_distance

    def estimate_maximum_directivity(self, freq):
        # estimate the maximum possible directivity based upon the smallest enclosing sphere and Hannan's relation between projected area and directivity
        total_points = self.export_all_points()
        # calculate bounding box
        bounding_box = total_points.get_oriented_bounding_box()
        max_points = bounding_box.get_max_bound()
        min_points = bounding_box.get_min_bound()
        center = bounding_box.get_center()
        max_dist = np.sqrt(
            (max_points[0] - center[0]) ** 2
            + (max_points[1] - center[1]) ** 2
            + (max_points[2] - center[2]) ** 2
        )
        min_dist = np.sqrt(
            (center[0] - min_points[0]) ** 2
            + (center[1] - min_points[1]) ** 2
            + (center[2] - min_points[2]) ** 2
        )
        a = np.mean([max_dist, min_dist])
        projected_area = np.pi * (a ** 2)
        wavelength = 3e8 / freq
        directivity = (4 * np.pi * projected_area) / (wavelength ** 2)
        return directivity

    def visualise_antenna(self,extras=[],origin=True):
        """
        This function uses open3d to display the antenna structure in the local coordinate frame
        """
        total_points = self.export_all_points()
        # calculate bounding box
        if np.asarray(total_points.points).shape[0]<4:
            max_dist=2.0
            min_dist=0.0
        else:
            bounding_box = total_points.get_oriented_bounding_box()
            max_points = bounding_box.get_max_bound()
            min_points = bounding_box.get_min_bound()
            center = bounding_box.get_center()
            max_dist = np.sqrt(
                (max_points[0] - center[0]) ** 2
                + (max_points[1] - center[1]) ** 2
                + (max_points[2] - center[2]) ** 2
            )
            min_dist = np.sqrt(
                (center[0] - min_points[0]) ** 2
                + (center[1] - min_points[1]) ** 2
                + (center[2] - min_points[2]) ** 2
            )
        a = np.mean([max_dist, min_dist])
        #scale antenna xyz to the structures
        self.antenna_xyz=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5*a, origin=[0, 0, 0])
        if origin:
            o3d.visualization.draw(
                self.points.points + self.structures.solids + [self.antenna_xyz] + extras
            )
        else:
            o3d.visualization.draw(
                self.points.points + self.structures.solids + extras
            )

    # def generate_farfield(self,excitation_vector,wavelength,elements=False,azimuth_resolution=37,elevation_resolution=37):
    #
    #
    #       if elements==False:
    #           resultant_pattern=antenna_pattern(azimuth_resolution=azimuth_resolution,elevation_resolution=elevation_resolution)
    #           resultant_pattern.pattern[:,:,0], resultant_pattern.pattern[:,:,1] =farfield_generator(self.points.export_points(),
    #                                                self.structures,
    #                                                excitation_vector,
    #                                                az_range=np.linspace(-180, 180, resultant_pattern.azimuth_resolution),
    #                                                el_range=np.linspace(-90, 90, resultant_pattern.elevation_resolution),
    #                                                wavelength=wavelength,
    #                                                farfield_distance=20,
    #                                                project_vectors=True,
    #                                                antenna_axes=self.antenna_axes,elements=elements)
    #       else:
    #           #generate antenna array pattern
    #           array_points=self.points.export_points()
    #           num_elements=np.asarray(np.asarray(array_points.points).shape[0])
    #           resultant_pattern = array_pattern(elements=num_elements,azimuth_resolution=azimuth_resolution,elevation_resolution=elevation_resolution)
    #           resultant_pattern.pattern[:,:, :, 0], resultant_pattern.pattern[:,:, :, 1] = farfield_generator(
    #               self.points.export_points(),
    #               self.structures,
    #               excitation_vector,
    #               az_range=np.linspace(-180, 180, resultant_pattern.azimuth_resolution),
    #               el_range=np.linspace(-90, 90, resultant_pattern.elevation_resolution),
    #               wavelength=wavelength,
    #               farfield_distance=20,
    #               project_vectors=True,
    #               antenna_axes=self.antenna_axes, elements=elements)
    #
    #       return resultant_pattern
    #
    # def frequency_domain_channel(self,sink_coords,excitation_function,wavelength=1.0,scatter_points=None,scattering=1,elements=True,):
    #
    #      Ex,Ey,Ez=frequency_domain_channel(self.points.export_points(),
    #                                    sink_coords,self.structures,excitation_function,scatter_points,wavelength,scattering,elements,project_vectors=True,antenna_axes=self.antenna_axes)
    #      return Ex,Ey,Ez

class antenna_pattern(object3d):
    """
    Antenna Pattern class which allows for patterns to be handled consistently
    across LyceanEM and other modules. The definitions assume that the pattern axes
    are consistent with the global axes set. If a different orientation is required,
    such as a premeasured antenna in a new orientation then the pattern rotate_function
    must be used.

    Antenna Pattern Frequency is in Hz
    Rotation Offset is Specified in terms of rotations around the x, y, and z axes as roll,pitch/elevation, and azimuth
    in radians.
    """

    def __init__(
        self,
        azimuth_resolution=37,
        elevation_resolution=37,
        pattern_frequency=1e9,
        arbitary_pattern=False,
        arbitary_pattern_type="isotropic",
        arbitary_pattern_format="Etheta/Ephi",
        file_location=None
    ):
        super().__init__()

        if file_location!=None:
            self.pattern_frequency = pattern_frequency
            self.arbitary_pattern_format = arbitary_pattern_format
            self.import_pattern(file_location)
            self.field_radius = 1.0
        else:
            self.azimuth_resolution = azimuth_resolution
            self.elevation_resolution = elevation_resolution
            self.pattern_frequency = pattern_frequency
            self.arbitary_pattern_type = arbitary_pattern_type
            self.arbitary_pattern_format = arbitary_pattern_format
            self.field_radius = 1.0
            az_mesh, elev_mesh = np.meshgrid(
                np.linspace(-180, 180, self.azimuth_resolution),
                np.linspace(-90, 90, self.elevation_resolution),
            )
            self.az_mesh = az_mesh
            self.elev_mesh = elev_mesh
            if self.arbitary_pattern_format == "Etheta/Ephi":

                self.pattern = np.zeros(
                    (self.elevation_resolution, self.azimuth_resolution, 2),
                    dtype=np.complex64,
                )

            elif self.arbitary_pattern_format == "ExEyEz":

                self.pattern = np.zeros(
                    (self.elevation_resolution, self.azimuth_resolution, 3),
                    dtype=np.complex64,
                )

            if arbitary_pattern == True:
                self.initilise_pattern()

    def initilise_pattern(self):
        """
        pattern initialisation function, providing an isotopic pattern
        or quasi-isotropic pattern

        Returns
        -------
        Populated antenna pattern
        """

        if self.arbitary_pattern_type == "isotropic":
            self.pattern[:, :, 0] = 1.0
        elif self.arbitary_pattern_type == "xhalfspace":
            az_angles = self.az_mesh[0, :]
            az_index = np.where(np.abs(az_angles) < 90)
            self.pattern[:, az_index, 0] = 1.0
        elif self.arbitary_pattern_type == "yhalfspace":
            az_angles = self.az_mesh[0, :]
            az_index = np.where(az_angles > 90)
            self.pattern[:, az_index, 0] = 1.0
        elif self.arbitary_pattern_type == "zhalfspace":
            elev_angles = self.elev_mesh[:, 0]
            elev_index = np.where(elev_angles > 0)
            self.pattern[elev_index, :, 0] = 1.0



    def import_pattern(self, file_location):
        """
        takes the file location and imports the individual pattern file, replacing exsisting values with those of the saved file.

        It is import to note that for CST ASCII export format, that you select a plot range of -180 to 180 for phi, and that by defauly CST exports from 0 to 180 in theta, which
        is the opposite direction to the default for LyceanEM, so this data structures are flipped for consistency.

        Parameters
        -----------
        file location : :class:`pathlib.Path`
            file location

        Returns
        ---------
        None
        """
        if file_location.suffix == ".txt":
            # file is a CST ffs format
            datafile = np.loadtxt(file_location, skiprows=2)
            theta = datafile[:, 0]
            phi = datafile[:, 1]
            freq_index1 = file_location.name.find("f=")
            freq_index2 = file_location.name.find(")")
            file_frequency = (
                float(file_location.name[freq_index1 + 2 : freq_index2]) * 1e9
            )
            phi_steps = np.linspace(np.min(phi), np.max(phi), np.unique(phi).size)
            theta_steps = np.linspace(
                np.min(theta), np.max(theta), np.unique(theta).size
            )
            phi_res = np.unique(phi).size
            theta_res = np.unique(theta).size
            etheta = (
                (datafile[:, 3] * np.exp(1j * np.deg2rad(datafile[:, 4])))
                .reshape(phi_res, theta_res)
                .transpose()
            )
            ephi = (
                (datafile[:, 5] * np.exp(1j * np.deg2rad(datafile[:, 6])))
                .reshape(phi_res, theta_res)
                .transpose()
            )
            self.azimuth_resolution = phi_res
            self.elevation_resolution = theta_res
            self.elev_mesh = np.flipud(
                GF.thetatoelevation(theta).reshape(phi_res, theta_res).transpose()
            )
            self.az_mesh = np.flipud(phi.reshape(phi_res, theta_res).transpose())
            self.pattern = np.zeros(
                (self.elevation_resolution, self.azimuth_resolution, 2),
                dtype=np.complex64,
            )
            self.pattern[:, :, 0] = np.flipud(etheta)
            self.pattern[:, :, 1] = np.flipud(ephi)
            self.pattern_frequency = file_frequency
        elif file_location.suffix == ".dat":
            # file is .dat format from anechoic chamber measurements
            Ea, Eb, freq, norm, theta_values, phi_values = EM.importDat(file_location)
            az_mesh, elev_mesh = np.meshgrid(
                phi_values, GF.thetatoelevation(theta_values)
            )
            self.azimuth_resolution = np.unique(phi_values).size
            self.elevation_resolution = np.unique(theta_values).size
            self.az_mesh = az_mesh
            self.elev_mesh = elev_mesh
            self.pattern = np.zeros(
                (self.elevation_resolution, self.azimuth_resolution, 2),
                dtype=np.complex64,
            )
            self.pattern_frequency = freq * 1e6
            self.pattern[:, :, 0] = Ea.transpose() + norm
            self.pattern[:, :, 1] = Eb.transpose() + norm

    def export_pattern(self, file_location):
        """
        takes the file location and exports the pattern as a .dat file
        unfinished, must be in Etheta/Ephi format,

        Parameters
        -----------
        file_location : :class:`pathlib.Path`
            the path for the output file, including name

        Returns
        --------
        None
        """
        if self.arbitary_pattern_format == "ExEyEz":
            self.transmute_pattern()

        theta_flat = GF.elevationtotheta(self.elev_mesh).transpose().reshape(-1, 1)
        phi_mesh = self.az_mesh.transpose().reshape(-1, 1)
        planes = self.azimuth_resolution
        copolardb = 20 * np.log10(
            np.abs(self.pattern[:, :, 0].transpose().reshape(-1, 1))
        )
        copolarangle = np.degrees(
            np.angle(self.pattern[:, :, 0].transpose().reshape(-1, 1))
        )
        crosspolardb = 20 * np.log10(
            np.abs(self.pattern[:, :, 1].transpose().reshape(-1, 1))
        )
        crosspolarangle = np.degrees(
            np.angle(self.pattern[:, :, 1].transpose().reshape(-1, 1))
        )
        norm = np.nanmax(np.array([np.nanmax(copolardb), np.nanmax(crosspolardb)]))
        copolardb -= norm
        crosspolardb -= norm
        infoarray = np.array(
            [
                planes,
                np.min(self.az_mesh),
                np.max(self.az_mesh),
                norm,
                self.pattern_frequency / 1e6,
            ]
        ).reshape(1, -1)

        dataarray = np.concatenate(
            (
                theta_flat.reshape(-1, 1),
                copolardb.reshape(-1, 1),
                copolarangle.reshape(-1, 1),
                crosspolardb.reshape(-1, 1),
                crosspolarangle.reshape(-1, 1),
            ),
            axis=1,
        )
        outputarray = np.concatenate((infoarray, dataarray), axis=0)
        np.savetxt(file_location, outputarray, delimiter=",", fmt="%.2e")

    def export_global_weights(self):
        """
        Calculates the global frame coordinates and xyz weightings of the antenna pattern using the pattern pose to translate and rotate into the correction position and orientation within the model environment. If this function is used as a source for models, then the antenna pattern pose must be updated for the desired position and orientation before export.

        Returns
        -------
        points :

        weights :

        """
        xyz = self.cartesian_points()
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(xyz)
        points.normals = o3d.utility.Vector3dVector(xyz)
        points.transform(self.pose)

        if self.arbitary_pattern_format == "Etheta/Ephi":
            self.transmute_pattern(desired_format="ExEyEz")

        #import to export the weights in cartesian format, then orientate correctly within global reference frame.
        rotation_matrix = self.pose[:3, :3]

        #weightsx= self.pattern[:,:,0].ravel()
        #weightsy = self.pattern[:, :, 1].ravel()
        #weightsz = self.pattern[:, :, 2].ravel()
        weights=np.array([self.pattern[:,:,0].reshape(-1),self.pattern[:,:,1].reshape(-1),self.pattern[:,:,2].reshape(-1)])
        weights=np.matmul(weights,rotation_matrix)
        return points, weights

    def display_pattern(
        self, plottype="Polar", desired_pattern="both", pattern_min=-40, plot_max=0
    ):
        """
        Displays the Antenna Pattern using :func:`lyceanem.electromagnetics.beamforming.PatternPlot`

        Parameters
        ----------
        plottype : str
            the plot type, either [Polar], [Cartesian-Surf], or [Contour]. The default is [Polar]
        desired_pattern : str
            the desired pattern, default is [both], but is Pattern format is 'Etheta/Ephi' then options are [Etheta] or [Ephi], and if Pattern format is 'ExEyEz', then options are [Ex], [Ey], or [Ez].
        pattern_min : float
            the desired scale minimum in dB, the default is [-40]

        Returns
        -------
        None
        """

        if self.arbitary_pattern_format == "Etheta/Ephi":
            if desired_pattern == "both":
                BM.PatternPlot(
                    self.pattern[:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Etheta",
                    plot_max=plot_max,
                )
                BM.PatternPlot(
                    self.pattern[:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ephi",
                    plot_max=plot_max,
                )
            elif desired_pattern == "Etheta":
                BM.PatternPlot(
                    self.pattern[:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Etheta",
                    plot_max=plot_max,
                )
            elif desired_pattern == "Ephi":
                BM.PatternPlot(
                    self.pattern[:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ephi",
                    plot_max=plot_max,
                )
            elif desired_pattern =="Power":
                BM.PatternPlot(
                    self.pattern[:, :, 0]**2+self.pattern[:, :, 1]**2,
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    logtype='power',
                    plottype=plottype,
                    title_text="Power Pattern",
                    plot_max=plot_max,
                )
        elif self.arbitary_pattern_format == "ExEyEz":
            if desired_pattern == "both":
                BM.PatternPlot(
                    self.pattern[:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ex",
                    plot_max=plot_max,
                )
                BM.PatternPlot(
                    self.pattern[:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ey",
                    plot_max=plot_max,
                )
                BM.PatternPlot(
                    self.pattern[:, :, 2],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ez",
                    plot_max=plot_max,
                )
            elif desired_pattern == "Ex":
                BM.PatternPlot(
                    self.pattern[:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ex",
                    plot_max=plot_max,
                )
            elif desired_pattern == "Ey":
                BM.PatternPlot(
                    self.pattern[:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ey",
                    plot_max=plot_max,
                )
            elif desired_pattern == "Ez":
                BM.PatternPlot(
                    self.pattern[:, :, 2],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ez",
                    plot_max=plot_max,
                )

    def transmute_pattern(self,desired_format="Etheta/Ephi"):
        """
        convert the pattern from Etheta/Ephi format to Ex, Ey,Ez format, or back again
        """
        if self.arbitary_pattern_format == "Etheta/Ephi":
            if desired_format=="ExEyEz":
                oldformat = self.pattern.reshape(-1, self.pattern.shape[2])
                old_shape = oldformat.shape
                self.arbitary_pattern_format = "ExEyEz"
                self.pattern = np.zeros(
                    (self.elevation_resolution, self.azimuth_resolution, 3),
                    dtype=np.complex64,
                )
                theta = GF.elevationtotheta(self.elev_mesh)
                # convrsion part, move to transmute pattern
                conversion_matrix1 = np.asarray(
                    [
                        np.cos(np.deg2rad(theta.ravel()))
                        * np.cos(np.deg2rad(self.az_mesh.ravel())),
                        np.cos(np.deg2rad(theta.ravel()))
                        * np.sin(np.deg2rad(self.az_mesh.ravel())),
                        -np.sin(np.deg2rad(theta.ravel())),
                    ]
                ).transpose()
                conversion_matrix2 = np.asarray(
                    [
                        -np.sin(np.deg2rad(self.az_mesh.ravel())),
                        np.cos(np.deg2rad(self.az_mesh.ravel())),
                        np.zeros(self.az_mesh.size),
                    ]
                ).transpose()
                decomposed_fields = (
                    oldformat[:, 0].reshape(-1, 1) * conversion_matrix1
                    + oldformat[:, 1].reshape(-1, 1) * conversion_matrix2
                )
                self.pattern[:, :, 0] = decomposed_fields[:, 0].reshape(
                    self.elevation_resolution, self.azimuth_resolution
                )
                self.pattern[:, :, 1] = decomposed_fields[:, 1].reshape(
                    self.elevation_resolution, self.azimuth_resolution
                )
                self.pattern[:, :, 2] = decomposed_fields[:, 2].reshape(
                    self.elevation_resolution, self.azimuth_resolution
                )
            elif desired_format == "Circular":
                #recalculate pattern using Right Hand Circular and Left Hand Circular Polarisation
                self.arbitary_pattern_format = desired_format

        else:
            if desired_format == "Etheta/Ephi":
                oldformat = self.pattern.reshape(-1, self.pattern.shape[2])
                old_shape = oldformat.shape
                self.arbitary_pattern_format = "Etheta/Ephi"
                self.pattern = np.zeros(
                    (self.elevation_resolution, self.azimuth_resolution, 2),
                    dtype=np.complex64,
                )
                theta = GF.elevationtotheta(self.elev_mesh)
                costhetacosphi = (
                    np.cos(np.deg2rad(self.az_mesh.ravel()))
                    * np.cos(np.deg2rad(theta.ravel()))
                ).astype(np.complex64)
                sinphicostheta = (
                    np.sin(np.deg2rad(self.az_mesh.ravel()))
                    * np.cos(np.deg2rad(theta.ravel()))
                ).astype(np.complex64)
                sintheta = (np.sin(np.deg2rad(theta.ravel()))).astype(np.complex64)
                sinphi = (np.sin(np.deg2rad(self.az_mesh.ravel()))).astype(np.complex64)
                cosphi = (np.cos(np.deg2rad(self.az_mesh.ravel()))).astype(np.complex64)
                new_etheta = (
                    oldformat[:, 0] * costhetacosphi
                    + oldformat[:, 1] * sinphicostheta
                    - oldformat[:, 2] * sintheta
                )
                new_ephi = -oldformat[:, 0] * sinphi + oldformat[:, 1] * cosphi
                self.pattern[:, :, 0] = new_etheta.reshape(
                    self.elevation_resolution, self.azimuth_resolution
                )
                self.pattern[:, :, 1] = new_ephi.reshape(
                    self.elevation_resolution, self.azimuth_resolution
                )

    def cartesian_points(self):
        """
        exports the cartesian points for all pattern points.
        """
        x, y, z = MF.sph2cart(
            np.deg2rad(self.az_mesh.ravel()),
            np.deg2rad(self.elev_mesh.ravel()),
            np.ones((self.pattern[:, :, 0].size)) * self.field_radius,
        )
        field_points = (
            np.array([x, y, z]).transpose().astype(np.float32)
        )

        return field_points

    def rotate_pattern(self, rotation_matrix=None):
        """
        Rotate the self pattern from the assumed global axes into the new direction

        Parameters
        ----------
        new_axes : 3x3 numpy float array
            the new vectors for the antenna x,y,z axes

        Returns
        -------
        Updates self.pattern with the new pattern reflecting the antenna
        orientation within the global models

        """
        # generate pattern_coordinates for rotation
        theta = GF.elevationtotheta(self.elev_mesh)
        x, y, z = RF.sph2cart(
            np.deg2rad(self.az_mesh.ravel()),
            np.deg2rad(self.elev_mesh.ravel()),
            np.ones((self.pattern[:, :, 0].size)),
        )
        field_points = np.array([x, y, z]).transpose().astype(np.float32)

        # convert to ExEyEz for rotation
        if self.arbitary_pattern_format == "Etheta/Ephi":
            self.transmute_pattern(desired_format="ExEyEz")
            desired_format = "Etheta/Ephi"
        else:
            desired_format = "ExEyEz"

        rotated_points = np.dot(field_points, rotation_matrix)
        decomposed_fields = self.pattern.reshape(-1, 3)
        # rotation part
        xyzfields = np.dot(decomposed_fields, rotation_matrix)
        # resample
        resampled_xyzfields = self.resample_pattern(
            rotated_points, xyzfields, field_points
        )
        self.pattern[:, :, 0] = resampled_xyzfields[:, 0].reshape(
            self.elevation_resolution, self.azimuth_resolution
        )
        self.pattern[:, :, 1] = resampled_xyzfields[:, 1].reshape(
            self.elevation_resolution, self.azimuth_resolution
        )
        self.pattern[:, :, 2] = resampled_xyzfields[:, 2].reshape(
            self.elevation_resolution, self.azimuth_resolution
        )

        if desired_format == "Etheta/Ephi":
            # convert back
            self.transmute_pattern(desired_format=desired_format)

    def resample_pattern_angular(
        self, new_azimuth_resolution, new_elevation_resolution
    ):
        """
        resample pattern based upon provided azimuth and elevation resolution
        """
        new_az_mesh, new_elev_mesh = np.meshgrid(
            np.linspace(-180, 180, new_azimuth_resolution),
            np.linspace(-90, 90, new_elevation_resolution),
        )
        x, y, z = RF.sph2cart(
            np.deg2rad(new_az_mesh.ravel()),
            np.deg2rad(new_elev_mesh.ravel()),
            np.ones((self.pattern[:, :, 0].size)),
        )
        new_field_points = np.array([x, y, z]).transpose().astype(np.float32)
        old_field_points = self.cartesian_points()
        old_pattern = self.pattern.reshape(-1, 2)
        new_pattern = self.resample_pattern(
            old_field_points, old_pattern, new_field_points
        )

    def resample_pattern(self, old_points, old_pattern, new_points):
        """

        Parameters
        ----------
        old_points : float xyz
            xyz coordinates that the pattern has been sampled at
        old_pattern : 2 or 3 by n complex array of the antenna pattern at the old_poitns
            DESCRIPTION.
        new_points : desired_grid points in xyz float array
            DESCRIPTION.

        Returns
        -------
        new points, new_pattern

        """
        pol_format = old_pattern.shape[-1]  # will be 2 or three
        new_pattern = np.zeros((new_points.shape[0], pol_format), dtype=np.complex64)
        smoothing_factor = 4
        for component in range(pol_format):
            mag_interpolate = sp.Rbf(
                old_points[:, 0],
                old_points[:, 1],
                old_points[:, 2],
                np.abs(old_pattern[:, component]),
                smooth=smoothing_factor,
            )
            phase_interpolate = sp.Rbf(
                old_points[:, 0],
                old_points[:, 1],
                old_points[:, 2],
                np.angle(old_pattern[:, component]),
                smooth=smoothing_factor,
            )
            new_mag = mag_interpolate(
                new_points[:, 0], new_points[:, 1], new_points[:, 2]
            )
            new_angles = phase_interpolate(
                new_points[:, 0], new_points[:, 1], new_points[:, 2]
            )
            new_pattern[:, component] = new_mag * np.exp(1j * new_angles)

        return new_pattern

    def directivity(self):
        """

        Returns
        -------
        Dtheta : numpy array
            directivity for Etheta farfield
        Dphi : numpy array
            directivity for Ephi farfield
        Dtotal : numpy array
            overall directivity pattern
        Dmax : numpy array
            the maximum directivity for each pattern

        """
        Dtheta, Dphi, Dtotal, Dmax = BM.directivity_transformv2(
            self.pattern[:, :, 0],
            self.pattern[:, :, 1],
            az_range=self.az_mesh[0, :],
            elev_range=self.elev_mesh[:, 0],
        )
        return Dtheta, Dphi, Dtotal, Dmax

class array_pattern:
    """
    Array Pattern class which allows for patterns to be handled consistently
    across LyceanEM and other modules. The definitions assume that the pattern axes
    are consistent with the global axes set. If a different orientation is required,
    such as a premeasured antenna in a new orientation then the pattern rotate_function
    must be used.

    Antenna Pattern Frequency is in Hz
    Rotation Offset is Specified in terms of rotations around the x, y, and z axes as roll,pitch/elevation, and azimuth
    in radians.
    """

    def __init__(
        self,
        azimuth_resolution=37,
        elevation_resolution=37,
        pattern_frequency=1e9,
        arbitary_pattern=False,
        arbitary_pattern_type="isotropic",
        arbitary_pattern_format="Etheta/Ephi",
        position_mapping=np.zeros((3), dtype=np.float32),
        rotation_offset=np.zeros((3), dtype=np.float32),
        elements=2,
    ):

        self.azimuth_resolution = azimuth_resolution
        self.elevation_resolution = elevation_resolution
        self.pattern_frequency = pattern_frequency
        self.arbitary_pattern_type = arbitary_pattern_type
        self.arbitary_pattern_format = arbitary_pattern_format
        self.position_mapping = position_mapping
        self.rotation_offset = rotation_offset
        self.field_radius = 1.0
        self.elements=elements
        self.beamforming_weights=np.ones((self.elements),dtype=np.complex64)
        az_mesh, elev_mesh = np.meshgrid(
            np.linspace(-180, 180, self.azimuth_resolution),
            np.linspace(-90, 90, self.elevation_resolution),
        )
        self.az_mesh = az_mesh
        self.elev_mesh = elev_mesh
        if self.arbitary_pattern_format == "Etheta/Ephi":
            self.pattern = np.zeros(
                ( elements,self.elevation_resolution, self.azimuth_resolution, 2),
                dtype=np.complex64,
            )
        elif self.arbitary_pattern_format == "ExEyEz":

            self.pattern = np.zeros(
                ( elements,self.elevation_resolution, self.azimuth_resolution, 3),
                dtype=np.complex64,
            )
        if arbitary_pattern == True:
            self.initilise_pattern()

    def _rotation_matrix(self):
        """_rotation_matrix getter method

        Calculates and returns the (3D) axis rotation matrix.

        Returns
        -------
        : :class:`numpy.ndarray` of shape (3, 3)
            The model (3D) rotation matrix.
        """
        x_rot = R.from_euler("X", self.rotation_offset[0], degrees=True).as_matrix()
        y_rot = R.from_euler("Y", self.rotation_offset[1], degrees=True).as_matrix()
        z_rot = R.from_euler("Z", self.rotation_offset[2], degrees=True).as_matrix()

        # x_rot=np.array([[1,0,0],
        #                [0,np.cos(np.deg2rad(self.rotation_offset[0])),-np.sin(np.deg2rad(self.rotation_offset[0]))],
        #                [0,np.sin(np.deg2rad(self.rotation_offset[0])),np.cos(np.deg2rad(self.rotation_offset[0]))]])

        total_rotation = np.dot(np.dot(z_rot, y_rot), x_rot)
        return total_rotation

    def initilise_pattern(self):
        """
        pattern initialisation function, providing an isotopic pattern
        or quasi-isotropic pattern

        Returns
        -------
        Populated antenna pattern
        """
        if self.arbitary_pattern_type == "isotropic":
            self.pattern[:, :,:, 0] = 1.0
        elif self.arbitary_pattern_type == "xhalfspace":
            az_angles = self.az_mesh[0, :]
            az_index = np.where(np.abs(az_angles) < 90)
            self.pattern[:, az_index,:, 0] = 1.0
        elif self.arbitary_pattern_type == "yhalfspace":
            az_angles = self.az_mesh[0, :]
            az_index = np.where(az_angles > 90)
            self.pattern[:, az_index, :,0] = 1.0
        elif self.arbitary_pattern_type == "zhalfspace":
            elev_angles = self.elev_mesh[:, 0]
            elev_index = np.where(elev_angles > 0)
            self.pattern[elev_index, :, :,0] = 1.0

    def display_pattern(
        self, plottype="Polar", desired_pattern="both", pattern_min=-40
    ):
        """
        Displays the Antenna Array Pattern using :func:`lyceanem.electromagnetics.beamforming.PatternPlot` and the stored weights

        Parameters
        ----------
        plottype : str
            the plot type, either [Polar], [Cartesian-Surf], or [Contour]. The default is [Polar]
        desired_pattern : str
            the desired pattern, default is [both], but is Pattern format is 'Etheta/Ephi' then options are [Etheta] or [Ephi], and if Pattern format is 'ExEyEz', then options are [Ex], [Ey], or [Ez].
        pattern_min : float
            the desired scale minimum in dB, the default is [-40]

        Returns
        -------
        None
        """

        if self.arbitary_pattern_format == "Etheta/Ephi":
            if desired_pattern == "both":
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Etheta",
                )
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ephi",
                )
            elif desired_pattern == "Etheta":
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Etheta",
                )
            elif desired_pattern == "Ephi":
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ephi",
                )
            elif desired_pattern =="Power":
                BM.PatternPlot(
                    (self.beamforming_weights*self.pattern[:,:, :, 0])**2+(self.beamforming_weights*self.pattern[:,:, :, 1])**2,
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Power Pattern",
                )
        elif self.arbitary_pattern_format == "ExEyEz":
            if desired_pattern == "both":
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ex",
                )
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ey",
                )
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 2],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ez",
                )
            elif desired_pattern == "Ex":
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ex",
                )
            elif desired_pattern == "Ey":
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ey",
                )
            elif desired_pattern == "Ez":
                BM.PatternPlot(
                    self.beamforming_weights*self.pattern[:,:, :, 2],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ez",
                )


    def cartesian_points(self):
        """
        exports the cartesian points for all pattern points.
        """
        x, y, z = RF.sph2cart(
            np.deg2rad(self.az_mesh.ravel()),
            np.deg2rad(self.elev_mesh.ravel()),
            np.ones((self.pattern[:, :, 0].size)) * self.field_radius,
        )
        field_points = (
            np.array([x, y, z]).transpose().astype(np.float32)
        )

        return field_points

    def directivity(self):
        """

        Returns
        -------
        Dtheta : numpy array
            directivity for Etheta farfield
        Dphi : numpy array
            directivity for Ephi farfield
        Dtotal : numpy array
            overall directivity pattern
        Dmax : numpy array
            the maximum directivity for each pattern

        """
        Dtheta, Dphi, Dtotal, Dmax = BM.directivity_transformv2(
            self.beamforming_weights*self.pattern[:,:, :, 0],
            self.beamforming_weights*self.pattern[:,:, :, 1],
            az_range=self.az_mesh[0, :],
            elev_range=self.elev_mesh[:, 0],
        )
        return Dtheta, Dphi, Dtotal, Dmax

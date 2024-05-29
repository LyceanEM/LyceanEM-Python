import copy

import numpy as np
import meshio
from scipy import interpolate as sp
from scipy.spatial.transform import Rotation as R
import pyvista as pv

from . import base_types as base_types
from .electromagnetics import beamforming as BM
from .electromagnetics import empropagation as EM
from .geometry import geometryfunctions as GF
from .raycasting import rayfunctions as RF
from .utility import math_functions as MF


# from .models.frequency_domain import calculate_farfield as farfield_generator
# from .models.frequency_domain import calculate_scattering as frequency_domain_channel


class object3d:
    def __init__(self):
        # define object pose as rotation matrix from world frame using Denavit-Hartenbeg convention
        self.pose = np.eye(4)

    def rotate_euler(self, x, y, z, degrees=True, replace=True):
        rot = R.from_euler("xyz", [x, y, z], degrees=degrees)
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        if replace == True:
            self.pose = transform
        else:
            self.pose = np.matmul(self.pose, transform)
        return self.pose

    def rotate_matrix(self, new_axes, replace=True):
        rot = R.from_matrix(new_axes)
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        if replace == True:
            self.pose = transform
        else:
            self.pose = np.matmul(self.pose, transform)
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
        if points == None:
            # no points provided at creation,
            print("Empty Object Created, please add points")
            self.points = []
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

        def create_points(self, points, normals):
            """
            create points within the class based upon the provided numpy arrays of floats in local coordinates

            Parameters
            points : numpy 2d array
                the coordinates of all the poitns
            normals : numpy 2d array
                the normal vectors of each point

            Returns
            None
            """
            mesh_vertices = points.reshape(-1, 3)
            mesh_normals = normals.reshape(-1, 3)
            new_point_cloud = meshio.Mesh(points=mesh_vertices, cells=[], point_data={"Normals": mesh_normals})

            self.add_points(new_point_cloud)

    def rotate_points(
        self, rotation_vector, rotation_centre=np.zeros((3, 1), dtype=np.float32)
    ):
        """
        rotates the components of the structure around a common point, default is the origin

        Parameters
        ----------
        rotation_matrix : open3d rotation vector
            3,1numpy array
        rotation_centre : 1*3 numpy float array
            centre of rotation for the structures

        Returns
        --------
        None
        """
        # warning, current commond just rotates around the origin, and until Open3D can be brought up to the
        # latest version without breaking BlueCrystal reqruiements, this will require additional code.
        assert (rotation_vector.shape == (3,) or rotation_vector.shape == (3,1) or rotation_vector.shape == (1,3) or rotation_vector.shape == (3,3)), "Rotation vector must be a 3x1 or 3x3 array"
        for item in range(len(self.points)):
            self.points[item] = GF.mesh_rotate(
                self.points[item], rotation_vector, rotation_centre
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
            self.points[item] = GF.translate_mesh(self.points[item], vector)

    def export_points(self, point_index=None):
        """
        combines all the points in the collection as a combined point cloud for modelling

        Returns
        -------
        combined points
        """
        if point_index == None:
            
            for item in range(len(self.points)):
                if item == 0:
                    points = np.array(self.points[item].points)
                else:
                    points = np.append(points, self.points[item].points, axis=0)
            point_data = {}
            for key in self.points[0].point_data.keys():
                for item in range(len(self.points)):  
                    if item == 0:
                        point_data[key] = np.array(self.points[item].point_data[key])
                    else:
                        point_data_element = np.array(self.points[item].point_data[key])
                        point_data[key] = np.append(point_data[key], point_data_element, axis=0)

            combinded_points = meshio.Mesh(points, cells = [], point_data=point_data)
            combinded_points = GF.mesh_transform(combinded_points, self.pose, False)
            return combinded_points
                    



        else:
            for item in point_index:
                if item == 0:
                    points = np.array(self.points[item].points)
                else:
                    points = np.append(points, self.points[item].points, axis=0)
            point_data = {}
            for key in self.points[0].point_data.keys():
                for item in point_index:  
                    if item == 0:
                        point_data[key] = np.array(self.points[item].point_data[key])
                    else:
                        point_data_element = np.array(self.points[item].point_data[key])
                        point_data[key] = np.append(point_data[key], point_data_element, axis=0)
                

            combinded_points = meshio.Mesh(points, point_data=point_data)
            combinded_points = GF.mesh_transform(combinded_points, self.pose, False)

        
            return combinded_points


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
        rotation_matrix : numpy array of appropriate shape (4,4)
        
        rotation_centre : 1*3 numpy float array
            centre of rotation for the structures

        Returns
        --------
        None
        """


        for item in range(len(self.solids)):
            self.solids[item] = GF.mesh_rotate(
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
            self.solids[item] = GF.translate_mesh(self.solids[item],vector, relative=True)


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
            temp_object = GF.mesh_transform(temp_object, self.pose, False)

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
        #self.antenna_xyz = o3d.geometry.TriangleMesh.create_coordinate_frame(
           # size=1, origin=self.pose[:3, 3]
       # )



    def export_all_points(self, point_index=None):
        if point_index == None:
            point_cloud = self.points.export_points()
        else:
            point_cloud = self.points.export_points(point_index=point_index)

        point_cloud = GF.mesh_transform(point_cloud, self.pose, False)


        return point_cloud

    def excitation_function(
        self,
        desired_e_vector,
        transmit_amplitude=1,
        point_index=None,
        phase_shift="none",
        wavelength=1.0,
        steering_vector=np.zeros((1, 3)),
    ):
        # generate the local excitation function and then convert into the global coordinate frame.
        if point_index == None:
            aperture_points = self.export_all_points()
        else:
            aperture_points = self.export_all_points(point_index=point_index)

        #as export all points imposes the transformation from local to global frame on the points and associated normal vectors, no rotation is required within calculate_conformalVectors
        aperture_weights = EM.calculate_conformalVectors(
            desired_e_vector, aperture_points.point_data['Normals'], np.eye(3)
        )
        if phase_shift == "wavefront":
            source_points = np.asarray(aperture_points.points)
            phase_weights = BM.WavefrontWeights(
                source_points, steering_vector, wavelength
            )
            aperture_weights = aperture_weights * phase_weights.reshape(-1, 1)

        return aperture_weights * transmit_amplitude

    
    def export_all_structures(self):
        objects = copy.deepcopy(self.structures.solids)
        for item in range(len(objects)):
            if objects[item] is None:
                print("Structure does not exist")
            else:
                objects[item] = GF.mesh_transform(objects[item], self.pose, False)

        return objects
    def rotate_antenna(self, rotation_vector):
        self.structures.rotate_structures(rotation_vector)
        self.points.rotate_points(rotation_vector)



  



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
        file_location=None,
    ):
        super().__init__()

        if file_location != None:
            self.pattern_frequency = pattern_frequency
            self.arbitary_pattern_format = arbitary_pattern_format
            antenna_pattern_import_implemented = False
            assert antenna_pattern_import_implemented
            ## needs implementing
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






    def export_pyvista_object(self):
        """
        Return Antenna Pattern as a PyVista Structured Mesh Data Object
        """
        import pyvista as pv
        def cell_bounds(points, bound_position=0.5):
            """
            Calculate coordinate cell boundaries.

            Parameters
            ----------
            points: numpy.ndarray
                One-dimensional array of uniformly spaced values of shape (M,).

            bound_position: bool, optional
                The desired position of the bounds relative to the position
                of the points.

            Returns
            -------
            bounds: numpy.ndarray
                Array of shape (M+1,)

            Examples
            --------
            >>> a = np.arange(-1, 2.5, 0.5)
            >>> a
            array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
            >>> cell_bounds(a)
            array([-1.25, -0.75, -0.25,  0.25,  0.75,  1.25,  1.75,  2.25])
            """
            if points.ndim != 1:
                raise ValueError("Only 1D points are allowed.")
            diffs = np.diff(points)
            delta = diffs[0] * bound_position
            bounds = np.concatenate([[points[0] - delta], points + delta])
            return bounds

        self.transmute_pattern(desired_format="ExEyEz")
        #hack for cst patterns in spatial intelligence project
        ex=self.pattern[:,:,0]
        ey=self.pattern[:,:,1]
        ez=self.pattern[:,:,2]
        az=np.linspace(0,360,self.azimuth_resolution)
        elevation=np.linspace(-90,90,self.elevation_resolution)
        xx_bounds = cell_bounds(az)
        yy_bounds = cell_bounds(90-elevation)
        self.transmute_pattern(desired_format="Etheta/Ephi")
        et=self.pattern[:,:,0]
        ep=self.pattern[:,:,1]
        vista_pattern=pv.grid_from_sph_coords(xx_bounds, yy_bounds,  self.field_radius)
        vista_pattern['Real']=np.real(np.append(np.append(ex.transpose().ravel().reshape(ex.size,1),
                                                          ey.transpose().ravel().reshape(ex.size,1),axis=1),
                                                ez.transpose().ravel().reshape(ex.size,1),axis=1))
        vista_pattern['Imag']=np.imag(np.append(np.append(ex.transpose().ravel().reshape(ex.size,1),
                                                          ey.transpose().ravel().reshape(ex.size,1),axis=1),
                                                ez.transpose().ravel().reshape(ex.size,1),axis=1))
        vista_pattern['E(theta) Magnitude']=np.abs(et.transpose().ravel())
        vista_pattern['E(theta) Phase']=np.angle(et.transpose().ravel())
        vista_pattern['E(phi) Magnitude']=np.abs(ep.transpose().ravel())
        vista_pattern['E(phi) Phase']=np.angle(ep.transpose().ravel())
        vista_pattern['Magnitude']=np.abs(vista_pattern['Real']+1j*vista_pattern['Imag'])
        vista_pattern['Phase']=np.angle(vista_pattern['Real']+1j*vista_pattern['Imag'])
        return vista_pattern
    def pattern_mesh(self):
        points = self.cartesian_points()
        mesh = pv.StructuredGrid(points[:, 0], points[:, 1], points[:, 2])
        mesh.dimensions = [self.azimuth_resolution, self.elevation_resolution, 1]
        if self.arbitary_pattern_format == "Etheta/Ephi":
            mesh.point_data['Etheta'] = self.pattern[:,:,0]
            mesh.point_data['Ephi'] = self.pattern[:, :, 1]
        elif self.arbitary_pattern_format == "ExEyEz":
            mesh.point_data['Ex'] = self.pattern[:, :, 0]
            mesh.point_data['Ey'] = self.pattern[:, :, 1]
            mesh.point_data['Ez'] = self.pattern[:, :, 2]

        return mesh

    def display_pattern(
        self, plottype="Polar", desired_pattern="both", pattern_min=-40, plot_max=0,plotengine="matplotlib"
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
        if plotengine=="pyvista":
            def PatternPlot(
                    data,
                    az,
                    elev,
                    pattern_min=-40,
                    plot_max=0.0,
                    plottype="Polar",
                    logtype="amplitude",
                    ticknum=6,
                    title_text=None,
                    shell_radius=1.0
            ):
                # points = spherical_mesh(az, elev,
                # (data / np.nanmax(data) * shell_radius))
                # mesh = pv.StructuredGrid(points[:, 0], points[:, 1], points[:, 2])
                # mesh.dimensions = [az.size, elev.size, 1]
                # mesh.point_data['Beamformed Directivity (dBi)'] = 10 * np.log10(data.ravel())
                mesh = self.pattern_mesh(shell_radius)
                sargs = dict(
                    title_font_size=20,
                    label_font_size=16,
                    shadow=True,
                    n_labels=ticknum,
                    italic=True,
                    fmt="%.1f",
                    font_family="arial",
                )
                pl = pv.Plotter()
                pl.add_mesh(mesh, scalar_bar_args=sargs, clim=[pattern_min, plot_max])
                pl.show_axes()
                pl.show()

                return pl
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
            elif desired_pattern == "Power":
                BM.PatternPlot(
                    self.pattern[:, :, 0] ** 2 + self.pattern[:, :, 1] ** 2,
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    logtype="power",
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

    def transmute_pattern(self, desired_format="Etheta/Ephi"):
        """
        convert the pattern from Etheta/Ephi format to Ex, Ey,Ez format, or back again
        """
        if self.arbitary_pattern_format == "Etheta/Ephi":
            if desired_format == "ExEyEz":
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
                # recalculate pattern using Right Hand Circular and Left Hand Circular Polarisation
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
        field_points = np.array([x, y, z]).transpose().astype(np.float32)

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
        self.elements = elements
        self.beamforming_weights = np.ones((self.elements), dtype=np.complex64)
        az_mesh, elev_mesh = np.meshgrid(
            np.linspace(-180, 180, self.azimuth_resolution),
            np.linspace(-90, 90, self.elevation_resolution),
        )
        self.az_mesh = az_mesh
        self.elev_mesh = elev_mesh
        if self.arbitary_pattern_format == "Etheta/Ephi":
            self.pattern = np.zeros(
                (elements, self.elevation_resolution, self.azimuth_resolution, 2),
                dtype=np.complex64,
            )
        elif self.arbitary_pattern_format == "ExEyEz":
            self.pattern = np.zeros(
                (elements, self.elevation_resolution, self.azimuth_resolution, 3),
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
            self.pattern[:, :, :, 0] = 1.0
        elif self.arbitary_pattern_type == "xhalfspace":
            az_angles = self.az_mesh[0, :]
            az_index = np.where(np.abs(az_angles) < 90)
            self.pattern[:, az_index, :, 0] = 1.0
        elif self.arbitary_pattern_type == "yhalfspace":
            az_angles = self.az_mesh[0, :]
            az_index = np.where(az_angles > 90)
            self.pattern[:, az_index, :, 0] = 1.0
        elif self.arbitary_pattern_type == "zhalfspace":
            elev_angles = self.elev_mesh[:, 0]
            elev_index = np.where(elev_angles > 0)
            self.pattern[elev_index, :, :, 0] = 1.0

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
                    self.beamforming_weights * self.pattern[:, :, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Etheta",
                )
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ephi",
                )
            elif desired_pattern == "Etheta":
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Etheta",
                )
            elif desired_pattern == "Ephi":
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ephi",
                )
            elif desired_pattern == "Power":
                BM.PatternPlot(
                    (self.beamforming_weights * self.pattern[:, :, :, 0]) ** 2
                    + (self.beamforming_weights * self.pattern[:, :, :, 1]) ** 2,
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Power Pattern",
                )
        elif self.arbitary_pattern_format == "ExEyEz":
            if desired_pattern == "both":
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ex",
                )
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ey",
                )
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 2],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ez",
                )
            elif desired_pattern == "Ex":
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 0],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ex",
                )
            elif desired_pattern == "Ey":
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 1],
                    self.az_mesh,
                    self.elev_mesh,
                    pattern_min=pattern_min,
                    plottype=plottype,
                    title_text="Ey",
                )
            elif desired_pattern == "Ez":
                BM.PatternPlot(
                    self.beamforming_weights * self.pattern[:, :, :, 2],
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
        field_points = np.array([x, y, z]).transpose().astype(np.float32)

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
            self.beamforming_weights * self.pattern[:, :, :, 0],
            self.beamforming_weights * self.pattern[:, :, :, 1],
            az_range=self.az_mesh[0, :],
            elev_range=self.elev_mesh[:, 0],
        )
        return Dtheta, Dphi, Dtotal, Dmax

import copy
import math
from packaging import version
import numpy as np
from numba import float32, from_dtype, njit, guvectorize
from scipy import interpolate as sp
from scipy.spatial.transform import Rotation as R

import open3d as o3d
from .electromagnetics import beamforming as BM
from .electromagnetics import empropagation as EM
from .geometry import geometryfunctions as GF
from .raycasting import rayfunctions as RF
from glob import glob
import shutil
import os
from sphinx_gallery.scrapers import figure_rst

# A numpy record array (like a struct) to record triangle
point_data = np.dtype([
    # conductivity, permittivity and permiability
    # free space values should be
    # permittivity of free space 8.8541878176e-12F/m
    # permeability of free space 1.25663706212e-6H/m
    ('permittivity', 'c8'), ('permeability', 'c8'),
    # electric or magnetic current sources? E if True
    ('Electric', '?'),
], align=True)
point_t = from_dtype(point_data)  # Create a type that numba can recognize!

# A numpy record array (like a struct) to record triangle
triangle_data = np.dtype([
    # v0 data
    ('v0x', 'f4'), ('v0y', 'f4'), ('v0z', 'f4'),
    # v1 data
    ('v1x', 'f4'), ('v1y', 'f4'), ('v1z', 'f4'),
    # v2 data
    ('v2x', 'f4'), ('v2y', 'f4'), ('v2z', 'f4'),
    # normal vector
    # ('normx', 'f4'),  ('normy', 'f4'), ('normz', 'f4'),
    # ('reflection', np.float64),
    # ('diffuse_c', np.float64),
    # ('specular_c', np.float64),
], align=True)
triangle_t = from_dtype(triangle_data)  # Create a type that numba can recognize!

# ray class, to hold the ray origin, direction, and eventuall other data.
ray_data = np.dtype([
    # origin data
    ('ox', 'f4'), ('oy', 'f4'), ('oz', 'f4'),
    # direction vector
    ('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4'),
    # target
    # direction vector
    # ('tx','f4'),('ty','f4'),('tz','f4'),
    # distance traveled
    ('dist', 'f4'),
    # intersection
    ('intersect', '?'),
], align=True)
ray_t = from_dtype(ray_data)  # Create a type that numba can recognize!
# We can use that type in our device functions and later the kernel!

scattering_point = np.dtype([
    # position data
    ('px', 'f4'), ('py', 'f4'), ('pz', 'f4'),
    # velocity
    ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
    # normal
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    # weights
    ('ex', 'c8'), ('ey', 'c8'), ('ez', 'c8'),
    # conductivity, permittivity and permiability
    # free space values should be
    # permittivity of free space 8.8541878176e-12F/m
    # permeability of free space 1.25663706212e-6H/m
    ('permittivity', 'c8'), ('permeability', 'c8'),
    # electric or magnetic current sources? E if True
    ('Electric', '?'),
], align=True)

scattering_t = from_dtype(scattering_point)  # Create a type that numba can recognize!

class PNGScraper(object):
    def __init__(self):
        self.seen = set()

    def __repr__(self):
        return 'PNGScraper'

    def __call__(self, block, block_vars, gallery_conf):
        # Find all PNG files in the directory of this example.
        path_current_example = os.path.dirname(block_vars['src_file'])
        pngs = sorted(glob(os.path.join(path_current_example, '*.png')))

        # Iterate through PNGs, copy them to the sphinx-gallery output directory
        image_names = list()
        image_path_iterator = block_vars['image_path_iterator']
        for png in pngs:
            if png not in self.seen:
                self.seen |= set(png)
                this_image_path = image_path_iterator.next()
                image_names.append(this_image_path)
                shutil.move(png, this_image_path)
        # Use the `figure_rst` helper function to generate rST for image files
        return figure_rst(image_names, gallery_conf['src_dir'])

@njit
def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


@njit
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


@njit
def cart2sph(x, y, z):
    # radians
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


@njit
def sph2cart(az, el, r):
    # radians
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


@njit
def calc_normals(T):
    # calculate triangle norm
    e1x = T['v1x'] - T['v0x']
    e1y = T['v1y'] - T['v0y']
    e1z = T['v1z'] - T['v0z']

    e2x = T['v2x'] - T['v0x']
    e2y = T['v2y'] - T['v0y']
    e2z = T['v2z'] - T['v0z']

    dirx = e1y * e2z - e1z * e2y
    diry = e1z * e2x - e1x * e2z
    dirz = e1x * e2y - e1y * e2x

    normconst = math.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2)

    T['normx'] = dirx / normconst
    T['normy'] = diry / normconst
    T['normz'] = dirz / normconst

    return T


@guvectorize([(float32[:], float32[:], float32[:], float32)], '(n),(n)->(n),()', target='parallel')
def fast_calc_dv(source, target, dv, normconst):
    dirx = target[0] - source[0]
    diry = target[1] - source[1]
    dirz = target[2] - source[2]
    normconst = math.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2)
    dv = np.array([dirx, diry, dirz]) / normconst


@njit
def calc_dv(source, target):
    dirx = target[0] - source[0]
    diry = target[1] - source[1]
    dirz = target[2] - source[2]
    normconst = np.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2)
    dv = np.array([dirx, diry, dirz]) / normconst
    return dv[0], dv[1], dv[2], normconst


@njit
def calc_dv_norm(source, target, direction, length):
    length[:, 0] = np.sqrt(
        (target[:, 0] - source[:, 0]) ** 2 + (target[:, 1] - source[:, 1]) ** 2 + (target[:, 2] - source[:, 2]) ** 2)
    direction = (target - source) / length
    return direction, length


class structures:
    """
    Structure class to store information about the geometry and materials in the environment, holding the seperate
    shapes as open3D trianglemesh data structures. Everything in the class will be considered an integrated unit, rotating and moving together.
    This class will be developed to include material parameters to enable more complex modelling.

    Units should be SI, metres

    This is the default class for passing structures to the different models.
    """

    def __init__(self,
                 solids):
        # solids is a list of open3D trianglemesh structures
        self.solids = []
        for item in solids:
            self.solids.append(item)
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
        new_solids : open3d trianglemesh
            the solid to be added to the structure

        Returns
        --------
        None

        """
        self.solids.append(new_solids)
        # self.materials.append(new_materials)

    def rotate_structures(self, rotation_matrix, rotation_centre=np.zeros((3,1), dtype=np.float32)):
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
        for item in self.solids:
            self.solids[item]=GF.open3drotate(self.solids[item],rotation_matrix,rotation_centre)


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
        for item in self.solids:
            self.solids[item].translate(vector)

    def triangles_base_raycaster(self):
        """
        generates the triangles for all the trianglemesh objects in the structure, and outputs them as a continuous array of
        triangle_t format triangles

        Parameters
        -----------
        None

        Returns
        --------
        triangles : N by 1 numpy array of triangle_t triangles
            a continuous array of all the triangles in the structure
        """
        triangles = np.empty((0), dtype=triangle_t)
        for item in self.solids:
            triangles = np.append(triangles, RF.convertTriangles(copy.deepcopy(item)))

        return triangles


class antenna_pattern:
    """
    Antenna Pattern class which allows for patterns to be handled consistently
    across LyceanEM and other modules. The definitions assume that the pattern axes
    are consistent with the global axes set. If a different orientation is required,
    such as a premeasured antenna in a new orientation then the pattern rotate_function
    must be used.

    Antenna Pattern Frequency is in Hz, eventually I may update the class for multi point patterns
    Rotation Offset is Specified in terms of rotations around the x, y, and z axes as roll,pitch/elevation, and azimuth
    in radians.
    """

    def __init__(self,
                 azimuth_resolution=37,
                 elevation_resolution=37,
                 pattern_frequency=1e9,
                 arbitary_pattern=False,
                 arbitary_pattern_type='isotropic',
                 arbitary_pattern_format='Etheta/Ephi',
                 position_mapping=np.zeros((3), dtype=np.float32),
                 rotation_offset=np.zeros((3), dtype=np.float32)):

        self.azimuth_resolution = azimuth_resolution
        self.elevation_resolution = elevation_resolution
        self.pattern_frequency = pattern_frequency
        self.arbitary_pattern_type = arbitary_pattern_type
        self.arbitary_pattern_format = arbitary_pattern_format
        self.position_mapping = position_mapping
        self.rotation_offset = rotation_offset
        self.field_radius=1.0
        az_mesh, elev_mesh = np.meshgrid(np.linspace(-180, 180, self.azimuth_resolution),
                                         np.linspace(-90, 90, self.elevation_resolution))
        self.az_mesh = az_mesh
        self.elev_mesh = elev_mesh
        if self.arbitary_pattern_format == 'Etheta/Ephi':
            self.pattern = np.zeros((self.elevation_resolution, self.azimuth_resolution, 2),
                                    dtype=np.complex64)
        elif self.arbitary_pattern_format == 'ExEyEz':
            self.pattern = np.zeros((self.elevation_resolution, self.azimuth_resolution, 3),
                                    dtype=np.complex64)
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
        x_rot = R.from_euler('X', self.rotation_offset[0], degrees=True).as_matrix()
        y_rot = R.from_euler('Y', self.rotation_offset[1], degrees=True).as_matrix()
        z_rot = R.from_euler('Z', self.rotation_offset[2], degrees=True).as_matrix()

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

        if self.arbitary_pattern_type == 'isotropic':
            self.pattern[:, :, 0] = 1.0
        elif self.arbitary_pattern_type == 'xhalfspace':
            az_angles = self.az_mesh[0, :]
            az_index = np.where(np.abs(az_angles) < 90)
            self.pattern[:, az_index, 0] = 1.0
        elif self.arbitary_pattern_type == 'yhalfspace':
            az_angles = self.az_mesh[0, :]
            az_index = np.where(az_angles > 90)
            self.pattern[:, az_index, 0] = 1.0
        elif self.arbitary_pattern_type == 'zhalfspace':
            elev_angles = self.elev_mesh[:, 0]
            elev_index = np.where(elev_angles > 0)
            self.pattern[elev_index, :, 0] = 1.0

    def import_pattern(self, file_location):
        """
        takes the file location and imports the individual pattern file, replacing exsisting values with those of the saved file.

        It is import to note that for CST ASCII export format, that you select a plot range of -180 to 180 for phi, and that by defauly CST exports from 0 to 180 in theta, which
        is the opposite direction to the default for LyceanEM, so this data structures are flipped for consistency.
        Inputs : file location

        Returns : None
        """
        if file_location.suffix == '.txt':
            # file is a CST ffs format
            datafile = np.loadtxt(file_location, skiprows=2)
            theta = datafile[:, 0]
            phi = datafile[:, 1]
            freq_index1 = file_location.name.find('f=')
            freq_index2 = file_location.name.find(')')
            file_frequency = float(file_location.name[freq_index1 + 2:freq_index2]) * 1e9
            phi_steps = np.linspace(np.min(phi), np.max(phi), np.unique(phi).size)
            theta_steps = np.linspace(np.min(theta), np.max(theta), np.unique(theta).size)
            phi_res = np.unique(phi).size
            theta_res = np.unique(theta).size
            etheta = (datafile[:, 3] * np.exp(1j * np.deg2rad(datafile[:, 4]))).reshape(phi_res,theta_res).transpose()
            ephi = (datafile[:, 5] * np.exp(1j * np.deg2rad(datafile[:, 6]))).reshape(phi_res,theta_res).transpose()
            self.azimuth_resolution = phi_res
            self.elevation_resolution = theta_res
            self.elev_mesh = np.flipud(GF.thetatoelevation(theta).reshape(phi_res,theta_res).transpose())
            self.az_mesh = np.flipud(phi.reshape(phi_res,theta_res).transpose())
            self.pattern = np.zeros((self.elevation_resolution, self.azimuth_resolution, 2),
                                    dtype=np.complex64)
            self.pattern[:, :, 0] = np.flipud(etheta)
            self.pattern[:, :, 1] = np.flipud(ephi)
            self.pattern_frequency = file_frequency
        elif file_location.suffix == '.dat':
            # file is .dat format from anechoic chamber measurements
            Ea, Eb, freq, norm, theta_values, phi_values = EM.importDat(file_location)
            az_mesh, elev_mesh = np.meshgrid(phi_values, GF.thetatoelevation(theta_values))
            self.azimuth_resolution = np.unique(phi_values).size
            self.elevation_resolution = np.unique(theta_values).size
            self.az_mesh = az_mesh
            self.elev_mesh = elev_mesh
            self.pattern = np.zeros((self.elevation_resolution, self.azimuth_resolution, 2),
                                    dtype=np.complex64)
            self.pattern_frequency = freq * 1e6
            self.pattern[:, :, 0] = Ea.transpose() + norm
            self.pattern[:, :, 1] = Eb.transpose() + norm

    def export_pattern(self, file_location):
        """
        takes the file location and exports the pattern as a .dat file
        unfinished, must be in Etheta/Ephi format,

        Parameters
        -----------
        file_location : posix path
            the path for the output file, including name

        Returns
        --------
        None
        """
        if self.arbitary_pattern_format == 'ExEyEz':
            self.transmute_pattern()

        theta_flat = GF.elevationtotheta(self.elev_mesh).transpose().reshape(-1,1)
        phi_mesh = self.az_mesh.transpose().reshape(-1,1)
        planes=self.azimuth_resolution
        copolardb=20*np.log10(np.abs(self.pattern[:,:,0].transpose().reshape(-1,1)))
        copolarangle=np.degrees(np.angle(self.pattern[:,:,0].transpose().reshape(-1,1)))
        crosspolardb = 20 * np.log10(np.abs(self.pattern[:, :, 1].transpose().reshape(-1,1)))
        crosspolarangle = np.degrees(np.angle(self.pattern[:, :, 1].transpose().reshape(-1,1)))
        norm=np.nanmax(np.array([np.nanmax(copolardb),np.nanmax(crosspolardb)]))
        copolardb -= norm
        crosspolardb -= norm
        infoarray=np.array([planes,np.min(self.az_mesh),np.max(self.az_mesh),norm,self.pattern_frequency/1e6]).reshape(1,-1)

        dataarray = np.concatenate((theta_flat.reshape(-1,1),
                                    copolardb.reshape(-1,1),
                                    copolarangle.reshape(-1,1),
                                    crosspolardb.reshape(-1,1),
                                    crosspolarangle.reshape(-1,1)),axis=1)
        outputarray=np.concatenate((infoarray,dataarray),axis=0)
        np.savetxt(file_location,outputarray, delimiter=',',fmt="%.2e")


    def display_pattern(self, desired_pattern='both', pattern_min=-40):
        """
        displays the antenna pattern
        """

        if self.arbitary_pattern_format == 'Etheta/Ephi':
            if desired_pattern == 'both':
                BM.PatternPlot(self.pattern[:, :, 0], self.az_mesh, self.elev_mesh, pattern_min=pattern_min,
                               title_text='Etheta')
                BM.PatternPlot(self.pattern[:, :, 1], self.az_mesh, self.elev_mesh, pattern_min=pattern_min,
                               title_text='Ephi')
        elif self.arbitary_pattern_format == 'ExEyEz':
            if desired_pattern == 'both':
                BM.PatternPlot(self.pattern[:, :, 0], self.az_mesh, self.elev_mesh, pattern_min=pattern_min,
                               title_text='Ex')
                BM.PatternPlot(self.pattern[:, :, 1], self.az_mesh, self.elev_mesh, pattern_min=pattern_min,
                               title_text='Ey')
                BM.PatternPlot(self.pattern[:, :, 2], self.az_mesh, self.elev_mesh, pattern_min=pattern_min,
                               title_text='Ez')

    def transmute_pattern(self):
        """
        convert the pattern from Etheta/Ephi format to Ex, Ey,Ez format
        """
        if self.arbitary_pattern_format == 'Etheta/Ephi':
            oldformat = self.pattern.reshape(-1, self.pattern.shape[2])
            old_shape = oldformat.shape
            self.arbitary_pattern_format = 'ExEyEz'
            self.pattern = np.zeros((self.elevation_resolution,
                                     self.azimuth_resolution,
                                     3),
                                    dtype=np.complex64)
            theta = GF.elevationtotheta(self.elev_mesh)
            # convrsion part, move to transmute pattern
            conversion_matrix1 = np.asarray(
                [np.cos(np.deg2rad(theta.ravel())) * np.cos(np.deg2rad(self.az_mesh.ravel())),
                 np.cos(np.deg2rad(theta.ravel())) * np.sin(np.deg2rad(self.az_mesh.ravel())),
                 -np.sin(np.deg2rad(theta.ravel()))]).transpose()
            conversion_matrix2 = np.asarray([-np.sin(np.deg2rad(self.az_mesh.ravel())),
                                             np.cos(np.deg2rad(self.az_mesh.ravel())),
                                             np.zeros(self.az_mesh.size)]).transpose()
            decomposed_fields = oldformat[:, 0].reshape(-1, 1) * conversion_matrix1 + oldformat[:, 1].reshape(-1,
                                                                                                              1) * conversion_matrix2
            self.pattern[:, :, 0] = decomposed_fields[:, 0].reshape(self.elevation_resolution, self.azimuth_resolution)
            self.pattern[:, :, 1] = decomposed_fields[:, 1].reshape(self.elevation_resolution, self.azimuth_resolution)
            self.pattern[:, :, 2] = decomposed_fields[:, 2].reshape(self.elevation_resolution, self.azimuth_resolution)
        else:
            oldformat = self.pattern.reshape(-1, self.pattern.shape[2])
            old_shape = oldformat.shape
            self.arbitary_pattern_format = 'Etheta/Ephi'
            self.pattern = np.zeros((self.elevation_resolution,
                                     self.azimuth_resolution,
                                     2),
                                    dtype=np.complex64)
            theta = GF.elevationtotheta(self.elev_mesh)
            costhetacosphi = (np.cos(np.deg2rad(self.az_mesh.ravel())) * np.cos(np.deg2rad(theta.ravel()))).astype(
                np.complex64)
            sinphicostheta = (np.sin(np.deg2rad(self.az_mesh.ravel())) * np.cos(np.deg2rad(theta.ravel()))).astype(
                np.complex64)
            sintheta = (np.sin(np.deg2rad(theta.ravel()))).astype(np.complex64)
            sinphi = (np.sin(np.deg2rad(self.az_mesh.ravel()))).astype(np.complex64)
            cosphi = (np.cos(np.deg2rad(self.az_mesh.ravel()))).astype(np.complex64)
            new_etheta = oldformat[:, 0] * costhetacosphi + oldformat[:, 1] * sinphicostheta - oldformat[:,
                                                                                               2] * sintheta
            new_ephi = -oldformat[:, 0] * sinphi + oldformat[:, 1] * cosphi
            self.pattern[:, :, 0] = new_etheta.reshape(self.elevation_resolution, self.azimuth_resolution)
            self.pattern[:, :, 1] = new_ephi.reshape(self.elevation_resolution, self.azimuth_resolution)

    def cartesian_points(self):
        """
        exports the cartesian points for all pattern points.
        """
        x, y, z = RF.sph2cart(np.deg2rad(self.az_mesh.ravel()),
                              np.deg2rad(self.elev_mesh.ravel()),
                              np.ones((self.pattern[:, :, 0].size))*self.field_radius)
        field_points = np.array([x, y, z]).transpose().astype(np.float32) + self.position_mapping

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
        x, y, z = RF.sph2cart(np.deg2rad(self.az_mesh.ravel()),
                              np.deg2rad(self.elev_mesh.ravel()),
                              np.ones((self.pattern[:, :, 0].size)))
        field_points = np.array([x, y, z]).transpose().astype(np.float32)

        # convert to ExEyEz for rotation
        if self.arbitary_pattern_format == 'Etheta/Ephi':
            self.transmute_pattern()
            desired_format = 'Etheta/Ephi'
        else:
            desired_format = 'ExEyEz'

        if rotation_matrix == None:
            rot_mat = self._rotation_matrix()
        else:
            rot_mat = rotation_matrix

        rotated_points = np.dot(field_points, rot_mat)
        decomposed_fields = self.pattern.reshape(-1, 3)
        # rotation part
        xyzfields = np.dot(decomposed_fields, rot_mat)
        # resample
        resampled_xyzfields = self.resample_pattern(rotated_points, xyzfields, field_points)
        self.pattern[:, :, 0] = resampled_xyzfields[:, 0].reshape(self.elevation_resolution, self.azimuth_resolution)
        self.pattern[:, :, 1] = resampled_xyzfields[:, 1].reshape(self.elevation_resolution, self.azimuth_resolution)
        self.pattern[:, :, 2] = resampled_xyzfields[:, 2].reshape(self.elevation_resolution, self.azimuth_resolution)

        if desired_format == 'Etheta/Ephi':
            # convert back
            self.transmute_pattern()

    def resample_pattern_angular(self, new_azimuth_resolution, new_elevation_resolution):
        """
        resample pattern based upon provided azimuth and elevation resolution
        """
        new_az_mesh, new_elev_mesh = np.meshgrid(np.linspace(-180, 180, new_azimuth_resolution),
                                                 np.linspace(-90, 90, new_elevation_resolution))
        x, y, z = RF.sph2cart(np.deg2rad(new_az_mesh.ravel()),
                              np.deg2rad(new_elev_mesh.ravel()),
                              np.ones((self.pattern[:, :, 0].size)))
        new_field_points = np.array([x, y, z]).transpose().astype(np.float32)
        old_field_points = self.cartesian_points()
        old_pattern = self.pattern.reshape(-1, 2)
        new_pattern = self.resample_pattern(old_field_points, old_pattern, new_field_points)

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
            mag_interpolate = sp.Rbf(old_points[:, 0], old_points[:, 1], old_points[:, 2],
                                     np.abs(old_pattern[:, component]), smooth=smoothing_factor)
            phase_interpolate = sp.Rbf(old_points[:, 0], old_points[:, 1], old_points[:, 2],
                                       np.angle(old_pattern[:, component]), smooth=smoothing_factor)
            new_mag = mag_interpolate(new_points[:, 0], new_points[:, 1], new_points[:, 2])
            new_angles = phase_interpolate(new_points[:, 0], new_points[:, 1], new_points[:, 2])
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
        Dtheta, Dphi, Dtotal, Dmax = EM.directivity_transformv2(self.pattern[:,:,0],
                                                                self.pattern[:,:,1],
                                                                az_range=self.az_mesh[0,:],
                                                                elev_range=self.elev_mesh[:,0])
        return Dtheta,Dphi,Dtotal,Dmax

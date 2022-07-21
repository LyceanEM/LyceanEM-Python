import numpy as np

# provide idealised patterns to allow testing of the different models
import open3d as o3d

from lyceanem.base_classes import antenna_pattern
import lyceanem.geometry.geometryfunctions as GF



def electriccurrentsource(prime_vector, theta, phi):
    """
    create an idealised electric current source that can be used to test the outputs of the model

    Parameters
    ----------
    prime_vector : 1D numpy array of floats
        orientation of the electric current source in xyz
    theta : 2D numpy array of floats
        theta angles of desired pattern in degrees
    phi : 2D numpy array of floats
        phi angles of desired pattern in degrees

    Returns
    -------
    etheta : 2D numpy array of complex
        Etheta polarisation
    ephi : 2D numpy array of complex
        Ephi polarisation
    """

    etheta = np.zeros(theta.shape, dtype=np.complex64)
    ephi = np.zeros(theta.shape, dtype=np.complex64)
    etheta = (
        prime_vector[0] * np.cos(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))
        + prime_vector[1] * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))
        - prime_vector[2] * np.sin(np.deg2rad(theta))
    )
    ephi = -prime_vector[0] * np.sin(np.deg2rad(phi)) + prime_vector[1] * np.cos(
        np.deg2rad(phi)
    )
    return etheta, ephi


def antenna_pattern_source(radius, import_antenna=False, antenna_file=None):
    """
    This function generates an antenna pattern and `opaque' sphere as the base, representing an inserted antenna with measured pattern.

    This function is not yet complete
    Parameters
    ----------
    radius : float
        radius of the sphere, setting the minimum enclosing volume of the antenna
    import_antenna : boolean,
        if [True] the provided antenna_file location will be used to import an antenna file to populate the variable
    antenna_file : PosixPath
        a file location for the antenna file to be used. The initial set will be based upon the .dat files used by the University of Bristol Anechoic Chamber

    Returns
    --------
    solid : :class:`open3d.geometry.TriangleMesh`
        the enclosing sphere for the antenna
    points : :class:`open3d.geometry.PointCloud`
        the sample points for the antenna pattern, to be used as source points for the frequency domain model
    pattern : 3 by N numpy array of complex
        array of the sample points of the antenna pattern, specified as Ex,Ey,Ez components

    """
    if import_antenna:
        # import antenna pattern
        pattern = antenna_pattern()
        pattern.import_pattern(antenna_file)
        pattern.field_radius = radius
    else:
        print("arbitary pattern")
        # generate an arbitary locally Z directed electric current source
        prime_vector = np.zeros((3), dtype=np.float32)
        prime_vector[2] = 1.0
        az_res = 37
        elev_res = 37
        az_mesh, elev_mesh = np.meshgrid(
            np.linspace(-180, 180, az_res), np.linspace(-90, 90, elev_res)
        )
        _, theta = np.meshgrid(
            np.linspace(-180.0, 180.0, az_res),
            GF.elevationtotheta(np.linspace(-90, 90, elev_res)),
        )
        etheta, ephi = electriccurrentsource(prime_vector, theta, az_mesh)
        pattern = antenna_pattern()
        pattern.pattern[:, :, 0] = etheta
        pattern.pattern[:, :, 1] = ephi
        pattern.field_radius = radius

    field_points = pattern.cartesian_points()
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(field_points)
    points.normals = o3d.utility.Vector3dVector(field_points)
    points.normalize_normals()
    solid = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=30)

    return solid, points, pattern

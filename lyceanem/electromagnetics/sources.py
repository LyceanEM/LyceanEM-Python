import numpy as np
import lyceanem.geometry.geometryfunctions as GF
from lyceanem.base_classes import antenna_pattern


def electriccurrentsource(prime_vector, theta, phi):
    """
    Create an idealised electric current source based upon the provided electric field vector

    Parameters
    ----------
    prime_vector : numpy.ndarray of floats
        orientation of the electric current source in xyz
    theta : numpy.ndarray of floats
        theta angles of desired pattern in degrees
    phi : numpy.ndarray of floats
        phi angles of desired pattern in degrees

    Returns
    -------
    etheta : numpy.ndarray of complex
        Etheta polarisation
    ephi : numpy.ndarray of complex
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
    import_antenna : bool
        if [True] the provided antenna_file location will be used to import an antenna file to populate the variable
    antenna_file : PosixPath
        a file location for the antenna file to be used. The initial set will be based upon the .dat files used by the University of Bristol Anechoic Chamber

    Returns
    --------
    solid : meshio.Mesh
        the enclosing sphere for the antenna
    pattern : numpy.ndarray of floats
        array of the sample points of the antenna pattern, specified as Ex,Ey,Ez components

    """
    if import_antenna:
        # import antenna pattern
        pattern = antenna_pattern()
        ## import the pattern not implemented
        pattern.import_pattern(antenna_file)
        pattern.field_radius = radius
        az_res=pattern.azimuth_resolution
        elev_res=pattern.elevation_resolution

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

    from lyceanem.geometry.targets import spherical_field
    # create a sphere to represent the antenna
    solid = spherical_field(radius=radius, az_res=az_res, elev_res=elev_res)
    from lyceanem.electromagnetics.emfunctions import update_electric_fields
    solid = update_electric_fields(solid, *field_points)
    return solid, pattern

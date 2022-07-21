import numpy as np

# provide idealised patterns to allow testing of the different models
import open3d as o3d

#import lyceanem.base_classes as base_classes
import lyceanem.geometry.geometryfunctions as GF
import lyceanem.raycasting.rayfunctions as RF


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




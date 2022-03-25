import numpy as np
# provide idealised patterns to allow testing of the different models
import open3d as o3d

import lyceanem.base as base
import lyceanem.geometry.geometryfunctions as GF
import lyceanem.raycasting.rayfunctions as RF


def electriccurrentsource(prime_vector,theta,phi):
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

    etheta=np.zeros(theta.shape,dtype=np.complex64)
    ephi=np.zeros(theta.shape,dtype=np.complex64)
    etheta = prime_vector[0]* np.cos(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)) + prime_vector[1]*np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)) - prime_vector[2]* np.sin(np.deg2rad(theta))
    ephi= -prime_vector[0] * np.sin(np.deg2rad(phi)) + prime_vector[1]*np.cos(np.deg2rad(phi))
    return etheta,ephi

def antenna_pattern_source(local_axes_global,radius,import_antenna=False,antenna_file=None):
    """
    This function generates an antenna pattern and `opaque' sphere as the base, representing an inserted antenna with measured pattern.

    This function is not yet complete
    Parameters
    ----------
    loacl_axes_global : 2D numpy array of float
        3 by 3 array of the local axis vectors (x,y,z), in the global coordinate set.
    radius : float
        radius of the sphere, setting the minimum enclosing volume of the antenna
    import_antenna : boolean,
        if [True] the provided antenna_file location will be used to import an antenna file to populate the variable
    antenna_file : PosixPath
        a file location for the antenna file to be used. The initial set will be based upon the .dat files used by the University of Bristol Anechoic Chamber

    Returns
    --------
    solid : open3d trianglemesh
        the enclosing sphere for the antenna
    points : open3d point cloud
        the sample points for the antenna pattern, to be used as source points for the frequency domain model
    pattern : 3 by N numpy array of complex
        array of the sample points of the antenna pattern, specified as Ex,Ey,Ez components
    """
    if ~import_antenna:
        #generate an arbitary locally Z directed electric current source
        prime_vector=np.zeros((3),dtype=np.float32)
        prime_vector[2]=1.0
        az_res=37
        elev_res=37
        az_mesh,elev_mesh=np.meshgrid(np.linspace(-180,180,az_res),np.linspace(-90,90,elev_res))
        _, theta = np.meshgrid(np.linspace(-180.0, 180.0, az_res), GF.elevationtotheta(np.linspace(-90,90,elev_res)))
        etheta,ephi=electriccurrentsource(prime_vector,theta,az_mesh)

    field_points=np.size(etheta)
    points=o3d.geometry.PointCloud()
    x,y,z=RF.sph2cart(az_mesh.ravel(),elev_mesh.ravel(),radius*np.ones((field_points)))
    solid=o3d.geometry.TriangleMesh.create_sphere(radius=radius,resolution=field_points)
    #define the material characteristics for the source
    local_information = np.empty((len(field_points)), dtype=base.scattering_t)
    # set all sources as magnetic current sources, and permittivity and permeability as free space
    local_information[:]['Electric'] = True
    local_information[:]['permittivity'] = 8.8541878176e-12
    local_information[:]['permeability'] = 1.25663706212e-6
    local_E_vector=np.zeros((field_points,2),dtype=np.complex)
    local_E_vector[:,0]=etheta.ravel()
    local_E_vector[:,1]=ephi.ravel()
    pattern=base.antenna_pattern
    #outgoing_E_vector=launchtransform(source_normal, departure_vector, local_E_vector, local_information)
    return solid,points,pattern


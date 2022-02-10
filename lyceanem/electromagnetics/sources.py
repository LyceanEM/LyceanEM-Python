import numpy as np


#provide idealised patterns to allow testing of the different models
import open3d as o3d

from ..electromagnetics.empropagation import launchtransform
from ..geometry import geometryfunctions as GF
from ..raycasting import rayfunctions as RF
from ..base import scattering_t


def electriccurrentsource(prime_vector,theta,phi):
    """
    create an idealised electric current source that can be used to test the outputs of the model
    Parameters
    prime vector : orientation of the electric current source
    sinks       : angular position of the sinks 2d*2 (phi,theta)
    """
    etheta=np.zeros(theta.shape)
    ephi=np.zeros(theta.shape)
    etheta = prime_vector[0]* np.cos(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)) + prime_vector[1]*np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta)) - prime_vector[2]* np.sin(np.deg2rad(theta))
    ephi= -prime_vector[0] * np.sin(np.deg2rad(phi)) + prime_vector[1]*np.cos(np.deg2rad(phi))
    return etheta,ephi

def antenna_pattern_source(local_axes_global,radius,import_antenna=False,antenna_file=None):
    """
    This function generates an antenna pattern and `opaque' sphere as the base, representing an inserted antenna with measured pattern.
    Inputs
    loacl_axes_global : a 3 by 3 float matrix pf the local axis vectors (x,y,z), in the global coordinate set.
    radius : of the sphere, setting the minimum enclosing volume of the antenna
    import_antenna : boolean, to select whether to import a measure or simulated antenna file
    antenna_file : a filename for the antenna file to be used. The initial set will be based upon the .dat files used by the University of Bristol Anechoic Chamber

    Outputs
    solid : open3d trianglemesh of the enclosing sphere
    points : open3d point cloud of the N relevent antenna pattern sample points
    pattern : 3 by N complex matrix of the sample points of the antenna pattern, specified as Ex,Ey,Ez components
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
    local_information = np.empty((len(field_points)), dtype=scattering_t)
    # set all sources as magnetic current sources, and permittivity and permeability as free space
    local_information[:]['Electric'] = True
    local_information[:]['permittivity'] = 8.8541878176e-12
    local_information[:]['permeability'] = 1.25663706212e-6
    local_E_vector=np.zeros((field_points,2),dtype=np.complex)
    local_E_vector[:,0]=etheta.ravel()
    local_E_vector[:,1]=ephi.ravel()
    outgoing_E_vector=launchtransform(source_normal, departure_vector, local_E_vector, local_information)
    return solid,points,pattern


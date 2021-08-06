

import numpy as np
import pathlib
from tqdm import tqdm
from math import sqrt
import cupy as cp
import cmath
import rayfunctions as RF
import scipy.stats
import math
import copy
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Wedge
import mpl_toolkits.mplot3d.art3d as art3d

from scipy.spatial import distance
from numpy.linalg import norm
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy.random import default_rng

from numba import cuda, int16, float32, float64, complex64, complex128, from_dtype, jit, njit, guvectorize, prange
from timeit import default_timer as timer

# A numpy record array (like a struct) to record triangle
point_data = np.dtype([
    # conductivity, permittivity and permiability
    #free space values should be
    #permittivity of free space 8.8541878176e-12F/m
    #permeability of free space 1.25663706212e-6H/m
    ('permittivity', 'c8'), ('permeability', 'c8'),
    #electric or magnetic current sources? E if True
    ('Electric', '?'),
    ], align=True)
point_t = from_dtype(point_data) # Create a type that numba can recognize!

# A numpy record array (like a struct) to record triangle
triangle_data = np.dtype([
    # v0 data
    ('v0x', 'f4'), ('v0y', 'f4'), ('v0z', 'f4'),
    # v1 data
    ('v1x', 'f4'),  ('v1y', 'f4'), ('v1z', 'f4'),
    # v2 data
    ('v2x', 'f4'),  ('v2y', 'f4'), ('v2z', 'f4'),
    # normal vector
    #('normx', 'f4'),  ('normy', 'f4'), ('normz', 'f4'),
    # ('reflection', np.float64),
    # ('diffuse_c', np.float64),
    # ('specular_c', np.float64),
    ], align=True)
triangle_t = from_dtype(triangle_data) # Create a type that numba can recognize!

# ray class, to hold the ray origin, direction, and eventuall other data.
ray_data=np.dtype([
    #origin data
    ('ox','f4'),('oy','f4'),('oz','f4'),
    #direction vector
    ('dx','f4'),('dy','f4'),('dz','f4'),
    #target
    #direction vector
    #('tx','f4'),('ty','f4'),('tz','f4'),
    #distance traveled
    ('dist','f4'),
    #intersection
    ('intersect','?'),
    ],align=True)
ray_t = from_dtype(ray_data) # Create a type that numba can recognize!
# We can use that type in our device functions and later the kernel!

scattering_point = np.dtype([
    #position data
    ('px', 'f4'), ('py', 'f4'), ('pz', 'f4'),
    #velocity
    ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
    #normal
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    #weights
    ('ex','c8'),('ey','c8'),('ez','c8'),
    # conductivity, permittivity and permiability
    #free space values should be
    #permittivity of free space 8.8541878176e-12F/m
    #permeability of free space 1.25663706212e-6H/m
    ('permittivity', 'c8'), ('permeability', 'c8'),
    #electric or magnetic current sources? E if True
    ('Electric', '?'),
    ], align=True)
scattering_t = from_dtype(scattering_point) # Create a type that numba can recognize!

@njit
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

@njit
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

@njit
def cart2sph(x, y, z):
    #radians
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

@njit
def sph2cart(az, el, r):
    #radians
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

@njit
def calc_normals(T):
    #calculate triangle norm
    e1x=T['v1x']-T['v0x']
    e1y=T['v1y']-T['v0y']
    e1z=T['v1z']-T['v0z']

    e2x=T['v2x']-T['v0x']
    e2y=T['v2y']-T['v0y']
    e2z=T['v2z']-T['v0z']

    dirx=e1y*e2z-e1z*e2y
    diry=e1z*e2x-e1x*e2z
    dirz=e1x*e2y-e1y*e2x

    normconst=math.sqrt(dirx**2+diry**2+dirz**2)

    T['normx'] = dirx/normconst
    T['normy'] = diry/normconst
    T['normz'] = dirz/normconst

    return T

@guvectorize([(float32[:], float32[:], float32[:], float32)], '(n),(n)->(n),()',target='parallel')
def fast_calc_dv(source,target,dv,normconst):
    dirx=target[0]-source[0]
    diry=target[1]-source[1]
    dirz=target[2]-source[2]
    normconst=math.sqrt(dirx**2+diry**2+dirz**2)
    dv=np.array([dirx,diry,dirz])/normconst

@njit
def calc_dv(source,target):
    dirx=target[0]-source[0]
    diry=target[1]-source[1]
    dirz=target[2]-source[2]
    normconst=np.sqrt(dirx**2+diry**2+dirz**2)
    dv=np.array([dirx,diry,dirz])/normconst
    return dv[0],dv[1],dv[2],normconst

@njit
def calc_dv_norm(source,target,direction,length):
    length[:,0]=np.sqrt((target[:,0]-source[:,0])**2+(target[:,1]-source[:,1])**2+(target[:,2]-source[:,2])**2)
    direction=(target-source)/length
    return direction, length

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

#provide idealised patterns to allow testing of the different models

def electriccurrentsource(prime_vector,sinks):
    """
    create an idealised electric current source that can be used to test the outputs of the model
    Parameters
    prime vector : orientation of the electric current source
    sinks       : angular position of the sinks 2d*2 (phi,theta)
    """
    etheta=np.zeros((1))
    ephi=np.zeros((1))
    etheta = prime_vector[0]* np.cos(np.deg2rad(sinks[:,:,0])) * np.cos(np.deg2rad(sinks[:,:,1])) + prime_vector[1]*np.sin(np.deg2rad(sinks[:,:,0])) * np.cos(np.deg2rad(sinks[:,:,1])) - prime_vector[2]* np.sin(np.deg2rad(sinks[:,:,1]))
    ephi= -prime_vector[0] * np.sin(np.deg2rad(sinks[:,:,0])) + prime_vector[1]*np.cos(np.deg2rad(sinks[:,:,0]))
    return etheta,ephi
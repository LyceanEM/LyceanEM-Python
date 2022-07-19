
import math
from packaging import version
import numpy as np
from numba import float32, from_dtype, njit, guvectorize

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
    e1x = T["v1x"] - T["v0x"]
    e1y = T["v1y"] - T["v0y"]
    e1z = T["v1z"] - T["v0z"]

    e2x = T["v2x"] - T["v0x"]
    e2y = T["v2y"] - T["v0y"]
    e2z = T["v2z"] - T["v0z"]

    dirx = e1y * e2z - e1z * e2y
    diry = e1z * e2x - e1x * e2z
    dirz = e1x * e2y - e1y * e2x

    normconst = math.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2)

    T["normx"] = dirx / normconst
    T["normy"] = diry / normconst
    T["normz"] = dirz / normconst

    return T


@guvectorize(
    [(float32[:], float32[:], float32[:], float32)],
    "(n),(n)->(n),()",
    target="parallel",
)
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
        (target[:, 0] - source[:, 0]) ** 2
        + (target[:, 1] - source[:, 1]) ** 2
        + (target[:, 2] - source[:, 2]) ** 2
    )
    direction = (target - source) / length
    return direction, length
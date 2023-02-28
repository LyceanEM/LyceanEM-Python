# -*- coding: utf-8 -*-
import copy
import math
from timeit import default_timer as timer

import numba as nb
import numpy as np
import open3d as o3d
import scipy.stats
from matplotlib import cm
from numba import cuda, float32, jit, njit, guvectorize, prange
from numpy.linalg import norm
from scipy.spatial import distance

import lyceanem.base_types as base_types
from ..utility import math_functions as math_functions
import lyceanem.electromagnetics.empropagation as EM

EPSILON = 1e-6  # how close to zero do we consider zero? example used 1e-7

# # A numpy record array (like a struct) to record triangle
# point_data = np.dtype([
#     # conductivity, permittivity and permiability
#     #free space values should be
#     #permittivity of free space 8.8541878176e-12F/m
#     #permeability of free space 1.25663706212e-6H/m
#     ('permittivity', 'c8'), ('permeability', 'c8'),
#     #electric or magnetic current sources? E if True
#     ('Electric', '?'),
#     ], align=True)
# point_t = from_dtype(point_data) # Create a type that numba can recognize!
#
# # A numpy record array (like a struct) to record triangle
# triangle_data = np.dtype([
#     # v0 data
#     ('v0x', 'f4'), ('v0y', 'f4'), ('v0z', 'f4'),
#     # v1 data
#     ('v1x', 'f4'),  ('v1y', 'f4'), ('v1z', 'f4'),
#     # v2 data
#     ('v2x', 'f4'),  ('v2y', 'f4'), ('v2z', 'f4'),
#     # normal vector
#     #('normx', 'f4'),  ('normy', 'f4'), ('normz', 'f4'),
#     # ('reflection', np.float64),
#     # ('diffuse_c', np.float64),
#     # ('specular_c', np.float64),
#     ], align=True)
# triangle_t = from_dtype(triangle_data) # Create a type that numba can recognize!
#
# # ray class, to hold the ray origin, direction, and eventuall other data.
# ray_data=np.dtype([
#     #origin data
#     ('ox','f4'),('oy','f4'),('oz','f4'),
#     #direction vector
#     ('dx','f4'),('dy','f4'),('dz','f4'),
#     #target
#     #direction vector
#     #('tx','f4'),('ty','f4'),('tz','f4'),
#     #distance traveled
#     ('dist','f4'),
#     #intersection
#     ('intersect','?'),
#     ],align=True)
# ray_t = from_dtype(ray_data) # Create a type that numba can recognize!
# # We can use that type in our device functions and later the kernel!
#
# scattering_point = np.dtype([
#     #position data
#     ('px', 'f4'), ('py', 'f4'), ('pz', 'f4'),
#     #velocity
#     ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
#     #normal
#     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#     #weights
#     ('ex','c8'),('ey','c8'),('ez','c8'),
#     # conductivity, permittivity and permiability
#     #free space values should be
#     #permittivity of free space 8.8541878176e-12F/m
#     #permeability of free space 1.25663706212e-6H/m
#     ('permittivity', 'c8'), ('permeability', 'c8'),
#     #electric or magnetic current sources? E if True
#     ('Electric', '?'),
#     ], align=True)
# scattering_t = from_dtype(scattering_point) # Create a type that numba can recognize!

#
# @njit
# def cart2pol(x, y):
#     rho = np.sqrt(x ** 2 + y ** 2)
#     phi = np.arctan2(y, x)
#     return (rho, phi)
#
#
# @njit
# def pol2cart(rho, phi):
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return (x, y)
#
#
# @njit
# def cart2sph(x, y, z):
#     # radians
#     hxy = np.hypot(x, y)
#     r = np.hypot(hxy, z)
#     el = np.arctan2(z, hxy)
#     az = np.arctan2(y, x)
#     return az, el, r
#
#
# @njit
# def sph2cart(az, el, r):
#     # radians
#     rcos_theta = r * np.cos(el)
#     x = rcos_theta * np.cos(az)
#     y = rcos_theta * np.sin(az)
#     z = r * np.sin(el)
#     return x, y, z
#
#
# @njit
# def calc_normals(T):
#     # calculate triangle norm
#     e1x = T["v1x"] - T["v0x"]
#     e1y = T["v1y"] - T["v0y"]
#     e1z = T["v1z"] - T["v0z"]
#
#     e2x = T["v2x"] - T["v0x"]
#     e2y = T["v2y"] - T["v0y"]
#     e2z = T["v2z"] - T["v0z"]
#
#     dirx = e1y * e2z - e1z * e2y
#     diry = e1z * e2x - e1x * e2z
#     dirz = e1x * e2y - e1y * e2x
#
#     normconst = math.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2)
#
#     T["normx"] = dirx / normconst
#     T["normy"] = diry / normconst
#     T["normz"] = dirz / normconst
#
#     return T
#
#
# @guvectorize(
#     [(float32[:], float32[:], float32[:], float32)],
#     "(n),(n)->(n),()",
#     target="parallel",
# )
# def fast_calc_dv(source, target, dv, normconst):
#     dirx = target[0] - source[0]
#     diry = target[1] - source[1]
#     dirz = target[2] - source[2]
#     normconst = math.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2)
#     dv = np.array([dirx, diry, dirz]) / normconst
#
#
# @njit
# def calc_dv(source, target):
#     dirx = target[0] - source[0]
#     diry = target[1] - source[1]
#     dirz = target[2] - source[2]
#     normconst = np.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2)
#     dv = np.array([dirx, diry, dirz]) / normconst
#     return dv[0], dv[1], dv[2], normconst


# @njit
# def calc_dv_norm(source,target,direction,length):
#    length[:,0]=np.sqrt((target[:,0]-source[:,0])**2+(target[:,1]-source[:,1])**2+(target[:,2]-source[:,2])**2)
#    direction=(target-source)/length
#    return direction, length


@cuda.jit(device=True)
def dot(ax1, ay1, az1, ax2, ay2, az2):
    result = ax1 * ax2 + ay1 * ay2 + az1 * az2
    return result


@cuda.jit(device=True)
def cross(ax1, ay1, az1, ax2, ay2, az2):
    rx = ay1 * az2 - az1 * ay2
    ry = az1 * ax2 - ax1 * az2
    rz = ax1 * ay2 - ay1 * ax2
    return rx, ry, rz


# @cuda.jit(boolean,float32(ray_t, triangle_t), device=True, inline=False)
@cuda.jit(device=True, inline=False)
def hit(ray, triangle):
    """ Compute compute whether the defined ray will intersect with the defined triangle
    using the  Möller–Trumbore ray-triangle intersection algorithm
    """
    # find edge vectors
    e1x = triangle.v1x - triangle.v0x
    e1y = triangle.v1y - triangle.v0y
    e1z = triangle.v1z - triangle.v0z

    e2x = triangle.v2x - triangle.v0x
    e2y = triangle.v2y - triangle.v0y
    e2z = triangle.v2z - triangle.v0z
    # calculate determinant cross product of DV and E2
    pvecx, pvecy, pvecz = cross(ray.dx, ray.dy, ray.dz, e2x, e2y, e2z)
    # pvecx=ray.dy*e2z-ray.dz*e2y
    # pvecy=ray.dz*e2x-ray.dx*e2z
    # pvecz=ray.dx*e2y-ray.dy*e2x
    # determinant is dot product of edge 1 and pvec
    A = dot(pvecx, pvecy, pvecz, e1x, e1y, e1z)
    # if A is near zero, then ray lies in the plane of the triangle
    if -EPSILON < A < EPSILON:
        # print('miss')
        return False, math.inf

    # if A is less than zero, then the ray is coming from behind the triangle
    # cull backface triangles
    if A < 0:
        return False, math.inf
    # calculate distance from vertice 0 to ray origin
    tvecx = ray.ox - triangle.v0x  # s
    tvecy = ray.oy - triangle.v0y  # s
    tvecz = ray.oz - triangle.v0z  # s
    # inverse determinant and calculate bounds of triangle
    F = 1.0 / A
    U = F * dot(tvecx, tvecy, tvecz, pvecx, pvecy, pvecz)
    # U=F*(tvecx*pvecx+tvecy*pvecy+tvecz*pvecz)
    # print('U',U)
    if U < 0.0 or U > (1.0):
        # in U coordinates, if U is less than 0 or greater than 1, it is outside the triangle
        # print('miss')
        return False, math.inf

    # cross product of tvec and E1
    qx, qy, qz = cross(tvecx, tvecy, tvecz, e1x, e1y, e1z)
    # qx=tvecy*e1z-tvecz*e1y
    # qy=tvecz*e1x-tvecx*e1z
    # qz=tvecx*e1y-tvecy*e1x
    V = F * dot(ray.dx, ray.dy, ray.dz, qx, qy, qz)
    # print('V,V+U',V,V+U)
    # V=F*(ray.dx*qx+ray.dy*qy+ray.dz*qz)
    if V < 0.0 or (U + V) > (1.0):
        # in UV coordinates, intersection is within triangle
        # print('miss')
        return False, math.inf

    # intersect_distance = F*(e2x*qx + e2y*qy + e2z*qz)
    intersect_distance = F * dot(e2x, e2y, e2z, qx, qy, qz)
    # print('dist',intersect_distance)
    # if (intersect_distance>EPSILON and intersect_distance<(1-EPSILON)):
    if (intersect_distance > (2 * EPSILON)) and (
        intersect_distance < (ray.dist - (2 * EPSILON))
    ):
        # intersection on triangle
        # print('hit')
        return True, intersect_distance

    return False, math.inf


@cuda.jit
def integratedkernal(ray_index, point_information, environment, ray_flag):
    cu_ray_num = cuda.grid(1)  # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),

    if cu_ray_num < ray_index.shape[0]:
        # there are rays to process but must create and launch on the spot, then if succesful go to the next one until the index has been traversed
        # print(ray_index[cu_ray_num,0],ray_index[cu_ray_num,1],ray_index[cu_ray_num,2])
        # ray_components[cu_ray_num,:]=0.0
        # print(scattering_matrix.shape[0],scattering_matrix.shape[1])
        for i in range(ray_index.shape[1] - 1):
            # print('integrated ray',cu_ray_num)
            if ray_index[cu_ray_num, i + 1] != 0:
                #     #print(i,cu_ray_num,network_index[cu_ray_num,i],network_index[cu_ray_num,i+1])
                #     #convert source point field to ray

                ray = cuda.local.array(shape=(1), dtype=base_types.ray_t)
                point1 = point_information[ray_index[cu_ray_num, i] - 1]
                point2 = point_information[ray_index[cu_ray_num, i + 1] - 1]
                ray[0]["ox"] = point1["px"]
                ray[0]["oy"] = point1["py"]
                ray[0]["oz"] = point1["pz"]
                normconst = math.sqrt(
                    (point2["px"] - point1["px"]) ** 2
                    + (point2["py"] - point1["py"]) ** 2
                    + (point2["pz"] - point1["pz"]) ** 2
                )
                ray[0]["dx"] = (point2["px"] - point1["px"]) / normconst
                ray[0]["dy"] = (point2["py"] - point1["py"]) / normconst
                ray[0]["dz"] = (point2["pz"] - point1["pz"]) / normconst
                ray[0]["dist"] = normconst
                ray[0]["intersect"] = False
                # ray=rayprep(point1,point2,ray)
                # print('integrated ray',cu_ray_num,i,normconst)
                for tri_inc in range(len(environment)):
                    hit_bool, dist_temp = hit(ray[0], environment[tri_inc])
                    if hit_bool and (dist_temp < ray[0]["dist"]):
                        # hit something, so break out of loop and null index the first entry of the ray_index
                        ray[0]["dist"] = dist_temp
                        ray[0]["intersect"] = True

            # if ray[0]['intersect']:
            #    break

        ray_flag[cu_ray_num] = ray[0]["intersect"]
        # print('integrated',cu_ray_num,i)


@cuda.jit
def kernel1D(rays, environment, ray_flag, distances):

    cu_ray_num = cuda.grid(1)  # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
    #           threadIdx.y + ( blockIdx.y * blockDim.y )
    # margin=1e-5
    dist_min = math.inf
    intersect_check = False
    max_tri_num = len(environment)
    i = 0  # emulate a C-style for-loop, exposing the idx increment logic
    while i < max_tri_num:
        hit_bool, dist_temp = hit(rays[cu_ray_num], environment[i])
        i += 1
        if hit_bool and (dist_temp < dist_min):
            # hit something
            intersect_check = True
            dist_min = dist_temp

    ray_flag[cu_ray_num] = intersect_check
    distances[cu_ray_num] = dist_min


@cuda.jit
def kernel1Dv2(rays, environment):

    cu_ray_num = cuda.grid(1)  # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
    #           threadIdx.y + ( blockIdx.y * blockDim.y )
    # margin=1e-5
    # dist_min=math.inf
    # intersect_check=False
    max_tri_num = len(environment)
    i = 0  # emulate a C-style for-loop, exposing the idx increment logic
    while i < max_tri_num:
        hit_bool, dist_temp = hit(rays[cu_ray_num], environment[i])
        i += 1
        # if hit_bool:
        #    print(cu_ray_num)
        # if (cu_ray_num==5):
        #    #if (i==58):
        #    hit_bool, dist_temp = hit(rays[cu_ray_num], environment[i])
        #    print(i,dist_temp)

        if hit_bool and (dist_temp < rays[cu_ray_num]["dist"]):
            # hit something
            # print(cu_ray_num,i,dist_temp,'hit')
            # dist_min=dist_temp
            rays[cu_ray_num]["dist"] = dist_temp
            rays[cu_ray_num]["intersect"] = True
            # ray_flag[cu_ray_num]=True

    # if rays[cu_ray_num]['intersect']==False:
    #    print(cu_ray_num,i,dist_temp,'miss')

    nb.cuda.syncthreads()
    # if (rays[cu_ray_num]['intersect']==True):
    #    print(cu_ray_num,rays[cu_ray_num]['dist'])
    # if (cu_ray_num==1):
    #    print(i)


@cuda.jit
def kernel1Dv3(rays, environment):

    cu_ray_num = cuda.grid(1)  # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),

    if (
        cu_ray_num < rays.shape[0]
    ):  #           threadIdx.y + ( blockIdx.y * blockDim.y )
        # margin=1e-5
        # dist_min=math.inf
        # intersect_check=False
        max_tri_num = len(environment)
        i = 0  # emulate a C-style for-loop, exposing the idx increment logic
        while i < max_tri_num:
            hit_bool, dist_temp = hit(rays[cu_ray_num], environment[i])
            i += 1
            # if hit_bool:
            #    print(cu_ray_num)
            # if (cu_ray_num==5):
            #    #if (i==58):
            #    hit_bool, dist_temp = hit(rays[cu_ray_num], environment[i])
            #    print(i,dist_temp)
            if hit_bool and (dist_temp < rays[cu_ray_num]["dist"]):
                # hit something
                # print(cu_ray_num,i,dist_temp,'hit')
                # dist_min=dist_temp
                # print('conventional',cu_ray_num,i,dist_temp,rays[cu_ray_num]['dist'])
                rays[cu_ray_num]["dist"] = dist_temp
                rays[cu_ray_num]["intersect"] = True

                # ray_flag[cu_ray_num]=True

        # print('conventional',cu_ray_num,dist_temp)
        nb.cuda.syncthreads()


def integratedRaycaster(ray_index, scattering_points, environment_local):
    start = timer()

    prep_dt = timer() - start
    raystart = timer()
    # Create a container for the pixel RGBA information of our image
    chunk_size = 2 ** 11
    threads_in_block = 1024
    # for idx in range(len(triangle_chunk)-1):
    d_environment = cuda.device_array(len(environment_local), dtype=base_types.triangle_t)
    d_ray_index = cuda.device_array(
        (ray_index.shape[0], ray_index.shape[1]), dtype=np.int32
    )
    d_ray_index = cuda.to_device(ray_index)
    ray_flags = np.full(ray_index.shape[0], False, dtype=np.bool)
    d_ray_flag = cuda.device_array(ray_index.shape[0], dtype=np.bool)
    d_ray_flag = cuda.to_device(ray_flags)
    cuda.to_device(environment_local, to=d_environment)
    d_point_information = cuda.device_array(
        scattering_points.shape[0], dtype=base_types.scattering_t
    )
    d_point_information = cuda.to_device(scattering_points)
    grids = math.ceil(ray_index.shape[0] / threads_in_block)
    threads = threads_in_block
    # Execute the kernel
    # cuda.profile_start()
    # kernel1Dv2[grids, threads](d_chunk_payload,d_environment,d_ray_flag)
    integratedkernal[grids, threads](
        d_ray_index, d_point_information, d_environment, d_ray_flag
    )
    # cuda.profile_stop()
    # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
    propagation_index = d_ray_index.copy_to_host()
    ray_flags = d_ray_flag.copy_to_host()
    kernel_dt = timer() - raystart
    start = timer()

    mem_dt = timer() - start
    # deallocate memory on gpu
    #ctx = cuda.current_context()
    #deallocs = ctx.deallocations
    #deallocs.clear()
    # final_index=np.delete(propagation_index,np.where(propagation_index[:,0]==0),axis=0)
    final_index = np.delete(ray_index, ray_flags, axis=0)

    return final_index


def visiblespace(
    source_coords,
    source_normals,
    environment,
    vertex_area=0,
    az_range=np.linspace(-180.0, 180.0, 19),
    elev_range=np.linspace(-90.0, 90.0, 19),
    shell_range=0.5,
):
    """
    Visiblespace generates a matrix stack of visible space for each element, indexed by source_coordinate.

    Parameters
    --------------
    source_coords : n by 3 numpy array of floats
        xyz coordinates of the sources
    source_normals : n by 3 numpy array of floats
        normal vectors for each source point
    environment : :class:`lyceanem.base_classes.triangles`
        blocking environment
    vertex_area : float or array of floats
        the area associated with each source point, defaults to 0, but can also be specified for each source
    az_range : array of float
        array of azimuth planes in degrees
    elev_range : array of float
        array of elevation points in degrees
    shell_range: float
        radius of point cloud shell

    Returns
    -------------------
    visible_patterns : m by l by n array of floats
        3D antenna patterns
    resultant_pcd : open3d pointcloud
        colour data to scale the points fractional visibility from the source aperture
    """

    azaz, elel = np.meshgrid(az_range, elev_range)
    sourcenum = len(source_coords)

    sinks = np.zeros((len(np.ravel(azaz)), 3), dtype=np.float32)
    sinks[:, 0], sinks[:, 1], sinks[:, 2] = azeltocart(
        np.ravel(azaz), np.ravel(elel), 1e9
    )
    sinknum = len(sinks)
    #
    # initial_index=np.full((len(source_coords),2),np.nan,dtype=np.int32)
    # initial_index[:,0]=np.arange(len(source_coords))
    # pointingindex=np.reshape(np.arange(0,len(sinks)),(len(az_range),len(elev_range)))

    # need to create farfield sinks in az,elev coordinates, then convert to xyz sink coordinates, and generate index
    # missed_points,hit_points,missed_index,hit_index,shadow_rays=chunkingRaycaster1D(source_coords,sinks,np.zeros((1,3),dtype=np.float32),initial_index,environment,1,terminate_flag=True)
    hit_index, _ = workchunkingv2(
        source_coords, sinks, np.empty((0, 3), dtype=np.float32), environment, 1
    )
    unified_model = np.append(
        source_coords.astype(np.float32), sinks.astype(np.float32), axis=0
    )
    # filtered_network2,final_network2,filtered_index2,final_index2,shadow_rays
    directions = np.zeros((len(hit_index), 3), dtype=np.float32)
    norm_length = np.zeros((len(hit_index), 1), dtype=np.float32)
    hit_directions, hit_norms = math_functions.calc_dv_norm(
        unified_model[hit_index[:, 0].astype(int) - 1, :],
        unified_model[hit_index[:, 1].astype(int) - 1, :],
        directions,
        norm_length,
    )
    # angles=Angular_distance(hit_directions,source_normals[hit_index[:,0].astype(int)-1,:].astype(np.float32))
    angles = np.zeros((hit_directions.shape[0], 1), dtype=np.float32)
    angles = angle_pop(
        hit_directions,
        source_normals[hit_index[:, 0].astype(int) - 1, :].astype(np.float32),
        angles,
    )

    # angles[np.isnan(angles)]=0
    # visible_patterns=quickpatterncreator(az_range,elev_range,source_coords,angles,vertex_area,hit_index)
    if len(vertex_area) == 1:
        if vertex_area == 0:
            portion = np.zeros((len(angles), 1), dtype=np.float32)
            portion[:] = 1
        else:
            portion = vertex_area * np.abs(np.cos(angles))
            portion[portion < 0] = 0
    else:
        portion = np.ravel(vertex_area[hit_index[:, 0].astype(int) - 1]) * np.abs(
            np.ravel(np.cos(angles))
        )

    portion[np.where(np.abs(angles) > (np.pi / 2))[0]] = 0.0
    visible_patterns = np.empty((len(az_range) * len(elev_range)), dtype=np.float32)
    visible_patterns[:] = 0
    visible_patterns = patternsort(
        visible_patterns, sourcenum, sinknum, portion, hit_index
    ).reshape(len(elev_range), len(az_range))
    shell_coords = np.zeros((len(np.ravel(azaz)), 3), dtype=np.float32)
    shell_coords[:, 0], shell_coords[:, 1], shell_coords[:, 2] = azeltocart(
        np.ravel(azaz), np.ravel(elel), shell_range
    )
    maxarea = np.nanmax(np.nanmax(visible_patterns))
    resultant_pcd = patterntocloud(
        np.reshape(visible_patterns, (len(az_range) * len(elev_range), 1)),
        shell_coords,
        maxarea,
    )

    return visible_patterns, resultant_pcd


@njit(cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


@njit(cache=True, nogil=True)
def angle(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)


@njit(parallel=True)
def angle_pop(hit_directions, source_normals, angles):
    # parallised angle calculation
    for angle_inc in prange(len(angles)):
        angles[angle_inc] = angle(
            hit_directions[angle_inc, :], source_normals[angle_inc, :]
        )

    return angles


#@njit(parallel=True)
def patternsort(visible_patterns, sourcenum, sinknum, portion, hit_index):
    # faster method of summing the array contributions to each farfield point
    for n in range(sinknum):
        visible_patterns[n] = np.sum(portion[hit_index[:, 1] - 1 - sourcenum == n])

    return visible_patterns


def patterncreator(az_range, elev_range, source_coords, pointingindex, hit_index):
    # creates visibility pattern from azimuth and elevation ranges
    visible_patterns = np.empty(
        (len(az_range), len(elev_range), len(source_coords)), dtype=np.int32
    )
    visible_patterns[:, :, :] = 0
    for n in range(len(az_range)):
        for m in range(len(elev_range)):
            for element in range(len(source_coords)):
                if np.any(
                    np.all(
                        np.asarray(
                            [
                                hit_index[:, 0] == element,
                                hit_index[:, 1] == pointingindex[n, m],
                            ]
                        ),
                        axis=0,
                    )
                ):
                    visible_patterns[n, m, element] = 1
    return visible_patterns


def quickpatterncreator(
    az_range, elev_range, source_coords, angles, vertex_area, hit_index
):
    # creates visibility pattern from azimuth and elevation ranges
    visible_patterns = np.empty((len(az_range), len(elev_range)), dtype=np.float32)
    visible_patterns[:, :] = 0
    pointingindex = np.reshape(
        np.arange(0, len(az_range) * len(elev_range)), (len(az_range), len(elev_range))
    )
    sourcenum = len(source_coords)
    if vertex_area == 0:
        for n in range(len(az_range)):
            for m in range(len(elev_range)):
                visible_patterns[n, m] = np.sum(
                    hit_index[:, 1] - sourcenum - 1 == pointingindex[n, m]
                )
    else:
        portion = vertex_area * np.cos(angles)
        portion[portion < 0] = 0
        for n in range(len(az_range)):
            for m in range(len(elev_range)):
                visible_patterns[n, m] = np.sum(
                    portion[hit_index[:, 1] - sourcenum - 1 == pointingindex[n, m]]
                )

    return visible_patterns


# @njit
def patterntocloud(pattern_data, shell_coords, maxarea):
    # takes the pattern_data and shell_coordinates, and creates an open3d point cloud based upon the data.
    point_cloud = points2pointcloud(shell_coords)
    points, elements = np.shape(pattern_data)
    # normdata
    viridis = cm.get_cmap("viridis", 40)
    visible_elements = np.sum(pattern_data / maxarea, axis=1)
    np_colors = viridis(visible_elements)
    point_cloud.colors = o3d.utility.Vector3dVector(np_colors[:, 0:3])
    return point_cloud


def azeltocart(az_data, el_data, radius):
    # convert from az,el and radius data to xyz
    x_data = radius * np.cos(np.deg2rad(el_data)) * np.cos(np.deg2rad(az_data))
    y_data = radius * np.cos(np.deg2rad(el_data)) * np.sin(np.deg2rad(az_data))
    z_data = radius * np.sin(np.deg2rad(el_data))
    return x_data, y_data, z_data


def convertTriangles(triangle_object):
    """
    convert o3d triangle object to ray tracer triangle class
    """
    if triangle_object == None:
        triangles = np.empty(0, dtype=base_types.triangle_t)
    else:
        vertices = np.asarray(triangle_object.vertices)
        tri_index = np.asarray(triangle_object.triangles)
        normals = np.asarray(triangle_object.triangle_normals)
        triangles = np.empty(len(tri_index), dtype=base_types.triangle_t)
        for idx in range(len(tri_index)):
            triangles[idx]["v0x"] = np.single(vertices[tri_index[idx, 0], 0])
            triangles[idx]["v0y"] = np.single(vertices[tri_index[idx, 0], 1])
            triangles[idx]["v0z"] = np.single(vertices[tri_index[idx, 0], 2])
            triangles[idx]["v1x"] = np.single(vertices[tri_index[idx, 1], 0])
            triangles[idx]["v1y"] = np.single(vertices[tri_index[idx, 1], 1])
            triangles[idx]["v1z"] = np.single(vertices[tri_index[idx, 1], 2])
            triangles[idx]["v2x"] = np.single(vertices[tri_index[idx, 2], 0])
            triangles[idx]["v2y"] = np.single(vertices[tri_index[idx, 2], 1])
            triangles[idx]["v2z"] = np.single(vertices[tri_index[idx, 2], 2])
            # triangles[idx]['normx']=np.single(normals[idx,0])
            # triangles[idx]['normy']=np.single(normals[idx,1])
            # triangles[idx]['normz']=np.single(normals[idx,2])

    return triangles


def pick_points(pcd):
    """
    test function based on open3d example to pick points, can be used as the basis of a selection function to pick vertices.
    """
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def points2pointcloud(xyz):
    """
    turns numpy array of xyz data into a o3d format point cloud
    Parameters
    ----------
    xyz : TYPE
        DESCRIPTION.

    Returns
    -------
    pcd : point cloud format

    """
    pcd = o3d.geometry.PointCloud()
    if xyz.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(xyz)
    else:
        reshaped = xyz.reshape((int(len(xyz.ravel()) / 3), 3))
        pcd.points = o3d.utility.Vector3dVector(reshaped)

    return pcd


@guvectorize([(float32[:, :], float32[:])], "(m,n)->(m)", target="parallel")
def PathLength(partial_network, path_length):
    """
    for each row in the partial_network, calculate the distance travelled from the origin to the sink point
    #
    Parameters
    ----------
    partial_network : n by (3*m) array of coordinates
    #
    Returns
    -------
    path_lengths : n by 1 array of path lengths between each point.
    #
    """
    if partial_network.shape[1] == 6:
        for i in range(partial_network.shape[0]):
            path_length[i] = np.sqrt(
                (partial_network[i, 3] - partial_network[i, 0]) ** 2
                + (partial_network[i, 4] - partial_network[i, 1]) ** 2
                + (partial_network[i, 5] - partial_network[i, 2]) ** 2
            )
    elif partial_network.shape[1] == 9:
        for i in range(partial_network.shape[0]):
            path_length[i] = np.sqrt(
                (partial_network[i, 3] - partial_network[i, 0]) ** 2
                + (partial_network[i, 4] - partial_network[i, 1]) ** 2
                + (partial_network[i, 5] - partial_network[i, 2]) ** 2
            ) + np.sqrt(
                (partial_network[i, 6] - partial_network[i, 3]) ** 2
                + (partial_network[i, 7] - partial_network[i, 4]) ** 2
                + (partial_network[i, 8] - partial_network[i, 5]) ** 2
            )
    elif partial_network.shape[1] == 12:
        for i in range(partial_network.shape[0]):
            path_length[i] = (
                np.sqrt(
                    (partial_network[i, 3] - partial_network[i, 0]) ** 2
                    + (partial_network[i, 4] - partial_network[i, 1]) ** 2
                    + (partial_network[i, 5] - partial_network[i, 2]) ** 2
                )
                + np.sqrt(
                    (partial_network[i, 6] - partial_network[i, 3]) ** 2
                    + (partial_network[i, 7] - partial_network[i, 4]) ** 2
                    + (partial_network[i, 8] - partial_network[i, 5]) ** 2
                )
                + np.sqrt(
                    (partial_network[i, 9] - partial_network[i, 6]) ** 2
                    + (partial_network[i, 10] - partial_network[i, 7]) ** 2
                    + (partial_network[i, 11] - partial_network[i, 8]) ** 2
                )
            )


# @guvectorize([(int32[:,:], float32[:,:], float32[:,:], float32[:,:], float32, complex64[:,:])], '(m,n)->(m,n)',target='parallel')
@njit(parallel=True, nogil=True, cache=True)
def ScatteringNetworkGenerator(
    network_index,
    scattering_points,
    scattering_normals,
    scattering_weights,
    point_information,
    scattering_coefficient,
    wavelength,
    local_channel,
):
    """
    for each row in the partial_network, calculate the distance travelled from the origin to the sink point
    #
    Parameters
    ----------
    partial_network : n by (3*m) array of coordinates
    #
    Returns
    -------
    local_channel : n by 3 array of electric field vectors at each sink
    #
    """
    if network_index.shape[1] == 2:
        for i in prange(network_index.shape[0]):
            local_channel[i, :] = EM.losChannel(
                scattering_points[network_index[i, 0] - 1, :],
                scattering_normals[network_index[i, 0] - 1, :],
                scattering_weights[network_index[i, 0] - 1, :],
                point_information[network_index[i, 0] - 1],
                scattering_points[network_index[i, 1] - 1, :],
                scattering_normals[network_index[i, 1] - 1, :],
                scattering_weights[network_index[i, 1] - 1, :],
                point_information[network_index[i, 1] - 1],
                scattering_coefficient,
                wavelength,
            )
    elif network_index.shape[1] == 3:
        for i in prange(network_index.shape[0]):
            local_channel[i, :] = EM.losplus1Channel(
                scattering_points[network_index[i, 0] - 1, :],
                scattering_normals[network_index[i, 0] - 1, :],
                scattering_weights[network_index[i, 0] - 1, :],
                point_information[network_index[i, 0] - 1],
                scattering_points[network_index[i, 1] - 1, :],
                scattering_normals[network_index[i, 1] - 1, :],
                scattering_weights[network_index[i, 1] - 1, :],
                point_information[network_index[i, 1] - 1],
                scattering_points[network_index[i, 2] - 1, :],
                scattering_normals[network_index[i, 2] - 1, :],
                scattering_weights[network_index[i, 2] - 1, :],
                point_information[network_index[i, 2] - 1],
                scattering_coefficient,
                wavelength,
            )
    elif network_index.shape[1] == 4:
        for i in prange(network_index.shape[0]):
            local_channel[i, :] = EM.losplus2Channel(
                scattering_points[network_index[i, 0] - 1, :],
                scattering_normals[network_index[i, 0] - 1, :],
                scattering_weights[network_index[i, 0] - 1, :],
                point_information[network_index[i, 0] - 1],
                scattering_points[network_index[i, 1] - 1, :],
                scattering_normals[network_index[i, 1] - 1, :],
                scattering_weights[network_index[i, 1] - 1, :],
                point_information[network_index[i, 1] - 1],
                scattering_points[network_index[i, 2] - 1, :],
                scattering_normals[network_index[i, 2] - 1, :],
                scattering_weights[network_index[i, 2] - 1, :],
                point_information[network_index[i, 2] - 1],
                scattering_points[network_index[i, 3] - 1, :],
                scattering_normals[network_index[i, 3] - 1, :],
                scattering_weights[network_index[i, 3] - 1, :],
                point_information[network_index[i, 3] - 1],
                scattering_coefficient,
                wavelength,
            )

    return local_channel


@njit(parallel=True, nogil=True, cache=True)
def ScatteringNetworkGeneratorTime(
    network_index,
    scattering_points,
    scattering_normals,
    scattering_weights,
    point_information,
    wavelength,
    local_channel,
    times,
):
    """
    for each row in the partial_network, calculate the distance travelled from the origin to the sink point
    #
    Parameters
    ----------
    partial_network : n by (3*m) array of coordinates
    #
    Returns
    -------
    local_channel : n by 3 array of electric field vectors at each sink
    #
    """
    if network_index.shape[1] == 2:
        for i in prange(network_index.shape[0]):
            local_channel[i, :], times[i] = EM.losChannelv2(
                scattering_points[network_index[i, 0] - 1, :],
                scattering_normals[network_index[i, 0] - 1, :],
                scattering_weights[network_index[i, 0] - 1, :],
                point_information[network_index[i, 0] - 1],
                scattering_points[network_index[i, 1] - 1, :],
                scattering_normals[network_index[i, 1] - 1, :],
                scattering_weights[network_index[i, 1] - 1, :],
                point_information[network_index[i, 1] - 1],
                wavelength,
            )
    elif network_index.shape[1] == 3:
        for i in prange(network_index.shape[0]):
            local_channel[i, :], times[i] = EM.losplus1Channelv2(
                scattering_points[network_index[i, 0] - 1, :],
                scattering_normals[network_index[i, 0] - 1, :],
                scattering_weights[network_index[i, 0] - 1, :],
                point_information[network_index[i, 0] - 1],
                scattering_points[network_index[i, 1] - 1, :],
                scattering_normals[network_index[i, 1] - 1, :],
                scattering_weights[network_index[i, 1] - 1, :],
                point_information[network_index[i, 1] - 1],
                scattering_points[network_index[i, 2] - 1, :],
                scattering_normals[network_index[i, 2] - 1, :],
                scattering_weights[network_index[i, 2] - 1, :],
                point_information[network_index[i, 2] - 1],
                wavelength,
            )
    elif network_index.shape[1] == 4:
        for i in prange(network_index.shape[0]):
            local_channel[i, :], times[i] = EM.losplus2Channelv2(
                scattering_points[network_index[i, 0] - 1, :],
                scattering_normals[network_index[i, 0] - 1, :],
                scattering_weights[network_index[i, 0] - 1, :],
                point_information[network_index[i, 0] - 1],
                scattering_points[network_index[i, 1] - 1, :],
                scattering_normals[network_index[i, 1] - 1, :],
                scattering_weights[network_index[i, 1] - 1, :],
                point_information[network_index[i, 1] - 1],
                scattering_points[network_index[i, 2] - 1, :],
                scattering_normals[network_index[i, 2] - 1, :],
                scattering_weights[network_index[i, 2] - 1, :],
                point_information[network_index[i, 2] - 1],
                scattering_points[network_index[i, 3] - 1, :],
                scattering_normals[network_index[i, 3] - 1, :],
                scattering_weights[network_index[i, 3] - 1, :],
                point_information[network_index[i, 3] - 1],
                wavelength,
            )

    return local_channel, times


@njit(parallel=True)
def ScatteringNetworkGeneratorv2(
    network_index,
    scattering_points,
    scattering_normals,
    scattering_weights,
    point_information,
    wavelength,
    local_channel,
):
    """
    for each row in the partial_network, calculate the distance travelled from the origin to the sink point
    #
    Parameters
    ----------
    partial_network : n by (3*m) array of coordinates
    #
    Returns
    -------
    local_channel : n by 3 array of electric field vectors at each sink
    #
    """
    if network_index.shape[1] == 2:
        local_channel = EM.losChannel(
            scattering_points[network_index[0] - 1],
            scattering_normals[network_index[0] - 1],
            scattering_weights[network_index[0] - 1],
            point_information[network_index[0] - 1],
            scattering_points[network_index[1] - 1],
            scattering_normals[network_index[1] - 1],
            scattering_weights[network_index[1] - 1],
            point_information[network_index[1] - 1],
            wavelength,
        )
    elif network_index.shape[1] == 3:
        local_channel = EM.losplus1Channel(
            scattering_points[network_index[0] - 1],
            scattering_normals[network_index[0] - 1],
            scattering_weights[network_index[0] - 1],
            point_information[network_index[0] - 1],
            scattering_points[network_index[1] - 1],
            scattering_normals[network_index[1] - 1],
            scattering_weights[network_index[1] - 1],
            point_information[network_index[1] - 1],
            scattering_points[network_index[2] - 1],
            scattering_normals[network_index[2] - 1],
            scattering_weights[network_index[2] - 1],
            point_information[network_index[2] - 1],
            wavelength,
        )
    elif network_index.shape[1] == 4:
        local_channel = EM.losplus2Channel(
            scattering_points[network_index[0] - 1],
            scattering_normals[network_index[0] - 1],
            scattering_weights[network_index[0] - 1],
            point_information[network_index[0] - 1],
            scattering_points[network_index[1] - 1],
            scattering_normals[network_index[1] - 1],
            scattering_weights[network_index[1] - 1],
            point_information[network_index[1] - 1],
            scattering_points[network_index[2] - 1],
            scattering_normals[network_index[2] - 1],
            scattering_weights[network_index[2] - 1],
            point_information[network_index[2] - 1],
            scattering_points[network_index[3] - 1],
            scattering_normals[network_index[3] - 1],
            scattering_weights[network_index[3] - 1],
            point_information[network_index[3] - 1],
            wavelength,
        )

    return local_channel


def PoyntingVector(partial_network):
    """
    for each row in the partial_network, calculate the distance travelled from the origin to the sink point
    #
    Parameters
    ----------
    partial_network : n by 6 array of coordinates
    #
    Returns
    -------
    pointing_vectors : n by 3 array of path lengths between each point.
    #
    """
    pointing_vectors = np.zeros((partial_network.shape[0], 3), dtype=np.float32)
    pointing_mags = np.zeros((partial_network.shape[0], 1), dtype=np.float32)
    for i in range(partial_network.shape[0]):
        pointing_vectors[i, :] = partial_network[i, -3:] - partial_network[i, -6:-3]
        pointing_mags[i] = np.sqrt(
            pointing_vectors[i, 0] ** 2
            + pointing_vectors[i, 1] ** 2
            + pointing_vectors[i, 2] ** 2
        )

    pointing_vectors = pointing_vectors / pointing_mags
    return pointing_vectors


def AngleofArrivalVectors(partial_network):
    """
    calculate the vector from the receiving point to the incoming ray, to calculate angle of arrival correctly
    #
    Parameters
    ----------
    partial_network : n by 6 array of coordinates
    #
    Returns
    -------
    pointing_vectors : n by 3 array of path lengths between each point.
    #
    """
    pointing_vectors = np.zeros((partial_network.shape[0], 3), dtype=np.float32)
    pointing_mags = np.zeros((partial_network.shape[0], 1), dtype=np.float32)
    for i in range(partial_network.shape[0]):
        pointing_vectors[i, :] = partial_network[i, -6:-3] - partial_network[i, -3:]
        pointing_mags[i] = np.sqrt(
            pointing_vectors[i, 0] ** 2
            + pointing_vectors[i, 1] ** 2
            + pointing_vectors[i, 2] ** 2
        )

    pointing_vectors = pointing_vectors / pointing_mags
    return pointing_vectors


def CalculatePoyntingVectors(
    total_network,
    wavelength,
    scattering_index,
    ideal_vector=np.asarray([[1, 0, 0]]),
    az_bins=np.linspace(-np.pi, np.pi, 19),
    el_bins=np.linspace(-np.pi / 2.0, np.pi / 2.0, 19),
    time_bins=np.linspace(-1e-9, 1.9e-8, 20),
    impulse=False,
    aoa=True,
):
    """
    Takes the total network generated using the raycasting process, and calculates the angle of arrival spectrum, delay spectrum, and angular standard deviation as a measure of `farfield ness' of the arriving waves'

    Parameters
    ----------
    total_network : array of n*(m*3) tuples
        the coordinates of each interaction point
    wavelength : float
        wavelength of interest, currently a single value (SI units)
    scattering_index : n*2 array of integers
        the source and sink index of each ray
    ideal_vector : 1*3 array
        the direction vector of the `ideal' incoming ray
    time bins : 1d numpy array of floats
        default is 20 bins, but this should always be set by the user to a sampling rate sufficent for twice the expected highest frequency

    Returns
    -------
    aoa_spectrum : array of size sinknum*len(theta_bins)
        angle of arrival spectrum
    impulse_response :

    travel_times :

    """
    max_path_divergence = 2.0
    if len(total_network) == 0:
        sourcenum = np.int(np.nanmax(scattering_index[:, 0])) + 1
        sinknum = np.int(np.nanmax(scattering_index[:, 1])) + 1
        aoa_spectrum = np.zeros((sinknum, az_bins.shape[0]), dtype=np.float32)
        delay_spectrum = np.zeros((sinknum, az_bins.shape[0]), dtype=np.float32)
    else:
        wave_vector = (2 * np.pi) / wavelength
        sourcenum = np.int(np.nanmax(scattering_index[:, 0])) + 1
        sinknum = np.int(np.nanmax(scattering_index[:, 1])) + 1
        scatterdepth = int((len(total_network[0, :]) - 3) / 3)
        impulse_response = np.zeros((len(time_bins), sinknum), dtype="complex")
        angle_response = np.zeros((len(az_bins), sinknum), dtype="complex")
        aoa_spectrum = np.zeros((sinknum, az_bins.shape[0]), dtype=np.float32)
        delay_spectrum = np.zeros((sinknum, az_bins.shape[0]), dtype=np.float32)
        az_std = np.zeros((sinknum), dtype=np.float32)
        averaged_response = np.zeros((sinknum, 1), dtype="complex")
        phase_std = np.zeros((sinknum), dtype=np.float32)
        total_pointing_vectors = np.empty((0, 3), dtype=np.float32)
        paths = np.empty((0, 1), dtype=np.float32).ravel()
        for idx in range(scatterdepth):
            if idx == (scatterdepth - 1):
                # indexing
                if scatterdepth == 1:
                    start_row = 0
                    end_row = len(total_network[:, 1]) - 1
                else:
                    start_row = (
                        np.max(
                            np.where((np.isnan(total_network[:, 6 + ((idx - 1) * 3)])))
                        )
                        + 1
                    )
                    end_row = len(total_network[:, 1]) - 1

                end_col = (idx * 3) + 6
                # engine here
                paths = np.append(
                    paths,
                    PathLength(total_network[start_row : end_row + 1, :end_col]),
                    axis=0,
                )
                total_pointing_vectors = np.append(
                    total_pointing_vectors,
                    AngleofArrivalVectors(
                        total_network[start_row : end_row + 1, :end_col]
                    ),
                    axis=0,
                )

            elif idx == 0:
                # indexing
                start_row = 0
                end_row = np.max(np.where((np.isnan(total_network[:, 6 + (idx * 3)]))))
                end_col = (idx * 3) + 6
                # engine here
                paths = np.append(
                    paths,
                    PathLength(total_network[start_row : end_row + 1, :end_col]),
                    axis=0,
                )
                total_pointing_vectors = np.append(
                    total_pointing_vectors,
                    AngleofArrivalVectors(
                        total_network[start_row : end_row + 1, :end_col]
                    ),
                    axis=0,
                )

            else:
                # indexing
                start_row = (
                    np.max(np.where((np.isnan(total_network[:, 3 + (idx * 3)])))) + 1
                )
                end_row = np.max(
                    np.where((np.isnan(total_network[:, 6 + ((idx) * 3)])))
                )
                end_col = (idx * 3) + 6
                # engine here
                paths = np.append(
                    paths,
                    PathLength(total_network[start_row : end_row + 1, :end_col]),
                    axis=0,
                )
                total_pointing_vectors = np.append(
                    total_pointing_vectors,
                    AngleofArrivalVectors(
                        total_network[start_row : end_row + 1, :end_col]
                    ),
                    axis=0,
                )

            #
            # scatter_map indexing
        path_loss = wavelength / (4 * np.pi * paths)
        ray_components = path_loss * (np.exp(paths * wave_vector * 1j))
        phase_components = paths * wave_vector
        angular_dist = Angular_distance(ideal_vector, total_pointing_vectors)
        travel_times = paths / scipy.constants.c

        for sink_index in range(sinknum):
            sink_indexing = np.where((scattering_index[:, 1] == sink_index))
            if impulse:
                impulse_response = ImpulseStack(
                    impulse_response,
                    ray_components,
                    sink_index,
                    sink_indexing,
                    travel_times,
                    time_bins,
                )
                # impulse_bin_indexing=np.digitize(travel_times[sink_indexing],time_bins+travel_times[np.argmax(ray_components)],right=True)
                # for time_index in range(len(ray_components[sink_indexing])):
                #    impulse_response[impulse_bin_indexing[time_index],sink_index]=impulse_response[impulse_bin_indexing[time_index],sink_index]+ray_components[sink_indexing][time_index]

            if aoa:
                angle_response = angleofarrivalStack(
                    angle_response,
                    ray_components,
                    sink_index,
                    sink_indexing,
                    angular_dist,
                    az_bins,
                )
                # angle_bin_indexing=np.digitize(angular_dist[sink_indexing],az_bins,right=True)
                # for angle_index in range(len(ray_components[sink_indexing])):
                #    angle_response[angle_bin_indexing[angle_index],sink_index]=angle_response[angle_bin_indexing[angle_index],sink_index]+ray_components[sink_indexing][angle_index]

    return angle_response, impulse_response, travel_times[np.argmax(ray_components)]


@njit(parallel=True)
def ImpulseStack(
    impulse_response, ray_components, sink_index, sink_indexing, travel_times, time_bins
):
    # attempt to impove the efficiency of the digitize action
    impulse_bin_indexing = np.digitize(
        travel_times[sink_indexing],
        time_bins + travel_times[np.argmax(ray_components)],
        right=True,
    )
    for time_index in prange(len(ray_components[sink_indexing])):
        impulse_response[impulse_bin_indexing[time_index], sink_index] = (
            impulse_response[impulse_bin_indexing[time_index], sink_index]
            + ray_components[sink_indexing][time_index]
        )

    return impulse_response


@njit(parallel=True)
def angleofarrivalStack(
    angle_response, ray_components, sink_index, sink_indexing, angular_dist, az_bins
):
    # improved efficiency digitizatoin
    angle_bin_indexing = np.digitize(angular_dist[sink_indexing], az_bins, right=True)
    for angle_index in prange(len(ray_components[sink_indexing])):
        angle_response[angle_bin_indexing[angle_index], sink_index] = (
            angle_response[angle_bin_indexing[angle_index], sink_index]
            + ray_components[sink_indexing][angle_index]
        )

    return angle_response


@njit(parallel=True)
def Angular_distance(center_vector, pointing_vectors):
    if len(center_vector) == 1:
        angle_lists = np.zeros((len(pointing_vectors), 1), dtype=np.float32)
        for i in range(len(pointing_vectors)):
            angle_lists[i] = np.arccos(np.dot(center_vector, pointing_vectors[i, :]))
            # angle_lists[i]=np.arctan2(pointing_vectors[i,1],pointing_vectors[i,0])
    else:
        angle_lists = np.zeros((len(pointing_vectors), 1), dtype=np.float32)
        for i in range(len(pointing_vectors)):
            angle_lists[i] = np.arccos(
                np.dot(center_vector[i, :], pointing_vectors[i, :])
            )
            # angle_lists[i]=np.arctan2(pointing_vectors[i,1],pointing_vectors[i,0])

    return angle_lists


@jit
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation
    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


# @jit
def VectorNetworkProcessEM(
    sourcenum,
    sinknum,
    unified_points,
    unified_normals,
    unified_weights,
    point_information,
    scattering_index,
    scattering_coefficient,
    wavelength,
):
    """
    A more efficienct version of BaseNetworkProcess, using the vectorised path length function to process the total_network array more efficienctly.
    #
    Parameters
    ----------
    total_network : array of n*(m*3) tuples, the coordinates of each interaction point
    wavelength : scaler
        wavelength of interest, currently a single value (SI units)
    scattering_index : n*2 array of integers, the source and sink index of each ray
    #
    Returns
    -------
    scatter_map : source_size*sink_size array
    #
    """
    if scattering_index.shape[0] == 0:
        scatter_map = np.zeros((sourcenum, sinknum, 3), dtype="complex")
    else:
        wave_vector = (2 * np.pi) / wavelength
        scatterdepth = scattering_index.shape[1] - 1
        scatter_map = np.zeros((sourcenum, sinknum, 3, scatterdepth), dtype="complex")
        for idx in range(scatterdepth):
            if idx == (scatterdepth - 1):
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), :
                ]
                ray_components = np.zeros(
                    (ray_analysis_index.shape[0], 3), dtype=np.complex64
                )
                ray_components = ScatteringNetworkGenerator(
                    ray_analysis_index,
                    unified_points,
                    unified_normals,
                    unified_weights,
                    point_information,
                    scattering_coefficient,
                    wavelength,
                    ray_components,
                )
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]

            elif idx == 0:
                ray_analysis_index = scattering_index[
                    np.equal(scattering_index[:, idx + 2], 0), 0 : idx + 2
                ]
                ray_components = np.zeros(
                    (ray_analysis_index.shape[0], 3), dtype=np.complex64
                )
                ray_components = ScatteringNetworkGenerator(
                    ray_analysis_index,
                    unified_points,
                    unified_normals,
                    unified_weights,
                    point_information,
                    scattering_coefficient,
                    wavelength,
                    ray_components,
                )
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index

            else:
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), 0 : idx + 2
                ]
                ray_components = np.zeros(
                    (ray_analysis_index.shape[0], 3), dtype=np.complex64
                )
                ray_components = ScatteringNetworkGenerator(
                    ray_analysis_index,
                    unified_points,
                    unified_normals,
                    unified_weights,
                    point_information,
                    scattering_coefficient,
                    wavelength,
                    ray_components,
                )
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]
            #
            # scatter_map indexing
            # for source_index in range(sourcenum):
            #    for sink_index in range(sinknum):
            #        scatter_map[source_index,sink_index,idx]=np.sum(ray_components[(depth_slice[:,0]-1==0)&(depth_slice[:,1]-sourcenum-1==0)])

            scatter_map = scatter_net_sortEMtest(
                sourcenum, sinknum, scatter_map, depth_slice, ray_components, idx
            )

    return scatter_map


def VectorNetworkProcessTimeEM(
    sourcenum,
    sinknum,
    unified_points,
    unified_normals,
    unified_weights,
    point_information,
    scattering_index,
    wavelength,
):
    """
    A more efficienct version of BaseNetworkProcess, using the vectorised path length function to process the total_network array more efficienctly.
    #
    Parameters
    ----------
    total_network : array of n*(m*3) tuples, the coordinates of each interaction point
    wavelength : scaler
        wavelength of interest, currently a single value (SI units)
    scattering_index : n*2 array of integers, the source and sink index of each ray
    #
    Returns
    -------
    scatter_map : source_size*sink_size array
    #
    """
    if scattering_index.shape[0] == 0:
        scatter_map = np.zeros((sourcenum, sinknum, 3), dtype="complex")
    else:
        time_index = np.linspace(1e-8, 2e-6, 100000)
        wave_vector = (2 * np.pi) / wavelength
        scatterdepth = scattering_index.shape[1] - 1
        scatter_map = np.zeros(
            (sourcenum, sinknum, 3, len(time_index), scatterdepth), dtype="complex"
        )
        for idx in range(scatterdepth):
            if idx == (scatterdepth - 1):
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), :
                ]
                ray_components = np.zeros(
                    (ray_analysis_index.shape[0], 3), dtype=np.complex64
                )
                time_components = np.zeros(
                    (ray_analysis_index.shape[0]), dtype=np.float32
                )
                ray_components, time_components = ScatteringNetworkGeneratorTime(
                    ray_analysis_index,
                    unified_points,
                    unified_normals,
                    unified_weights,
                    point_information,
                    wavelength,
                    ray_components,
                    time_components,
                )
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]

            elif idx == 0:
                ray_analysis_index = scattering_index[
                    np.equal(scattering_index[:, idx + 2], 0), 0 : idx + 2
                ]
                ray_components = np.zeros(
                    (ray_analysis_index.shape[0], 3), dtype=np.complex64
                )
                time_components = np.zeros(
                    (ray_analysis_index.shape[0]), dtype=np.float32
                )
                ray_components, time_components = ScatteringNetworkGeneratorTime(
                    ray_analysis_index,
                    unified_points,
                    unified_normals,
                    unified_weights,
                    point_information,
                    wavelength,
                    ray_components,
                    time_components,
                )
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index

            else:
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), 0 : idx + 2
                ]
                ray_components = np.zeros(
                    (ray_analysis_index.shape[0], 3), dtype=np.complex64
                )
                time_components = np.zeros(
                    (ray_analysis_index.shape[0]), dtype=np.float32
                )
                ray_components, time_components = ScatteringNetworkGeneratorTime(
                    ray_analysis_index,
                    unified_points,
                    unified_normals,
                    unified_weights,
                    point_information,
                    wavelength,
                    ray_components,
                    time_components,
                )
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]
            #
            # scatter_map indexing
            # for source_index in range(sourcenum):
            #    for sink_index in range(sinknum):
            #        scatter_map[source_index,sink_index,idx]=np.sum(ray_components[(depth_slice[:,0]-1==0)&(depth_slice[:,1]-sourcenum-1==0)])

            scatter_map = scatter_net_sortTimeEM(
                sourcenum,
                sinknum,
                scatter_map,
                depth_slice,
                ray_components,
                time_components,
                time_index,
                idx,
            )

    return scatter_map


def VectorNetworkProcessEMv2(
    sourcenum,
    sinknum,
    unified_points,
    unified_normals,
    unified_weights,
    point_information,
    scattering_index,
    wavelength,
):
    """
    A more efficienct version of BaseNetworkProcess, using the vectorised path length function to process the total_network array more efficienctly.
    #
    Parameters
    ----------
    total_network : array of n*(m*3) tuples, the coordinates of each interaction point
    wavelength : scaler
        wavelength of interest, currently a single value (SI units)
    scattering_index : n*2 array of integers, the source and sink index of each ray
    #
    Returns
    -------
    scatter_map : source_size*sink_size array
    #
    """
    if scattering_index.shape[0] == 0:
        scatter_map = np.zeros((sourcenum, sinknum, 3), dtype="complex")
    else:
        wave_vector = (2 * np.pi) / wavelength
        scatterdepth = scattering_index.shape[1] - 1
        scatter_map = np.zeros((sourcenum, sinknum, 3, scatterdepth), dtype="complex")
        ray_components = np.zeros((scattering_index.shape[0], 3), dtype=np.complex64)
        ray_components = EM.EMGPUWrapper(
            scattering_index, point_information, wavelength
        )
        for idx in range(scatterdepth):
            if idx == (scatterdepth - 1):
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), :
                ]
                temp_components = ray_components[
                    ~np.equal(scattering_index[:, idx + 1], 0), :
                ]
                # ray_components=ScatteringNetworkGenerator(ray_analysis_index,unified_points,unified_normals,unified_weights,point_information,wavelength,ray_components)
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]

            elif idx == 0:
                ray_analysis_index = scattering_index[
                    np.equal(scattering_index[:, idx + 2], 0), 0 : idx + 2
                ]
                temp_components = ray_components[
                    ~np.equal(scattering_index[:, idx + 2], 0), :
                ]
                # ray_components=np.zeros((ray_analysis_index.shape[0],3),dtype=np.complex64)
                # ray_components=EM.EMGPUWrapper(ray_analysis_index,point_information,wavelength)
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index

            else:
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), 0 : idx + 2
                ]
                temp_components = ray_components[
                    ~np.equal(scattering_index[:, idx + 1], 0), :
                ]
                # ray_components=EM.EMGPUWrapper(ray_analysis_index,point_information,wavelength)
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]
            #
            # scatter_map indexing
            # for source_index in range(sourcenum):
            #    for sink_index in range(sinknum):
            #        scatter_map[source_index,sink_index,idx]=np.sum(ray_components[(depth_slice[:,0]-1==0)&(depth_slice[:,1]-sourcenum-1==0)])

            scatter_map = scatter_net_sortEM(
                sourcenum, sinknum, scatter_map, depth_slice, temp_components, idx
            )

    return scatter_map


# @jit
def VectorNetworkProcessv2(
    sources, sinks, scattering_points, scattering_index, wavelength
):
    """
    A more efficienct version of BaseNetworkProcess, using the vectorised path length function to process the total_network array more efficienctly.
    #
    Parameters
    ----------
    total_network : array of n*(m*3) tuples, the coordinates of each interaction point
    wavelength : scaler
        wavelength of interest, currently a single value (SI units)
    scattering_index : n*2 array of integers, the source and sink index of each ray
    #
    Returns
    -------
    scatter_map : source_size*sink_size array
    #
    """
    if scattering_index.shape[0] == 0:
        sourcenum = np.int(np.nanmax(scattering_index[:, 0])) + 1
        sinknum = np.int(np.nanmax(scattering_index[:, 1])) + 1
        scatter_map = np.zeros((sourcenum, sinknum, 1), dtype="complex")
    else:
        unified_model = np.append(
            np.append(sources.astype(np.float32), sinks.astype(np.float32), axis=0),
            scattering_points.astype(np.float32),
            axis=0,
        )
        wave_vector = (2 * np.pi) / wavelength
        sourcenum = sources.shape[0]
        sinknum = sinks.shape[0]
        scatterdepth = scattering_index.shape[1] - 1
        scatter_map = np.zeros((sourcenum, sinknum, scatterdepth), dtype="complex")
        for idx in range(scatterdepth):
            if idx == (scatterdepth - 1):
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), :
                ]
                if idx == 0:
                    paths = PathLength(
                        np.append(
                            unified_model[ray_analysis_index[:, 0], :] - 1,
                            unified_model[ray_analysis_index[:, 1] - 1, :],
                            axis=1,
                        )
                    )
                elif idx == 1:
                    paths = PathLength(
                        np.append(
                            np.append(
                                unified_model[ray_analysis_index[:, 0] - 1, :],
                                unified_model[ray_analysis_index[:, 1] - 1, :],
                                axis=1,
                            ),
                            unified_model[ray_analysis_index[:, 2] - 1, :],
                            axis=1,
                        )
                    )
                elif idx == 2:
                    paths = PathLength(
                        np.append(
                            np.append(
                                np.append(
                                    unified_model[ray_analysis_index[:, 0] - 1, :],
                                    unified_model[ray_analysis_index[:, 1] - 1, :],
                                    axis=1,
                                ),
                                unified_model[ray_analysis_index[:, 2] - 1, :],
                                axis=1,
                            ),
                            unified_model[ray_analysis_index[:, 3] - 1, :],
                            axis=1,
                        )
                    )

                # path_loss=((wavelength/(4*np.pi*paths))**2)*np.exp(paths*wave_vector*1j)
                path_loss = wavelength / (4 * np.pi * paths)
                ray_components = path_loss * (np.exp(paths * wave_vector * 1j))
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]

            elif idx == 0:
                ray_analysis_index = scattering_index[
                    np.equal(scattering_index[:, idx + 2], 0), 0 : idx + 2
                ]
                paths = PathLength(
                    np.append(
                        unified_model[ray_analysis_index[:, 0] - 1, :],
                        unified_model[ray_analysis_index[:, 1] - 1, :],
                        axis=1,
                    )
                )
                # path_loss=((wavelength/(4*np.pi*paths))**2)*np.exp(paths*wave_vector*1j)
                path_loss = wavelength / (4 * np.pi * paths)
                ray_components = path_loss * (np.exp(paths * wave_vector * 1j))
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index

            else:
                ray_analysis_index = scattering_index[
                    ~np.equal(scattering_index[:, idx + 1], 0), :
                ]
                if idx == 0:
                    paths = PathLength(
                        np.append(
                            unified_model[ray_analysis_index[:, 0], :] - 1,
                            unified_model[ray_analysis_index[:, 1] - 1, :],
                            axis=1,
                        )
                    )
                elif idx == 1:
                    paths = PathLength(
                        np.append(
                            np.append(
                                unified_model[ray_analysis_index[:, 0] - 1, :],
                                unified_model[ray_analysis_index[:, 1] - 1, :],
                                axis=1,
                            ),
                            unified_model[ray_analysis_index[:, 2] - 1, :],
                            axis=1,
                        )
                    )
                elif idx == 2:
                    paths = PathLength(
                        np.append(
                            np.append(
                                np.append(
                                    unified_model[ray_analysis_index[:, 0] - 1, :],
                                    unified_model[ray_analysis_index[:, 1] - 1, :],
                                    axis=1,
                                ),
                                unified_model[ray_analysis_index[:, 2] - 1, :],
                                axis=1,
                            ),
                            unified_model[ray_analysis_index[:, 3] - 1, :],
                            axis=1,
                        )
                    )

                # path_loss=((wavelength/(4*np.pi*paths))**2)*np.exp(paths*wave_vector*1j)
                path_loss = wavelength / (4 * np.pi * paths)
                ray_components = path_loss * (np.exp(paths * wave_vector * 1j))
                # path_loss=((1j*wave_vector)/(4*np.pi))*(np.exp(paths*wave_vector*1j)/paths)
                depth_slice = ray_analysis_index[:, [0, -1]]
            #
            # scatter_map indexing
            # for source_index in range(sourcenum):
            #    for sink_index in range(sinknum):
            #        scatter_map[source_index,sink_index,idx]=np.sum(ray_components[(depth_slice[:,0]-1==0)&(depth_slice[:,1]-sourcenum-1==0)])

            scatter_map = scatter_net_sorttest(
                sourcenum, sinknum, scatter_map, depth_slice, ray_components, idx
            )

    return scatter_map


@njit(parallel=True)
def scatter_net_sort(sourcenum, sinknum, scatter_map, depth_slice, ray_components, idx):
    # simplified scattering sort to create a scattering network using Numbas prange and loop unrolling.
    for sink_index in range(sinknum):
        for source_index in range(sourcenum):
            scatter_map[source_index, sink_index, idx] = np.nansum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index)
                ]
            )

    return scatter_map


def scatter_net_sorttest(
    sourcenum, sinknum, scatter_map, depth_slice, ray_components, idx
):
    # simplified scattering sort to create a scattering network using Numbas prange and loop unrolling.
    for sink_index in range(sinknum):
        for source_index in range(sourcenum):
            scatter_map[source_index, sink_index, idx] = np.nansum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index)
                ]
            )

    return scatter_map


def scatter_net_sortEMtest(
    sourcenum, sinknum, scatter_map, depth_slice, ray_components, idx
):
    # simplified scattering sort to create a scattering network using Numbas prange and loop unrolling.
    for sink_index in range(sinknum):
        for source_index in range(sourcenum):
            scatter_map[source_index, sink_index, 0, idx] = np.nansum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    0,
                ]
            )
            scatter_map[source_index, sink_index, 1, idx] = np.nansum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    1,
                ]
            )
            scatter_map[source_index, sink_index, 2, idx] = np.nansum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    2,
                ]
            )

    return scatter_map


@njit(parallel=True)
def scatter_net_sortEM(
    sourcenum, sinknum, scatter_map, depth_slice, ray_components, idx
):
    # simplified scattering sort to create a scattering network using Numbas prange and loop unrolling.
    for sink_index in range(sinknum):
        for source_index in range(sourcenum):
            scatter_map[source_index, sink_index, 0, idx] = np.sum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    0,
                ]
            )
            scatter_map[source_index, sink_index, 1, idx] = np.sum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    1,
                ]
            )
            scatter_map[source_index, sink_index, 2, idx] = np.sum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    2,
                ]
            )

    return scatter_map


# @njit(parallel=True)
def scatter_net_sortTimeEM(
    sourcenum,
    sinknum,
    scatter_map,
    depth_slice,
    ray_components,
    time_components,
    time_index,
    idx,
):
    # simplified scattering sort to create a scattering network using Numbas prange and loop unrolling.
    for sink_index in range(sinknum):
        for source_index in range(sourcenum):
            time_address = np.digitize(
                time_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index)
                ],
                time_index,
            )
            scatter_map[source_index, sink_index, 0, time_address, idx] = np.sum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    0,
                ]
            )
            scatter_map[source_index, sink_index, 1, time_address, idx] = np.sum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    1,
                ]
            )
            scatter_map[source_index, sink_index, 2, time_address, idx] = np.sum(
                ray_components[
                    (depth_slice[:, 0] - 1 == source_index)
                    & (depth_slice[:, 1] - sourcenum - 1 == sink_index),
                    2,
                ]
            )

    return scatter_map


def BaseNetworkProcess(total_network, wavelength, scattering_index):
    # take the total_network matrix, and work out the path length of each connection, and hence superposition
    wave_vector = (2 * np.pi) / wavelength
    sourcenum = np.int(np.nanmax(scattering_index[:, 0])) + 1
    sinknum = np.int(np.nanmax(scattering_index[:, 1])) + 1
    scatterdepth = int((len(total_network[1, :]) - 3) / 3)
    scatter_map = np.zeros((sourcenum, sinknum, scatterdepth), dtype="complex")
    depth_num = 0
    for idx in range(scatterdepth):
        if idx == (scatterdepth - 1):
            if scatterdepth == 1:
                start_row = 0
                end_row = len(total_network[:, 1]) - 1
            else:
                start_row = (
                    np.max(np.where((np.isnan(total_network[:, 6 + ((idx - 1) * 3)]))))
                    + 1
                )
                end_row = len(total_network[:, 1]) - 1

        elif idx == 0:
            start_row = 0
            end_row = np.max(np.where((np.isnan(total_network[:, 6 + (idx * 3)]))))
        else:
            start_row = (
                np.max(np.where((np.isnan(total_network[:, 3 + (idx * 3)])))) + 1
            )
            end_row = np.max(np.where((np.isnan(total_network[:, 6 + ((idx) * 3)]))))

        for idr in range(start_row, end_row + 1):
            source_index = np.int(scattering_index[idr, 0])
            sink_index = np.int(scattering_index[idr, 1])
            path_distance = np.zeros((1))
            if idx == 0:
                path_distance = (
                    distance.euclidean(
                        total_network[idr, np.asarray([0, 1, 2]) + ((0) * 3)],
                        total_network[idr, np.asarray([3, 4, 5]) + ((0) * 3)],
                    )
                    + path_distance
                )
            else:
                for idy in range(idx + 1):
                    path_distance = (
                        distance.euclidean(
                            total_network[idr, np.asarray([0, 1, 2]) + ((idy) * 3)],
                            total_network[idr, np.asarray([3, 4, 5]) + ((idy) * 3)],
                        )
                        + path_distance
                    )

            path_loss = (4 * np.pi * path_distance) / wavelength
            scatter_map[source_index, sink_index, idx] = (1 / path_loss) * np.exp(
                -path_distance * wave_vector * 1j
            ) + scatter_map[source_index, sink_index, idx]

    return scatter_map


def charge_rays_environment1Dv2(sources, sinks, environment_points, point_indexing):
    """
    Generate Ray Payload from numpy arrays of sources, sinks, and environment points, in a 1 dimensional array to match the point_indexing demands

    Parameters
    ----------
    sources : TYPE
        DESCRIPTION.
    sinks : TYPE
        DESCRIPTION.
    environment_points : numpy array of xyz
        points for scattering in environment
        point_indexing : (n*m) by 2 array, 0 has the source index, and 1 has the sink index, otherwise nans'
    Returns
    -------
    temp_ray_payload : array of ray_t type
        ray payload to be sent to GPU

    """
    unified_model = np.append(
        np.append(sources, sinks, axis=0), environment_points, axis=0
    )
    temp_ray_payload = np.empty(point_indexing.shape[0], dtype=base_types.ray_t)
    local_sources = unified_model[point_indexing[:, 0] - 1, :]
    directions = np.zeros(
        (len(unified_model[point_indexing[:, 1] - 1, :]), 3), dtype=np.float32
    )
    norm_length = np.zeros(
        (len(unified_model[point_indexing[:, 1] - 1, :]), 1), dtype=np.float32
    )
    directions, norm_length = math_functions.calc_dv_norm(
        unified_model[point_indexing[:, -2] - 1, :],
        unified_model[point_indexing[:, -1] - 1, :],
        directions,
        norm_length,
    )
    temp_ray_payload[:]["ox"] = unified_model[point_indexing[:, 0] - 1, 0]
    temp_ray_payload[:]["oy"] = unified_model[point_indexing[:, 0] - 1, 1]
    temp_ray_payload[:]["oz"] = unified_model[point_indexing[:, 0] - 1, 2]
    temp_ray_payload[:]["dx"] = directions[:, 0]
    temp_ray_payload[:]["dy"] = directions[:, 1]
    temp_ray_payload[:]["dz"] = directions[:, 2]
    # temp_ray_payload[:]['tx']=unified_model[point_indexing[:,1]-1,0]
    # temp_ray_payload[:]['ty']=unified_model[point_indexing[:,1]-1,1]
    # temp_ray_payload[:]['tz']=unified_model[point_indexing[:,1]-1,2]
    temp_ray_payload[:]["dist"] = norm_length[:, 0]
    temp_ray_payload[:]["intersect"] = False

    return temp_ray_payload


def rayHits1Dv2(ray_payload, point_indexing, sink_index):
    """
    filtering process for each raycasting stage, seprating rays which hit sinks from non-terminating rays.
    Parameters
    ----------
    ray_payload : TYPE
        DESCRIPTION.
    point_indexing : TYPE
        DESCRIPTION.
    scatter_inc : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    filtered_network : TYPE
        DESCRIPTION.
    final_network : TYPE
        DESCRIPTION.
    filtered_index : TYPE
        DESCRIPTION.
    final_index : TYPE
        DESCRIPTION.
    """
    hitmap = ~ray_payload[:]["intersect"]
    final_index = point_indexing[
        np.all(np.array([hitmap, np.isin(point_indexing[:, -1], sink_index)]), axis=0),
        :,
    ]
    filtered_index = point_indexing[
        np.all(np.array([hitmap, ~np.isin(point_indexing[:, -1], sink_index)]), axis=0),
        :,
    ]

    return filtered_index, final_index


def launchRaycaster1Dv2(
    sources, sinks, scattering_points, io_indexing, environment_local
):
    # cuda.select_device(0)
    start = timer()
    first_ray_payload = charge_rays_environment1Dv2(
        sources, sinks, scattering_points, io_indexing
    )
    prep_dt = timer() - start
    raystart = timer()
    ray_num = len(first_ray_payload)
    tri_num = len(environment_local)
    max_tris = 2 ** 20
    triangle_chunk = np.linspace(
        0, tri_num, math.ceil(tri_num / max_tris) + 1, dtype=np.int32
    )
    chunk_size = 2 ** 10
    threads_in_block = 1024
    ray_chunks = np.linspace(
        0, ray_num, math.ceil(ray_num / chunk_size) + 1, dtype=np.int32
    )
    # for idx in range(len(triangle_chunk)-1):
    d_environment = cuda.device_array(len(environment_local), dtype=base_types.triangle_t)
    cuda.to_device(environment_local, to=d_environment)
    for n in range(len(ray_chunks) - 1):
        chunk_payload = first_ray_payload[ray_chunks[n] : ray_chunks[n + 1]]
        chunk_ray_size = len(chunk_payload)
        d_chunk_payload = cuda.device_array([chunk_ray_size], dtype=base_types.ray_t)
        cuda.to_device(chunk_payload, to=d_chunk_payload)
        #
        # ray_temp=np.empty((chunk_ray_size),dtype=np.bool)
        # ray_temp[:]=first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']
        # distance_temp=np.empty((chunk_ray_size),dtype=np.float32)
        # distance_temp[:]=math.inf
        # dist_list=cuda.to_device(distance_temp)
        # d_ray_flag=cuda.to_device(ray_temp)
        # Here, we choose the granularity of the threading on our device. We want
        # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
        # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
        grids = math.ceil(chunk_ray_size / threads_in_block)
        threads = min(ray_chunks[1] - ray_chunks[0], threads_in_block)
        # Execute the kernel
        # cuda.profile_start()
        # kernel1Dv2[grids, threads](d_chunk_payload,d_environment,d_ray_flag)
        kernel1Dv2[grids, threads](d_chunk_payload, d_environment)
        # cuda.profile_stop()
        # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
        first_ray_payload[
            ray_chunks[n] : ray_chunks[n + 1]
        ] = d_chunk_payload.copy_to_host()
        # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']=d_chunk_payload['intersect'].copy_to_host()
        # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['dist']=d_chunk_payload['dist'].copy_to_host()

    # cuda.close()
    kernel_dt = timer() - raystart
    start = timer()
    RAYS_CAST = ray_num
    sink_index = np.arange(
        sources.shape[0] + 1, sources.shape[0] + 1 + sinks.shape[0]
    ).reshape(sinks.shape[0], 1)
    filtered_index, final_index = rayHits1Dv2(
        first_ray_payload, io_indexing, sink_index
    )
    mem_dt = timer() - start
    # print("First Stage: Prep {:3.1f} s, Raycasting  {:3.1f} s, Path Processing {:3.1f} s".format(prep_dt,kernel_dt,mem_dt) )
    return filtered_index, final_index, RAYS_CAST


def launchRaycaster1Dv3(
    sources, sinks, scattering_points, io_indexing, environment_local
):
    # cuda.select_device(0)
    start = timer()
    first_ray_payload = charge_rays_environment1Dv2(
        sources, sinks, scattering_points, io_indexing
    )
    prep_dt = timer() - start
    raystart = timer()
    ray_num = len(first_ray_payload)
    tri_num = len(environment_local)
    # print('Launch Raycaster Triangles ', len(environment_local))
    max_tris = 2 ** 20
    triangle_chunk = np.linspace(
        0, tri_num, math.ceil(tri_num / max_tris) + 1, dtype=np.int32
    )
    chunk_size = 2 ** 18
    threads_in_block = 256
    # ray_chunks=np.linspace(0,ray_num,math.ceil(ray_num/chunk_size)+1,dtype=np.int32)
    # for idx in range(len(triangle_chunk)-1):
    d_environment = cuda.device_array(len(environment_local), dtype=base_types.triangle_t)
    cuda.to_device(environment_local, to=d_environment)

    d_chunk_payload = cuda.device_array([ray_num], dtype=base_types.ray_t)
    cuda.to_device(first_ray_payload, to=d_chunk_payload)
    #
    # ray_temp=np.empty((chunk_ray_size),dtype=np.bool)
    # ray_temp[:]=first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']
    # distance_temp=np.empty((chunk_ray_size),dtype=np.float32)
    # distance_temp[:]=math.inf
    # dist_list=cuda.to_device(distance_temp)
    # d_ray_flag=cuda.to_device(ray_temp)
    # Here, we choose the granularity of the threading on our device. We want
    # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
    # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
    grids = math.ceil(ray_num / threads_in_block)
    threads = threads_in_block
    # Execute the kernel
    # cuda.profile_start()
    # kernel1Dv2[grids, threads](d_chunk_payload,d_environment,d_ray_flag)
    kernel1Dv3[grids, threads](d_chunk_payload, d_environment)
    # cuda.profile_stop()
    # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
    first_ray_payload = d_chunk_payload.copy_to_host()
    # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']=d_chunk_payload['intersect'].copy_to_host()
    # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['dist']=d_chunk_payload['dist'].copy_to_host()

    # cuda.close()
    kernel_dt = timer() - raystart
    start = timer()
    RAYS_CAST = ray_num
    sink_index = np.arange(
        sources.shape[0] + 1, sources.shape[0] + 1 + sinks.shape[0]
    ).reshape(sinks.shape[0], 1)
    filtered_index, final_index = rayHits1Dv2(
        first_ray_payload, io_indexing, sink_index
    )
    mem_dt = timer() - start
    # deallocate memory on gpu
    #ctx = cuda.current_context()
    #deallocs = ctx.deallocations
    #deallocs.clear()
    # print("First Stage: Prep {:3.1f} s, Raycasting  {:3.1f} s, Path Processing {:3.1f} s".format(prep_dt,kernel_dt,mem_dt) )
    return filtered_index, final_index, RAYS_CAST


def chunkingRaycaster1Dv2(
    sources, sinks, scattering_points, filtered_index, environment_local, terminate_flag
):
    # cuda.select_device(0)
    start = timer()
    sink_index = np.arange(
        sources.shape[0] + 1, sources.shape[0] + 1 + sinks.shape[0]
    ).reshape(sinks.shape[0], 1)
    scattering_point_index = np.arange(
        np.max(sink_index) + 1, np.max(sink_index) + 1 + scattering_points.shape[0]
    ).reshape(scattering_points.shape[0], 1)
    if not terminate_flag:
        target_indexing = create_model_index(
            filtered_index, sink_index, scattering_point_index
        )
    else:
        target_indexing = create_model_index(
            filtered_index, sink_index, np.empty((0, 0), dtype=np.int32)
        )  # only target rays at sinks

    second_ray_payload = charge_rays_environment1Dv2(
        sources, sinks, scattering_points, target_indexing
    )
    prep_dt = timer() - start
    raystart = timer()
    ray_num = len(second_ray_payload)
    tri_num = len(environment_local)
    max_tris = 2 ** 18
    triangle_chunk = np.linspace(
        0, tri_num, math.ceil(tri_num / max_tris) + 1, dtype=np.int32
    )
    # Create a container for the pixel RGBA information of our image
    chunk_size = 2 ** 10
    threads_in_block = 1024
    ray_chunks = np.linspace(
        0, ray_num, math.ceil(ray_num / chunk_size) + 1, dtype=np.int32
    )
    # for idx in range(len(triangle_chunk)-1):
    d_environment = cuda.device_array(len(environment_local), dtype=base_types.triangle_t)
    cuda.to_device(environment_local, to=d_environment)
    for n in range(len(ray_chunks) - 1):
        chunk_payload = second_ray_payload[ray_chunks[n] : ray_chunks[n + 1]]
        chunk_ray_size = len(chunk_payload)
        # distance_temp=np.empty((chunk_ray_size),dtype=np.float32)
        # distance_temp[:]=math.inf
        # ray_temp=np.empty((chunk_ray_size),dtype=np.bool)
        # ray_temp[:]=False
        # dist_list=cuda.to_device(distance_temp)
        # d_ray_flag=cuda.to_device(ray_temp)
        # d_chunk_payload = cuda.device_array([chunk_ray_size], dtype=ray_t)
        d_chunk_payload = cuda.device_array([chunk_ray_size], dtype=base_types.ray_t)
        cuda.to_device(chunk_payload, to=d_chunk_payload)
        # Here, we choose the granularity of the threading on our device. We want
        # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
        # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
        grids = math.ceil(chunk_ray_size / threads_in_block)
        threads = min(ray_chunks[1] - ray_chunks[0], threads_in_block)
        # Execute the kernel
        # cuda.profile_start()
        kernel1Dv2[grids, threads](d_chunk_payload, d_environment)
        # cuda.profile_stop()
        # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
        # second_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
        # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
        second_ray_payload[
            ray_chunks[n] : ray_chunks[n + 1]
        ] = d_chunk_payload.copy_to_host()

    # cuda.close()
    kernel_dt = timer() - raystart
    start = timer()
    RAYS_CAST = ray_num
    filtered_index2, final_index2 = rayHits1Dv2(
        second_ray_payload, target_indexing, sink_index
    )
    mem_dt = timer() - start
    # print("Second Stage: Prep {:3.1f} s, Raycasting  {:3.1f} s, Path Processing {:3.1f} s".format(prep_dt,kernel_dt,mem_dt) )
    return filtered_index2, final_index2, RAYS_CAST


def chunkingRaycaster1Dv3(
    sources, sinks, scattering_points, filtered_index, environment_local, terminate_flag
):
    start = timer()
    sink_index = np.arange(
        sources.shape[0] + 1, sources.shape[0] + 1 + sinks.shape[0]
    ).reshape(sinks.shape[0], 1)
    scattering_point_index = np.arange(
        np.max(sink_index) + 1, np.max(sink_index) + 1 + scattering_points.shape[0]
    ).reshape(scattering_points.shape[0], 1)
    if not terminate_flag:
        target_indexing = create_model_index(
            filtered_index, sink_index, scattering_point_index
        )
    else:
        target_indexing = create_model_index(
            filtered_index, sink_index, np.empty((0, 0), dtype=np.int32)
        )  # only target rays at sinks

    prep_dt = timer() - start
    raystart = timer()
    RAYS_CAST = 0
    # the rays must fit in GPU memory, aim for no more than 80% utilisation
    # establish memory limits
    free_mem, total_mem = cuda.current_context().get_memory_info()
    max_mem = np.ceil(free_mem * 0.5).astype(np.int64)
    ray_limit = (
        np.floor(np.floor((max_mem - environment_local.nbytes) / base_types.ray_t.size) / 1e7)
        * 1e7
    ).astype(np.int64)
    if target_indexing.shape[0] >= ray_limit:
        # need to split the array and process seperatly

        sub_target = np.array_split(
            target_indexing, np.ceil(target_indexing.shape[0] / ray_limit).astype(int)
        )
        chunknum = len(sub_target)
        filtered_index2 = np.empty((0, target_indexing.shape[1]), dtype=np.int32)
        final_index2 = np.empty((0, target_indexing.shape[1]), dtype=np.int32)
        # print('chunking total of ',target_indexing.shape[0],' rays in ',chunknum,' batches')
        for chunkindex in range(chunknum):
            # cycle the raycaster over the sub arrays
            threads_in_block = 256
            second_ray_payload = charge_rays_environment1Dv2(
                sources, sinks, scattering_points, sub_target[chunkindex]
            )
            # ray_chunks=np.linspace(0,ray_num,math.ceil(ray_num/chunk_size)+1,dtype=np.int32)
            # for idx in range(len(triangle_chunk)-1):
            d_environment = cuda.device_array(
                len(environment_local), dtype=base_types.triangle_t
            )
            cuda.to_device(environment_local, to=d_environment)
            d_chunk_payload = cuda.device_array(
                [second_ray_payload.shape[0]], dtype=base_types.ray_t
            )
            cuda.to_device(second_ray_payload, to=d_chunk_payload)
            #
            # ray_temp=np.empty((chunk_ray_size),dtype=np.bool)
            # ray_temp[:]=first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']
            # distance_temp=np.empty((chunk_ray_size),dtype=np.float32)
            # distance_temp[:]=math.inf
            # dist_list=cuda.to_device(distance_temp)
            # d_ray_flag=cuda.to_device(ray_temp)
            # Here, we choose the granularity of the threading on our device. We want
            # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
            # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
            grids = math.ceil(second_ray_payload.shape[0] / threads_in_block)
            threads = threads_in_block
            # Execute the kernel
            # cuda.profile_start()
            # kernel1Dv2[grids, threads](d_chunk_payload,d_environment,d_ray_flag)
            kernel1Dv3[grids, threads](d_chunk_payload, d_environment)
            # cuda.profile_stop()
            # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
            second_ray_payload = d_chunk_payload.copy_to_host()
            # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']=d_chunk_payload['intersect'].copy_to_host()
            # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['dist']=d_chunk_payload['dist'].copy_to_host()

            # cuda.close()
            kernel_dt = timer() - raystart
            start = timer()
            RAYS_CAST += second_ray_payload.shape[0]
            temp_filtered_index2, temp_final_index2 = rayHits1Dv2(
                second_ray_payload, sub_target[chunkindex], sink_index
            )
            mem_dt = timer() - start
            filtered_index2 = np.append(filtered_index2, temp_filtered_index2, axis=0)
            final_index2 = np.append(final_index2, temp_final_index2, axis=0)
            # deallocate memory on gpu
            #ctx = cuda.current_context()
            #ctx.reset()
    else:
        second_ray_payload = charge_rays_environment1Dv2(
            sources, sinks, scattering_points, target_indexing
        )
        # print('Chunking Raycaster Triangles ', len(environment_local))
        # max_tris=2**18
        # triangle_chunk=np.linspace(0,tri_num,math.ceil(tri_num/max_tris)+1,dtype=np.int32)

        # chunk_size=2**11
        threads_in_block = 256
        # ray_chunks=np.linspace(0,ray_num,math.ceil(ray_num/chunk_size)+1,dtype=np.int32)
        # for idx in range(len(triangle_chunk)-1):
        d_environment = cuda.device_array(len(environment_local), dtype=base_types.triangle_t)
        cuda.to_device(environment_local, to=d_environment)
        d_chunk_payload = cuda.device_array(
            [second_ray_payload.shape[0]], dtype=base_types.ray_t
        )
        cuda.to_device(second_ray_payload, to=d_chunk_payload)
        #
        # ray_temp=np.empty((chunk_ray_size),dtype=np.bool)
        # ray_temp[:]=first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']
        # distance_temp=np.empty((chunk_ray_size),dtype=np.float32)
        # distance_temp[:]=math.inf
        # dist_list=cuda.to_device(distance_temp)
        # d_ray_flag=cuda.to_device(ray_temp)
        # Here, we choose the granularity of the threading on our device. We want
        # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
        # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
        grids = math.ceil(second_ray_payload.shape[0] / threads_in_block)
        threads = threads_in_block
        # Execute the kernel
        # cuda.profile_start()
        # kernel1Dv2[grids, threads](d_chunk_payload,d_environment,d_ray_flag)
        kernel1Dv3[grids, threads](d_chunk_payload, d_environment)
        # cuda.profile_stop()
        # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
        second_ray_payload = d_chunk_payload.copy_to_host()
        # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['intersect']=d_chunk_payload['intersect'].copy_to_host()
        # first_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['dist']=d_chunk_payload['dist'].copy_to_host()

        # cuda.close()
        kernel_dt = timer() - raystart
        start = timer()
        RAYS_CAST += second_ray_payload.shape[0]
        filtered_index2, final_index2 = rayHits1Dv2(
            second_ray_payload, target_indexing, sink_index
        )
        mem_dt = timer() - start
        # deallocate memory on gpu
        #ctx = cuda.current_context()
        #deallocs = ctx.deallocations
        #deallocs.clear()
        # print("Second Stage: Prep {:3.1f} s, Raycasting  {:3.1f} s, Path Processing {:3.1f} s".format(prep_dt,kernel_dt,mem_dt) )
    return filtered_index2, final_index2, RAYS_CAST


def charge_shadow_rays1D(sources, sinks, point_indexing, target_indexing):
    """
    Generate Ray Payload from numpy arrays of sources, sinks, and environment points, in a 1 dimensional array

    Parameters
    ----------
    sources : TYPE
        DESCRIPTION.
    sinks : TYPE
        DESCRIPTION.
    environment_points : numpy array of xyz
        points for scattering in environment
        point_indexing : (n*m) by 2 array, 0 has the source index, and 1 has the sink index, otherwise nans'
    Returns
    -------
    temp_ray_payload : array of ray_t type
        ray payload to be sent to GPU

    """
    idx = 0
    targets = sinks
    target_num = len(targets)
    source_num, origin_width = np.shape(sources)
    origins = np.zeros((source_num * target_num, origin_width), dtype=np.float32)
    source_sink_index = np.full((source_num * target_num, 2), np.nan, dtype=np.int32)
    temp_ray_payload = np.empty([source_num * target_num], dtype=base_types.ray_t)
    temp_ray_payload, origins, source_sink_index = ray_charge_core_final_vector(
        temp_ray_payload,
        origins,
        source_sink_index,
        sources,
        targets,
        point_indexing,
        target_indexing,
    )

    return temp_ray_payload, origins, source_sink_index


def ray_charge_core_final_vector(
    temp_ray_payload,
    origins,
    source_sink_index,
    sources,
    targets,
    point_indexing,
    target_indexing,
):
    # using array allocation
    # idx=0
    target_num = len(targets)
    source_num, origin_width = np.shape(sources)
    directions = np.zeros((len(targets), 3), dtype=np.float32)
    local_sources = np.zeros((len(targets), origin_width), dtype=np.float32)
    norm_length = np.zeros((len(targets), 1), dtype=np.float32)
    source_chunking = np.linspace(
        0,
        len(temp_ray_payload),
        math.ceil(len(temp_ray_payload) / target_num) + 1,
        dtype=np.int32,
    )
    for n in range(len(source_chunking) - 1):
        local_sources[:, -origin_width:] = sources[n, :]
        directions, norm_length = math_functions.calc_dv_norm(
            local_sources[:, -3:], targets, directions, norm_length
        )
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "ox"
        ] = local_sources[:, -3]
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "oy"
        ] = local_sources[:, -2]
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "oz"
        ] = local_sources[:, -1]
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "dx"
        ] = directions[:, 0]
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "dy"
        ] = directions[:, 1]
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "dz"
        ] = directions[:, 2]
        # temp_ray_payload[source_chunking[n]:source_chunking[n+1]]['tx']=targets[:,0]
        # temp_ray_payload[source_chunking[n]:source_chunking[n+1]]['ty']=targets[:,1]
        # temp_ray_payload[source_chunking[n]:source_chunking[n+1]]['tz']=targets[:,2]
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "dist"
        ] = norm_length[:, 0]
        temp_ray_payload[source_chunking[n] : source_chunking[n + 1]][
            "intersect"
        ] = False
        origins[source_chunking[n] : source_chunking[n + 1], -origin_width:] = sources[
            n, :
        ]
        source_sink_index[
            source_chunking[n] : source_chunking[n + 1], 0
        ] = point_indexing[n, 0]
        source_sink_index[
            source_chunking[n] : source_chunking[n + 1], 1
        ] = target_indexing.ravel()

    return temp_ray_payload, origins, source_sink_index


def shadowRaycaster(filtered_network, sinks, filtered_index, environment_local):
    start = timer()
    target_indexing = np.full((len(sinks), 1), np.nan, dtype=np.float32).ravel()
    target_indexing[:] = range(len(sinks))
    second_ray_payload, second_origin, io_index2 = charge_shadow_rays1D(
        filtered_network, sinks, filtered_index, target_indexing
    )
    prep_dt = timer() - start
    raystart = timer()
    ray_num = len(second_ray_payload)
    tri_num = len(environment_local)
    max_tris = 2 ** 16
    triangle_chunk = np.linspace(
        0, tri_num, math.ceil(tri_num / max_tris) + 1, dtype=np.int32
    )
    # Create a container for the pixel RGBA information of our image
    chunk_size = 2 ** 21
    threads_in_block = 1024
    ray_chunks = np.linspace(
        0, ray_num, math.ceil(ray_num / chunk_size) + 1, dtype=np.int32
    )
    # sync_stream=cuda.stream()
    for idx in range(len(triangle_chunk) - 1):
        d_environment = cuda.device_array(
            len(environment_local[triangle_chunk[idx] : triangle_chunk[idx + 1]]),
            dtype=base_types.triangle_t,
        )
        cuda.to_device(
            environment_local[triangle_chunk[idx] : triangle_chunk[idx + 1]],
            to=d_environment,
        )
        for n in range(len(ray_chunks) - 1):
            chunk_payload = second_ray_payload[ray_chunks[n] : ray_chunks[n + 1]]
            chunk_ray_size = len(chunk_payload)
            distance_temp = np.empty((chunk_ray_size), dtype=np.float32)
            distance_temp[:] = math.inf
            ray_temp = np.empty((chunk_ray_size), dtype=np.bool)
            ray_temp[:] = False
            dist_list = cuda.to_device(distance_temp)
            d_ray_flag = cuda.to_device(ray_temp)
            # d_chunk_payload = cuda.device_array([chunk_ray_size], dtype=ray_t)
            d_chunk_payload = cuda.device_array([chunk_ray_size], dtype=base_types.ray_t)
            cuda.to_device(chunk_payload, to=d_chunk_payload)
            # Here, we choose the granularity of the threading on our device. We want
            # to try to cover the entire workload of rays and targets with simulatenous threads, so we'll
            # choose a grid of (source_num/16. target_num/16) blocks, each with (16, 16) threads
            grids = math.ceil(chunk_ray_size / threads_in_block)
            threads = min(ray_chunks[1] - ray_chunks[0], threads_in_block)
            # Execute the kernel
            # cuda.profile_start()
            kernel1D[grids, threads](
                d_chunk_payload, d_environment, d_ray_flag, dist_list
            )
            # second_ray_payload[ray_chunks[n]:ray_chunks[n+1]]=d_chunk_payload.copy_to_host()
            second_ray_payload[ray_chunks[n] : ray_chunks[n + 1]][
                "intersect"
            ] = d_ray_flag.copy_to_host()
            # second_ray_payload[ray_chunks[n]:ray_chunks[n+1]]['dist']=dist_list.copy_to_host()
            # cuda.profile_stop()
            # distmap[source_chunks[n]:source_chunks[n+1],target_chunks[m]:target_chunks[m+1]] = d_distmap_chunked.copy_to_host()
            # sync_stream.synchronize()

    kernel_dt = timer() - raystart
    menstart = timer()
    RAYS_CAST = ray_num
    filtered_network2, final_network2, filtered_index2, final_index2 = rayHits1D(
        second_ray_payload, io_index2, scatter_inc=1, origins_local=second_origin
    )
    mem_dt = timer() - menstart
    print(
        "Stage: Prep {:3.1f} s, Raycasting  {:3.1f} s, Path Processing {:3.1f} s".format(
            prep_dt, kernel_dt, mem_dt
        )
    )
    return filtered_network2, final_network2, filtered_index2, final_index2, RAYS_CAST


def rayHits1D(
    ray_payload, point_indexing, scatter_inc=1, origins_local=None, initial_network=None
):
    """
    filtering process for each raycasting stage, seprating rays which hit sinks from non-terminating rays.
    ----------
    ray_payload : TYPE
        DESCRIPTION.
    point_indexing : TYPE
        DESCRIPTION.
    scatter_inc : TYPE, optional
        DESCRIPTION. The default is 1.
    origins_local : TYPE, optional
        DESCRIPTION. The default is None.
    initial_network : TYPE, optional
        DESCRIPTION. The default is None.
    Returns
    -------
    filtered_network : TYPE
        DESCRIPTION.
    final_network : TYPE
        DESCRIPTION.
    filtered_index : TYPE
        DESCRIPTION.
    final_index : TYPE
        DESCRIPTION.
    """
    hitmap = ~ray_payload[:]["intersect"]
    filter_vector = np.ravel(
        np.nonzero(np.all(np.asarray([hitmap, np.isnan(point_indexing[:, 1])]), axis=0))
    )
    output_vector = np.ravel(
        np.nonzero(
            np.all(np.asarray([hitmap, ~np.isnan(point_indexing[:, 1])]), axis=0)
        )
    )
    #
    final_network = np.zeros(
        (np.max(np.shape(output_vector)), 3 + (scatter_inc * 3)), dtype=np.float32
    )
    final_network[:, 0:-3] = origins_local[output_vector, :]
    final_network[:, -3] = ray_payload[output_vector]["tx"]
    final_network[:, -2] = ray_payload[output_vector]["ty"]
    final_network[:, -1] = ray_payload[output_vector]["tz"]
    #
    filtered_network = np.zeros(
        (np.max(np.shape(filter_vector)), 3 + (scatter_inc * 3)), dtype=np.float32
    )
    filtered_network[:, 0:-3] = origins_local[filter_vector, :]
    filtered_network[:, -3] = ray_payload[filter_vector]["tx"]
    filtered_network[:, -2] = ray_payload[filter_vector]["ty"]
    filtered_network[:, -1] = ray_payload[filter_vector]["tz"]
    #
    # final_network=hit_network[output_vector,:]
    # filtered_network=np.delete(hit_network,output_vector,axis=0)
    final_index = point_indexing[output_vector, :]
    filtered_index = point_indexing[filter_vector, :]
    return filtered_network, final_network, filtered_index, final_index


def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(2.0, 0.0)
    return False


# @njit
def create_model_index(source_index, sink_index, scattering_point_index):
    """
    Create an index for the model environment, first sources, then sinks, the scattering points in an uninteruppted chain
    Thus

    Parameters
    ----------
    sources : TYPE
        DESCRIPTION.
    sinks : TYPE
        DESCRIPTION.
    scattering_points : TYPE
        DESCRIPTION.

    Returns
    -------
    io_indexing : TYPE
        DESCRIPTION.

    """
    io_indexing = np.zeros(
        (
            source_index.shape[0]
            * (sink_index.shape[0] + scattering_point_index.shape[0]),
            source_index.shape[1] + 1,
        ),
        dtype=np.int32,
    )
    idx = 0
    for n in np.arange(0, source_index.shape[0]):
        if scattering_point_index.shape[0] == 0:
            io_indexing[
                np.arange(
                    idx, idx + (sink_index.shape[0] + scattering_point_index.shape[0])
                ),
                -1,
            ] = sink_index.ravel()
        else:
            io_indexing[
                np.arange(
                    idx, idx + (sink_index.shape[0] + scattering_point_index.shape[0])
                ),
                -1,
            ] = np.append(sink_index, scattering_point_index, axis=0).ravel()

        io_indexing[
            np.arange(
                idx, idx + (sink_index.shape[0] + scattering_point_index.shape[0])
            ),
            0 : source_index.shape[1],
        ] = source_index[n, :]
        idx += sink_index.shape[0] + scattering_point_index.shape[0]

    # then boolean check and filter to ensure there are no loops, aiming next ray at originating point
    return io_indexing[np.not_equal(io_indexing[:, -2], io_indexing[:, -1]), :]


def workchunkingv2(
    sources, sinks, scattering_points, environment, max_scatter, line_of_sight=True
):
    """
    Raycasting index creation and assignment to raycaster, upper bound is around 4.7e8 rays at a time, there is already chunking to prevent overflow of the GPU memory and timeouts

    Parameters
    ----------
    sources : n*3 numpy array of float
    sinks : m*3 numpy array of float
    scattering_points : o*3 numpy array of float
    environment : numpy array of triangle_t
    max_scatter : int
    line_of_sight : boolean

    Returns
    ---------
    full_index : 2D numpy array of ints
        the index for all successful rays cast from source coordinates, to any scattering points, to the sink point for each entry
    RAYS_CAST : int
        the number of rays cast in this launch.
    """
    # temp function to chunk the number of rays to prevent creation of ray arrays to large for memory
    # print('WorkChunking Triangles ',len(environment))
    RAYS_CAST=0
    raycasting_timestamp = timer()
    ray_estimate = (
        sources.shape[0] * (sinks.shape[0] + scattering_points.shape[0])
    ) + (
        sources.shape[0] * (scattering_points.shape[0] * sinks.shape[0] * (max_scatter))
    )
    # print("Total of {:3.1f} rays required".format(ray_estimate))
    # establish memory limits
    free_mem, total_mem = cuda.current_context().get_memory_info()
    max_mem = np.ceil(free_mem * 0.8).astype(np.int64)
    ray_limit = (
        np.floor(np.floor((max_mem - environment.nbytes) / base_types.ray_t.size) / 1e7) * 1e7
    ).astype(np.int64)
    # establish index boundaries
    source_index = np.arange(1, sources.shape[0] + 1).reshape(
        sources.shape[0], 1
    )  # index starts at 1, leaving 0 as a null
    sink_index = np.arange(
        sources.shape[0] + 1, sources.shape[0] + 1 + sinks.shape[0]
    ).reshape(sinks.shape[0], 1)
    scattering_point_index = np.arange(
        np.max(sink_index) + 1, np.max(sink_index) + 1 + scattering_points.shape[0]
    ).reshape(scattering_points.shape[0], 1)
    # implement chunking
    if max_scatter == 1:
        full_index = np.empty((0, 2), dtype=np.int32)
        io_indexing = create_model_index(
            source_index, sink_index, scattering_point_index
        )
        if io_indexing.shape[0] >= ray_limit:
            # need to split the array and process seperatly
            sub_io = np.array_split(
                io_indexing, np.ceil(io_indexing.shape[0] / ray_limit).astype(int)
            )
            chunknum = len(sub_io)

            for chunkindex in range(chunknum):
                _, temp_index, first_wave_Rays = launchRaycaster1Dv3(
                    sources,
                    sinks,
                    scattering_points,
                    sub_io[chunkindex],
                    copy.deepcopy(environment),
                )
                full_index = np.append(full_index, temp_index, axis=0)
                RAYS_CAST += first_wave_Rays
        else:
            _, temp_index, first_wave_Rays = launchRaycaster1Dv3(
                sources,
                sinks,
                scattering_points,
                io_indexing,
                copy.deepcopy(environment),
            )
            full_index = np.append(full_index, temp_index, axis=0)
            RAYS_CAST = first_wave_Rays

    elif max_scatter == 2:
        full_index = np.empty((0, 3), dtype=np.int32)
        if line_of_sight:
            io_indexing = create_model_index(
                source_index, sink_index, scattering_point_index
            )
        else:
            # drop line of sight rays from queue
            io_indexing = create_model_index(
                source_index, np.empty((0, 1)), scattering_point_index
            )

        if io_indexing.shape[0] >= ray_limit:
            # need to split the array and process separately
            sub_io = np.array_split(
                io_indexing, np.ceil(io_indexing.shape[0] / ray_limit).astype(int)
            )
            chunknum = len(sub_io)
            final_index = np.empty((0, 3), dtype=np.int32)
            RAYS_CAST = 0
            for chunkindex in range(chunknum):
                (
                    temp_filtered_index,
                    temp_final_index,
                    first_wave_Rays,
                ) = launchRaycaster1Dv3(
                    sources,
                    sinks,
                    scattering_points,
                    sub_io[chunkindex],
                    copy.deepcopy(environment),
                )
                # filtered_index = np.append(filtered_index, temp_filtered_index, axis=0)
                # final_index = np.append(final_index, temp_final_index, axis=0)
                if temp_filtered_index.shape[0] == 0:
                    temp_filtered_index2 = np.empty((0, 3), dtype=np.int32)
                    temp_final_index2 = np.empty((0, 3), dtype=np.int32)
                    second_wave_rays = 0
                else:
                    (
                        temp_filtered_index2,
                        temp_final_index2,
                        second_wave_rays,
                    ) = chunkingRaycaster1Dv3(
                        sources,
                        sinks,
                        scattering_points,
                        temp_filtered_index,
                        environment,
                        terminate_flag=True,
                    )
                RAYS_CAST += first_wave_Rays + second_wave_rays
                # create combined path network
                host_size = np.shape(temp_final_index)
                host_padding = np.zeros((host_size[0], 1), dtype=np.int32)
                temp_full_index = np.append(
                    np.append(temp_final_index, host_padding, axis=1),
                    temp_final_index2,
                    axis=0,
                )
                full_index = np.append(full_index, temp_full_index, axis=0)
        else:
            filtered_index, final_index, first_wave_Rays = launchRaycaster1Dv3(
                sources,
                sinks,
                scattering_points,
                io_indexing,
                copy.deepcopy(environment),
            )

            if filtered_index.shape[0] == 0:
                filtered_index2 = np.empty((0, 3), dtype=np.int32)
                final_index2 = np.empty((0, 3), dtype=np.int32)
                second_wave_rays = 0
            else:
                filtered_index2, final_index2, second_wave_rays = chunkingRaycaster1Dv3(
                    sources,
                    sinks,
                    scattering_points,
                    filtered_index,
                    environment,
                    terminate_flag=True,
                )
                # cuda.profile_stop()
                # print('WorkChunkingMaxScatter2 Triangles ', len(environment), 'Filtered Rays Stage 2', len(filtered_index2), 'Final Rays Stage 2',len(final_index2))
                start = timer()
                # filtered_network2,final_network2,filtered_index2,final_index2=rayHits(second_ray_payload,distmap2,io_index2,scatter_inc=2,origins_local=second_origin)
                RAYS_CAST = first_wave_Rays + second_wave_rays
                # create combined path network
                host_size = np.shape(final_index)
                host_padding = np.zeros((host_size[0], 1), dtype=np.int32)
                temp_full_index = np.append(
                    np.append(final_index, host_padding, axis=1), final_index2, axis=0
                )
                full_index = np.append(full_index, temp_full_index, axis=0)

    elif max_scatter == 3:
        full_index = np.empty((0, 4), dtype=np.int32)
        chunknum = np.minimum(sources.shape[0], np.int(np.ceil(ray_estimate / 2e8)))
        chunknum = np.maximum(2, chunknum)
        source_chunking = np.linspace(0, sources.shape[0], chunknum, dtype=np.int32)
        for chunkindex in range(source_chunking.size - 1):
            if line_of_sight:
                io_indexing = create_model_index(
                    source_index[
                        source_chunking[chunkindex] : source_chunking[chunkindex + 1]
                    ],
                    sink_index,
                    scattering_point_index,
                )
                filtered_index, final_index, first_wave_Rays = launchRaycaster1Dv3(
                    sources, sinks, scattering_points, io_indexing, environment
                )
            else:
                # drop line of sight rays from queue
                io_indexing = create_model_index(
                    source_index[
                        source_chunking[chunkindex] : source_chunking[chunkindex + 1]
                    ],
                    np.empty((0, 1)),
                    scattering_point_index,
                )
                filtered_index, final_index, first_wave_Rays = launchRaycaster1Dv3(
                    sources, sinks, scattering_points, io_indexing, environment
                )
            filtered_index2, final_index2, second_wave_rays = chunkingRaycaster1Dv3(
                sources, sinks, scattering_points, filtered_index, environment, False
            )
            if filtered_index2.shape[0] * sinks.shape[0] > 2e8:
                temp_chunks = np.int(
                    np.ceil((filtered_index2.shape[0] * sinks.shape[0]) / 2e8)
                )
                temp_chunking = np.linspace(
                    0, filtered_index2.shape[0], temp_chunks, dtype=np.int32
                )
                temp_index = np.empty((0, 4), dtype=np.int32)
                for temp_chunkindex in range(temp_chunking.shape[0] - 1):
                    (
                        filtered_index3,
                        final_index3_part,
                        third_wave_rays,
                    ) = chunkingRaycaster1Dv3(
                        sources,
                        sinks,
                        scattering_points,
                        filtered_index2[
                            temp_chunking[temp_chunkindex] : temp_chunking[
                                temp_chunkindex + 1
                            ],
                            :,
                        ],
                        environment,
                        True,
                    )
                    temp_index = np.append(temp_index, final_index3_part, axis=0)

                final_index3 = temp_index
            else:
                filtered_index3, final_index3, third_wave_rays = chunkingRaycaster1Dv3(
                    sources,
                    sinks,
                    scattering_points,
                    filtered_index2,
                    environment,
                    True,
                )

            # cuda.profile_stop()
            start = timer()
            # filtered_network2,final_network2,filtered_index2,final_index2=rayHits(second_ray_payload,distmap2,io_index2,scatter_inc=2,origins_local=second_origin)
            RAYS_CAST = first_wave_Rays + second_wave_rays + third_wave_rays
            # create combined path network
            host_size = np.shape(final_index)
            host_nans = np.zeros((host_size[0], 1), dtype=np.int32)
            temp_total = np.append(
                np.append(final_index, host_nans, axis=1), final_index2, axis=0
            )
            host_size2 = np.shape(temp_total)
            host_nans2 = np.zeros((host_size2[0], 1), dtype=np.int32)
            full_index_temp = np.append(
                np.append(temp_total, host_nans2, axis=1), final_index3, axis=0
            )
            full_index = np.append(full_index, full_index_temp, axis=0)
            # process for scattering matrix

    raycastingduration = raycasting_timestamp - timer()
    # print("Raycasting Duration:  {:3.1f} s, Total Rays: {:3.1f}".format(raycastingduration,RAYS_CAST) )
    return full_index, RAYS_CAST


def create_scatter_index(
    source_index, sink_index, scattering_point_index, scattering_mask
):
    # create target index for given scattering depth
    target_index = np.empty((0, np.max(scattering_mask) + 2), dtype=np.int32)
    # temp_index=create_model_index(source_index, sink_index, scattering_point_index)
    # stop_index=temp_index[np.isin(temp_index[:,-1],sink_index),:]
    # start_index=temp_index[~np.isin(temp_index[:,-1],sink_index),:]
    # final_index=create_model_index(start_index, sink_index, scattering_point_index)
    # target_index=np.append(np.append(stop_index,np.zeros((stop_index.shape[0],1),dtype=np.int32),axis=1),final_index,axis=0)
    for launch_round in range(np.max(scattering_mask) + 1):
        # create list of allowed points which always has all sinks.
        if launch_round == 0:
            allowed_illuminators = source_index
            # allowed_points=np.append(np.where(scattering_mask>=(launch_round+1))[0]+1,(np.where(scattering_mask==0)[0]+1))
            # allowed_points=copy.deepcopy(sink_index).ravel()
            allowed_scatter_points = (
                np.where(scattering_mask >= (launch_round + 1))[0] + 1
            )
            temp_index = create_model_index(
                allowed_illuminators,
                sink_index,
                allowed_scatter_points.reshape(allowed_scatter_points.shape[0], 1),
            )
            target_index = np.append(
                target_index,
                np.append(
                    temp_index,
                    np.zeros(
                        (
                            temp_index.shape[0],
                            target_index.shape[1] - temp_index.shape[1],
                        ),
                        dtype=np.int32,
                    ),
                    axis=1,
                ),
                axis=0,
            )
        else:
            allowed_illuminators = (
                np.where(scattering_mask >= (launch_round + 1))[0] + 1
            )
            # allowed_points=np.append(np.where(scattering_mask>=(launch_round+1))[0]+1,(np.where(scattering_mask==0)[0]+1))
            allowed_scatter_points = scattering_point_index
            propagate_index = np.where(
                target_index[:, launch_round] > np.max(sink_index)
            )[0]
            drain_trim = np.delete(
                propagate_index,
                np.isin(
                    target_index[propagate_index, launch_round], allowed_illuminators
                ),
            )
            drain_temp = create_model_index(
                target_index[drain_trim, 0 : launch_round + 1],
                sink_index,
                np.empty((0)),
            )
            illuminator_index = propagate_index[
                np.isin(
                    target_index[propagate_index, launch_round], allowed_illuminators
                )
            ]
            forward_temp = create_model_index(
                target_index[illuminator_index, 0 : launch_round + 1],
                sink_index,
                allowed_scatter_points.reshape(allowed_scatter_points.shape[0], 1),
            )
            trimmed_index = np.delete(target_index, propagate_index, axis=0)
            target_index = np.append(
                trimmed_index,
                np.append(
                    np.append(drain_temp, forward_temp, axis=0),
                    np.zeros(
                        (
                            drain_temp.shape[0] + forward_temp.shape[0],
                            target_index.shape[1] - drain_temp.shape[1],
                        ),
                        dtype=np.int32,
                    ),
                    axis=1,
                ),
                axis=0,
            )

        # so far this

    return target_index


def integratedraycastersetup(
    sources, sinks, scattering_points, environment, scattering_mask
):
    # temp function to chunk the number of rays to prevent creation of ray arrays to large for memory
    raycasting_timestamp = timer()
    max_scatter = np.max(scattering_mask)
    if max_scatter == 0:
        ray_estimate = sources * sinks
    elif max_scatter == 1:
        ray_estimate = (
            sources * (sinks + scattering_points.shape[0])
            + sources * scattering_points.shape[0] * sinks
        )
    elif max_scatter == 2:
        ray_estimate = (
            sources * (sinks + scattering_points.shape[0])
            + sources * scattering_points.shape[0] * sinks
            + sources
            * scattering_points.shape[0]
            * (scattering_points.shape[0])
            * sinks
        )
    # print("Total of {:3.1f} rays required".format(ray_estimate))
    # establish index boundaries
    source_index = np.arange(1, sources + 1).reshape(
        sources, 1
    )  # index starts at 1, leaving 0 as a null
    sink_index = np.arange(sources + 1, sources + 1 + sinks).reshape(sinks, 1)
    scattering_point_index = np.arange(
        np.max(sink_index) + 1,
        np.max(sink_index) + 1 + (scattering_points.shape[0] - sources - sinks),
    ).reshape((scattering_points.shape[0] - sources - sinks), 1)
    # implement chunking
    full_index = np.empty((0, np.max(scattering_mask) + 2), dtype=np.int32)
    chunknum = np.minimum(sources, np.int(np.ceil(ray_estimate / 2e8)))
    chunknum = np.maximum(2, chunknum)
    source_chunking = np.linspace(0, sources, chunknum, dtype=np.int32)
    for chunkindex in range(source_chunking.size - 1):
        io_indexing = create_scatter_index(
            source_index[source_chunking[chunkindex] : source_chunking[chunkindex + 1]],
            sink_index,
            scattering_point_index,
            scattering_mask,
        )
        # io_indexing=create_model_index(source_index[source_chunking[chunkindex]:source_chunking[chunkindex+1]],sink_index,np.empty((0,1),dtype=np.int32))
        full_index = np.append(
            full_index,
            integratedRaycaster(io_indexing, scattering_points, environment),
            axis=0,
        )

    # full_index=create_scatter_index(source_index,sink_index,scattering_point_index,scattering_mask)
    raycastingduration = timer() - raycasting_timestamp
    # print("Raycasting Duration:  {:3.1f} s, Total Rays: {:3.1f}".format(raycastingduration,full_index.shape[0]) )
    return full_index, io_indexing


if __name__ == "__main__":
    print("this is a library")

#pragma once

#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"
#include "gpu_error.cuh"


#include <iostream>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>




__device__ __inline__ bool hit(float3 ray_direction, float ray_length,float3 origin,float3 edge1,float3 edge2, float3 triangle_origin){

    // checking if ray is parallel to the triangle
    //equivilant to if dot_product(ray_direction, cross_product(edge1, edge2)) == 0
    float3 pvec = cross(ray_direction, edge2);
    float A = dot(edge1, pvec);

    if (std::abs(A)<EPSILON)
    {

    return 0;
    }
    if (A < 0)
    {
    return 0;
    }
    float3 tvec = origin - triangle_origin;


    // inverse determinant and calculate bounds of triangle
    float F = 1.0 / A;
    float U = F * (dot(tvec, pvec));


    // print('U',U)
    if (U < 0.0 || U > (1.0))
    {
    // in U coordinates, if U is less than 0 or greater than 1, it is outside the triangle
    return 0;
    }
    // cross product of tvec and E1
    float3 qvec = cross(tvec, edge1);
    float V = F * (dot(ray_direction, qvec));

    // print('V,V+U',V,V+U)
    if (V < 0.0 || (U + V) > (1.0))
    {
    // in UV coordinates, intersection is within triangle
    return 0;
    }
    float intersect_distance = F * (dot(edge2, qvec));

    if ((intersect_distance > (2 * EPSILON)) && (intersect_distance < (ray_length - (2 * EPSILON))))
    {
    // intersection on triangle
    return 1;
    }

    return 0;
}











__device__ __inline__  void points_to_rays(int tid,float3 *source, float3 *end, float4 *ray, int source_num, int end_num, int flag)
{

        // rays stored in end order major
        int e = tid %end_num;

        int c = tid /end_num;

        float4 r = make_float4(end[e].x - source[c].x,end[e].y - source[c].y,end[e].z - source[c].z,0);
        r.w = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        r.x = r.x/r.w;
        r.y = r.y/r.w;
        r.z = r.z/r.w;
        ray[tid] = r;

}

__global__ void set_values(int2 *ray_index, int ray_num, int2 value,int end_num)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  
    for(int i = thread; i < ray_num; i+=stride)
    {
        int c = i / end_num;
        int d = i % end_num;
        ray_index[i] = make_int2(c,d);

    }
}
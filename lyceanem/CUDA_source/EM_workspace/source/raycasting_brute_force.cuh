#pragma once

#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"
#include "gpu_error.cuh"
#include "em.cuh"
#include "raycasting.cuh"


#include <iostream>
#include <pybind11/numpy.h>
#include <assert.h>
#include <math.h> // for math functions like floorf

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__device__ __inline__ void intersection_brute_force(int i, float4 *ray, float3 *tri_vertex,int3* triangle_index, int tri_num, int ray_num, float3 *origin,int2 * ray_index,int end_num,int flag)
{
    


        //rays are stored in end order major
        int c = i / end_num;
        int d = i % end_num;
        if(flag || c!= d)
        {
            int b;
            bool intersected = false;
            float3 o = origin[c];
            float4 r = ray[i];

            for(int t = 0 ; t < tri_num ; t+=32)
            {
                __shared__ float3 edge1[32];
                __shared__ float3 edge2[32];
                __shared__ float3 triangle_origin[32];
                for(int j = 0; j < 32; j++){
                        edge1[j] = tri_vertex[triangle_index[t+j].y] - tri_vertex[triangle_index[t+j].x];
                        edge2[j] = tri_vertex[triangle_index[t+j].z] - tri_vertex[triangle_index[t+j].x];
                        triangle_origin[j] = tri_vertex[triangle_index[t+j].x];
                    
                }
                for(int j = 0; j < 32; j++){
                    b = hit(
                        make_float3(r.x,r.y,r.z),
                        r.w,
                        o,
                        edge1[j],
                        edge2[j],
                        triangle_origin[j]
                    );
                    intersected = (b == 1) ? 1 : intersected;


                }   
            }

            ray_index[i] = (intersected) ? make_int2(-1,-1): make_int2(c,d);

        }
 


    
}
__global__ void raycast_brute_force(float3 *source, float3 *end, float4 *ray, int source_num, int end_num, int flag, float3 *tri_vertex,int3 *triangle_index, int tri_num, int ray_num, int2 *ray_index,
                                PointData* points, float wave_length, complex_float3* scattering_network, const float2 alpha_beta)

{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  
    for(int i = thread; i < ray_num; i+=stride)
    {
        points_to_rays(i,source,end,ray,source_num,end_num,flag);
        intersection_brute_force(i,ray,tri_vertex,triangle_index,tri_num,ray_num,source,ray_index,end_num,flag);


        if(ray_index[i].x != -1){
            complex_float3 ray_field = em_wave(alpha_beta, ray[i],points[ray_index[i].x],points[ray_index[i].y],wave_length);
            scattering_network[i] = ray_field;
        }


    }

}

void raycast_wrapper_brute_force (float *source, float *end, int source_num, int end_num, float3 *d_tri_vertex,int3 *d_triangle_index, int tri_num, 
    PointData* points, float wave_length, complex_float3* h_scattering_network, float2 alpha_beta, bool not_self_to_self)
{
    // declare device memory
    float3 *d_source;
    float3 *d_end;
    float4 *d_ray;
    int2 *d_ray_index;


    // calculate size of arrays
    int source_size = source_num * sizeof(float3);
    int end_size = end_num * sizeof(float3);
    int ray_size = (source_num * end_num) * sizeof(float4);
    int ray_index_size = (source_num * end_num) * sizeof(int2);

    // allocate memory on device
    cudaMalloc((void**)&d_source, source_size);
    cudaMalloc((void**)&d_end, end_size);
    cudaMalloc((void**)&d_ray, ray_size);
    cudaMalloc((void**)&d_ray_index, ray_index_size);
    gpuErrchk( cudaGetLastError() );




    // copy data to device
    cudaMemcpy(d_source,(float3*) source, source_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_end,(float3*) end, end_size, cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );

    
    // set data to zero in 
    float3 zero = make_float3(0,0,0);

    cudaMemset(d_ray, 0, ray_size);
    set_values<<<32,512>>>(d_ray_index, (source_num * end_num),make_int2(-1,-1),end_num);
    gpuErrchk( cudaGetLastError() );
    PointData* d_points;
    complex_float3* d_scattering_network;

    // calculate size of arrays
    int points_size = (source_num+end_num) * sizeof(PointData);

    int scattering_network_size = (end_num*source_num) * sizeof(complex_float3);

    
    // allocate device memory
    cudaMalloc((void**)&d_points, points_size);
    cudaMalloc((void**)&d_scattering_network, scattering_network_size);
    gpuErrchk( cudaGetLastError() );

    // copy data to device
    cudaMemcpy(d_points, points, points_size, cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );



    // set to 0
    cudaMemset(d_scattering_network, 0, scattering_network_size);
    cudaDeviceSynchronize();


 
    //launch kernel for each ray_wave

    // source to sink


    raycast_brute_force<<<32,512>>>(d_source,d_end,d_ray,source_num,end_num,not_self_to_self,d_tri_vertex,d_triangle_index,tri_num,source_num*end_num, d_ray_index,
                                d_points, wave_length, d_scattering_network, alpha_beta);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    gpuErrchk( cudaGetLastError() );
  
    cudaFree(d_source);
    cudaFree(d_end);
    cudaFree(d_points);
    cudaFree(d_scattering_network);
    cudaFree(d_ray);
    cudaFree(d_ray_index);
    gpuErrchk( cudaGetLastError() );
    


}


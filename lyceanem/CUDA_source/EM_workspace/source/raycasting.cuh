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
#include "terrain_acceleration_build.cuh"




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

    





__device__ __inline__ void intersection(int i, float4 *ray, float3 *tri_vertex,int3* triangle_index, int tri_num, int ray_num, float3 *origin,int2 * ray_index,int end_num,int flag,int x_offset, int y_offset)
{
    


        //rays are stored in end order major
        int c = i / end_num;
        int d = i % end_num;
        if(flag || c!= d)
        {
            int b;
            int result = 0;
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
                    result = (b == 1) ? 1 : result;


                }   
            }

            ray_index[i] = (result == 0) ? make_int2(c+x_offset,d+y_offset) : make_int2(-1,-1);

        }
 


    
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
__global__ void raycast(float3 *source, float3 *end, float4 *ray, int source_num, int end_num, int flag, float3 *tri_vertex,int3 *triangle_index, int tri_num, int ray_num, int2 *ray_index, int x_offset, int y_offset)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  
    for(int i = thread; i < ray_num; i+=stride)
    {
        points_to_rays(i,source,end,ray,source_num,end_num,flag);
        intersection(i,ray,tri_vertex,triangle_index,tri_num,ray_num,source,ray_index,end_num,flag,x_offset,y_offset);


    }

}

std::pair<float4*,int2*>  raycast_wrapper (float *source, float *end, float* scatter, int source_num, int end_num, int scatter_num, float *tri_vertex,int *triangle_index, int tri_num)
{
    // declare device memory
    float3 *d_source;
    float3 *d_end;
    float3 *d_scatter;
    float4 *d_ray;
    int2 *d_ray_index;
    float3 *d_tri_vertex;
    int3 *d_triangle_index;

    // calculate size of arrays
    int source_size = source_num * sizeof(float3);
    int end_size = end_num * sizeof(float3);
    int scatter_size = scatter_num * sizeof(float3);
    int ray_size = (source_num * end_num+ source_num * scatter_num + scatter_num * end_num + scatter_num * scatter_num) * sizeof(float4);
    int ray_index_size = (source_num * end_num+ source_num * scatter_num + scatter_num * end_num + scatter_num * scatter_num) * sizeof(int2);
    int tri_vertex_size = tri_num * sizeof(float3);
    int triangle_index_size = tri_num * sizeof(int3);

    // allocate memory on device
    cudaMalloc((void**)&d_source, source_size);
    cudaMalloc((void**)&d_end, end_size);
    cudaMalloc((void**)&d_scatter, scatter_size);
    cudaMalloc((void**)&d_ray, ray_size);
    cudaMalloc((void**)&d_ray_index, ray_index_size);
    cudaMalloc((void**)&d_tri_vertex, tri_vertex_size);
    cudaMalloc((void**)&d_triangle_index, triangle_index_size);
    gpuErrchk( cudaGetLastError() );




    // copy data to device
    cudaMemcpy(d_source,(float3*) source, source_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_end,(float3*) end, end_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scatter,(float3*) scatter, scatter_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tri_vertex,(float3*) tri_vertex, tri_vertex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangle_index,(int3*) triangle_index, triangle_index_size, cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );

    
    // set data to zero in 
    float3 zero = make_float3(0,0,0);

    cudaMemset(d_ray, 0, ray_size);
    cudaMemset(d_ray_index, -1, ray_index_size);
    gpuErrchk( cudaGetLastError() );


 
    //launch kernel for each ray_wave

    // source to sink
    bool not_self_to_self = (d_source != d_end);
    printf("source_sink");
    raycast<<<32,32>>>(d_source,d_end,d_ray,source_num,end_num,not_self_to_self,d_tri_vertex,d_triangle_index,tri_num,source_num*end_num, d_ray_index,0,source_num);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    gpuErrchk( cudaGetLastError() );
    int rays_filled = source_num * end_num;


    // source to scatter
    printf("source_scatter");
    not_self_to_self = (d_source != d_scatter);
    raycast<<<32,32>>>(d_source,d_scatter,&d_ray[rays_filled],source_num,scatter_num,not_self_to_self,d_tri_vertex,d_triangle_index,tri_num,source_num*scatter_num, &d_ray_index[rays_filled],0,source_num+end_num);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    gpuErrchk( cudaGetLastError() );
    rays_filled += source_num * scatter_num;

    // scatter to sink
    printf("scatter_sink");
    not_self_to_self = (d_scatter != d_end);
    raycast<<<32,32>>>(d_scatter,d_end,&d_ray[rays_filled],scatter_num,end_num,not_self_to_self,d_tri_vertex,d_triangle_index,tri_num,scatter_num*end_num, &d_ray_index[rays_filled],source_num+end_num,source_num);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    gpuErrchk( cudaGetLastError() );
    rays_filled += scatter_num * end_num;


    // scatter to scatter
    printf("scatter_scatter");
    not_self_to_self = (d_scatter != d_scatter);
    raycast<<<32,32>>>(d_scatter,d_scatter,&d_ray[rays_filled],scatter_num,scatter_num,not_self_to_self,d_tri_vertex,d_triangle_index,tri_num,scatter_num*scatter_num, &d_ray_index[rays_filled],source_num+end_num,source_num+end_num);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    gpuErrchk( cudaGetLastError() );
    printf("error %s\n",cudaGetErrorString(cudaGetLastError()));
    // free unneeded arrays
    cudaFree(d_source);
    cudaFree(d_end);
    cudaFree(d_scatter);
    cudaFree(d_tri_vertex);
    cudaFree(d_triangle_index);
    
    return std::make_pair(d_ray,d_ray_index);


}




#pragma once

#include <cuda_runtime.h>

#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"
#include "gpu_error.cuh"


#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tuple>
#include <assert.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <math.h>
namespace py = pybind11;



__device__ __inline__ static int bin_point( float2 y_range, float2 z_range, int2 num_bins, float3 point){
    /* Should only be used to build the grid structure as makes assertions about points being in the bounds of grid*/


    int biny = floorf((point.y-y_range.x)/y_range.y);
    int binz = floorf((point.z-z_range.x)/z_range.y);
    ////printf("BINNING %f %f %f  %i\n",point.y,y_range.x,y_range.y,biny);
    float lower_y = y_range.x + biny * y_range.y;
    float upper_y = y_range.x + (biny+1) * y_range.y;
    float lower_z = z_range.x + binz * z_range.y;
    float upper_z = z_range.x + (binz+1) * z_range.y;
    
    
    bool all_triangles_are_within_provided_yz_bounds = (point.y >= y_range.x && point.y <= y_range.y * num_bins.x + y_range.x && point.z >= z_range.x && point.z <= z_range.y * num_bins.y + z_range.x);
    bool check_in_bin = (point.y >= lower_y && point.y < upper_y && point.z >= lower_z && point.z < upper_z);
    if (!check_in_bin){
        printf("Point %f %f %f not in bin %f %f %f %f %f %f\n",point.x,point.y,point.z,lower_y,upper_y,lower_z,upper_z,y_range.y,z_range.y);
    }
    if (!all_triangles_are_within_provided_yz_bounds){
        printf("Point %f %f %f not in bounds %f %f %f %f\n",point.x,point.y,point.z,y_range.x,y_range.y*num_bins.x+y_range.x,z_range.x,z_range.y*num_bins.y+z_range.x);
        printf("num_bins %i %i\n",num_bins.x,num_bins.y);
        printf("tilewidth %f %f\n",y_range.y,z_range.y);
    }

    assert(check_in_bin && all_triangles_are_within_provided_yz_bounds);


    return biny * num_bins.y + binz;
}

__device__ __inline__ void count_triangles_in_bin(float3* points, float2 y_range, float2 z_range, int2 num_bins, int triangle_number, int3* triangles, int* bin_counts, int tid,int  stride){

    for(int t = tid; t < triangle_number; t += stride){

        // unroll the loop
        int bin_x = bin_point( y_range, z_range, num_bins, points[triangles[t].x]);
        atomicAdd(&bin_counts[bin_x], 1);
        int bin_y = bin_point( y_range, z_range, num_bins, points[triangles[t].y]);
        if(bin_y != bin_x){
            atomicAdd(&bin_counts[bin_y], 1);
        }
        int bin_z = bin_point( y_range, z_range, num_bins, points[triangles[t].z]);
        if(bin_z != bin_x && bin_z != bin_y){
            atomicAdd(&bin_counts[bin_z], 1);
        }
        if(triangles[t].x == 7 || triangles[t].y == 7 || triangles[t].z == 7){
            int binnn = bin_point( y_range, z_range, num_bins, points[7]);
        }
        if(bin_z != bin_x && bin_z != bin_y && bin_y != bin_x){
            int2 binx = make_int2(bin_x/num_bins.y,bin_x%num_bins.y);
            int2 biny = make_int2(bin_y/num_bins.y,bin_y%num_bins.y);
            int2 binz = make_int2(bin_z/num_bins.y,bin_z%num_bins.y);
            // need to add to 4th bin in case of 4th bin
            int2 temp = make_int2(binx.x-biny.x,binx.y-biny.y);
            int2 result = make_int2(-1,-1);
            if (temp.x == 0)
            {
                result.x = binz.x;
            }
            if (temp.y == 0)
            {
                result.y = binz.y;
            }
            temp = make_int2(binx.x-binz.x,binx.y-binz.y);
            if (temp.x == 0)
            {
                result.x = biny.x;
            }
            if (temp.y == 0)
            {
                result.y = biny.y;
            }
            temp = make_int2(biny.x-binz.x,biny.y-binz.y);
            if (temp.x == 0)
            {
                result.x = binx.x;
            }
            if (temp.y == 0)
            {
                result.y = binx.y;
            }
            int bin_4 = result.x*num_bins.y + result.y;

            atomicAdd(&bin_counts[bin_4], 1);
        }
    }
}
__device__ __inline__ static int check_triangle_side_length(int3* triangle, float3* points, int tid, int stride, int triangle_number, float2 y_range, float2 z_range, int2 num_bins){
    for(int i = tid; i < triangle_number; i+= stride){
        int bin1 = bin_point(y_range, z_range, num_bins, points[triangle[i].x]);
        int bin2 = bin_point(y_range, z_range, num_bins, points[triangle[i].y]);
        int bin3 = bin_point(y_range, z_range, num_bins, points[triangle[i].z]);
        //check bins are only one away from each other
        int2 bin1_2 = make_int2(bin1/num_bins.y,bin1%num_bins.y);
        int2 bin2_2 = make_int2(bin2/num_bins.y,bin2%num_bins.y);
        int2 bin3_2 = make_int2(bin3/num_bins.y,bin3%num_bins.y);
        bool side12 = (abs(bin1_2.x-bin2_2.x) <= 1 && abs(bin1_2.y-bin2_2.y) <= 1);
        bool side13 = (abs(bin1_2.x-bin3_2.x) <= 1 && abs(bin1_2.y-bin3_2.y) <= 1);
        bool side23 = (abs(bin2_2.x-bin3_2.x) <= 1 && abs(bin2_2.y-bin3_2.y) <= 1);
        

        bool bins_ajacent = ((side12 && side13 && side23));
        if (!bins_ajacent){
            printf("Triangle %i %i %i not in ajacent bins %i %i %i %i %i %i\n",triangle[i].x,triangle[i].y,triangle[i].z,bin1_2.x,bin1_2.y,bin2_2.x,bin2_2.y,bin3_2.x,bin3_2.y);
        }
        assert(bins_ajacent);
    }
}
__device__ __inline__ void compute_bin_offsets(int* bin_counts, int2 num_bins, int2* bin_offsets,int tid,int stride){
    for(int i = tid; i <num_bins.x * num_bins.y; i+=stride){
        for(int j = 0; j < i; j++){
            bin_offsets[i].x += bin_counts[j];
        }
        bin_offsets[i].y = bin_counts[i];
    }
}

__device__ __inline__ void insert_triangle_in_bin(int2 bin_offset, int triangle_index, int* binned_triangles){
    int offset = bin_offset.x;
    int old_value;
    
    do {
        old_value = atomicCAS(&binned_triangles[offset], -1, triangle_index);
        offset++;
    } while (old_value != -1);
}

__device__ __inline__ void bin_triangles(int3* triangles, float3* points, float2 y_range, float2 z_range, int2 num_bins, int* bin_counts, int2* bin_offsets, int* binned_triangles, int triangle_number, int tid,int  stride){
    for(int t = tid; t < triangle_number; t += stride){
        int bin_x = bin_point( y_range, z_range, num_bins, points[triangles[t].x]);
        insert_triangle_in_bin(bin_offsets[bin_x], t, binned_triangles);
        int bin_y = bin_point( y_range, z_range, num_bins, points[triangles[t].y]);
        if(bin_y != bin_x){
            insert_triangle_in_bin(bin_offsets[bin_y], t, binned_triangles);
        }
        int bin_z = bin_point( y_range, z_range, num_bins, points[triangles[t].z]);
        if(bin_z != bin_x && bin_z != bin_y){
            insert_triangle_in_bin(bin_offsets[bin_z], t, binned_triangles);
        }
        if(bin_z != bin_x && bin_z != bin_y && bin_y != bin_x){
            int2 binx = make_int2(bin_x/num_bins.y,bin_x%num_bins.y);
            int2 biny = make_int2(bin_y/num_bins.y,bin_y%num_bins.y);
            int2 binz = make_int2(bin_z/num_bins.y,bin_z%num_bins.y);
            // need to add to 4th bin in case of 4th bin
            int2 temp = make_int2(binx.x-biny.x,binx.y-biny.y);
            int2 result = make_int2(-1,-1);
            if (temp.x == 0)
            {
                result.x = binz.x;
            }
            if (temp.y == 0)
            {
                result.y = binz.y;
            }
            temp = make_int2(binx.x-binz.x,binx.y-binz.y);
            if (temp.x == 0)
            {
                result.x = biny.x;
            }
            if (temp.y == 0)
            {
                result.y = biny.y;
            }
            temp = make_int2(biny.x-binz.x,biny.y-binz.y);
            if (temp.x == 0)
            {
                result.x = binx.x;
            }
            if (temp.y == 0)
            {
                result.y = binx.y;
            }
            int bin_4 = result.x*num_bins.y + result.y;

            insert_triangle_in_bin(bin_offsets[bin_4], t, binned_triangles);

        }
    }
}

__global__ void bin_triangles_kernel(int3* triangles, float3* points, float2 y_range, float2 z_range, int2 num_bins, int* bin_counts, int2* bin_offsets, int* binned_triangles, int triangle_number, int3* binned_triangles_index){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i < num_bins.x * num_bins.y; i+=stride){
        bin_counts[i] = 0;
        bin_offsets[i] = make_int2(0,0);
    }
    for(int i = tid; i < triangle_number*4 ; i+=stride){
        binned_triangles[i] = -1;
    }
    check_triangle_side_length(triangles, points, tid, stride, triangle_number, y_range, z_range, num_bins);

    count_triangles_in_bin(points, y_range, z_range, num_bins, triangle_number, triangles, bin_counts, tid, stride);
    __syncthreads();
    compute_bin_offsets(bin_counts, num_bins, bin_offsets, tid, stride);
    __syncthreads();
    bin_triangles(triangles, points, y_range, z_range, num_bins, bin_counts, bin_offsets, binned_triangles, triangle_number, tid, stride);
    __syncthreads();
    for(int i = tid; i < triangle_number*4; i+=stride){
        if (binned_triangles[i] != -1){
            binned_triangles_index[i] = triangles[binned_triangles[i]]; 
        }
        else {
            binned_triangles_index[i] = make_int3(-1,-1,-1);
        }
    }
}
std::tuple<int3*,int*,float3*>   grid_structure_builder_wrapper(float *tri_vertex,int *triangle_index, int tri_num, int2 n_cells, float2 x_top_bottom, float2 y_range, float2 z_range, int vertex_num)
{
    // declare device memory
    float3 *d_tri_vertex;
    int3 *d_triangle_index_binned;
    int3* d_triangles;
    int *d_binned_triangles;
    int *d_bin_counts;
    int2 *d_bin_offsets;


    // allocate memory on device
    cudaMalloc((void**)&d_tri_vertex, vertex_num * sizeof(float3));
    gpuErrchk( cudaGetLastError() );

    cudaMalloc((void**)&d_triangle_index_binned, tri_num * 3*sizeof(int3));
    cudaMalloc((void**)&d_triangles, tri_num * sizeof(int3));
    gpuErrchk( cudaGetLastError() );

    cudaMalloc((void**)&d_binned_triangles, tri_num * 3 * sizeof(int));
    gpuErrchk( cudaGetLastError() );

    cudaMalloc((void**)&d_bin_counts, n_cells.x * n_cells.y * sizeof(int));
    gpuErrchk( cudaGetLastError() );

    cudaMalloc((void**)&d_bin_offsets, n_cells.x * n_cells.y * sizeof(int2));
    gpuErrchk( cudaGetLastError() );




    // copy data to device
    cudaMemcpy(d_tri_vertex,(float3*) tri_vertex, vertex_num * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles,(int3*) triangle_index, tri_num * sizeof(int3), cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );


    
 
    //launch kernel for each ray_wave
    bin_triangles_kernel<<<32, 256>>>(d_triangles, d_tri_vertex, y_range, z_range, n_cells, d_bin_counts, d_bin_offsets, d_binned_triangles, tri_num, d_triangle_index_binned);

    //alocate host memory
    // free unneeded arrays
    cudaFree(d_triangles);
    cudaFree(d_binned_triangles);
    cudaFree(d_bin_offsets);
    gpuErrchk( cudaGetLastError() );
    
    return std::make_tuple(d_triangle_index_binned,d_bin_counts,d_tri_vertex);


}
__global__ void triangle_counter_kernel(float3* triangle_vertex, float2 y_range, float2 z_range, int2 num_bins, int triangle_number, int3* triangles, int* bin_counts){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    check_triangle_side_length(triangles, triangle_vertex, tid, stride, triangle_number, y_range, z_range, num_bins);

    count_triangles_in_bin(triangle_vertex, y_range, z_range, num_bins, triangle_number, triangles, bin_counts, tid, stride);
}


py::array_t<int> bin_counts_to_numpy(py::array_t<float> tri_vertex, py::array_t<int> tri_index,int ncells_x,int ncells_y, float ymin, float cell_width, float z_min ){
    
    //validate input
    if (tri_vertex.ndim() != 2 || tri_vertex.shape(1) != 3){throw std::runtime_error("tri_vertex must be a 2D array with 3 columns");}
    if (tri_index.ndim() != 2 || tri_index.shape(1) != 3){throw std::runtime_error("tri_index must be a 2D array with 3 columns");}

    //get buffers
    auto tri_vertex_buf = tri_vertex.request();
    auto tri_index_buf = tri_index.request();

    //get pointers
    float *tri_vertex_ptr = (float *) tri_vertex_buf.ptr;
    int *tri_index_ptr = (int *) tri_index_buf.ptr;


    // device memory
    float3 *d_tri_vertex;
    int3 *d_triangles;
    int *d_bin_counts;

    // allocate memory on device
    cudaMalloc((void**)&d_tri_vertex, tri_vertex.shape(0) * sizeof(float3));
    cudaMalloc((void**)&d_triangles, tri_index.shape(0) * sizeof(int3));
    cudaMalloc((void**)&d_bin_counts, ncells_x * ncells_y * sizeof(int));
    gpuErrchk( cudaGetLastError() );

    // copy data to device
    cudaMemcpy(d_tri_vertex,(float3*) tri_vertex_ptr, tri_vertex.shape(0) * sizeof(float3), cudaMemcpyHostToDevice);

    cudaMemcpy(d_triangles,(int3*) tri_index_ptr, tri_index.shape(0) * sizeof(int3), cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );
    cudaMemset(d_bin_counts, 0, ncells_x * ncells_y * sizeof(int));

    //launch kernel for each ray_wave
    triangle_counter_kernel<<<32,256>>>(d_tri_vertex, make_float2(ymin,cell_width), make_float2(z_min,cell_width), make_int2(ncells_x,ncells_y), tri_index.shape(0), d_triangles, d_bin_counts);

    //declare new numpy array
    py::array_t<int> bin_counts = py::array_t<int>(ncells_x * ncells_y);
    auto bin_counts_buf = bin_counts.request();
    int *bin_counts_ptr = (int *) bin_counts_buf.ptr;

    cudaMemcpy(bin_counts_ptr, d_bin_counts, ncells_x * ncells_y * sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk( cudaGetLastError() );
    cudaFree(d_tri_vertex);
    cudaFree(d_triangles);
    cudaFree(d_bin_counts);
    gpuErrchk( cudaGetLastError() );
    return bin_counts;




}
__global__ void bin_triangles_kernel_precounted(int3* triangles, float3* points, float2 y_range, float2 z_range, int2 num_bins, int* bin_counts, int2* bin_offsets, int* binned_triangles, int triangle_number, int3* binned_triangles_index,int sum_bin_counts){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i < sum_bin_counts; i+=stride){
        binned_triangles[i] = -1;
    }

    compute_bin_offsets(bin_counts, num_bins, bin_offsets, tid, stride);
    __syncthreads();
    bin_triangles(triangles, points, y_range, z_range, num_bins, bin_counts, bin_offsets, binned_triangles, triangle_number, tid, stride);
    __syncthreads();
    for(int i = tid; i < sum_bin_counts; i+=stride){
        if (binned_triangles[i] != -1){
            binned_triangles_index[i] = triangles[binned_triangles[i]]; 
        }
        else {
            binned_triangles_index[i] = make_int3(-1,-1,-1);
        }
    }
}

py::array_t<int> bin_triangles_to_numpy(py::array_t<float> tri_vertex, py::array_t<int> tri_index,int ncells_x,int ncells_y, float ymin, float cell_width, float z_min,py::array_t<int> bin_counts, int sum_bin_counts){
    
    //validate input
    if (tri_vertex.ndim() != 2 || tri_vertex.shape(1) != 3){throw std::runtime_error("tri_vertex must be a 2D array with 3 columns");}
    if (tri_index.ndim() != 2 || tri_index.shape(1) != 3){throw std::runtime_error("tri_index must be a 2D array with 3 columns");}
    if (bin_counts.ndim() != 1|| bin_counts.shape(0)!= ncells_x*ncells_y){throw std::runtime_error("bincounts is a 1d flattened 2d array thus must have 1d with ncountsy *ncountz");}

    //get buffers
    auto tri_vertex_buf = tri_vertex.request();
    auto tri_index_buf = tri_index.request();
    auto bin_counts_buf = bin_counts.request();

    //get pointers
    float *tri_vertex_ptr = (float *) tri_vertex_buf.ptr;
    int *tri_index_ptr = (int *) tri_index_buf.ptr;
    int *bin_counts_ptr = (int *) bin_counts_buf.ptr;

    // device memory
    float3 *d_tri_vertex;
    int3 *d_triangles;
    int *d_bin_counts;
    int2 *d_bin_offsets;
    int3 *d_binned_triangles;
    int *d_binnedtriangle_number;
    
    // allocate memory on device
    cudaMalloc((void**)&d_tri_vertex, tri_vertex.shape(0) * sizeof(float3));
    cudaMalloc((void**)&d_triangles, tri_index.shape(0) * sizeof(int3));
    cudaMalloc((void**)&d_bin_counts, ncells_x * ncells_y * sizeof(int));
    cudaMalloc((void**)&d_bin_offsets, ncells_x * ncells_y * sizeof(int2));
    cudaMalloc((void**)&d_binned_triangles, sum_bin_counts * sizeof(int3));
    cudaMalloc((void**)&d_binnedtriangle_number, sum_bin_counts* sizeof(int));
    gpuErrchk( cudaGetLastError() );

    // copy data to device
    cudaMemcpy(d_tri_vertex,(float3*) tri_vertex_ptr, tri_vertex.shape(0) * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles,(int3*) tri_index_ptr, tri_index.shape(0) * sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_counts,(int*) bin_counts_ptr, bin_counts.shape(0) * sizeof(int), cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );

    cudaMemset(d_binnedtriangle_number, 0, sum_bin_counts * sizeof(int));
    cudaMemset(d_bin_offsets, 0, ncells_x * ncells_y * sizeof(int2));
    cudaMemset(d_binned_triangles, 0, sum_bin_counts * sizeof(int3));

    //launch kernel for each ray_wave
    bin_triangles_kernel_precounted<<<1, 512>>>(d_triangles, d_tri_vertex, make_float2(ymin,cell_width), make_float2(z_min,cell_width), make_int2(ncells_x,ncells_y), d_bin_counts, d_bin_offsets, d_binnedtriangle_number, tri_index.shape(0), d_binned_triangles, sum_bin_counts);
    //new numpy array
    py::array_t<int> binned_triangles = py::array_t<int>(sum_bin_counts*3);
    auto binned_triangles_buf = binned_triangles.request();
    int *binned_triangles_ptr = (int *) binned_triangles_buf.ptr;

    cudaMemcpy(binned_triangles_ptr, d_binned_triangles, sum_bin_counts*3 * sizeof(int), cudaMemcpyDeviceToHost);   
    gpuErrchk( cudaGetLastError() );
    cudaFree(d_tri_vertex);
    cudaFree(d_triangles);
    cudaFree(d_bin_counts);
    cudaFree(d_bin_offsets);
    cudaFree(d_binned_triangles);
    cudaFree(d_binnedtriangle_number);
    gpuErrchk( cudaGetLastError() );
    return binned_triangles;
}





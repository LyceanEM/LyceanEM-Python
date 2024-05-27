#pragma once
#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"

#include "vector_types.h"

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "cuComplex.h"
#include <math.h>
#include "math_constants.h"

#include "gpu_error.cuh"

__device__ __inline__ complex_float3 ray_launch(const complex_float3 & e_field,float3 ray){
    //vectors that are the 3 axis of cartesian space
    float3 x_axis = make_float3(1,0,0);
    float3 y_axis = make_float3(0,1,0);
    float3 z_axis = make_float3(0,0,1);

    float x_norm = length(cross(x_axis,ray));
    float y_norm = length(cross(y_axis,ray));
    float z_norm = length(cross(z_axis,ray));


    float3 ray_u;
    if (abs(x_norm) > abs(y_norm) && abs(x_norm) > abs(z_norm)){
        ray_u = normalize(cross(x_axis,ray));
    }
    else if (abs(y_norm) >= abs(x_norm) && abs(y_norm) > abs(z_norm)){
        ray_u = normalize(cross(y_axis,ray));
    }
    else if (abs(z_norm) >= abs(x_norm) && abs(z_norm) >= abs(y_norm)){
        ray_u = normalize(cross(z_axis,ray));
    }
    float3 ray_v = normalize(cross(ray,ray_u));


    cuFloatComplex e_field_axis_u = complex_dot_real(e_field,ray_u);
    cuFloatComplex e_field_axis_v  = complex_dot_real(e_field,ray_v);
    
    complex_float3 ray_field;
    /*
    ray_field.x =e_field_axis_u * ray_u.x + e_field_axis_v * ray_v.x;
    ray_field.y =e_field_axis_u * ray_u.y + e_field_axis_v * ray_v.y;
    ray_field.z =e_field_axis_u * ray_u.z + e_field_axis_v * ray_v.z;
    */
    //change to explicit __fmaf_rn
    ray_field.x.x = __fmaf_rn(e_field_axis_u.x, ray_u.x, __fmaf_rn(e_field_axis_v.x, ray_v.x, 0));
    ray_field.x.y = __fmaf_rn(e_field_axis_u.y, ray_u.x, __fmaf_rn(e_field_axis_v.y, ray_v.x, 0));
    ray_field.y.x = __fmaf_rn(e_field_axis_u.x, ray_u.y, __fmaf_rn(e_field_axis_v.x, ray_v.y, 0));
    ray_field.y.y = __fmaf_rn(e_field_axis_u.y, ray_u.y, __fmaf_rn(e_field_axis_v.y, ray_v.y, 0));
    ray_field.z.x = __fmaf_rn(e_field_axis_u.x, ray_u.z, __fmaf_rn(e_field_axis_v.x, ray_v.z, 0));
    ray_field.z.y = __fmaf_rn(e_field_axis_u.y, ray_u.z, __fmaf_rn(e_field_axis_v.y, ray_v.z, 0));
    

    return ray_field;
}


__device__ __inline__ complex_float3 frequency_em_wave(const complex_float3& ray_electric_field, int ray_wave, const float4& ray, const PointData& origin, const PointData& end) {


    complex_float3 ray_field;
    if (ray_wave == 0){

        if (origin.is_electric){
            ray_field = origin.electric_field;

        }
        else{
            float source_impedance = sqrt(origin.permeability.x / origin.permittivity.x);
            //printf("source_impedance %f\n",source_impedance);
            ray_field =  complex_cross_real(origin.electric_field, make_float3(ray.x,ray.y,ray.z)); 
            ray_field /= source_impedance;

        }

    }
    else {
         ray_field = ray_launch(ray_electric_field,origin.normal);
   
         
    }

    
    
    complex_float3 ray_field2 = ray_launch(ray_field,make_float3(ray.x,ray.y,ray.z));



    ray_field2 *= end.electric_field;


    return ray_field2;

}
__device__ __inline__ void frequency_em_loss( complex_float3 & ray_electric_field, float distance,float wave_vector){
    cuFloatComplex loss;
    
    sincosf(-distance*wave_vector, &loss.y, &loss.x);
    loss.y /= (distance*wave_vector*2);
    loss.x /= (distance*wave_vector*2);
    ray_electric_field *= loss;
}


template <int scatter_depth>
__global__ void frequency_em(
     float4* ray, PointData* points, int2* ray_index, float wave_vector, complex_float3* scattering_network, int source_num, int scatter_num, int end_num){
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int2 segment[scatter_depth+1];
    complex_float3 ray_electric_field;
    ray_electric_field.x.x = 0;
    ray_electric_field.x.y = 0;
    ray_electric_field.y.x = 0;
    ray_electric_field.y.y = 0;
    ray_electric_field.z.x = 0;
    ray_electric_field.z.y = 0;


    int source_end_chunk_size = source_num*end_num;
    int source_scatter_chunk_size = source_num*scatter_num;
    int scatter_end_chunk_size = scatter_num*end_num;
    int scatter_scatter_chunk_size = scatter_num*scatter_num;

    int source_end_chunk_start = 0;
    int source_scatter_chunk_start = source_end_chunk_size;
    int scatter_end_chunk_start = source_scatter_chunk_start + source_scatter_chunk_size;
    int scatter_scatter_chunk_start = scatter_end_chunk_start + scatter_end_chunk_size;

    if constexpr(scatter_depth == 0){
        segment[0] = make_int2(source_end_chunk_start,source_end_chunk_size);
    }
    else if constexpr(scatter_depth == 1){
        segment[0] = make_int2(source_scatter_chunk_start, source_scatter_chunk_size);
        segment[1] = make_int2(scatter_end_chunk_start, scatter_end_chunk_size);
    }


    int ray_wave;
    int source = 0;
    int2 ray_index1;
    int2 ray_index2;
    for(int tid = thread; tid < segment[0].y; tid+=stride){
        bool check_first_segment_exists = ((ray_index[segment[0].x+tid].x!= -1));
        if(check_first_segment_exists){  
            ray_wave = 0;

            ray_index1 = ray_index[segment[0].x+tid];
 
            complex_float3 ray_electric_field1 = frequency_em_wave(ray_electric_field, ray_wave, ray[segment[0].x+tid], points[ray_index1.x], points[ray_index1.y]);

            if constexpr (scatter_depth >0){
                int rays_from_point_wave1 = segment[1].y/scatter_num;
                ray_wave = 1;
                int index_second_segment = segment[1].x + tid%scatter_num * rays_from_point_wave1;
                for(int j = 0 ; j < rays_from_point_wave1; j++){

                    bool check_second_segment_exists = ((ray_index[index_second_segment+j].x != -1));
                    if(check_second_segment_exists){
                        ray_index2 = ray_index[index_second_segment+j];
                        int sink = j;
                        int source = tid/scatter_num;
                        complex_float3 ray_electric_field2 = frequency_em_wave(ray_electric_field1, ray_wave, ray[index_second_segment+j], points[ray_index2.x],points[ray_index2.y]);
            
                        /*if constexpr (scatter_depth >1){
                            int rays_from_point_wave2 = segment[2].y/scatter_num;
                            ray_wave = 2;
                            for(int k = 0 ; k < rays_from_point_wave2; k++){
                                bool check_third_segment_exists = ((ray_index[segment[2].x+k].x != -1) && (ray_index[segment[2].x+k].y != -1));
                                if(check_third_segment_exists){
                                    complex_float3 ray_electric_field3 = frequency_em_wave(ray_electric_field2, ray_wave, ray[segment[2].x+k], points[ray_index[segment[2].x+k].x],points[ray_index[segment[2].x+k].y]);
                                
                                    if constexpr (scatter_depth >2){
                                        int rays_from_point_wave3 = segment[3].y/scatter_num;
                                        ray_wave = 3;
                                        for(int l = 0 ; l < rays_from_point_wave3; l++){
                                            bool check_fourth_segment_exists = ((ray_index[segment[3].x+l].x != -1) && (ray_index[segment[3].x+l].y != -1) );
                                            if(check_fourth_segment_exists){
                                                complex_float3 ray_electric_field4 = frequency_em_wave(ray_electric_field3, ray_wave, ray[segment[3].x+l], points[ray_index[segment[3].x+l].x],points[ray_index[segment[3].x+l].y]);
                                            
                                                if constexpr (scatter_depth ==4 ){
                                                    ray_wave = 4;
                                                    int rays_from_point_wave4 = segment[4].y/scatter_num;
                                                    for(int o = 0 ; o < rays_from_point_wave4; o++){
                                                        bool check_fifth_segment_exists = ((ray_index[segment[4].x+o].x != -1) && (ray_index[segment[4].x+o].y != -1));
                                                        if(check_fifth_segment_exists){
                                                            
                                                            complex_float3 ray_electric_field5 = frequency_em_wave(ray_electric_field4, ray_wave, ray[segment[4].x+o], points[ray_index[segment[4].x+o].x],points[ray_index[segment[4].x+o].y]);
                                                        
                                                            float distance = ray[tid+segment[0].x].w + ray[segment[1].x+j].w + ray[segment[2].x+k].w + ray[segment[3].x+l].w+ ray[segment[4].x+o].w;
                                                            frequency_em_loss(ray_electric_field5, distance, wave_vector);
                                                            int sink = o/scatter_num;
                                                            int source = tid%source_num;
                                                            scattering_network[end_num*source + sink].plusEqualAtomic(ray_electric_field5);
                                                        }
                                                    }

                                                }
                                                else if constexpr (scatter_depth==3){
                                                    float distance = ray[segment[0].x+tid].w + ray[segment[1].x+j].w + ray[segment[2].x+k].w + ray[segment[3].x+l].w;
                                                    frequency_em_loss(ray_electric_field4, distance, wave_vector);
                                                    int sink = l/scatter_num;
                                                    int source = tid%source_num;
                                                    scattering_network[end_num*source + sink].plusEqualAtomic(ray_electric_field4);
                                                }
                                            }
                                        }
                                    }
                                    else if constexpr(scatter_depth==2){
                                        float distance = ray[segment[0].x+tid].w + ray[segment[1].x+j].w + ray[segment[2].x+k].w;
                                        frequency_em_loss(ray_electric_field3, distance, wave_vector);
                                        int sink = k/scatter_num;
                                        int source = tid%source_num;
                                        scattering_network[end_num*source + sink].plusEqualAtomic(ray_electric_field3);
                                    }
                                }
                            }
                        }  */ 
                         if constexpr(scatter_depth==1){
                            float distance = ray[segment[0].x+tid].w + ray[index_second_segment+j].w;

                            frequency_em_loss(ray_electric_field2, distance, wave_vector);
                            int sink = j;
                            int source = tid/scatter_num;
                            scattering_network[end_num*source + sink].plusEqualAtomic(ray_electric_field2);


                        }
                    }
                }
            }
            else if constexpr(scatter_depth==0){
                float distance = ray[segment[0].x+tid].w;
                int sink = tid%end_num;
                int source = tid/end_num;
                frequency_em_loss(ray_electric_field1, distance, wave_vector);
                scattering_network[end_num*source + sink].plusEqualAtomic(ray_electric_field1);
            }
        }
    }
}
#include <chrono>
// concrete versions
void frequency_wrapper(int scatter_depth, float4* d_ray, PointData* points, int2* d_ray_index,
     float wave_vector, int source_num, int scatter_num, int end_num, complex_float3* h_scattering_network){
    //declare device pointers
    PointData* d_points;
    complex_float3* d_scattering_network;

    // calculate size of arrays
    int points_size = (source_num+scatter_num+end_num) * sizeof(PointData);

    int scattering_network_size = (end_num*source_num) * sizeof(complex_float3);
    auto start = std::chrono::high_resolution_clock::now();

    
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
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time to copy points to device: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    if (scatter_depth == 0){


        frequency_em<0><<<1,1,1>>>(d_ray, d_points, d_ray_index, wave_vector, d_scattering_network, source_num, scatter_num, end_num);

    }
    else if (scatter_depth == 1){
        frequency_em<1><<<1,1,1>>>(d_ray, d_points, d_ray_index, wave_vector, d_scattering_network, source_num, scatter_num, end_num);
        

        //frequency_em<0><<<1,1,1>>>(d_ray, d_points, d_ray_index, wave_vector, d_scattering_network, source_num, scatter_num, end_num);
// Wait for the kernel to finish
 // Wait for the kernel to finish

    }
    //sync threads
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time to run kernes: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count() << "ms\n";


    gpuErrchk( cudaGetLastError() );
    //printf("error %s\n",cudaGetErrorString(cudaGetLastError()));



    //allocate host memory


    // copy data back to host
    cudaMemcpy(h_scattering_network, d_scattering_network, scattering_network_size, cudaMemcpyDeviceToHost);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::cout << "Time to copy scattering network to host: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2).count() << "ms\n";
    



    // free device memory
    cudaFree(d_points);
    cudaFree(d_scattering_network);
    cudaFree(d_ray);
    cudaFree(d_ray_index);
    gpuErrchk( cudaGetLastError() );





}

                        








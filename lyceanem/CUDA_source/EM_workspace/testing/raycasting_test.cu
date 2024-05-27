#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../source/math_utility.cuh"
#include "raycasting.cuh"

#include "vector_types.h"

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "em_test.cuh"
#include "gpu_error.cuh"


namespace py = pybind11;
// test hit function global kernel to call hit
__global__ void hit_test_runner(int* return_value, float3* ray_origin, float3* ray_direction, float3* edge1, float3* edge2, float3* triangle_origin) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("tid %d\n", tid);
    if (tid < 5){
        return_value[tid] = hit(ray_direction[tid], 1000, ray_origin[tid], edge1[tid], edge2[tid], triangle_origin[tid]);
        printf("return_value %d\n", return_value[tid]);

    }
}

py::array_t<int> hit_test(py::array_t<float> ray_direction, py::array_t<float> ray_origin, py::array_t<float> triangle_origin, py::array_t<float> edge1, py::array_t<float> edge2) {
    py::buffer_info ray_direction_info = ray_direction.request();
    py::buffer_info ray_origin_info = ray_origin.request();
    py::buffer_info triangle_origin_info = triangle_origin.request();
    py::buffer_info edge1_info = edge1.request();
    py::buffer_info edge2_info = edge2.request();


    float* ray_direction_data = static_cast<float *>(ray_direction_info.ptr);
    float* ray_origin_data = static_cast<float *>(ray_origin_info.ptr);
    float* triangle_origin_data = static_cast<float *>(triangle_origin_info.ptr);
    float* edge1_data = static_cast<float *>(edge1_info.ptr);
    float* edge2_data = static_cast<float *>(edge2_info.ptr);

    //create py::array_t<int> to return
    py::array_t<int> return_value = py::array_t<int>(5);
    py::buffer_info return_value_info = return_value.request();
    int* return_value_data = static_cast<int *>(return_value_info.ptr);
    for (int i = 0; i < 5; i++) {
        return_value_data[i] = -2;
    }

    float3* device_ray_direction;
    float3* device_ray_origin;
    float3* device_triangle_origin;
    float3* device_edge1;
    float3* device_edge2;
    int* device_return_value;

    cudaMalloc((void**)&device_ray_direction, 5 * sizeof(float3));
    cudaMalloc((void**)&device_ray_origin, 5 * sizeof(float3));
    cudaMalloc((void**)&device_triangle_origin, 5 * sizeof(float3));
    cudaMalloc((void**)&device_edge1, 5 * sizeof(float3));
    cudaMalloc((void**)&device_edge2, 5 * sizeof(float3));
    cudaMalloc((void**)&device_return_value, 5 * sizeof(int));
    gpuErrchk( cudaGetLastError() );


    cudaMemcpy(device_ray_direction, ray_direction_data, 5 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ray_origin, ray_origin_data, 5 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_triangle_origin, triangle_origin_data, 5 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_edge1, edge1_data, 5 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_edge2, edge2_data, 5 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_return_value, return_value_data, 5 * sizeof(int), cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );


    hit_test_runner<<<5, 5,5>>>(device_return_value, 
        device_ray_origin,
         device_ray_direction, 
         device_edge1,
          device_edge2, 
          device_triangle_origin);
    gpuErrchk( cudaGetLastError() );
    cudaDeviceSynchronize();



    cudaMemcpy(return_value_data, device_return_value, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_ray_direction);
    cudaFree(device_ray_origin);
    cudaFree(device_triangle_origin);
    cudaFree(device_edge1);
    cudaFree(device_edge2);
    cudaFree(device_return_value);
    gpuErrchk( cudaGetLastError() );


    return return_value;
}





py::array_t<int> ray_cast_test(  py::array_t<int> triangle_index, py::array_t<float> triangle_points,
    py::array_t<float> source, py::array_t<float> end) {
    py::buffer_info triangle_index_info = triangle_index.request();
    py::buffer_info triangle_points_info = triangle_points.request();
    py::buffer_info source_info = source.request();
    py::buffer_info end_info = end.request();


    py::array_t<int> bool_list = py::array_t<int>(source_info.shape[0]*end_info.shape[0]*2);
    py::buffer_info bool_list_info = bool_list.request();

    // checks dimensions are N by 3
    if (triangle_index_info.shape[1] != 3 || triangle_points_info.shape[1] != 3 || source_info.shape[1] != 3 || end_info.shape[1] != 3) {
        throw std::runtime_error("1st dimension should be 3 for all arrays");
    }

    int* triangle_index_data = static_cast<int *>(triangle_index_info.ptr);
    float* triangle_points_data = static_cast<float *>(triangle_points_info.ptr);
    float* source_data = static_cast<float *>(source_info.ptr);
    float* end_data = static_cast<float *>(end_info.ptr);
    int* bool_list_data = static_cast<int *>(bool_list_info.ptr);

    float4* device_ray_direction;
    int3* device_triangle_index;
    float3* device_triangle_points;
    float3* device_source;
    float3* device_end;
    int2* device_bool_list;
    int ray_num = source_info.shape[0] * end_info.shape[0];

    cudaMalloc((void**)&device_ray_direction,  ray_num * 4 * sizeof(float));
    cudaMalloc((void**)&device_triangle_index, triangle_index_info.shape[0] * 3 * sizeof(int));
    cudaMalloc((void**)&device_triangle_points, triangle_points_info.shape[0] * 3 *sizeof(float));
    cudaMalloc((void**)&device_source, source_info.shape[0] * 3 * sizeof(float));
    cudaMalloc((void**)&device_end, end_info.shape[0] *3* sizeof(float));
    cudaMalloc((void**)&device_bool_list, bool_list_info.shape[0] * sizeof(int));
    gpuErrchk( cudaGetLastError() );

    cudaMemcpy(device_triangle_index, triangle_index_data, triangle_index_info.shape[0] * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_triangle_points, triangle_points_data, triangle_points_info.shape[0] * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_source, source_data, source_info.shape[0] * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_end, end_data, end_info.shape[0] * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_bool_list, 0, ray_num * sizeof(int2));
    gpuErrchk( cudaGetLastError() );

    raycast<<<32, 256,1>>>(device_source, 
        device_end, device_ray_direction,
         source_info.shape[0],
          end_info.shape[0],
           1, device_triangle_points, 
           device_triangle_index, 
           triangle_index_info.shape[0],
             ray_num, 
             device_bool_list,0,source_info.shape[0]);
    gpuErrchk( cudaGetLastError() );

    cudaMemcpy(bool_list_data, device_bool_list, bool_list_info.shape[0] * sizeof(int2), cudaMemcpyDeviceToHost);
    cudaFree(device_ray_direction);
    cudaFree(device_triangle_index);
    cudaFree(device_triangle_points);
    cudaFree(device_source);
    cudaFree(device_end);
    cudaFree(device_bool_list);
    gpuErrchk( cudaGetLastError() );

    return bool_list;
}
// write a test function to test these 2 functions




PYBIND11_MODULE(ray_tests, m) {
    m.def("ray_cast_test", &ray_cast_test, "A function to test the ray_cast function");
    m.def("hit_test", &hit_test, "A function to test the hit function");
    m.def("em_test", &em_test, "A function to test the frequency_wrapper function");


}




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
# include <complex>
#include "em.cuh"
#include "raycasting.cuh"
#include "gpu_error.cuh"
#include "raycasting_accelerated.cuh"
#include "terrain_acceleration_build.cuh"
#include "raycasting_brute_force.cuh"
#include <utility>
#include <tuple>
#include <chrono>   

namespace py = pybind11;
long calculate_memory_requirement_tiles(int source_size, int end_size, int tri_vertex_size, int binned_tri_num, int n_cellsx, int n_cellsy){
    long memory_requirement = 0;
    memory_requirement += source_size * 3 * sizeof(float);
    memory_requirement += end_size * 3 * sizeof(float);
    memory_requirement += tri_vertex_size * 3 * sizeof(float);
    memory_requirement += binned_tri_num * 3 * sizeof(int);
    memory_requirement += n_cellsx * n_cellsy * 2 * sizeof(int);
    memory_requirement += source_size * end_size * (sizeof(complex_float3)+sizeof(float4)+sizeof(int2));
    return memory_requirement;
}
long calculate_memory_requirement_brute_force(int source_size, int end_size, int tri_vertex_size, int triangle_size){
    long memory_requirement = 0;
    memory_requirement += source_size * 3 * sizeof(float);
    memory_requirement += end_size * 3 * sizeof(float);
    memory_requirement += tri_vertex_size * 3 * sizeof(float);
    memory_requirement += triangle_size * 3 * sizeof(int);
    memory_requirement += source_size * end_size * (sizeof(complex_float3)+sizeof(float4)+sizeof(int2));
    return memory_requirement;
}
py::array_t<std::complex<float>> calculate_scattering_tiles(py::array_t<float> source, py::array_t<float> end, py::array_t<float> triangle_vertex, 
                                                        float wave_length,
                                                        py::array_t<float> ex_real, py::array_t<float> ex_imag, py::array_t<float> ey_real, py::array_t<float> ey_imag, py::array_t<float> ez_real, py::array_t<float> ez_imag, 
                                                        py::array_t<float> normal, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float tile_size, py::array_t<int> bin_count, py::array_t<int> bin_triangles, int n_cellsx, int n_cellsy, int binned_tri_num,float alpha, float beta, bool not_self_to_self){

    // validate data about input arrays
    if (source.ndim() != 2 || source.shape(1) != 3){throw std::runtime_error("source must be a 2D array with 3 columns");}
    if (end.ndim() != 2 || end.shape(1) != 3){throw std::runtime_error("end must be a 2D array with 3 columns");}
    if (triangle_vertex.ndim() != 2 || triangle_vertex.shape(1) != 3){throw std::runtime_error("triangle_vertex must be a 2D array with 3 columns");}
    if (ex_real.ndim() != 1){throw std::runtime_error("ex_real must be a 1D array");}
    if (ex_imag.ndim() != 1){throw std::runtime_error("ex_imag must be a 1D array");}
    if (ey_real.ndim() != 1){throw std::runtime_error("ey_real must be a 1D array");}
    if (ey_imag.ndim() != 1){throw std::runtime_error("ey_imag must be a 1D array");}
    if (ez_real.ndim() != 1){throw std::runtime_error("ez_real must be a 1D array");}
    if (ez_imag.ndim() != 1){throw std::runtime_error("ez_imag must be a 1D array");}
    if (normal.ndim() != 2 || normal.shape(1) != 3){throw std::runtime_error("normal must be a 2D array with 3 columns");}
    if (bin_count.ndim() != 1){throw std::runtime_error("bin_count must be a 1D array");}
    if (bin_triangles.ndim() != 1){throw std::runtime_error("bin_triangles must be a 1D array");}
    if (bin_count.shape(0) != n_cellsx * n_cellsy){throw std::runtime_error("bin_count must have n_cellsx * n_cellsy elements");}
    if (bin_triangles.shape(0) != binned_tri_num*3 ){throw std::runtime_error("bin_triangles 3* bintrinum elements");}
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free*.95 < calculate_memory_requirement_tiles(source.shape(0), end.shape(0), triangle_vertex.shape(0), binned_tri_num, n_cellsx, n_cellsy)){
        throw std::runtime_error("Not enough memory on the GPU- increase chunk number");
    }

    //get the size of the input arrays
    int source_size = source.shape(0);
    int end_size = end.shape(0);
    int tri_vertex_size = triangle_vertex.shape(0);

    int points_size = source_size + end_size;


    if (ex_real.shape(0) != source_size) {
        throw std::runtime_error("Mismatch: ex_real.shape(0) != source_size");
    } 
    if (ex_imag.shape(0) != source_size) {
        throw std::runtime_error("Mismatch: ex_imag.shape(0) != source_size");
    } 
    if (ey_real.shape(0) != source_size) {
        throw std::runtime_error("Mismatch: ey_real.shape(0) != source_size");
    } 
    if (ey_imag.shape(0) != source_size) {
        throw std::runtime_error("Mismatch: ey_imag.shape(0) != source_size");
    } 
    if (ez_real.shape(0) != source_size) {
        throw std::runtime_error("Mismatch: ez_real.shape(0) != source_size");
    } 
    if (ez_imag.shape(0) != source_size) {
        throw std::runtime_error("Mismatch: ez_imag.shape(0) != source_size");
    } 
    if (normal.shape(0) != points_size) {
        throw std::runtime_error("Mismatch: normal.shape(0) != points_size");
    }


    //get the pointers to the data
    float* source_ptr = (float*) source.request().ptr;
    float* end_ptr = (float*) end.request().ptr;
    float* triangle_vertex_ptr = (float*) triangle_vertex.request().ptr;
    float* ex_real_ptr = (float*) ex_real.request().ptr;
    float* ex_imag_ptr = (float*) ex_imag.request().ptr;
    float* ey_real_ptr = (float*) ey_real.request().ptr;
    float* ey_imag_ptr = (float*) ey_imag.request().ptr;
    float* ez_real_ptr = (float*) ez_real.request().ptr;
    float* ez_imag_ptr = (float*) ez_imag.request().ptr;
    float* normal_ptr = (float*) normal.request().ptr;
    int* bin_count_ptr = (int*) bin_count.request().ptr;
    int* bin_triangles_ptr = (int*) bin_triangles.request().ptr;

    int3* d_binned_triangles;
    int2* d_bin_count;
    float3* d_tri_vertex;
    std::vector<int2> binned_num(n_cellsx * n_cellsy);
    int sum = 0;
    for (int i = 0; i < n_cellsx * n_cellsy; i++){
        binned_num[i] = make_int2( bin_count_ptr[i],sum);
        sum += bin_count_ptr[i];    
    }

    int2* bin_count_ptr2 = binned_num.data();
    //print free toral memory



    cudaMalloc(&d_binned_triangles, binned_tri_num * sizeof(int3));
    cudaMalloc(&d_bin_count, n_cellsx * n_cellsy * sizeof(int2));
    cudaMalloc(&d_tri_vertex, tri_vertex_size * sizeof(float3));
    //gpuerror check
    cudaMemGetInfo(&free, &total);
    gpuErrchk( cudaGetLastError() );


    cudaMemcpy(d_binned_triangles, bin_triangles_ptr, binned_tri_num * sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_count, bin_count_ptr2, n_cellsx * n_cellsy * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tri_vertex, triangle_vertex_ptr, tri_vertex_size * sizeof(float3), cudaMemcpyHostToDevice);
    //gpuerror check
    gpuErrchk( cudaGetLastError() );

    std::vector<PointData> points_vec(points_size);
    PointData* points = points_vec.data();


    for (int i = 0; i < source_size; i++){
        points_vec[i] = create_point_data(ex_real_ptr[i], ex_imag_ptr[i], ey_real_ptr[i], ey_imag_ptr[i], ez_real_ptr[i], ez_imag_ptr[i],normal_ptr[i*3], normal_ptr[i*3+1], normal_ptr[i*3+2]);
    }
    for (int i = 0; i < end_size; i++){

        points_vec[i+source_size] = create_point_data(0, 0, 0, 0, 0, 0,normal_ptr[(i+source_size)*3], normal_ptr[(i+source_size)*3+1], normal_ptr[(i+source_size)*3+2]);
    }

    //pointer to the points
    // declare numpy complex array
    py::array_t<std::complex<float>> scattering_network_py = py::array_t<std::complex<float>>(source_size * end_size *3);
    std::complex<float>* scattering_network_py_ptr = (std::complex<float>*) scattering_network_py.request().ptr;
    int source_index = 0;
    std::vector<complex_float3> scattering_network(source_size* end_size);
    complex_float3* scattering_network_ptr = scattering_network.data();

    raycast_wrapper_tiles(&source_ptr[source_index], end_ptr, source_size, end_size,d_tri_vertex, d_binned_triangles, d_bin_count
            ,make_int2(n_cellsx,n_cellsy) , make_float2(xmin,xmax), make_float2(ymin,tile_size), make_float2(zmin,tile_size),points, wave_length,scattering_network_ptr,make_float2(alpha,beta),not_self_to_self);
    for (int i = 0; i < scattering_network.size() ; i++)
    {
        scattering_network_py_ptr[i*3+0] = std::complex<float>(scattering_network_ptr[i].x.x, scattering_network_ptr[i].x.y);
        scattering_network_py_ptr[i*3+1] = std::complex<float>(scattering_network_ptr[i].y.x, scattering_network_ptr[i].y.y);
        scattering_network_py_ptr[i*3+2] = std::complex<float>(scattering_network_ptr[i].z.x, scattering_network_ptr[i].z.y);
    }
    cudaFree(d_binned_triangles);
    cudaFree(d_bin_count);
    cudaFree(d_tri_vertex);
    return scattering_network_py;
}


py::array_t<std::complex<float>> calculate_scattering_brute_force(py::array_t<float> source, py::array_t<float> end, 
    py::array_t<int> triangles, py::array_t<float> triangle_vertex, 
    float wave_length,
    py::array_t<float> ex_real, py::array_t<float> ex_imag, py::array_t<float> ey_real, py::array_t<float> ey_imag, py::array_t<float> ez_real,
     py::array_t<float> ez_imag, 
    py::array_t<float> normal,float alpha, float beta, bool not_self_to_self)
{

    // validate data about input arrays
    if (source.ndim() != 2 || source.shape(1) != 3){throw std::runtime_error("source must be a 2D array with 3 columns");}
    if (end.ndim() != 2 || end.shape(1) != 3){throw std::runtime_error("end must be a 2D array with 3 columns");}
    if (triangles.ndim() != 2 || triangles.shape(1) != 3){throw std::runtime_error("triangles must be a 2D array with 3 columns");}
    if (triangle_vertex.ndim() != 2 || triangle_vertex.shape(1) != 3){throw std::runtime_error("triangle_vertex must be a 2D array with 3 columns");}
    if (ex_real.ndim() != 1){throw std::runtime_error("ex_real must be a 1D array");}
    if (ex_imag.ndim() != 1){throw std::runtime_error("ex_imag must be a 1D array");}
    if (ey_real.ndim() != 1){throw std::runtime_error("ey_real must be a 1D array");}
    if (ey_imag.ndim() != 1){throw std::runtime_error("ey_imag must be a 1D array");}
    if (ez_real.ndim() != 1){throw std::runtime_error("ez_real must be a 1D array");}
    if (ez_imag.ndim() != 1){throw std::runtime_error("ez_imag must be a 1D array");}
    if (normal.ndim() != 2 || normal.shape(1) != 3){throw std::runtime_error("normal must be a 2D array with 3 columns");}
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free*.95 < calculate_memory_requirement_brute_force(source.shape(0), end.shape(0), triangle_vertex.shape(0), triangles.shape(0))){
        throw std::runtime_error("Not enough memory on the GPU- increase chunk number");
    }

    //get the size of the input arrays
    int source_size = source.shape(0);
    int end_size = end.shape(0);
    int triangle_size = triangles.shape(0);
    int tri_vertex_size = triangle_vertex.shape(0);

    int points_size = source_size + end_size;


    if (ex_real.shape(0) != source_size) {
    throw std::runtime_error("Mismatch: ex_real.shape(0) != source_size");
    } 
    if (ex_imag.shape(0) != source_size) {
    throw std::runtime_error("Mismatch: ex_imag.shape(0) != source_size");
    } 
    if (ey_real.shape(0) != source_size) {
    throw std::runtime_error("Mismatch: ey_real.shape(0) != source_size");
    } 
    if (ey_imag.shape(0) != source_size) {
    throw std::runtime_error("Mismatch: ey_imag.shape(0) != source_size");
    } 
    if (ez_real.shape(0) != source_size) {
    throw std::runtime_error("Mismatch: ez_real.shape(0) != source_size");
    } 
    if (ez_imag.shape(0) != source_size) {
    throw std::runtime_error("Mismatch: ez_imag.shape(0) != source_size");
    } 
    if (normal.shape(0) != points_size) {
    throw std::runtime_error("Mismatch: normal.shape(0) != points_size");
    }

    //get the pointers to the data
    float* source_ptr = (float*) source.request().ptr;
    float* end_ptr = (float*) end.request().ptr;
    int* triangles_ptr = (int*) triangles.request().ptr;
    float* triangle_vertex_ptr = (float*) triangle_vertex.request().ptr;
    float* ex_real_ptr = (float*) ex_real.request().ptr;
    float* ex_imag_ptr = (float*) ex_imag.request().ptr;
    float* ey_real_ptr = (float*) ey_real.request().ptr;
    float* ey_imag_ptr = (float*) ey_imag.request().ptr;
    float* ez_real_ptr = (float*) ez_real.request().ptr;
    float* ez_imag_ptr = (float*) ez_imag.request().ptr;
    float* normal_ptr = (float*) normal.request().ptr;

    int3* d_triangles;
    float3* d_tri_vertex;

    //print free toral memory



    cudaMalloc(&d_triangles, triangle_size * sizeof(int3));
    cudaMalloc(&d_tri_vertex, tri_vertex_size * sizeof(float3));
    //gpuerror check
    gpuErrchk( cudaGetLastError() );

    cudaMemcpy(d_triangles, triangles_ptr,triangle_size * sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tri_vertex, triangle_vertex_ptr, tri_vertex_size * sizeof(float3), cudaMemcpyHostToDevice);
    //gpuerror check
    gpuErrchk( cudaGetLastError() );
    std::vector<PointData> points_vec(points_size);
    PointData* points = points_vec.data();


    for (int i = 0; i < source_size; i++){
    points_vec[i] = create_point_data(ex_real_ptr[i], ex_imag_ptr[i], ey_real_ptr[i], ey_imag_ptr[i], ez_real_ptr[i], ez_imag_ptr[i],normal_ptr[i*3], normal_ptr[i*3+1], normal_ptr[i*3+2]);
    }
    for (int i = 0; i < end_size; i++){
    points_vec[i+source_size] = create_point_data(0, 0, 0, 0, 0, 0,normal_ptr[(i+source_size)*3], normal_ptr[(i+source_size)*3+1], normal_ptr[(i+source_size)*3+2]);
    }
    py::array_t<std::complex<float>> scattering_network_py = py::array_t<std::complex<float>>(source_size * end_size *3);
    std::complex<float>* scattering_network_py_ptr = (std::complex<float>*) scattering_network_py.request().ptr;
    int source_index = 0;
    std::vector<complex_float3> scattering_network(source_size* end_size);
    complex_float3* scattering_network_ptr = scattering_network.data();

    raycast_wrapper_brute_force(&source_ptr[source_index], end_ptr, source_size, end_size,d_tri_vertex, d_triangles, triangle_size
    ,points, wave_length,scattering_network_ptr,make_float2(alpha,beta),not_self_to_self);

    for (int i = 0; i < scattering_network.size() ; i++)
    {
    scattering_network_py_ptr[i*3+0] = std::complex<float>(scattering_network_ptr[i].x.x, scattering_network_ptr[i].x.y);
    scattering_network_py_ptr[i*3+1] = std::complex<float>(scattering_network_ptr[i].y.x, scattering_network_ptr[i].y.y);
    scattering_network_py_ptr[i*3+2] = std::complex<float>(scattering_network_ptr[i].z.x, scattering_network_ptr[i].z.y);
    std::cout << scattering_network_py_ptr[i*3+0] << " " << scattering_network_py_ptr[i*3+1] << " " << scattering_network_py_ptr[i*3+2] << std::endl;
    }
    cudaFree(d_triangles);
    cudaFree(d_tri_vertex);
    return scattering_network_py;
}


PYBIND11_MODULE(em, m) {
    m.def("calculate_scattering_tiles", &calculate_scattering_tiles, "Calculate scattering with tile based raycaster");
    m.def("calculate_scattering_brute_force", &calculate_scattering_brute_force, "Calculate scattering with brute force raycaster");
    m.def("bin_counts_to_numpy", &bin_counts_to_numpy);
    m.def("bin_triangles_to_numpy", &bin_triangles_to_numpy);

}

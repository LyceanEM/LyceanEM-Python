#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../source/math_utility.cuh"
#include "raycasting.cuh"

#include "vector_types.h"

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <complex>
#include <cmath>
#include <stdexcept>
#include "cuComplex.h"
void validate_point_data(py::array_t<float> ex_real, py::array_t<float> ex_imag, py::array_t<float> ey_real, py::array_t<float> ey_imag, py::array_t<float> ez_real, py::array_t<float> ez_imag, 
    py::array_t<bool> is_electric, py::array_t<float> permittivity_real, py::array_t<float> permittivity_imag, py::array_t<float> permeability_real, py::array_t<float> permeability_imag,
    py::array_t<float> normal, int points_size)
{    if (ex_real.ndim() != 1){throw std::runtime_error("ex_real must be a 1D array");}
    if (ex_imag.ndim() != 1){throw std::runtime_error("ex_imag must be a 1D array");}
    if (ey_real.ndim() != 1){throw std::runtime_error("ey_real must be a 1D array");}
    if (ey_imag.ndim() != 1){throw std::runtime_error("ey_imag must be a 1D array");}
    if (ez_real.ndim() != 1){throw std::runtime_error("ez_real must be a 1D array");}
    if (ez_imag.ndim() != 1){throw std::runtime_error("ez_imag must be a 1D array");}
    if (is_electric.ndim() != 1){throw std::runtime_error("is_electric must be a 1D array");}
    if (permittivity_real.ndim() != 1){throw std::runtime_error("permittivity_real must be a 1D array");}
    if (permittivity_imag.ndim() != 1){throw std::runtime_error("permittivity_imag must be a 1D array");}
    if (permeability_real.ndim() != 1){throw std::runtime_error("permeability_real must be a 1D array");}
    if (permeability_imag.ndim() != 1){throw std::runtime_error("permeability_imag must be a 1D array");}
    if (normal.ndim() != 2 || normal.shape(1) != 3){throw std::runtime_error("normal must be a 2D array with 3 columns");}


    if (ex_real.shape(0) != ex_imag.shape(0) ||
    ex_real.shape(0) != ey_real.shape(0) ||
    ex_real.shape(0) != ey_imag.shape(0) ||
    ex_real.shape(0) != ez_real.shape(0) ||
    ex_real.shape(0) != ez_imag.shape(0) ||
    ex_real.shape(0) != is_electric.shape(0) ||
    ex_real.shape(0) != permittivity_real.shape(0) ||
    ex_real.shape(0) != permittivity_imag.shape(0) ||
    ex_real.shape(0) != permeability_real.shape(0) ||
    ex_real.shape(0) != permeability_imag.shape(0) ||
    ex_real.shape(0) != normal.shape(0)||
    ex_real.shape(0) != points_size)
    {
        throw std::runtime_error("All point data arrays must have the same size = source_size + end_size + scatter_size");
    }
}

void em_wave_test(py::array_t<float> source, py::array_t<float> end, py::array_t<float> scatter, py::array_t<int> index, py::array_t<float> ex_real, py::array_t<float> ex_imag, py::array_t<float> ey_real,
     py::array_t<float> ey_imag, py::array_t<float> ez_real, py::array_t<float> ez_imag, 
    py::array_t<bool> is_electric, py::array_t<float> permittivity_real, py::array_t<float> permittivity_imag, py::array_t<float> permeability_real, py::array_t<float> permeability_imag,
    py::array_t<float> normal, float wave_vector)

{
    validate_point_data(ex_real, ex_imag, ey_real, ey_imag, ez_real, ez_imag, is_electric, permittivity_real, permittivity_imag, permeability_real, permeability_imag, normal, source.shape(0) + end.shape(0) + scatter.shape(0));

    //validate points must be 2 by 3
    if (source.ndim() != 2 || source.shape(1) != 3 || source.shape(0) !=2){throw std::runtime_error("source must be a 2D array with 3 columns and 2 rows");}
    if (end.ndim() != 2 || end.shape(1) != 3 || end.shape(0) !=2){throw std::runtime_error("end must be a 2D array with 3 columns and 2 rows");}
    if (scatter.ndim() != 2 || scatter.shape(1) != 3 || scatter.shape(0) !=2){throw std::runtime_error("scatter must be a 2D array with 3 columns and 2 rows");}
    if (index.ndim() != 2 || index.shape(1) != 2 || index.shape(0) !=3){throw std::runtime_error("index must be a 2D array with 3 columns and 2 rows");}


    float* source_ptr = (float*) source.request().ptr;
    float* end_ptr = (float*) end.request().ptr;
    float* scatter_ptr = (float*) scatter.request().ptr;
    float* ex_real_ptr = (float*) ex_real.request().ptr;
    float* ex_imag_ptr = (float*) ex_imag.request().ptr;
    float* ey_real_ptr = (float*) ey_real.request().ptr;
    float* ey_imag_ptr = (float*) ey_imag.request().ptr;
    float* ez_real_ptr = (float*) ez_real.request().ptr;
    float* ez_imag_ptr = (float*) ez_imag.request().ptr;
    bool* is_electric_ptr = (bool*) is_electric.request().ptr;
    float* permittivity_real_ptr = (float*) permittivity_real.request().ptr;
    float* permittivity_imag_ptr = (float*) permittivity_imag.request().ptr;
    float* permeability_real_ptr = (float*) permeability_real.request().ptr;
    float* permeability_imag_ptr = (float*) permeability_imag.request().ptr;
    float* normal_ptr = (float*) normal.request().ptr;

    PointData points[points_size];

    for (int i = 0; i < points_size; i++){
        points[i] = create_point_data(ex_real_ptr[i], ex_imag_ptr[i], ey_real_ptr[i], ey_imag_ptr[i], ez_real_ptr[i], ez_imag_ptr[i], is_electric_ptr[i], permittivity_real_ptr[i], permittivity_imag_ptr[i], permeability_real_ptr[i], permeability_imag_ptr[i], normal_ptr[i*3], normal_ptr[i*3+1], normal_ptr[i*3+2]);
    }


    float3 source = make_float3(source_ptr[0], source_ptr[1], source_ptr[2]);
    float3 end = make_float3(end_ptr[0], end_ptr[1], end_ptr[2]);
    float3 scatter = make_float3(scatter_ptr[0], scatter_ptr[1], scatter_ptr[2]);

    float4 ray[3];
    float3 direction = normalize(end-source);
    ray[0] = make_float4(direction.x, direction.y, direction.z, length(end-source));
    direction = normalize(scatter-source);
    ray[1] = make_float4(direction.x, direction.y, direction.z, length(scatter-source));
    direction = normalize(end-scatter);
    ray[2] = make_float4(direction.x, direction.y, direction.z, length(end-scatter));

    int2 indexs[3];
    indexs[0] = make_int2(index_ptr[0], index_ptr[1]);
    indexs[1] = make_int2(index_ptr[1], index_ptr[2]);
    indexs[2] = make_int2(index_ptr[0], index_ptr[2]);

    // decalre d_ray and d_ray_index
    float4* d_ray;
    int2* d_ray_index;

    complex_float3 scattering_network[2];

    // calculate size of arrays
    int ray_size = 3 * sizeof(float4);
    int ray_index_size = 3 * sizeof(int2);

    // allocate device memory
    cudaMalloc((void**)&d_ray, ray_size);
    cudaMalloc((void**)&d_ray_index, ray_index_size);

    // copy data to device
    cudaMemcpy(d_ray, ray, ray_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_index, indexs, ray_index_size, cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );

    // call the function

    frequency_wrapper(1, d_ray, points, d_ray_index, wave_vector, 1, 1, 1, scattering_network);
    gpuErrchk( cudaGetLastError() );
    cudaFree(d_ray);
    cudaFree(d_ray_index);




    // copy data from scattering_network to numpy array
    py::array_t<std::complex<float>> scattering_network_np = py::array_t<std::complex<float>>(3)
    py::buffer_info scattering_network_info = scattering_network_np.request();
    std::complex<float>* scattering_network_data = static_cast<std::complex<float>*>(scattering_network_info.ptr);
    for (int i = 0; i < 1; i++){
        scattering_network_data[i] = std::complex<float>(scattering_network[i].x);
        scattering_network_data[i+1] = std::complex<float>(scattering_network[i].y);
        scattering_network_data[i+2] = std::complex<float>(scattering_network[i].z);
    }
    return scattering_network_np;
}

m.def("em_wave_test", &em_wave_test, "A function to test the frequency_wrapper function");



       





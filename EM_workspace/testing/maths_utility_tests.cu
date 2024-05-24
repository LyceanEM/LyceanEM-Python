//define a kernel that takes 3 3by N arrays and converts them to origin and direction vectors between points and allocates the rays onto shared memory
#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"


#include <iostream>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace py = pybind11;

/* Test for math_utility kernels*/
/*testing of functions*/

// boiler plate for all tests that calls kernel to be tested and checks appropriate input and output sizes
template <typename F, int size_input, typename T, typename RT, int size_return>
void test(py::array_t<float> a, py::array_t<float> b,py::array_t<float>  c, F function) {
    
    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();
    py::buffer_info c_info = c.request();
    if (a_info.shape[0] != size_input || b_info.shape[0] != size_input || c_info.shape[0] != size_return) {
        throw std::runtime_error("Input arrays must have same size"+ std::to_string(a_info.shape[0]) + " " + std::to_string(b_info.shape[0]) + " " + std::to_string(c_info.shape[0]) + " " + std::to_string(size_input) + " " + std::to_string(size_return));
    }

    float* a_data = static_cast<float *>(a_info.ptr);
    float* b_data = static_cast<float *>(b_info.ptr);
    float* c_data = static_cast<float *>(c_info.ptr);
    

    T* device_a;
    T* device_b;
    RT* device_c;

    cudaMalloc((void**)&device_a, 1 * sizeof(T));
    cudaMalloc((void**)&device_b, 1* sizeof(T));
    cudaMalloc((void**)&device_c,  1* sizeof(RT));
    cudaMemcpy(device_a, a_data, 1 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b_data, 1 * sizeof(T), cudaMemcpyHostToDevice);
    function<<<1,1,1>>>(device_a, device_b, device_c);


    cudaMemcpy(c_data, device_c, 1 * sizeof(RT), cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

}





/* Switch to function that takes function pointers and 2 arrays as input and returns an array */




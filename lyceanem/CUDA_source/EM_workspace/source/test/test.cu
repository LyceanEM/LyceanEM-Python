#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <iostream>

namespace py = pybind11;

// CUDA kernel
__global__ void test_assert() {
    printf("hi from assert\n");
    bool assert_works = false;
    assert(assert_works);
}

// Wrapper function to launch CUDA kernel
void launch_kernel() {
    test_assert<<<1, 1>>>();
    cudaDeviceSynchronize();
}

// Binding code
PYBIND11_MODULE(cuda_module, m) {
    m.def("launch_kernel", &launch_kernel, "Launch CUDA kernel");
}

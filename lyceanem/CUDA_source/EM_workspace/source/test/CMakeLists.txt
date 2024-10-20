cmake_minimum_required(VERSION 3.10)
set (PYBIND11_PYTHON_VERSION "3.10.9") 
project(test LANGUAGES CXX CUDA)
find_package( PkgConfig )


# Find CUDA
find_package(CUDAToolkit REQUIRED)
# Include Pybind11
find_package(pybind11 REQUIRED)
# Add executable
pybind11_add_module(cuda_module  ~/lycean_em/LyceanEM_CUDA_MVP/source/frequency_domain_models.cu)

# Link CUDA
target_link_libraries(cuda_module PRIVATE ${CUDA_LIBRARIES})

# Include CUDA headers
target_include_directories(cuda_module PRIVATE ${CUDA_INCLUDE_DIRS})


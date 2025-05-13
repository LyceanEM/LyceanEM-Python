cmake_minimum_required(VERSION 3.22)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python_version: ${Python_VERSION}")
find_package(pybind11 CONFIG REQUIRED)
project(test LANGUAGES CXX CUDA)
## set the generator
set(CMAKE_GENERATOR "Ninja")


find_package( PkgConfig )
# build release with debug info


# Find CUDA
find_package(CUDAToolkit REQUIRED)
# print done
# Include Pybind11
# Add executable
pybind11_add_module(em lyceanem/src/frequency_domain_models.cu)
## all architecture abobe and including turing

set_property(TARGET em PROPERTY CUDA_ARCHITECTURES 75)


# Link CUDA
target_link_libraries(em PRIVATE ${CUDA_LIBRARIES})

# Include CUDA headers
target_include_directories(em PRIVATE ${CUDA_INCLUDE_DIRS})

#install
install(TARGETS em DESTINATION .)
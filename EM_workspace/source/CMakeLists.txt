cmake_minimum_required(VERSION 3.22)
set (PYBIND11_PYTHON_VERSION "3.10.9") 
#find_package(CUDA REQUIRED)

project(frequency_domain LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

# set  -std=c++20
set(CMAKE_CXX_STANDARD 23)

#set(CMAKE_CXX_COMPILER /usr/bin/g++)
#set(CMAKE_FIND_DEBUG_MODE 1)

find_package(pybind11  REQUIRED)

find_package( PkgConfig )
include_directories( ${EIGEN3_INCLUDE_DIRS} )
 
# Set the source file for the C++ extension module
set(EMSOURCE  ~/lycean_em/LyceanEM_CUDA_MVP/source/frequency_domain_models.cu)



# Add the C++ extension module


pybind11_add_module(em ${EMSOURCE})

target_include_directories(em PRIVATE ./)
target_compile_options(em PRIVATE --expt-relaxed-constexpr -arch=sm_75)
target_compile_options(em PRIVATE -lineinfo)

## set target output to be current conda env
set_target_properties(em PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CONDA_PREFIX}/lib/python3.10/site-packages)

set_target_properties(em PROPERTIES LIBRARY_OUTPUT_DIRECTORY ~/lycean_em/LyceanEM_CUDA_MVP/testing)









cmake_minimum_required(VERSION 3.22)
message(STATUS "CMake version: ${CMAKE_VERSION}")
set (PYBIND11_PYTHON_VERSION "3.10.9") 
#find_package(CUDA REQUIRED)

project(frequency_domain LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA: ${CUDA_VERSION_STRING}")

# set  -std=c++20
set(CMAKE_CXX_STANDARD 23)

#set(CMAKE_CXX_COMPILER /usr/bin/g++)
#set(CMAKE_FIND_DEBUG_MODE 1)

find_package(pybind11  REQUIRED)

find_package( PkgConfig )
 
# Set the source file for the C++ extension module
set(EMSOURCE  ~/lycean_em/LyceanEM_CUDA_MVP/source/frequency_domain_models.cu)



# Add the C++ extension module


pybind11_add_module(em ${EMSOURCE})



target_include_directories(em PRIVATE ~/lycean_em/LyceanEM_CUDA_MVP/source)
target_compile_options(em PRIVATE --expt-relaxed-constexpr -arch=sm_75)
target_compile_options(em PRIVATE -lineinfo)

install(TARGETS em DESTINATION lyceanem)



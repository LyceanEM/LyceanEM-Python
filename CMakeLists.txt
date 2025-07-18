cmake_minimum_required(VERSION 3.23)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python_version: ${Python_VERSION}")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
find_package(pybind11 CONFIG REQUIRED)
project(test LANGUAGES CXX CUDA)

# Attempt to locate the specified Python library file
if(EXISTS "${PYTHON_LIBRARY}")
  message(STATUS "Found Python library: ${PYTHON_LIBRARY}")
else()
  message(FATAL_ERROR "Python library not found at: ${PYTHON_LIBRARY}")
endif()

find_package( PkgConfig )
# build release with debug info


# Find CUDA
find_package(CUDAToolkit REQUIRED)
# print done
# Include Pybind11
# Add executable
pybind11_add_module(em lyceanem/src/frequency_domain_models.cu)
## all architecture abobe and including turing

set_property(TARGET em PROPERTY CUDA_ARCHITECTURES all-major)
target_link_libraries(em PRIVATE ${Python_LIBRARIES} ${Python_INCLUDE_DIRS})

# Link CUDA
target_link_libraries(em PRIVATE ${CUDA_LIBRARIES})

# Include CUDA headers
target_include_directories(em PRIVATE ${CUDA_INCLUDE_DIRS})

#install
install(TARGETS em DESTINATION .)
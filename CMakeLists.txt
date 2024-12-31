cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
PROJECT(polar_coordinate_trans LANGUAGES CXX CUDA)


# required packages
FIND_PACKAGE(OpenCV REQUIRED)
ADD_DEFINITIONS(-std=c++11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(CUDA)
find_package(cuBLAS)

set(CUDA_SOURCE_FILES src/cuda_fun_api/polar2cart.cu)
cuda_add_library(cuda_function_api ${CUDA_SOURCE_FILES} STATIC)
target_link_libraries(cuda_function_api  ${OpenCV_LIBS} ${CUBLAS_LIBRARIES})


FILE(GLOB SOURCES src/main.cpp)

ADD_EXECUTABLE(main ${SOURCES})
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} cuda_function_api)
target_compile_options(main PRIVATE "/MT")

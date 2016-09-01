# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)
project(datamininglibs C CXX)

find_package(GTest)
find_package(CUDA REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_64_BIT_DEVICE_CODE ON)

add_definitions(-DUNICODE -D_UNICODE)
if(WIN32)
	add_definitions(-DWindowsBuild)
	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-gencode arch=compute_30,code=sm_30;-g;-G;-lineinfo)
	set(CUDA_NVCC_FLAGS_OPTIMIZED ${CUDA_NVCC_FLAGS_OPTIMIZED};-gencode arch=compute_30,code=sm_30;-O3)
else(WIN32)
	add_definitions(-DUnixBuild)
	set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )
	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-gencode arch=compute_20,code=sm_20;-g;-G;-lineinfo;-std=c++11)
	set(CUDA_NVCC_FLAGS_OPTIMIZED ${CUDA_NVCC_FLAGS_OPTIMIZED};-gencode arch=compute_20,code=sm_20;-O3;-std=c++11)
endif(WIN32)

set (LIBRARY_OUTPUT_PATH Build_Output/Libs CACHE PATH "Lib path")
set (EXECUTABLE_OUTPUT_PATH Build_Output/bin CACHE PATH "Exe path")
mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

add_subdirectory(./lib_core)
add_subdirectory(./lib_algorithms)
add_subdirectory(./lib_julia)
add_subdirectory(./lib_parsing)
add_subdirectory(./lib_models)
add_subdirectory(./lib_data)
add_subdirectory(./lib_gpu)

if(GTEST_FOUND)
	add_subdirectory(./runnable_test)
endif(GTEST_FOUND)

# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
	../lib_models/include
	../lib_data/include
	../lib_gpu/include
	../lib_core/include
	../lib_parsing/include
)

set(cpp_files
	./test/test_gpurf.h

	../source_shared/precomp.h
	../source_shared/include/global_defines.h

	./include/ml_algorithm_params.h
	./include/lib_algorithms.h
	./include/ml_algorithm.h
	./include/algorithms_interface.h

	./source/ml_algorithm_params_impl.h
	./source/gpu_algorithm.h
	./source/gpurf.h
	./source/gpudte.h


	./source/ml_algorithm_params_impl.cc
	./source/algorithms_interface.cc
	./source/ml_algorithm.cc
	../source_shared/precomp.cc
	./source/gpu_algorithm.cc
	./source/gpurf.cu
	./source/gpudte.cu
)

source_group(source FILES
	./source/ml_algorithm_params_impl.h
	./source/gpu_algorithm.h
	./source/gpurf.h
	./source/gpudte.h

	./source/ml_algorithm_params_impl.cc
	./source/gpu_algorithm.cc
	./source/gpurf.cu
	./source/gpudte.cu
)

source_group(interface FILES
	./include/ml_algorithm_params.h
	./include/lib_algorithms.h
	./include/ml_algorithm.h
	./include/algorithms_interface.h

	./source/algorithms_interface.cc
	./source/ml_algorithm.cc
)

source_group(test FILES
	./test/test_gpurf.h
)

source_group(precomp FILES
	../source_shared/include/global_defines.h
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

CUDA_ADD_LIBRARY(lib_algorithms SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_algorithms PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_algorithms
	${CUDA_cusparse_LIBRARY}
	${CUDA_curand_LIBRARY}
	${CUDA_LIBRARIES}

	lib_models
	lib_data
	lib_parsing
	lib_gpu
	lib_core
)

include_directories(
	${dir_list}
	${CUDA_INCLUDE_DIRS}
)

if (MSVC)
   set_target_properties(lib_algorithms PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)

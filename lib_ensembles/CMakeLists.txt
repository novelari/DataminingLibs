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
	../lib_algorithms/include
)

set(cpp_files
	./test/test_gpurf.h
	./test/test_gpuert.h
	./test/test_resources.h

	../source_shared/precomp.h
	./include/lib_ensembles.h
	./include/ensembles_interface.h	
	./source/gpurf.h
	./source/gpudte.h
	./source/gpuert.h
	./source/gpudte_algorithm.h
	./source/gpudte_algorithm_shared.h

	./source/gpurf.cu
	./source/gpudte.cu
	./source/gpuert.cu
	./source/gpudte_algorithm.cc
	./source/gpudte_algorithm_shared.cc
	../source_shared/precomp.cc
	./source/ensembles_interface.cc
)

source_group(source FILES
	./source/gpurf.h
	./source/gpudte.h
	./source/gpuert.h
	./source/gpudte_algorithm.h
	./source/gpudte_algorithm_shared.h

	./source/gpurf.cu
	./source/gpudte.cu
	./source/gpuert.cu
	./source/gpudte_algorithm.cc
	./source/gpudte_algorithm_shared.cc
)

source_group(interface FILES
	./include/lib_ensembles.h
	./include/ensembles_interface.h

	./source/ensembles_interface.cc
)

source_group(test FILES
	./test/test_gpurf.h
	./test/test_gpuert.h
	./test/test_resources.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

CUDA_ADD_LIBRARY(lib_ensembles SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_ensembles PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_ensembles 
	${CUDA_cusparse_LIBRARY}
	${CUDA_curand_LIBRARY}
	${CUDA_LIBRARIES}

	lib_models
	lib_data
	lib_parsing
	lib_gpu
	lib_core
	lib_algorithms
)

include_directories(lib_ensembles 
	${dir_list}
	${CUDA_INCLUDE_DIRS}
)

if (MSVC)
   set_target_properties(lib_ensembles PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)

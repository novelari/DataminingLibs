# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./
	../source_shared
	../source_shared/include
	../lib_algorithms/source
	../lib_julia/source
	../lib_parsing/source
	../lib_models/source
	../lib_data/source
	../lib_gpu/source
	../lib_core/source
	../lib_ensembles/source
	../lib_algorithms/test
	../lib_julia/test
	../lib_parsing/test 
	../lib_models/test
	../lib_data/test
	../lib_gpu/test
	../lib_core/test
	../lib_ensembles/test
	../lib_algorithms/include
	../lib_julia/include
	../lib_parsing/include
	../lib_models/include
	../lib_data/include
	../lib_gpu/include
	../lib_core/include
	../lib_ensembles/include
)

set(cpp_files
	julia_to_c.cc
)

source_group(source FILES
	julia_to_c.cc
)

add_library(lib_juliacall SHARED ${cpp_files})
set_property(TARGET lib_juliacall PROPERTY COMPILE_FLAGS "-fPIC")

#add_executable(runnable_test ${cpp_files})
if(UNIX)
	target_compile_features(lib_juliacall PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_juliacall
	${GTEST_LIBRARIES}

	lib_algorithms
	lib_julia
	lib_parsing
	lib_models
	lib_data
	lib_gpu
	lib_core
	lib_ensembles
)

include_directories(lib_juliacall
	${GTEST_INCLUDE_DIRS}
	${dir_list}
)

if (MSVC)
   set_target_properties(lib_juliacall PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)
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
	../lib_algorithms/test
	../lib_julia/test
	../lib_parsing/test 
	../lib_models/test
	../lib_data/test
	../lib_gpu/test
	../lib_core/test
	../lib_algorithms/include
	../lib_julia/include
	../lib_parsing/include
	../lib_models/include
	../lib_data/include
	../lib_gpu/include
	../lib_core/include
)

set(cpp_files
	test_main.cc
)

source_group(source FILES
	test_main.cc
)

add_executable(runnable_test ${cpp_files})
target_link_libraries(runnable_test
	${GTEST_LIBRARIES}

	lib_algorithms
	lib_julia
	lib_parsing
	lib_models
	lib_data
	lib_gpu
	lib_core
)

include_directories(runnable_test
	${GTEST_INCLUDE_DIRS}
	${dir_list}
)

if(WIN32)
	add_custom_command(TARGET runnable_test POST_BUILD
		COMMAND ../../runnable_test/copy_shared_libs.bat
	)
endif(WIN32)
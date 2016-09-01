# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
)

set(cpp_files
	./test/test_main_interface.h

	../source_shared/precomp.h
	./include/lib_core.h
	./include/core_interface.h
	./include/barrier.h
	./source/barrier_impl.h

	../source_shared/precomp.cc
	./source/core_interface.cc
	./source/barrier_impl.cc
)

source_group(source FILES
	./source/barrier_impl.h

	./source/barrier_impl.cc
)

source_group(interface FILES
	./include/barrier.h
	./include/lib_core.h
	./include/core_interface.h

	./source/core_interface.cc
)

source_group(test FILES
	./test/test_main_interface.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

add_library(lib_core SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_core PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_core
)

include_directories(lib_core
	${dir_list}
)

if (MSVC)
   set_target_properties(lib_core PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)

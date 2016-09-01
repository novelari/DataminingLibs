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
	./include/lib_julia.h
	./include/julia_interface.h

	../source_shared/precomp.cc
	./source/julia_interface.cc
)

source_group(source FILES
)

source_group(interface FILES
	./include/lib_julia.h
	./include/julia_interface.h

	./source/julia_interface.cc
)

source_group(test FILES
	./test/test_main_interface.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

add_library(lib_julia SHARED ${cpp_files})
target_link_libraries(lib_julia
	lib_algorithms
	lib_parsing
)

include_directories(lib_julia
	${dir_list}
)

if (MSVC)
   set_target_properties(lib_julia PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)
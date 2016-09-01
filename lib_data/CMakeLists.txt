# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
)

set(cpp_files
	./test/test_dataframe.h

	../source_shared/precomp.h
	./include/ml_dataframe.h
	./include/ml_resultdata.h
	./include/lib_data.h
	./include/data_interface.h
	./source/ml_dataframe_impl.h
	./source/ml_resultdata_impl.h

	../source_shared/precomp.cc
	./source/data_interface.cc
	./source/ml_dataframe_impl.cc
	./source/ml_resultdata_impl.cc
)

source_group(source FILES
	./source/ml_dataframe_impl.h
	./source/ml_resultdata_impl.h

	./source/ml_dataframe_impl.cc
	./source/ml_resultdata_impl.cc
)

source_group(interface FILES
	./include/ml_dataframe.h
	./include/lib_data.h
	./include/data_interface.h
	./include/ml_resultdata.h

	./source/data_interface.cc
)

source_group(test FILES
	./test/test_dataframe.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

CUDA_ADD_LIBRARY(lib_data SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_data PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_data
)

include_directories(lib_data
	${dir_list}
)

if (MSVC)
   set_target_properties(lib_data PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)

# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
)

set(cpp_files
	./test/test_ml_model.h

	../source_shared/precomp.h
	./include/lib_models.h
	./include/ml_model.h
	./include/models_interface.h
	./source/ml_model_impl.h

	../source_shared/precomp.cc
	./source/models_interface.cc
	./source/ml_model_impl.cc
)

source_group(source FILES
	./source/ml_model_impl.h

	./source/ml_model_impl.cc
)

source_group(interface FILES
	./include/lib_models.h
	./include/ml_model.h
	./include/models_interface.h

	./source/models_interface.cc
)

source_group(test FILES
	./test/test_ml_model.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

CUDA_ADD_LIBRARY(lib_models SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_models PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_models
)

include_directories(lib_models
	${dir_list}
)

if (MSVC)
   set_target_properties(lib_models PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)

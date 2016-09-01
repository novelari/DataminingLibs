# required cmake version                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

set(dir_list
	./include
	./source
	../source_shared
	../source_shared/include
	../lib_data/include
	../lib_core/include
)

set(cpp_files
	./test/test_csv_parsing.h

	../source_shared/precomp.h
	./include/parsing_interface.h
	./include/lib_parsing.h
	./include/ml_parser.h
	./source/rds_parser.h
	./source/csv_parser.h
	./source/arff_parser.h

	../source_shared/precomp.cc
	./source/parsing_interface.cc
	./source/ml_parser.cc
	./source/rds_parser.cc
	./source/csv_parser.cc
	./source/arff_parser.cc
)

source_group(source FILES
	./source/rds_parser.h
	./source/csv_parser.h
	./source/arff_parser.h

	./source/rds_parser.cc
	./source/csv_parser.cc
	./source/arff_parser.cc
)

source_group(interface FILES
	./include/lib_parsing.h
	./include/parsing_interface.h
	./include/ml_parser.h

	./source/ml_parser.cc
	./source/parsing_interface.cc
)

source_group(test FILES
	./test/test_csv_parsing.h
)

source_group(precomp FILES
	../source_shared/precomp.h

	../source_shared/precomp.cc
)

add_library(lib_parsing SHARED ${cpp_files})
if(UNIX)
	target_compile_features(lib_parsing PRIVATE cxx_range_for cxx_contextual_conversions cxx_generic_lambdas)
endif(UNIX)

target_link_libraries(lib_parsing
	lib_data
	lib_core
)

include_directories(lib_parsing
	${dir_list}
)

if (MSVC)
   set_target_properties(lib_parsing PROPERTIES COMPILE_FLAGS "/Yuprecomp.h")
   set_source_files_properties(../source_shared/precomp.cc PROPERTIES COMPILE_FLAGS "/Ycprecomp.h")
endif(MSVC)

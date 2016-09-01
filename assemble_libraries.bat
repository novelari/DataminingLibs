rem datamininglibs_x64
mkdir datamininglibs_x64& pushd datamininglibs_x64
mkdir include
mkdir lib
mkdir bin
popd

set libs[0]=lib_algorithms
set libs[1]=lib_data
set libs[2]=lib_julia
set libs[3]=lib_models
set libs[4]=lib_parsing
set libs[5]=lib_gpu
set libs[6]=lib_core

xcopy /s ".\source_shared\include\*" ".\datamininglibs_x64\include\" /Y
for /F "tokens=2 delims==" %%s in ('set libs[') do xcopy /s ".\%%s\include\*" ".\datamininglibs_x64\include\%%s\include\" /Y
for /F "tokens=2 delims==" %%s in ('set libs[') do if exist __project_files_cmake_win64__\%%s\Build_Output\Libs\Release\%%s.lib (
	xcopy "__project_files_cmake_win64__\%%s\Build_Output\Libs\Release\%%s.lib" ".\datamininglibs_x64\lib\" /Y
)
for /F "tokens=2 delims==" %%s in ('set libs[') do if exist __project_files_cmake_win64__\%%s\Build_Output\Libs\Release\%%s.dll (
	xcopy "__project_files_cmake_win64__\%%s\Build_Output\Libs\Release\%%s.dll" ".\datamininglibs_x64\bin\" /Y
)
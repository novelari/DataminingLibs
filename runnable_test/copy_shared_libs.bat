set libs[0]=lib_algorithms
set libs[1]=lib_data
set libs[2]=lib_julia
set libs[3]=lib_models
set libs[4]=lib_parsing
set libs[5]=lib_gpu
set libs[6]=lib_core
set libs[7]=lib_ensembles

for /F "tokens=2 delims==" %%s in ('set libs[') do if exist ..\%%s\Build_Output\Libs\Debug\%%s.dll (
    xcopy ..\%%s\Build_Output\Libs\Debug\%%s.dll .\Build_Output\bin\Debug\ /Y
)
for /F "tokens=2 delims==" %%s in ('set libs[') do if exist ..\%%s\Build_Output\Libs\Release\%%s.dll (
	xcopy ..\%%s\Build_Output\Libs\Release\%%s.dll .\Build_Output\bin\Release\ /Y
)
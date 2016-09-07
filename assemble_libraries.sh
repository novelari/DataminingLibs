rm -rf datamininglibs_x64
mkdir datamininglibs_x64
libs_array=(lib_algorithms lib_data lib_julia lib_models lib_parsing lib_gpu lib_core lib_ensembles)
cd datamininglibs_x64
mkdir include
cd ./include
for i in "${libs_array[@]}"
do
mkdir $i
cd ./$i
mkdir include
cd ..
done
cd ..
mkdir lib
mkdir bin
cd ..

cp __project_files_cmake_unix64__/runnable_test/Build_Output/bin/runnable_test ./datamininglibs_x64/bin/
cp ./source_shared/include/* ./datamininglibs_x64/include/
for i in "${libs_array[@]}"
do
 cp ./$i/include/* ./datamininglibs_x64/include/$i/include/
done
for i in "${libs_array[@]}"
do
 cp __project_files_cmake_unix64__/$i/Build_Output/Libs/lib$i.so ./datamininglibs_x64/lib/
done

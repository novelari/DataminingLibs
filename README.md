Datamining libs
=====================
Datamining libraries for decision tree ensembles running on Gpu's.

Requirements
---------------------
    Compiler with c++11 support (GCC 5.4, MSVC 2013)
    Cmake: https://cmake.org/download/
    The latest Cuda toolkit: https://developer.nvidia.com/cuda-toolkit

    Optional requirement for compilation of the tests:
    Google test: https://github.com/google/googletest

Compiling on Windows
---------------------
Build the visual studio projects using the following script:
    
    ./build_cmake_vs_projects.bat

Browse into the newly created __project_files_cmake_win64__ directory where you will find the solution file that can be opened and compiled in visual studio.

Compiling on Unix
---------------------
Tested to work on Ubuntu 16.04 LTS (GCC 5.4) and Cuda toolkit 8.0rc (with the 8.0.27.1 patch for GCC 5.4 support)

Build the make files using the following script:
    
    ./build_cmake_unix_make.sh

Browse into the newly created __project_files_cmake_unix64__ directory and execute the make command. 
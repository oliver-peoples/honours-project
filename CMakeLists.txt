cmake_minimum_required(VERSION 3.18)

project(tem_mode_trilateration)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

find_package(Eigen3)

add_executable(tem_modes tem_modes.cu)
target_link_libraries(tem_modes rt pthread dl)
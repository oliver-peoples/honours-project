cmake_minimum_required(VERSION 3.18)

project(qclm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(ensmallen REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/optim)

add_executable(tem_localization_2d ${CMAKE_SOURCE_DIR}/tem_localization_2d.cpp)
add_executable(render_tem ${CMAKE_SOURCE_DIR}/render_tem.cpp)
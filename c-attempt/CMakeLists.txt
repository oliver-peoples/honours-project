cmake_minimum_required(VERSION 3.18)

project(qclm)

set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/eigen3)
include_directories(${CMAKE_SOURCE_DIR}/optim)

add_executable(tem_render_3d ${CMAKE_SOURCE_DIR}/tem-render/tem_render_3d.cpp)
target_link_libraries(tem_render_3d PUBLIC OpenMP::OpenMP_CXX rt dl pthread)
target_compile_options(tem_render_3d PUBLIC -fopenmp)

add_executable(multicore_localization
    ${CMAKE_SOURCE_DIR}/multicore-localization-inf-time/main.cpp
    ${CMAKE_SOURCE_DIR}/multicore-localization-inf-time/main_simple.cpp
    ${CMAKE_SOURCE_DIR}/multicore-localization-inf-time/main_noise_response.cpp
)
target_link_libraries(multicore_localization PUBLIC OpenMP::OpenMP_CXX rt dl pthread)
target_compile_options(multicore_localization PUBLIC -fopenmp)

add_executable(finite_time
    ${CMAKE_SOURCE_DIR}/finite-time/main.cpp
    ${CMAKE_SOURCE_DIR}/finite-time/main_simple.cpp
    ${CMAKE_SOURCE_DIR}/finite-time/main_noise_response.cpp
)
target_link_libraries(finite_time PUBLIC OpenMP::OpenMP_CXX rt dl pthread)
target_compile_options(finite_time PUBLIC -fopenmp)

add_executable(avg_finite_time
    ${CMAKE_SOURCE_DIR}/avg-finite-time/main.cpp
    ${CMAKE_SOURCE_DIR}/avg-finite-time/main_simple.cpp
)
target_link_libraries(avg_finite_time PUBLIC OpenMP::OpenMP_CXX rt dl pthread)
target_compile_options(avg_finite_time PUBLIC -fopenmp)

add_executable(multicore_monte_carlo ${CMAKE_SOURCE_DIR}/multicore-monte-carlo/main.cpp)
target_link_libraries(multicore_monte_carlo PUBLIC OpenMP::OpenMP_CXX rt dl pthread)
target_compile_options(multicore_monte_carlo PUBLIC -fopenmp)

add_executable(queue
    ${CMAKE_SOURCE_DIR}/queue-inf-time/main.cpp
    ${CMAKE_SOURCE_DIR}/queue-inf-time/noiseResponse.cpp
)
target_link_libraries(queue PUBLIC OpenMP::OpenMP_CXX rt dl pthread)
target_compile_options(queue PUBLIC -fopenmp)

add_executable(tem_localization ${CMAKE_SOURCE_DIR}/tem-localization/main.cpp)
target_link_libraries(tem_localization PUBLIC OpenMP::OpenMP_CXX rt dl pthread)
target_compile_options(tem_localization PUBLIC -fopenmp)
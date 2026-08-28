cmake_minimum_required(VERSION 3.18)
project(simulated_bifurcation LANGUAGES CXX)

option(SBM_NATIVE_OPTIMIZATION "Enable native CPU SIMD instructions" OFF)

add_library(sbm
    src/sbm/model.cpp
    src/sbm/cpu_solver.cpp
)
target_include_directories(sbm PUBLIC include)
target_compile_features(sbm PUBLIC cxx_std_17)
if(SBM_NATIVE_OPTIMIZATION AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sbm PRIVATE -march=native)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sbm PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(sbm_solve src/sbm/main.cpp)
target_link_libraries(sbm_solve PRIVATE sbm)

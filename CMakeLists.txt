cmake_minimum_required(VERSION 3.18)
project(simulated_bifurcation LANGUAGES CXX)

option(SBM_ENABLE_CUDA "Build the CUDA backend" OFF)
option(SBM_BUILD_FPGA_SIM "Build the software simulation of the FPGA HLS kernel" ON)
option(SBM_NATIVE_OPTIMIZATION "Enable native CPU SIMD instructions" OFF)

add_library(sbm
    src/sbm/model.cpp
    src/sbm/maxcut.cpp
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

if(SBM_BUILD_FPGA_SIM)
    target_sources(sbm PRIVATE fpga/dsb_hls.cpp src/sbm/fpga_sim_solver.cpp)
    target_include_directories(sbm PRIVATE fpga)
    target_compile_definitions(sbm PUBLIC SBM_HAS_FPGA_SIM=1)
endif()

if(SBM_ENABLE_CUDA)
    enable_language(CUDA)
    target_sources(sbm PRIVATE cuda/gpu_solver.cu)
    target_compile_definitions(sbm PUBLIC SBM_HAS_CUDA=1)
    set_target_properties(sbm PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endif()

add_executable(sbm_solve src/sbm/main.cpp)
target_link_libraries(sbm_solve PRIVATE sbm)

add_executable(maxcut_generate src/benchmark/generate_maxcut.cpp)
target_link_libraries(maxcut_generate PRIVATE sbm)

add_executable(maxcut_benchmark src/benchmark/maxcut_benchmark.cpp)
target_link_libraries(maxcut_benchmark PRIVATE sbm)

add_executable(maxcut_batch_benchmark src/benchmark/maxcut_batch_benchmark.cpp)
target_link_libraries(maxcut_batch_benchmark PRIVATE sbm)

add_executable(maxcut_large_benchmark src/benchmark/maxcut_large_benchmark.cpp)
target_link_libraries(maxcut_large_benchmark PRIVATE sbm)

add_executable(bqm_margin_benchmark src/benchmark/bqm_margin_benchmark.cpp)
target_link_libraries(bqm_margin_benchmark PRIVATE sbm)

include(CTest)
if(BUILD_TESTING)
    add_executable(sbm_tests tests/sbm_tests.cpp)
    target_link_libraries(sbm_tests PRIVATE sbm)
    add_test(NAME sbm_tests COMMAND sbm_tests)
endif()

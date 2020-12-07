cmake_minimum_required(VERSION 3.8)
project(densecrf LANGUAGES CXX)

# provide FindCUDAToolkit for cmake versions below 3.17
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
enable_language(CUDA)

# CPU Example Target
add_executable(example_cpu
  examples/example_cpu.cpp
  examples/util.cpp
  examples/util.h
  include/densecrf_base.h
  include/densecrf_cpu.h
  include/pairwise_cpu.h
  include/permutohedral_cpu.h
)
target_include_directories(example_cpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(example_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >)
target_compile_features(example_cpu PUBLIC cxx_std_11)

# GPU Example Target
add_executable(example_gpu
  examples/example_gpu.cu
  examples/util.cpp
  examples/util.h
  include/densecrf_base.h
  include/densecrf_cpu.h
  include/pairwise_cpu.h
  include/permutohedral_cpu.h
)
target_include_directories(example_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(example_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math >
  $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >
)
target_compile_features(example_gpu PUBLIC cxx_std_11)
set_target_properties( example_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

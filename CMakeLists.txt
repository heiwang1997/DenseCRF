cmake_minimum_required(VERSION 3.8)
project(densecrf_cuda LANGUAGES CXX)

# provide FindCUDAToolkit for cmake versions below 3.17
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
enable_language(CUDA)

add_library(${PROJECT_NAME}_cpu INTERFACE)
target_include_directories(${PROJECT_NAME}_cpu INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/")

add_library(${PROJECT_NAME}_cuda INTERFACE)
target_include_directories(${PROJECT_NAME}_cuda INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/")

# CPU Example Target
add_executable(example_cpu
  examples/example_cpu.cpp
  examples/util.cpp
  examples/util.h
)
target_link_libraries(example_cpu ${PROJECT_NAME}_cpu)
target_compile_options(example_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >)
target_compile_features(example_cpu PUBLIC cxx_std_11)

# GPU Example Target
add_executable(example_gpu
  examples/example_gpu.cu
  examples/util.cpp
  examples/util.h
)
target_link_libraries(example_gpu ${PROJECT_NAME}_cuda)
target_compile_options(example_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math >
  $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >
)
target_compile_features(example_gpu PUBLIC cxx_std_11)
set_target_properties( example_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS ${PROJECT_NAME}_cpu EXPORT ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}_cuda EXPORT ${PROJECT_NAME})

include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

install(EXPORT ${PROJECT_NAME}
  DESTINATION share/${PROJECT_NAME}/cmake
  FILE ${PROJECT_NAME}Config.cmake
)

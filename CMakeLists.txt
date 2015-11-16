project(cnn)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -Wall -Wpedantic -std=c++11 -Ofast -g -DEIGEN_FAST_MATH -Wno-unused-local-typedefs -march=native")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

function(find_cudnn)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
    DOC "CUDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
    DOC "CUDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found CUDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find CUDNN in path: ${CUDNN_ROOT} (Did you set CUDNN_ROOT properly?)")
  endif()
endfunction()

set(WITH_CUDA_BACKEND 1)

if (WITH_CUDA_BACKEND)
  find_package(CUDA REQUIRED)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  add_definitions(-DHAVE_CUDA)
  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  #find_cudnn()
  #include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
endif()

CUDA_ADD_EXECUTABLE(mm mm.cu)
  set(CUDA_SEPARABLE_COMPILATION ON)
  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-std=c++11;-O2;-DVERBOSE")
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
CUDA_ADD_CUBLAS_TO_TARGET(mm)
add_dependencies(mm ${CUDA_curand_LIBRARY})
target_link_libraries(mm ${CUDA_curand_LIBRARY})


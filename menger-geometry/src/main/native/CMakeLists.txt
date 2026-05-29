cmake_minimum_required(VERSION 3.18)
project(mengergeometry CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_MESSAGE LAZY)

#=============================================================================
# CUDA (required for PTX shader compilation)
#=============================================================================
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CUDA compiler not found.")
endif()

set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;75-virtual")
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 12.0 REQUIRED)

#=============================================================================
# OptiX SDK
#=============================================================================
if(NOT DEFINED OptiX_INSTALL_DIR)
  if(DEFINED ENV{OPTIX_ROOT})
    set(OptiX_INSTALL_DIR "$ENV{OPTIX_ROOT}")
  else()
    file(GLOB OPTIX_CANDIDATES
      "/usr/local/optix"
      "/opt/optix"
      "/usr/local/NVIDIA-OptiX-SDK-*"
      "/opt/NVIDIA-OptiX-SDK-*"
      "$ENV{HOME}/NVIDIA-OptiX-SDK-*"
      "$ENV{HOME}/Downloads/NVIDIA-OptiX-SDK-*"
    )
    if(OPTIX_CANDIDATES)
      list(SORT OPTIX_CANDIDATES ORDER DESCENDING)
      list(GET OPTIX_CANDIDATES 0 OptiX_INSTALL_DIR)
    else()
      message(FATAL_ERROR "OptiX SDK not found. Set OPTIX_ROOT environment variable.")
    endif()
  endif()
endif()
set(OptiX_INCLUDE_DIR "${OptiX_INSTALL_DIR}/include")
if(NOT EXISTS "${OptiX_INCLUDE_DIR}/optix.h")
  message(FATAL_ERROR "optix.h not found at ${OptiX_INCLUDE_DIR}")
endif()
message(STATUS "✓ OptiX: ${OptiX_INSTALL_DIR}")

#=============================================================================
# optix-jni headers (project reference — replaced by published artifact in Sprint 25)
#=============================================================================
set(OPTIX_JNI_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../optix-jni/src/main/native/include")

#=============================================================================
# Stub PTX shader — validates MengerParams struct compiles with nvcc + OptiX
#=============================================================================
add_library(menger_stub_shader OBJECT
    shaders/menger_stub.cu
)
target_include_directories(menger_stub_shader PRIVATE
    ${OptiX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_JNI_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(menger_stub_shader PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75-virtual"
)

#=============================================================================
# Stub shared library (placeholder for Sprint 25 JNI bindings)
#=============================================================================
file(WRITE ${CMAKE_BINARY_DIR}/stub.cpp "// stub\n")
add_library(mengergeometry SHARED ${CMAKE_BINARY_DIR}/stub.cpp)
target_link_libraries(mengergeometry PRIVATE CUDA::cudart)

install(TARGETS mengergeometry LIBRARY DESTINATION .)
install(FILES
    "${CMAKE_BINARY_DIR}/CMakeFiles/menger_stub_shader.dir/shaders/menger_stub.ptx"
    DESTINATION .
    RENAME "menger_stub.ptx"
    OPTIONAL
)

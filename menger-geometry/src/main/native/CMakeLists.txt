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
# optix-jni source paths (project reference — headers replaced by published artifact post-Sprint 25)
#=============================================================================
set(OPTIX_JNI_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../optix-jni/src/main/native/include")
set(OPTIX_JNI_SHADER_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../optix-jni/src/main/native/shaders")

#=============================================================================
# Full menger PTX shader: base OptiX shaders + 4D fractal shaders + caustics
# Compiled to optix_shaders_menger.ptx, used by MengerRenderer instead of
# optix-jni's optix_shaders.ptx.
#=============================================================================
add_library(menger_shaders OBJECT
    shaders/optix_shaders_menger.cu
)
target_include_directories(menger_shaders PRIVATE
    ${OptiX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_JNI_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders  # local 4D shaders found before OPTIX_JNI_SHADER_DIR
    ${OPTIX_JNI_SHADER_DIR}              # base shaders (helpers.cu, raygen, miss, hits, etc.)
)
set_target_properties(menger_shaders PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75-virtual"
)

#=============================================================================
# menger-geometry shared library: project4d.cu + CausticsRenderer
# CausticsRenderer calls symbols from liboptixjni.so — resolved at runtime
# when both JNI libraries are loaded by the JVM.
#=============================================================================
add_library(mengergeometry SHARED
    project4d.cu
    CausticsRenderer.cpp
)
target_include_directories(mengergeometry PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OptiX_INCLUDE_DIR}
    ${OPTIX_JNI_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mengergeometry PRIVATE CUDA::cudart)
target_link_options(mengergeometry PRIVATE "LINKER:--allow-shlib-undefined")
set_target_properties(mengergeometry PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

install(TARGETS mengergeometry LIBRARY DESTINATION .)
install(FILES
    "${CMAKE_BINARY_DIR}/CMakeFiles/menger_shaders.dir/shaders/optix_shaders_menger.ptx"
    DESTINATION .
    RENAME "optix_shaders_menger.ptx"
    OPTIONAL
)

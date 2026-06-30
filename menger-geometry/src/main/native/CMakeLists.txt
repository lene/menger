cmake_minimum_required(VERSION 3.18)
project(mengergeometry CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_MESSAGE LAZY)

# AddressSanitizer (ASan) — optional, enabled for leak-check builds.
# Usage: cmake -DENABLE_ASAN=ON ...
# Works with GCC and Clang. CUDA-side ASan support is compiler-dependent;
# compute-sanitizer covers GPU-side leak detection.
option(ENABLE_ASAN "Enable AddressSanitizer for CPU-side leak detection" OFF)
if(ENABLE_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
  message(STATUS "✓ AddressSanitizer enabled")
endif()

# ccache compiler launcher — speeds up repeated native rebuilds (sprint 28.2).
# No-op when ccache is not installed.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

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
# JNI (required for MengerJNIBindings)
#=============================================================================
find_package(JNI REQUIRED)
message(STATUS "✓ JNI: ${JNI_INCLUDE_DIRS}")

#=============================================================================
# FFmpeg/libav (required for video texture decoding)
#=============================================================================
find_package(FFmpeg REQUIRED COMPONENTS avcodec avformat avutil swscale)
message(STATUS "✓ FFmpeg/libav: avcodec ${FFmpeg_avcodec_VERSION}")

#=============================================================================
# optix-jni native API from the published artifact
#=============================================================================
if(NOT DEFINED OPTIX_JNI_INCLUDE_DIR)
  message(FATAL_ERROR "OPTIX_JNI_INCLUDE_DIR is required; run through sbt nativeCompile.")
endif()
if(NOT EXISTS "${OPTIX_JNI_INCLUDE_DIR}/OptiXData.h")
  message(FATAL_ERROR "OptiXData.h not found in OPTIX_JNI_INCLUDE_DIR: ${OPTIX_JNI_INCLUDE_DIR}")
endif()

if(NOT DEFINED OPTIX_JNI_SHADER_DIR)
  message(FATAL_ERROR "OPTIX_JNI_SHADER_DIR is required; run through sbt nativeCompile.")
endif()
if(NOT EXISTS "${OPTIX_JNI_SHADER_DIR}/helpers.cu")
  message(FATAL_ERROR "helpers.cu not found in OPTIX_JNI_SHADER_DIR: ${OPTIX_JNI_SHADER_DIR}")
endif()

message(STATUS "✓ optix-jni include API: ${OPTIX_JNI_INCLUDE_DIR}")
message(STATUS "✓ optix-jni shader API: ${OPTIX_JNI_SHADER_DIR}")

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
    VideoLoader.cpp
    MengerJNIBindings.cpp
)
target_include_directories(mengergeometry PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OptiX_INCLUDE_DIR}
    ${OPTIX_JNI_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mengergeometry PRIVATE
    CUDA::cudart
    FFmpeg::avcodec
    FFmpeg::avformat
    FFmpeg::avutil
    FFmpeg::swscale
)
# libmengergeometry.so calls OptiXWrapper symbols from liboptixjni.so.
# --allow-shlib-undefined avoids a hard build-time dependency on liboptixjni.so.
# At runtime, MengerJNIBindings.cpp's constructor promotes liboptixjni.so to
# RTLD_GLOBAL so lazy binding can resolve the symbols regardless of load order.
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

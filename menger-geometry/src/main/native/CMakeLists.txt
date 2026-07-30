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

# Generated JNI headers (sbt JniJavah, -DJAVAH_INCLUDE_DIR): MengerJNIBindings.cpp and
# VideoLoader.cpp #include these so a Scala @native / C++ signature mismatch is a compile
# error (Sprint 35 Task 1.4).
if(DEFINED JAVAH_INCLUDE_DIR)
  include_directories(${JAVAH_INCLUDE_DIR})
  message(STATUS "✓ Generated JNI headers: ${JAVAH_INCLUDE_DIR}")
endif()

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
# Registrable 4D-fractal PTX module (Sprint 35 Task 1.2b).
# Compiled to menger_4d.ptx: ONLY the 4D intersection/closest-hit/shadow programs
# plus the device helpers they call (helpers.cu). MengerRenderer registers it through
# optix-jni's custom-geometry SPI, on top of optix-jni's own base pipeline
# (optix_shaders.ptx) — no more forking the whole shader layer.
#=============================================================================
add_library(menger_4d_shaders OBJECT
    shaders/menger_4d_module.cu
)
target_include_directories(menger_4d_shaders PRIVATE
    ${OptiX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_JNI_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders  # local 4D shaders found before OPTIX_JNI_SHADER_DIR
    ${OPTIX_JNI_SHADER_DIR}              # helpers.cu (device-helper infrastructure)
)
set_target_properties(menger_4d_shaders PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75-virtual"
)

#=============================================================================
# menger-geometry shared library: video decoding only.
# As of Sprint 35 Task 1.3 the caustics/project4d fork was deleted — optix-jni
# owns caustics, reached via OptiXRenderer's JNI in liboptixjni.so. This library
# no longer references any optix-jni symbol, so it links cleanly (no
# --allow-shlib-undefined, no RTLD_GLOBAL promotion).
#=============================================================================
add_library(mengergeometry SHARED
    VideoLoader.cpp
    MengerJNIBindings.cpp
)
target_include_directories(mengergeometry PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mengergeometry PRIVATE
    FFmpeg::avcodec
    FFmpeg::avformat
    FFmpeg::avutil
    FFmpeg::swscale
)

install(TARGETS mengergeometry LIBRARY DESTINATION .)
install(FILES
    "${CMAKE_BINARY_DIR}/CMakeFiles/menger_4d_shaders.dir/shaders/menger_4d_module.ptx"
    DESTINATION .
    RENAME "menger_4d.ptx"
    OPTIONAL
)

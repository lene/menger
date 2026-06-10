include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)

if(NOT FFmpeg_FIND_COMPONENTS)
  set(FFmpeg_FIND_COMPONENTS avcodec avformat avutil swscale)
endif()

set(_FFmpeg_SUPPORTED_COMPONENTS avcodec avformat avutil swscale)
set(_FFmpeg_REQUIRED_VARS)
set(FFmpeg_INCLUDE_DIRS)
set(FFmpeg_LIBRARIES)

foreach(_component IN LISTS FFmpeg_FIND_COMPONENTS)
  if(NOT _component IN_LIST _FFmpeg_SUPPORTED_COMPONENTS)
    message(FATAL_ERROR "Unsupported FFmpeg component requested: ${_component}")
  endif()

  set(_pkg "lib${_component}")
  set(_pc "PC_FFmpeg_${_component}")
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(${_pc} QUIET "${_pkg}")
  endif()

  set(_header "lib${_component}/${_component}.h")
  find_path(FFmpeg_${_component}_INCLUDE_DIR
    NAMES "${_header}"
    HINTS ${${_pc}_INCLUDE_DIRS}
  )
  find_library(FFmpeg_${_component}_LIBRARY
    NAMES "${_component}"
    HINTS ${${_pc}_LIBRARY_DIRS}
  )

  set(FFmpeg_${_component}_VERSION "${${_pc}_VERSION}")

  if(FFmpeg_${_component}_INCLUDE_DIR AND FFmpeg_${_component}_LIBRARY)
    set(FFmpeg_${_component}_FOUND TRUE)
    list(APPEND FFmpeg_INCLUDE_DIRS "${FFmpeg_${_component}_INCLUDE_DIR}")
    list(APPEND FFmpeg_LIBRARIES "${FFmpeg_${_component}_LIBRARY}")

    if(NOT TARGET FFmpeg::${_component})
      add_library(FFmpeg::${_component} UNKNOWN IMPORTED)
      set_target_properties(FFmpeg::${_component} PROPERTIES
        IMPORTED_LOCATION "${FFmpeg_${_component}_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFmpeg_${_component}_INCLUDE_DIR}"
      )
    endif()
  endif()

  list(APPEND _FFmpeg_REQUIRED_VARS
    FFmpeg_${_component}_INCLUDE_DIR
    FFmpeg_${_component}_LIBRARY
  )
endforeach()

list(REMOVE_DUPLICATES FFmpeg_INCLUDE_DIRS)
list(REMOVE_DUPLICATES FFmpeg_LIBRARIES)

if(FFmpeg_avcodec_VERSION AND FFmpeg_avcodec_VERSION VERSION_LESS 58)
  set(FFmpeg_avcodec_FOUND FALSE)
  message(STATUS "FFmpeg avcodec ${FFmpeg_avcodec_VERSION} is too old; require >= 58")
endif()

find_package_handle_standard_args(FFmpeg
  REQUIRED_VARS ${_FFmpeg_REQUIRED_VARS}
  HANDLE_COMPONENTS
)

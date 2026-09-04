cmake_minimum_required(VERSION 3.15)
project(ls_eend_ax LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# BSP layout: <BSP_MSP_DIR>/include/ax_engine_api.h and <BSP_MSP_DIR>/lib/*.so
if(NOT BSP_MSP_DIR)
  set(BSP_MSP_DIR "$ENV{BSP_MSP_DIR}")
endif()
if(NOT BSP_MSP_DIR)
  message(FATAL_ERROR
    "Set BSP_MSP_DIR to <ax650n_bsp_sdk>/msp/out "
    "(see cpp/README.md / download_toolchains.sh)")
endif()

include_directories(
  ${BSP_MSP_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft
)
link_directories(${BSP_MSP_DIR}/lib)

# kissfft is C, so it needs its own target (a C++-only project silently skips it).
add_library(ls_eend_kissfft STATIC third_party/kissfft/kiss_fft.c)
target_include_directories(ls_eend_kissfft PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft)

add_library(ls_eend_sdk STATIC
  src/audio_reader.cpp
  src/feature_extractor.cpp
  src/ax_engine_runner.cpp
  src/streaming_diarizer.cpp
)
# AX_NPU_* symbols live in libax_interpreter, not libax_engine.
target_link_libraries(ls_eend_sdk PUBLIC
  ls_eend_kissfft ax_engine ax_interpreter ax_sys pthread dl m)

add_executable(ls_eend_ax650 ls_eend_ax.cpp)
target_link_libraries(ls_eend_ax650 PRIVATE ls_eend_sdk)

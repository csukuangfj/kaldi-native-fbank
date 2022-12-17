if("x${CMAKE_SOURCE_DIR}" STREQUAL "x${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "\
In-source build is not a good practice.
Please use:
  mkdir build
  cd build
  cmake ..
to build this project"
  )
endif()

if(CMAKE_TOOLCHAIN_FILE)
  set(_BUILD_PYTHON OFF)
  set(_BUILD_TESTS OFF)
  set(_ENABLE_CHECK OFF)
else()
  set(_BUILD_PYTHON ON)
  set(_BUILD_TESTS ON)
  set(_ENABLE_CHECK ON)
endif()


if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(kaldi-native-fbank CXX C)

set(KALDI_NATIVE_FBANK_VERSION "1.7")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(kaldi_native_fbank_rpath_origin "$ORIGIN")
else()
  set(kaldi_native_fbank_rpath_origin "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${kaldi_native_fbank_rpath_origin})
set(CMAKE_BUILD_RPATH ${kaldi_native_fbank_rpath_origin})

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")

if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()
message(STATUS "BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

if(WIN32)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for Windows")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

option(KALDI_NATIVE_FBANK_BUILD_TESTS "Whether to build tests or not" ${_BUILD_TESTS})
option(KALDI_NATIVE_FBANK_BUILD_PYTHON "Whether to build Python extension" ${_BUILD_PYTHON})
option(KALDI_NATIVE_FBANK_ENABLE_CHECK "Whether to build with log" ${_ENABLE_CHECK})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

message(STATUS "KALDI_NATIVE_FBANK_ENABLE_CHECK: ${KALDI_NATIVE_FBANK_ENABLE_CHECK}")

if(KALDI_NATIVE_FBANK_BUILD_PYTHON)
  include(pybind11)
endif()

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  enable_testing()
  include(googletest)
endif()

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h KNF_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h KNF_HAVE_EXECINFO_H)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(kaldi-native-fbank)

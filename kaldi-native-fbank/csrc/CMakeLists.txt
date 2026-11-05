
include_directories(${PROJECT_SOURCE_DIR})
set(sources
  feature-cmvn.cc
  feature-delta.cc
  feature-fbank.cc
  feature-functions.cc
  feature-mfcc.cc
  feature-raw-audio-samples.cc
  feature-splice.cc
  feature-window.cc
  istft.cc
  kaldi-math.cc
  mel-computations.cc
  online-feature.cc
  online-nemo-fbank.cc
  rfft.cc
  stft.cc
  whisper-feature.cc
)

if(KALDI_NATIVE_FBANK_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()

add_library(kaldi-native-fbank-core STATIC ${sources})
if(KALDI_NATIVE_FBANK_ENABLE_CHECK)
  target_compile_definitions(kaldi-native-fbank-core PUBLIC KNF_ENABLE_CHECK=1)

  if(KNF_HAVE_EXECINFO_H)
    target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_EXECINFO_H=1)
  endif()

  if(KNF_HAVE_CXXABI_H)
    target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_CXXABI_H=1)
  endif()
endif()

# We are using std::call_once() in log.h,which requires us to link with -pthread
if(NOT WIN32 AND KALDI_NATIVE_FBANK_ENABLE_CHECK)
  target_link_libraries(kaldi-native-fbank-core -pthread)
endif()
target_link_libraries(kaldi-native-fbank-core kissfft)
target_link_libraries(kaldi-native-fbank-core Eigen3::Eigen)

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  add_executable(test-online-fbank test-online-fbank.cc)
  target_link_libraries(test-online-fbank kaldi-native-fbank-core)
endif()

function(kaldi_native_fbank_add_test source)
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} "${source}")
  target_link_libraries(${name}
    PRIVATE
      kaldi-native-fbank-core
      gtest
      gtest_main
  )

  # Pass the target (not $<TARGET_FILE:>) so CMAKE_CROSSCOMPILING_EMULATOR
  # is used when cross-compiling (e.g. qemu-arm-static for arm32 CI).
  add_test(NAME "Test.${name}"
    COMMAND
    ${name}
  )
endfunction()

# please sort the source files alphabetically
set(test_srcs
  # test-online-feature.cc
  test-cmvn.cc
  test-delta.cc
  test-fbank-eps.cc
  test-log.cc
  test-online-nemo-fbank.cc
  test-rfft.cc
  test-splice.cc
)

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  foreach(source IN LISTS test_srcs)
    kaldi_native_fbank_add_test(${source})
  endforeach()
endif()

install(TARGETS kaldi-native-fbank-core
  DESTINATION lib
)

# ---- clang-tidy (top-level project only) ------------------------------------
# Sources to lint with `make check` (clang-tidy).  Keep this list in sync
# with scripts/check_style_clang_format.sh.  Don't pass --config-file:
# clang-tidy picks up the repo-root .clang-tidy automatically.
# Only lint files that are actually compiled (they have compile commands).
set(srcs_to_check)
foreach(source IN LISTS sources test_srcs)
  list(APPEND srcs_to_check ${CMAKE_CURRENT_SOURCE_DIR}/${source})
endforeach()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

# The python bindings are compiled into the extension module.
file(GLOB _python_srcs ${PROJECT_SOURCE_DIR}/kaldi-native-fbank/python/csrc/*.cc)
list(APPEND srcs_to_check ${_python_srcs})

# For clang-tidy.  Requires CMAKE_EXPORT_COMPILE_COMMANDS (set at the
# top level) so that compile_commands.json exists in the build dir.
#
# On macOS, pass the SDK path so that the system headers (<cmath> etc.)
# are found; without it clang-tidy reports bogus parse errors.
set(_tidy_extra_args)
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE _tidy_sdk
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND _tidy_extra_args "--extra-arg=-isysroot${_tidy_sdk}")
endif()

add_custom_target(
  clang-tidy-check
  clang-tidy ${_tidy_extra_args} -p ${CMAKE_BINARY_DIR}/compile_commands.json
             ${srcs_to_check}
  DEPENDS ${sources})

add_custom_target(check DEPENDS clang-tidy-check)

endif()  # top-level project only
if(KALDI_NATIVE_FBANK_BUILD_PYTHON AND WIN32)
  install(TARGETS kaldi-native-fbank-core
    DESTINATION ..
  )
endif()

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  install(TARGETS test-online-fbank
    DESTINATION bin
  )
endif()

file(MAKE_DIRECTORY
    ${PROJECT_BINARY_DIR}/include/kaldi-native-fbank/csrc
)
file(GLOB_RECURSE all_headers *.h)
# test-*.h are unit-test helpers; do not install them.
list(FILTER all_headers EXCLUDE REGEX "/test-[^/]*\.h$")

file(COPY
  ${all_headers}
  DESTINATION
    ${PROJECT_BINARY_DIR}/include/kaldi-native-fbank/csrc
)

install(FILES ${all_headers}
  DESTINATION include/kaldi-native-fbank/csrc
)

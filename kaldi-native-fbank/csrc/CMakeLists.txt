
include_directories(${PROJECT_SOURCE_DIR})
add_library(kaldi-native-fbank-core
  feature-fbank.cc
  feature-functions.cc
  feature-window.cc
  fftsg.c
  log.cc
  mel-computations.cc
  online-feature.cc
  rfft.cc
)
# We are using std::call_once() in log.h,which requires us to link with -pthread
target_link_libraries(kaldi-native-fbank-core -pthread)

if(KNF_HAVE_EXECINFO_H)
  target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_EXECINFO_H=1)
endif()

if(KNF_HAVE_CXXABI_H)
  target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_CXXABI_H=1)
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

  add_test(NAME "Test.${name}"
    COMMAND
    $<TARGET_FILE:${name}>
  )
endfunction()

# please sort the source files alphabetically
set(test_srcs
  # test-online-feature.cc
  test-log.cc
  test-rfft.cc
)

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  foreach(source IN LISTS test_srcs)
    kaldi_native_fbank_add_test(${source})
  endforeach()
endif()

install(TARGETS kaldi-native-fbank-core
  DESTINATION lib
)

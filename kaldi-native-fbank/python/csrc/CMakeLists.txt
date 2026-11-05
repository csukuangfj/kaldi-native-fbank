pybind11_add_module(_kaldi_native_fbank
  feature-cmvn.cc
  feature-delta.cc
  feature-fbank.cc
  feature-raw-audio-samples.cc
  feature-splice.cc
  feature-transform.cc
  feature-mfcc.cc
  feature-pitch.cc
  feature-window.cc
  istft.cc
  kaldi-native-fbank.cc
  mel-computations.cc
  online-feature.cc
  online-nemo-fbank.cc
  rfft.cc
  stft.cc
  utils.cc
)

# Like sherpa-onnx: one self-contained module inside the package lib dir.
set_target_properties(_kaldi_native_fbank PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../kaldi_native_fbank/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../kaldi_native_fbank/lib"
)

if(APPLE)
  if(PYTHON_EXECUTABLE)
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
    )
    message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  else()
    message(STATUS "PYTHON_EXECUTABLE is empty; skip probing PYTHON_SITE_PACKAGE_DIR")
  endif()
  if(PYTHON_SITE_PACKAGE_DIR STREQUAL "")
    message(WARNING "PYTHON_SITE_PACKAGE_DIR is empty!")
  else()
    target_link_libraries(_kaldi_native_fbank PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
  endif()
endif()

if(NOT WIN32)
  target_link_libraries(_kaldi_native_fbank PRIVATE "-Wl,-rpath,${kaldi_native_fbank_rpath_origin}/kaldi_native_fbank/lib")
endif()

target_link_libraries(_kaldi_native_fbank PRIVATE kaldi-native-fbank-core)

install(TARGETS _kaldi_native_fbank
  DESTINATION lib
)

add_subdirectory(csrc)

if(kALDI_NATIVE_FBANK_BUILD_PYTHON)
  add_subdirectory(python)
endif()

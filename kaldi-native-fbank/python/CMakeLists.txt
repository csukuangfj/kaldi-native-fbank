# PYTHON_EXECUTABLE is normally passed in by setup.py; fall back to the
# interpreter found by CMake so python tests also run in plain cmake builds.
if(NOT PYTHON_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()

add_subdirectory(csrc)
add_subdirectory(tests)

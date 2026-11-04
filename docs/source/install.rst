Installation
============

From PyPI::

   pip install kaldi-native-fbank

From source::

   python setup.py install

CMake::

   cmake -B build -DKALDI_NATIVE_FBANK_BUILD_PYTHON=ON -DKALDI_NATIVE_FBANK_BUILD_TESTS=ON
   cmake --build build
   ctest --test-dir build --output-on-failure

#!/usr/bin/env bash
#
# Generate the .pyi type stubs for the python module with pybind11-stubgen.
#
# Usage (after building the python module, e.g. `cd build && make -j2`):
#
#   ./scripts/generate_python_stubs.sh
#
# The generated files are written into kaldi-native-fbank/python/kaldi_native_fbank/:
#   __init__.pyi, lib/__init__.pyi, lib/_kaldi_native_fbank.pyi
#
# pybind11-stubgen reads the built module, so the stubs always match the
# bindings (constructor defaults included). Re-run this script whenever the
# bindings change.

set -e

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
kaldi_native_fbank_dir=$(cd $cur_dir/.. && pwd)
python_dir=$kaldi_native_fbank_dir/kaldi-native-fbank/python

PYTHON=${PYTHON:-python3}

PYTHONPATH=$python_dir $PYTHON -m pybind11_stubgen \
  kaldi_native_fbank \
  -o "$python_dir"

echo "Stubs written to $python_dir/kaldi_native_fbank/"

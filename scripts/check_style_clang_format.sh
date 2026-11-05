#!/bin/bash
#
# Copyright      2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#
# (1) To check all files in the folders listed below
#  ./scripts/check_style_clang_format.sh
#
# (2) To check the given files only
#  ./scripts/check_style_clang_format.sh path/to/a.cc path/to/b.h

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
kaldi_native_fbank_dir=$(cd $cur_dir/.. && pwd)

source $kaldi_native_fbank_dir/scripts/utils.sh

if ! command -v clang-format &> /dev/null; then
  abort "Please install clang-format first"
fi

# We always use the .clang-format in the repository root.
style=file:$kaldi_native_fbank_dir/.clang-format

# The folders to be checked. Keep them in sync with
# .github/workflows/style_check.yaml
folders="
kaldi-native-fbank/csrc
kaldi-native-fbank/python/csrc
"

# return true if the given file is a c/c++ source file
# return false otherwise
function is_source_code_file() {
  case "$1" in
    *.c|*.cc|*.h|*.cu)
      echo true;;
    *)
      echo false;;
  esac
}

function check_style() {
  # A line marked with NOLINT is allowed to stay unformatted, e.g., long
  # string literals that must not be split into pieces.
  local report
  report=$(clang-format --dry-run -Werror --style=$style "$1" 2>&1 | grep "code should be clang-formatted" || true)
  if [ -z "$report" ]; then
    return
  fi

  local bad_lines
  bad_lines=$(echo "$report" | sed 's/^[^:]*:\([0-9]*\):[0-9]*:.*$/\1/' | sort -un | while read -r n; do
    sed -n "${n}p" "$1" | grep -q NOLINT || echo "$n"
  done)

  if [ -n "$bad_lines" ]; then
    abort "$1 (lines: $(echo $bad_lines | tr '\n' ' '))"
  fi
}

function do_check() {
  if [ $# -gt 0 ]; then
    files=$@
  else
    files=$(cd $kaldi_native_fbank_dir && find $folders -type f \( -name "*.c" -o -name "*.cc" -o -name "*.h" -o -name "*.cu" \))
  fi

  for f in $files; do
    need_check=$(is_source_code_file $f)
    if $need_check; then
      [[ -f $f ]] && check_style $f
    fi
  done
}

function main() {
  do_check "$@"

  ok "Great! Style check passed!"
}

cd $kaldi_native_fbank_dir

main "$@"

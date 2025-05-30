/**
 * Copyright (c)  2022-2023  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kaldi-native-fbank/python/csrc/rfft.h"

#include <cstdint>
#include <vector>

#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

void PybindRfft(py::module &m) {  // NOLINT
  py::class_<Rfft>(m, "Rfft")
      .def(py::init<int32_t, bool>(), py::arg("n"), py::arg("inverse") = false)
      .def("compute",
           [](Rfft &self, std::vector<float> &d) -> std::vector<float> {
             self.Compute(d.data());
             return d;
           });
}

}  // namespace knf

/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/python/csrc/feature-window.h"

#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace knf {

static void PybindFrameExtractionOptions(py::module &m) {  // NOLINT
  using PyClass = FrameExtractionOptions;
  py::class_<PyClass>(m, "FrameExtractionOptions")
      .def(py::init([](float samp_freq, float frame_shift_ms,
                       float frame_length_ms, float dither, float preemph_coeff,
                       bool remove_dc_offset, const std::string &window_type,
                       bool round_to_power_of_two, float blackman_coeff,
                       bool snip_edges) {
             FrameExtractionOptions opts;
             opts.samp_freq = samp_freq;
             opts.frame_shift_ms = frame_shift_ms;
             opts.frame_length_ms = frame_length_ms;
             opts.dither = dither;
             opts.preemph_coeff = preemph_coeff;
             opts.remove_dc_offset = remove_dc_offset;
             opts.window_type = window_type;
             opts.round_to_power_of_two = round_to_power_of_two;
             opts.blackman_coeff = blackman_coeff;
             opts.snip_edges = snip_edges;
             return opts;
           }),
           py::arg("samp_freq") = 16000.0f, py::arg("frame_shift_ms") = 10.0f,
           py::arg("frame_length_ms") = 25.0f, py::arg("dither") = 0.00003f,
           py::arg("preemph_coeff") = 0.97f, py::arg("remove_dc_offset") = true,
           py::arg("window_type") = "povey",
           py::arg("round_to_power_of_two") = true,
           py::arg("blackman_coeff") = 0.42f, py::arg("snip_edges") = true)
      .def_readwrite("samp_freq", &PyClass::samp_freq)
      .def_readwrite("frame_shift_ms", &PyClass::frame_shift_ms)
      .def_readwrite("frame_length_ms", &PyClass::frame_length_ms)
      .def_readwrite("dither", &PyClass::dither)
      .def_readwrite("preemph_coeff", &PyClass::preemph_coeff)
      .def_readwrite("remove_dc_offset", &PyClass::remove_dc_offset)
      .def_readwrite("window_type", &PyClass::window_type)
      .def_readwrite("round_to_power_of_two", &PyClass::round_to_power_of_two)
      .def_readwrite("blackman_coeff", &PyClass::blackman_coeff)
      .def_readwrite("snip_edges", &PyClass::snip_edges)
      .def("as_dict",
           [](const PyClass &self) -> py::dict { return AsDict(self); })
      .def_static("from_dict",
                  [](py::dict dict) -> PyClass {
                    return FrameExtractionOptionsFromDict(dict);
                  })
#if 0
      .def_readwrite("allow_downsample",
                     &PyClass::allow_downsample)
      .def_readwrite("allow_upsample", &PyClass::allow_upsample)
#endif
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass {
            return FrameExtractionOptionsFromDict(dict);
          }));
}
static void PybindFeatureWindowFunction(py::module *m) {
  using PyClass = FeatureWindowFunction;
  py::class_<PyClass>(*m, "FeatureWindowFunction")
      .def(py::init<const FrameExtractionOptions &>(), py::arg("opts"))
      .def("apply",
           [](const PyClass &self,
              std::vector<float> &wave) -> std::vector<float> {
             self.Apply(wave.data());
             return wave;
           })
      .def_property_readonly("window", &PyClass::GetWindow);
}

void PybindFeatureWindow(py::module &m) {  // NOLINT
  PybindFrameExtractionOptions(m);
  PybindFeatureWindowFunction(&m);
}

}  // namespace knf

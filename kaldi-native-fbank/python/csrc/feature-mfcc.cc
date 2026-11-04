/**
 * Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/csrc/feature-mfcc.h"

#include <string>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/python/csrc/feature-mfcc.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace knf {

static void PybindMfccOptions(py::module &m) {  // NOLINT
  using PyClass = MfccOptions;
  py::class_<PyClass>(m, "MfccOptions")
      .def(
          py::init([](const FrameExtractionOptions &frame_opts,
                      const MelBanksOptions &mel_opts, int32_t num_ceps,
                      bool use_energy, float energy_floor, bool raw_energy,
                      float cepstral_lifter, bool htk_compat) {
            MfccOptions opts;
            opts.frame_opts = frame_opts;
            opts.mel_opts = mel_opts;
            opts.num_ceps = num_ceps;
            opts.use_energy = use_energy;
            opts.energy_floor = energy_floor;
            opts.raw_energy = raw_energy;
            opts.cepstral_lifter = cepstral_lifter;
            opts.htk_compat = htk_compat;
            return opts;
          }),
          py::arg("frame_opts") = FrameExtractionOptions{},
          py::arg("mel_opts") =
              [] {
                MelBanksOptions o;
                o.num_bins = 23;
                return o;
              }(),
          py::arg("num_ceps") = 13, py::arg("use_energy") = true,
          py::arg("energy_floor") = 0.0f, py::arg("raw_energy") = true,
          py::arg("cepstral_lifter") = 22.0f, py::arg("htk_compat") = false)
      .def_readwrite("frame_opts", &PyClass::frame_opts)
      .def_readwrite("mel_opts", &PyClass::mel_opts)
      .def_readwrite("num_ceps", &PyClass::num_ceps)
      .def_readwrite("use_energy", &PyClass::use_energy)
      .def_readwrite("energy_floor", &PyClass::energy_floor)
      .def_readwrite("raw_energy", &PyClass::raw_energy)
      .def_readwrite("cepstral_lifter", &PyClass::cepstral_lifter)
      .def_readwrite("htk_compat", &PyClass::htk_compat)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def("as_dict",
           [](const PyClass &self) -> py::dict { return AsDict(self); })
      .def_static(
          "from_dict",
          [](py::dict dict) -> PyClass { return MfccOptionsFromDict(dict); })
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass { return MfccOptionsFromDict(dict); }));
}

void PybindFeatureMfcc(py::module &m) {  // NOLINT
  PybindMfccOptions(m);
}

}  // namespace knf

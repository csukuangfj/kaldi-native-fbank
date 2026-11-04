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

#include "kaldi-native-fbank/python/csrc/feature-fbank.h"

#include <limits>
#include <string>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace knf {

static void PybindFbankOptions(py::module &m) {  // NOLINT
  using PyClass = FbankOptions;
  auto cls_FbankOptions = py::class_<PyClass>(m, "FbankOptions");
  cls_FbankOptions.attr("__module__") = "kaldi_native_fbank";
  cls_FbankOptions
      .def(py::init([](const FrameExtractionOptions &frame_opts,
                       const MelBanksOptions &mel_opts, bool use_energy,
                       float energy_floor, bool raw_energy, bool htk_compat,
                       bool use_log_fbank, bool use_log_floor, float eps,
                       bool use_power) {
             FbankOptions opts;
             opts.frame_opts = frame_opts;
             opts.mel_opts = mel_opts;
             opts.use_energy = use_energy;
             opts.energy_floor = energy_floor;
             opts.raw_energy = raw_energy;
             opts.htk_compat = htk_compat;
             opts.use_log_fbank = use_log_fbank;
             opts.use_log_floor = use_log_floor;
             opts.eps = eps;
             opts.use_power = use_power;
             return opts;
           }),
           py::arg_v("frame_opts", FrameExtractionOptions{},
                     "FrameExtractionOptions()"),
           py::arg_v(
               "mel_opts",
               [] {
                 MelBanksOptions o;
                 o.num_bins = 23;
                 return o;
               }(),
               "MelBanksOptions(num_bins=23)"),
           py::arg("use_energy") = false, py::arg_v("energy_floor", 0.0f, "0"),
           py::arg("raw_energy") = true, py::arg("htk_compat") = false,
           py::arg("use_log_fbank") = true, py::arg("use_log_floor") = true,
           py::arg_v("eps", std::numeric_limits<float>::epsilon(),
                     "1.1920929e-07"),
           py::arg("use_power") = true)
      .def_readwrite("frame_opts", &PyClass::frame_opts)
      .def_readwrite("mel_opts", &PyClass::mel_opts)
      .def_readwrite("use_energy", &PyClass::use_energy)
      .def_readwrite("energy_floor", &PyClass::energy_floor)
      .def_readwrite("raw_energy", &PyClass::raw_energy)
      .def_readwrite("htk_compat", &PyClass::htk_compat)
      .def_readwrite("use_log_fbank", &PyClass::use_log_fbank)
      .def_readwrite("use_log_floor", &PyClass::use_log_floor)
      .def_readwrite("eps", &PyClass::eps)
      .def_readwrite("use_power", &PyClass::use_power)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def("as_dict",
           [](const PyClass &self) -> py::dict { return AsDict(self); })
      .def_static(
          "from_dict",
          [](py::dict dict) -> PyClass { return FbankOptionsFromDict(dict); })
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass { return FbankOptionsFromDict(dict); }));
}

void PybindFeatureFbank(py::module &m) {  // NOLINT
  PybindFbankOptions(m);
}

}  // namespace knf

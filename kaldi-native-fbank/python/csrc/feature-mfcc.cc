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
  auto cls_MfccOptions = py::class_<PyClass>(
      m, "MfccOptions", R"doc(Options for computing MFCC features.

This is kaldi's `MfccOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-mfcc.h>`_.

.. seealso::
   :class:`FrameExtractionOptions`
      Frame extraction settings (``frame_opts``).
   :class:`MelBanksOptions`
      Mel filter-bank settings (``mel_opts``).
   :class:`kaldi_native_fbank.OnlineMfcc`
      The streaming extractor that consumes these options.)doc");
  cls_MfccOptions.attr("__module__") = "kaldi_native_fbank";
  cls_MfccOptions
      .def(py::init([](const FrameExtractionOptions &frame_opts,
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
           py::arg("num_ceps") = 13, py::arg("use_energy") = true,
           py::arg_v("energy_floor", 0.0f, "0"), py::arg("raw_energy") = true,
           py::arg_v("cepstral_lifter", 22.0f, "22"),
           py::arg("htk_compat") = false,
           R"doc(Construct with the given options.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param frame_opts: Frame-extraction options; see :class:`FrameExtractionOptions`.
:param mel_opts: Mel filter-bank options (default 23 bins); see :class:`MelBanksOptions`.
:param num_ceps: Number of cepstra in the MFCC computation (including C0).
:param use_energy: Use energy instead of C0.
:param energy_floor: Absolute floor on energy (only with use_energy; needed when dither is 0).
:param raw_energy: Compute energy before preemphasis/windowing.
:param cepstral_lifter: Liftering constant; 0 disables liftering.
:param htk_compat: Put energy/C0 last and use a factor of sqrt(2) on C0.)doc")
      .def_readwrite(
          "frame_opts", &PyClass::frame_opts,
          "Frame-extraction options; see :class:`FrameExtractionOptions`.")
      .def_readwrite("mel_opts", &PyClass::mel_opts,
                     "Mel filter-bank options; see :class:`MelBanksOptions`.  "
                     "The default has 23 mel bins.")
      .def_readwrite(
          "num_ceps", &PyClass::num_ceps,
          "Number of cepstra in the MFCC computation (including C0).")
      .def_readwrite("use_energy", &PyClass::use_energy,
                     "Use energy (not C0) in the MFCC computation.")
      .def_readwrite(
          "energy_floor", &PyClass::energy_floor,
          "Floor on energy (absolute, not relative).  Only makes a difference "
          "if ``use_energy`` is True; only necessary if ``frame_opts.dither`` "
          "is 0.  Suggested values: 0.1 or 1.0.")
      .def_readwrite(
          "raw_energy", &PyClass::raw_energy,
          "If True, compute energy before preemphasis and windowing.")
      .def_readwrite("cepstral_lifter", &PyClass::cepstral_lifter,
                     "Constant that controls scaling of the MFCCs (liftering). "
                     " If 0.0, no liftering is done.")
      .def_readwrite("htk_compat", &PyClass::htk_compat,
                     "If True, put energy/C0 last and use a factor of sqrt(2) "
                     "on C0.  Warning: not sufficient for HTK-compatible "
                     "features (other parameters must change too).")
      .def(
          "__str__",
          [](const PyClass &self) -> std::string { return self.ToString(); },
          "Return the options as a string.")
      .def(
          "as_dict",
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          "Return the options as a dict.")
      .def_static(
          "from_dict",
          [](py::dict dict) -> PyClass { return MfccOptionsFromDict(dict); },
          "Build options from a dict.")
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass { return MfccOptionsFromDict(dict); }));
}

void PybindFeatureMfcc(py::module &m) {  // NOLINT
  PybindMfccOptions(m);
}

}  // namespace knf

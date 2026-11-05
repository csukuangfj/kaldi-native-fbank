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
  auto cls_FbankOptions = py::class_<PyClass>(
      m, "FbankOptions", R"doc(Options for computing filter-bank features.

This is kaldi's `FbankOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-fbank.h>`_.  It only includes things that can be done
in a "stateless" way: it does not include energy max-normalization and it
does not include delta computation.

.. seealso::
   :class:`FrameExtractionOptions`
      Frame extraction settings (``frame_opts``).
   :class:`MelBanksOptions`
      Mel filter-bank settings (``mel_opts``).
   :class:`kaldi_native_fbank.OnlineFbank`
      The streaming extractor that consumes these options.)doc");
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
           py::arg("use_power") = true,
           R"doc(Construct with the given options.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param frame_opts: Frame-extraction options; see :class:`FrameExtractionOptions`.
:param mel_opts: Mel filter-bank options (default 23 bins); see :class:`MelBanksOptions`.
:param use_energy: Append an extra energy dimension to the filter banks.
:param energy_floor: Absolute floor on energy (only with use_energy; needed when dither is 0).
:param raw_energy: Compute energy before preemphasis/windowing (only with use_energy).
:param htk_compat: Put energy last instead of first (only with use_energy).
:param use_log_fbank: Produce log-filterbank (True) or linear (False).
:param use_log_floor: Use kaldi's hard floor (True) or the additive NeMo guard (False).
:param eps: Value of the log-zero guard (default FLT_EPSILON).
:param use_power: Use power (True) or magnitude (False) in the analysis.)doc")
      .def_readwrite(
          "frame_opts", &PyClass::frame_opts,
          "Frame-extraction options; see :class:`FrameExtractionOptions`.")
      .def_readwrite("mel_opts", &PyClass::mel_opts,
                     "Mel filter-bank options; see :class:`MelBanksOptions`.  "
                     "The default has 23 mel bins.")
      .def_readwrite(
          "use_energy", &PyClass::use_energy,
          "If True, append an extra dimension with energy to the filter banks.")
      .def_readwrite(
          "energy_floor", &PyClass::energy_floor,
          "Floor on energy (absolute, not relative).  Active only if "
          "``use_energy`` is True; only necessary if ``frame_opts.dither`` is "
          "0.  Suggested values: 0.1 or 1.0.")
      .def_readwrite(
          "raw_energy", &PyClass::raw_energy,
          "If True, compute log energy before preemphasis and windowing; if "
          "False, after.  Active only if ``use_energy`` is True.")
      .def_readwrite(
          "htk_compat", &PyClass::htk_compat,
          "If True, put energy last instead of first.  Active only if "
          "``use_energy`` is True.  Warning: not sufficient for HTK-compatible "
          "features (other parameters must change too).")
      .def_readwrite("use_log_fbank", &PyClass::use_log_fbank,
                     "If True (default), produce log-filterbank; else linear.")
      .def_readwrite(
          "use_log_floor", &PyClass::use_log_floor,
          "If True (default), use kaldi's hard floor ``log(max(e, eps))``; if "
          "False, the additive guard ``log(e + eps)`` used by NeMo models.")
      .def_readwrite(
          "eps", &PyClass::eps,
          "Value of the log-zero guard selected by ``use_log_floor``; defaults "
          "to ``FLT_EPSILON`` (NeMo models use ``FLT_EPSILON / 2``).")
      .def_readwrite("use_power", &PyClass::use_power,
                     "If True (default), use power in the filterbank analysis; "
                     "else magnitude.")
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
          [](py::dict dict) -> PyClass { return FbankOptionsFromDict(dict); },
          "Build options from a dict.")
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass { return FbankOptionsFromDict(dict); }));
}

void PybindFeatureFbank(py::module &m) {  // NOLINT
  PybindFbankOptions(m);
}

}  // namespace knf

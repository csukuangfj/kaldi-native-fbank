/**
 * Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/csrc/feature-raw-audio-samples.h"

#include <string>

#include "kaldi-native-fbank/python/csrc/feature-raw-audio-samples.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace knf {

static void PybindRawAudioSamplesOptions(py::module &m) {  // NOLINT
  using PyClass = RawAudioSamplesOptions;
  auto cls_RawAudioSamplesOptions = py::class_<PyClass>(
      m, "RawAudioSamplesOptions",
      R"doc(Options for framing raw audio samples without any processing.

The default ``frame_opts`` performs no processing on the samples:
rectangular window, no dithering, no preemphasis, no DC offset removal, no
zero-padding.

.. seealso:: :class:`kaldi_native_fbank.OnlineRawAudioSamples`)doc");
  cls_RawAudioSamplesOptions.attr("__module__") = "kaldi_native_fbank";
  cls_RawAudioSamplesOptions
      .def(py::init([](const FrameExtractionOptions &frame_opts) {
             RawAudioSamplesOptions opts;
             opts.frame_opts = frame_opts;
             return opts;
           }),
           py::arg_v(
               "frame_opts",
               [] {
                 RawAudioSamplesOptions o;
                 return o.frame_opts;
               }(),
               "RawAudioSamplesOptions().frame_opts"),
           R"doc(Construct; ``frame_opts`` defaults to un-processed framing.

The default is a rectangular window with no dithering, preemphasis, DC
offset removal or zero-padding.

:param frame_opts: Frame-extraction options; see
   :class:`FrameExtractionOptions`.)doc")
      .def_readwrite(
          "frame_opts", &PyClass::frame_opts,
          "Frame-extraction options; see :class:`FrameExtractionOptions`.")
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
          [](py::dict dict) -> PyClass {
            return RawAudioSamplesOptionsFromDict(dict);
          },
          "Build options from a dict.")
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass {
            return RawAudioSamplesOptionsFromDict(dict);
          }));
}

void PybindFeatureRawAudioSamples(py::module &m) {  // NOLINT
  PybindRawAudioSamplesOptions(m);
}

}  // namespace knf

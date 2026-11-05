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

#include "kaldi-native-fbank/python/csrc/online-feature.h"

#include <cstdint>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-mfcc.h"
#include "kaldi-native-fbank/csrc/feature-raw-audio-samples.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/whisper-feature.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {

static void PybindWhisperFeatureOptions(py::module &m) {  // NOLINT
  using PyClass = WhisperFeatureOptions;
  auto cls_WhisperFeatureOptions =
      py::class_<PyClass>(m, "WhisperFeatureOptions",
                          R"doc(Options for computing Whisper mel features.

Note that most fields of ``frame_opts`` are ignored and reset inside
WhisperFeatureComputer: 16 kHz sampling rate, 25 ms window, 10 ms shift,
"hann" window, ``snip_edges=False``, etc.

.. seealso:: :class:`kaldi_native_fbank.OnlineWhisperFbank`)doc");
  cls_WhisperFeatureOptions.attr("__module__") = "kaldi_native_fbank";
  cls_WhisperFeatureOptions
      .def(py::init<const FrameExtractionOptions &, int32_t>(),
           py::arg_v("frame_opts", FrameExtractionOptions{},
                     "FrameExtractionOptions()"),
           py::arg("dim") = 80,
           R"doc(Construct with the given frame options and number of mel bins.

:param frame_opts: Frame options (mostly ignored and reset by the computer).
:param dim: Number of mel bins: 80 for most Whisper models, 128 for
   Whisper large v3.)doc")
      .def_readwrite("frame_opts", &PyClass::frame_opts,
                     "Frame-extraction options (mostly ignored and reset by "
                     "the computer).")
      .def_readwrite("dim", &PyClass::dim,
                     "Number of mel bins: 80 for most Whisper models, 128 for "
                     "Whisper large v3.")
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
            return WhisperFeatureOptionsFromDict(dict);
          },
          "Build options from a dict.")
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass {
            return WhisperFeatureOptionsFromDict(dict);
          }));
}

template <typename C>
void PybindOnlineFeatureTpl(py::module &m,  // NOLINT
                            const std::string &class_name,
                            const std::string &class_help_doc = "") {
  using PyClass = OnlineGenericBaseFeature<C>;
  using Options = typename C::Options;
  auto cls = py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str());
  cls.attr("__module__") = "kaldi_native_fbank";
  cls.def(py::init<const Options &>(), py::arg("opts"),
          R"doc(Construct from the options of the wrapped feature computer.

:param opts: Options of the computer, e.g. :class:`FbankOptions` for
   :class:`kaldi_native_fbank.OnlineFbank`.)doc")
      .def_property_readonly("dim", &PyClass::Dim,
                             R"doc(The dimension of the output features.)doc")
      .def_property_readonly("frame_shift_in_seconds",
                             &PyClass::FrameShiftInSeconds,
                             R"doc(The frame shift in seconds.)doc")
      .def_property_readonly(
          "num_frames_ready", &PyClass::NumFramesReady,
          R"doc(The number of frames that have already been computed.)doc")
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"),
           R"doc(Return true if ``frame`` is the last frame.

Note: this can only return true after :meth:`input_finished` has been
called.

:param frame: Frame index.
:return: True if the frame is the last one.)doc")
      .def(
          "get_frame",
          [](py::object obj, int32_t frame) {
            auto *self = obj.cast<PyClass *>();
            const float *f = self->GetFrame(frame);
            return py::array_t<float>({self->Dim()},    // shape
                                      {sizeof(float)},  // stride in bytes
                                      f,                // ptr
                                      obj);  // it will increase the reference
                                             // count of **this** vector
          },
          py::arg("frame"),
          R"doc(Return the ``frame``-th feature vector.

:param frame: Frame index.
:return: np.ndarray of shape ``(dim,)`` and dtype float32.  The buffer is
   owned by the extractor.)doc")
      .def(
          "accept_waveform",
          [](PyClass &self, float sampling_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sampling_rate, waveform.data(),
                                waveform.size());
          },
          py::arg("sampling_rate"), py::arg("waveform"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Add more waveform samples.

:param sampling_rate: The sampling rate of ``waveform``; must match the
   rate in the options.
:param waveform: 1-D float32 samples.)doc")
      .def("input_finished", &PyClass::InputFinished,
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Tell the extractor that no more waveform will be added.

This flushes out the last frame or two when ``snip_edges`` is False and
also affects the return value of :meth:`is_last_frame`.)doc")
      .def("pop", &PyClass::Pop, py::arg("n"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Discard the first ``n`` frames.

:param n: Number of frames to discard.)doc");
}

void PybindOnlineFeature(py::module &m) {  // NOLINT
  PybindOnlineFeatureTpl<FbankComputer>(m, "OnlineFbank");
  PybindOnlineFeatureTpl<MfccComputer>(m, "OnlineMfcc");

  PybindWhisperFeatureOptions(m);

  PybindOnlineFeatureTpl<WhisperFeatureComputer>(m, "OnlineWhisperFbank");
  PybindOnlineFeatureTpl<RawAudioSamplesComputer>(m, "OnlineRawAudioSamples");
}

}  // namespace knf

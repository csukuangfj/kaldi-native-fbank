/**
 * Copyright (c)  2026  Julian Pscheid
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

#include "kaldi-native-fbank/python/csrc/online-nemo-fbank.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "kaldi-native-fbank/csrc/online-nemo-fbank.h"

namespace knf {
namespace {

py::dict AsDict(const NemoFbankOptions &opts) {
  py::dict ans;
  ans["samp_freq"] = opts.samp_freq;
  ans["frame_shift_ms"] = opts.frame_shift_ms;
  ans["frame_length_ms"] = opts.frame_length_ms;
  ans["num_mel_bins"] = opts.num_mel_bins;
  ans["low_freq"] = opts.low_freq;
  ans["high_freq"] = opts.high_freq;
  ans["preemph_coeff"] = opts.preemph_coeff;
  ans["buffered_capacity"] = opts.buffered_capacity;
  return ans;
}

NemoFbankOptions FromDict(py::dict dict) {
  NemoFbankOptions opts;
  if (dict.contains("samp_freq")) {
    opts.samp_freq = dict["samp_freq"].cast<float>();
  }
  if (dict.contains("frame_shift_ms")) {
    opts.frame_shift_ms = dict["frame_shift_ms"].cast<float>();
  }
  if (dict.contains("frame_length_ms")) {
    opts.frame_length_ms = dict["frame_length_ms"].cast<float>();
  }
  if (dict.contains("num_mel_bins")) {
    opts.num_mel_bins = dict["num_mel_bins"].cast<int32_t>();
  }
  if (dict.contains("low_freq")) {
    opts.low_freq = dict["low_freq"].cast<float>();
  }
  if (dict.contains("high_freq")) {
    opts.high_freq = dict["high_freq"].cast<float>();
  }
  if (dict.contains("preemph_coeff")) {
    opts.preemph_coeff = dict["preemph_coeff"].cast<float>();
  }
  if (dict.contains("buffered_capacity")) {
    opts.buffered_capacity = dict["buffered_capacity"].cast<int32_t>();
  }
  return opts;
}

}  // namespace

void PybindOnlineNemoFbank(py::module &m) {  // NOLINT
  using Options = NemoFbankOptions;
  auto cls_NemoFbankOptions = py::class_<Options>(
      m, "NemoFbankOptions", R"doc(Options for :class:`OnlineNemoFbank`.

.. seealso:: :class:`OnlineNemoFbank`)doc");
  cls_NemoFbankOptions.attr("__module__") = "kaldi_native_fbank";
  cls_NemoFbankOptions
      .def(py::init([](float samp_freq, float frame_shift_ms,
                       float frame_length_ms, int32_t num_mel_bins,
                       float low_freq, float high_freq, float preemph_coeff,
                       int32_t buffered_capacity) {
             NemoFbankOptions opts;
             opts.samp_freq = samp_freq;
             opts.frame_shift_ms = frame_shift_ms;
             opts.frame_length_ms = frame_length_ms;
             opts.num_mel_bins = num_mel_bins;
             opts.low_freq = low_freq;
             opts.high_freq = high_freq;
             opts.preemph_coeff = preemph_coeff;
             opts.buffered_capacity = buffered_capacity;
             return opts;
           }),
           py::arg("samp_freq") = 16000.0f, py::arg("frame_shift_ms") = 10.0f,
           py::arg("frame_length_ms") = 25.0f, py::arg("num_mel_bins") = 80,
           py::arg("low_freq") = 0, py::arg("high_freq") = 0,
           py::arg_v("preemph_coeff", 0.97f, "0.97"),
           py::arg("buffered_capacity") = 0,
           R"doc(Construct with the given options.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param samp_freq: Waveform sample frequency.
:param frame_shift_ms: Frame shift in milliseconds.
:param frame_length_ms: Frame length in milliseconds.
:param num_mel_bins: Number of mel bins.
:param low_freq: Lower frequency cutoff for the mel bins.
:param high_freq: Upper cutoff; non-positive values are offsets from Nyquist.
:param preemph_coeff: Preemphasis coefficient.
:param buffered_capacity: Full-buffer length in frames; 0 disables the final sample-length cap.)doc")
      .def_readwrite("samp_freq", &Options::samp_freq,
                     "Waveform sample frequency.")
      .def_readwrite("frame_shift_ms", &Options::frame_shift_ms,
                     "Frame shift in milliseconds.")
      .def_readwrite("frame_length_ms", &Options::frame_length_ms,
                     "Frame length in milliseconds.")
      .def_readwrite("num_mel_bins", &Options::num_mel_bins,
                     "Number of mel bins.")
      .def_readwrite("low_freq", &Options::low_freq,
                     "Lower frequency cutoff for the mel bins.")
      .def_readwrite("high_freq", &Options::high_freq,
                     "Upper frequency cutoff; non-positive values are offsets "
                     "from Nyquist.")
      .def_readwrite("preemph_coeff", &Options::preemph_coeff,
                     "Preemphasis coefficient.")
      .def_readwrite("buffered_capacity", &Options::buffered_capacity,
                     "Full-buffer length in frames.  0 disables the final "
                     "sample-length cap.")
      .def("__str__", &Options::ToString, "Return the options as a string.")
      .def("as_dict", &AsDict, "Return the options as a dict.")
      .def_static("from_dict", &FromDict, "Build options from a dict.")
      .def(py::pickle(&AsDict, &FromDict));

  using PyClass = OnlineNemoFbank;
  auto cls_OnlineNemoFbank = py::class_<PyClass>(
      m, "OnlineNemoFbank",
      R"doc(NeMo-compatible buffered streaming filter-bank features.

NeMo buffered streaming re-featurizes each requested audio window:
preemphasis over that window, then a centered, constant-zero-padded STFT, a
symmetric Hann window, Slaney mel filters, and
:math:`\log(\mathrm{mel} + 2^{-24})`.  Input samples are floats in
``[-1, 1]``.  This class does not resample or normalize.

.. seealso:: :class:`NemoFbankOptions`, :class:`kaldi_native_fbank.OnlineFbank`)doc");
  cls_OnlineNemoFbank.attr("__module__") = "kaldi_native_fbank";
  cls_OnlineNemoFbank
      .def(py::init<const Options &>(), py::arg("opts"),
           R"doc(Construct from the given options.

:param opts: Options; see :class:`NemoFbankOptions`.)doc")
      .def_property_readonly(
          "dim", &PyClass::Dim,
          R"doc(The feature dimension (number of mel bins).)doc")
      .def_property_readonly("frame_shift_in_seconds",
                             &PyClass::FrameShiftInSeconds,
                             R"doc(The frame shift in seconds.)doc")
      .def_property_readonly(
          "num_frames_ready", &PyClass::NumFramesReady,
          R"doc(The number of whole frames that are ready.)doc")
      .def_property_readonly(
          "num_trailing_samples", &PyClass::NumTrailingSamples,
          R"doc(Buffered samples that do not make up a whole hop.)doc")
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"),
           "Return true if frame is the last frame; only after "
           "input_finished().")
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

:param sampling_rate: The sampling rate of ``waveform``; must match
   ``opts.samp_freq``.
:param waveform: 1-D float32 samples in ``[-1, 1]``.)doc")
      .def("input_finished", &PyClass::InputFinished,
           py::call_guard<py::gil_scoped_release>(),
           "Tell the extractor that no more waveform will be added.")
      .def(
          "get_frames",
          [](PyClass &self, int32_t frame_index, int32_t n, bool is_final) {
            int32_t offset = 0;
            std::vector<float> features;
            {
              py::gil_scoped_release release;
              features = self.GetFrames(frame_index, n, is_final, &offset);
            }
            py::array_t<float> ans({n, self.Dim()});
            std::copy(features.begin(), features.end(), ans.mutable_data());
            return py::make_tuple(ans, offset);
          },
          py::arg("frame_index"), py::arg("n"), py::arg("is_final") = false,
          "Return (frames, start_sample_offset). Frames own their storage; "
          "samples before this window are discarded.",
          R"doc(Return ``n`` frames starting at the absolute ``frame_index``.

Samples preceding the requested window are discarded; later calls cannot
rewind.  If ``is_final`` reaches the last ready frame after
:meth:`input_finished`, the partial hop is included.  When ``n`` equals
``buffered_capacity``, the window is capped at ``n`` hops by shifting its
start by the remainder; that shift (in samples) is reported as the second
element of the returned tuple.

:param frame_index: Absolute index of the first frame to return.
:param n: Number of frames.
:param is_final: True if this is the final request.
:return: A tuple ``(frames, start_sample_offset)`` where ``frames`` is a
   float32 array of shape ``(n, dim)`` and ``start_sample_offset`` is 0
   unless the buffer cap shifted the window.)doc");
}

}  // namespace knf

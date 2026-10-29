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
  py::class_<Options>(m, "NemoFbankOptions")
      .def(py::init<>())
      .def_readwrite("samp_freq", &Options::samp_freq)
      .def_readwrite("frame_shift_ms", &Options::frame_shift_ms)
      .def_readwrite("frame_length_ms", &Options::frame_length_ms)
      .def_readwrite("num_mel_bins", &Options::num_mel_bins)
      .def_readwrite("low_freq", &Options::low_freq)
      .def_readwrite("high_freq", &Options::high_freq)
      .def_readwrite("preemph_coeff", &Options::preemph_coeff)
      .def_readwrite("buffered_capacity", &Options::buffered_capacity)
      .def("__str__", &Options::ToString)
      .def("as_dict", &AsDict)
      .def_static("from_dict", &FromDict)
      .def(py::pickle(&AsDict, &FromDict));

  using PyClass = OnlineNemoFbank;
  py::class_<PyClass>(m, "OnlineNemoFbank")
      .def(py::init<const Options &>(), py::arg("opts"))
      .def_property_readonly("dim", &PyClass::Dim)
      .def_property_readonly("frame_shift_in_seconds",
                             &PyClass::FrameShiftInSeconds)
      .def_property_readonly("num_frames_ready", &PyClass::NumFramesReady)
      .def_property_readonly("num_trailing_samples",
                             &PyClass::NumTrailingSamples)
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"))
      .def(
          "accept_waveform",
          [](PyClass &self, float sampling_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sampling_rate, waveform.data(),
                                waveform.size());
          },
          py::arg("sampling_rate"), py::arg("waveform"),
          py::call_guard<py::gil_scoped_release>())
      .def("input_finished", &PyClass::InputFinished,
           py::call_guard<py::gil_scoped_release>())
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
          "samples before this window are discarded.");
}

}  // namespace knf

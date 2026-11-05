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

#include "kaldi-native-fbank/csrc/stft.h"

#include <cstdint>
#include <string>
#include <vector>

#include "kaldi-native-fbank/python/csrc/stft.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {

static void PybindStftConfig(py::module *m) {
  using PyClass = StftConfig;
  auto cls_StftConfig = py::class_<PyClass>(
      *m, "StftConfig",
      R"doc(Configuration for :class:`Stft` and :class:`IStft`.

.. seealso:: :class:`Stft`, :class:`IStft`)doc");
  cls_StftConfig.attr("__module__") = "kaldi_native_fbank";
  cls_StftConfig
      .def(py::init<int32_t, int32_t, int32_t, const std::string &, bool,
                    const std::string &, bool, const std::vector<float> &>(),
           py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"),
           py::arg("window_type") = "", py::arg("center") = true,
           py::arg("pad_mode") = "reflect", py::arg("normalized") = false,
           py::arg("window") = std::vector<float>{},
           R"doc(Construct with the given parameters.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param n_fft: FFT size; should be a power of two.
:param hop_length: Number of samples between successive frames.
:param win_length: Window size in samples (must be <= n_fft).
:param window_type: Window type, e.g. "hann".
:param center: If True, pad by n_fft/2 and center the frames.
:param pad_mode: Padding mode when center is True ("reflect").
:param normalized: If True, scale the FFT by 1/sqrt(n_fft).
:param window: Explicit window coefficients (overrides window_type).)doc")
      .def_readwrite("n_fft", &PyClass::n_fft,
                     "FFT size; should be a power of two.")
      .def_readwrite("hop_length", &PyClass::hop_length,
                     "Number of samples between successive frames.")
      .def_readwrite("win_length", &PyClass::win_length,
                     "Window size in samples; must be <= n_fft.  The window is "
                     "zero-padded to n_fft.")
      .def_readwrite("n_fft", &PyClass::n_fft,
                     "FFT size; should be a power of two.")
      .def_readwrite("window_type", &PyClass::window_type,
                     "Window type, e.g. \"hann\"; ignored if explicit window "
                     "coefficients are given.")
      .def_readwrite("center", &PyClass::center,
                     "If True, pad the signal by ``n_fft // 2`` on both sides "
                     "(using ``pad_mode``) and center the frames.")
      .def_readwrite("pad_mode", &PyClass::pad_mode,
                     "Padding mode used when ``center`` is True; currently "
                     "only \"reflect\".")
      .def_readwrite("normalized", &PyClass::normalized,
                     "If True, scale the FFT by ``1 / sqrt(n_fft)``.")
      .def("__str__", &PyClass::ToString, "Return the config as a string.");
}

static void PybindStftResult(py::module *m) {
  using PyClass = StftResult;
  auto cls_StftResult = py::class_<PyClass>(
      *m, "StftResult",
      R"doc(Result of :meth:`Stft.compute`: the complex spectra of all frames.

.. seealso:: :class:`Stft`, :class:`IStft`)doc");
  cls_StftResult.attr("__module__") = "kaldi_native_fbank";
  cls_StftResult
      .def(py::init<const std::vector<float> &, const std::vector<float> &,
                    int32_t>(),
           py::arg("real"), py::arg("imag"), py::arg("num_frames"),
           R"doc(Construct a StftResult.

:param real: Real parts, shape (num_frames, n_fft // 2 + 1), flattened row-major.
:param imag: Imaginary parts, same shape.
:param num_frames: Number of frames in real/imag.)doc")
      .def_property_readonly(
          "real", [](const PyClass &self) { return self.real; },
          "Real parts of the spectra, shape (num_frames, n_fft // 2 + 1), "
          "flattened row-major.")
      .def_property_readonly(
          "imag", [](const PyClass &self) { return self.imag; },
          "Imaginary parts of the spectra, shape (num_frames, n_fft // 2 + 1), "
          "flattened row-major.")
      .def_property_readonly(
          "num_frames", [](const PyClass &self) { return self.num_frames; },
          "Number of frames in real/imag.");
}

void PybindStft(py::module *m) {
  PybindStftConfig(m);
  PybindStftResult(m);
  using PyClass = Stft;
  auto cls_Stft = py::class_<Stft>(
      *m, "Stft", R"doc(Compute the short-time Fourier transform of a waveform.

.. seealso::
   :class:`IStft`
      The inverse transform.
   :class:`StftConfig`
      Configuration.)doc");
  cls_Stft.attr("__module__") = "kaldi_native_fbank";
  cls_Stft
      .def(py::init<const StftConfig &>(), py::arg("config"),
           R"doc(Construct with the given config.

:param config: STFT configuration; see :class:`StftConfig`.)doc")
      .def(
          "compute",
          [](Stft &self, const std::vector<float> &d) -> StftResult {
            return self.Compute(d.data(), d.size());
          },
          py::arg("input"), py::call_guard<py::gil_scoped_release>(),
          R"doc(Compute the STFT of a waveform.

:param data: 1-D float32 samples.
:return: A :class:`StftResult` with the complex spectra of all frames.)doc")

      .def(
          "__call__",
          [](Stft &self, const std::vector<float> &d) -> StftResult {
            return self.Compute(d.data(), d.size());
          },
          py::arg("input"), py::call_guard<py::gil_scoped_release>());
}

}  // namespace knf

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
  auto cls_FrameExtractionOptions =
      py::class_<PyClass>(m, "FrameExtractionOptions",
                          R"doc(Options for extracting frames from a waveform.

This is the equivalent of kaldi's `FrameExtractionOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.h>`_.  It controls
how a waveform is split into overlapping frames and how each frame is
pre-processed (dithering, DC offset removal, preemphasis, windowing) before
the FFT.

.. seealso::
   :class:`FeatureWindowFunction`
      The windowing function implied by these options.
   :class:`kaldi_native_fbank.OnlineFbank`
      Streaming extractors that accept these options via ``frame_opts``.)doc");
  cls_FrameExtractionOptions.attr("__module__") = "kaldi_native_fbank";
  cls_FrameExtractionOptions
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
           py::arg("frame_length_ms") = 25.0f,
           py::arg_v("dither", 0.00003f, "0.00003"),
           py::arg_v("preemph_coeff", 0.97f, "0.97"),
           py::arg("remove_dc_offset") = true, py::arg("window_type") = "povey",
           py::arg("round_to_power_of_two") = true,
           py::arg_v("blackman_coeff", 0.42f, "0.42"),
           py::arg("snip_edges") = true,
           R"doc(Construct with the given options.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param samp_freq: Waveform sample frequency in Hz (must match the waveform).
:param frame_shift_ms: Frame shift in milliseconds.
:param frame_length_ms: Frame length in milliseconds.
:param dither: Amount of dithering; 0.0 disables it.  The default 0.00003 is equivalent to kaldi's 1.0.
:param preemph_coeff: Preemphasis coefficient.
:param remove_dc_offset: Subtract the mean of the waveform before the FFT.
:param window_type: Window type, e.g. "povey"; see :attr:`window_type` for the formulas of each type.
:param round_to_power_of_two: If True, round the FFT window size up to a power of two.
:param blackman_coeff: Constant coefficient for the generalized Blackman window.
:param snip_edges: If True, emit only frames that completely fit in the file; if False, reflect at the edges.)doc")
      .def_readwrite(
          "samp_freq", &PyClass::samp_freq,
          "Waveform data sample frequency (must match the waveform).")
      .def_readwrite("frame_shift_ms", &PyClass::frame_shift_ms,
                     "Frame shift in milliseconds.")
      .def_readwrite("frame_length_ms", &PyClass::frame_length_ms,
                     "Frame length in milliseconds.")
      .def_readwrite("dither", &PyClass::dither,
                     "Amount of dithering; 0.0 means no dither.  The default "
                     "0.00003 is equivalent to dither=1.0 in kaldi.")
      .def_readwrite("preemph_coeff", &PyClass::preemph_coeff,
                     "Preemphasis coefficient.")
      .def_readwrite("remove_dc_offset", &PyClass::remove_dc_offset,
                     "Subtract mean of the waveform before the FFT.")
      .def_readwrite(
          "window_type", &PyClass::window_type,
          R"doc(Window type: one of ``"hanning"``, ``"hann"``, ``"sine"``, ``"hamming"``, ``"povey"``, ``"rectangular"`` or ``"blackman"``.

Let :math:`N` be the window size, :math:`n = 0, \ldots, N-1` the sample index, and :math:`a = 2\pi/(N-1)`.  The formulas are:

* ``"hanning"``: :math:`w[n] = 0.5 - 0.5 \cos(a n)`
* ``"hann"``: :math:`w[n] = 0.5 - 0.5 \cos(a n)` with :math:`a = 2\pi/N`
  (periodic variant; see the hann_window docs of PyTorch)
* ``"sine"``: :math:`w[n] = \sin(0.5\,a\,n) = \sin(\pi n/(N-1))`
* ``"hamming"``: :math:`w[n] = 0.54 - 0.46 \cos(a n)`
* ``"povey"``: :math:`w[n] = (0.5 - 0.5 \cos(a n))^{0.85}`
  (like hamming but goes to zero at the edges)
* ``"rectangular"``: :math:`w[n] = 1`
* ``"blackman"``: :math:`w[n] = c - 0.5 \cos(a n) + (0.5 - c) \cos(2 a n)`
  where :math:`c` is ``blackman_coeff``.)doc")
      .def_readwrite("round_to_power_of_two", &PyClass::round_to_power_of_two,
                     "If True, round the window size to a power of two by "
                     "zero-padding the FFT input.")
      .def_readwrite(
          "blackman_coeff", &PyClass::blackman_coeff,
          "Constant coefficient for the generalized Blackman window.")
      .def_readwrite("snip_edges", &PyClass::snip_edges,
                     "If True, output only frames that completely fit in the "
                     "file (the frame count depends on the frame length).  If "
                     "False, the frame count depends only on the frame shift "
                     "and the data is reflected at the edges.")
      .def(
          "as_dict",
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          "Return the options as a dict.")
      .def_static(
          "from_dict",
          [](py::dict dict) -> PyClass {
            return FrameExtractionOptionsFromDict(dict);
          },
          "Build options from a dict.")
#if 0
      .def_readwrite("allow_downsample",
                     &PyClass::allow_downsample)
      .def_readwrite("allow_upsample", &PyClass::allow_upsample)
#endif
      .def(
          "__str__",
          [](const PyClass &self) -> std::string { return self.ToString(); },
          "Return the options as a string.")
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass {
            return FrameExtractionOptionsFromDict(dict);
          }));
}
static void PybindFeatureWindowFunction(py::module *m) {
  using PyClass = FeatureWindowFunction;
  auto cls_FeatureWindowFunction = py::class_<PyClass>(
      *m, "FeatureWindowFunction",
      "Stores the windowing coefficients and applies them to a frame.");
  cls_FeatureWindowFunction.attr("__module__") = "kaldi_native_fbank";
  cls_FeatureWindowFunction
      .def(
          py::init<const FrameExtractionOptions &>(), py::arg("opts"),
          R"doc(Construct the windowing function implied by ``opts.window_type``.

The window has ``opts.WindowSize()`` samples.

:param opts: Frame-extraction options selecting the window type and size.)doc")
      .def(
          "apply",
          [](const PyClass &self,
             std::vector<float> &wave) -> std::vector<float> {
            self.Apply(wave.data());
            return wave;
          },
          R"doc(Apply the window to a frame, in place.

``wave[i]`` becomes ``wave[i] * window[i]``.

:param wave: One frame of ``window_size`` samples.
:return: The windowed frame (the same list, modified in place).)doc")
      .def_property_readonly(
          "window", &PyClass::GetWindow,
          R"doc(The window coefficients, a list of ``window_size`` floats.)doc");
}

void PybindFeatureWindow(py::module &m) {  // NOLINT
  PybindFrameExtractionOptions(m);
  PybindFeatureWindowFunction(&m);
}

}  // namespace knf

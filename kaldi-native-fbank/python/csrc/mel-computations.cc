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

#include "kaldi-native-fbank/python/csrc/mel-computations.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

#define C_CONTIGUOUS py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_

namespace knf {

static void PybindMelBanksOptions(py::module &m) {  // NOLINT
  using PyClass = MelBanksOptions;
  auto cls_MelBanksOptions = py::class_<PyClass>(
      m, "MelBanksOptions",
      R"doc(Options for constructing triangular mel-filterbank bins.

The kaldi-compatible fields correspond to kaldi's `MelBanksOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/mel-computations.h>`_; the
``is_librosa``-related fields select librosa-compatible filters instead.

.. seealso:: :class:`MelBanks`)doc");
  cls_MelBanksOptions.attr("__module__") = "kaldi_native_fbank";
  cls_MelBanksOptions
      .def(py::init([](int32_t num_bins, float low_freq, float high_freq,
                       float vtln_low, float vtln_high, bool debug_mel,
                       bool htk_mode, bool is_librosa, const std::string &norm,
                       bool use_slaney_mel_scale, bool floor_to_int_bin) {
             MelBanksOptions opts;
             opts.num_bins = num_bins;
             opts.low_freq = low_freq;
             opts.high_freq = high_freq;
             opts.vtln_low = vtln_low;
             opts.vtln_high = vtln_high;
             opts.debug_mel = debug_mel;
             opts.htk_mode = htk_mode;
             opts.is_librosa = is_librosa;
             opts.norm = norm;
             opts.use_slaney_mel_scale = use_slaney_mel_scale;
             opts.floor_to_int_bin = floor_to_int_bin;
             return opts;
           }),
           py::arg("num_bins") = 25, py::arg_v("low_freq", 20.0f, "20"),
           py::arg_v("high_freq", 0.0f, "0"),
           py::arg_v("vtln_low", 100.0f, "100"),
           py::arg_v("vtln_high", -500.0f, "-500"),
           py::arg("debug_mel") = false, py::arg("htk_mode") = false,
           py::arg("is_librosa") = false, py::arg("norm") = "slaney",
           py::arg("use_slaney_mel_scale") = true,
           py::arg("floor_to_int_bin") = false,
           R"doc(Construct with the given options.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param num_bins: Number of triangular mel-frequency bins.
:param low_freq: Lower frequency cutoff.
:param high_freq: Upper cutoff; 0 means none, negative values are offsets from Nyquist.
:param vtln_low: VTLN lower cutoff of the warping function.
:param vtln_high: VTLN upper cutoff of the warping function; negative values are offsets from Nyquist.
:param debug_mel: Print debugging information for mel bin computation.
:param htk_mode: Exact HTK compatibility (affects mel-energy flooring).
:param is_librosa: Use librosa-style mel filters.
:param norm: Librosa filter normalization ("" or "slaney").
:param use_slaney_mel_scale: Use the Slaney mel scale (librosa mode only).
:param floor_to_int_bin: Floor the bin edges to integers (librosa mode only).)doc")
      .def_readwrite("num_bins", &PyClass::num_bins,
                     "Number of triangular mel-frequency bins.")
      .def_readwrite("low_freq", &PyClass::low_freq,
                     R"doc(Lower frequency cutoff of the mel bins.

See also :attr:`high_freq`.)doc")
      .def_readwrite("high_freq", &PyClass::high_freq,
                     "Upper frequency cutoff; 0 means no cutoff, negative "
                     "values are offsets from Nyquist.")
      .def_readwrite("vtln_low", &PyClass::vtln_low,
                     "VTLN lower cutoff of the warping function.")
      .def_readwrite("vtln_high", &PyClass::vtln_high,
                     "VTLN upper cutoff of the warping function; negative "
                     "values are offsets from Nyquist.")
      .def_readwrite("debug_mel", &PyClass::debug_mel,
                     "Print debugging information for mel bin computation.")
      .def_readwrite(
          "htk_mode", &PyClass::htk_mode,
          "Enable exact HTK compatibility (affects mel-energy flooring).")
      .def_readwrite("is_librosa", &PyClass::is_librosa,
                     R"doc(Use librosa-style mel filters.

If True, you probably want :attr:`low_freq` to be 0.  See the librosa
mel filter documentation.)doc")
      .def_readwrite("norm", &PyClass::norm,
                     "Librosa filter normalization: \"\" or \"slaney\" (used "
                     "only when is_librosa=True).")
      .def_readwrite(
          "use_slaney_mel_scale", &PyClass::use_slaney_mel_scale,
          "Use the Slaney mel scale (used only when is_librosa=True).")
      .def_readwrite(
          "floor_to_int_bin", &PyClass::floor_to_int_bin,
          "Floor the bin edges to integers (used only when is_librosa=True).")
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
            return MelBanksOptionsFromDict(dict);
          },
          "Build options from a dict.")
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass {
            return MelBanksOptionsFromDict(dict);
          }));
}

void PybindMelComputations(py::module &m) {  // NOLINT
  PybindMelBanksOptions(m);
  using PyClass = MelBanks;
  auto cls_MelBanks = py::class_<PyClass>(m, "MelBanks",
                                          R"doc(Triangular mel filter banks.

A ``MelBanks`` object holds the weights of a set of triangular mel-frequency
bins and can project FFT bin energies onto them.

.. seealso:: :class:`MelBanksOptions`)doc");
  cls_MelBanks.attr("__module__") = "kaldi_native_fbank";
  cls_MelBanks
      .def(py::init<const MelBanksOptions &, const FrameExtractionOptions &,
                    float>(),
           py::arg_v("opts", MelBanksOptions{}, "MelBanksOptions()"),
           py::arg_v("frame_opts", FrameExtractionOptions{},
                     "FrameExtractionOptions()"),
           py::arg("vtln_warp_factor") = 1.0,
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Construct kaldi-compatible (or librosa-style) mel banks.

The mel bin edges are placed uniformly on the mel scale between
``opts.low_freq`` and ``opts.high_freq``; each bin is a triangle in the
frequency domain.  If ``opts.is_librosa`` is true, librosa-style filters are
computed instead.

:param opts: Mel-bin configuration; see :class:`MelBanksOptions`.
:param frame_opts: Frame-extraction configuration (provides ``samp_freq``
   and the FFT size).
:param vtln_warp_factor: VTLN warping factor (1.0 means no warping).)doc")
      .def(
          "get_matrix",
          [](const PyClass &self) -> py::array_t<float> {
            std::vector<float> matrix = self.GetMatrix();
            int32_t num_rows = self.NumBins();
            int32_t num_cols = matrix.size() / num_rows;

            py::array_t<float> ans({num_rows, num_cols});
            py::buffer_info buf = ans.request();
            auto p = static_cast<float *>(buf.ptr);
            std::copy(matrix.begin(), matrix.end(), p);
            return ans;
          },
          R"doc(Return the mel filter weights.

:return: A float32 array of shape ``(num_mel_bins, num_fft_bins // 2 + 1)``;
   ``weights[i, k]`` is the weight of fft bin :math:`k` for mel bin :math:`i`.)doc")
      .def(
          "compute",
          [](const PyClass &self, const py::array_t<float> &fft_energies) {
            if (!(C_CONTIGUOUS == (fft_energies.flags() & C_CONTIGUOUS))) {
              throw py::value_error(
                  "input fft_energies should be contiguous. Please use "
                  "np.ascontiguousarray(fft_energies)");
            }

            int num_dim = fft_energies.ndim();
            if (num_dim != 1) {
              std::ostringstream os;
              os << "Expect an array of 1 dimension (num_fft_bins/2+1,)"
                    "Given dim: "
                 << num_dim << "\n";
              throw py::value_error(os.str());
            }

            py::array_t<float> ans(self.NumBins());

            py::buffer_info buf = ans.request();
            auto p = static_cast<float *>(buf.ptr);
            self.Compute(fft_energies.data(), p);

            return ans;
          },
          py::arg("fft_energies"),
          R"doc(Compute mel energies (note: NOT log) from a power spectrum.

For each mel bin :math:`i`, the output is the weighted sum of the FFT bin
energies that fall into that bin:

.. math::

   \mathrm{mel}[i] = \sum_{k} w[i][k] \cdot
   \mathrm{fft\_energies}[\mathrm{offset}[i] + k]

where :math:`\mathrm{offset}[i]` is the first nonzero fft bin of mel bin
:math:`i` and :math:`w[i]` are its triangle weights.

:param fft_energies: 1-D float32 array of size ``num_fft_bins // 2 + 1``
   holding the power spectrum (not log).  Must be C-contiguous.
:return: 1-D float32 array of size ``num_mel_bins`` with the mel energies.)doc")
      .def_property_readonly("dim", &PyClass::NumBins)
      .def_static("inverse_mel_scale", &PyClass::InverseMelScale,
                  py::arg("mel"),
                  R"doc(HTK mel scale: convert mel frequency to Hz.

.. math::

   \mathrm{Hz} = 700 \left( \exp(\mathrm{mel} / 1127) - 1 \right)

:param mel: Mel frequency.
:return: The frequency in Hz.)doc")
      .def_static("mel_scale", &PyClass::MelScale, py::arg("hz"),
                  R"doc(HTK mel scale: convert Hz to mel frequency.

.. math::

   \mathrm{mel} = 1127 \log(1 + \mathrm{Hz} / 700)

:param hz: The frequency in Hz.
:return: The mel frequency.)doc")
      .def_static("inverse_mel_scale_slaney", &PyClass::InverseMelScaleSlaney,
                  py::arg("mel"),
                  R"doc(Slaney mel scale: convert mel frequency to Hz.

.. math::

   \mathrm{Hz} = \left\{ \begin{array}{ll}
   (200/3)\,\mathrm{mel} & \mathrm{mel} \le 15 \\
   1000 \exp((\mathrm{mel} - 15) \log(6.4) / 27) & \mathrm{otherwise}
   \end{array} \right.

:param mel: Mel frequency (Slaney scale).
:return: The frequency in Hz.)doc")
      .def_static("mel_scale_slaney", &PyClass::MelScaleSlaney, py::arg("hz"),
                  R"doc(Slaney mel scale: convert Hz to mel frequency.

.. math::

   \mathrm{mel} = \left\{ \begin{array}{ll}
   (3/200)\,\mathrm{Hz} & \mathrm{Hz} \le 1000 \\
   15 + 27 \log(\mathrm{Hz} / 1000) / \log(6.4) & \mathrm{otherwise}
   \end{array} \right.

:param hz: The frequency in Hz.
:return: The mel frequency (Slaney scale).)doc");
}

}  // namespace knf

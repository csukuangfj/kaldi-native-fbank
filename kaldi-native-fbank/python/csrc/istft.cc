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

#include "kaldi-native-fbank/python/csrc/istft.h"

#include "kaldi-native-fbank/csrc/istft.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {
struct StftConfig;

void PybindIStft(py::module *m) {
  using PyClass = IStft;
  auto cls_IStft = py::class_<IStft>(
      *m, "IStft", R"doc(Reconstruct a waveform from :class:`Stft` results.

This is the inverse of :class:`Stft` (compatible with librosa's istft()):
the frames are inverse-transformed, windowed, overlap-added, and divided by
the summed window envelope.

.. seealso:: :class:`Stft`, :class:`StftConfig`)doc");
  cls_IStft.attr("__module__") = "kaldi_native_fbank";
  cls_IStft
      .def(
          py::init<const StftConfig &>(), py::arg("config"),
          R"doc(Construct with the same config that was used for the forward :class:`Stft`.

:param config: The config used for the forward transform.)doc")
      .def("compute", &PyClass::Compute, py::arg("stft_result"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Reconstruct the waveform.

:param stft_result: Output of :meth:`Stft.compute`.
:return: 1-D list of samples.  If ``config.center`` is True, the
   ``n_fft // 2`` padding samples at both ends are removed.)doc")
      .def("__call__", &PyClass::Compute, py::arg("stft_result"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Same as :meth:`compute`.)doc");
}

}  // namespace knf

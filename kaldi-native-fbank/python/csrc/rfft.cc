/**
 * Copyright (c)  2022-2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/python/csrc/rfft.h"

#include <cstdint>
#include <vector>

#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

void PybindRfft(py::module &m) {  // NOLINT
  auto cls_Rfft = py::class_<Rfft>(
      m, "Rfft", R"doc(n-point real discrete Fourier transform, where n is even.

Forward transform:

.. math::

   R[k] = \sum_{j=0}^{n-1} \mathrm{in}[j] \cos(2\pi j k / n),
   \quad 0 \le k \le n/2

   I[k] = \sum_{j=0}^{n-1} \mathrm{in}[j] \sin(2\pi j k / n),
   \quad 0 < k < n/2

.. seealso:: :class:`Stft`)doc");
  cls_Rfft.attr("__module__") = "kaldi_native_fbank";
  cls_Rfft
      .def(py::init<int32_t, bool>(), py::arg("n"), py::arg("inverse") = false,
           R"doc(Construct for an n-point transform.

:param n: Number of fft points; must be even and >= 2.
:param inverse: If True, :meth:`compute` performs the inverse transform.)doc")
      .def(
          "compute",
          [](Rfft &self, std::vector<float> &d) -> std::vector<float> {
            self.Compute(d.data());
            return d;
          },
          R"doc(Transform in place and return the packed spectrum.

On return of the forward transform:

* ``out[0] = R[0]``
* ``out[1] = R[n/2]``
* for ``1 < k < n/2``: ``out[2*k] = R[k]``, ``out[2*k+1] = I[k]``

:param d: 1-D list of n floats.
:return: The packed spectrum, a 1-D list of n floats.)doc");
}

}  // namespace knf

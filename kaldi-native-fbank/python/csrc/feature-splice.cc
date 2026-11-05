/**
 * Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/python/csrc/feature-splice.h"

#include <pybind11/eigen.h>

#include <cstdint>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/feature-splice.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {

void PybindFeatureSplice(py::module &m) {  // NOLINT
  auto cls_SpliceOptions =
      py::class_<SpliceOptions>(m, "SpliceOptions",
                                R"doc(Options for :func:`splice_frames`.

Mirrors the flags of kaldi's `splice-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/splice-feats.cc>`_.

.. seealso:: :func:`splice_frames`)doc");
  cls_SpliceOptions.attr("__module__") = "kaldi_native_fbank";
  cls_SpliceOptions
      .def(py::init([](int32_t left_context, int32_t right_context) {
             SpliceOptions opts;
             opts.left_context = left_context;
             opts.right_context = right_context;
             return opts;
           }),
           py::arg("left_context") = 3, py::arg("right_context") = 3,
           R"doc(Construct with the given contexts.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param left_context: Number of preceding frames to append
   (kaldi ``splice-feats --left-context``).
:param right_context: Number of following frames to append
   (kaldi ``splice-feats --right-context``).)doc")
      .def_readwrite("left_context", &SpliceOptions::left_context,
                     R"doc(Number of preceding frames to append.

Same as kaldi ``splice-feats --left-context``.)doc")
      .def_readwrite("right_context", &SpliceOptions::right_context,
                     R"doc(Number of following frames to append.

Same as kaldi ``splice-feats --right-context``.)doc");

  m.def(
      "splice_frames",
      [](const MatrixF &feats, const SpliceOptions &opts) -> MatrixF {
        return SpliceFrames(feats, opts);
      },
      py::arg("feats"), py::arg("opts") = SpliceOptions{},
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Splice each frame with its left/right context neighbors.

This is kaldi's `splice-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/splice-feats.cc>`_
(same arithmetic): for every frame index :math:`t` the output row is

.. math::

   y_t = [ x_{\max(0, t-L)}, \ldots, x_{\min(T-1, t+R)} ]

where :math:`L` is ``opts.left_context``, :math:`R` is
``opts.right_context`` and :math:`T` is the number of input frames.  Edge
frames are replicated: out-of-range indices clamp to the first/last frame.
The output therefore has the same number of frames as the input and
``dim * (1 + L + R)`` dimensions.

:param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
:type feats: numpy.ndarray
:param opts: Contexts; see :class:`SpliceOptions`.
:return: The spliced matrix of shape ``(num_frames, dim * (1 + L + R))``,
   float32.
:rtype: numpy.ndarray

Example
-------

.. code-block:: python

   opts = knf.SpliceOptions(left_context=3, right_context=3)
   spliced = knf.splice_frames(feats, opts)  # (T, 7 * dim)

.. seealso:: :class:`SpliceOptions`)doc");
}

}  // namespace knf

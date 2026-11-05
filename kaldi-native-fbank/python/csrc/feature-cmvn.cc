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

#include "kaldi-native-fbank/python/csrc/feature-cmvn.h"

#include <pybind11/eigen.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-cmvn.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {

void PybindFeatureCmvn(py::module &m) {  // NOLINT
  auto cls_CmvnOptions = py::class_<CmvnOptions>(
      m, "CmvnOptions",
      R"doc(Options matching kaldi's `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_
flags.

The flags control how :func:`apply_cmvn` normalizes features.

.. seealso:: :func:`apply_cmvn`
   For the normalization formulas.)doc");
  cls_CmvnOptions.attr("__module__") = "kaldi_native_fbank";
  cls_CmvnOptions
      .def(py::init([](bool norm_vars, bool reverse) {
             CmvnOptions opts;
             opts.norm_vars = norm_vars;
             opts.reverse = reverse;
             return opts;
           }),
           py::arg("norm_vars") = false, py::arg("reverse") = false,
           R"doc(Construct with the given flags.

:param norm_vars: If ``True``, normalize the variances as well
   (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--norm-vars``).
:param reverse: If ``True``, undo cmvn instead of applying it
   (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--reverse``).)doc")
      .def_readwrite("norm_vars", &CmvnOptions::norm_vars,
                     R"doc(If ``True``, normalize the variances as well: after
subtracting the mean, each dimension is divided by its standard deviation.
Requires 2-row stats (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--norm-vars``).)doc")
      .def_readwrite("reverse", &CmvnOptions::reverse,
                     R"doc(If ``True``, undo cmvn instead of applying it: the
mean is added back and, if ``norm_vars`` is ``True``, the standard deviation is
multiplied back (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--reverse``).)doc")
      .def(
          "__str__",
          [](const CmvnOptions &opts) -> std::string {
            std::ostringstream os;
            os << "CmvnOptions(norm_vars="
               << (opts.norm_vars ? "True" : "False")
               << ", reverse=" << (opts.reverse ? "True" : "False") << ")";
            return os.str();
          },
          "Return the options as a string.");

  m.def("compute_cmvn_stats", &ComputeCmvnStats, py::arg("feats"),
        py::call_guard<py::gil_scoped_release>(),
        R"doc(Compute full 2-row cmvn stats for a feature matrix.

Equivalent to initializing zero stats and accumulating every row of ``feats``
with weight 1.0 (kaldi `AccCmvnStats <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_).

The returned stats matrix matches kaldi's ``cmvn.ark`` layout exactly:

.. math::

   \mathrm{stats} =
   \left( \begin{array}{cccc|c}
     S_1 & S_2 & \cdots & S_D & c \\
     Q_1 & Q_2 & \cdots & Q_D & -
   \end{array} \right)

* Row 0, columns ``0..D-1`` hold the sums
  :math:`S_d = \sum_t x_t(d)`; the last column :math:`c` holds the frame
  count (sum of weights).
* Row 1, columns ``0..D-1`` hold the sums of squares
  :math:`Q_d = \sum_t x_t(d)^2`; the last column is unused.

:param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
:type feats: numpy.ndarray
:return: Stats matrix of shape ``(2, dim + 1)``, float64.
:rtype: numpy.ndarray

.. seealso::
   :func:`apply_cmvn`
      Apply these stats to features.
   :class:`CmvnAccumulator`
      Incremental version of this computation.)doc");

  m.def(
      "apply_cmvn",
      [](const MatrixD &stats, const CmvnOptions &opts, const MatrixF &feats)
          -> MatrixF { return ApplyCmvn(stats, opts, feats); },
      py::arg("stats"), py::arg("opts"), py::arg("feats"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Apply cmvn to features and return the normalized copy.

This follows kaldi's `ApplyCmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_ / `ApplyCmvnReverse <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_ semantics.
Let :math:`c` be the frame count,
:math:`\mu_d = \mathrm{stats}(0, d)/c` the mean and, when ``opts.norm_vars``
is true, :math:`\sigma_d^2 = \mathrm{stats}(1, d)/c - \mu_d^2` the variance
of dimension :math:`d`.

Forward (``opts.reverse == False``), applied to every frame :math:`x`:

.. math::

   x_d \leftarrow x_d - \mu_d

and, if ``opts.norm_vars``:

.. math::

   x_d \leftarrow (x_d - \mu_d) / \sigma_d

Reverse (``opts.reverse == True``) undoes the normalization:

.. math::

   x_d \leftarrow x_d + \mu_d

and, if ``opts.norm_vars``:

.. math::

   x_d \leftarrow x_d \cdot \sigma_d + \mu_d

1-row (mean-only) stats are accepted when ``norm_vars`` is false; requesting
variance normalization with 1-row stats raises an exception (same as kaldi).

:param stats: 1-row or 2-row cmvn stats, e.g. from
   :func:`compute_cmvn_stats`.
:param opts: Options controlling the normalization; see :class:`CmvnOptions`.
:param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
:type feats: numpy.ndarray
:return: A normalized copy of ``feats``.
:rtype: numpy.ndarray

Example
-------

.. code-block:: python

   stats = knf.compute_cmvn_stats(feats)
   opts = knf.CmvnOptions()  # norm_vars=False, reverse=False
   normed = knf.apply_cmvn(stats, opts, feats)

.. seealso:: :func:`apply_cmvn_`, :func:`compute_cmvn_stats`,
   :class:`CmvnAccumulator`)doc");

  m.def(
      "apply_cmvn_",
      [](const MatrixD &stats, const CmvnOptions &opts,
         Eigen::Ref<MatrixF> feats) { ApplyCmvn(stats, opts, feats); },
      py::arg("stats"), py::arg("opts"), py::arg("feats"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(In-place version of :func:`apply_cmvn`.

``feats`` is modified directly; no copy of the feature matrix is made.

:param stats: 1-row or 2-row cmvn stats; see :func:`compute_cmvn_stats`.
:param opts: Options controlling the normalization; see :class:`CmvnOptions`.
:param feats: Feature matrix of shape ``(num_frames, dim)``.  Must be a
   writeable, C-contiguous float32 ``numpy.ndarray``; it is normalized
   in place.

.. seealso:: :func:`apply_cmvn`
   For the normalization formulas.)doc");

  auto cls_CmvnAccumulator = py::class_<CmvnAccumulator>(
      m, "CmvnAccumulator",
      R"doc(Incremental accumulator for speaker-level / streaming cmvn.

It always keeps the full 2-row layout (see :func:`compute_cmvn_stats`) and
adds up weighted frames one by one, so stats can be collected over many
utterances of the same speaker and normalized later with
:func:`apply_cmvn`.

.. seealso:: :func:`compute_cmvn_stats`, :func:`apply_cmvn`)doc");

  cls_CmvnAccumulator.attr("__module__") = "kaldi_native_fbank";

  cls_CmvnAccumulator
      .def(py::init<int32_t>(), py::arg("dim") = 0,
           R"doc(Construct an accumulator for features of dimension ``dim``.

:param dim: Feature dimension.  ``0`` (the default) creates an empty
   accumulator whose stats are grown by the first :meth:`add` or
   :meth:`accumulate` call.)doc")
      .def("accumulate", &CmvnAccumulator::Accumulate, py::arg("feats"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Accumulate stats for a feature matrix (rows = frames).

Equivalent to calling :meth:`add` for each row with weight 1.0.

:param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
:type feats: numpy.ndarray)doc")
      .def(
          "add",
          [](CmvnAccumulator &self, int32_t dim, std::vector<float> frame,
             float weight) { self.Add(dim, frame.data(), weight); },
          py::arg("dim"), py::arg("frame"), py::arg("weight") = 1.0f,
          R"doc(Accumulate stats for one frame of dimension ``dim``.

Applies the same updates as kaldi's `AccCmvnStats <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_; with
:math:`w = \mathrm{weight}` and :math:`x_d = \mathrm{frame}[d]`:

.. math::

   c \mathrel{+}= w, \qquad
   S_d \mathrel{+}= x_d\, w, \qquad
   Q_d \mathrel{+}= x_d^2\, w

:param dim: Feature dimension; must match the dimension given to the
   constructor.
:param frame: One frame of shape ``(dim,)``.
:type frame: Sequence[float]
:param weight: Weight of this frame, e.g. 1.0.)doc")
      .def_property_readonly(
          "stats",
          [](const CmvnAccumulator &self) -> MatrixD { return self.Stats(); },
          R"doc(The accumulated 2-row stats.

Row 0 is ``[sums, count]``, row 1 is ``[sumsq, (unused)]``; see
:func:`compute_cmvn_stats` for the layout.

:rtype: numpy.ndarray)doc");
}

}  // namespace knf

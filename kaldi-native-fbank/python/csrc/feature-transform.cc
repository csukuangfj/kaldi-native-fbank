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

#include "kaldi-native-fbank/python/csrc/feature-transform.h"

#include <pybind11/eigen.h>

#include <cstdint>
#include <string>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/feature-transform.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {

void PybindFeatureTransform(py::module &m) {  // NOLINT
  auto cls_FeatureTransform = py::class_<FeatureTransform>(
      m, "FeatureTransform",
      R"doc(A linear (or affine) feature transform, e.g. an LDA matrix.

This is kaldi's `transform-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/transform-feats.cc>`_
(same semantics).  The transform is a matrix :math:`M` of shape
:math:`(D_{out} \times n)`:

* if :math:`n` equals the input feature dimension, the transform is
  linear: :math:`y = x\, M^{T}`;
* if :math:`n` is one larger, the transform is affine and the last column
  of :math:`M` is the offset:
  :math:`y = x\, M_{[:,\,0:n-1]}^{T} + M_{[:,\,n-1]}`;
* otherwise :meth:`apply` raises an exception.

Example
-------

.. code-block:: python

   lda = knf.FeatureTransform.read("lda.mat")
   out = lda.apply(feats)            # (T, lda.output_dim)
   out = knf.transform_features(feats, lda)

.. seealso:: :func:`transform_features`)doc");
  cls_FeatureTransform.attr("__module__") = "kaldi_native_fbank";
  cls_FeatureTransform
      .def(py::init<const MatrixF &>(), py::arg("mat"),
           R"doc(Construct a transform from a matrix directly.

:param mat: The transform matrix of shape ``(output_dim, input_dim)``, or
   one column wider for an affine transform (the last column is the
   offset).)doc")
      .def_static(
          "read",
          [](const std::string &filename) -> FeatureTransform {
            return FeatureTransform::Read(filename);
          },
          py::arg("filename"), py::call_guard<py::gil_scoped_release>(),
          R"doc(Read a transform matrix from a kaldi matrix file.

Supports the kaldi binary format (``"FM"``/``"DM"``, as written by kaldi's
``Matrix::Write``) and the kaldi text format (``[ ... ]``).

:param filename: Path of the matrix file, e.g. ``lda.mat``.
:return: The transform.)doc")
      .def_property_readonly("input_dim", &FeatureTransform::InputDim,
                             R"doc(Number of columns of the stored matrix.

This is the feature dimension in the linear form; an affine transform
consumes ``input_dim - 1`` features plus the implicit 1.0.)doc")
      .def_property_readonly(
          "output_dim", &FeatureTransform::OutputDim,
          R"doc(Number of rows of the stored matrix (the output dimension).)doc")
      .def(
          "is_affine",
          [](const FeatureTransform &self, int32_t input_dim) -> bool {
            return self.IsAffine(input_dim);
          },
          py::arg("input_dim"),
          R"doc(Whether the transform is affine for inputs of dimension ``input_dim``.

The matrix alone cannot record the intended input dimension, so the rule
is the one kaldi's ``transform-feats`` applies: the transform is affine iff
it has one column more than the feature dimension (the extra column is the
offset).

:param input_dim: The input feature dimension.
:return: True for an affine transform.)doc")
      .def(
          "apply",
          [](const FeatureTransform &self, const MatrixF &feats) -> MatrixF {
            return self.Apply(feats);
          },
          py::arg("feats"), py::call_guard<py::gil_scoped_release>(),
          R"doc(Apply the transform to a feature matrix.

kaldi ``transform-feats`` semantics (rows = frames): if the transform has
``feats.shape[1]`` columns the output is :math:`Y = X M^{T}` (linear); if it
has one column more, the transform is affine and the last column is the
offset.  Otherwise an exception is raised.

:param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
:type feats: numpy.ndarray
:return: The transformed matrix of shape ``(num_frames, output_dim)``,
   float32.
:rtype: numpy.ndarray)doc");

  m.def(
      "transform_features",
      [](const MatrixF &feats, const FeatureTransform &transform) -> MatrixF {
        return transform.Apply(feats);
      },
      py::arg("feats"), py::arg("transform"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Apply a transform to features and return the result.

Shorthand for ``transform.apply(feats)``; matches kaldi's
`transform-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/transform-feats.cc>`_.

:param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
:type feats: numpy.ndarray
:param transform: The transform; see :class:`FeatureTransform`.
:return: The transformed matrix of shape ``(num_frames, transform.output_dim)``.
:rtype: numpy.ndarray

.. seealso:: :class:`FeatureTransform`)doc");
}

}  // namespace knf

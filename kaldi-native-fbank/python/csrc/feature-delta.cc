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

#include "kaldi-native-fbank/python/csrc/feature-delta.h"

#include <pybind11/eigen.h>

#include <cstdint>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-delta.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {

void PybindFeatureDelta(py::module &m) {  // NOLINT
  auto cls_DeltaOptions =
      py::class_<DeltaOptions>(m, "DeltaOptions",
                               R"doc(Options for :func:`compute_deltas`.

Mirrors the flags of kaldi's `add-deltas <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/add-deltas.cc>`_.

.. seealso:: :func:`compute_deltas`)doc");
  cls_DeltaOptions.attr("__module__") = "kaldi_native_fbank";
  cls_DeltaOptions
      .def(py::init([](int32_t order, int32_t window) {
             DeltaOptions opts;
             opts.order = order;
             opts.window = window;
             return opts;
           }),
           py::arg("order") = 2, py::arg("window") = 2,
           R"doc(Construct with the given order and window.

All fields are optional keyword arguments; the defaults are shown in the
signature.

:param order: Order of the delta computation (kaldi ``add-deltas --delta-order``).
:param window: Delta regression window (kaldi ``add-deltas --delta-window``).)doc")
      .def_readwrite("order", &DeltaOptions::order,
                     R"doc(Order of the delta computation.

Same as kaldi ``add-deltas --delta-order``.  2 gives ``[x, delta,
delta-delta]``.)doc")
      .def_readwrite("window", &DeltaOptions::window,
                     R"doc(Delta regression window.

Same as kaldi ``add-deltas --delta-window``.  The actual regression spans
``2 * window + 1`` frames.)doc");

  m.def(
      "compute_deltas",
      [](const DeltaOptions &opts, const MatrixF &feats) -> MatrixF {
        return ComputeDeltas(opts, feats);
      },
      py::arg("opts"), py::arg("feats"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Compute delta (dynamic) features.

This is kaldi's `add-deltas <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/add-deltas.cc>`_
(same arithmetic).  The :math:`i`-th block of the output is the :math:`i`-th
order regression delta; with :math:`N` = ``opts.window``, the first-order
delta of frame :math:`t` is

.. math::

   \Delta_t = \frac{\sum_{j=-N}^{N} j\, x_{\mathrm{clamp}(t+j)}}
                   {2 \sum_{j=1}^{N} j^2}

where out-of-range frames clamp to the first/last frame (edge frames are
replicated).  Higher orders apply the same operator recursively to the
previous order's window.  The output rows are
:math:`[x_t, \Delta_t, \Delta^2_t, \ldots]`, i.e. the block order is
baseline first, then increasing delta order.

The regression windows themselves are computed as follows.  With
:math:`N` = ``opts.window``, the first-order window is

.. math::

   w^{(1)}_j = \frac{j}{\sum_{k=-N}^{N} k^2}
             = \frac{j}{2 \sum_{k=1}^{N} k^2},
   \qquad j = -N, \ldots, N

and each higher-order window is the discrete convolution of the previous
window with :math:`w^{(1)}` (kaldi's ``scales_`` recursion):

.. math::

   w^{(i)} = w^{(i-1)} * w^{(1)}, \qquad w^{(0)} = [1]

so :math:`w^{(i)}` is the :math:`i`-fold convolution of :math:`w^{(1)}`
(its length is ``1 + 2 * i * window``).

**Motivation.**  The spectrum of speech is strongly non-stationary: the
*trajectory* of the spectrum over time carries information that the
static spectrum does not.  Furui (1986) showed that appending the time
derivative of the spectrum (the "delta") significantly improves
recognition.  HTK refined this into the regression-based formulation
used here: instead of a raw difference such as
:math:`x_{t+1} - x_{t-1}`, the local slope of a least-squares line is
estimated over a window of :math:`2N+1` frames, which is less sensitive
to noise and to isolated outliers.  kaldi's ``DeltaFeatures`` implements
the same HTK formulation, so the outputs are compatible with models
trained on HTK- or kaldi-produced features.

**Why this definition.**  The first-order window is the slope of the
least-squares straight line fitted to the frames around :math:`t` — the
classic HTK regression delta that kaldi's ``DeltaFeatures`` implements.
Fitting :math:`y(j) = a + b\, j` to :math:`x_{t+j}`,
:math:`j = -N, \ldots, N`, by minimizing
:math:`\sum_j (x_{t+j} - a - b\, j)^2`, and using
:math:`\sum_j j = 0` (which decouples the intercept from the slope), the
slope estimate is

.. math::

   b = \frac{\sum_j j\, x_{t+j}}{\sum_j j^2}
     = \sum_j w^{(1)}_j\, x_{t+j}

so :math:`w^{(1)}_j = j / \sum_k k^2` is exactly the least-squares slope
estimator: :math:`\Delta_t = b` is the local slope of the fitted line,
i.e. a least-squares estimate of the first derivative (in feature units
per frame — the :math:`\sum_j j^2` normalizer is the OLS denominator).

Higher orders apply the *same* slope estimator to the previous order's
output (:math:`\Delta^2` is "the slope of the slope").  Since the slope
estimator is a linear filter with kernel :math:`w^{(1)}`, filtering an
already-filtered sequence convolves the kernels, hence
:math:`w^{(i)} = w^{(i-1)} * w^{(1)}` with support growing to
:math:`\pm i N`.  Note this is *recursive* slope-of-slope — it is not
identical to the :math:`i`-th derivative of a degree-:math:`i`
least-squares polynomial fitted over the wider :math:`\pm iN` window.

Two further details fall out of the construction:

* Normalizing by :math:`\sum_j j^2` at every order keeps each block on
  the scale of a true :math:`i`-th derivative.
* Edge frames replicate the first/last frame so the frame count is
  preserved and the regression "sees" a locally constant signal at the
  borders (instead of zero-padding, which would bias the deltas).

**References.**

* The ported implementation: kaldi's ``DeltaFeatures``
  (`feature-functions.cc <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-functions.cc>`_,
  `feature-functions.h <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-functions.h>`_).
* The regression-delta formulation: *The HTK Book*, S. Young et al.,
  Cambridge University Engineering Department, 2006, chapter 5
  ("Speech Input/Output"), section 5.9 ("Delta, Acceleration and Third
  Differential Coefficients"); PDF:
  https://www.danielpovey.com/files/htkbook.pdf
* Delta features originate from S. Furui, "Speaker-independent isolated
  word recognition using dynamic features of speech spectrum", IEEE
  Trans. Acoustics, Speech, and Signal Processing, 1986,
  https://doi.org/10.1109/TASSP.1986.1164788

**How the convolution is computed.**  The discrete convolution of two
windows :math:`a` (indices :math:`-A..A`) and :math:`b` (indices
:math:`-B..B`) is the window :math:`c = a * b` (indices
:math:`-(A+B)..A+B`) defined by

.. math::

   c_m = \sum_{k} a_k \, b_{m-k}

where :math:`a_k` and :math:`b_k` are zero outside their index ranges.
For the second-order window this is

.. math::

   w^{(2)}_m = \sum_{k=-N}^{N} w^{(1)}_k \, w^{(1)}_{m-k}

**Window example 1 — N = 1, orders 1 to 3.**
:math:`\sum_{k=-1}^{1} k^2 = 2`, so the first-order window is
:math:`w^{(1)}_j = j/2 = [-0.5, 0, 0.5]`.  The convolution
:math:`w^{(2)}_m = \sum_k w^{(1)}_k w^{(1)}_{m-k}` term by term:

* :math:`m = -2`: :math:`w^{(1)}_{-1} w^{(1)}_{-1} = (-0.5)(-0.5) = 0.25`
* :math:`m = -1`: :math:`w^{(1)}_{-1} w^{(1)}_{0} + w^{(1)}_{0} w^{(1)}_{-1}
  = (-0.5)(0) + (0)(-0.5) = 0`
* :math:`m = 0`: :math:`w^{(1)}_{-1} w^{(1)}_{1} + w^{(1)}_{0} w^{(1)}_{0}
  + w^{(1)}_{1} w^{(1)}_{-1} = (-0.5)(0.5) + 0 + (0.5)(-0.5) = -0.5`
* :math:`m = 1`: :math:`w^{(1)}_{0} w^{(1)}_{1} + w^{(1)}_{1} w^{(1)}_{0} = 0`
* :math:`m = 2`: :math:`w^{(1)}_{1} w^{(1)}_{1} = (0.5)(0.5) = 0.25`

which gives :math:`w^{(2)} = [0.25, 0, -0.5, 0, 0.25]`.  Convolving once
more, :math:`w^{(3)} = w^{(2)} * w^{(1)}`; e.g.
:math:`w^{(3)}_{-3} = w^{(1)}_{-1} w^{(2)}_{-2} = (-0.5)(0.25) = -0.125` and
:math:`w^{(3)}_{-1} = w^{(1)}_{-1} w^{(2)}_{0} + w^{(1)}_{1} w^{(2)}_{-2}
= (-0.5)(-0.5) + (0.5)(0.25) = 0.375`.  The full chain for N = 1 is:

* :math:`w^{(0)} = [1]`
* :math:`w^{(1)} = [-0.5, 0, 0.5]`
* :math:`w^{(2)} = [0.25, 0, -0.5, 0, 0.25]`
* :math:`w^{(3)} = [-0.125, 0, 0.375, 0, -0.375, 0, 0.125]`

**Window example 2 — N = 2.**
:math:`\sum_{k=-2}^{2} k^2 = 10`, so
:math:`w^{(1)}_j = j/10 = [-0.2, -0.1, 0, 0.1, 0.2]`.  Two sample
convolution sums for :math:`w^{(2)} = w^{(1)} * w^{(1)}`:

* :math:`w^{(2)}_{0} = \sum_k w^{(1)}_k w^{(1)}_{-k} = 2(-0.2)(0.2) +
  2(-0.1)(0.1) + (0)(0) = -0.1`
* :math:`w^{(2)}_{4} = w^{(1)}_{2} w^{(1)}_{2} = (0.2)(0.2) = 0.04`

The full 9-tap window is
:math:`w^{(2)} = [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]`.

**Example 1 — full derivation** (``x = [1, 2, 4]``, ``order=2``, ``window=1``).
With the windows derived above,
:math:`w^{(1)} = [-0.5, 0, 0.5]` and :math:`w^{(2)} = [0.25, 0, -0.5, 0,
0.25]`, i.e.

.. math::

   \Delta_t = -0.5\, x_{t-1} + 0.5\, x_{t+1}

   \Delta^2_t = 0.25\, x_{t-2} - 0.5\, x_t + 0.25\, x_{t+2}

For the 1-D input ``x = [1, 2, 4]`` (out-of-range frames clamp to the
first/last frame):

* ``t = 0``: :math:`\Delta = -0.5 \cdot 1 + 0.5 \cdot 2 = 0.5`,
  :math:`\Delta^2 = 0.25 \cdot 1 - 0.5 \cdot 1 + 0.25 \cdot 4 = 0.75`
* ``t = 1``: :math:`\Delta = -0.5 \cdot 1 + 0.5 \cdot 4 = 1.5`,
  :math:`\Delta^2 = 0.25 \cdot 1 - 0.5 \cdot 2 + 0.25 \cdot 4 = 0.25`
* ``t = 2``: :math:`\Delta = -0.5 \cdot 2 + 0.5 \cdot 4 = 1.0`,
  :math:`\Delta^2 = 0.25 \cdot 1 - 0.5 \cdot 4 + 0.25 \cdot 4 = -0.75`

.. code-block:: python

   out = knf.compute_deltas(knf.DeltaOptions(order=2, window=1), x)
   # out rows [x, delta, delta2]:
   #   [1, 0.5,  0.75]
   #   [2, 1.5,  0.25]
   #   [4, 1.0, -0.75]

**Example 2 — larger window** (``x = [3, 1, 4, 1, 5]``, ``order=1``,
``window=2``).  With :math:`N = 2`,
:math:`\sum_{k=-2}^{2} k^2 = 10`, so
:math:`w^{(1)} = [-0.2, -0.1, 0, 0.1, 0.2]` (each entry is :math:`j/10`).
For instance at :math:`t = 2`:
:math:`\Delta = -0.2 \cdot 3 - 0.1 \cdot 1 + 0.1 \cdot 1 + 0.2 \cdot 5 = 0.4`.
The output rows ``[x, delta]`` are:

.. code-block:: python

   out = knf.compute_deltas(knf.DeltaOptions(order=1, window=2), x)
   #   [3,  0.0]
   #   [1, -0.3]
   #   [4,  0.4]
   #   [1,  0.9]
   #   [5,  0.6]

**Example 3 — two feature dimensions** (``D = 2``, ``order=2``,
``window=1``).  Each dimension is processed independently and the blocks
are laid out ``[x, delta, delta2]`` with the dimensions interleaved per
block:

.. code-block:: python

   x = [[1, 10], [2, 20], [4, 40]]
   out = knf.compute_deltas(knf.DeltaOptions(order=2, window=1), x)
   # out rows [x1, x2, d1, d2, dd1, dd2]:
   #   [1, 10,  0.5,  5.0,  0.75,  7.5]
   #   [2, 20,  1.5, 15.0,  0.25,  2.5]
   #   [4, 40,  1.0, 10.0, -0.75, -7.5]

:param opts: Order/window; see :class:`DeltaOptions`.
:param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
:type feats: numpy.ndarray
:return: The matrix of shape ``(num_frames, dim * (order + 1))``, float32.
:rtype: numpy.ndarray

Example
-------

.. code-block:: python

   opts = knf.DeltaOptions(order=2, window=2)
   out = knf.compute_deltas(opts, feats)  # (T, 3 * dim): [x, delta, d-delta]

.. seealso:: :class:`DeltaOptions`, :func:`splice_frames`)doc");
}

}  // namespace knf

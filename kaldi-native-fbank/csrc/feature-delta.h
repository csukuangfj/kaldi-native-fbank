/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-delta.h
//
// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
//                2014  IMSL, PKU-HKUST (author: Wei Shi)
//                2016  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/feat/feature-functions.h
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

/**
 * @file feature-delta.h
 * @brief Delta (dynamic) features.
 *
 * This is kaldi's `add-deltas` (see
 * @kfile{src/feat/feature-functions.cc}, class `DeltaFeatures`): each output
 * frame concatenates the input features with their first- and
 * second-order regression deltas, i.e. the output is
 * `[x, delta, delta-delta]` with `(order + 1) * dim` dimensions.
 *
 * @code
 *   knf::DeltaOptions opts;              // order=2, window=2
 *   knf::MatrixF out = knf::ComputeDeltas(opts, feats);
 * @endcode
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_DELTA_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_DELTA_H_

#include <stdint.h>

#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/// @brief Options for ComputeDeltas(); mirrors kaldi's add-deltas flags.
///
/// @see ComputeDeltas
struct DeltaOptions {
  /// Order of the delta computation (kaldi add-deltas
  /// @c --delta-order).  2 gives `[x, delta, delta-delta]`.
  int32_t order = 2;

  /// Delta regression window (kaldi add-deltas @c --delta-window).  The
  /// actual regression spans `2 * window + 1` frames.
  int32_t window = 2;
};

/**
   @brief Compute delta (dynamic) features.

   This is kaldi's `add-deltas` (kaldi `DeltaFeatures` /
   `ComputeDeltas` in @kfile{src/feat/feature-functions.cc}), same
   arithmetic.  The \f$ i \f$-th block of the output is the \f$ i \f$-th
   order regression delta; with \f$ N = \f$ `opts.window`, the first-order
   delta of frame \f$ t \f$ is

   \f[ \Delta_t = \frac{\sum_{j=-N}^{N} j\, x_{\mathrm{clamp}(t+j)}}
                       {2 \sum_{j=1}^{N} j^2} \f]

   where out-of-range frames clamp to the first/last frame (edge frames are
   replicated).  Higher orders apply the same operator recursively to the
   previous order's window (kaldi's `scales_` recursion).  The output rows
   are `[x_t, \Delta_t, \Delta^2_t, \ldots]`, i.e. the block order is
   baseline first, then increasing delta order.

   @param [in] opts   Order/window; see DeltaOptions.
   @param [in] feats  Feature matrix of size T x D (T > 0, D > 0).
   @return  The matrix of size T x (D * (order + 1)).

   The regression windows themselves are computed as follows.  With
   \f$ N \f$ = `opts.window`, the first-order window is

   \f[ w^{(1)}_j = \frac{j}{\sum_{k=-N}^{N} k^2}
                  = \frac{j}{2 \sum_{k=1}^{N} k^2},
        \qquad j = -N, \ldots, N \f]

   and each higher-order window is the discrete convolution of the
   previous window with \f$ w^{(1)} \f$ (kaldi's `scales_` recursion):

   \f[ w^{(i)} = w^{(i-1)} * w^{(1)}, \qquad w^{(0)} = [1] \f]

   so \f$ w^{(i)} \f$ is the \f$ i \f$-fold convolution of
   \f$ w^{(1)} \f$ (its length is `1 + 2 * i * window`).

   **Motivation.**  The spectrum of speech is strongly non-stationary: the
   *trajectory* of the spectrum over time carries information that the
   static spectrum does not.  Furui (1986) showed that appending the time
   derivative of the spectrum (the "delta") significantly improves
   recognition.  HTK refined this into the regression-based formulation
   used here: instead of a raw difference such as
   \f$ x_{t+1} - x_{t-1} \f$, the local slope of a least-squares line is
   estimated over a window of \f$ 2N+1 \f$ frames, which is less
   sensitive to noise and to isolated outliers.  kaldi's `DeltaFeatures`
   implements the same HTK formulation.

   **Why this definition.**  The first-order window is the slope of the
   least-squares straight line fitted to the frames around \f$ t \f$ — the
   classic HTK regression delta that kaldi's `DeltaFeatures` implements.
   Fitting \f$ y(j) = a + b\, j \f$ to \f$ x_{t+j} \f$,
   \f$ j = -N, \ldots, N \f$, by minimizing
   \f$ \sum_j (x_{t+j} - a - b\, j)^2 \f$, and using
   \f$ \sum_j j = 0 \f$ (which decouples the intercept from the slope),
   the slope estimate is

   \f[ b = \frac{\sum_j j\, x_{t+j}}{\sum_j j^2}
          = \sum_j w^{(1)}_j\, x_{t+j} \f]

   so \f$ w^{(1)}_j = j / \sum_k k^2 \f$ is exactly the least-squares
   slope estimator: \f$ \Delta_t = b \f$ is the local slope of the fitted
   line, i.e. a least-squares estimate of the first derivative (the
   \f$ \sum_j j^2 \f$ normalizer is the OLS denominator).

   Higher orders apply the *same* slope estimator to the previous order's
   output (\f$ \Delta^2 \f$ is "the slope of the slope").  Since the
   slope estimator is a linear filter with kernel \f$ w^{(1)} \f$,
   filtering an already-filtered sequence convolves the kernels, hence
   \f$ w^{(i)} = w^{(i-1)} * w^{(1)} \f$ with support growing to
   \f$ \pm i N \f$.  Note this is *recursive* slope-of-slope — it is not
   identical to the \f$ i \f$-th derivative of a degree-\f$ i \f$
   least-squares polynomial fitted over the wider \f$ \pm iN \f$ window.

   Normalizing by \f$ \sum_j j^2 \f$ at every order keeps each block on
   the scale of a true \f$ i \f$-th derivative, and edge frames replicate
   the first/last frame so the frame count is preserved and the regression
   "sees" a locally constant signal at the borders.  This is the standard
   HTK/kaldi dynamic-feature formulation.

   **References.**

   * The ported implementation: kaldi's `DeltaFeatures`
     (@kfile{src/feat/feature-functions.cc},
     @kfile{src/feat/feature-functions.h}).
   * The regression-delta formulation: *The HTK Book*, S. Young et al.,
     Cambridge University Engineering Department, 2006, chapter 5
     ("Speech Input/Output"), section 5.9 ("Delta, Acceleration and Third
     Differential Coefficients"); PDF:
     [htkbook.pdf](https://www.danielpovey.com/files/htkbook.pdf)
   * Delta features originate from S. Furui, "Speaker-independent isolated
     word recognition using dynamic features of speech spectrum", IEEE
     Trans. Acoustics, Speech, and Signal Processing, 1986,
     https://doi.org/10.1109/TASSP.1986.1164788

   **How the convolution is computed.**  The discrete convolution of two
   windows \f$ a \f$ (indices \f$ -A..A \f$) and \f$ b \f$ (indices
   \f$ -B..B \f$) is the window \f$ c = a * b \f$ (indices
   \f$ -(A+B)..A+B \f$) defined by

   \f[ c_m = \sum_{k} a_k \, b_{m-k} \f]

   where \f$ a_k \f$ and \f$ b_k \f$ are zero outside their index
   ranges.  For the second-order window this is

   \f[ w^{(2)}_m = \sum_{k=-N}^{N} w^{(1)}_k \, w^{(1)}_{m-k} \f]

   **Window example 1 - N = 1, orders 1 to 3.**
   \f$ \sum_{k=-1}^{1} k^2 = 2 \f$, so the first-order window is
   \f$ w^{(1)}_j = j/2 = [-0.5, 0, 0.5] \f$.  The convolution
   \f$ w^{(2)}_m = \sum_k w^{(1)}_k w^{(1)}_{m-k} \f$ term by term:

   * \f$ m = -2 \f$: \f$ w^{(1)}_{-1} w^{(1)}_{-1} = (-0.5)(-0.5) = 0.25 \f$
   * \f$ m = -1 \f$: \f$ w^{(1)}_{-1} w^{(1)}_{0} + w^{(1)}_{0} w^{(1)}_{-1}
     = (-0.5)(0) + (0)(-0.5) = 0 \f$
   * \f$ m = 0 \f$: \f$ w^{(1)}_{-1} w^{(1)}_{1} + w^{(1)}_{0} w^{(1)}_{0}
     + w^{(1)}_{1} w^{(1)}_{-1} = (-0.5)(0.5) + 0 + (0.5)(-0.5) = -0.5 \f$
   * \f$ m = 1 \f$: \f$ w^{(1)}_{0} w^{(1)}_{1} + w^{(1)}_{1} w^{(1)}_{0}
     = 0 \f$
   * \f$ m = 2 \f$: \f$ w^{(1)}_{1} w^{(1)}_{1} = (0.5)(0.5) = 0.25 \f$

   which gives \f$ w^{(2)} = [0.25, 0, -0.5, 0, 0.25] \f$.  Convolving
   once more, \f$ w^{(3)} = w^{(2)} * w^{(1)} \f$; e.g.
   \f$ w^{(3)}_{-3} = w^{(1)}_{-1} w^{(2)}_{-2} = (-0.5)(0.25) = -0.125 \f$
   and \f$ w^{(3)}_{-1} = w^{(1)}_{-1} w^{(2)}_{0} + w^{(1)}_{1} w^{(2)}_{-2}
   = (-0.5)(-0.5) + (0.5)(0.25) = 0.375 \f$.  The full chain for N = 1:

   * \f$ w^{(0)} = [1] \f$
   * \f$ w^{(1)} = [-0.5, 0, 0.5] \f$
   * \f$ w^{(2)} = [0.25, 0, -0.5, 0, 0.25] \f$
   * \f$ w^{(3)} = [-0.125, 0, 0.375, 0, -0.375, 0, 0.125] \f$

   **Window example 2 - N = 2.**
   \f$ \sum_{k=-2}^{2} k^2 = 10 \f$, so
   \f$ w^{(1)}_j = j/10 = [-0.2, -0.1, 0, 0.1, 0.2] \f$.  Two sample
   convolution sums for \f$ w^{(2)} = w^{(1)} * w^{(1)} \f$:

   * \f$ w^{(2)}_{0} = \sum_k w^{(1)}_k w^{(1)}_{-k} = 2(-0.2)(0.2) +
     2(-0.1)(0.1) + (0)(0) = -0.1 \f$
   * \f$ w^{(2)}_{4} = w^{(1)}_{2} w^{(1)}_{2} = (0.2)(0.2) = 0.04 \f$

   The full 9-tap window is
   \f$ w^{(2)} = [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]
   \f$.

   **Example 1 - full derivation** (`x = [1, 2, 4]`, `order=2`, `window=1`).
   With the windows derived above,
   \f$ w^{(1)} = [-0.5, 0, 0.5] \f$ and
   \f$ w^{(2)} = [0.25, 0, -0.5, 0, 0.25] \f$, i.e.

   \f[ \Delta_t = -0.5\, x_{t-1} + 0.5\, x_{t+1} \f]
   \f[ \Delta^2_t = 0.25\, x_{t-2} - 0.5\, x_t + 0.25\, x_{t+2} \f]

   For the 1-D input `x = [1, 2, 4]` (out-of-range frames clamp to the
   first/last frame) the output rows `[x_t, \Delta_t, \Delta^2_t]` are:

   @code
     [1, 0.5,  0.75]   // t=0: D = -0.5*1 + 0.5*2;  D2 = 0.25*1 - 0.5*1 + 0.25*4
     [2, 1.5,  0.25]   // t=1: D = -0.5*1 + 0.5*4;  D2 = 0.25*1 - 0.5*2 + 0.25*4
     [4, 1.0, -0.75]   // t=2: D = -0.5*2 + 0.5*4;  D2 = 0.25*1 - 0.5*4 + 0.25*4
   @endcode

   **Example 2 - larger window** (`x = [3, 1, 4, 1, 5]`, `order=1`,
   `window=2`).  With \f$ N = 2 \f$, \f$ \sum_{k=-2}^{2} k^2 = 10 \f$,
   so \f$ w^{(1)} = [-0.2, -0.1, 0, 0.1, 0.2] \f$ (each entry is
   \f$ j/10 \f$).  At \f$ t = 2 \f$:
   \f$ \Delta = -0.2 \cdot 3 - 0.1 \cdot 1 + 0.1 \cdot 1 + 0.2 \cdot 5
   = 0.4 \f$.  The output rows `[x, delta]` are:

   @code
     [3,  0.0]
     [1, -0.3]
     [4,  0.4]
     [1,  0.9]
     [5,  0.6]
   @endcode

   **Example 3 - two feature dimensions** (`D = 2`, `order=2`,
   `window=1`).  Each dimension is processed independently; the blocks are
   `[x, delta, delta2]` with dimensions interleaved per block:

   @code
     x = [[1, 10], [2, 20], [4, 40]]
     out rows [x1, x2, d1, d2, dd1, dd2]:
       [1, 10,  0.5,  5.0,  0.75,  7.5]
       [2, 20,  1.5, 15.0,  0.25,  2.5]
       [4, 40,  1.0, 10.0, -0.75, -7.5]
   @endcode

   @code
     knf::DeltaOptions opts;  // order=2, window=2 -> 3*D dims
     knf::MatrixF out = knf::ComputeDeltas(opts, feats);
   @endcode

   @par Used by:
   Pipelines that add dynamic features after cmvn (mfcc -> cmvn ->
   deltas).

   @see DeltaOptions, SpliceFrames
*/
MatrixF ComputeDeltas(const DeltaOptions &opts, const MatrixF &feats);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_DELTA_H_

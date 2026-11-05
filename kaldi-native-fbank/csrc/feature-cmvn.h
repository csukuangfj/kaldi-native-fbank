/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-cmvn.h
//
// Copyright 2009-2013  Microsoft Corporation
//                     Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/transform/cmvn.h
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

/**
 * @file feature-cmvn.h
 * @brief Cepstral mean and variance normalization (CMVN).
 *
 * The semantics follow kaldi's @kfile{src/transform/cmvn.cc} exactly (same
 * arithmetic and stats layout).
 *
 * Cmvn stats layout (matches kaldi's cmvn.ark format exactly):
 *
 * 2-row stats (full stats, produced by InitCmvnStats/ComputeCmvnStats):
 * row 0 = [sum_1 .. sum_D, count], row 1 = [sumsq_1 .. sumsq_D, (unused)].
 * Supports both mean and variance normalization.
 *
 * 1-row stats (mean-only stats; some pipelines only collect means):
 * row 0 = [sum_1 .. sum_D, count]. Supports mean normalization only;
 * requesting norm_vars with 1-row stats is an error (same as kaldi).
 *
 * In both forms the frame count lives in the LAST column of row 0
 * (index `dim`); columns 0..dim-1 hold sums / sumsqs.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_CMVN_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_CMVN_H_

#include <stdint.h>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/// @brief Options matching kaldi's @kbin{apply-cmvn} flags.
///
/// @par Used by:
/// ApplyCmvn().
///
/// @see ApplyCmvn, ComputeCmvnStats
struct CmvnOptions {
  /// If true, normalize the variances as well: after subtracting the mean,
  /// each dimension is divided by its standard deviation. Requires 2-row
  /// stats (kaldi @kbin{apply-cmvn} @c --norm-vars).
  bool norm_vars = false;

  /// If true, undo cmvn instead of applying it: the mean is added back and,
  /// if norm_vars is true, the standard deviation is multiplied back
  /// (kaldi @kbin{apply-cmvn} @c --reverse).
  bool reverse = false;
};

/**
   @brief Initialize cmvn stats to zeros in the FULL 2-row layout.

   The output matrix `stats` is resized to 2 x (dim+1) and set to zero. Its
   layout matches kaldi's cmvn.ark format exactly:
   \f[ \mathrm{stats} =
       \left( \begin{array}{cccc|c}
         S_1 & S_2 & \cdots & S_D & c \\
         Q_1 & Q_2 & \cdots & Q_D & -
       \end{array} \right) \f]

   Row 0 (above): columns 0 .. dim-1 hold the sums
   \f$ S_d = \sum_t x_t(d) \f$ of each feature dimension; the last
   column \f$ c \f$ holds the frame count (sum of weights).

   Row 1 (below): columns 0 .. dim-1 hold the sums of squares
   \f$ Q_d = \sum_t x_t(d)^2 \f$; the last column is unused.

   This is the form produced/consumed by AccCmvnStats(), ComputeCmvnStats()
   and ApplyCmvn() with norm_vars=true.

   Note: ApplyCmvn() also accepts 1-row (mean-only) stats, where only row 0
   above is present; this function, however, only produces the 2-row form
   used for accumulating both mean and variance.

   @param [in] dim    Feature dimension of the frames that will be
                      accumulated, e.g. 39 for 39-dim MFCC.
   @param [out] stats The output stats matrix, resized to 2 x (dim+1).

   @code
     knf::MatrixD stats;
     knf::InitCmvnStats(39, &stats);  // stats: 2 x 40, all zeros
   @endcode

   @see AccCmvnStats, ComputeCmvnStats, ApplyCmvn
*/
void InitCmvnStats(int32_t dim, MatrixD *stats);

/**
   @brief Accumulate stats for one frame (kaldi
   [AccCmvnStats](https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc),
   same op order).

   The following updates are applied to `stats`, which must use the full
   2-row layout (see InitCmvnStats()); with \f$ w = \f$ `weight` and
   \f$ x_d = \f$ `feat[d]`:
   \f[ c \mathrel{+}= w, \qquad
       S_d \mathrel{+}= x_d\, w, \qquad
       Q_d \mathrel{+}= x_d^2\, w
       \qquad (0 \le d < \mathrm{dim}) \f]

   @param [in] feat   Pointer to one frame, a 1-D array of size dim.
   @param [in] dim    Feature dimension of the frame; must equal
                      stats->cols() - 1.
   @param [in] weight Weight of this frame, e.g. 1.0.
   @param [in,out] stats  Stats to update; must use the full 2-row layout.

   @code
     knf::MatrixD stats;
     knf::InitCmvnStats(dim, &stats);
     for (int32_t t = 0; t < num_frames; ++t) {
       knf::AccCmvnStats(feats.row(t).data(), dim, 1.0f, &stats);
     }
   @endcode

   @see InitCmvnStats, ComputeCmvnStats, CmvnAccumulator::Add
*/
void AccCmvnStats(const float *feat, int32_t dim, float weight, MatrixD *stats);

/**
   @brief Accumulate stats for a feature matrix (rows = frames).

   This is equivalent to calling the single-frame AccCmvnStats() above for
   every row of `feats`, with weight 1.0.

   @param [in] feats  Feature matrix of size num_frames x dim.
   @param [in,out] stats  Stats to update; must use the full 2-row layout.

   @see InitCmvnStats, CmvnAccumulator::Accumulate
*/
void AccCmvnStats(const MatrixF &feats, MatrixD *stats);

/**
   @brief Compute full 2-row cmvn stats for a whole matrix.

   Equivalent to InitCmvnStats() followed by AccCmvnStats() over all rows.

   @param [in] feats  Feature matrix of size num_frames x dim.
   @return  The stats matrix of size 2 x (dim+1); see InitCmvnStats() for
            the layout. The frame count in stats(0, dim) is num_frames.

   @see InitCmvnStats, AccCmvnStats, ApplyCmvn
*/
MatrixD ComputeCmvnStats(const MatrixF &feats);

/**
   @brief Apply cmvn to feats in place (kaldi
   [ApplyCmvn](https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc)
   /
   [ApplyCmvnReverse](https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc)).

   Let \f$ c = \mathrm{stats}(0, \mathrm{dim}) \f$ be the frame count,
   \f$ \mu_d = \mathrm{stats}(0, d) / c \f$ the mean and, when norm_vars is
   true, \f$ \sigma_d^2 = \mathrm{stats}(1, d) / c - \mu_d^2 \f$ the
   variance of dimension \f$ d \f$.

   Forward (opts.reverse == false), applied to every frame \f$ x \f$:
   \f[ x_d \leftarrow x_d - \mu_d \f]
   and, if opts.norm_vars:
   \f[ x_d \leftarrow (x_d - \mu_d) / \sigma_d \f]

   Reverse (opts.reverse == true) undoes the normalization:
   \f[ x_d \leftarrow x_d + \mu_d \f]
   and, if opts.norm_vars:
   \f[ x_d \leftarrow x_d \cdot \sigma_d + \mu_d \f]

   Accepts 1-row (mean-only) or 2-row stats; requesting norm_vars with 1-row
   stats is an error (same as kaldi).

   @param [in] stats  1-row or 2-row cmvn stats; see InitCmvnStats().
   @param [in] opts   Options: norm_vars (normalize variances), reverse
                      (undo cmvn).
   @param [in,out] feats  Feature matrix of size num_frames x dim; each row
                          is normalized in place.

   @code
     knf::MatrixD stats = knf::ComputeCmvnStats(feats);
     knf::CmvnOptions opts;  // norm_vars=false, reverse=false
     knf::ApplyCmvn(stats, opts, &feats);  // feats normalized in place
   @endcode

   @see ComputeCmvnStats, CmvnOptions, CmvnAccumulator
*/
void ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts, MatrixF *feats);

/**
   @brief In-place variant taking a mutable Eigen view of the features.

   Same semantics as the pointer version above. The view must already
   have the right shape. It enables zero-copy application to a mapped
   buffer (e.g. a numpy array via pybind11).

   @see ApplyCmvn
*/
void ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts,
               Eigen::Ref<MatrixF> feats);

/**
   @brief Convenience version of ApplyCmvn(): returns the normalized copy.

   Same semantics as the in-place ApplyCmvn() above, but `feats` is left
   unchanged and a normalized copy is returned.

   @param [in] stats  1-row or 2-row cmvn stats; see InitCmvnStats().
   @param [in] opts   Options: norm_vars (normalize variances), reverse
                      (undo cmvn).
   @param [in] feats  Feature matrix of size num_frames x dim.
   @return  A copy of feats with cmvn applied.
*/
MatrixF ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts,
                  const MatrixF &feats);

/// @brief Incremental accumulator for speaker-level / streaming cmvn.
///
/// It always keeps the full 2-row layout (see InitCmvnStats()) and adds up
/// weighted frames one by one, so stats can be collected over many
/// utterances of the same speaker and normalized later with ApplyCmvn().
///
/// @par Used by:
/// Pipelines that collect speaker-level stats incrementally (e.g.
/// accumulate all utterances of a speaker, then normalize).
///
/// @see AccCmvnStats, InitCmvnStats, ApplyCmvn
class CmvnAccumulator {
 public:
  /// Construct an accumulator for features of dimension `dim`.
  ///
  /// @param dim  Feature dimension; 0 (the default) creates an empty
  ///             accumulator whose stats are grown by the first Add().
  explicit CmvnAccumulator(int32_t dim = 0);

  /// Accumulate stats for a feature matrix (rows = frames); equivalent to
  /// calling Add() for each row with weight 1.0.
  void Accumulate(const MatrixF &feats);

  /// Accumulate stats for one frame of dimension `dim`, applying the same
  /// updates as the free function AccCmvnStats().
  ///
  /// @param dim     Feature dimension; must match the dimension given to the
  ///                constructor.
  /// @param frame   Pointer to one frame, a 1-D array of size dim.
  /// @param weight  Weight of this frame, e.g. 1.0.
  void Add(int32_t dim, const float *frame, float weight = 1.0f);

  /// Return the accumulated 2-row stats: row 0 = [sums, count], row 1 =
  /// [sumsq, (unused)]; see InitCmvnStats() for the layout.
  const MatrixD &Stats() const { return stats_; }

 private:
  MatrixD stats_;  ///< 2-row stats; see InitCmvnStats() for the layout.
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_CMVN_H_

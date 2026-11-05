/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-splice.h
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
 * @file feature-splice.h
 * @brief Splice frames with their left/right context neighbors.
 *
 * This is kaldi's `splice-feats` (see
 * @kfile{src/feat/feature-functions.cc}): each output frame concatenates the
 * input frame with `left_context` preceding and `right_context` following
 * frames.  Edge frames are replicated (a frame index below 0 clamps to 0 and
 * an index past the end clamps to the last frame), so the output has the SAME
 * number of frames as the input and `dim * (1 + left_context + right_context)`
 * dimensions.
 *
 * @code
 *   knf::SpliceOptions opts;       // left_context=3, right_context=3
 *   knf::MatrixF spliced = knf::SpliceFrames(feats, opts);
 * @endcode
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_SPLICE_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_SPLICE_H_

#include <stdint.h>

#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/// @brief Options for SpliceFrames(); mirrors kaldi's splice-feats flags.
///
/// @see SpliceFrames
struct SpliceOptions {
  /// Number of preceding frames to append (kaldi splice-feats
  /// @c --left-context).
  int32_t left_context = 3;

  /// Number of following frames to append (kaldi splice-feats
  /// @c --right-context).
  int32_t right_context = 3;
};

/**
   @brief Concatenate each frame with its left/right context neighbors.

   This is kaldi's `splice-feats` (kaldi `SpliceFrames` in
   @kfile{src/feat/feature-functions.cc}), same arithmetic: for every frame
   index \f$ t \f$ the output row is

   \f[ y_t = [ x_{\max(0, t-L)}, \ldots, x_{\min(T-1, t+R)} ] \f]

   where \f$ L = \f$ `opts.left_context`, \f$ R = \f$ `opts.right_context`
   and \f$ T \f$ is the number of input frames.  Edge frames are replicated:
   out-of-range indices clamp to the first/last frame.  The output therefore
   has the same number of frames as the input and
   `dim * (1 + L + R)` dimensions.

   @param [in] feats  Feature matrix of size T x D (T > 0, D > 0).
   @param [in] opts   Contexts; see SpliceOptions.
   @return  The spliced matrix of size T x (D * (1 + L + R)).

   @code
     knf::SpliceOptions opts;  // left=3, right=3 -> 7*D dims
     knf::MatrixF spliced = knf::SpliceFrames(feats, opts);
   @endcode

   @par Used by:
   Pipelines that build xvector/MFA-style features (mfcc -> cmvn ->
   splice -> lda).

   @see SpliceOptions
*/
MatrixF SpliceFrames(const MatrixF &feats, const SpliceOptions &opts);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_SPLICE_H_

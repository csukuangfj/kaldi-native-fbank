/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-splice.cc
//
// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
//                2014  IMSL, PKU-HKUST (author: Wei Shi)
//                2016  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/feat/feature-functions.cc
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include "kaldi-native-fbank/csrc/feature-splice.h"

#include <stdexcept>

#include "Eigen/Core"

namespace knf {

MatrixF SpliceFrames(const MatrixF &feats, const SpliceOptions &opts) {
  if (feats.rows() == 0 || feats.cols() == 0) {
    throw std::invalid_argument("SpliceFrames: empty input");
  }
  if (opts.left_context < 0 || opts.right_context < 0) {
    throw std::invalid_argument(
        "SpliceFrames: left_context/right_context must be >= 0");
  }

  int32_t num_frames = feats.rows();
  int32_t dim = feats.cols();
  int32_t num_cols = 1 + opts.left_context + opts.right_context;

  MatrixF ans(num_frames, dim * num_cols);

  // Output block j holds the frames at offset (j - left_context) with the
  // edge frames replicated, i.e. row t comes from frame
  // clamp(t + shift, 0, T-1) with shift = j - left_context.
  for (int32_t j = 0; j < num_cols; ++j) {
    int32_t shift = j - opts.left_context;

    // Middle rows are the ones whose source frame t + shift lies in [0, T);
    // the head/tail rows replicate the first/last frame.
    int32_t mid_lo = shift < 0 ? -shift : 0;
    if (mid_lo > num_frames) {
      mid_lo = num_frames;
    }
    int32_t mid_hi = num_frames - shift;
    if (mid_hi > num_frames) {
      mid_hi = num_frames;
    }
    if (mid_hi < mid_lo) {
      mid_hi = mid_lo;
    }

    if (mid_lo > 0) {
      // rows [0, mid_lo): replicate the first frame
      ans.block(0, j * dim, mid_lo, dim).rowwise() = feats.row(0);
    }
    if (mid_hi > mid_lo) {
      // rows [mid_lo, mid_hi): direct copy from the shifted rows
      ans.block(mid_lo, j * dim, mid_hi - mid_lo, dim) =
          feats.middleRows(mid_lo + shift, mid_hi - mid_lo);
    }
    if (mid_hi < num_frames) {
      // rows [mid_hi, T): replicate the last frame
      ans.block(mid_hi, j * dim, num_frames - mid_hi, dim).rowwise() =
          feats.row(num_frames - 1);
    }
  }

  return ans;
}

}  // namespace knf

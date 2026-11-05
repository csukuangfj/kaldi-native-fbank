/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-delta.cc
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

#include "kaldi-native-fbank/csrc/feature-delta.h"

#include <stdexcept>
#include <vector>

#include "Eigen/Core"

namespace knf {

namespace {

// rows(t) = feats.row(clamp(t + shift, 0, T-1)); edge rows replicate the
// first/last frame (kaldi's clamping in DeltaFeatures::Process).
MatrixF ShiftClamped(const MatrixF &feats, int32_t shift) {
  int32_t num_frames = feats.rows();
  int32_t dim = feats.cols();

  // middle rows whose source frame t + shift lies in [0, T)
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

  MatrixF ans(num_frames, dim);
  if (mid_lo > 0) {
    ans.topRows(mid_lo).rowwise() = feats.row(0);
  }
  if (mid_hi > mid_lo) {
    ans.middleRows(mid_lo, mid_hi - mid_lo) =
        feats.middleRows(mid_lo + shift, mid_hi - mid_lo);
  }
  if (mid_hi < num_frames) {
    ans.bottomRows(num_frames - mid_hi).rowwise() = feats.row(num_frames - 1);
  }
  return ans;
}

// kaldi DeltaFeatures::scales_ recursion: scales[i] is the regression
// window of order i, built by convolving scales[i-1] with the weights
// j / sum(j^2) for j in [-window, window].
std::vector<VectorF> ComputeScales(const DeltaOptions &opts) {
  std::vector<VectorF> scales(opts.order + 1);
  scales[0] = VectorF::Ones(1);

  for (int32_t i = 1; i <= opts.order; ++i) {
    const VectorF &prev = scales[i - 1];
    VectorF cur = VectorF::Zero(prev.size() + 2 * opts.window);

    float normalizer = 0.0f;
    for (int32_t j = -opts.window; j <= opts.window; ++j) {
      normalizer += static_cast<float>(j) * j;
      cur.segment(j + opts.window, prev.size()) += static_cast<float>(j) * prev;
    }
    cur /= normalizer;
    scales[i] = cur;
  }
  return scales;
}

}  // namespace

MatrixF ComputeDeltas(const DeltaOptions &opts, const MatrixF &feats) {
  if (feats.rows() == 0 || feats.cols() == 0) {
    throw std::invalid_argument("ComputeDeltas: empty input");
  }
  if (opts.order < 0 || opts.order >= 1000) {
    throw std::invalid_argument("ComputeDeltas: order must be in [0, 1000)");
  }
  if (opts.window <= 0 || opts.window >= 1000) {
    throw std::invalid_argument("ComputeDeltas: window must be in (0, 1000)");
  }

  int32_t num_frames = feats.rows();
  int32_t dim = feats.cols();

  std::vector<VectorF> scales = ComputeScales(opts);

  MatrixF ans(num_frames, dim * (opts.order + 1));

  // Block 0: the baseline features.
  ans.leftCols(dim) = feats;

  for (int32_t i = 1; i <= opts.order; ++i) {
    const VectorF &cur_scales = scales[i];
    int32_t max_offset = (cur_scales.size() - 1) / 2;
    MatrixF block = MatrixF::Zero(num_frames, dim);
    // Same accumulation order as kaldi DeltaFeatures::Process: j ascending.
    for (int32_t j = -max_offset; j <= max_offset; ++j) {
      float scale = cur_scales(j + max_offset);
      if (scale != 0.0f) {
        block += scale * ShiftClamped(feats, j);
      }
    }
    ans.middleCols(i * dim, dim) = block;
  }

  return ans;
}

}  // namespace knf

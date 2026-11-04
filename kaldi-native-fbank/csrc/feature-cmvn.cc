/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-cmvn.cc
//
// Copyright 2009-2013  Microsoft Corporation
//                     Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/transform/cmvn.cc
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include "kaldi-native-fbank/csrc/feature-cmvn.h"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace knf {

namespace {
void CheckStats(const MatrixD &stats, int32_t dim) {
  if (stats.rows() > 2 || stats.rows() < 1 || stats.cols() != dim + 1) {
    std::ostringstream os;
    os << "Dim mismatch: cmvn " << stats.rows() << "x" << stats.cols()
       << ", expected 2x" << (dim + 1);
    throw std::invalid_argument(os.str());
  }
}
}  // namespace

void InitCmvnStats(int32_t dim, MatrixD *stats) {
  if (dim <= 0) {
    throw std::invalid_argument("dim must be > 0");
  }
  *stats = MatrixD::Zero(2, dim + 1);
}

void AccCmvnStats(const float *feat, int32_t dim, float weight,
                  MatrixD *stats) {
  if (stats == nullptr) {
    throw std::invalid_argument("stats is null");
  }
  if (stats->rows() != 2 || stats->cols() != dim + 1) {
    throw std::invalid_argument("stats must be 2 x (dim+1)");
  }
  /// kaldi AccCmvnStats, same arithmetic:
  ///   count += weight; sum += x*w; sumsq += x*x*w
  double *mean_ptr = stats->row(0).data();
  double *var_ptr = stats->row(1).data();
  double *count_ptr = mean_ptr + dim;
  const float *feats_ptr = feat;
  *count_ptr += weight;
  for (; mean_ptr < count_ptr; mean_ptr++, var_ptr++, feats_ptr++) {
    *mean_ptr += *feats_ptr * weight;
    *var_ptr += *feats_ptr * *feats_ptr * weight;
  }
}

void AccCmvnStats(const MatrixF &feats, MatrixD *stats) {
  for (int32_t i = 0; i < feats.rows(); ++i) {
    AccCmvnStats(feats.row(i).data(), feats.cols(), 1.0f, stats);
  }
}

MatrixD ComputeCmvnStats(const MatrixF &feats) {
  MatrixD stats;
  InitCmvnStats(feats.cols(), &stats);
  AccCmvnStats(feats, &stats);
  return stats;
}

void ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts, MatrixF *feats) {
  int32_t dim = feats->cols();
  CheckStats(stats, dim);
  if (stats.rows() == 1 && opts.norm_vars) {
    throw std::invalid_argument(
        "variance normalization requested but no variance stats supplied");
  }
  double count = stats(0, dim);
  if (count < 1.0) {
    throw std::invalid_argument("Insufficient stats for cmvn: count = " +
                                std::to_string(count));
  }

  if (!opts.reverse) {
    if (!opts.norm_vars) {
      /// kaldi: offset = -1.0/count * mean_stats; feats += offset
      VectorF offset(dim);
      for (int32_t d = 0; d < dim; ++d) {
        offset[d] = static_cast<float>(-1.0 / count * stats(0, d));
      }
      for (int32_t i = 0; i < feats->rows(); ++i) {
        for (int32_t d = 0; d < dim; ++d) (*feats)(i, d) += offset[d];
      }
      return;
    }
    /// kaldi: x(d) <-- x(d) * scale + offset,  offset = -(mean*scale)
    VectorF norm0(dim), norm1(dim);
    for (int32_t d = 0; d < dim; ++d) {
      double mean = stats(0, d) / count;
      double var = (stats(1, d) / count) - mean * mean;
      const double floor_val = 1.0e-20;
      if (var < floor_val) {
        fprintf(stderr, "WARNING: Flooring cepstral variance from %f to %f\n",
                var, floor_val);
        var = floor_val;
      }
      double scale = 1.0 / std::sqrt(var);
      double offset = -(mean * scale);
      norm0[d] = static_cast<float>(offset);
      norm1[d] = static_cast<float>(scale);
    }
    for (int32_t i = 0; i < feats->rows(); ++i) {
      for (int32_t d = 0; d < dim; ++d) {
        (*feats)(i, d) = (*feats)(i, d) * norm1[d] + norm0[d];
      }
    }
    return;
  }

  /// reverse: x(d) <-- x(d) * scale + offset with offset=mean,
  // scale = (var_norm ? sqrt(var) : 1)
  VectorF norm0(dim), norm1(dim);
  for (int32_t d = 0; d < dim; ++d) {
    double mean = stats(0, d) / count;
    double scale = 1.0;
    if (opts.norm_vars) {
      double var = (stats(1, d) / count) - mean * mean;
      const double floor_val = 1.0e-20;
      if (var < floor_val) {
        fprintf(stderr, "WARNING: Flooring cepstral variance from %f to %f\n",
                var, floor_val);
        var = floor_val;
      }
      scale = std::sqrt(var);
    }
    norm0[d] = static_cast<float>(mean);
    norm1[d] = static_cast<float>(scale);
  }
  for (int32_t i = 0; i < feats->rows(); ++i) {
    for (int32_t d = 0; d < dim; ++d) {
      (*feats)(i, d) = (*feats)(i, d) * norm1[d] + norm0[d];
    }
  }
}

MatrixF ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts,
                  const MatrixF &feats) {
  MatrixF ans = feats;
  ApplyCmvn(stats, opts, &ans);
  return ans;
}

CmvnAccumulator::CmvnAccumulator(int32_t dim) {
  if (dim > 0) {
    InitCmvnStats(dim, &stats_);
  }
}

void CmvnAccumulator::Accumulate(const MatrixF &feats) {
  if (stats_.rows() == 0) {
    InitCmvnStats(feats.cols(), &stats_);
  }
  AccCmvnStats(feats, &stats_);
}

void CmvnAccumulator::Add(int32_t dim, const float *frame, float weight) {
  if (stats_.rows() == 0) {
    InitCmvnStats(dim, &stats_);
  }
  AccCmvnStats(frame, dim, weight, &stats_);
}

}  // namespace knf

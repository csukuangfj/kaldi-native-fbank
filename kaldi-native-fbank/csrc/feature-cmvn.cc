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

#include <sstream>
#include <stdexcept>
#include <string>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/macros.h"

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
  Eigen::Map<const VectorF> x(feat, dim);
  (*stats)(0, dim) += weight;
  stats->row(0).head(dim) += x.cast<double>() * static_cast<double>(weight);
  stats->row(1).head(dim) +=
      x.cast<double>().array().square().matrix() * static_cast<double>(weight);
}

void AccCmvnStats(const MatrixF &feats, MatrixD *stats) {
  if (stats == nullptr) {
    throw std::invalid_argument("stats is null");
  }
  int32_t dim = feats.cols();
  if (stats->rows() != 2 || stats->cols() != dim + 1) {
    throw std::invalid_argument("stats must be 2 x (dim+1)");
  }
  /// kaldi AccCmvnStats applied to every row at once:
  ///   count += T; sum += colwise-sum; sumsq += colwise-sum-of-squares
  (*stats)(0, dim) += static_cast<double>(feats.rows());
  stats->row(0).head(dim) += feats.cast<double>().colwise().sum();
  stats->row(1).head(dim) +=
      feats.cast<double>().array().square().colwise().sum().matrix();
}

MatrixD ComputeCmvnStats(const MatrixF &feats) {
  MatrixD stats;
  InitCmvnStats(feats.cols(), &stats);
  AccCmvnStats(feats, &stats);
  return stats;
}

namespace {
void ApplyCmvnInPlace(const MatrixD &stats, const CmvnOptions &opts,
                      Eigen::Ref<MatrixF> feats) {
  int32_t dim = feats.cols();
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

  /// kaldi ApplyCmvn/ApplyCmvnReverse: x(d) <- x(d) * scale(d) + offset(d)
  ///   forward, no var-norm: scale = 1,        offset = -1/count * mean_stats
  ///   forward, var-norm:     scale = 1/sigma, offset = -(mean*scale)
  ///   reverse:               scale = sigma|1, offset = mean
  /// with mean = stats.row(0)/count, var = stats.row(1)/count - mean^2.
  Eigen::RowVectorXd scale(dim);
  Eigen::RowVectorXd offset(dim);

  if (!opts.norm_vars) {
    scale.setOnes();
    if (!opts.reverse) {
      /// kaldi: offset = -1.0/count * mean_stats; feats += offset
      offset = (-1.0 / count) * stats.row(0).head(dim);
    } else {
      /// kaldi reverse: offset = mean
      offset = stats.row(0).head(dim) / count;
    }
  } else {
    Eigen::RowVectorXd mean = stats.row(0).head(dim) / count;
    Eigen::RowVectorXd var =
        stats.row(1).head(dim) / count - mean.array().square().matrix();
    // kaldi floors the per-dim variance at 1e-20 (and warns).
    if ((var.array() < 1.0e-20).any()) {
      KNF_LOGE("WARNING: Flooring cepstral variance to %e", 1.0e-20);
    }
    var = var.cwiseMax(1.0e-20);
    if (!opts.reverse) {
      /// kaldi: scale = 1/sigma, offset = -(mean*scale)
      scale = var.array().sqrt().inverse().matrix();
      offset = -(mean.array() * scale.array()).matrix();
    } else {
      /// kaldi reverse: scale = sigma, offset = mean
      scale = var.array().sqrt().matrix();
      offset = mean;
    }
  }

  Eigen::RowVectorXf scale_f = scale.cast<float>();
  Eigen::RowVectorXf offset_f = offset.cast<float>();
  feats.array().rowwise() *= scale_f.array();
  feats.array().rowwise() += offset_f.array();
}
}  // namespace

void ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts, MatrixF *feats) {
  ApplyCmvnInPlace(stats, opts, *feats);
}

void ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts,
               Eigen::Ref<MatrixF> feats) {
  ApplyCmvnInPlace(stats, opts, feats);
}

MatrixF ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts,
                  const MatrixF &feats) {
  MatrixF ans = feats;
  ApplyCmvnInPlace(stats, opts, ans);
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

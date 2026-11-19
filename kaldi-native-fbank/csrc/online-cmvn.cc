// kaldi-native-fbank/kaldi-native-fbank/csrc/online-cmvn.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include "kaldi-native-fbank/csrc/online-cmvn.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace knf {

std::string OnlineCmvnOptions::ToString() const {
  std::ostringstream os;
  os << "OnlineCmvnOptions(cmn_window=" << cmn_window
     << ", speaker_frames=" << speaker_frames
     << ", global_frames=" << global_frames
     << ", normalize_mean=" << (normalize_mean ? "True" : "False")
     << ", normalize_variance=" << (normalize_variance ? "True" : "False")
     << ", modulus=" << modulus << ", ring_buffer_size=" << ring_buffer_size
     << ", skip_dims=\"" << skip_dims << "\")";
  return os.str();
}

namespace {

std::vector<int32_t> ParseSkipDims(const std::string &skip_dims) {
  // Colon-separated list of integers, like kaldi's --skip-dims.
  std::vector<int32_t> dims;
  if (skip_dims.empty()) return dims;
  std::istringstream is(skip_dims);
  std::string item;
  while (std::getline(is, item, ':')) {
    if (item.empty()) {
      throw std::invalid_argument("Bad --skip-dims option: '" + skip_dims +
                                  "'");
    }
    size_t pos = 0;
    int value;
    try {
      value = std::stoi(item, &pos);
    } catch (const std::exception &) {
      throw std::invalid_argument("Bad --skip-dims option: '" + skip_dims +
                                  "'");
    }
    if (pos != item.size()) {
      throw std::invalid_argument("Bad --skip-dims option: '" + skip_dims +
                                  "'");
    }
    dims.push_back(value);
  }
  return dims;
}

}  // namespace

void OnlineCmvn::SmoothOnlineCmvnStats(const MatrixD &speaker_stats,
                                       const MatrixD &global_stats,
                                       const OnlineCmvnOptions &opts,
                                       MatrixD *stats) {
  int32_t dim = stats->cols() - 1;
  double cur_count = (*stats)(0, dim);
  if (cur_count > 1.001 * opts.cmn_window) {
    throw std::runtime_error("SmoothOnlineCmvnStats: count exceeds cmn_window");
  }
  if (cur_count >= opts.cmn_window) return;

  // Kaldi smooths row 1 (sum-squared stats) unless it takes the
  // row-0-only shortcut, which applies exactly when there are 2-row speaker
  // stats and variance normalization is off.  Keep its branch condition so
  // the smoothed stats (e.g. frozen state) match kaldi exactly.
  bool smooth_row1 = opts.normalize_variance || speaker_stats.rows() != 2;

  if (speaker_stats.rows() != 0) {  // if we have speaker stats..
    double count_from_speaker = opts.cmn_window - cur_count,
           speaker_count = speaker_stats(0, dim);
    if (count_from_speaker > opts.speaker_frames)
      count_from_speaker = opts.speaker_frames;
    if (count_from_speaker > speaker_count) count_from_speaker = speaker_count;
    if (count_from_speaker > 0.0) {
      double scale = count_from_speaker / speaker_count;
      // When normalize_variance is false, kaldi smooths only row 0 (see the
      // row-range shortcut in its SmoothOnlineCmvnStats).
      stats->row(0).noalias() += scale * speaker_stats.row(0);
      if (smooth_row1) stats->row(1).noalias() += scale * speaker_stats.row(1);
    }
    cur_count = (*stats)(0, dim);
  }
  if (cur_count >= opts.cmn_window) return;

  if (global_stats.rows() != 0) {
    double count_from_global = opts.cmn_window - cur_count,
           global_count = global_stats(0, dim);
    if (global_count <= 0.0) {
      throw std::runtime_error(
          "SmoothOnlineCmvnStats: global count must be "
          "positive");
    }
    if (count_from_global > opts.global_frames)
      count_from_global = opts.global_frames;
    if (count_from_global > 0.0) {
      double scale = count_from_global / global_count;
      stats->row(0).noalias() += scale * global_stats.row(0);
      if (smooth_row1) stats->row(1).noalias() += scale * global_stats.row(1);
    }
  } else {
    throw std::runtime_error("Global CMN stats are required");
  }
}

void OnlineCmvn::Init(const OnlineCmvnState &state) {
  // Kaldi only documents global_frames <= speaker_frames <= cmn_window
  // (its OnlineCmvnOptions::Check()); its runtime accepts looser values and
  // clamps in SmoothOnlineCmvnStats, so only structural limits are enforced.
  if (!(opts_.cmn_window > 0 && opts_.modulus > 0 &&
        opts_.ring_buffer_size >= opts_.modulus)) {
    throw std::invalid_argument(
        "Invalid OnlineCmvnOptions: expect 0 < modulus <= ring_buffer_size "
        "and cmn_window > 0");
  }
  if (!opts_.normalize_mean && opts_.normalize_variance) {
    throw std::invalid_argument(
        "normalize_variance requires normalize_mean to be true");
  }
  if (state.global_cmvn_stats.rows() == 0) {
    throw std::invalid_argument("Global CMVN stats are required");
  }
  if (state.global_cmvn_stats.rows() != 2) {
    throw std::invalid_argument("Global CMVN stats must have 2 rows");
  }
  dim_ = state.global_cmvn_stats.cols() - 1;
  if (dim_ <= 0) {
    throw std::invalid_argument("Global CMVN stats must have dim+1 columns");
  }
  if (state.speaker_cmvn_stats.rows() != 0) {
    if (state.speaker_cmvn_stats.rows() != 2 ||
        state.speaker_cmvn_stats.cols() != dim_ + 1) {
      throw std::invalid_argument("Speaker CMVN stats must be 2 x (dim+1)");
    }
    prev_speaker_cmvn_stats_ = state.speaker_cmvn_stats;
    speaker_cmvn_stats_ = state.speaker_cmvn_stats;
  } else {
    prev_speaker_cmvn_stats_.resize(0, 0);
    speaker_cmvn_stats_.resize(0, 0);
  }
  if (state.frozen_state.rows() != 0) {
    if (state.frozen_state.rows() != 2 ||
        state.frozen_state.cols() != dim_ + 1) {
      throw std::invalid_argument("Frozen CMVN state must be 2 x (dim+1)");
    }
    frozen_state_ = state.frozen_state;
  } else {
    frozen_state_.resize(0, 0);
  }
  global_cmvn_stats_ = state.global_cmvn_stats;

  skip_dims_ = ParseSkipDims(opts_.skip_dims);
  for (int32_t d : skip_dims_) {
    if (d < 0 || d >= dim_) {
      throw std::invalid_argument("skip-dims entry out of range");
    }
  }

  stats_.resize(2, dim_ + 1);
  stats_.setZero();
  raw_ring_.assign(opts_.cmn_window, VectorF());
}

OnlineCmvn::OnlineCmvn(const OnlineCmvnOptions &opts,
                       const OnlineCmvnState &state)
    : opts_(opts) {
  Init(state);
}

VectorF OnlineCmvn::AcceptFeature(const float *feat, int32_t dim) {
  if (dim != dim_) {
    throw std::invalid_argument("Feature dim mismatch");
  }
  VectorF v(dim);
  for (int32_t i = 0; i < dim; ++i) v[i] = feat[i];
  return AcceptFeature(v);
}

VectorF OnlineCmvn::AcceptFeature(const VectorF &feat) {
  if (feat.size() != dim_) {
    throw std::invalid_argument("Feature dim mismatch");
  }
  Eigen::VectorXd feat_dbl = feat.cast<double>();

  // Speaker stats accumulate over all frames of the utterance (raw features),
  // like kaldi's OnlineCmvn::GetState().
  if (speaker_cmvn_stats_.rows() == 0) {
    speaker_cmvn_stats_.resize(2, dim_ + 1);
    speaker_cmvn_stats_.setZero();
  }
  speaker_cmvn_stats_(0, dim_) += 1.0;
  speaker_cmvn_stats_.row(0).head(dim_) += feat_dbl.transpose();
  speaker_cmvn_stats_.row(1).head(dim_) +=
      feat_dbl.array().square().matrix().transpose();

  if (!frozen_state_.rows()) {
    UpdateStats(feat);
  }

  VectorF out = feat;
  NormalizeFrame(&out);
  ++num_frames_;
  return out;
}

MatrixF OnlineCmvn::AcceptFeatures(const MatrixF &feats) {
  MatrixF out(feats.rows(), feats.cols());
  for (int32_t t = 0; t < feats.rows(); ++t) {
    out.row(t) = AcceptFeature(feats.row(t).data(), feats.cols()).transpose();
  }
  return out;
}

void OnlineCmvn::UpdateStats(const VectorF &feat) {
  // Sliding-window stats over the last <= cmn_window frames; same add/sub
  // order as kaldi's OnlineCmvn::ComputeStatsForFrame().
  int32_t t = num_frames_;
  Eigen::VectorXd feat_dbl = feat.cast<double>();

  stats_.row(0).head(dim_) += feat_dbl.transpose();
  if (opts_.normalize_variance) {
    stats_.row(1).head(dim_) += feat_dbl.array().square().matrix().transpose();
  }
  stats_(0, dim_) += 1.0;

  if (t >= opts_.cmn_window) {
    // the frame leaving the window is stored in this slot
    const VectorF &prev = raw_ring_[t % opts_.cmn_window];
    Eigen::VectorXd prev_dbl = prev.cast<double>();
    stats_.row(0).head(dim_) -= prev_dbl.transpose();
    if (opts_.normalize_variance) {
      stats_.row(1).head(dim_) -=
          prev_dbl.array().square().matrix().transpose();
    }
    stats_(0, dim_) -= 1.0;
  }
  raw_ring_[t % opts_.cmn_window] = feat;
}

void OnlineCmvn::NormalizeFrame(VectorF *feat) const {
  MatrixD stats(2, dim_ + 1);
  if (frozen_state_.rows() != 0) {
    stats = frozen_state_;
  } else {
    stats = stats_;
    SmoothOnlineCmvnStats(prev_speaker_cmvn_stats_, global_cmvn_stats_, opts_,
                          &stats);
  }

  if (!skip_dims_.empty()) {
    // kaldi's FakeStatsForSomeDims: mean 0, var 1 for skipped dims.
    double count = stats(0, dim_);
    for (int32_t d : skip_dims_) {
      stats(0, d) = 0.0;
      stats(1, d) = count;
    }
  }

  if (!opts_.normalize_mean) return;

  // kaldi's ApplyCmvn, formulas kept expression-for-expression.
  double count = stats(0, dim_);
  if (count < 1.0) {
    throw std::runtime_error(
        "Insufficient stats for cepstral mean and variance normalization");
  }
  if (!opts_.normalize_variance) {
    double inv = -1.0 / count;
    for (int32_t d = 0; d < dim_; ++d) {
      double offset = inv * stats(0, d);
      (*feat)[d] += offset;
    }
    return;
  }
  for (int32_t d = 0; d < dim_; ++d) {
    double mean = stats(0, d) / count;
    double var = (stats(1, d) / count) - mean * mean;
    const double floor = 1.0e-20;
    if (var < floor) var = floor;
    double scale = 1.0 / std::sqrt(var);
    if (scale != scale || 1 / scale == 0.0) {
      throw std::runtime_error(
          "NaN or infinity in cepstral mean/variance computation");
    }
    double offset = -(mean * scale);
    (*feat)[d] = (*feat)[d] * scale + offset;
  }
}

void OnlineCmvn::Freeze() {
  MatrixD stats = stats_;
  SmoothOnlineCmvnStats(prev_speaker_cmvn_stats_, global_cmvn_stats_, opts_,
                        &stats);
  frozen_state_ = stats;
}

OnlineCmvnState OnlineCmvn::GetState() const {
  OnlineCmvnState state;
  state.speaker_cmvn_stats = speaker_cmvn_stats_;
  state.global_cmvn_stats = global_cmvn_stats_;
  state.frozen_state = frozen_state_;
  return state;
}

}  // namespace knf

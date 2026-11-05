/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-matrix.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

/**
 * @file feature-matrix.h
 * @brief Eigen matrix aliases used to hold features and statistics.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_MATRIX_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_MATRIX_H_

#include <stdint.h>

#include "Eigen/Dense"

namespace knf {

/// Row-major feature matrix: rows are frames, columns are features.
/// Row-major matches kaldi's Matrix<float> memory layout, so golden data
/// produced by kaldi can be compared directly.
using MatrixF =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Double-precision accumulator/statistics matrix (e.g. cmvn stats),
/// matching kaldi's Matrix<double> layout.
using MatrixD =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// 1-D feature/aux vector (float).
using VectorF = Eigen::VectorXf;

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_MATRIX_H_

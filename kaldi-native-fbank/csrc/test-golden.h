/// kaldi-native-fbank/kaldi-native-fbank/csrc/test-golden.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

/**
 * @file test-golden.h
 * @brief Shared helpers for golden-file unit tests.
 *
 * The golden test data (kaldi-generated .npy matrices) lives in
 * tests/data/feat_golden; see the README there for how it was generated.
 * These helpers read that data and locate it from the source tree; they are
 * shared by the test-*.cc gtests.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_TEST_GOLDEN_H_
#define KALDI_NATIVE_FBANK_CSRC_TEST_GOLDEN_H_

#include <stdint.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/// Minimal .npy (v1/v2, C-order, float32/float64) reader for golden files.
///
/// @param [in] path  Path of the .npy file.
/// @param [out] out  Matrix filled with the file contents.
/// @return  true on success.
inline bool LoadNpy(const std::string &path, MatrixD *out) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    return false;
  }
  char magic[6];
  is.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    return false;
  }
  char ver[2];
  is.read(ver, 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16 = 0;
    is.read(reinterpret_cast<char *>(&h16), 2);
    hlen = h16;
  } else {
    is.read(reinterpret_cast<char *>(&hlen), 4);
  }
  std::string header(hlen, '\0');
  is.read(&header[0], hlen);

  auto dp = header.find("'descr': '") + 10;
  std::string descr = header.substr(dp, 3);
  auto sp = header.find("'shape': (") + 10;
  auto ep = header.find(')', sp);
  std::string shape = header.substr(sp, ep - sp);
  int32_t rows = 0;
  int32_t cols = 0;
  if (shape.find(',') != std::string::npos) {
    std::string col_part = shape.substr(shape.find(',') + 1);
    rows = std::stoi(shape.substr(0, shape.find(',')));
    // 1-D arrays have shape "(N,)", i.e. an empty column part.
    cols = col_part.find_first_not_of(" \t") == std::string::npos
               ? 1
               : std::stoi(col_part);
  } else {
    rows = std::stoi(shape);
    cols = 1;
  }
  *out = MatrixD(rows, cols);
  if (descr == "<f8") {
    for (int32_t r = 0; r < rows; ++r) {
      is.read(reinterpret_cast<char *>((*out).row(r).data()),
              cols * sizeof(double));
    }
  } else if (descr == "<i2") {
    // int16 (e.g. raw wav samples stored as .npy).
    std::vector<int16_t> tmp(cols);
    for (int32_t r = 0; r < rows; ++r) {
      is.read(reinterpret_cast<char *>(tmp.data()), cols * sizeof(int16_t));
      for (int32_t c = 0; c < cols; ++c) {
        (*out)(r, c) = tmp[c];
      }
    }
  } else {
    std::vector<float> tmp(cols);
    for (int32_t r = 0; r < rows; ++r) {
      is.read(reinterpret_cast<char *>(tmp.data()), cols * sizeof(float));
      for (int32_t c = 0; c < cols; ++c) {
        (*out)(r, c) = tmp[c];
      }
    }
  }
  return true;
}

/// Locate tests/data/feat_golden relative to this header.
///
/// Uses plain string operations (std::filesystem is not portable).  The
/// header lives in the same directory as the test-*.cc files, so the
/// resolved path is the same for every test.
inline std::string GoldenDir() {
  std::string dir = __FILE__;
  std::string::size_type pos = dir.find_last_of("/\\");
  dir = (pos == std::string::npos) ? std::string(".") : dir.substr(0, pos);
  return dir + "/../../tests/data/feat_golden";
}

/// Return the maximum absolute difference between a float matrix and a
/// double matrix of the same shape (for golden comparisons).
inline double MaxAbsDiff(const MatrixF &a, const MatrixD &b) {
  double max_diff = 0.0;
  for (int32_t r = 0; r < a.rows(); ++r) {
    for (int32_t c = 0; c < a.cols(); ++c) {
      double diff = std::fabs(static_cast<double>(a(r, c)) - b(r, c));
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
  }
  return max_diff;
}

/// Comparison tolerance for golden matrices whose kaldi reference was
/// produced by matrix multiplications or long dot products (the LDA
/// transform-feats outputs and the pitch NCCF): kaldi's BLAS and Eigen
/// accumulate the sums in different orders, which gives per-platform float
/// noise of up to a few 1e-5.  Observed maxima for the LDA goldens:
/// 1.5e-5 (linux-arm32), 1.1e-5 (windows-x64), 2.7e-5 (windows-arm64);
/// the golden-generating platform agrees at <= 1e-5.  1e-4 leaves headroom
/// while still catching real implementation differences (>= 1e-3).
/// Pipelines of sequential reductions (cmvn/delta/splice) compare at the
/// tighter 1e-5 instead.
constexpr double kGoldenDotProductTol = 1e-4;

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_TEST_GOLDEN_H_

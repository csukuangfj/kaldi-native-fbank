// kaldi-native-fbank/kaldi-native-fbank/csrc/test-cmvn.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include <stdint.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-cmvn.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

namespace {

// Minimal .npy (v1, C-order, float32/float64) reader for golden files.
bool LoadNpy(const std::string &path, MatrixD *out) {
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
    rows = std::stoi(shape.substr(0, shape.find(',')));
    cols = std::stoi(shape.substr(shape.find(',') + 1));
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

std::string GoldenDir() {
  // Locate tests/data/feat_golden relative to this source file, without
  // std::filesystem (it is not portable).
  std::string dir = __FILE__;
  std::string::size_type pos = dir.find_last_of("/\\");
  dir = (pos == std::string::npos) ? std::string(".") : dir.substr(0, pos);
  return dir + "/../../tests/data/feat_golden";
}

}  // namespace

TEST(Cmvn, InitAndAccumulate) {
  MatrixD stats;
  InitCmvnStats(3, &stats);
  EXPECT_EQ(stats.rows(), 2);
  EXPECT_EQ(stats.cols(), 4);
  EXPECT_EQ(stats.sum(), 0.0);

  // hand case: two frames [1 2 3], [3 4 5]
  float f0[3] = {1, 2, 3};
  float f1[3] = {3, 4, 5};
  AccCmvnStats(f0, 3, 1.0f, &stats);
  AccCmvnStats(f1, 3, 1.0f, &stats);
  EXPECT_DOUBLE_EQ(stats(0, 3), 2.0);   // count
  EXPECT_DOUBLE_EQ(stats(0, 0), 4.0);   // sum of dim0
  EXPECT_DOUBLE_EQ(stats(1, 0), 10.0);  // sumsq of dim0
  EXPECT_DOUBLE_EQ(stats(1, 2), 34.0);  // sumsq of dim2: 9+25
}

TEST(Cmvn, GoldenAishell3Typical) {
  MatrixD feats;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/aishell3_typical/mfcc.utt.npy", &feats));
  MatrixF feats_f = feats.cast<float>();

  MatrixD stats = ComputeCmvnStats(feats_f);
  MatrixD ref_stats;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/aishell3_typical/cmvn_stats.utt.npy", &ref_stats));
  ASSERT_EQ(stats.rows(), ref_stats.rows());
  ASSERT_EQ(stats.cols(), ref_stats.cols());
  for (int32_t r = 0; r < stats.rows(); ++r) {
    for (int32_t c = 0; c < stats.cols(); ++c) {
      double rel = std::abs(stats(r, c) - ref_stats(r, c)) /
                   (std::abs(ref_stats(r, c)) + 1.0);
      EXPECT_LE(rel, 1e-5) << "row " << r << " col " << c;
    }
  }

  // mean-only cmvn against golden
  CmvnOptions opts;
  MatrixF normed = ApplyCmvn(stats, opts, feats_f);
  MatrixD ref_normed;
  ASSERT_TRUE(LoadNpy(GoldenDir() + "/golden/aishell3_typical/cmvn.utt.npy",
                      &ref_normed));
  ASSERT_EQ(normed.rows(), ref_normed.rows());
  ASSERT_EQ(normed.cols(), ref_normed.cols());
  for (int32_t r = 0; r < normed.rows(); ++r) {
    for (int32_t c = 0; c < normed.cols(); ++c) {
      EXPECT_NEAR(normed(r, c), ref_normed(r, c), 1e-5) << "row " << r;
    }
  }

  // norm_vars consistency: mean ~0, var ~1
  opts.norm_vars = true;
  MatrixF nv = ApplyCmvn(stats, opts, feats_f);
  for (int32_t c = 0; c < nv.cols(); ++c) {
    double m = nv.col(c).mean();
    EXPECT_NEAR(m, 0.0, 1e-3);
  }
}

TEST(Cmvn, GoldenEdgeSilence) {
  MatrixD feats;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/edge_cases/silence_0.5s.mfcc.utt.npy", &feats));
  MatrixF feats_f = feats.cast<float>();
  MatrixD stats = ComputeCmvnStats(feats_f);
  MatrixD ref_stats;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/edge_cases/silence_0.5s.cmvn_stats.utt.npy",
      &ref_stats));
  for (int32_t r = 0; r < 2; ++r) {
    for (int32_t c = 0; c < stats.cols(); ++c) {
      EXPECT_NEAR(stats(r, c), ref_stats(r, c), 1e-5);
    }
  }

  // apply-cmvn must reproduce the golden output
  MatrixD ref_cmvn;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/edge_cases/silence_0.5s.cmvn.utt.npy", &ref_cmvn));
  CmvnOptions m;
  MatrixF mm = ApplyCmvn(stats, m, feats_f);
  for (int32_t r = 0; r < mm.rows(); ++r) {
    for (int32_t c = 0; c < mm.cols(); ++c) {
      EXPECT_NEAR(mm(r, c), ref_cmvn(r, c), 1e-5);
    }
  }
}

TEST(Cmvn, AccumulatorMatchesBatch) {
  MatrixD feats;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/aishell3_typical/mfcc.utt.npy", &feats));
  MatrixF feats_f = feats.cast<float>();
  CmvnAccumulator acc;
  acc.Accumulate(feats_f);
  MatrixD batch = ComputeCmvnStats(feats_f);
  for (int32_t r = 0; r < 2; ++r) {
    for (int32_t c = 0; c < batch.cols(); ++c) {
      EXPECT_NEAR(acc.Stats()(r, c), batch(r, c), 1e-6);
    }
  }
}

}  // namespace knf

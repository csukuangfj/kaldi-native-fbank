// kaldi-native-fbank/kaldi-native-fbank/csrc/test-cmvn.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include <cmath>
#include <cstdint>
#include <string>

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-cmvn.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/test-golden.h"

namespace knf {

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

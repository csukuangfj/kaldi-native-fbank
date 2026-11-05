/// kaldi-native-fbank/kaldi-native-fbank/csrc/test-delta.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include <stdexcept>
#include <string>

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-delta.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/test-golden.h"

namespace knf {

TEST(Delta, HandCase) {
  // x = [1, 2, 3] (T=3, D=1), order=1, window=1:
  // scales1 = [-0.5, 0, 0.5], delta(t) = -0.5*x(t-1) + 0.5*x(t+1) with
  // clamped edges -> [0.5, 1.0, 0.5].
  MatrixF feats(3, 1);
  feats << 1.0f, 2.0f, 3.0f;

  DeltaOptions opts;
  opts.order = 1;
  opts.window = 1;
  MatrixF out = ComputeDeltas(opts, feats);

  ASSERT_EQ(out.rows(), 3);
  ASSERT_EQ(out.cols(), 2);
  EXPECT_FLOAT_EQ(out(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(out(1, 0), 2.0f);
  EXPECT_FLOAT_EQ(out(2, 0), 3.0f);
  EXPECT_FLOAT_EQ(out(0, 1), 0.5f);
  EXPECT_FLOAT_EQ(out(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(out(2, 1), 0.5f);
}

TEST(Delta, OrderZeroIsBaseline) {
  MatrixF feats(4, 3);
  feats.setRandom();
  DeltaOptions opts;
  opts.order = 0;
  MatrixF out = ComputeDeltas(opts, feats);
  ASSERT_EQ(out.rows(), feats.rows());
  ASSERT_EQ(out.cols(), feats.cols());
  EXPECT_EQ((out - feats).cwiseAbs().maxCoeff(), 0.0f);
}

TEST(Delta, GoldenAishell3Typical) {
  MatrixD feats;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/aishell3_typical/cmvn.utt.npy", &feats));
  MatrixF feats_f = feats.cast<float>();

  DeltaOptions opts;  // order=2, window=2
  MatrixF out = ComputeDeltas(opts, feats_f);

  MatrixD ref;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/aishell3_typical/deltas.utt.npy", &ref));
  ASSERT_EQ(out.rows(), ref.rows());
  ASSERT_EQ(out.cols(), ref.cols());
  EXPECT_LT(MaxAbsDiff(out, ref), 1e-5);
}

TEST(Delta, GoldenEnglishMfaTypical) {
  MatrixD feats;
  ASSERT_TRUE(LoadNpy(GoldenDir() + "/golden/english_mfa_typical/cmvn.utt.npy",
                      &feats));
  MatrixF feats_f = feats.cast<float>();

  DeltaOptions opts;
  MatrixF out = ComputeDeltas(opts, feats_f);

  MatrixD ref;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/english_mfa_typical/deltas.utt.npy", &ref));
  ASSERT_EQ(out.rows(), ref.rows());
  ASSERT_EQ(out.cols(), ref.cols());
  EXPECT_LT(MaxAbsDiff(out, ref), 1e-5);
}

TEST(Delta, GoldenEdgeShort2f) {
  MatrixD feats;
  ASSERT_TRUE(LoadNpy(GoldenDir() + "/golden/edge_cases/short_2f.cmvn.utt.npy",
                      &feats));
  MatrixF feats_f = feats.cast<float>();

  DeltaOptions opts;
  MatrixF out = ComputeDeltas(opts, feats_f);

  MatrixD ref;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/edge_cases/short_2f.deltas.utt.npy", &ref));
  ASSERT_EQ(out.rows(), ref.rows());
  ASSERT_EQ(out.cols(), ref.cols());
  EXPECT_LT(MaxAbsDiff(out, ref), 1e-5);
}

TEST(Delta, Errors) {
  DeltaOptions opts;
  MatrixF empty(0, 3);
  EXPECT_THROW(ComputeDeltas(opts, empty), std::invalid_argument);

  MatrixF feats(2, 2);
  feats.setZero();
  DeltaOptions bad_order;
  bad_order.order = -1;
  EXPECT_THROW(ComputeDeltas(bad_order, feats), std::invalid_argument);

  DeltaOptions bad_window;
  bad_window.window = 0;
  EXPECT_THROW(ComputeDeltas(bad_window, feats), std::invalid_argument);
}

}  // namespace knf

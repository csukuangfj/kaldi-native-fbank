/// kaldi-native-fbank/kaldi-native-fbank/csrc/test-splice.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include <stdint.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/feature-splice.h"

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

// kaldi splice-feats is a pure copy/concatenate: expect exact equality.
double MaxAbsDiff(const MatrixF &a, const MatrixD &b) {
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

}  // namespace

TEST(Splice, HandCaseEdgeReplication) {
  // 3 frames of dim 2: [1 2], [3 4], [5 6]; left=1, right=1 -> dim 6.
  MatrixF feats(3, 2);
  feats << 1, 2, 3, 4, 5, 6;

  SpliceOptions opts;
  opts.left_context = 1;
  opts.right_context = 1;
  MatrixF spliced = SpliceFrames(feats, opts);

  ASSERT_EQ(spliced.rows(), 3);
  ASSERT_EQ(spliced.cols(), 6);

  // t=0: clamps t-1 to 0 -> [1 2  1 2  3 4]
  EXPECT_FLOAT_EQ(spliced(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(spliced(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(spliced(0, 2), 1.0f);
  EXPECT_FLOAT_EQ(spliced(0, 3), 2.0f);
  EXPECT_FLOAT_EQ(spliced(0, 4), 3.0f);
  EXPECT_FLOAT_EQ(spliced(0, 5), 4.0f);
  // t=1: [1 2  3 4  5 6]
  EXPECT_FLOAT_EQ(spliced(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(spliced(1, 3), 4.0f);
  EXPECT_FLOAT_EQ(spliced(1, 5), 6.0f);
  // t=2: clamps t+1 to 2 -> [3 4  5 6  5 6]
  EXPECT_FLOAT_EQ(spliced(2, 0), 3.0f);
  EXPECT_FLOAT_EQ(spliced(2, 2), 5.0f);
  EXPECT_FLOAT_EQ(spliced(2, 5), 6.0f);
}

TEST(Splice, SingleFrame) {
  MatrixF feats(1, 2);
  feats << 7, 8;
  SpliceOptions opts;
  opts.left_context = 2;
  opts.right_context = 0;
  MatrixF spliced = SpliceFrames(feats, opts);
  ASSERT_EQ(spliced.rows(), 1);
  ASSERT_EQ(spliced.cols(), 6);
  for (int32_t c = 0; c < 6; c += 2) {
    EXPECT_FLOAT_EQ(spliced(0, c), 7.0f);
    EXPECT_FLOAT_EQ(spliced(0, c + 1), 8.0f);
  }
}

TEST(Splice, GoldenAishell3Typical) {
  MatrixD feats;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/aishell3_typical/cmvn.utt.npy", &feats));
  MatrixF feats_f = feats.cast<float>();

  SpliceOptions opts;  // left=3, right=3
  MatrixF spliced = SpliceFrames(feats_f, opts);

  MatrixD ref;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/aishell3_typical/spliced.utt.npy", &ref));
  ASSERT_EQ(spliced.rows(), ref.rows());
  ASSERT_EQ(spliced.cols(), ref.cols());
  EXPECT_EQ(MaxAbsDiff(spliced, ref), 0.0);
}

TEST(Splice, GoldenEdgeShort2f) {
  MatrixD feats;
  ASSERT_TRUE(LoadNpy(GoldenDir() + "/golden/edge_cases/short_2f.cmvn.utt.npy",
                      &feats));
  MatrixF feats_f = feats.cast<float>();

  SpliceOptions opts;
  MatrixF spliced = SpliceFrames(feats_f, opts);

  MatrixD ref;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/edge_cases/short_2f.spliced.utt.npy", &ref));
  ASSERT_EQ(spliced.rows(), ref.rows());
  ASSERT_EQ(spliced.cols(), ref.cols());
  EXPECT_EQ(MaxAbsDiff(spliced, ref), 0.0);
}

TEST(Splice, GoldenEdgeSilence) {
  MatrixD feats;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/edge_cases/silence_0.5s.cmvn.utt.npy", &feats));
  MatrixF feats_f = feats.cast<float>();

  SpliceOptions opts;
  MatrixF spliced = SpliceFrames(feats_f, opts);

  MatrixD ref;
  ASSERT_TRUE(LoadNpy(
      GoldenDir() + "/golden/edge_cases/silence_0.5s.spliced.utt.npy", &ref));
  ASSERT_EQ(spliced.rows(), ref.rows());
  ASSERT_EQ(spliced.cols(), ref.cols());
  EXPECT_EQ(MaxAbsDiff(spliced, ref), 0.0);
}

TEST(Splice, Errors) {
  SpliceOptions opts;
  MatrixF empty(0, 3);
  EXPECT_THROW(SpliceFrames(empty, opts), std::invalid_argument);

  MatrixF feats(2, 2);
  feats.setZero();
  SpliceOptions bad;
  bad.left_context = -1;
  EXPECT_THROW(SpliceFrames(feats, bad), std::invalid_argument);
}

}  // namespace knf

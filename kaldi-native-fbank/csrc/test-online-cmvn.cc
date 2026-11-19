/// kaldi-native-fbank/kaldi-native-fbank/csrc/test-online-cmvn.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/online-cmvn.h"

namespace knf {

namespace {

// Global stats with zero mean and unit variance per dim.
MatrixD UnitGlobalStats(int32_t dim, double count = 1000.0) {
  MatrixD stats = MatrixD::Zero(2, dim + 1);
  stats(0, dim) = count;
  for (int32_t d = 0; d < dim; d++) stats(1, d) = count;
  return stats;
}

OnlineCmvn MakeCmvn(int32_t dim, const OnlineCmvnOptions &opts,
                    const OnlineCmvnState &state) {
  return OnlineCmvn(opts, state);
}

MatrixF RandomFeats(int32_t T, int32_t dim, unsigned seed = 42) {
  // deterministic, stationary pseudo-random values in a realistic range
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  MatrixF feats(T, dim);
  for (int32_t t = 0; t < T; t++) {
    for (int32_t d = 0; d < dim; d++) feats(t, d) = dist(gen);
  }
  return feats;
}

}  // namespace

TEST(OnlineCmvn, OptionsDefaultsMatchKaldi) {
  OnlineCmvnOptions opts;
  EXPECT_EQ(opts.cmn_window, 600);
  EXPECT_EQ(opts.speaker_frames, 600);
  EXPECT_EQ(opts.global_frames, 200);
  EXPECT_TRUE(opts.normalize_mean);
  EXPECT_FALSE(opts.normalize_variance);
}

TEST(OnlineCmvn, ShapesAndCount) {
  OnlineCmvnOptions opts;
  OnlineCmvn cmvn(opts, OnlineCmvnState(UnitGlobalStats(6)));
  MatrixF feats = RandomFeats(200, 6);
  MatrixF out = cmvn.AcceptFeatures(feats);
  EXPECT_EQ(out.rows(), feats.rows());
  EXPECT_EQ(out.cols(), feats.cols());
  EXPECT_EQ(cmvn.NumFramesAccepted(), 200);
  EXPECT_EQ(cmvn.Dim(), 6);
}

TEST(OnlineCmvn, OutputFiniteAndCentered) {
  // global_frames=0: normalization uses only the running window, so the tail
  // is centered.
  OnlineCmvnOptions opts;
  opts.cmn_window = 100;
  opts.global_frames = 0;
  OnlineCmvn cmvn(opts, OnlineCmvnState(UnitGlobalStats(4)));
  MatrixF out = cmvn.AcceptFeatures(RandomFeats(300, 4));
  for (int32_t t = 0; t < out.rows(); t++) {
    for (int32_t d = 0; d < out.cols(); d++) {
      ASSERT_TRUE(std::isfinite(out(t, d)));
    }
  }
  for (int32_t d = 0; d < 4; d++) {
    double mean = 0.0;
    for (int32_t t = 200; t < 300; t++) mean += out(t, d);
    // residual is statistical noise of the random data; the exact per-frame
    // math is checked in MatchesNaiveReference.
    EXPECT_LT(std::fabs(mean / 100.0), 1.0) << "dim " << d;
  }
}

// Randomized comparison against a brute-force reference, following kaldi's
// UnitTestOnlineCmvn (feat/feature-functions-test.cc): many iterations with
// random dimensions, lengths and options; the expected output is computed
// from first principles and compared within 1e-4.
TEST(OnlineCmvn, MatchesNaiveReference) {
  for (int iter = 0; iter < 200; iter++) {
    int32_t T = 1 + iter % 7 * 10;         // 1..61 frames
    int32_t dim = 1 + iter % 8;
    OnlineCmvnOptions opts;
    opts.cmn_window = 5 + iter % 45;
    opts.speaker_frames = 1 + iter % 60;
    opts.global_frames = 1 + iter % 30;
    opts.normalize_variance = (iter % 2 == 0);
    bool with_speaker = (iter % 3 == 0);
    bool with_skip = (iter % 5 == 0);
    if (with_skip) opts.skip_dims = "0";

    MatrixF feats = RandomFeats(T, dim, 1000 + iter);
    // plausible global stats: stats of some random frames
    MatrixD glob = MatrixD::Zero(2, dim + 1);
    {
      MatrixF g = RandomFeats(50, dim, 7000 + iter);
      for (int32_t t = 0; t < 50; t++) {
        glob(0, dim) += 1.0;
        for (int32_t d = 0; d < dim; d++) {
          glob(0, d) += g(t, d);
          glob(1, d) += g(t, d) * g(t, d);
        }
      }
    }
    MatrixD spk;
    if (with_speaker) {
      spk = MatrixD::Zero(2, dim + 1);
      MatrixF g = RandomFeats(30, dim, 8000 + iter);
      for (int32_t t = 0; t < 30; t++) {
        spk(0, dim) += 1.0;
        for (int32_t d = 0; d < dim; d++) {
          spk(0, d) += g(t, d);
          spk(1, d) += g(t, d) * g(t, d);
        }
      }
    }
    OnlineCmvnState state;
    state.global_cmvn_stats = glob;
    state.speaker_cmvn_stats = spk;
    OnlineCmvn cmvn(opts, state);
    MatrixF out = cmvn.AcceptFeatures(feats);

    // brute-force reference (mirrors SmoothOnlineCmvnStats + ApplyCmvn)
    for (int32_t t = 0; t < T; t++) {
      int32_t lo = t - opts.cmn_window + 1 < 0 ? 0 : t - opts.cmn_window + 1;
      int32_t n = t - lo + 1;
      for (int32_t d = 0; d < dim; d++) {
        if (with_skip && d == 0) {
          EXPECT_EQ(out(t, d), feats(t, d));
          continue;
        }
        double sum = 0.0, sumsq = 0.0;
        for (int32_t u = lo; u <= t; u++) {
          sum += feats(u, d);
          sumsq += feats(u, d) * feats(u, d);
        }
        double count = n;
        // speaker smoothing
        if (with_speaker) {
          double take = std::min(static_cast<double>(opts.cmn_window) - count,
                                 static_cast<double>(opts.speaker_frames));
          take = std::min(take, spk(0, dim));
          if (take > 0.0) {
            sum += take / spk(0, dim) * spk(0, d);
            if (opts.normalize_variance) sumsq += take / spk(0, dim) * spk(1, d);
            count += take;
          }
        }
        // global smoothing
        if (count < opts.cmn_window) {
          double take = std::min(static_cast<double>(opts.cmn_window) - count,
                                 static_cast<double>(opts.global_frames));
          take = std::min(take, glob(0, dim));
          if (take > 0.0) {
            sum += take / glob(0, dim) * glob(0, d);
            if (opts.normalize_variance) sumsq += take / glob(0, dim) * glob(1, d);
            count += take;
          }
        }
        double mean = sum / count;
        double want = feats(t, d) - mean;
        if (opts.normalize_variance) {
          double var = sumsq / count - mean * mean;
          var = std::max(var, 1.0e-20);
          want = (feats(t, d) - mean) / std::sqrt(var);
        }
        EXPECT_NEAR(out(t, d), want, 1e-4) << "iter=" << iter << " t=" << t
                                          << " d=" << d;
      }
    }
  }
}

// Utterance start: too little data for the window, so the normalization must
// back off to global stats (kaldi's SmoothOnlineCmvnStats).
TEST(OnlineCmvn, UtteranceStartGlobalBacking) {
  const int32_t dim = 1;
  MatrixD glob = MatrixD::Zero(2, dim + 1);
  glob(0, 0) = 50.0;
  glob(0, dim) = 100.0;  // global mean 0.5

  OnlineCmvnOptions opts;
  opts.normalize_variance = false;
  OnlineCmvn cmvn(opts, OnlineCmvnState(glob));

  VectorF x(dim);
  x(0) = 2.0f;
  VectorF y = cmvn.AcceptFeature(x);

  // window: sum=2, count=1.  global takes min(cmn_window-1, global_frames)=200
  // frames worth: sum += 200/100 * 50 = 100 -> sum=102, count=201.
  double want = 2.0 - (2.0 + 200.0 / 100.0 * 50.0) / (1.0 + 200.0);
  EXPECT_NEAR(y(0), want, 1e-5);
}

// Utterance start with speaker stats: speaker back-off is applied first and
// clamped by speaker_frames; global back-off fills the rest.
TEST(OnlineCmvn, UtteranceStartSpeakerBacking) {
  const int32_t dim = 1;
  MatrixD glob = MatrixD::Zero(2, dim + 1);
  glob(0, 0) = 50.0;
  glob(0, dim) = 100.0;  // global mean 0.5
  MatrixD spk = MatrixD::Zero(2, dim + 1);
  spk(0, 0) = 30.0;
  spk(0, dim) = 10.0;  // speaker mean 3

  OnlineCmvnOptions opts;
  opts.speaker_frames = 5;  // clamp: only 5 of the 10 speaker frames count
  OnlineCmvnState state;
  state.global_cmvn_stats = glob;
  state.speaker_cmvn_stats = spk;
  OnlineCmvn cmvn(opts, state);

  VectorF x(dim);
  x(0) = 2.0f;
  VectorF y = cmvn.AcceptFeature(x);

  // window: sum=2, count=1.
  // speaker: take min(599, 5, 10) = 5 -> sum += 5/10 * 30 = 15 -> 17, count 6.
  // global:  take min(594, 200) = 200 -> sum += 200/100 * 50 = 100 -> 117,
  //          count 206.
  double want = 2.0 - (2.0 + 15.0 + 100.0) / (1.0 + 5.0 + 200.0);
  EXPECT_NEAR(y(0), want, 1e-5);
}

// First frames with a not-yet-full window and no back-off at all: frame t is
// normalized with the mean of frames 0..t (frame 0 maps to 0 exactly).
TEST(OnlineCmvn, FirstFramesPartialWindow) {
  const int32_t dim = 1, c = 5;
  OnlineCmvnOptions opts;
  opts.cmn_window = c;
  opts.global_frames = 0;  // no back-off: pure past-only window
  OnlineCmvn cmvn(opts, OnlineCmvnState(UnitGlobalStats(dim)));

  MatrixF feats(8, dim);
  for (int32_t t = 0; t < 8; t++) feats(t, 0) = 1.0f + t;
  MatrixF out = cmvn.AcceptFeatures(feats);

  for (int32_t t = 0; t < 8; t++) {
    int32_t lo = t - c + 1 < 0 ? 0 : t - c + 1;
    double sum = 0.0;
    for (int32_t u = lo; u <= t; u++) sum += feats(u, 0);
    double want = feats(t, 0) - sum / (t - lo + 1);
    EXPECT_NEAR(out(t, 0), want, 1e-5) << "t=" << t;
  }
  EXPECT_NEAR(out(0, 0), 0.0, 1e-6);  // single-frame window centers itself
}

TEST(OnlineCmvn, NormalizeVarianceUnitScale) {
  // With variance normalization the tail has unit-ish variance (sanity
  // check; the exact math is covered by MatchesNaiveReference).
  OnlineCmvnOptions opts;
  opts.cmn_window = 100;
  opts.normalize_variance = true;
  OnlineCmvn cmvn(opts, OnlineCmvnState(UnitGlobalStats(4)));
  MatrixF out = cmvn.AcceptFeatures(RandomFeats(400, 4));
  for (int32_t d = 0; d < 4; d++) {
    double mean = 0.0;
    for (int32_t t = 300; t < 400; t++) mean += out(t, d);
    mean /= 100.0;
    double var = 0.0;
    for (int32_t t = 300; t < 400; t++) {
      var += (out(t, d) - mean) * (out(t, d) - mean);
    }
    var /= 100.0;
    EXPECT_NEAR(var, 1.0, 0.2) << "dim " << d;
  }
}

TEST(OnlineCmvn, SkipDimsUntouched) {
  OnlineCmvnOptions opts;
  opts.skip_dims = "0:2";
  OnlineCmvn cmvn(opts, OnlineCmvnState(UnitGlobalStats(4)));
  MatrixF feats = RandomFeats(100, 4);
  MatrixF out = cmvn.AcceptFeatures(feats);
  for (int32_t t = 0; t < 100; t++) {
    EXPECT_EQ(out(t, 0), feats(t, 0));
    EXPECT_EQ(out(t, 2), feats(t, 2));
  }
}

TEST(OnlineCmvn, FreezeGivesConstantOffset) {
  OnlineCmvnOptions opts;
  OnlineCmvn cmvn(opts, OnlineCmvnState(UnitGlobalStats(5)));
  MatrixF feats = RandomFeats(200, 5);
  cmvn.AcceptFeatures(feats);
  cmvn.Freeze();
  MatrixF tail = cmvn.AcceptFeatures(feats.block(0, 0, 10, 5));
  for (int32_t d = 0; d < 5; d++) {
    double diff0 = feats(0, d) - tail(0, d);
    for (int32_t t = 1; t < 10; t++) {
      EXPECT_NEAR(feats(t, d) - tail(t, d), diff0, 1e-4);
    }
  }
}

TEST(OnlineCmvn, StateHandoverFrozen) {
  OnlineCmvnOptions opts;
  MatrixF feats = RandomFeats(200, 5);

  OnlineCmvn one(opts, OnlineCmvnState(UnitGlobalStats(5)));
  one.AcceptFeatures(feats.block(0, 0, 80, 5));
  one.Freeze();
  MatrixF tail = one.AcceptFeatures(feats.block(80, 0, 120, 5));

  OnlineCmvn first(opts, OnlineCmvnState(UnitGlobalStats(5)));
  first.AcceptFeatures(feats.block(0, 0, 80, 5));
  first.Freeze();
  OnlineCmvn second(opts, first.GetState());
  MatrixF tail2 = second.AcceptFeatures(feats.block(80, 0, 120, 5));

  for (int32_t t = 0; t < 120; t++) {
    for (int32_t d = 0; d < 5; d++) {
      EXPECT_NEAR(tail(t, d), tail2(t, d), 1e-5);
    }
  }
}

TEST(OnlineCmvn, StateHandoverUsesSpeakerStats) {
  OnlineCmvnOptions opts;
  MatrixF feats = RandomFeats(200, 5);

  OnlineCmvn first(opts, OnlineCmvnState(UnitGlobalStats(5)));
  first.AcceptFeatures(feats.block(0, 0, 80, 5));
  OnlineCmvn second(opts, first.GetState());
  MatrixF adapted = second.AcceptFeatures(feats.block(0, 0, 80, 5));

  OnlineCmvn fresh(opts, OnlineCmvnState(UnitGlobalStats(5)));
  MatrixF plain = fresh.AcceptFeatures(feats.block(0, 0, 80, 5));

  double max_diff = 0.0;
  for (int32_t t = 0; t < 80; t++) {
    for (int32_t d = 0; d < 5; d++) {
      max_diff = std::max(max_diff,
                          static_cast<double>(std::fabs(adapted(t, d) -
                                                        plain(t, d))));
    }
  }
  EXPECT_GT(max_diff, 1e-3);  // speaker stats must change the normalization
}

TEST(OnlineCmvn, SmoothOnlineCmvnStatsCount) {
  OnlineCmvnOptions opts;
  opts.global_frames = 100;
  MatrixD stats = MatrixD::Zero(2, 6);
  stats(0, 5) = 10;  // window count
  MatrixD smoothed = stats;
  OnlineCmvn::SmoothOnlineCmvnStats(MatrixD(), UnitGlobalStats(5), opts,
                                    &smoothed);
  EXPECT_NEAR(smoothed(0, 5), 110.0, 1e-9);  // 10 window + 100 global
}

TEST(OnlineCmvn, InvalidOptionsThrow) {
  OnlineCmvnOptions opts;
  opts.normalize_mean = false;
  opts.normalize_variance = true;
  EXPECT_THROW(OnlineCmvn(opts, OnlineCmvnState(UnitGlobalStats(3))),
               std::invalid_argument);
  EXPECT_THROW(OnlineCmvn(OnlineCmvnOptions(), OnlineCmvnState()),
               std::invalid_argument);  // no global stats
}

}  // namespace knf

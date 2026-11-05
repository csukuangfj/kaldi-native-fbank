// kaldi-native-fbank/kaldi-native-fbank/csrc/test-pitch.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/feature-pitch.h"
#include "kaldi-native-fbank/csrc/test-golden.h"

namespace knf {

namespace {

// Load the raw samples of a golden utterance as float (kaldi semantics:
// raw int16 magnitudes, not divided by 32768).
std::vector<float> LoadSamples(const std::string &name) {
  MatrixD tmp;
  EXPECT_TRUE(LoadNpy(GoldenDir() + "/" + name, &tmp));
  std::vector<float> samples(tmp.rows());
  for (int32_t i = 0; i < tmp.rows(); ++i) samples[i] = tmp(i, 0);
  return samples;
}

// The mandarin_mfa-style options of golden group "mandarin_mfa_pitch"
// (see tests/data/feat_golden/README.md and generate.sh):
// compute-and-process-kaldi-pitch-feats --sample-frequency=16000
//   --frame-shift=10 --frame-length=25 --min-f0=50 --max-f0=500
//   --delta-pitch=0.005 --penalty-factor=0.1 --snip-edges=true
//   --add-normalized-log-pitch=true --add-delta-pitch=true
PitchExtractionOptions MandarinPitchOpts() {
  PitchExtractionOptions opts;
  opts.samp_freq = 16000;
  opts.frame_shift_ms = 10;
  opts.frame_length_ms = 25;
  opts.min_f0 = 50;
  opts.max_f0 = 500;
  opts.delta_pitch = 0.005;
  opts.penalty_factor = 0.1;
  opts.snip_edges = true;
  return opts;
}

}  // namespace

TEST(Pitch, MatchesGolden) {
  std::vector<float> samples = LoadSamples("zh0_1s.npy");
  MatrixD ref;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/mandarin_mfa_pitch/pitch.utt.npy", &ref));
  ASSERT_EQ(ref.cols(), 3);  // POV, normalized-log-pitch, delta-pitch

  MatrixF got =
      ComputeAndProcessKaldiPitch(MandarinPitchOpts(), ProcessPitchOptions(),
                                  samples.data(), samples.size());
  ASSERT_EQ(got.rows(), ref.rows());
  ASSERT_EQ(got.cols(), ref.cols());
  // Value-by-value agreement with kaldi (dither-free pipeline): the noise
  // added to the delta-pitch is drawn from a deterministic stream (see
  // feature-pitch.cc), so the whole matrix must match within the
  // dot-product float noise of the NCCF computation (kaldi's BLAS vs
  // Eigen accumulation order; see kGoldenDotProductTol).
  EXPECT_LE(MaxAbsDiff(got, ref), kGoldenDotProductTol);
}

TEST(Pitch, StreamingMatchesGolden) {
  std::vector<float> samples = LoadSamples("zh0_1s.npy");
  MatrixD ref;
  ASSERT_TRUE(
      LoadNpy(GoldenDir() + "/golden/mandarin_mfa_pitch/pitch.utt.npy", &ref));

  OnlinePitch pitch(MandarinPitchOpts(), ProcessPitchOptions());
  EXPECT_EQ(pitch.Dim(), 3);
  pitch.AcceptWaveform(16000, samples.data(), samples.size());
  pitch.InputFinished();
  ASSERT_EQ(pitch.NumFramesReady(), ref.rows());
  EXPECT_TRUE(pitch.IsLastFrame(pitch.NumFramesReady() - 1));

  MatrixF got(pitch.NumFramesReady(), pitch.Dim());
  for (int32_t t = 0; t < pitch.NumFramesReady(); ++t) {
    std::vector<float> row = pitch.GetFrame(t);
    ASSERT_EQ(row.size(), 3);
    for (int32_t d = 0; d < 3; ++d) got(t, d) = row[d];
  }
  EXPECT_LE(MaxAbsDiff(got, ref), kGoldenDotProductTol);
}

}  // namespace knf

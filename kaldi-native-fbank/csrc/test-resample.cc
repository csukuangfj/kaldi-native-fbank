/**
 * Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/resample.h"

namespace knf {
namespace {

constexpr double kPi = 3.14159265358979323846;

VectorF MakeSine(int32_t n, float freq_hz, float samp_rate_hz) {
  VectorF wave(n);
  for (int32_t i = 0; i != n; ++i) {
    wave[i] = std::sin(2.0 * kPi * freq_hz * i / samp_rate_hz);
  }
  return wave;
}

// The float-pointer overload must give exactly the same output as the
// VectorF one.
TEST(ResampleTest, PointerOverloadMatchesVectorApi) {
  int32_t samp_rate_in = 44100;
  int32_t samp_rate_out = 16000;

  LinearResample resampler1(samp_rate_in, samp_rate_out, 0.99 * 0.5 * 16000, 6);
  LinearResample resampler2(samp_rate_in, samp_rate_out, 0.99 * 0.5 * 16000, 6);

  VectorF wave = MakeSine(44100, 440.0f, samp_rate_in);

  VectorF out1;
  resampler1.Resample(wave, true, &out1);

  std::vector<float> out2;
  resampler2.Resample(wave.data(), wave.size(), true, &out2);

  ASSERT_EQ(out1.size(), out2.size());
  for (size_t i = 0; i != out1.size(); ++i) {
    EXPECT_EQ(out1[i], out2[i]);
  }
}

// Piecewise processing must equal one-shot processing.
TEST(ResampleTest, PiecewiseEqualsOneShot) {
  int32_t samp_rate_in = 44100;
  int32_t samp_rate_out = 16000;

  LinearResample resampler1(samp_rate_in, samp_rate_out, 0.99 * 0.5 * 16000, 6);
  LinearResample resampler2(samp_rate_in, samp_rate_out, 0.99 * 0.5 * 16000, 6);

  VectorF wave = MakeSine(44100, 440.0f, samp_rate_in);

  VectorF out1;
  resampler1.Resample(wave, true, &out1);

  std::vector<float> out2;
  int32_t chunk = 1000;
  for (int32_t offset = 0; offset < wave.size(); offset += chunk) {
    int32_t n = std::min<int32_t>(chunk, wave.size() - offset);
    bool flush = (offset + n == wave.size());
    std::vector<float> piece;
    resampler2.Resample(wave.data() + offset, n, flush, &piece);
    out2.insert(out2.end(), piece.begin(), piece.end());
  }

  ASSERT_EQ(out1.size(), out2.size());
  for (size_t i = 0; i != out1.size(); ++i) {
    // A window that straddles two pieces is summed with a different code
    // path than the one-shot DotProduct, so results may differ by rounding.
    EXPECT_NEAR(out1[i], out2[i], 1e-5);
  }
}

// An empty last piece with flush == true must be accepted.
TEST(ResampleTest, EmptyFlushIsAcceptable) {
  LinearResample resampler(44100, 16000, 0.99 * 0.5 * 16000, 6);

  VectorF wave = MakeSine(4410, 440.0f, 44100);

  std::vector<float> out;
  resampler.Resample(wave.data(), wave.size(), false, &out);

  float dummy = 0;
  std::vector<float> tail;
  resampler.Resample(&dummy, 0, true, &tail);

  out.insert(out.end(), tail.begin(), tail.end());
  EXPECT_FALSE(out.empty());
}

}  // namespace
}  // namespace knf

/**
 * Copyright (c)  2026  kyo-zzz
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

#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"

namespace knf {
namespace {

std::vector<float> ComputeOneFrame(const FbankOptions &opts) {
  FbankComputer computer(opts);
  std::vector<float> frame(opts.frame_opts.PaddedWindowSize(), 0.0f);
  // Use a sample in the middle of the window so the window function does not
  // zero it out at the edge.
  frame[frame.size() / 2] = 1.0f;

  std::vector<float> feature(computer.Dim());
  computer.Compute(0, 1.0f, &frame, feature.data());
  return feature;
}

TEST(FbankEpsTest, DefaultIsFloatEpsilon) {
  FbankOptions opts;
  EXPECT_EQ(opts.eps, std::numeric_limits<float>::epsilon());
}

TEST(FbankEpsTest, AddsEpsBeforeLog) {
  FbankOptions opts;
  opts.frame_opts.dither = 0;
  opts.mel_opts.num_bins = 10;

  opts.eps = 0.0f;  // log(mel_energy)
  auto no_guard = ComputeOneFrame(opts);

  opts.eps = 1.0f;  // log(mel_energy + eps)
  auto with_guard = ComputeOneFrame(opts);

  for (size_t i = 0; i != no_guard.size(); ++i) {
    ASSERT_TRUE(std::isfinite(no_guard[i])) << "bin " << i;
    EXPECT_NEAR(with_guard[i], std::log(std::exp(no_guard[i]) + 1.0f), 1e-5)
        << "bin " << i;
  }
}

TEST(FbankEpsTest, ZeroFrameProducesLogEps) {
  FbankOptions opts;
  opts.frame_opts.dither = 0;
  opts.mel_opts.num_bins = 10;
  opts.eps = 5.9604644775390625e-08f;  // 2^-24, NeMo's log-zero guard

  FbankComputer computer(opts);
  std::vector<float> frame(opts.frame_opts.PaddedWindowSize(), 0.0f);
  std::vector<float> feature(computer.Dim());
  computer.Compute(0, 1.0f, &frame, feature.data());

  for (float v : feature) EXPECT_FLOAT_EQ(v, std::log(opts.eps));
}

}  // namespace
}  // namespace knf

/**
 * Copyright (c)  2026  Julian Pscheid
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
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/online-nemo-fbank.h"

namespace knf {

TEST(OnlineNemoFbank, CenteredFeatures) {
  NemoFbankOptions config;
  config.buffered_capacity = 672;
  config.num_mel_bins = 128;
  config.low_freq = 0;
  config.high_freq = 8000;
  OnlineNemoFbank stream(config);
  std::vector<float> audio(1281);
  for (int32_t i = 0; i < audio.size(); ++i) {
    audio[i] = 0.1 * std::cos(0.031 * i) + 0.025 * std::sin(0.137 * i);
  }
  audio[0] += 0.3f;
  audio[640] -= 0.2f;
  stream.AcceptWaveform(16000, audio.data(), audio.size());
  stream.InputFinished();
  ASSERT_EQ(stream.NumFramesReady(), 8);
  EXPECT_EQ(stream.NumTrailingSamples(), 1);
  int32_t offset = -1;
  auto frames = stream.GetFrames(0, 8, true, &offset);
  EXPECT_EQ(offset, 0);
  // NumPy centered FFT, symmetric 400-point Hann and librosa Slaney mel.
  const int32_t times[] = {0, 1, 3, 7};
  const int32_t bins[] = {0, 17, 63, 127};
  const float expected[4][4] = {
      {-7.6648445f, -7.6372275f, -6.3784108f, -4.1939735f},
      {-7.3691812f, -9.5039988f, -11.0779839f, -8.8404980f},
      {-7.3689537f, -9.3721800f, -12.4265671f, -10.1413660f},
      {-7.2432251f, -9.3878937f, -16.6143456f, -16.6329727f}};
  for (int32_t t = 0; t < 4; ++t) {
    for (int32_t b = 0; b < 4; ++b) {
      EXPECT_NEAR(frames[times[t] * 128 + bins[b]], expected[t][b], 2e-4);
    }
  }
}

TEST(OnlineNemoFbank, PartialFinalBuffer) {
  NemoFbankOptions config;
  config.buffered_capacity = 8;
  OnlineNemoFbank stream(config);
  std::vector<float> audio(11 * 160 + 79);
  for (int32_t i = 0; i < audio.size(); ++i) {
    audio[i] = 0.1 * std::sin(0.037 * i);
  }
  stream.AcceptWaveform(16000, audio.data(), audio.size());
  int32_t offset = -1;
  stream.GetFrames(0, 8, false, &offset);
  EXPECT_EQ(offset, 0);
  stream.InputFinished();
  auto frames = stream.GetFrames(3, 8, true, &offset);
  EXPECT_EQ(offset, 79);
  OnlineNemoFbank reference(config);
  reference.AcceptWaveform(16000, audio.data() + audio.size() - 1280, 1280);
  reference.InputFinished();
  auto expected = reference.GetFrames(0, 8, true, &offset);
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(frames, expected);
}

TEST(OnlineNemoFbank, FullCapacityPartialHop) {
  NemoFbankOptions config;
  config.buffered_capacity = 8;
  for (int32_t remainder : {1, 79}) {
    OnlineNemoFbank stream(config);
    std::vector<float> audio(8 * 160 + remainder);
    for (int32_t i = 0; i < audio.size(); ++i) {
      audio[i] = 0.1 * std::sin(0.037 * i);
    }
    stream.AcceptWaveform(16000, audio.data(), audio.size());
    stream.InputFinished();
    int32_t offset = 0;
    auto frames = stream.GetFrames(0, 8, true, &offset);
    EXPECT_EQ(offset, remainder);
    OnlineNemoFbank reference(config);
    reference.AcceptWaveform(16000, audio.data() + remainder, 1280);
    reference.InputFinished();
    auto expected = reference.GetFrames(0, 8, true, &offset);
    EXPECT_EQ(offset, 0);
    EXPECT_EQ(frames, expected);
  }
}

TEST(OnlineNemoFbank, EmptyAndPartialInput) {
  OnlineNemoFbank fbank(NemoFbankOptions{});
  EXPECT_FALSE(fbank.IsLastFrame(-1));
  std::vector<float> audio(79, 0.1f);
  fbank.AcceptWaveform(16000, audio.data(), audio.size());
  fbank.InputFinished();
  EXPECT_EQ(fbank.NumFramesReady(), 0);
  EXPECT_EQ(fbank.NumTrailingSamples(), 79);
  EXPECT_FALSE(fbank.IsLastFrame(-1));
  EXPECT_FALSE(fbank.IsLastFrame(0));
  int32_t offset = -1;
  EXPECT_TRUE(fbank.GetFrames(0, 0, true, &offset).empty());
  EXPECT_EQ(offset, 0);

  OnlineNemoFbank empty(NemoFbankOptions{});
  empty.InputFinished();
  EXPECT_FALSE(empty.IsLastFrame(-1));
}

TEST(OnlineNemoFbank, PrefixDiscard) {
  NemoFbankOptions opts;
  opts.buffered_capacity = 8;
  OnlineNemoFbank fbank(opts);
  std::vector<float> audio(12 * 160, 0.1f);
  fbank.AcceptWaveform(16000, audio.data(), audio.size());
  EXPECT_EQ(fbank.GetFrames(4, 8).size(), 8 * fbank.Dim());
  fbank.InputFinished();
  EXPECT_TRUE(fbank.IsLastFrame(11));
  EXPECT_FALSE(fbank.IsLastFrame(10));
}

}  // namespace knf

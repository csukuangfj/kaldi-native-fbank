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

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"

namespace knf {
namespace {

// A flat power spectrum makes every filter with a positive weight
// contribute a positive energy, so a zero energy means the filter has no
// FFT bin inside its triangle (or only zero weights).
std::vector<float> ComputeEnergies(const MelBanks &mel_banks,
                                   int32_t num_fft_bins) {
  std::vector<float> power(num_fft_bins, 1.0f);
  std::vector<float> energies(mel_banks.NumBins(), -1.0f);
  mel_banks.Compute(power.data(), energies.data());
  return energies;
}

// The mel bins of MedASR (128 bins over [125, 7500] Hz with a 512-point
// FFT): the triangle of bin 0 ([125, 154.9] Hz) is narrower than the FFT
// bin spacing (31.25 Hz) and touches the 125 Hz FFT bin exactly at its
// left edge. The bin must get an all-zero filter instead of aborting,
// which matches librosa/torchaudio.
TEST(MelComputationsTest, EmptyTriangleBecomesZeroFilter) {
  MelBanksOptions mel_opts;
  mel_opts.num_bins = 128;
  mel_opts.low_freq = 125;
  mel_opts.high_freq = 7500;

  FrameExtractionOptions frame_opts;  // 25 ms window -> 512-point FFT

  MelBanks *mel_banks = nullptr;
  EXPECT_NO_THROW(mel_banks = new MelBanks(mel_opts, frame_opts, 1.0f));

  std::vector<float> energies = ComputeEnergies(*mel_banks, 256);
  EXPECT_EQ(energies[0], 0.0f);
  for (int32_t i = 1; i != mel_banks->NumBins(); ++i) {
    EXPECT_GT(energies[i], 0.0f);
  }

  delete mel_banks;
}

TEST(MelComputationsTest, EmptyTriangleBecomesZeroFilterLibrosa) {
  MelBanksOptions mel_opts;
  mel_opts.num_bins = 128;
  mel_opts.low_freq = 125;
  mel_opts.high_freq = 7500;
  mel_opts.is_librosa = true;
  mel_opts.use_slaney_mel_scale = false;  // HTK scale, same as the kaldi path

  FrameExtractionOptions frame_opts;  // 25 ms window -> 512-point FFT

  MelBanks *mel_banks = nullptr;
  EXPECT_NO_THROW(mel_banks = new MelBanks(mel_opts, frame_opts, 1.0f));

  std::vector<float> energies = ComputeEnergies(*mel_banks, 257);
  EXPECT_EQ(energies[0], 0.0f);
  for (int32_t i = 1; i != mel_banks->NumBins(); ++i) {
    EXPECT_GT(energies[i], 0.0f);
  }

  delete mel_banks;
}

// A usual configuration must keep all filters non-empty.
TEST(MelComputationsTest, NormalConfigHasNoEmptyBins) {
  MelBanksOptions mel_opts;
  mel_opts.num_bins = 80;
  mel_opts.low_freq = 20;
  mel_opts.high_freq = 7600;

  FrameExtractionOptions frame_opts;

  MelBanks mel_banks(mel_opts, frame_opts, 1.0f);

  std::vector<float> energies = ComputeEnergies(mel_banks, 256);
  for (int32_t i = 0; i != mel_banks.NumBins(); ++i) {
    EXPECT_GT(energies[i], 0.0f);
  }
}

// Even a grossly oversized num_bins must not throw; the affected bins
// simply get all-zero filters.
TEST(MelComputationsTest, TooManyBinsDoesNotThrow) {
  MelBanksOptions mel_opts;
  mel_opts.num_bins = 2000;
  mel_opts.low_freq = 125;
  mel_opts.high_freq = 7500;

  FrameExtractionOptions frame_opts;

  MelBanks *mel_banks = nullptr;
  EXPECT_NO_THROW(mel_banks = new MelBanks(mel_opts, frame_opts, 1.0f));

  std::vector<float> energies = ComputeEnergies(*mel_banks, 256);
  for (int32_t i = 0; i != mel_banks->NumBins(); ++i) {
    EXPECT_GE(energies[i], 0.0f);
  }

  delete mel_banks;
}

}  // namespace
}  // namespace knf

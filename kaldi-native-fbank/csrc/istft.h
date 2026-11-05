/**
 * Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
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

/**
 * @file istft.h
 * @brief Inverse short-time Fourier transform (ISTFT).
 *
 * This is not from kaldi. It is the inverse of Stft (see stft.h) and is
 * compatible with librosa's istft(): the frames are inverse-transformed,
 * windowed, overlap-added, and divided by the summed window envelope.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_ISTFT_H_
#define KALDI_NATIVE_FBANK_CSRC_ISTFT_H_

#include <memory>
#include <vector>

#include "kaldi-native-fbank/csrc/stft.h"

namespace knf {
struct StftConfig;
struct StftResult;

/// @brief Reconstruct a waveform from Stft results.
///
/// @par Used by:
/// Inverse pipelines that consume Stft::Compute() output (e.g. speech
/// enhancement / vocoder front ends).
///
/// @see Stft, StftConfig, StftResult
class IStft {
 public:
  /// Construct with the same config that was used for the forward Stft.
  explicit IStft(const StftConfig &config);

  /// Destructor.
  ~IStft();

  /// Reconstruct the waveform.
  ///
  /// @param stft_result  Output of Stft::Compute(). Its num_frames and the
  ///                     config determine the length of the result.
  /// @return  The reconstructed samples; if config.center is true, the
  ///          n_fft/2 padding samples at both ends are removed.
  std::vector<float> Compute(const StftResult &stft_result) const;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_ISTFT_H_

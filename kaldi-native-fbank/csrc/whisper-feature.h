/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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
 * @file whisper-feature.h
 * @brief Mel features for OpenAI Whisper models.
 *
 * This is not from kaldi. It computes the mel power spectrum used by Whisper
 * models. Note it does not apply the final log compression; the caller is
 * expected to do that. For a mel power spectrum \f$ x \f$:
 *
 * \f[ x \leftarrow \max(x, 10^{-10}) \f]
 * \f[ y = \log_{10} x \f]
 * \f[ y \leftarrow \max(y,\, \max(y) - 8) \f]
 * \f[ \mathrm{mel} = (y + 4) / 4 \f]
 *
 * Input samples should be normalized into the range [-1, 1].
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_
#define KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"

namespace knf {

/// @brief Options for computing Whisper mel features.
///
/// @see WhisperFeatureComputer
struct WhisperFeatureOptions {
  /// Construct with the given frame options and number of mel bins.
  explicit WhisperFeatureOptions(const FrameExtractionOptions &frame_opts = {},
                                 int32_t dim = 80)
      : frame_opts(frame_opts), dim(dim) {}

  /// Frame-extraction options. Note: most fields are ignored and reset
  /// inside WhisperFeatureComputer (16 kHz, 25 ms window, 10 ms shift,
  /// "hann" window, snip_edges=false, etc.).
  FrameExtractionOptions frame_opts;

  /// Number of mel bins: 80 for most Whisper models, 128 for large v3.
  int32_t dim = 80;

  /// Return the options as a string.
  std::string ToString() const;
};

/// @brief Class for computing Whisper mel features.
///
/// For each frame it computes the power spectrum and projects it onto
/// librosa-style mel filter banks, producing mel power energies (not log).
///
/// @par Used by:
/// OnlineWhisperFbank = OnlineGenericBaseFeature<WhisperFeatureComputer>
/// (see online-feature.h).
///
/// @see WhisperFeatureOptions, MelBanks, OnlineWhisperFbank
class WhisperFeatureComputer {
 public:
  /// Construct from the given options.
  ///
  /// note: opts.frame_opts is ignored and we reset it inside.
  explicit WhisperFeatureComputer(
      const WhisperFeatureOptions &opts = WhisperFeatureOptions());

  /// Return the number of mel bins.
  int32_t Dim() const { return opts_.dim; }

  /// Return the (reset) frame-extraction options.
  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  /// Compute one frame of features from one frame of windowed signal; see
  /// FbankComputer::Compute() for the argument meanings.
  void Compute(float /*signal_raw_log_energy*/, float /*vtln_warp*/,
               std::vector<float> *signal_frame, float *feature);

  /// Always false: raw log energy is not used by this feature type.
  bool NeedRawLogEnergy() const { return false; }

  /// Option type accepted by this computer.
  using Options = WhisperFeatureOptions;

 private:
  std::unique_ptr<MelBanks> mel_banks_;  ///< Librosa-style mel filter banks.
  WhisperFeatureOptions opts_;           ///< Options given to the constructor.
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_

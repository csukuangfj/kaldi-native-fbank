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
 * @file feature-raw-audio-samples.h
 * @brief Frame raw audio samples without any processing.
 *
 * This is not from kaldi. It wraps raw waveform samples into frames so they
 * can be fed through the same streaming interface as the other feature
 * types (see OnlineRawAudioSamples in online-feature.h).
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"

namespace knf {

/// @brief Options for framing raw audio samples.
///
/// The default constructor sets up frame_opts so that no processing is done
/// on the samples: rectangular window, no dithering, no preemphasis, no DC
/// offset removal, no zero-padding.
///
/// @see RawAudioSamplesComputer
struct RawAudioSamplesOptions {
  /// Options for frame extraction; see FrameExtractionOptions.
  FrameExtractionOptions frame_opts;

  /// Construct with frame_opts set up for unprocessed samples: rectangular
  /// window, no dithering, no preemphasis, no DC offset removal, no
  /// zero-padding.
  RawAudioSamplesOptions() {
    frame_opts.window_type = "rectangular";
    frame_opts.dither = 0;
    frame_opts.preemph_coeff = 0;
    frame_opts.remove_dc_offset = false;
    frame_opts.round_to_power_of_two = false;
    frame_opts.snip_edges = true;
  }

  /// Return the options as a string.
  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";
    os << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const RawAudioSamplesOptions &opts);

/// @brief "Computer" that frames raw audio samples without any processing.
///
/// It has the same interface as FbankComputer/MfccComputer so it can be used
/// with OnlineGenericBaseFeature (see online-feature.h).
///
/// @par Used by:
/// OnlineRawAudioSamples = OnlineGenericBaseFeature<RawAudioSamplesComputer>
/// (see online-feature.h).
///
/// @see RawAudioSamplesOptions, OnlineRawAudioSamples
class RawAudioSamplesComputer {
 public:
  /// Option type accepted by this computer.
  using Options = RawAudioSamplesOptions;

  /// Construct from the given options.
  explicit RawAudioSamplesComputer(const RawAudioSamplesOptions &opts)
      : opts_(opts) {}

  /// Return the frame size in samples.
  int32_t Dim() const {
    return opts_.frame_opts.frame_length_ms * opts_.frame_opts.samp_freq / 1000;
  }

  /// Always false: raw log energy is not used by this feature type.
  bool NeedRawLogEnergy() const { return false; }

  /// Return the frame-extraction options.
  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  /// Return the options this computer was constructed with.
  const RawAudioSamplesOptions &GetOptions() const { return opts_; }

  /// Copy one frame of raw samples to `feature`; the two energy/vtln
  /// arguments are unused.
  void Compute(float unused_signal_raw_log_energy, float unused_vtln_warp,
               std::vector<float> *signal_frame, float *feature);

 private:
  RawAudioSamplesOptions opts_;  ///< Options given to the constructor.
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_

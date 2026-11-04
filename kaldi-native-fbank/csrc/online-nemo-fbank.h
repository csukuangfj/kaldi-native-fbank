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

/**
 * @file online-nemo-fbank.h
 * @brief Buffered streaming NeMo-compatible filter-bank features.
 *
 * This is not from kaldi. It reproduces NeMo's buffered streaming feature
 * extraction; see OnlineNemoFbank for the details.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_ONLINE_NEMO_FBANK_H_
#define KALDI_NATIVE_FBANK_CSRC_ONLINE_NEMO_FBANK_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/stft.h"

namespace knf {

/// @brief Options for OnlineNemoFbank.
struct NemoFbankOptions {
  /// Waveform sample frequency.
  float samp_freq = 16000;

  /// Frame shift in milliseconds.
  float frame_shift_ms = 10;

  /// Frame length in milliseconds.
  float frame_length_ms = 25;

  /// Number of mel bins.
  int32_t num_mel_bins = 80;

  /// Lower frequency cutoff for mel bins.
  float low_freq = 0;

  /// Upper frequency cutoff for mel bins; non-positive values are offsets
  /// from Nyquist.
  float high_freq = 0;

  /// Preemphasis coefficient.
  float preemph_coeff = 0.97f;

  /// Full-buffer length in frames. Zero disables the final sample-length cap.
  int32_t buffered_capacity = 0;

  /// Return the options as a string.
  std::string ToString() const;
};

/// @brief Write the options to the stream, one option per line.
std::ostream &operator<<(std::ostream &os, const NemoFbankOptions &opts);

/// @brief NeMo-compatible buffered streaming filter-bank features.
///
/// NeMo buffered streaming re-featurizes each requested audio window. Apply
/// preemphasis over that window, then a centered, constant-zero-padded STFT,
/// a symmetric Hann window, Slaney mel filters, and
/// \f$ \log(\mathrm{mel} + 2^{-24}) \f$.
/// Input samples are floats in [-1, 1]. This class does not resample or
/// normalize. Like OnlineFbank, instances require external synchronization.
///
/// @see NemoFbankOptions, Stft, MelBanks, OnlineFbank
class OnlineNemoFbank {
 public:
  /// Construct from the given options.
  explicit OnlineNemoFbank(const NemoFbankOptions &opts);

  /// Return the number of mel bins.
  int32_t Dim() const { return opts_.num_mel_bins; }

  /// Return the frame shift in seconds.
  float FrameShiftInSeconds() const { return hop_ / opts_.samp_freq; }

  /// Return the number of whole frames that are ready.
  int32_t NumFramesReady() const { return num_samples_ / hop_; }

  /// Return the number of buffered samples that do not make up a whole hop.
  int32_t NumTrailingSamples() const { return num_samples_ % hop_; }

  /// Return true if `frame` is the last frame. Only returns true after
  /// InputFinished() has been called.
  bool IsLastFrame(int32_t frame) const;

  /// Add more waveform samples; the sampling rate must match
  /// opts.samp_freq.
  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);

  /// Tell the class that no more waveform will be added.
  void InputFinished() { input_finished_ = true; }

  /// Return n row-major frames starting at the absolute frame_index. Samples
  /// preceding the requested window are discarded; later calls cannot rewind.
  /// If is_final reaches the last ready frame after InputFinished(), include
  /// the partial hop. When n equals buffered_capacity, cap the window at n hops
  /// by shifting its start by the remainder. Report that shift (in samples)
  /// through start_sample_offset, which is otherwise zero. A zero-frame request
  /// is empty.
  std::vector<float> GetFrames(int32_t frame_index, int32_t n,
                               bool is_final = false,
                               int32_t *start_sample_offset = nullptr);

 private:
  NemoFbankOptions opts_;          ///< Options given to the constructor.
  int32_t hop_;                    ///< Hop size in samples.
  int64_t num_samples_ = 0;        ///< Total number of samples seen so far.
  int64_t sample_offset_ = 0;      ///< Samples discarded from samples_.
  bool input_finished_ = false;    ///< True after InputFinished().
  std::deque<float> samples_;      ///< Buffered samples not yet consumed.
  std::unique_ptr<Stft> stft_;     ///< STFT engine.
  std::unique_ptr<MelBanks> mel_;  ///< Slaney mel filter banks.
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_ONLINE_NEMO_FBANK_H_

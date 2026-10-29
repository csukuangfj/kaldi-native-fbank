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

struct NemoFbankOptions {
  float samp_freq = 16000;
  float frame_shift_ms = 10;
  float frame_length_ms = 25;
  int32_t num_mel_bins = 80;
  float low_freq = 0;
  float high_freq = 0;  // Non-positive values are offsets from Nyquist.
  float preemph_coeff = 0.97f;

  // Full-buffer length in frames. Zero disables the final sample-length cap.
  int32_t buffered_capacity = 0;

  std::string ToString() const;
};

std::ostream &operator<<(std::ostream &os, const NemoFbankOptions &opts);

// NeMo buffered streaming re-featurizes each requested audio window. Apply
// preemphasis over that window, then a centered, constant-zero-padded STFT,
// a symmetric Hann window, Slaney mel filters, and log(mel + 2^-24).
// Input samples are floats in [-1, 1]. This class does not resample or
// normalize. Like OnlineFbank, instances require external synchronization.
class OnlineNemoFbank {
 public:
  explicit OnlineNemoFbank(const NemoFbankOptions &opts);

  int32_t Dim() const { return opts_.num_mel_bins; }
  float FrameShiftInSeconds() const { return hop_ / opts_.samp_freq; }
  int32_t NumFramesReady() const { return num_samples_ / hop_; }
  int32_t NumTrailingSamples() const { return num_samples_ % hop_; }
  bool IsLastFrame(int32_t frame) const;

  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);
  void InputFinished() { input_finished_ = true; }

  // Return n row-major frames starting at the absolute frame_index. Samples
  // preceding the requested window are discarded; later calls cannot rewind.
  // If is_final reaches the last ready frame after InputFinished(), include the
  // partial hop. When n equals buffered_capacity, cap the window at n hops by
  // shifting its start by the remainder. Report that shift (in samples) through
  // start_sample_offset, which is otherwise zero. A zero-frame request is
  // empty.
  std::vector<float> GetFrames(int32_t frame_index, int32_t n,
                               bool is_final = false,
                               int32_t *start_sample_offset = nullptr);

 private:
  NemoFbankOptions opts_;
  int32_t hop_;
  int64_t num_samples_ = 0;
  int64_t sample_offset_ = 0;
  bool input_finished_ = false;
  std::deque<float> samples_;
  std::unique_ptr<Stft> stft_;
  std::unique_ptr<MelBanks> mel_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_ONLINE_NEMO_FBANK_H_

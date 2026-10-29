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

#include "kaldi-native-fbank/csrc/online-nemo-fbank.h"

#include <cmath>
#include <sstream>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/log.h"

namespace knf {

std::string NemoFbankOptions::ToString() const {
  std::ostringstream os;
  os << "samp_freq: " << samp_freq << "\n";
  os << "frame_shift_ms: " << frame_shift_ms << "\n";
  os << "frame_length_ms: " << frame_length_ms << "\n";
  os << "num_mel_bins: " << num_mel_bins << "\n";
  os << "low_freq: " << low_freq << "\n";
  os << "high_freq: " << high_freq << "\n";
  os << "preemph_coeff: " << preemph_coeff << "\n";
  os << "buffered_capacity: " << buffered_capacity << "\n";
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const NemoFbankOptions &opts) {
  return os << opts.ToString();
}

OnlineNemoFbank::OnlineNemoFbank(const NemoFbankOptions &opts) : opts_(opts) {
  FrameExtractionOptions frame_opts;
  frame_opts.samp_freq = opts.samp_freq;
  frame_opts.frame_shift_ms = opts.frame_shift_ms;
  frame_opts.frame_length_ms = opts.frame_length_ms;
  hop_ = frame_opts.WindowShift();
  int32_t window_size = frame_opts.WindowSize();
  int32_t n_fft = frame_opts.PaddedWindowSize();
  MelBanksOptions mel_opts;
  mel_opts.num_bins = Dim();
  mel_opts.low_freq = opts.low_freq;
  mel_opts.high_freq = opts.high_freq;
  mel_opts.is_librosa = true;
  mel_ = std::make_unique<MelBanks>(mel_opts, frame_opts, 1.0f);

  StftConfig stft_config;
  stft_config.n_fft = n_fft;
  stft_config.hop_length = hop_;
  stft_config.win_length = n_fft;
  stft_config.center = true;
  stft_config.pad_mode = "constant";
  stft_config.window.resize(n_fft);
  int32_t pad = (n_fft - window_size) / 2;
  constexpr double kPi = 3.14159265358979323846;
  for (int32_t i = 0; i < window_size; ++i) {
    stft_config.window[pad + i] =
        0.5 - 0.5 * std::cos(2 * kPi * i / (window_size - 1));
  }
  stft_ = std::make_unique<Stft>(stft_config);
}

void OnlineNemoFbank::AcceptWaveform(float sampling_rate, const float *waveform,
                                     int32_t n) {
  KNF_CHECK_EQ(sampling_rate, opts_.samp_freq);
  KNF_CHECK_GE(n, 0);
  KNF_CHECK(n == 0 || waveform != nullptr);
  if (n == 0) {
    return;
  }
  KNF_CHECK(!input_finished_);
  samples_.insert(samples_.end(), waveform, waveform + n);
  num_samples_ += n;
}

bool OnlineNemoFbank::IsLastFrame(int32_t frame) const {
  return frame >= 0 && input_finished_ && frame == NumFramesReady() - 1;
}

std::vector<float> OnlineNemoFbank::GetFrames(int32_t frame_index, int32_t n,
                                              bool is_final,
                                              int32_t *start_sample_offset) {
  if (start_sample_offset) {
    *start_sample_offset = 0;
  }
  if (n == 0) {
    return {};
  }
  int64_t begin = static_cast<int64_t>(frame_index) * hop_;
  int64_t end = begin + static_cast<int64_t>(n) * hop_;
  if (is_final && input_finished_ &&
      static_cast<int64_t>(frame_index) + n == NumFramesReady()) {
    // Keep the partial hop, shifting the start only for a full-capacity buffer.
    end = num_samples_;
    if (n == opts_.buffered_capacity) {
      begin = end - static_cast<int64_t>(n) * hop_;
    }
  }
  KNF_CHECK_GE(frame_index, 0);
  KNF_CHECK_GE(n, 0);
  KNF_CHECK_GE(begin, sample_offset_);
  KNF_CHECK_LE(end, num_samples_);
  if (start_sample_offset) {
    *start_sample_offset = begin - static_cast<int64_t>(frame_index) * hop_;
  }
  samples_.erase(samples_.begin(), samples_.begin() + (begin - sample_offset_));
  sample_offset_ = begin;
  std::vector<float> audio(samples_.begin(), samples_.begin() + (end - begin));
  KNF_CHECK(opts_.preemph_coeff >= 0.0 && opts_.preemph_coeff <= 1.0);
  for (int32_t i = static_cast<int32_t>(audio.size()) - 1; i > 0; --i) {
    audio[i] -= opts_.preemph_coeff * audio[i - 1];
  }
  auto spectrum = stft_->Compute(audio.data(), audio.size());
  int32_t fft_bins = spectrum.real.size() / spectrum.num_frames;
  std::vector<float> power(fft_bins);
  std::vector<float> features(static_cast<size_t>(n) * Dim());
  constexpr float kLogGuard = 0x1p-24f;
  for (int32_t t = 0; t < n; ++t) {
    for (int32_t k = 0; k < fft_bins; ++k) {
      int32_t i = t * fft_bins + k;
      power[k] = spectrum.real[i] * spectrum.real[i] +
                 spectrum.imag[i] * spectrum.imag[i];
    }
    float *frame = features.data() + static_cast<size_t>(t) * Dim();
    mel_->Compute(power.data(), frame);
    for (int32_t k = 0; k < Dim(); ++k) {
      frame[k] = std::log(frame[k] + kLogGuard);
    }
  }
  return features;
}

}  // namespace knf

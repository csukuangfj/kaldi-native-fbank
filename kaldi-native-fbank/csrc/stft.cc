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
#include "kaldi-native-fbank/csrc/stft.h"

#include <sstream>

#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {
std::string StftConfig::ToString() const {
  std::ostringstream os;
  os << "StftConfig(";
  os << "n_fft=" << n_fft << ",";
  os << "hop_length=" << hop_length << ",";
  os << "win_length=" << win_length << ",";
  os << "window_type=\"" << window_type << "\",";
  os << "center=" << (center ? "True" : "False") << ",";
  os << "pad_mode=\"" << pad_mode << "\")";
  return os.str();
}

class Stft::Impl {
 public:
  explicit Impl(const StftConfig &config) : config_(config) {}

  StftResult Compute(const float *data, int32_t n) const {
    int32_t n_fft = config_.n_fft;
    int32_t hop_length = config_.hop_length;
    int64_t num_frames = 1 + (n - n_fft) / hop_length;

    Rfft rfft(n_fft);

    StftResult ans;
    ans.num_frames = num_frames;
    ans.real.resize(num_frames * (n_fft / 2 + 1));
    ans.imag.resize(num_frames * (n_fft / 2 + 1));

    std::vector<float> tmp(config_.n_fft);

    for (int32_t i = 0; i < num_frames; ++i) {
      tmp = {data + i * hop_length, data + i * hop_length + n_fft};
      rfft.Compute(tmp.data());

      for (int32_t k = 0; k < n_fft / 2; ++k) {
        if (k == 0) {
          ans.real[i * (n_fft / 2 + 1)] = tmp[0];
          ans.real[i * (n_fft / 2 + 1) + n_fft / 2] = tmp[1];
        } else {
          ans.real[i * (n_fft / 2 + 1) + k] = tmp[2 * k];

          // we use -1 here so it matches the results of torch.stft
          ans.imag[i * (n_fft / 2 + 1) + k] = -1 * tmp[2 * k + 1];
        }
      }
    }
    return ans;
  }

 private:
  StftConfig config_;
};

Stft::Stft(const StftConfig &config) : impl_(std::make_unique<Impl>(config)) {}

Stft::~Stft() = default;

StftResult Stft::Compute(const float *data, int32_t n) const {
  return impl_->Compute(data, n);
}

}  // namespace knf

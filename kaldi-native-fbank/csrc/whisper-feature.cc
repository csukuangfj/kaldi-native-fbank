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

#include "kaldi-native-fbank/csrc/whisper-feature.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-functions.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"

namespace knf {

std::string WhisperFeatureOptions::ToString() const {
  std::ostringstream os;
  os << "WhisperFeatureOptions(";
  os << "frame_opts=" << frame_opts.ToString() << ", ";
  os << "dim=" << dim << ")";
  return os.str();
}

WhisperFeatureComputer::WhisperFeatureComputer(
    const WhisperFeatureOptions &opts /*= {}*/)
    : opts_(opts) {
  opts_.frame_opts.samp_freq = 16000;
  opts_.frame_opts.frame_shift_ms = 10;
  opts_.frame_opts.frame_length_ms = 25;
  opts_.frame_opts.dither = 0;
  opts_.frame_opts.preemph_coeff = 0;
  opts_.frame_opts.remove_dc_offset = false;
  opts_.frame_opts.window_type = "hann";
  opts_.frame_opts.round_to_power_of_two = false;
  opts_.frame_opts.snip_edges = false;

  MelBanksOptions mel_opts;
  mel_opts.num_bins = opts_.dim;
  mel_opts.low_freq = 0;
  mel_opts.is_librosa = true;

  mel_banks_ = std::make_unique<MelBanks>(mel_opts, opts_.frame_opts, 1.0f);
  rfft_ = std::make_unique<Rfft>(opts_.frame_opts.PaddedWindowSize());
}

void WhisperFeatureComputer::Compute(float /*signal_raw_log_energy*/,
                                     float /*vtln_warp*/,
                                     std::vector<float> *signal_frame,
                                     float *feature) {
  if (signal_frame->size() !=
      static_cast<size_t>(opts_.frame_opts.PaddedWindowSize())) {
    throw std::runtime_error("signal_frame has an unexpected size");
  }
  // we have already applied window function to signal_frame before
  // calling this method
  rfft_->Compute(signal_frame->data());  // signal_frame is modified in-place
  ComputePowerSpectrum(signal_frame);

  // feature is pre-allocated by the user
  mel_banks_->Compute(signal_frame->data(), feature);
}

}  // namespace knf

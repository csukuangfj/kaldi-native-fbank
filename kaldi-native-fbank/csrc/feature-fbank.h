/**
 * Copyright 2009-2012  Karel Vesely
 *                2016  Johns Hopkins University (author: Daniel Povey)
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * This file is copied/modified from kaldi/src/feat/feature-fbank.h
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
 * @file feature-fbank.h
 * @brief Kaldi-compatible filter-bank (fbank) features.
 *
 * This file is copied/modified from @kfile{src/feat/feature-fbank.h}. It
 * computes mel-filterbank features, equal to kaldi's @kbin{compute-fbank-feats}
 * for identical parameters.
 *
 * @code
 *   knf::FbankOptions opts;
 *   opts.frame_opts.samp_freq = 16000;
 *   opts.mel_opts.num_bins = 23;
 *
 *   knf::OnlineFbank fbank(opts);  // see online-feature.h
 * @endcode
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_

#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

/// @brief Options for computing filter-bank features.
///
/// This is kaldi's FbankOptions (@kfile{src/feat/feature-fbank.h}). It only
/// includes things that can be done in a "stateless" way, i.e. it does not
/// include energy max-normalization and it does not include delta computation.
///
/// @see FbankComputer, FrameExtractionOptions, MelBanksOptions
struct FbankOptions {
  /// Options for frame extraction; see FrameExtractionOptions.
  FrameExtractionOptions frame_opts;

  /// Options for the mel filter banks; see MelBanksOptions.
  MelBanksOptions mel_opts;

  /// If true, append an extra dimension with energy to the filter banks.
  bool use_energy = false;

  /// Floor on energy (absolute, not relative) in fbank computation. Only
  /// makes a difference if use_energy is true; only necessary if
  /// frame_opts.dither is 0.0. Suggested values: 0.1 or 1.0.
  float energy_floor = 0.0f;  // active iff use_energy==true

  /// If true, compute log energy before preemphasis and windowing; if false,
  /// compute it after preemphasis and windowing.
  bool raw_energy = true;  // active iff use_energy==true

  /// If true, put energy last (if using energy); if false, put energy first.
  ///
  /// Warning: not sufficient to get HTK compatible features (need to change
  /// other parameters).
  bool htk_compat = false;  // active iff use_energy==true

  /// If true (default), produce log-filterbank; else linear.
  bool use_log_fbank = true;

  /// If true (default), use Kaldi's hard floor before the log, i.e.
  /// \f$ \log(\max(e, \epsilon)) \f$. If false, use the additive guard
  /// \f$ \log(e + \epsilon) \f$, where \f$ e \f$ is the mel energy. NeMo
  /// models use the additive guard with eps = FLT_EPSILON / 2 (2^-24).
  bool use_log_floor = true;

  /// Value used by the log-zero guard selected by use_log_floor.
  /// It defaults to FLT_EPSILON; NeMo models use FLT_EPSILON / 2
  /// (2^-24, 5.9604644775390625e-08).
  float eps = std::numeric_limits<float>::epsilon();

  /// If true (default), use power in filterbank analysis; else magnitude.
  bool use_power = true;

  /// Construct with the default options: 23 mel bins.
  FbankOptions() { mel_opts.num_bins = 23; }

  /// Return the options as a string.
  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";
    os << "\n";

    os << "mel_opts: \n";
    os << mel_opts << "\n";

    os << "use_energy: " << use_energy << "\n";
    os << "energy_floor: " << energy_floor << "\n";
    os << "raw_energy: " << raw_energy << "\n";
    os << "htk_compat: " << htk_compat << "\n";
    os << "use_log_fbank: " << use_log_fbank << "\n";
    os << "use_log_floor: " << use_log_floor << "\n";
    os << "eps: " << eps << "\n";
    os << "use_power: " << use_power << "\n";
    return os.str();
  }
};

/// @brief Write the options to the stream, one option per line.
std::ostream &operator<<(std::ostream &os, const FbankOptions &opts);

/// @brief Class for computing mel-filterbank features.
///
/// It computes one frame of features at a time from one frame of signal; see
/// OnlineFbank in online-feature.h for the streaming interface that feeds it.
/// For more information, see @kfile{src/feat/feature-fbank.h}.
///
/// @par Used by:
/// OnlineFbank = OnlineGenericBaseFeature<FbankComputer>
/// (see online-feature.h) wraps this class for streaming use.
///
/// @see FbankOptions, MelBanks, OnlineFbank, ExtractWindow
class FbankComputer {
 public:
  /// Option type accepted by this computer.
  using Options = FbankOptions;

  /// Construct from the given options.
  explicit FbankComputer(const FbankOptions &opts);

  /// Destructor; frees the cached MelBanks.
  ~FbankComputer();

  /// Return the dimension of the output features: num_bins, plus 1 if
  /// use_energy is true.
  int32_t Dim() const {
    return opts_.mel_opts.num_bins + (opts_.use_energy ? 1 : 0);
  }

  /// If true, compute log_energy_pre_window but after dithering and dc
  /// removal.
  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

  /// Return the frame-extraction options.
  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  /// Return the options this computer was constructed with.
  const FbankOptions &GetOptions() const { return opts_; }

  /**
     Function that computes one frame of features from
     one frame of signal.

     @param [in] signal_raw_log_energy The log-energy of the frame of the signal
         prior to windowing and pre-emphasis, or
         log(numeric_limits<float>::min()), whichever is greater.  Must be
         ignored by this function if this class returns false from
         this->NeedsRawLogEnergy().
     @param [in] vtln_warp  The VTLN warping factor that the user wants
         to be applied when computing features for this utterance.  Will
         normally be 1.0, meaning no warping is to be done.  The value will
         be ignored for feature types that don't support VLTN, such as
         spectrogram features.
     @param [in] signal_frame  One frame of the signal,
       as extracted using the function ExtractWindow() using the options
       returned by this->GetFrameOptions().  The function will use the
       vector as a workspace, which is why it's a non-const pointer.
     @param [out] feature  Pointer to a vector of size this->Dim(), to which
         the computed feature will be written. It should be pre-allocated.
  */
  void Compute(float signal_raw_log_energy, float vtln_warp,
               std::vector<float> *signal_frame, float *feature);

 private:
  /// Return the mel banks for the given VTLN factor, creating them on first
  /// use.
  const MelBanks *GetMelBanks(float vtln_warp);

  FbankOptions opts_;       ///< Options given to the constructor.
  float log_energy_floor_;  ///< Floor for log energy, derived from opts_.
  std::map<float, MelBanks *> mel_banks_;  ///< Mel banks per VTLN coefficient.
  Rfft rfft_;  ///< FFT engine, of size opts_.PaddedWindowSize().
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_

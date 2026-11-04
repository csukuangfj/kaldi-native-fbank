/**
 * Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
 *           2014-2016  Johns Hopkins University (author: Daniel Povey)
 * Copyright 2024       Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * This file is copied/modified from kaldi/src/feat/feature-mfcc.h
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
 * @file feature-mfcc.h
 * @brief Kaldi-compatible MFCC features.
 *
 * This file is copied/modified from @kfile{src/feat/feature-mfcc.h}. It
 * computes Mel-frequency cepstral coefficients, equal to kaldi's
 * @kbin{compute-mfcc-feats} value-by-value for identical parameters.
 *
 * @code
 *   knf::MfccOptions opts;
 *   opts.frame_opts.samp_freq = 16000;
 *   opts.num_ceps = 13;
 *
 *   knf::OnlineMfcc mfcc(opts);  // see online-feature.h
 * @endcode
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

/// @brief Options for computing MFCC features.
///
/// This is kaldi's MfccOptions (@kfile{src/feat/feature-mfcc.h}).
///
/// @see MfccComputer, FrameExtractionOptions, MelBanksOptions
struct MfccOptions {
  /// Options for frame extraction; see FrameExtractionOptions.
  FrameExtractionOptions frame_opts;

  /// Options for the mel filter banks; see MelBanksOptions.
  MelBanksOptions mel_opts;

  /// Number of cepstra in MFCC computation (including C0), e.g. 13.
  int32_t num_ceps = 13;

  /// Use energy (not C0) in MFCC computation.
  bool use_energy = true;

  /// Floor on energy (absolute, not relative) in MFCC computation. Only
  /// makes a difference if use_energy is true; only necessary if
  /// frame_opts.dither is 0.0. Suggested values: 0.1 or 1.0.
  float energy_floor = 0.0;

  /// If true, compute energy before preemphasis and windowing.
  bool raw_energy = true;

  /// Constant that controls scaling of MFCCs (HTK liftering). If 0.0, no
  /// liftering is done.
  float cepstral_lifter = 22.0;

  /// If true, put energy or C0 last and use a factor of sqrt(2) on C0.
  ///
  /// Warning: not sufficient to get HTK compatible features (need to change
  /// other parameters).
  bool htk_compat = false;

  /// Construct with the default options: 23 mel bins, 13 cepstra.
  MfccOptions() { mel_opts.num_bins = 23; }

  /// Return the options as a string.
  std::string ToString() const {
    std::ostringstream os;
    os << "MfccOptions(";
    os << "frame_opts=" << frame_opts.ToString() << ", ";
    os << "mel_opts=" << mel_opts.ToString() << ", ";

    os << "num_ceps=" << num_ceps << ", ";
    os << "use_energy=" << (use_energy ? "True" : "False") << ", ";
    os << "energy_floor=" << energy_floor << ", ";
    os << "raw_energy=" << (raw_energy ? "True" : "False") << ", ";
    os << "cepstral_lifter=" << cepstral_lifter << ", ";
    os << "htk_compat=" << (htk_compat ? "True" : "False") << ")";

    return os.str();
  }
};

/// @brief Write the options to the stream, one option per line.
std::ostream &operator<<(std::ostream &os, const MfccOptions &opts);

/// @brief Class for computing MFCC features.
///
/// It computes one frame of features at a time from one frame of signal; see
/// OnlineMfcc in online-feature.h for the streaming interface that feeds it.
/// For more information, see @kfile{src/feat/feature-mfcc.h}.
///
/// @par Used by:
/// OnlineMfcc = OnlineGenericBaseFeature<MfccComputer>
/// (see online-feature.h) wraps this class for streaming use.
///
/// @see MfccOptions, MelBanks, ComputeLifterCoeffs, OnlineMfcc, ExtractWindow
class MfccComputer {
 public:
  /// Option type accepted by this computer.
  using Options = MfccOptions;

  /// Construct from the given options.
  explicit MfccComputer(const MfccOptions &opts);

  /// Destructor; frees the cached MelBanks.
  ~MfccComputer();

  /// Return the dimension of the output features: num_ceps.
  int32_t Dim() const { return opts_.num_ceps; }

  /// If true, compute log_energy_pre_window but after dithering and dc
  /// removal.
  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

  /// Return the frame-extraction options.
  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  /// Return the options this computer was constructed with.
  const MfccOptions &GetOptions() const { return opts_; }

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

  MfccOptions opts_;        ///< Options given to the constructor.
  float log_energy_floor_;  ///< Floor for log energy, derived from opts_.
  std::map<float, MelBanks *> mel_banks_;  ///< Mel banks per VTLN coefficient.
  Rfft rfft_;  ///< FFT engine, of size opts_.PaddedWindowSize().

  /// Temp buffer of size num_mel_bins = opts.mel_opts.num_bins.
  std::vector<float> mel_energies_;

  std::vector<float>
      lifter_coeffs_;  ///< Liftering coeffs, size opts_.num_ceps.

  /// DCT matrix of shape [num_ceps][num_mel_bins], flattened in row major;
  /// we left-multiply by it to get the cepstra.
  std::vector<float> dct_matrix_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_

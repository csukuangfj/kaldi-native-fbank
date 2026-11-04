/**
 * Copyright 2009-2011  Phonexia s.r.o.;  Microsoft Corporation
 *                2016  Johns Hopkins University (author: Daniel Povey)
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * This file is copied/modified from kaldi/src/feat/mel-computations.h
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
 * @file mel-computations.h
 * @brief Mel-filterbank construction and related computations.
 *
 * This file provides the options and the class for computing triangular
 * mel-filterbank bins, plus liftering coefficients used by MFCC. It is
 * copied/modified from @kfile{src/feat/mel-computations.h} and additionally
 * supports librosa-style mel filters.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_
#define KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"

namespace knf {
struct FrameExtractionOptions;

/// @brief Options for constructing triangular mel-filterbank bins.
///
/// The kaldi-compatible fields correspond to kaldi's MelBanksOptions
/// (@kfile{src/feat/mel-computations.h}); the is_librosa-related fields select
/// librosa-compatible filters instead.
///
/// @par Used by:
/// The `mel_opts` member of FbankOptions and MfccOptions, and the
/// constructors of MelBanks.
///
/// @see MelBanks, FbankOptions, MfccOptions
struct MelBanksOptions {
  /// Number of triangular mel-frequency bins, e.g. 25.
  int32_t num_bins = 25;

  /// Lower frequency cutoff, e.g. 20.
  float low_freq = 20;

  /// Upper frequency cutoff; 0 means no cutoff, negative values are added to
  /// the Nyquist frequency to get the cutoff.
  float high_freq = 0;

  /// VTLN lower cutoff of the warping function.
  float vtln_low = 100;

  /// VTLN upper cutoff of the warping function: if negative, added to the
  /// Nyquist frequency to get the cutoff.
  float vtln_high = -500;

  /// Print out debugging information for mel bin computation.
  bool debug_mel = false;

  /// htk_mode is a "hidden" config. It enables more exact compatibility with
  /// HTK, for testing purposes. Affects mel-energy flooring and reproduces a
  /// bug in HTK.
  bool htk_mode = false;

  /// If true, use librosa-style mel filters instead of kaldi-style ones.
  ///
  /// Note that if you set is_librosa, you probably need to set low_freq to 0.
  /// See https://librosa.org/doc/main/generated/librosa.filters.mel.html
  bool is_librosa = false;

  /// Used only when is_librosa is true. Possible values: "", "slaney". We
  /// don't support a numeric value here, but it can be added on demand.
  /// See https://librosa.org/doc/main/generated/librosa.filters.mel.html
  std::string norm = "slaney";

  /// Used only when is_librosa is true: use the Slaney mel scale.
  bool use_slaney_mel_scale = true;

  /// Used only when is_librosa is true: floor the bin edges to integers.
  bool floor_to_int_bin = false;

  /// Return the options as a string.
  std::string ToString() const {
    std::ostringstream os;
    os << "num_bins: " << num_bins << "\n";
    os << "low_freq: " << low_freq << "\n";
    os << "high_freq: " << high_freq << "\n";
    os << "vtln_low: " << vtln_low << "\n";
    os << "vtln_high: " << vtln_high << "\n";
    os << "debug_mel: " << debug_mel << "\n";
    os << "htk_mode: " << htk_mode << "\n";
    os << "is_librosa: " << is_librosa << "\n";
    os << "norm: " << norm << "\n";
    os << "use_slaney_mel_scale: " << use_slaney_mel_scale << "\n";
    os << "floor_to_int_bin: " << floor_to_int_bin << "\n";
    return os.str();
  }
};

/// @brief Write the options to the stream, one option per line.
std::ostream &operator<<(std::ostream &os, const MelBanksOptions &opts);

/// @brief Triangular mel-filterbank bins.
///
/// A MelBanks object holds the weights of a set of triangular mel-frequency
/// bins and can project FFT bin energies onto them. See also
/// https://en.wikipedia.org/wiki/Mel_scale
///
/// @par Used by:
/// FbankComputer, MfccComputer, WhisperFeatureComputer, and
/// OnlineNemoFbank.
///
/// @see MelBanksOptions, FbankComputer, MfccComputer, ComputeLifterCoeffs
class MelBanks {
 public:
  /// @brief HTK mel scale: convert mel frequency to Hz.
  ///
  /// \f[ \mathrm{Hz} = 700 \left( \exp(\mathrm{mel} / 1127) - 1 \right) \f]
  static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
  }

  /// @brief HTK mel scale: convert Hz to mel frequency.
  ///
  /// \f[ \mathrm{mel} = 1127 \log(1 + \mathrm{Hz} / 700) \f]
  static inline float MelScale(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
  }

  /// @brief Slaney mel scale: convert mel frequency to Hz.
  ///
  /// \f[ \mathrm{Hz} = \left\{ \begin{array}{ll}
  ///     (200/3)\,\mathrm{mel} & \mathrm{mel} \le 15 \\
  ///     1000 \exp((\mathrm{mel} - 15) \log(6.4) / 27) & \mathrm{otherwise}
  ///   \end{array} \right. \f]
  static inline float InverseMelScaleSlaney(float mel_freq) {
    if (mel_freq <= 15) {
      return 200.0f / 3 * mel_freq;
    }

    // return 1000 * expf((mel_freq - 15) * logf(6.4f) / 27);

    // Note: log(6.4)/27 = 0.06875177742094911

    return 1000 * expf((mel_freq - 15) * 0.06875177742094911f);
  }

  /// @brief Slaney mel scale: convert Hz to mel frequency.
  ///
  /// \f[ \mathrm{mel} = \left\{ \begin{array}{ll}
  ///     (3/200)\,\mathrm{Hz} & \mathrm{Hz} \le 1000 \\
  ///     15 + 27 \log(\mathrm{Hz} / 1000) / \log(6.4) & \mathrm{otherwise}
  ///   \end{array} \right. \f]
  static inline float MelScaleSlaney(float freq) {
    if (freq <= 1000) {
      return freq * 3 / 200.0f;
    }

    // return 15 + 27 * logf(freq / 1000) / logf(6.4f)
    //
    // Note: 27/log(6.4) = 14.545078505785561

    return 15 + 14.545078505785561f * logf(freq / 1000);
  }

  /**
     @brief Piecewise-linear VTLN warping function in the frequency domain.

     This computes a VTLN warping function that is not the same as HTK's one,
     but has similar inputs (this function has the advantage of never
     producing empty bins).

     This function computes a warp function \f$ F(\mathrm{freq}) \f$,
     defined between low_freq and high_freq inclusive, with the following
     properties:

     - \f$ F(\mathrm{low\_freq}) = \mathrm{low\_freq} \f$
     - \f$ F(\mathrm{high\_freq}) = \mathrm{high\_freq} \f$

     The function is continuous and piecewise linear with two inflection
     points. The lower inflection point (measured in terms of the unwarped
     frequency) is at frequency \f$ l \f$; the higher inflection point is at
     a frequency \f$ h \f$. If \f$ l \le f \le h \f$, then
     \f$ F(f) = f/\mathrm{vtln\_warp\_factor} \f$.

     If the higher inflection point (measured in terms of the unwarped
     frequency) is at \f$ h \f$, then
     \f$ \max(h, F(h)) = \mathrm{vtln\_high\_cutoff} \f$. Since
     \f$ F(h) = h/\mathrm{vtln\_warp\_factor} \f$, then
     \f$ \max(h, h/\mathrm{vtln\_warp\_factor}) =
     \mathrm{vtln\_high\_cutoff} \f$, so
     \f[ h = \frac{\mathrm{vtln\_high\_cutoff}}
                   {\max(1,\, 1/\mathrm{vtln\_warp\_factor})}
          = \mathrm{vtln\_high\_cutoff}
            \cdot \min(1,\, \mathrm{vtln\_warp\_factor}). \f]

     If the lower inflection point (measured in terms of the unwarped
     frequency) is at \f$ l \f$, then
     \f$ \min(l, F(l)) = \mathrm{vtln\_low\_cutoff} \f$. This implies that
     \f[ l = \frac{\mathrm{vtln\_low\_cutoff}}
                   {\min(1,\, 1/\mathrm{vtln\_warp\_factor})}
          = \mathrm{vtln\_low\_cutoff}
            \cdot \max(1,\, \mathrm{vtln\_warp\_factor}). \f]

     @param [in] vtln_low_cutoff   Lower discontinuity of the warp function;
                                   must be > low_freq.
     @param [in] vtln_high_cutoff  Upper discontinuity of the warp function;
                                   must be < high_freq.
     @param [in] low_freq          Lower frequency cutoff of the mel
                                   computation.
     @param [in] high_freq         Upper frequency cutoff of the mel
                                   computation.
     @param [in] vtln_warp_factor  VTLN warping factor (1.0 means no
                                   warping).
     @param [in] freq              The frequency to warp.
     @return  The warped frequency; freq itself if it lies outside
              [low_freq, high_freq].
  */
  static float VtlnWarpFreq(
      float vtln_low_cutoff,
      float vtln_high_cutoff,  // discontinuities in warp func
      float low_freq,
      float high_freq,  // upper+lower frequency cutoffs in
      // the mel computation
      float vtln_warp_factor, float freq);

  /**
     @brief Piecewise-linear VTLN warping function in the mel domain.

     Same as VtlnWarpFreq(), but the input and output are on the mel scale:
     it warps mel_freq by first converting it to Hz, applying VtlnWarpFreq(),
     and converting the result back to mel.
  */
  static float VtlnWarpMelFreq(float vtln_low_cutoff, float vtln_high_cutoff,
                               float low_freq, float high_freq,
                               float vtln_warp_factor, float mel_freq);

  /// Construct kaldi-compatible mel banks.
  ///
  /// The mel bin edges are placed uniformly on the mel scale between
  /// opts.low_freq and opts.high_freq; each bin is a triangle in the
  /// frequency domain, defined by a start fft bin and per-bin weights (see
  /// GetBins()/GetMatrix()). If opts.is_librosa is true, librosa-style
  /// filters are computed instead.
  ///
  /// @param [in] opts        Mel-bin configuration; see MelBanksOptions.
  /// @param [in] frame_opts  Frame-extraction configuration (provides
  ///                         samp_freq and the FFT size).
  /// @param [in] vtln_warp_factor  VTLN warping factor (1.0 means no
  ///                         warping); the bin edges are warped with
  ///                         VtlnWarpMelFreq() before the triangles are
  ///                         computed.
  // TODO(fangjun): Remove vtln_warp_factor
  MelBanks(const MelBanksOptions &opts,
           const FrameExtractionOptions &frame_opts, float vtln_warp_factor);

  /// Initialize with a 2-d weights matrix.
  ///
  /// @param [in] weights  Pointer to the start address of the matrix, in row
  ///                      major; element (i, j) is the weight of fft bin j
  ///                      for mel bin i.
  /// @param [in] num_rows It equls to number of mel bins
  /// @param [in] num_cols It equals to (number of fft bins)/2+1
  MelBanks(const float *weights, int32_t num_rows, int32_t num_cols);

  /**
     @brief Compute mel energies (note: NOT log energies).

     At input, "fft_energies" contains the FFT energies (not log). For each
     mel bin \f$ i \f$, the output is the weighted sum of the FFT bin
     energies that fall into that bin:

     \f[ \mathrm{mel\_energies\_out}[i] =
         \sum_{k} w[i][k] \cdot
         \mathrm{fft\_energies}[\mathrm{offset}[i] + k] \f]

     where \f$ \mathrm{offset}[i] \f$ is the first nonzero fft bin of mel
     bin \f$ i \f$ and \f$ w[i] \f$ are its triangle weights.

     @param [in] fft_energies  1-D array of size num_fft_bins/2+1, holding
                the power spectrum (not log).
     @param [out] mel_energies_out  1-D array of size num_mel_bins; the
                caller must pre-allocate it. On return it holds the mel
                energies (not log).
  */
  void Compute(const float *fft_energies, float *mel_energies_out) const;

  /// Return the number of mel bins.
  int32_t NumBins() const { return bins_.size(); }

  /// Return the mel filter weights as a flattened 2-d matrix of shape
  /// [num_mel_bins, num_fft_bins/2+1], in row major.
  std::vector<float> GetMatrix() const;

 private:
  // for kaldi-compatible
  void InitKaldiMelBanks(const MelBanksOptions &opts,
                         const FrameExtractionOptions &frame_opts,
                         float vtln_warp_factor);

  // for librosa-compatible
  // See https://librosa.org/doc/main/generated/librosa.filters.mel.html
  void InitLibrosaMelBanks(const MelBanksOptions &opts,
                           const FrameExtractionOptions &frame_opts,
                           float vtln_warp_factor);

 private:
  /// The "bins_" vector is a vector, one for each bin, of a pair:
  /// (the first nonzero fft-bin), (the vector of weights).
  std::vector<std::pair<int32_t, std::vector<float>>> bins_;

  // TODO(fangjun): Remove debug_ and htk_mode_
  bool debug_ = false;         ///< Copy of opts.debug_mel.
  bool htk_mode_ = false;      ///< Copy of opts.htk_mode.
  int32_t num_fft_bins_ = -1;  ///< Number of FFT bins, num_fft/2+1.
};

/// @brief Compute liftering coefficients (scaling on cepstral coeffs).
///
/// For \f$ i = 0, \ldots, n-1 \f$, where \f$ n = \f$ `coeffs->size()`:
///
/// \f[ \mathrm{coeffs}[i] = 1 + \frac{Q}{2} \sin\left(\frac{\pi i}{Q}\right)
/// \f]
///
/// Coeffs are numbered slightly differently from HTK: the zeroth index is C0,
/// which is not affected. Q must be > 0: with Q == 0 the result is undefined
/// (division by zero). "No liftering" is handled by the caller by not calling
/// this function at all (see MfccOptions::cepstral_lifter).
///
/// @param [in] Q      Liftering constant, e.g. 22.0 (kaldi default).
/// @param [out] coeffs  Pre-allocated output vector of size num_ceps.
///
/// @par Used by:
/// MfccComputer (via MfccOptions::cepstral_lifter).
///
/// @see MfccComputer, MfccOptions
void ComputeLifterCoeffs(float Q, std::vector<float> *coeffs);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_

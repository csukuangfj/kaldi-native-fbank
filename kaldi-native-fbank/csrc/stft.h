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
 * @file stft.h
 * @brief Short-time Fourier transform (STFT).
 *
 * This is not from kaldi. It is compatible with librosa's stft() and is used
 * by the NeMo filter-bank features (see online-nemo-fbank.h).
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_STFT_H_
#define KALDI_NATIVE_FBANK_CSRC_STFT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace knf {

/// @brief Configuration for Stft.
///
/// @see Stft, IStft
struct StftConfig {
  /// FFT size; should be a power of two.
  int32_t n_fft = 0;

  /// Number of samples between successive frames.
  int32_t hop_length = 0;

  /// Window size in samples; must be <= n_fft. The window is zero-padded to
  /// n_fft.
  int32_t win_length = 0;

  /// Window type; see GetWindow() in feature-window.h for the supported
  /// types and their formulas.
  std::string window_type;

  /// If true, the signal is padded by n_fft/2 on both sides (using pad_mode)
  /// and the frames are centered around time t.
  bool center = true;

  /// Padding mode used when center is true; currently only "reflect".
  std::string pad_mode = "reflect";

  /// If true, scale the FFT by 1/sqrt(n_fft) instead of 1.
  bool normalized = false;

  /// Explicit window coefficients; if specified, window_type is ignored.
  std::vector<float> window;

  /// Return the config as a string.
  std::string ToString() const;
};

/// @brief Result of Stft::Compute().
///
/// @see Stft, IStft
struct StftResult {
  /// Real parts of the spectra, [num_frames, n_fft/2+1], flattened in row
  /// major.
  std::vector<float> real;

  /// Imaginary parts of the spectra, [num_frames, n_fft/2+1], flattened in
  /// row major.
  std::vector<float> imag;

  /// Number of frames in real/imag.
  int32_t num_frames = 0;
};

/// @brief Compute the short-time Fourier transform of a waveform.
///
/// @par Used by:
/// OnlineNemoFbank computes its spectra with Stft; IStft inverts them.
///
/// @see StftConfig, StftResult, IStft, Rfft
class Stft {
 public:
  /// Construct with the given config.
  explicit Stft(const StftConfig &config);

  /// Destructor.
  ~Stft();

  /// Compute the STFT of a waveform.
  ///
  /// @param data Pointer to a 1-D array of size n.
  /// @param n    Number of samples in data.
  /// @return     The complex spectra of all frames; see StftResult.
  StftResult Compute(const float *data, int32_t n) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_STFT_H_

// kaldi-native-fbank/csrc/feature-window.h
//
// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
//           2014-2016  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
//
// This file is copied/modified from kaldi/src/feat/feature-window.h

/**
 * @file feature-window.h
 * @brief Waveform windowing and frame extraction.
 *
 * This file provides the options and the helper functions used to split a
 * waveform into overlapping frames and to apply dithering, DC offset removal,
 * preemphasis, and a windowing function to each frame before the FFT.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace knf {

/// @brief Round n up to the nearest power of two. Requires n > 0.
///
/// Copied from @kfile{src/base/kaldi-math.cc}.
inline int32_t RoundUpToNearestPowerOfTwo(int32_t n) {
  // copied from kaldi/src/base/kaldi-math.cc
  if (n <= 0) {
    throw std::invalid_argument("n must be > 0");
  }
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

/// @brief Options for extracting frames from a waveform.
///
/// This struct is the equivalent of kaldi's FrameExtractionOptions
/// (@kfile{src/feat/feature-window.h}).
///
/// @par Used by:
/// Every feature computer: it is the `frame_opts` member of FbankOptions,
/// MfccOptions, WhisperFeatureOptions, RawAudioSamplesOptions, and it is
/// also used by ExtractWindow()/ProcessWindow()/NumFrames().
///
/// @see FeatureWindowFunction, ExtractWindow, NumFrames, FbankOptions
struct FrameExtractionOptions {
  /// Waveform data sample frequency (must match the waveform file, if
  /// specified there).
  float samp_freq = 16000;

  /// Frame shift in milliseconds.
  float frame_shift_ms = 10.0f;

  /// Frame length in milliseconds.
  float frame_length_ms = 25.0f;

  /// Amount of dithering, 0.0 means no dither.
  ///
  /// Note: the default value 0.00003f here is equivalent to 1.0 in kaldi.
  /// If you turn this off, you should set the energy floor option, e.g. to
  /// 1.0 or 0.1.
  float dither = 0.00003f;

  /// Preemphasis coefficient.
  float preemph_coeff = 0.97f;

  /// Subtract mean of wave before FFT.
  bool remove_dc_offset = true;

  /// Type of the windowing function; may be "hamming", "rectangular",
  /// "povey", "hanning", "hann", "sine", or "blackman".
  ///
  /// "povey" is a window made to be similar to Hamming but to go to zero at
  /// the edges:
  /// \f$ w[n] = (0.5 - 0.5 \cos(2\pi n/(N-1)))^{0.85} \f$.
  std::string window_type = "povey";

  /// If true, round the window size to a power of two by zero-padding the
  /// input to the FFT.
  bool round_to_power_of_two = true;

  /// Constant coefficient for the generalized Blackman window.
  float blackman_coeff = 0.42f;

  /// If true, end effects are handled by outputting only frames that
  /// completely fit in the file, and the number of frames depends on the
  /// frame length. If false, the number of frames depends only on the frame
  /// shift, and we reflect the data at the ends.
  bool snip_edges = true;

  // bool allow_downsample = false;
  // bool allow_upsample = false;

  /// Return the window shift in samples.
  int32_t WindowShift() const {
    return static_cast<int32_t>(samp_freq * 0.001f * frame_shift_ms);
  }

  /// Return the window size in samples.
  int32_t WindowSize() const {
    return static_cast<int32_t>(samp_freq * 0.001f * frame_length_ms);
  }

  /// Return the window size after zero-padding to a power of two, if
  /// round_to_power_of_two is true; otherwise the same as WindowSize().
  int32_t PaddedWindowSize() const {
    return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize())
                                  : WindowSize());
  }

  /// Return the options as a string.
  std::string ToString() const {
    std::ostringstream os;
#define KNF_PRINT(x) os << #x << ": " << x << "\n"
    KNF_PRINT(samp_freq);
    KNF_PRINT(frame_shift_ms);
    KNF_PRINT(frame_length_ms);
    KNF_PRINT(dither);
    KNF_PRINT(preemph_coeff);
    KNF_PRINT(remove_dc_offset);
    KNF_PRINT(window_type);
    KNF_PRINT(round_to_power_of_two);
    KNF_PRINT(blackman_coeff);
    KNF_PRINT(snip_edges);
    // KNF_PRINT(allow_downsample);
    // KNF_PRINT(allow_upsample);
#undef KNF_PRINT
    return os.str();
  }
};

/// @brief Write the options to the stream, one option per line.
std::ostream &operator<<(std::ostream &os, const FrameExtractionOptions &opts);

/// @brief Stores the windowing coefficients and applies them to a frame.
///
/// The window type is specified either by a FrameExtractionOptions, by a
/// window-type name plus a window size, or directly by the coefficients.
///
/// @par Used by:
/// ExtractWindow(), ProcessWindow(), Stft/IStft, and
/// OnlineGenericBaseFeature (online-feature.h).
///
/// @see GetWindow, FrameExtractionOptions, ProcessWindow
class FeatureWindowFunction {
 public:
  /// Construct an empty window function. You must assign a real window to it
  /// before calling Apply().
  FeatureWindowFunction() = default;

  /// Construct the windowing function implied by `opts.window_type`, with a
  /// size of opts.WindowSize().
  explicit FeatureWindowFunction(const FrameExtractionOptions &opts);

  /// Construct the windowing function of the given type and size.
  ///
  /// See GetWindow() for the formula of each window type.
  ///
  /// @param window_type  One of "hamming", "rectangular", "povey", "hanning",
  ///                     "hann", "sine", "blackman".
  /// @param window_size  Number of samples in the window.
  /// @param blackman_coeff  Constant coefficient for generalized Blackman
  ///                        window; only used if window_type is "blackman".
  FeatureWindowFunction(const std::string &window_type, int32_t window_size,
                        float blackman_coeff = 0.42);

  /// Construct from explicit window coefficients.
  explicit FeatureWindowFunction(const std::vector<float> &window);

  /**
   * Apply the window to a frame, in place.
   *
   * @param wave Pointer to a 1-D array of shape [window_size].
   *             It is modified in-place: wave[i] = wave[i] * window_[i].
   */
  void Apply(float *wave) const;

  /// Return the window coefficients.
  const std::vector<float> &GetWindow() const { return window_; }

 private:
  std::vector<float> window_;  // of size opts.WindowSize()
};

/// @brief Return the index of the first sample of the frame indexed `frame`.
///
/// @see NumFrames, FrameExtractionOptions
///
/// If snip_edges=true, it just returns frame * opts.WindowShift(); if
/// snip_edges=false, the formula is a little more complicated and the result
/// may be negative.
int64_t FirstSampleOfFrame(int32_t frame, const FrameExtractionOptions &opts);

/**
   This function returns the number of frames that we can extract from a wave
   file with the given number of samples in it (assumed to have the same
   sampling rate as specified in 'opts').

      @param [in] num_samples  The number of samples in the wave file.
      @param [in] opts     The frame-extraction options class

      @param [in] flush   True if we are asserting that this number of samples
   is 'all there is', false if we expecting more data to possibly come in.  This
   only makes a difference to the answer
   if opts.snip_edges== false.  For offline feature extraction you always want
   flush == true.  In an online-decoding context, once you know (or decide) that
   no more data is coming in, you'd call it with flush == true at the end to
   flush out any remaining data.

      @return  The number of frames that can be extracted.

      @see FirstSampleOfFrame, ExtractWindow
*/
int32_t NumFrames(int64_t num_samples, const FrameExtractionOptions &opts,
                  bool flush = true);

/**
  ExtractWindow() extracts a windowed frame of waveform (possibly with a
  power-of-two, padded size, depending on the config), including all the
  processing done by ProcessWindow().

  @param [in] sample_offset  If 'wave' is not the entire waveform, but
                   part of it to the left has been discarded, then the
                   number of samples prior to 'wave' that we have
                   already discarded.  Set this to zero if you are
                   processing the entire waveform in one piece, or
                   if you get 'no matching function' compilation
                   errors when updating the code.
  @param [in] wave  The waveform
  @param [in] f     The frame index to be extracted, with
                    0 <= f < NumFrames(sample_offset + wave.Dim(), opts, true)
  @param [in] opts  The options class to be used
  @param [in] window_function  The windowing function, as derived from the
                    options class.
  @param [out] window  The windowed, possibly-padded waveform to be
                     extracted.  Will be resized as needed.
  @param [out] log_energy_pre_window  If non-NULL, the log-energy of
                   the signal prior to pre-emphasis and multiplying by
                   the windowing function will be written to here.

  @see ProcessWindow, NumFrames, FbankComputer::Compute, MfccComputer::Compute
*/
void ExtractWindow(int64_t sample_offset, const std::vector<float> &wave,
                   int32_t f, const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   std::vector<float> *window,
                   float *log_energy_pre_window = nullptr);

/**
  This function does all the windowing steps after actually
  extracting the windowed signal: depending on the
  configuration, it does dithering, dc offset removal,
  preemphasis, and multiplication by the windowing function.
   @param [in] opts  The options class to be used
   @param [in] window_function  The windowing function-- should have
                    been initialized using 'opts'.
   @param [in,out] window  A vector of size opts.WindowSize().  Note:
      it will typically be a sub-vector of a larger vector of size
      opts.PaddedWindowSize(), with the remaining samples zero,
      as the FFT code is more efficient if it operates on data with
      power-of-two size.
   @param [out]   log_energy_pre_window If non-NULL, then after dithering and
      DC offset removal, this function will write to this pointer the log of
      the total energy (i.e. sum-squared) of the frame.

   @see ExtractWindow, FeatureWindowFunction
 */
void ProcessWindow(const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function, float *window,
                   float *log_energy_pre_window = nullptr);

/// @brief Compute the inner product of two vectors of size n.
float InnerProduct(const float *a, const float *b, int32_t n);

/// @brief Return the window coefficients of the given window type.
///
/// Let \f$ N \f$ be `window_size`, \f$ n = 0, \ldots, N-1 \f$ the sample
/// index, and \f$ a = 2\pi/(N-1) \f$. The supported window types and their
/// formulas are:
///
/// * "hanning":     \f$ w[n] = 0.5 - 0.5 \cos(a n) \f$
/// * "hann":        \f$ w[n] = 0.5 - 0.5 \cos(a n) \f$ with
///   \f$ a = 2\pi/N \f$ (periodic variant; see
///   https://pytorch.org/docs/stable/generated/torch.hann_window.html)
/// * "sine":        \f$ w[n] = \sin(0.5\,a\,n) = \sin(\pi n/(N-1)) \f$
/// * "hamming":     \f$ w[n] = 0.54 - 0.46 \cos(a n) \f$
/// * "povey":       \f$ w[n] = (0.5 - 0.5 \cos(a n))^{0.85} \f$
///   (like hamming but goes to zero at the edges)
/// * "rectangular": \f$ w[n] = 1 \f$
/// * "blackman":    \f$ w[n] = c - 0.5 \cos(a n) + (0.5 - c) \cos(2 a n) \f$
///   where \f$ c = \f$ `blackman_coeff`.
///
/// If `window_type` is not one of the above, an error is logged with
/// KNF_LOGE() and std::invalid_argument is thrown.
///
/// @param window_type  One of "hamming", "rectangular", "povey", "hanning",
///                     "hann", "sine", "blackman".
/// @param window_size  Number of samples in the window.
/// @param blackman_coeff  Constant coefficient for generalized Blackman
///                        window; only used if window_type is "blackman".
/// @return  A vector of `window_size` window coefficients.
///
/// @see FeatureWindowFunction
std::vector<float> GetWindow(const std::string &window_type,
                             int32_t window_size, float blackman_coeff = 0.42);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_

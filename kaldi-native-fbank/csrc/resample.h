/// kaldi-native-fbank/kaldi-native-fbank/csrc/resample.h
//
// Copyright     2013  Pegah Ghahremani
//               2014  IMSL, PKU-HKUST (author: Wei Shi)
//               2014  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/feat/resample.h
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

/**
 * @file resample.h
 * @brief Classes for resampling signals (ported from kaldi's
 *        @kfile{src/feat/resample.h}).
 *
 * This header contains declarations of classes for resampling signals.  The
 * normal cases of resampling a signal are upsampling and downsampling
 * (increasing and decreasing the sample rate of a signal, respectively),
 * although the ArbitraryResample class allows a more generic case where
 * we want to get samples of a signal at uneven intervals (for instance,
 * log-spaced).
 *
 * The input signal is always evenly spaced, say sampled with frequency @f$ S
 * @f$, and we assume the original signal was band-limited to @f$ S/2 @f$ or
 * lower. The @f$ n @f$'th input sample @f$ x_n @f$ (with @f$ n = 0, 1, \ldots
 * @f$) is interpreted as the original signal's value at time @f$ n/S @f$.
 *
 * For resampling, it is convenient to view the input signal as a
 * continuous function @f$ x(t) @f$ of @f$ t @f$, where each sample @f$ x_n @f$
 * becomes a delta function with magnitude @f$ x_n/S @f$, at time @f$ n/S @f$.
 * If we band limit this to the Nyquist frequency @f$ S/2 @f$, we can show that
 * this is the same as the original signal that was sampled [assuming the
 * original signal was periodic and band limited].  In general we want to
 * bandlimit to lower than @f$ S/2 @f$, because we don't have a perfect filter
 * and also because if we want to resample at a lower frequency than @f$ S @f$,
 * we need to bandlimit to below half of that.
 *
 * Anyway, suppose we want to bandlimit to @f$ C @f$, with @f$ 0 < C < S/2 @f$.
 * The perfect rectangular filter with cutoff @f$ C @f$ is the sinc function,
 * @f[         f(t) = 2C \, \mathrm{sinc}(2Ct),                   @f]
 * where sinc is the normalized sinc function
 * @f$ \mathrm{sinc}(t) = \sin(\pi t) / (\pi t) @f$, with
 * @f$ \mathrm{sinc}(0) = 1 @f$.  This is not a practical filter, though,
 * because it has infinite support.  At the cost of less-than-perfect rolloff,
 * we can choose a suitable windowing function @f$ g(t) @f$, and use
 * @f$ f(t) g(t) @f$ as the filter.  For a windowing function we choose
 * raised-cosine (Hanning) window with support on @f$ [-w/2C, w/2C] @f$, where
 * @f$ w \ge 2 @f$ is an integer chosen by the user.  @f$ w = 1 @f$ means we
 * window the sinc function out to its first zero on the left and right,
 * @f$ w = 2 @f$ means the second zero, and so on; we normally choose @f$ w @f$
 * to be at least two.  We call this `num_zeros`, not @f$ w @f$, in the code.
 *
 * Convolving the signal @f$ x(t) @f$ with this windowed filter
 * @f$ h(t) = f(t)g(t) @f$ and evaluating the resulting signal @f$ s(t) @f$ at
 * an arbitrary time @f$ t @f$ is easy: we have
 *  @f[          s(t) = 1/S \sum_n x_n h(t - n/S).        @f]
 * This is true for arbitrary values of @f$ t @f$.  What the class
 * ArbitraryResample does is to allow you to evaluate the signal for specified
 * values of @f$ t @f$.
 *
 * @see LinearResample, ArbitraryResample, OnlinePitch
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_RESAMPLE_H_
#define KALDI_NATIVE_FBANK_CSRC_RESAMPLE_H_

#include <stdint.h>

#include <vector>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/**
 * @brief Resample a signal at arbitrary (e.g. log-spaced) time points.
 *
 * Class ArbitraryResample allows you to resample a signal (assumed zero outside
 * the sample region, not periodic) at arbitrary specified time values, which
 * don't have to be linearly spaced.  The low-pass filter cutoff
 * `filter_cutoff_hz` should be less than half the sample rate;
 * `num_zeros` should probably be at least two preferably more; higher numbers
 * give sharper filters but will be less efficient.
 *
 * @par Used by:
 * OnlinePitch resamples the NCCF from evenly spaced lags to the log-spaced
 * lags at which pitch is measured.
 *
 * @see LinearResample
 */
class ArbitraryResample {
 public:
  /**
   * @brief Construct the resampler.
   *
   * @param [in] num_samples_in   Number of samples in each input signal
   *                              (must be > 0).
   * @param [in] samp_rate_hz     Sample rate of the input signal in Hz
   *                              (must be > 0).
   * @param [in] filter_cutoff_hz Low-pass filter cutoff in Hz; must be > 0
   *                              and at most half of `samp_rate_hz`.
   * @param [in] sample_points_secs  The time points (in seconds) at which to
   *                              evaluate the resampled signal; one output
   *                              sample is produced per element.
   * @param [in] num_zeros        Filter sharpness: the raised-cosine window
   *                              spans `num_zeros` zeros of the sinc filter
   *                              (must be > 0; 2..10 recommended).
   */
  ArbitraryResample(int32_t num_samples_in, float samp_rate_hz,
                    float filter_cutoff_hz,
                    const std::vector<float> &sample_points_secs,
                    int32_t num_zeros);

  /// Return the number of samples in each input signal.
  int32_t NumSamplesIn() const { return num_samples_in_; }

  /// Return the number of samples in each resampled output signal
  /// (equals the `sample_points_secs` size given to the constructor).
  int32_t NumSamplesOut() const {
    return static_cast<int32_t>(weights_.size());
  }

  /**
   * @brief Resample a batch of signals (one per row).
   *
   * @param [in] input  Matrix of shape (num_rows, NumSamplesIn()); each row is
   *                    one input signal.
   * @param [out] output  Matrix of shape (num_rows, NumSamplesOut()); must
   *                    already be resized to that shape.  Each row is the
   *                    resampled version of the corresponding input row.
   */
  void Resample(const MatrixF &input, MatrixF *output) const;

  /**
   * @brief Resample a single signal.
   *
   * @param [in] input  Vector of size NumSamplesIn().
   * @param [out] output  Vector of size NumSamplesOut(); must already be
   *                    resized to that shape.
   */
  void Resample(const VectorF &input, VectorF *output) const;

 private:
  /// Compute, for each output sample, the index of the first input sample
  /// that has a nonzero weight (clamped to the valid input range).
  void SetIndexes(const std::vector<float> &sample_points);

  /// Compute the FIR filter weights for each output sample.
  void SetWeights(const std::vector<float> &sample_points);

  /// The windowed-sinc filter @f$ h(t) = f(t) g(t) @f$ evaluated at offset
  /// @f$ t @f$ seconds from the filter center.
  float FilterFunc(float t) const;

  int32_t num_samples_in_;  ///< Number of samples in each input signal.
  float samp_rate_in_;      ///< Input sample rate in Hz.
  float filter_cutoff_;     ///< Low-pass filter cutoff in Hz.
  int32_t num_zeros_;       ///< Filter width in sinc zeros.

  /// The first input-sample index that we sum over, for this output-sample
  /// index (clamped to [0, num_samples_in_-1]).
  std::vector<int32_t> first_index_;

  /// Filter weights on the input samples, for each output-sample index.
  std::vector<VectorF> weights_;
};

/**
 * @brief Efficient up/downsampling at linearly spaced output times.
 *
 * LinearResample is a special case of ArbitraryResample, where we want to
 * resample a signal at linearly spaced intervals (this means we want to
 * upsample or downsample the signal).  It is more efficient than
 * ArbitraryResample because we can construct it just once.
 *
 * We require that the input and output sampling rate be specified as
 * integers, as this is an easy way to specify that their ratio be rational.
 *
 * @par Used by:
 * OnlinePitch downsamples the waveform from `samp_freq` to `resample_freq`
 * before computing the NCCF.
 *
 * @see ArbitraryResample, ResampleWaveform
 */
class LinearResample {
 public:
  /**
   * @brief Construct the resampler.
   *
   * @param [in] samp_rate_in_hz   Input sample rate in Hz (integer, > 0).
   * @param [in] samp_rate_out_hz  Output sample rate in Hz (integer, > 0).
   * @param [in] filter_cutoff_hz  Low-pass filter cutoff in Hz; must be > 0
   *                               and at most half of both sample rates.
   * @param [in] num_zeros         Filter sharpness (sinc zeros under the
   *                               raised-cosine window); more == sharper but
   *                               less efficient.  Around 4 to 10 is typical.
   */
  LinearResample(int32_t samp_rate_in_hz, int32_t samp_rate_out_hz,
                 float filter_cutoff_hz, int32_t num_zeros);

  /**
   * @brief Resample (possibly a piece of) a signal.
   *
   * If you call it with `flush == true` and you have never called it with
   * `flush == false`, it just resamples the input signal (it resizes the
   * output to a suitable number of samples).
   *
   * You can also use this function to process a signal a piece at a time.
   * Suppose you break it into piece1, piece2, ... pieceN.  You can call
   * @code{.cc}
   * Resample(piece1, &output1, false);
   * Resample(piece2, &output2, false);
   * Resample(piece3, &output3, true);
   * @endcode
   * If you call it with `flush == false`, it won't output the last few samples
   * but will remember them, so that if you later give it a second piece of
   * the input signal it can process it correctly.
   *
   * If your most recent call to the object was with `flush == false`, it will
   * have internal state; you can remove this by calling Reset().  Calling
   * Reset() unnecessarily between signals will not do any harm.  Empty input
   * is acceptable.
   *
   * @param [in] input   The input signal (a piece of it).
   * @param [in] flush   True if this is the last piece of the signal.
   * @param [out] output The resampled output (resized internally).
   */
  void Resample(const VectorF &input, bool flush, VectorF *output);

  /**
   * @brief Reset the internal state prior to processing a new signal.
   *
   * It is only necessary if you have called Resample(x, y, false) for some
   * signal, leading to a remainder of the signal being saved, but then
   * abandon processing the signal before calling Resample(x, y, true) for
   * the last piece.
   */
  void Reset();

  /// Return the input sampling rate (for checks, for example).
  int32_t GetInputSamplingRate() const { return samp_rate_in_; }

  /// Return the output sampling rate (for checks, for example).
  int32_t GetOutputSamplingRate() const { return samp_rate_out_; }

 private:
  /**
   * @brief Return the number of output samples for a given input length.
   *
   * If `flush == true`, we return the largest n such that
   * (n/samp_rate_out_) is in the interval [ 0, input_num_samp/samp_rate_in_ ),
   * and note that the interval is half-open.  If `flush == false`, define
   * window_width as num_zeros / (2.0 * filter_cutoff_); we return the largest
   * n such that (n/samp_rate_out_) is in the interval
   * [ 0, input_num_samp/samp_rate_in_ - window_width ).
   */
  int64_t GetNumOutputSamples(int64_t input_num_samp, bool flush) const;

  /**
   * @brief Map an output-sample index to input-sample indexing info.
   *
   * Given an output-sample index, output to `*first_samp_in` the first
   * input-sample index that we have a weight on (may be negative), and to
   * `*samp_out_wrapped` the index into weights_ where we can get the
   * corresponding weights on the input.
   */
  void GetIndexes(int64_t samp_out, int64_t *first_samp_in,
                  int32_t *samp_out_wrapped) const;

  /// Save the trailing part of `input` that will be needed to process the
  /// next piece of the signal.
  void SetRemainder(const VectorF &input);

  /// Precompute the per-unit output-sample filter indexes and weights.
  void SetIndexesAndWeights();

  /// The windowed-sinc filter @f$ h(t) = f(t) g(t) @f$ evaluated at offset
  /// @f$ t @f$ seconds from the filter center.
  float FilterFunc(float t) const;

  // The following variables are provided by the user.
  int32_t samp_rate_in_;   ///< Input sample rate in Hz.
  int32_t samp_rate_out_;  ///< Output sample rate in Hz.
  float filter_cutoff_;    ///< Low-pass filter cutoff in Hz.
  int32_t num_zeros_;      ///< Filter width in sinc zeros.

  /// The number of input samples in the smallest repeating unit:
  /// num_samp_in_ = samp_rate_in_hz / Gcd(samp_rate_in_hz, samp_rate_out_hz)
  int32_t input_samples_in_unit_;

  /// The number of output samples in the smallest repeating unit:
  /// num_samp_out_ = samp_rate_out_hz / Gcd(samp_rate_in_hz, samp_rate_out_hz)
  int32_t output_samples_in_unit_;

  /// The first input-sample index that we sum over, for this output-sample
  /// index.  May be negative; any truncation at the beginning is handled
  /// separately.  This is just for the first few output samples, but we can
  /// extrapolate the correct input-sample index for arbitrary output samples.
  std::vector<int32_t> first_index_;

  /// Weights on the input samples, for this output-sample index.
  std::vector<VectorF> weights_;

  // The following variables keep track of where we are in a particular signal,
  // if it is being provided over multiple calls to Resample().

  /// The number of input samples we have already received for this signal
  /// (including anything in input_remainder_).
  int64_t input_sample_offset_ = 0;

  /// The number of samples we have already output for this signal.
  int64_t output_sample_offset_ = 0;

  /// A small trailing part of the previously seen input signal.
  VectorF input_remainder_;
};

/**
 * @brief Downsample or upsample a waveform (convenience wrapper).
 *
 * This is a convenience wrapper for the class 'LinearResample'.
 * The low-pass filter cutoff used in 'LinearResample' is 0.99 of the Nyquist,
 * where the Nyquist is half of the minimum of (orig_freq, new_freq).  The
 * resampling is done with a symmetric FIR filter with N_z (number of zeros)
 * as 6.
 *
 * We compared the downsampling results with those from the sox resampling
 * toolkit.  Sox's design is inspired by Laurent De Soras' paper,
 * https://ccrma.stanford.edu/~jos/resample/Implementation.html
 *
 * Note: we expect that while orig_freq and new_freq are floats, they are
 * actually required to have exact integer values (like 16000 or 8000) with
 * a ratio between them that can be expressed as a rational number with
 * reasonably small integer factors.
 *
 * @param [in] orig_freq  Original sample rate in Hz.
 * @param [in] wave       The input waveform.
 * @param [in] new_freq   Desired sample rate in Hz.
 * @param [out] new_wave  The resampled waveform (resized internally).
 */
void ResampleWaveform(float orig_freq, const VectorF &wave, float new_freq,
                      VectorF *new_wave);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_RESAMPLE_H_

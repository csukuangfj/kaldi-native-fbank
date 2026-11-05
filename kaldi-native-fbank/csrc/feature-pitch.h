/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-pitch.h
//
// Copyright     2013  Pegah Ghahremani
//               2014  IMSL, PKU-HKUST (author: Wei Shi)
//               2014  Yanqing Sun, Junjie Wang,
//                     Daniel Povey, Korbinian Riedhammer
//                     Xin Lei
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/feat/pitch-functions.h
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

/**
 * @file feature-pitch.h
 * @brief Pitch extraction (ported from kaldi's
 *        @kfile{src/feat/pitch-functions.h}).
 *
 * Implements the pitch extraction method described in *"A Pitch Extraction
 * Algorithm Tuned for Automatic Speech Recognition"*, Pegah Ghahremani,
 * Bagher BabaAli, Daniel Povey, Korbinian Riedhammer, Jan Trmal and Sanjeev
 * Khudanpur, ICASSP 2014
 * (https://www.danielpovey.com/files/icassp2014_pitch.pdf).
 *
 * The extraction happens in two stages, mirroring kaldi:
 *
 * 1. **Raw pitch** (`OnlinePitchFeature` in kaldi): the waveform is
 *    downsampled, the normalized cross-correlation function (NCCF) is
 *    evaluated at log-spaced lags between @f$ 1/\mathrm{max\_f0} @f$ and
 *    @f$ 1/\mathrm{min\_f0} @f$, and a Viterbi search over the lags picks a
 *    per-frame lag @f$ L_t @f$; the raw output is the 2-dimensional
 *    @f$ (\mathrm{NCCF}_t, 1/L_t) @f$.
 * 2. **Post-processing** (`OnlineProcessPitch` in kaldi): turns the raw
 *    output into features — by default the 3-dimensional
 *    (POV, normalized-log-pitch, delta-pitch) used by kaldi's
 *    @kbin{compute-and-process-kaldi-pitch-feats}.
 *
 * The streaming entry point is OnlinePitch; the batch entry point is
 * ComputeAndProcessKaldiPitch, which reproduces the output of kaldi's
 * `compute-and-process-kaldi-pitch-feats` bit-for-bit (with identical
 * options and input).
 *
 * @code
 *   knf::PitchExtractionOptions pitch_opts;
 *   pitch_opts.min_f0 = 50;
 *   pitch_opts.max_f0 = 500;
 *   knf::ProcessPitchOptions process_opts;   // 3-dim output by default
 *   knf::OnlinePitch pitch(pitch_opts, process_opts);
 *   pitch.AcceptWaveform(pitch_opts.samp_freq, samples, n);
 *   pitch.InputFinished();
 *   for (int32_t t = 0; t != pitch.NumFramesReady(); ++t) {
 *     std::vector<float> feat = pitch.GetFrame(t);  // size == pitch.Dim()
 *   }
 * @endcode
 *
 * @see PitchExtractionOptions, ProcessPitchOptions, OnlinePitch,
 *      ComputeAndProcessKaldiPitch, ResampleWaveform
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_PITCH_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_PITCH_H_

#include <stdint.h>

#include <memory>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/**
 * @brief Options for the raw pitch extraction stage (kaldi's
 *        `PitchExtractionOptions`).
 *
 * Mirrors the flags of kaldi's
 * [`compute-kaldi-pitch-feats`](https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-kaldi-pitch-feats.cc)
 * / `OnlinePitchFeature`; see @kfile{src/feat/pitch-functions.h}.
 *
 * @see OnlinePitch, ComputeAndProcessKaldiPitch
 */
struct PitchExtractionOptions {
  /// Waveform data sample frequency in Hz (must match the waveform).
  /// Kaldi @c --sample-frequency.
  float samp_freq = 16000;

  /// Frame shift in milliseconds.  Kaldi @c --frame-shift.
  float frame_shift_ms = 10.0;

  /// Frame length in milliseconds.  Kaldi @c --frame-length.
  float frame_length_ms = 25.0;

  /// Preemphasis coefficient (use is deprecated in kaldi; kept for parity).
  /// Kaldi @c --preemphasis-coefficient.
  float preemph_coeff = 0.0;

  /// Min F0 to search for (Hz).  Kaldi @c --min-f0.
  float min_f0 = 50;

  /// Max F0 to search for (Hz).  Kaldi @c --max-f0.
  float max_f0 = 400;

  /// Minimum f0, applied in a soft way; must not exceed `min_f0`.
  /// Kaldi @c --soft-min-f0.
  float soft_min_f0 = 10.0;

  /// Cost factor for F0 change (Viterbi inter-frame penalty).
  /// Kaldi @c --penalty-factor.
  float penalty_factor = 0.1;

  /// Cutoff frequency for the low-pass filter (Hz).  Kaldi @c --lowpass-cutoff.
  float lowpass_cutoff = 1000;

  /// Frequency that we down-sample the signal to.  Must be more than twice
  /// `lowpass_cutoff`.  Kaldi @c --resample-frequency.
  float resample_freq = 4000;

  /// The pitch tolerance in pruning lags: the relative change in pitch that
  /// the algorithm measures (the lags are spaced in a geometric progression
  /// with ratio @f$ 1 + \mathrm{delta\_pitch} @f$).
  /// Kaldi @c --delta-pitch.
  float delta_pitch = 0.005;

  /// Increasing this factor reduces NCCF for quiet frames, helping ensure
  /// pitch continuity in unvoiced regions.  Kaldi @c --nccf-ballast.
  float nccf_ballast = 7000;

  /// Integer that determines the filter width of the low-pass filter; more
  /// gives a sharper filter.  Kaldi @c --lowpass-filter-width.
  int32_t lowpass_filter_width = 1;

  /// Integer that determines the filter width when upsampling the NCCF.
  /// Kaldi @c --upsample-filter-width.
  int32_t upsample_filter_width = 5;

  /// The maximum number of frames of latency that we allow the pitch
  /// processing to introduce, for online operation.  If you set this to a
  /// large value, there would be no inaccuracy from the Viterbi traceback
  /// (but it might make you wait to see the pitch).  Normalization right
  /// context is more relevant for online latency; you can leave this at zero.
  /// Kaldi @c --max-frames-latency.
  int32_t max_frames_latency = 0;

  /// Only relevant for ComputeAndProcessKaldiPitch.  If nonzero, we provide
  /// the input as chunks of this size.  This affects the energy
  /// normalization which has a small effect on the resulting features,
  /// especially at the beginning of a file.  For best compatibility with
  /// online operation, you might want to set this to a small value, like one
  /// frame.  Kaldi @c --frames-per-chunk.
  int32_t frames_per_chunk = 0;

  /// Only relevant for ComputeAndProcessKaldiPitch, and only if
  /// `frames_per_chunk` is nonzero.  If true, we output the features as soon
  /// as they are available, which simulates the first-pass features you would
  /// get in online decoding.  If false, the features you get will be the same
  /// as those available at the end of the utterance, after InputFinished()
  /// has been called (e.g. during lattice rescoring).
  /// Kaldi @c --simulate-first-pass-online.
  bool simulate_first_pass_online = false;

  /// Only relevant for online operation or when emulating online operation
  /// (e.g. when setting `frames_per_chunk`).  This is the frame-index on
  /// which we recompute the NCCF (e.g. frame-index 500 = after 5 seconds);
  /// if the segment ends before this we do it when the segment ends.  We do
  /// this by re-computing the signal average energy, which affects the NCCF
  /// via the "ballast term", scaling the resampled NCCF by a factor derived
  /// from the average change in the "ballast term", and re-doing the
  /// backtrace computation.  Making this infinity would be the most exact,
  /// but would introduce unwanted latency at the end of long utterances, for
  /// little benefit.  Kaldi @c --recompute-frame.
  int32_t recompute_frame = 500;

  /// This is a "hidden config" used only for testing the online pitch
  /// extraction.  If true, we compute the signal root-mean-squared for the
  /// ballast term only up to the current frame, rather than the end of the
  /// current chunk of signal.  This makes the output insensitive to the
  /// chunking, which is useful for testing purposes.
  /// Kaldi @c --nccf-ballast-online.
  bool nccf_ballast_online = false;

  /// If false, the incomplete frames near the ending edge won't be snipped,
  /// so that the number of frames is the file size divided by the
  /// frame-shift.  This makes different types of features give the same
  /// number of frames.  Kaldi @c --snip-edges.  (Note: unlike the other
  /// feature types, kaldi's pitch extractor defaults this to true.)
  bool snip_edges = true;

  /// @return  The NCCF window size in samples after resampling.  This is the
  ///          "basic window size", not the full window size after extending
  ///          by max-lag.  Because of floating point representation, it is
  ///          more reliable to divide by 1000 instead of multiplying by
  ///          0.001, but it is a bit slower.
  int32_t NccfWindowSize() const {
    return static_cast<int32_t>(resample_freq * frame_length_ms / 1000.0);
  }

  /// @return  The NCCF window shift in samples after resampling.
  int32_t NccfWindowShift() const {
    return static_cast<int32_t>(resample_freq * frame_shift_ms / 1000.0);
  }
};

/**
 * @brief Options for the pitch post-processing stage (kaldi's
 *        `ProcessPitchOptions`).
 *
 * Determines which features are produced from the raw (NCCF, pitch) pair.
 * With the defaults, each frame is the 3-dimensional
 * (POV-feature, normalized-log-pitch, delta-log-pitch) used by kaldi's
 * [`compute-and-process-kaldi-pitch-feats`](https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc);
 * see @kfile{src/feat/pitch-functions.h}.
 *
 * @see OnlinePitch, ComputeAndProcessKaldiPitch
 */
struct ProcessPitchOptions {
  /// The final normalized-log-pitch feature is scaled with this value.
  /// Kaldi @c --pitch-scale.
  float pitch_scale = 2.0;

  /// The final POV feature is scaled with this value.  Kaldi @c --pov-scale.
  float pov_scale = 2.0;

  /// An offset that can be added to the final POV feature (useful for
  /// online decoding, where we don't do CMN to the pitch-derived features).
  /// Kaldi @c --pov-offset.
  float pov_offset = 0.0;

  /// Term to scale the final delta log-pitch feature.
  /// Kaldi @c --delta-pitch-scale.
  float delta_pitch_scale = 10.0;

  /// Standard deviation for the noise we add to the delta log-pitch (before
  /// scaling); should be about the same as the `delta_pitch` option to pitch
  /// creation.  The purpose is to get rid of peaks in the delta-pitch caused
  /// by discretization of pitch values.  (The noise is drawn from a
  /// deterministic stream — see ComputeAndProcessKaldiPitch — so the
  /// features are reproducible.)
  /// Kaldi @c --delta-pitch-noise-stddev.
  float delta_pitch_noise_stddev = 0.005;

  /// Left-context (in frames) for moving-window normalization of the log
  /// pitch.  Kaldi @c --normalization-left-context.
  int32_t normalization_left_context = 75;

  /// Right-context (in frames) for moving-window normalization of the log
  /// pitch; should be reduced in online decoding to reduce latency.
  /// Kaldi @c --normalization-right-context.
  int32_t normalization_right_context = 75;

  /// Number of frames on each side of the central frame to use for the delta
  /// window.  Kaldi @c --delta-window.
  int32_t delta_window = 2;

  /// Number of frames by which the pitch information is delayed.
  /// Kaldi @c --delay.
  int32_t delay = 0;

  /// If true, the warped NCCF is added to the output features (1 dim).
  /// Kaldi @c --add-pov-feature.
  bool add_pov_feature = true;

  /// If true, the log-pitch with POV-weighted mean subtraction over a
  /// 1.5 second window is added to the output features (1 dim).
  /// Kaldi @c --add-normalized-log-pitch.
  bool add_normalized_log_pitch = true;

  /// If true, the time derivative of log-pitch is added to the output
  /// features (1 dim).  Kaldi @c --add-delta-pitch.
  bool add_delta_pitch = true;

  /// If true, log(pitch) is added to the output features (1 dim).
  /// Kaldi @c --add-raw-log-pitch.
  bool add_raw_log_pitch = false;

  /// @return  The output feature dimension implied by the `add_*` flags.
  int32_t Dim() const {
    return (add_pov_feature ? 1 : 0) + (add_normalized_log_pitch ? 1 : 0) +
           (add_delta_pitch ? 1 : 0) + (add_raw_log_pitch ? 1 : 0);
  }
};

/**
 * @brief Streaming pitch extraction with post-processing (kaldi's
 *        `OnlinePitchFeature` + `OnlineProcessPitch`).
 *
 * This is the streaming interface of kaldi's
 * [`compute-and-process-kaldi-pitch-feats`](https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc):
 * it consumes waveform chunks and produces the processed pitch features
 * (by default 3-dimensional: POV, normalized-log-pitch, delta-pitch).
 *
 * Usage mirrors OnlineMfcc: feed the waveform with AcceptWaveform(), call
 * InputFinished() when no more data is coming, then read frames with
 * GetFrame().  Frames may become available before InputFinished() (with
 * latency: see `normalization_right_context` and `max_frames_latency`), and
 * InputFinished() flushes the remainder.
 *
 * Features for a given frame may be revised as more data arrives (the
 * Viterbi traceback and the normalization window settle over time), except
 * for the delta-pitch noise which is drawn once per frame.  After
 * InputFinished(), the features are final and stable.
 *
 * @par Used by:
 * Batch extraction via ComputeAndProcessKaldiPitch; MFA-style feature
 * pipelines that append pitch to MFCC (`paste-feats`).
 *
 * @see PitchExtractionOptions, ProcessPitchOptions,
 *      ComputeAndProcessKaldiPitch, OnlineMfcc
 */
class OnlinePitch {
 public:
  /**
   * @brief Construct with the given extraction and post-processing options.
   *
   * @param [in] opts   Raw pitch extraction options.
   * @param [in] popts  Post-processing options; at least one of the
   *                    `add_*` flags must be true.
   */
  explicit OnlinePitch(const PitchExtractionOptions &opts,
                       const ProcessPitchOptions &popts = {});

  ~OnlinePitch();

  /// @return  The output feature dimension (= `popts.Dim()`, 3 by default).
  int32_t Dim() const;

  /// @return  The frame shift in seconds.
  float FrameShiftInSeconds() const;

  /// @return  The number of frames that are already computed and readable
  ///          via GetFrame().
  int32_t NumFramesReady() const;

  /// @return  True if `frame` is the last frame (only true after
  ///          InputFinished() has been called and this is the final frame).
  bool IsLastFrame(int32_t frame) const;

  /**
   * @brief Return the `frame`-th feature vector.
   *
   * The frame is computed on demand and cached internally; calling this
   * repeatedly for the same frame returns the same values (as long as
   * `frame` < NumFramesReady()).
   *
   * @param [in] frame  Frame index, must be < NumFramesReady().
   * @return  Feature vector of size Dim().
   */
  std::vector<float> GetFrame(int32_t frame) const;

  /**
   * @brief Add waveform samples to the extractor.
   *
   * @param [in] sampling_rate  Sample rate of the waveform in Hz; must match
   *                            `opts.samp_freq`.
   * @param [in] waveform       Pointer to a 1-D array of n samples.
   * @param [in] n              Number of samples in `waveform`.
   */
  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);

  /// Tell the extractor no more waveform will be added; this flushes out the
  /// last frames and fixes the features of all frames.
  void InputFinished();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Batch pitch extraction (kaldi's `ComputeAndProcessKaldiPitch`).
 *
 * Combines the raw pitch extraction and the post-processing in one call;
 * the output equals what kaldi's
 * [`compute-and-process-kaldi-pitch-feats`](https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc)
 * produces for the same input and options.  It is sensitive to the
 * variables in `pitch_opts` that relate to online processing, i.e.
 * `max_frames_latency`, `frames_per_chunk`, `simulate_first_pass_online`,
 * `recompute_frame`.
 *
 * @param [in] pitch_opts    Raw pitch extraction options.
 * @param [in] process_opts  Post-processing options.
 * @param [in] wave          Pointer to a 1-D array of n waveform samples
 *                           (as returned by a WAV reader; kaldi semantics,
 *                           i.e. raw int16 magnitudes, not divided by 32768).
 * @param [in] n             Number of samples in `wave`.
 * @return  Feature matrix of shape (num_frames, process_opts.Dim()).
 */
MatrixF ComputeAndProcessKaldiPitch(const PitchExtractionOptions &pitch_opts,
                                    const ProcessPitchOptions &process_opts,
                                    const float *wave, int32_t n);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_PITCH_H_

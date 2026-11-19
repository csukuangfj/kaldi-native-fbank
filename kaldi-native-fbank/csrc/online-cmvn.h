// kaldi-native-fbank/kaldi-native-fbank/csrc/online-cmvn.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information
#ifndef KALDI_NATIVE_FBANK_CSRC_ONLINE_CMVN_H_
#define KALDI_NATIVE_FBANK_CSRC_ONLINE_CMVN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/**
 * @brief Options for OnlineCmvn, mirroring kaldi's OnlineCmvnOptions
 *   (kaldi/src/feat/online-feature.h).  The defaults match kaldi's.
 *
 * @details The options control two things: the sliding window of the running
 *   statistics, and how the sparse stats at the start of an utterance are
 *   backed off ("smoothed") with speaker and global stats.  See
 *   OnlineCmvn::SmoothOnlineCmvnStats for the smoothing formulas.
 */
struct OnlineCmvnOptions {
  /// Number of frames of sliding context for cepstral mean normalization
  /// (kaldi: --cmn-window).  In the steady state the normalization of frame
  /// t uses the stats of frames t-cmn_window+1 .. t.
  int32_t cmn_window = 600;
  /// Number of frames of previous utterance(s) from this speaker to back off
  /// to at the beginning of an utterance (kaldi: --speaker-frames).  Should
  /// be <= cmn_window; values above it are clamped at use time.
  int32_t speaker_frames = 600;
  /// Number of frames of global-average stats to back off to further, when
  /// even the speaker stats cannot fill the window (kaldi: --global-frames).
  int32_t global_frames = 200;
  /// If true (default), subtract the running mean.  Must be true when
  /// normalize_variance is true (kaldi: --norm-means).
  bool normalize_mean = true;
  /// If true, also divide by the running standard deviation (kaldi:
  /// --norm-vars).  Off by default; enable only if the features were trained
  /// with variance normalization.
  bool normalize_variance = false;
  /// How often kaldi snapshots stats internally (every `modulus` frames).
  /// Kept for configuration compatibility; this push-style port updates the
  /// stats incrementally, so the value does not affect results.
  int32_t modulus = 20;
  /// Size of kaldi's ring buffer for cached stats snapshots.  Kept for
  /// configuration compatibility (see `modulus`); must be >= modulus.
  int32_t ring_buffer_size = 20;
  /// Colon-separated list of dimensions to skip normalization of, e.g.
  /// "13:14:15" (kaldi: --skip-dims).  For skipped dims the stats are faked
  /// to mean 0 / variance 1, so the frame values pass through unchanged.
  std::string skip_dims;

  /// Return the options as a string.
  std::string ToString() const;
};

/**
 * @brief State of CMVN adaptation between utterances (not the
 *   within-utterance computation state).
 *
 * @details Stats matrices follow kaldi's layout: 2 x (dim+1) as
 *   `[ sum-stats count ; sum-squared-stats 0 ]`.  Pass the state of one
 *   utterance to a new OnlineCmvn to continue with the next utterance of the
 *   same speaker; the speaker stats then back off the normalization at the
 *   start of that utterance (see OnlineCmvn for the smoothing rules).
 */
struct OnlineCmvnState {
  /// Total CMVN stats for this speaker up till now, 2 x (dim+1); may be
  /// empty at the very first utterance.
  MatrixD speaker_cmvn_stats;
  /// Global CMVN stats, 2 x (dim+1), e.g. loaded from kaldi's
  /// `global_cmvn.stats`.  Required by OnlineCmvn.
  MatrixD global_cmvn_stats;
  /// If non-empty, 2 x (dim+1): the "frozen" normalization in use since
  /// OnlineCmvn::Freeze() was called (kaldi uses this to estimate fMLLR on
  /// top of a fixed CMVN).  Carries over to the next utterance.
  MatrixD frozen_state;

  OnlineCmvnState() = default;
  /// Construct with global stats (the required part of the state).
  explicit OnlineCmvnState(const MatrixD &global_stats)
      : global_cmvn_stats(global_stats) {}
};

/**
 * @brief Online (running) cepstral mean and optional variance normalization.
 *
 * @details A port of kaldi's OnlineCmvn (kaldi/src/feat/online-feature.h),
 *   push-style: feed raw feature frames one by one and get the normalized
 *   frames back (kaldi's version is a pull-style filter in an
 *   OnlineFeatureInterface chain).
 *
 *   Unlike the offline CMVN of feature-cmvn.h (which uses the whole
 *   utterance), this class looks only at the past.  In the middle of a long
 *   utterance it accumulates stats from the previous `cmn_window` frames
 *   (default 600, i.e. 6 seconds) and normalizes the current frame with
 *   them.  At the beginning of an utterance the window stats are augmented
 *   ("smoothed") with up to `speaker_frames` frames of speaker stats from
 *   previous utterances and up to `global_frames` frames of global stats, so
 *   the first frames are normalized against a reasonable distribution.
 *
 *   The frame being normalized is included in the stats that normalize it
 *   (exactly like kaldi).  Variance normalization floors the variance at
 *   1e-20.
 *
 * @note Features normalized this way are only *approximately* zero-mean,
 *   especially at utterance start.  Acoustic models must be trained with the
 *   same kind of normalization ("matched" online training).
 *
 * @note Parity with kaldi is tested frame-by-frame against
 *   `apply-cmvn-online` in tests/parity/parity_online_cmvn.py.
 *
 * Example:
 * @code
 *   knf::OnlineCmvnOptions opts;
 *   knf::OnlineCmvnState state(global_stats);  // MatrixD [2, dim+1]
 *   knf::OnlineCmvn cmvn(opts, state);
 *   for (const knf::VectorF &f : frames) {
 *     knf::VectorF y = cmvn.AcceptFeature(f);
 *   }
 *   auto next_state = cmvn.GetState();  // hand over to the next utterance
 * @endcode
 */
class OnlineCmvn {
 public:
  /**
   * @brief Construct with options and adaptation state.
   *
   * @param opts [in] The options (defaults match kaldi).
   * @param state [in] Adaptation state; `state.global_cmvn_stats` must be a
   *   2 x (dim+1) matrix.  `speaker_cmvn_stats` and `frozen_state` may be
   *   empty; if `frozen_state` is set, the normalization is frozen from the
   *   start (see Freeze()).
   */
  OnlineCmvn(const OnlineCmvnOptions &opts, const OnlineCmvnState &state);

  /// Feature dimension (dim of the global stats minus one).
  int32_t Dim() const { return dim_; }
  /// Number of frames accepted so far.
  int32_t NumFramesAccepted() const { return num_frames_; }

  /**
   * @brief Push one raw feature frame; return the normalized frame.
   *
   * @param feat [in] Pointer to `dim` raw feature values.
   * @param dim  [in] Feature dimension (must equal Dim()).
   * @return The normalized frame, same dimension.
   *
   * The frame is normalized with stats that include the frame itself (like
   * kaldi).  The raw frame also updates the speaker stats used by GetState().
   */
  VectorF AcceptFeature(const float *feat, int32_t dim);
  /// Same as AcceptFeature(const float *, int32_t).
  VectorF AcceptFeature(const VectorF &feat);

  /**
   * @brief Push a batch of raw feature frames.
   *
   * @param feats [in] Raw features, rows are frames, [T, dim].
   * @return The normalized features, [T, dim].
   */
  MatrixF AcceptFeatures(const MatrixF &feats);

  /**
   * @brief Freeze the normalization at the current frame.
   *
   * From now on frames are normalized with the current (smoothed) stats and
   * the running stats stop updating.  The frozen state is part of GetState()
   * so later utterances of the speaker keep the same normalization (kaldi
   * uses this to estimate fMLLR on top of a fixed CMVN).
   */
  void Freeze();

  /**
   * @brief Return the adaptation state for the next utterance.
   *
   * Contains the speaker stats accumulated over all accepted frames (plus
   * any incoming speaker stats), the global stats, and any frozen state —
   * equivalent to kaldi's OnlineCmvn::GetState(cur_frame).
   */
  OnlineCmvnState GetState() const;

  /**
   * @brief Smooth window stats with speaker and global stats.
   *
   * A port of kaldi's static OnlineCmvn::SmoothOnlineCmvnStats.  Adds up to
   * `speaker_frames` worth of speaker stats and up to `global_frames` worth
   * of global stats to `stats`, scaled by their counts, so the total count
   * approaches `cmn_window` (a no-op once it is reached).
   *
   * @param speaker_stats [in] Speaker stats [2, dim+1], may be empty.
   * @param global_stats  [in] Global stats [2, dim+1] (required at runtime).
   * @param opts          [in] The options in use.
   * @param stats         [in,out] Window stats to smooth.
   */
  static void SmoothOnlineCmvnStats(const MatrixD &speaker_stats,
                                    const MatrixD &global_stats,
                                    const OnlineCmvnOptions &opts,
                                    MatrixD *stats);

 private:
  void Init(const OnlineCmvnState &state);
  // Update sliding stats with frame t (add) and frame t - cmn_window
  // (subtract, if it exists), then cache the raw frame.
  void UpdateStats(const VectorF &feat);
  // Apply smoothed (or frozen) stats to one frame in place.
  void NormalizeFrame(VectorF *feat) const;

  OnlineCmvnOptions opts_;
  std::vector<int32_t> skip_dims_;
  int32_t dim_ = 0;
  int32_t num_frames_ = 0;

  // Sliding-window stats over the last <= cmn_window frames.
  MatrixD stats_;
  // Raw frame history for window subtraction: ring of cmn_window frames.
  std::vector<VectorF> raw_ring_;

  // Speaker stats of *previous* utterances, used for smoothing (kaldi's
  // orig_state_.speaker_cmvn_stats).  Fixed while frames are accepted.
  MatrixD prev_speaker_cmvn_stats_;
  // Speaker stats accumulated for GetState() (prev speaker stats + all
  // frames accepted so far), like kaldi's OnlineCmvn::GetState.
  MatrixD speaker_cmvn_stats_;
  MatrixD global_cmvn_stats_;
  MatrixD frozen_state_;  // empty unless Freeze() was called
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_ONLINE_CMVN_H_

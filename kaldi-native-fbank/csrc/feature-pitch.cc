/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-pitch.cc
//
// Copyright    2013  Pegah Ghahremani
//              2014  IMSL, PKU-HKUST (author: Wei Shi)
//              2014  Yanqing Sun, Junjie Wang,
//                    Daniel Povey, Korbinian Riedhammer
//                    Xin Lei
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/feat/pitch-functions.cc
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include "kaldi-native-fbank/csrc/feature-pitch.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-delta.h"
#include "kaldi-native-fbank/csrc/kaldi-math.h"
#include "kaldi-native-fbank/csrc/macros.h"
#include "kaldi-native-fbank/csrc/resample.h"

namespace knf {

namespace {

// kaldi draws the delta-pitch noise from RandGauss(), i.e. the Box-Muller
// transform over the global libc rand() stream.  That stream is
// platform-dependent (glibc's rand() differs from macOS's), so kaldi's
// output is not reproducible across platforms.  We reproduce the *reference*
// stream instead -- the one used to generate the golden test data in
// tests/data/feat_golden (see the README there): the classic Park-Miller
// LCG "minimal standard" that macOS/BSD rand() implements,
//
//     state <- state * 16807 mod (2^31 - 1),  state initially 1,
//
// together with kaldi's exact RandUniform()/RandGauss() formulas and
// RAND_MAX = 2^31-1.  This keeps the noise distribution and its semantics
// identical to kaldi's while making the features deterministic and
// identical on every platform (and identical to the golden data).
class ReferenceRandom {
 public:
  /// One step of the LCG; returns the new state (kaldi's Rand()).
  int32_t Rand() {
    // Schrage's trick avoids the 64-bit overflow of 16807 * state.
    int32_t hi = state_ / 127773;
    int32_t lo = state_ % 127773;
    int32_t x = 16807 * lo - 2836 * hi;
    if (x < 0) x += 2147483647;
    state_ = x;
    return x;
  }

  /// kaldi's RandUniform(): strictly between 0 and 1.
  float RandUniform() {
    return static_cast<float>((Rand() + 1.0) / (2147483647.0 + 2.0));
  }

  /// kaldi's RandGauss(): Box-Muller, sqrt(-2 log U1) * cos(2 pi U2).
  float RandGauss() {
    float u1 = RandUniform();
    float u2 = RandUniform();
    return static_cast<float>(sqrtf(-2 * Log(u1)) * cosf(2 * M_PI * u2));
  }

 private:
  int32_t state_ = 1;
};

// This function processes the NCCF n to a POV feature f by applying the
// formula
//   f = (1.0001 - n)^0.15  - 1.0
// This is a nonlinear function designed to make the output reasonably
// Gaussian distributed.  Before doing this, the NCCF distribution is in the
// range [-1, 1] but has a strong peak just before 1.0, which this function
// smooths out.
float NccfToPovFeature(float n) {
  if (n > 1.0) {
    n = 1.0;
  } else if (n < -1.0) {
    n = -1.0;
  }
  float f = pow((1.0001 - n), 0.15) - 1.0;
  assert(f - f == 0);  // check for NaN,inf.
  return f;
}

// This function processes the NCCF n to a reasonably accurate probability
// of voicing p by applying the formula:
//
//    n' = fabs(n)
//    r = -5.2 + 5.4 * exp(7.5 * (n' - 1.0)) +
//         4.8 * n' - 2.0 * exp(-10.0 * n') + 4.2 * exp(20.0 * (n' - 1.0));
//    p = 1.0 / (1 + exp(-1.0 * r));
//
// (Empirically fitted to the log-prob-ratio of voicing on the Keele
// database as a function of the NCCF; see kaldi's comment on NccfToPov in
// kaldi/src/feat/pitch-functions.cc.)
float NccfToPov(float n) {
  float ndash = fabsf(n);
  if (ndash > 1.0) ndash = 1.0;  // just in case it was slightly outside [-1, 1]

  float r = -5.2 + 5.4 * Exp(7.5 * (ndash - 1.0)) + 4.8 * ndash -
            2.0 * Exp(-10.0 * ndash) + 4.2 * Exp(20.0 * (ndash - 1.0));
  // r is the approximate log-prob-ratio of voicing, log(p/(1-p)).
  float p = 1.0 / (1 + Exp(-1.0 * r));
  assert(p - p == 0);  // Check for NaN/inf
  return p;
}

// This function computes some dot products that are required
// while computing the NCCF.
// For each integer lag from first_lag to last_lag-1, this function
// outputs to (*inner_prod)(lag - first_lag), the dot-product
// of a window starting at 0 with a window starting at
// lag.  All windows are of length nccf_window_size.  It
// outputs to (*norm_prod)(lag - first_lag), e1 * e2, where
// e1 is the dot-product of the un-shifted window with itself,
// and e2 is the dot-product of the window shifted by "lag"
// with itself.
void ComputeCorrelation(const VectorF &wave, int32_t first_lag,
                        int32_t last_lag, int32_t nccf_window_size,
                        VectorF *inner_prod, VectorF *norm_prod) {
  VectorF zero_mean_wave(wave);
  // TODO(kaldi): possibly fix this, the mean normalization is done in a
  // strange way.
  // subtract mean-frame from wave
  zero_mean_wave.array() -=
      wave.head(nccf_window_size).sum() / nccf_window_size;
  VectorF sub_vec1 = zero_mean_wave.head(nccf_window_size);
  float e1 = sub_vec1.dot(sub_vec1);
  for (int32_t lag = first_lag; lag <= last_lag; lag++) {
    VectorF sub_vec2 = zero_mean_wave.segment(lag, nccf_window_size);
    float e2 = sub_vec2.dot(sub_vec2);
    float sum = sub_vec1.dot(sub_vec2);
    (*inner_prod)(lag - first_lag) = sum;
    (*norm_prod)(lag - first_lag) = e1 * e2;
  }
}

// Computes the NCCF as a fraction of the numerator term (a dot product
// between two vectors) and a denominator term which equals
// sqrt(e1*e2 + nccf_ballast) where e1 and e2 are both dot-products of bits
// of the wave with themselves, and e1*e2 is supplied as "norm_prod".  These
// quantities are computed by "ComputeCorrelation".
void ComputeNccf(const VectorF &inner_prod, const VectorF &norm_prod,
                 float nccf_ballast, VectorF *nccf_vec) {
  assert(inner_prod.size() == norm_prod.size() &&
         inner_prod.size() == nccf_vec->size());
  for (int32_t lag = 0; lag < inner_prod.size(); lag++) {
    float numerator = inner_prod(lag);
    float denominator = pow(norm_prod(lag) + nccf_ballast, 0.5);
    float nccf = 0.0;
    if (denominator != 0.0) {
      nccf = numerator / denominator;
    } else {
      assert(numerator == 0.0);
      nccf = 0.0;
    }
    assert(nccf < 1.01 && nccf > -1.01);
    (*nccf_vec)(lag) = nccf;
  }
}

// This function selects the lags at which we measure the NCCF: we need
// to select lags from 1/max_f0 to 1/min_f0, in a geometric progression
// with ratio 1 + d.
void SelectLags(const PitchExtractionOptions &opts, VectorF *lags) {
  // choose lags relative to acceptable pitch tolerance
  float min_lag = 1.0 / opts.max_f0;
  float max_lag = 1.0 / opts.min_f0;

  std::vector<float> tmp_lags;
  // The float accumulation reproduces kaldi's exact log-spaced lag sequence.
  for (float lag = min_lag; lag <= max_lag;
       lag *= 1.0 + opts.delta_pitch)  // NOLINT
    tmp_lags.push_back(lag);
  lags->resize(tmp_lags.size());
  std::copy(tmp_lags.begin(), tmp_lags.end(), lags->data());
}

// This function computes the local-cost for the Viterbi computation,
// see eq. (5) in the paper.
//
// @param [in] nccf_pitch  The nccf as computed for the pitch computation
//                         (with ballast).
// @param [in] lags        The log-spaced lags at which nccf_pitch is sampled.
// @param [in] opts        The options as provided by the user.
// @param [out] local_cost The local-cost is output to here.
void ComputeLocalCost(const VectorF &nccf_pitch, const VectorF &lags,
                      const PitchExtractionOptions &opts, VectorF *local_cost) {
  // from the paper, eq. 5, local_cost = 1 - Phi(t,i)(1 - soft_min_f0 L_i)
  // nccf is the nccf on this frame measured at the lags in "lags".
  assert(nccf_pitch.size() == local_cost->size() &&
         nccf_pitch.size() == lags.size());
  local_cost->setOnes();
  // add the term -Phi(t,i) and the term soft_min_f0 Phi(t,i) L_i:
  *local_cost -= nccf_pitch;
  *local_cost += opts.soft_min_f0 * lags.cwiseProduct(nccf_pitch);
}

// class PitchFrameInfo is used inside class OnlinePitchFeatureImpl.
// It stores the information we need to keep around for a single frame
// of the pitch computation.
class PitchFrameInfo {
 public:
  /// This constructor is used for frame -1; it sets the costs to be all
  /// zeros the pov_nccf's to zero and the backpointers to -1.
  explicit PitchFrameInfo(int32_t num_states);

  /// This constructor is used for subsequent frames (not -1).
  explicit PitchFrameInfo(PitchFrameInfo *prev);

  /// Record the nccf_pov value.
  /// @param [in] nccf_pov  The nccf as computed for the POV computation
  ///                       (without ballast).
  void SetNccfPov(const VectorF &nccf_pov);

  /// This function may be called for frames apart from frame -1; the bulk of
  /// the Viterbi computation takes place inside this function.
  /// @param [in] opts        The options as provided by the user.
  /// @param [in] nccf_pitch  The nccf as computed for the pitch computation
  ///                         (with ballast).
  /// @param [in] lags        The log-spaced lags at which nccf_pitch and
  ///                         nccf_pov are sampled.
  /// @param [in] prev_forward_cost  The forward-cost vector for the previous
  ///                         frame.
  /// @param [out] index_info A pointer to a temporary vector used by this
  ///                         function.
  /// @param [out] this_forward_cost  The forward-cost vector for this frame
  ///                         (to be computed).
  void ComputeBacktraces(const PitchExtractionOptions &opts,
                         const VectorF &nccf_pitch, const VectorF &lags,
                         const VectorF &prev_forward_cost,
                         std::vector<std::pair<int32_t, int32_t>> *index_info,
                         VectorF *this_forward_cost);

  /// This function may be called on the last (most recent) PitchFrameInfo
  /// object with the best state (obtained from the externally held
  /// forward-costs).  It traces back as far as needed to set the
  /// cur_best_state_, and as it's going it sets the lag-index and pov_nccf in
  /// lag_nccf, which when it's called is an iterator to where to put the
  /// info for the final state; the iterator will be decremented inside this
  /// function.
  // The non-const reference matches kaldi's signature; it is written.
  void SetBestState(
      int32_t best_state,
      std::vector<std::pair<int32_t, float>> &lag_nccf);  // NOLINT

  /// This function may be called on the last (most recent) PitchFrameInfo
  /// object; it computes how many frames of latency there is because the
  /// traceback has not yet settled on a single value for frames in the past.
  /// It actually returns the minimum of max_latency and the actual latency,
  /// which is an optimization because we won't care about latency past
  /// a user-specified maximum latency.
  int32_t ComputeLatency(int32_t max_latency);

 private:
  // struct StateInfo is the information we keep for a single one of the
  // log-spaced lags, for a single frame.  This is a state in the Viterbi
  // computation.
  struct StateInfo {
    /// The state index on the previous frame that is the best preceding state
    /// for this state.
    int32_t backpointer = 0;
    /// the version of the NCCF we keep for the POV computation (without the
    /// ballast term).
    float pov_nccf = 0.0;
  };
  std::vector<StateInfo> state_info_;
  /// the state index of the first entry in "state_info"; this will initially
  /// be zero, but after cleanup might be nonzero.
  int32_t state_offset_;

  /// The current best state in the backtrace from the end.
  int32_t cur_best_state_;

  /// The structure for the previous frame (not owned; frame -1 is the end of
  /// the chain).
  PitchFrameInfo *prev_info_;
};

// This constructor is used for frame -1; it sets the costs to be all zeros
// the pov_nccf's to zero and the backpointers to -1.
PitchFrameInfo::PitchFrameInfo(int32_t num_states)
    : state_info_(num_states),
      state_offset_(0),
      cur_best_state_(-1),
      prev_info_(nullptr) {}

PitchFrameInfo::PitchFrameInfo(PitchFrameInfo *prev_info)
    : state_info_(prev_info->state_info_.size()),
      state_offset_(0),
      cur_best_state_(-1),
      prev_info_(prev_info) {}

void PitchFrameInfo::SetNccfPov(const VectorF &nccf_pov) {
  int32_t num_states = nccf_pov.size();
  assert(num_states == static_cast<int64_t>(state_info_.size()));
  for (int32_t i = 0; i < num_states; i++)
    state_info_[i].pov_nccf = nccf_pov(i);
}

void PitchFrameInfo::ComputeBacktraces(
    const PitchExtractionOptions &opts, const VectorF &nccf_pitch,
    const VectorF &lags, const VectorF &prev_forward_cost_vec,
    std::vector<std::pair<int32_t, int32_t>> *index_info,
    VectorF *this_forward_cost_vec) {
  int32_t num_states = nccf_pitch.size();

  VectorF local_cost(num_states);
  ComputeLocalCost(nccf_pitch, lags, opts, &local_cost);

  const float delta_pitch_sq = pow(Log(1.0 + opts.delta_pitch), 2.0);
  const float inter_frame_factor = delta_pitch_sq * opts.penalty_factor;

  // The cost-vector accesses below are the very inner loop of the pitch
  // tracker (a lot of time is taken here), so we use plain indexed access
  // like kaldi does, not Eigen expressions.
  const float *prev_forward_cost = prev_forward_cost_vec.data();
  float *this_forward_cost = this_forward_cost_vec->data();

  if (index_info->empty()) index_info->resize(num_states);

  // make it a reference for more concise indexing.
  std::vector<std::pair<int32_t, int32_t>> &bounds = *index_info;

  /* bounds[i].first will be a lower bound on the backpointer for state i,
     bounds[i].second will be an upper bound on it.  We progressively tighten
     these bounds till we know the backpointers exactly.
  */

  int32_t last_backpointer = 0;
  for (int32_t i = 0; i < num_states; i++) {
    int32_t start_j = last_backpointer;
    float best_cost = (start_j - i) * (start_j - i) * inter_frame_factor +
                      prev_forward_cost[start_j];
    int32_t best_j = start_j;

    for (int32_t j = start_j + 1; j < num_states; j++) {
      float this_cost =
          (j - i) * (j - i) * inter_frame_factor + prev_forward_cost[j];
      if (this_cost < best_cost) {
        best_cost = this_cost;
        best_j = j;
      } else {  // as soon as the costs stop improving, we stop searching.
        break;  // this is a loose lower bound we're getting.
      }
    }
    state_info_[i].backpointer = best_j;
    this_forward_cost[i] = best_cost;
    bounds[i].first = best_j;           // this is now a lower bound on the
                                        // backpointer.
    bounds[i].second = num_states - 1;  // we have no meaningful upper bound
                                        // yet.
    last_backpointer = best_j;
  }

  // We iterate, progressively refining the upper and lower bounds until they
  // meet and we know that the resulting backtraces are optimal.  Each
  // iteration takes time linear in num_states.  We won't normally iterate as
  // far as num_states; normally we only do two iterations; when printing out
  // the number of iterations, it's rarely more than that (once I saw seven
  // iterations).  Anyway, this part of the computation does not dominate.
  for (int32_t iter = 0; iter < num_states; iter++) {
    bool changed = false;
    if (iter % 2 == 0) {  // go backwards through the states
      last_backpointer = num_states - 1;
      for (int32_t i = num_states - 1; i >= 0; i--) {
        int32_t lower_bound = bounds[i].first;
        int32_t upper_bound = std::min(last_backpointer, bounds[i].second);
        if (upper_bound == lower_bound) {
          last_backpointer = lower_bound;
          continue;
        }
        float best_cost = this_forward_cost[i];
        int32_t best_j = state_info_[i].backpointer;
        int32_t initial_best_j = best_j;

        if (best_j == upper_bound) {
          // if best_j already equals upper bound, don't bother tightening the
          // upper bound, we'll tighten the lower bound when the time comes.
          last_backpointer = best_j;
          continue;
        }
        // Below, we have j > lower_bound + 1 because we know we've already
        // evaluated lower_bound and lower_bound + 1 [via knowledge of
        // this algorithm.]
        for (int32_t j = upper_bound; j > lower_bound + 1; j--) {
          float this_cost =
              (j - i) * (j - i) * inter_frame_factor + prev_forward_cost[j];
          if (this_cost < best_cost) {
            best_cost = this_cost;
            best_j = j;
          } else {  // as soon as the costs stop improving, we stop searching,
            // unless the best j is still lower than j, in which case
            // we obviously need to keep moving.
            if (best_j > j)
              break;  // this is a loose lower bound we're getting.
          }
        }
        // our "best_j" is now an upper bound on the backpointer.
        bounds[i].second = best_j;
        if (best_j != initial_best_j) {
          this_forward_cost[i] = best_cost;
          state_info_[i].backpointer = best_j;
          changed = true;
        }
        last_backpointer = best_j;
      }
    } else {  // go forwards through the states.
      last_backpointer = 0;
      for (int32_t i = 0; i < num_states; i++) {
        int32_t lower_bound = std::max(last_backpointer, bounds[i].first);
        int32_t upper_bound = bounds[i].second;
        if (upper_bound == lower_bound) {
          last_backpointer = lower_bound;
          continue;
        }
        float best_cost = this_forward_cost[i];
        int32_t best_j = state_info_[i].backpointer;
        int32_t initial_best_j = best_j;

        if (best_j == lower_bound) {
          // if best_j already equals lower bound, we don't bother tightening
          // the lower bound, we'll tighten the upper bound when the time
          // comes.
          last_backpointer = best_j;
          continue;
        }
        // Below, we have j < upper_bound because we know we've already
        // evaluated that point.
        for (int32_t j = lower_bound; j < upper_bound - 1; j++) {
          float this_cost =
              (j - i) * (j - i) * inter_frame_factor + prev_forward_cost[j];
          if (this_cost < best_cost) {
            best_cost = this_cost;
            best_j = j;
          } else {  // as soon as the costs stop improving, we stop searching,
            // unless the best j is still higher than j, in which case
            // we obviously need to keep moving.
            if (best_j < j)
              break;  // this is a loose lower bound we're getting.
          }
        }
        // our "best_j" is now a lower bound on the backpointer.
        bounds[i].first = best_j;
        if (best_j != initial_best_j) {
          this_forward_cost[i] = best_cost;
          state_info_[i].backpointer = best_j;
          changed = true;
        }
        last_backpointer = best_j;
      }
    }
    if (!changed) break;
  }
  // The next statement is needed due to RecomputeBacktraces: we have to
  // invalidate the previously computed best-state info.
  cur_best_state_ = -1;
  *this_forward_cost_vec += local_cost;
}

void PitchFrameInfo::SetBestState(
    int32_t best_state, std::vector<std::pair<int32_t, float>> &lag_nccf) {
  // This function would naturally be recursive, but we have coded this to
  // avoid recursion, which would otherwise eat up the stack.  Think of it as
  // a static member function, except we do use "this" right at the beginning.

  std::vector<std::pair<int32_t, float>>::reverse_iterator iter =
      lag_nccf.rbegin();

  PitchFrameInfo *this_info = this;  // it will change in the loop.
  while (this_info != nullptr) {
    PitchFrameInfo *prev_info = this_info->prev_info_;
    if (best_state == this_info->cur_best_state_) return;  // no change
    if (prev_info != nullptr)  // don't write anything for frame -1.
      iter->first = best_state;
    size_t state_info_index = best_state - this_info->state_offset_;
    assert(state_info_index < this_info->state_info_.size());
    this_info->cur_best_state_ = best_state;
    best_state = this_info->state_info_[state_info_index].backpointer;
    if (prev_info != nullptr)  // don't write anything for frame -1.
      iter->second = this_info->state_info_[state_info_index].pov_nccf;
    this_info = prev_info;
    if (this_info != nullptr) ++iter;
  }
}

int32_t PitchFrameInfo::ComputeLatency(int32_t max_latency) {
  if (max_latency <= 0) return 0;

  int32_t latency = 0;

  // This function would naturally be recursive, but we have coded this to
  // avoid recursion, which would otherwise eat up the stack.  Think of it as
  // a static member function, except we do use "this" right at the beginning.
  // This function is called only on the most recent PitchFrameInfo object.
  int32_t num_states = state_info_.size();
  int32_t min_living_state = 0;
  int32_t max_living_state = num_states - 1;
  PitchFrameInfo *this_info = this;  // it will change in the loop.

  for (; this_info != nullptr && latency < max_latency;) {
    int32_t offset = this_info->state_offset_;
    assert(min_living_state >= offset &&
           max_living_state - offset <
               static_cast<int64_t>(this_info->state_info_.size()));
    min_living_state =
        this_info->state_info_[min_living_state - offset].backpointer;
    max_living_state =
        this_info->state_info_[max_living_state - offset].backpointer;
    if (min_living_state == max_living_state) {
      return latency;
    }
    this_info = this_info->prev_info_;
    if (this_info != nullptr)  // avoid incrementing latency for frame -1,
      latency++;               // as it's not a real frame.
  }
  return latency;
}

// struct NccfInfo is used to cache certain quantities that we need for online
// operation, for the first "recompute_frame" frames of the file (e.g. 300);
// after that many frames, or after the user calls InputFinished(), we redo
// the initial backtraces, as we'll then have a better estimate of the
// average signal energy.
struct NccfInfo {
  VectorF nccf_pitch_resampled;  ///< resampled nccf_pitch
  float avg_norm_prod;           ///< average value of e1 * e2.
  float mean_square_energy;      ///< mean_square energy we used when computing
                                 ///< the original ballast term for
                                 ///< "nccf_pitch_resampled".

  NccfInfo(float avg_norm_prod, float mean_square_energy)
      : avg_norm_prod(avg_norm_prod), mean_square_energy(mean_square_energy) {}
};

// We could inherit from an online-feature interface as we have the same
// interface, but this will unnecessarily force a lot of our functions to be
// virtual.  This is kaldi's OnlinePitchFeatureImpl; it outputs the raw
// two-dimensional feature (NCCF, pitch).
class OnlinePitchFeatureImpl {
 public:
  explicit OnlinePitchFeatureImpl(const PitchExtractionOptions &opts);

  int32_t Dim() const { return 2; }

  /// Return the options this object was constructed with.
  const PitchExtractionOptions &opts() const { return opts_; }

  float FrameShiftInSeconds() const;

  int32_t NumFramesReady() const;

  bool IsLastFrame(int32_t frame) const;

  void GetFrame(int32_t frame, VectorF *feat);

  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);

  void InputFinished();

  ~OnlinePitchFeatureImpl();

 private:
  /// This function works out from the signal how many frames are currently
  /// available to process (this is called from inside AcceptWaveform()).
  /// Note: the number of frames differs slightly from the number the
  /// old pitch code gave.
  /// Note: the number this returns depends on whether input_finished_ ==
  /// true; if it is, it will "force out" a final frame or two.
  int32_t NumFramesAvailable(int64_t num_downsampled_samples,
                             bool snip_edges) const;

  /// This function extracts from the signal the samples numbered from
  /// "sample_index" (numbered in the full downsampled signal, not just this
  /// part), and of length equal to window.size().  It uses the data members
  /// downsampled_samples_discarded_ and downsampled_signal_remainder_, as
  /// well as the more recent part of the downsampled wave
  /// "downsampled_wave_part" which is provided.
  ///
  /// @param [in] downsampled_wave_part  One chunk of the downsampled wave,
  ///                      starting from sample-index
  ///                      downsampled_samples_processed_.
  /// @param [in] sample_index  The desired starting sample index (measured
  ///                      from the start of the whole signal, not just this
  ///                      part).
  /// @param [out] window  The part of the signal is output to here.
  void ExtractFrame(const VectorF &downsampled_wave_part, int64_t sample_index,
                    Eigen::Ref<VectorF> window);

  /// This function is called after we reach frame "recompute_frame", or when
  /// InputFinished() is called, whichever comes sooner.  It recomputes the
  /// backtraces for frames zero through recompute_frame, if needed because
  /// the average energy of the signal has changed, affecting the nccf ballast
  /// term.  It works out the average signal energy from
  /// downsampled_samples_processed_, signal_sum_ and signal_sumsq_ (which, if
  /// you see the calling code, might include more frames than just
  /// "recompute_frame", it might include up to the end of the current chunk).
  void RecomputeBacktraces();

  /// This function updates downsampled_signal_remainder_,
  /// downsampled_samples_processed_, signal_sum_ and signal_sumsq_; it's
  /// called from AcceptWaveform().
  void UpdateRemainder(const VectorF &downsampled_wave_part);

  // The following variables don't change throughout the lifetime
  // of this object.
  PitchExtractionOptions opts_;

  // the first lag of the downsampled signal at which we measure NCCF
  int32_t nccf_first_lag_;
  // the last lag of the downsampled signal at which we measure NCCF
  int32_t nccf_last_lag_;

  // The log-spaced lags at which we will resample the NCCF
  VectorF lags_;

  // This object is used to resample from evenly spaced to log-evenly-spaced
  // nccf values.
  std::unique_ptr<ArbitraryResample> nccf_resampler_;

  // The following objects may change during the lifetime of this object.

  // This object is used to resample the signal.
  std::unique_ptr<LinearResample> signal_resampler_;

  // frame_info_ is indexed by [frame-index + 1].  frame_info_[0] is an object
  // that corresponds to frame -1, which is not a real frame.
  std::vector<std::unique_ptr<PitchFrameInfo>> frame_info_;

  // nccf_info_ is indexed by frame-index, from frame 0 to at most
  // opts_.recompute_frame - 1.  It contains some information we'll
  // need to recompute the tracebacks after getting a better estimate
  // of the average energy of the signal.
  std::vector<std::unique_ptr<NccfInfo>> nccf_info_;

  // Current number of frames which we can't output because Viterbi has not
  // converged for them, or opts_.max_frames_latency if we have reached that
  // limit.
  int32_t frames_latency_ = 0;

  // The forward-cost at the current frame (the last frame in frame_info_);
  // this has the same dimension as lags_.  We normalize each time so
  // the lowest cost is zero, for numerical accuracy and so we can use float.
  VectorF forward_cost_;

  // stores the constant part of forward_cost_.
  double forward_cost_remainder_ = 0.0;

  // The resampled-lag index and the NCCF (as computed for POV, without
  // ballast term) for each frame, as determined by Viterbi traceback from the
  // best final state.
  std::vector<std::pair<int32_t, float>> lag_nccf_;

  bool input_finished_ = false;

  /// sum-squared of previously processed parts of signal; used to get NCCF
  /// ballast term.  Denominator is downsampled_samples_processed_.
  double signal_sumsq_ = 0.0;

  /// sum of previously processed parts of signal; used to do mean-subtraction
  /// when getting sum-squared, along with signal_sumsq_.
  double signal_sum_ = 0.0;

  /// downsampled_samples_processed is the number of samples (after
  /// downsampling) that we got in previous calls to AcceptWaveform().
  int64_t downsampled_samples_processed_ = 0;

  /// This is a small remainder of the previous downsampled signal;
  /// it's used by ExtractFrame for frames near the boundary of two
  /// waveforms supplied to AcceptWaveform().
  VectorF downsampled_signal_remainder_;
};

OnlinePitchFeatureImpl::OnlinePitchFeatureImpl(
    const PitchExtractionOptions &opts)
    : opts_(opts) {
  signal_resampler_ = std::make_unique<LinearResample>(
      opts.samp_freq, opts.resample_freq, opts.lowpass_cutoff,
      opts.lowpass_filter_width);

  double outer_min_lag = 1.0 / opts.max_f0 - (opts.upsample_filter_width /
                                              (2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / opts.min_f0 + (opts.upsample_filter_width /
                                              (2.0 * opts.resample_freq));
  nccf_first_lag_ = ceil(opts.resample_freq * outer_min_lag);
  nccf_last_lag_ = floor(opts.resample_freq * outer_max_lag);

  // Choose the lags at which we resample the NCCF.
  SelectLags(opts, &lags_);

  // upsample_cutoff is the filter cutoff for upsampling the NCCF, which is
  // the Nyquist of the resampling frequency.  The NCCF is (almost
  // completely) bandlimited to around "lowpass_cutoff" (1000 by default),
  // and when the spectrum of this bandlimited signal is convolved with the
  // spectrum of an impulse train with frequency "resample_freq", which are
  // separated by 4kHz, we get energy at -5000,-3000, -1000...1000,
  // 3000..5000, etc.  Filtering at half the Nyquist (2000 by default) is
  // sufficient to get only the first repetition.
  float upsample_cutoff = opts.resample_freq * 0.5;

  std::vector<float> lags_offset(lags_.data(), lags_.data() + lags_.size());
  // lags_offset equals lags_ (which are the log-spaced lag values we want to
  // measure the NCCF at) with nccf_first_lag_ / opts.resample_freq subtracted
  // from each element, so we can treat the measured NCCF values as starting
  // from sample zero in a signal that starts at the point start /
  // opts.resample_freq.  This is necessary because the ArbitraryResample code
  // assumes that the input signal starts from sample zero.
  for (float &lag_offset : lags_offset)
    lag_offset -= nccf_first_lag_ / opts.resample_freq;

  int32_t num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_;

  nccf_resampler_ = std::make_unique<ArbitraryResample>(
      num_measured_lags, opts.resample_freq, upsample_cutoff, lags_offset,
      opts.upsample_filter_width);

  // add a PitchFrameInfo object for frame -1 (not a real frame).
  frame_info_.push_back(std::make_unique<PitchFrameInfo>(lags_.size()));
  // zeroes forward_cost_; this is what we want for the fake frame -1.
  forward_cost_.resize(lags_.size());
  forward_cost_.setZero();
}

int32_t OnlinePitchFeatureImpl::NumFramesAvailable(
    int64_t num_downsampled_samples, bool snip_edges) const {
  int32_t frame_shift = opts_.NccfWindowShift();
  int32_t frame_length = opts_.NccfWindowSize();
  // Use the "full frame length" to compute the number
  // of frames only if the input is not finished.
  if (!input_finished_) frame_length += nccf_last_lag_;
  if (num_downsampled_samples < frame_length) {
    return 0;
  } else {
    if (!snip_edges) {
      if (input_finished_) {
        // The +0.5 rounding reproduces kaldi's exact frame-count formula.
        return static_cast<int32_t>(
            num_downsampled_samples * 1.0f / frame_shift + 0.5f);  // NOLINT
      } else {
        // kaldi computes frame_length / 2 as integer division and rounds by
        // adding 0.5 then truncating; keep that exact arithmetic.
        int64_t num = num_downsampled_samples - frame_length / 2;      // NOLINT
        return static_cast<int32_t>(num * 1.0f / frame_shift + 0.5f);  // NOLINT
      }
    } else {
      return static_cast<int32_t>(
          (num_downsampled_samples - frame_length) / frame_shift + 1);
    }
  }
}

void OnlinePitchFeatureImpl::UpdateRemainder(
    const VectorF &downsampled_wave_part) {
  // frame_info_ has an extra element at frame-1, so subtract
  // one from the length.
  int64_t num_frames = static_cast<int64_t>(frame_info_.size()) - 1;
  int64_t next_frame = num_frames;
  int64_t frame_shift = opts_.NccfWindowShift();
  int64_t next_frame_sample = frame_shift * next_frame;

  signal_sumsq_ += downsampled_wave_part.dot(downsampled_wave_part);
  signal_sum_ += downsampled_wave_part.sum();

  // next_frame_sample is the first sample index we'll need for the
  // next frame.
  int64_t next_downsampled_samples_processed =
      downsampled_samples_processed_ + downsampled_wave_part.size();

  if (next_frame_sample > next_downsampled_samples_processed) {
    // this could only happen in the weird situation that the full frame
    // length is less than the frame shift.
    int32_t full_frame_length = opts_.NccfWindowSize() + nccf_last_lag_;
    assert(full_frame_length < frame_shift && "Code error");
    downsampled_signal_remainder_.resize(0);
  } else {
    VectorF new_remainder(next_downsampled_samples_processed -
                          next_frame_sample);
    // note: next_frame_sample is the index into the entire signal, of
    // new_remainder(0).
    // i is the absolute index of the signal.
    for (int64_t i = next_frame_sample; i < next_downsampled_samples_processed;
         i++) {
      if (i >= downsampled_samples_processed_) {  // in current signal.
        new_remainder(i - next_frame_sample) =
            downsampled_wave_part(i - downsampled_samples_processed_);
      } else {  // in old remainder; only reach here if waveform supplied is
        new_remainder(i - next_frame_sample) =  //  tiny.
            downsampled_signal_remainder_(i - downsampled_samples_processed_ +
                                          downsampled_signal_remainder_.size());
      }
    }
    downsampled_signal_remainder_.swap(new_remainder);
  }
  downsampled_samples_processed_ = next_downsampled_samples_processed;
}

void OnlinePitchFeatureImpl::ExtractFrame(const VectorF &downsampled_wave_part,
                                          int64_t sample_index,
                                          Eigen::Ref<VectorF> window) {
  int32_t full_frame_length = window.size();
  int32_t offset =
      static_cast<int32_t>(sample_index - downsampled_samples_processed_);

  // Treat edge cases first
  if (sample_index < 0) {
    // Part of the frame is before the beginning of the signal. This
    // should only happen if opts_.snip_edges == false, when we are
    // processing the first few frames of signal. In this case
    // we pad with zeros.
    assert(opts_.snip_edges == false);
    int32_t sub_frame_length = sample_index + full_frame_length;
    int32_t sub_frame_index = full_frame_length - sub_frame_length;
    assert(sub_frame_length > 0 && sub_frame_index > 0);
    window.setZero();
    ExtractFrame(downsampled_wave_part, 0,
                 window.segment(sub_frame_index, sub_frame_length));
    return;
  }

  if (offset + full_frame_length > downsampled_wave_part.size()) {
    // Requested frame is past end of the signal.  This should only happen if
    // input_finished_ == true, when we're flushing out the last couple of
    // frames of signal.  In this case we pad with zeros.
    assert(input_finished_);
    int32_t sub_frame_length = downsampled_wave_part.size() - offset;
    assert(sub_frame_length > 0);
    window.setZero();
    ExtractFrame(downsampled_wave_part, sample_index,
                 window.head(sub_frame_length));
    return;
  }

  // "offset" is the offset of the start of the frame, into this
  // signal.
  if (offset >= 0) {
    // frame is full inside the new part of the signal.
    window = downsampled_wave_part.segment(offset, full_frame_length);
  } else {
    // frame is partly in the remainder and partly in the new part.
    int32_t remainder_offset = downsampled_signal_remainder_.size() + offset;
    assert(remainder_offset >= 0);  // or we didn't keep enough remainder.
    assert(offset + full_frame_length > 0);  // or we should have processed
                                             // this frame last time.

    int32_t old_length = -offset;
    int32_t new_length = offset + full_frame_length;
    window.head(old_length) =
        downsampled_signal_remainder_.segment(remainder_offset, old_length);
    window.segment(old_length, new_length) =
        downsampled_wave_part.head(new_length);
  }
  if (opts_.preemph_coeff != 0.0) {
    float preemph_coeff = opts_.preemph_coeff;
    for (int32_t i = window.size() - 1; i > 0; i--)
      window(i) -= preemph_coeff * window(i - 1);
    window(0) *= (1.0 - preemph_coeff);
  }
}

bool OnlinePitchFeatureImpl::IsLastFrame(int32_t frame) const {
  int32_t T = NumFramesReady();
  assert(frame < T);
  return (input_finished_ && frame + 1 == T);
}

float OnlinePitchFeatureImpl::FrameShiftInSeconds() const {
  return opts_.frame_shift_ms / 1000.0f;
}

int32_t OnlinePitchFeatureImpl::NumFramesReady() const {
  int32_t num_frames = lag_nccf_.size();
  int32_t latency = frames_latency_;
  assert(latency <= num_frames);
  return num_frames - latency;
}

void OnlinePitchFeatureImpl::GetFrame(int32_t frame, VectorF *feat) {
  assert(frame < NumFramesReady() && feat->size() == 2);
  (*feat)(0) = lag_nccf_[frame].second;
  (*feat)(1) = 1.0 / lags_(lag_nccf_[frame].first);
}

void OnlinePitchFeatureImpl::InputFinished() {
  input_finished_ = true;
  // Process an empty waveform; this has an effect because
  // after setting input_finished_ to true, NumFramesAvailable()
  // will return a slightly larger number.
  AcceptWaveform(opts_.samp_freq, nullptr, 0);
  int32_t num_frames = static_cast<int32_t>(frame_info_.size() - 1);
  if (num_frames < opts_.recompute_frame && !opts_.nccf_ballast_online)
    RecomputeBacktraces();
  frames_latency_ = 0;
}

// see comment with declaration.  This is only relevant for online
// operation (it gets called for non-online mode, but is a no-op).
void OnlinePitchFeatureImpl::RecomputeBacktraces() {
  assert(!opts_.nccf_ballast_online);
  int32_t num_frames = static_cast<int32_t>(frame_info_.size()) - 1;

  // The assertion reflects how we believe this function will be called.
  assert(num_frames <= opts_.recompute_frame);
  assert(nccf_info_.size() == static_cast<size_t>(num_frames));
  if (num_frames == 0) return;
  double num_samp = downsampled_samples_processed_;
  double sum = signal_sum_;
  double sumsq = signal_sumsq_;
  double mean = sum / num_samp;
  float mean_square = sumsq / num_samp - mean * mean;

  bool must_recompute = false;
  float threshold = 0.01;
  for (int32_t frame = 0; frame < num_frames; frame++)
    if (!ApproxEqual(nccf_info_[frame]->mean_square_energy, mean_square,
                     threshold))
      must_recompute = true;

  if (!must_recompute) {
    // Nothing to do.  We'll reach here, for instance, if everything was in
    // one chunk and opts_.nccf_ballast_online == false.  This is the case for
    // offline processing.
    nccf_info_.clear();
    return;
  }

  int32_t num_states = forward_cost_.size();
  int32_t basic_frame_length = opts_.NccfWindowSize();

  float new_nccf_ballast =
      pow(mean_square * basic_frame_length, 2) * opts_.nccf_ballast;

  double forward_cost_remainder = 0.0;
  VectorF forward_cost(num_states);  // start off at zero.
  forward_cost.setZero();
  VectorF next_forward_cost(forward_cost);
  std::vector<std::pair<int32_t, int32_t>> index_info;

  for (int32_t frame = 0; frame < num_frames; frame++) {
    NccfInfo &nccf_info = *nccf_info_[frame];
    float old_mean_square = nccf_info_[frame]->mean_square_energy;
    float avg_norm_prod = nccf_info_[frame]->avg_norm_prod;
    float old_nccf_ballast =
        powf(old_mean_square * basic_frame_length, 2) * opts_.nccf_ballast;
    float nccf_scale = powf(
        (old_nccf_ballast + avg_norm_prod) / (new_nccf_ballast + avg_norm_prod),
        static_cast<float>(0.5));
    // The "nccf_scale" is an estimate of the scaling factor by which the NCCF
    // would change on this frame, on average, by changing the ballast term
    // from "old_nccf_ballast" to "new_nccf_ballast".  It's not exact because
    // the "avg_norm_prod" is just an average of the product e1 * e2 of frame
    // energies of the (frame, shifted-frame), but these won't change that
    // much within a frame, and even if they do, the inaccuracy of the scaled
    // NCCF will still be very small if the ballast term didn't change much,
    // or if it's much larger or smaller than e1*e2.  By doing it as a simple
    // scaling, we save the overhead of the NCCF resampling, which is a
    // considerable part of the whole computation.
    nccf_info.nccf_pitch_resampled *= nccf_scale;

    frame_info_[frame + 1]->ComputeBacktraces(
        opts_, nccf_info.nccf_pitch_resampled, lags_, forward_cost, &index_info,
        &next_forward_cost);

    forward_cost.swap(next_forward_cost);
    float remainder = forward_cost.minCoeff();
    forward_cost_remainder += remainder;
    forward_cost.array() -= remainder;
  }

  forward_cost_remainder_ = forward_cost_remainder;
  forward_cost_.swap(forward_cost);

  Eigen::Index best_final_state = 0;
  forward_cost_.minCoeff(&best_final_state);

  if (lag_nccf_.size() != static_cast<size_t>(num_frames))
    lag_nccf_.resize(num_frames);

  frame_info_.back()->SetBestState(static_cast<int32_t>(best_final_state),
                                   lag_nccf_);
  frames_latency_ =
      frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
  nccf_info_.clear();
}

OnlinePitchFeatureImpl::~OnlinePitchFeatureImpl() = default;

void OnlinePitchFeatureImpl::AcceptWaveform(float sampling_rate,
                                            const float *waveform, int32_t n) {
  (void)sampling_rate;  // must match opts_.samp_freq; checked by OnlinePitch.
  // flush out the last few samples of input waveform only if
  // input_finished_ == true.
  const bool flush = input_finished_;

  VectorF wave = (waveform == nullptr || n == 0) ? VectorF() : VectorF(n);
  if (n > 0) {
    for (int32_t i = 0; i < n; ++i) wave(i) = waveform[i];
  }

  VectorF downsampled_wave;
  signal_resampler_->Resample(wave, flush, &downsampled_wave);

  // these variables will be used to compute the root-mean-square value of the
  // signal for the ballast term.
  double cur_sumsq = signal_sumsq_;
  double cur_sum = signal_sum_;
  int64_t cur_num_samp = downsampled_samples_processed_;
  int64_t prev_frame_end_sample = 0;
  if (!opts_.nccf_ballast_online) {
    cur_sumsq += downsampled_wave.dot(downsampled_wave);
    cur_sum += downsampled_wave.sum();
    cur_num_samp += downsampled_wave.size();
  }

  // end_frame is the total number of frames we can now process, including
  // previously processed ones.
  int32_t end_frame = NumFramesAvailable(
      downsampled_samples_processed_ + downsampled_wave.size(),
      opts_.snip_edges);
  // "start_frame" is the first frame-index we process
  int32_t start_frame = frame_info_.size() - 1;
  int32_t num_new_frames = end_frame - start_frame;

  if (num_new_frames == 0) {
    UpdateRemainder(downsampled_wave);
    return;
    // continuing to the rest of the code would generate
    // an error when sizing matrices with zero rows, and
    // anyway is a waste of time.
  }

  int32_t num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_;
  int32_t num_resampled_lags = lags_.size();
  int32_t frame_shift = opts_.NccfWindowShift();
  int32_t basic_frame_length = opts_.NccfWindowSize();
  int32_t full_frame_length = basic_frame_length + nccf_last_lag_;

  VectorF window(full_frame_length);
  VectorF inner_prod(num_measured_lags);
  VectorF norm_prod(num_measured_lags);
  MatrixF nccf_pitch(num_new_frames, num_measured_lags);
  MatrixF nccf_pov(num_new_frames, num_measured_lags);

  VectorF cur_forward_cost(num_resampled_lags);

  // Because the resampling of the NCCF is more efficient when grouped
  // together, we first compute the NCCF for all frames, then resample as a
  // matrix, then do the Viterbi [that happens inside the constructor of
  // PitchFrameInfo].

  for (int32_t frame = start_frame; frame < end_frame; frame++) {
    // start_sample is index into the whole wave, not just this part.
    int64_t start_sample = 0;
    if (opts_.snip_edges) {
      // Usual case: offset starts at 0
      start_sample = static_cast<int64_t>(frame) * frame_shift;
    } else {
      // When we are not snipping the edges, the first offsets may be
      // negative. In this case we will pad with zeros, it should not impact
      // the pitch tracker.
      start_sample = static_cast<int64_t>((frame + 0.5) * frame_shift) -
                     full_frame_length / 2;
    }
    ExtractFrame(downsampled_wave, start_sample, window);
    if (opts_.nccf_ballast_online) {
      // use only up to end of current frame to compute root-mean-square value.
      // end_sample will be the sample-index into "downsampled_wave", so
      // not really comparable to start_sample.
      int64_t end_sample =
          start_sample + full_frame_length - downsampled_samples_processed_;
      assert(end_sample > 0);  // or should have processed this frame last
                               // time.  Note: end_sample is one past last
                               // sample.
      if (end_sample > downsampled_wave.size()) {
        assert(input_finished_);
        end_sample = downsampled_wave.size();
      }
      VectorF new_part = downsampled_wave.segment(
          prev_frame_end_sample, end_sample - prev_frame_end_sample);
      cur_num_samp += new_part.size();
      cur_sumsq += new_part.dot(new_part);
      cur_sum += new_part.sum();
      prev_frame_end_sample = end_sample;
    }
    double mean_square =
        cur_sumsq / cur_num_samp - pow(cur_sum / cur_num_samp, 2.0);

    ComputeCorrelation(window, nccf_first_lag_, nccf_last_lag_,
                       basic_frame_length, &inner_prod, &norm_prod);
    double nccf_ballast_pov = 0.0;
    double nccf_ballast_pitch =
        pow(mean_square * basic_frame_length, 2) * opts_.nccf_ballast;
    double avg_norm_prod = norm_prod.sum() / norm_prod.size();
    VectorF nccf_pitch_row(num_measured_lags);
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pitch, &nccf_pitch_row);
    nccf_pitch.row(frame - start_frame) = nccf_pitch_row.transpose();
    VectorF nccf_pov_row(num_measured_lags);
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pov, &nccf_pov_row);
    nccf_pov.row(frame - start_frame) = nccf_pov_row.transpose();
    if (frame < opts_.recompute_frame)
      nccf_info_.push_back(
          std::make_unique<NccfInfo>(avg_norm_prod, mean_square));
  }

  MatrixF nccf_pitch_resampled(num_new_frames, num_resampled_lags);
  nccf_resampler_->Resample(nccf_pitch, &nccf_pitch_resampled);
  nccf_pitch.resize(0, 0);  // no longer needed.
  MatrixF nccf_pov_resampled(num_new_frames, num_resampled_lags);
  nccf_resampler_->Resample(nccf_pov, &nccf_pov_resampled);
  nccf_pov.resize(0, 0);  // no longer needed.

  // We've finished dealing with the waveform so we can call UpdateRemainder
  // now; we need to call it before we possibly call RecomputeBacktraces()
  // below, which is why we don't do it at the very end.
  UpdateRemainder(downsampled_wave);

  std::vector<std::pair<int32_t, int32_t>> index_info;

  for (int32_t frame = start_frame; frame < end_frame; frame++) {
    int32_t frame_idx = frame - start_frame;
    PitchFrameInfo *prev_info = frame_info_.back().get();
    std::unique_ptr<PitchFrameInfo> cur_info =
        std::make_unique<PitchFrameInfo>(prev_info);
    VectorF nccf_pov_row = nccf_pov_resampled.row(frame_idx).transpose();
    cur_info->SetNccfPov(nccf_pov_row);
    VectorF nccf_pitch_row = nccf_pitch_resampled.row(frame_idx).transpose();
    cur_info->ComputeBacktraces(opts_, nccf_pitch_row, lags_, forward_cost_,
                                &index_info, &cur_forward_cost);
    forward_cost_.swap(cur_forward_cost);
    // Renormalize forward_cost so smallest element is zero.
    float remainder = forward_cost_.minCoeff();
    forward_cost_remainder_ += remainder;
    forward_cost_.array() -= remainder;
    frame_info_.push_back(std::move(cur_info));
    if (frame < opts_.recompute_frame)
      nccf_info_[frame]->nccf_pitch_resampled =
          nccf_pitch_resampled.row(frame_idx).transpose();
    if (frame == opts_.recompute_frame - 1 && !opts_.nccf_ballast_online)
      RecomputeBacktraces();
  }

  // Trace back the best-path.
  Eigen::Index best_final_state = 0;
  forward_cost_.minCoeff(&best_final_state);
  lag_nccf_.resize(frame_info_.size() - 1);  // will keep any existing data.
  frame_info_.back()->SetBestState(static_cast<int32_t>(best_final_state),
                                   lag_nccf_);
  frames_latency_ =
      frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
}

// This online class implements post processing of pitch features.
// Inputs are original 2 dims (nccf, pitch).  It can produce various kinds of
// outputs; with the default options it will be (pov-feature,
// normalized-log-pitch, delta-log-pitch).  This is kaldi's OnlineProcessPitch.
class OnlineProcessPitch {
 public:
  OnlineProcessPitch(const ProcessPitchOptions &opts,
                     OnlinePitchFeatureImpl *src)
      : opts_(opts), src_(src), dim_(opts.Dim()) {
    if (dim_ == 0) {
      throw std::invalid_argument(
          "ProcessPitchOptions: at least one of the add_* features "
          "must be chosen");
    }
    if (src->Dim() != kRawFeatureDim) {
      throw std::invalid_argument(
          "OnlineProcessPitch: input feature must be pitch feature "
          "(should have dimension 2)");
    }
  }

  int32_t Dim() const { return dim_; }

  bool IsLastFrame(int32_t frame) const {
    if (frame <= -1)
      return src_->IsLastFrame(-1);
    else if (frame < opts_.delay)
      return src_->IsLastFrame(-1) == true ? false : src_->IsLastFrame(0);
    else
      return src_->IsLastFrame(frame - opts_.delay);
  }

  float FrameShiftInSeconds() const { return src_->FrameShiftInSeconds(); }

  int32_t NumFramesReady() const {
    int32_t src_frames_ready = src_->NumFramesReady();
    if (src_frames_ready == 0) {
      return 0;
    } else if (src_->IsLastFrame(src_frames_ready - 1)) {
      return src_frames_ready + opts_.delay;
    } else {
      return std::max(0, src_frames_ready - opts_.normalization_right_context +
                             opts_.delay);
    }
  }

  void GetFrame(int32_t frame, VectorF *feat) {
    int32_t frame_delayed = frame < opts_.delay ? 0 : frame - opts_.delay;
    if (!(feat->size() == dim_ && frame_delayed < NumFramesReady())) {
      throw std::runtime_error("OnlineProcessPitch: bad frame index");
    }
    int32_t index = 0;
    if (opts_.add_pov_feature) (*feat)(index++) = GetPovFeature(frame_delayed);
    if (opts_.add_normalized_log_pitch)
      (*feat)(index++) = GetNormalizedLogPitchFeature(frame_delayed);
    if (opts_.add_delta_pitch)
      (*feat)(index++) = GetDeltaPitchFeature(frame_delayed);
    if (opts_.add_raw_log_pitch)
      (*feat)(index++) = GetRawLogPitchFeature(frame_delayed);
    assert(index == dim_);
  }

 private:
  static constexpr int32_t kRawFeatureDim = 2;  // (nccf, pitch)

  // Note: we have to be a bit careful in this implementation because
  // the input features may change.  That is: if we call
  // src_->GetFrame(t, &vec) from GetFrame(), we can't guarantee that a later
  // call to src_->GetFrame(t, &vec) from another GetFrame() will return the
  // same value.  In fact, while designing this class we used some knowledge
  // of how the OnlinePitchFeature class works to minimize the amount of
  // re-querying we had to do.

  /// Computes and returns the POV feature for this frame.
  /// Called from GetFrame().
  float GetPovFeature(int32_t frame) const {
    VectorF tmp(kRawFeatureDim);
    src_->GetFrame(frame, &tmp);  // (NCCF, pitch) from pitch extractor
    float nccf = tmp(0);
    return opts_.pov_scale * NccfToPovFeature(nccf) + opts_.pov_offset;
  }

  /// Computes and returns the delta-log-pitch feature for this frame.
  /// Called from GetFrame().
  float GetDeltaPitchFeature(int32_t frame) {
    // Rather than computing the delta pitch directly in code here,
    // which might seem easier, we accumulate a small window of features
    // and call ComputeDeltas.  This might seem like overkill; the reason
    // we do it this way is to ensure that the end effects (at file
    // beginning and end) are handled in a consistent way.
    int32_t context = opts_.delta_window;
    int32_t start_frame = std::max(0, frame - context);
    int32_t end_frame = std::min(frame + context + 1, src_->NumFramesReady());
    int32_t frames_in_window = end_frame - start_frame;
    MatrixF feats(frames_in_window, 1);

    for (int32_t f = start_frame; f < end_frame; f++)
      feats(f - start_frame, 0) = GetRawLogPitchFeature(f);

    DeltaOptions delta_opts;
    delta_opts.order = 1;
    delta_opts.window = opts_.delta_window;
    MatrixF delta_feats = ComputeDeltas(delta_opts, feats);
    while (delta_feature_noise_.size() <= static_cast<size_t>(frame)) {
      delta_feature_noise_.push_back(rand_.RandGauss() *
                                     opts_.delta_pitch_noise_stddev);
    }
    // note: delta_feats will have two columns, second contains deltas.
    return (delta_feats(frame - start_frame, 1) + delta_feature_noise_[frame]) *
           opts_.delta_pitch_scale;
  }

  /// Computes and returns the raw log-pitch feature for this frame.
  /// Called from GetFrame().
  float GetRawLogPitchFeature(int32_t frame) const {
    VectorF tmp(kRawFeatureDim);
    src_->GetFrame(frame, &tmp);
    float pitch = tmp(1);
    assert(pitch > 0);
    return Log(pitch);
  }

  /// Computes and returns the mean-subtracted log-pitch feature for this
  /// frame.  Called from GetFrame().
  float GetNormalizedLogPitchFeature(int32_t frame) {
    UpdateNormalizationStats(frame);
    float log_pitch = GetRawLogPitchFeature(frame);
    float avg_log_pitch = normalization_stats_[frame].sum_log_pitch_pov /
                          normalization_stats_[frame].sum_pov;
    float normalized_log_pitch = log_pitch - avg_log_pitch;
    return normalized_log_pitch * opts_.pitch_scale;
  }

  /// Computes the normalization window sizes.
  void GetNormalizationWindow(int32_t t, int32_t src_frames_ready,
                              int32_t *window_begin,
                              int32_t *window_end) const {
    int32_t left_context = opts_.normalization_left_context;
    int32_t right_context = opts_.normalization_right_context;
    *window_begin = std::max(0, t - left_context);
    *window_end = std::min(t + right_context + 1, src_frames_ready);
  }

  struct NormalizationStats {
    int32_t cur_num_frames = -1;     ///< value of src_->NumFramesReady() when
                                     ///< "mean_pitch" was set.
    bool input_finished = false;     ///< true if input data was finished when
                                     ///< "mean_pitch" was computed.
    double sum_pov = 0.0;            ///< sum of pov over relevant range
    double sum_log_pitch_pov = 0.0;  ///< sum of log(pitch) * pov over relevant
                                     ///< range
  };

  /// Makes sure the entry in normalization_stats_ for this frame is up to
  /// date; called from GetNormalizedLogPitchFeature.
  void UpdateNormalizationStats(int32_t frame) {
    assert(frame >= 0);
    if (normalization_stats_.size() <= static_cast<size_t>(frame))
      normalization_stats_.resize(frame + 1);
    int32_t cur_num_frames = src_->NumFramesReady();
    bool input_finished = src_->IsLastFrame(cur_num_frames - 1);

    NormalizationStats &this_stats = normalization_stats_[frame];
    if (this_stats.cur_num_frames == cur_num_frames &&
        this_stats.input_finished == input_finished) {
      // Stats are fully up-to-date.
      return;
    }
    int32_t this_window_begin = 0;
    int32_t this_window_end = 0;
    GetNormalizationWindow(frame, cur_num_frames, &this_window_begin,
                           &this_window_end);

    if (frame > 0) {
      const NormalizationStats &prev_stats = normalization_stats_[frame - 1];
      if (prev_stats.cur_num_frames == cur_num_frames &&
          prev_stats.input_finished == input_finished) {
        // we'll derive this_stats efficiently from prev_stats.
        // Checking that cur_num_frames and input_finished have not changed
        // ensures that the underlying features will not have changed.
        this_stats = prev_stats;
        int32_t prev_window_begin = 0;
        int32_t prev_window_end = 0;
        GetNormalizationWindow(frame - 1, cur_num_frames, &prev_window_begin,
                               &prev_window_end);
        if (this_window_begin != prev_window_begin) {
          assert(this_window_begin == prev_window_begin + 1);
          VectorF tmp(kRawFeatureDim);
          src_->GetFrame(prev_window_begin, &tmp);
          float accurate_pov = NccfToPov(tmp(0));
          float log_pitch = Log(tmp(1));
          this_stats.sum_pov -= accurate_pov;
          this_stats.sum_log_pitch_pov -= accurate_pov * log_pitch;
        }
        if (this_window_end != prev_window_end) {
          assert(this_window_end == prev_window_end + 1);
          VectorF tmp(kRawFeatureDim);
          src_->GetFrame(prev_window_end, &tmp);
          float accurate_pov = NccfToPov(tmp(0));
          float log_pitch = Log(tmp(1));
          this_stats.sum_pov += accurate_pov;
          this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
        }
        return;
      }
    }
    // The way we do it here is not the most efficient way to do it;
    // we'll see if it becomes a problem.  The issue is we have to redo
    // this computation from scratch each time we process a new chunk, which
    // may be a little inefficient if the chunk-size is very small.
    this_stats.cur_num_frames = cur_num_frames;
    this_stats.input_finished = input_finished;
    this_stats.sum_pov = 0.0;
    this_stats.sum_log_pitch_pov = 0.0;
    VectorF tmp(kRawFeatureDim);
    for (int32_t f = this_window_begin; f < this_window_end; f++) {
      src_->GetFrame(f, &tmp);
      float accurate_pov = NccfToPov(tmp(0));
      float log_pitch = Log(tmp(1));
      this_stats.sum_pov += accurate_pov;
      this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
    }
  }

  ProcessPitchOptions opts_;
  OnlinePitchFeatureImpl *src_;  // Not owned.
  int32_t dim_;                  // Output feature dimension.

  /// Per-frame noise added to the delta-pitch feature, drawn lazily in
  /// frame order (kaldi's delta_feature_noise_).
  std::vector<float> delta_feature_noise_;

  /// Deterministic random stream for delta_feature_noise_ (see
  /// ReferenceRandom above).
  ReferenceRandom rand_;

  std::vector<NormalizationStats> normalization_stats_;
};

}  // namespace

class OnlinePitch::Impl {
 public:
  Impl(const PitchExtractionOptions &opts, const ProcessPitchOptions &popts)
      : feature_(opts), process_(popts, &feature_) {}

  OnlinePitchFeatureImpl feature_;
  OnlineProcessPitch process_;
};

OnlinePitch::OnlinePitch(const PitchExtractionOptions &opts,
                         const ProcessPitchOptions &popts)
    : impl_(new Impl(opts, popts)) {}

OnlinePitch::~OnlinePitch() = default;

int32_t OnlinePitch::Dim() const { return impl_->process_.Dim(); }

float OnlinePitch::FrameShiftInSeconds() const {
  return impl_->process_.FrameShiftInSeconds();
}

int32_t OnlinePitch::NumFramesReady() const {
  return impl_->process_.NumFramesReady();
}

bool OnlinePitch::IsLastFrame(int32_t frame) const {
  return impl_->process_.IsLastFrame(frame);
}

std::vector<float> OnlinePitch::GetFrame(int32_t frame) const {
  VectorF feat(Dim());
  impl_->process_.GetFrame(frame, &feat);
  return std::vector<float>(feat.data(), feat.data() + feat.size());
}

void OnlinePitch::AcceptWaveform(float sampling_rate, const float *waveform,
                                 int32_t n) {
  if (sampling_rate != impl_->feature_.opts().samp_freq) {
    throw std::invalid_argument(
        "OnlinePitch: sampling rate does not match the rate the object "
        "was constructed with");
  }
  impl_->feature_.AcceptWaveform(sampling_rate, waveform, n);
}

void OnlinePitch::InputFinished() { impl_->feature_.InputFinished(); }

MatrixF ComputeAndProcessKaldiPitch(const PitchExtractionOptions &pitch_opts,
                                    const ProcessPitchOptions &process_opts,
                                    const float *wave, int32_t n) {
  if (pitch_opts.simulate_first_pass_online &&
      pitch_opts.frames_per_chunk <= 0) {
    throw std::invalid_argument(
        "ComputeAndProcessKaldiPitch: --simulate-first-pass-online does not "
        "make sense unless --frames-per-chunk is set");
  }

  OnlinePitch pitch(pitch_opts, process_opts);

  int32_t cur_rows = 100;
  MatrixF feats(cur_rows, pitch.Dim());

  int32_t cur_offset = 0;
  int32_t cur_frame = 0;
  int32_t samp_per_chunk = pitch_opts.frames_per_chunk * pitch_opts.samp_freq *
                           pitch_opts.frame_shift_ms / 1000.0f;

  // We request the first-pass features as soon as they are available,
  // regardless of whether opts.simulate_first_pass_online == true.  If
  // opts.simulate_first_pass_online == true this should not affect the
  // features generated, but it helps us to test the code in a way that's
  // closer to what online decoding would see.
  while (cur_offset < n) {
    int32_t num_samp = 0;
    if (samp_per_chunk > 0)
      num_samp = std::min(samp_per_chunk, n - cur_offset);
    else  // user left opts.frames_per_chunk at zero.
      num_samp = n;
    pitch.AcceptWaveform(pitch_opts.samp_freq, wave + cur_offset, num_samp);
    cur_offset += num_samp;
    if (cur_offset == n) pitch.InputFinished();

    // Get each frame as soon as it is ready.
    for (; cur_frame < pitch.NumFramesReady(); cur_frame++) {
      if (cur_frame >= cur_rows) {
        cur_rows *= 2;
        MatrixF tmp(cur_rows, pitch.Dim());
        tmp.topRows(cur_frame) = feats.topRows(cur_frame);
        feats.swap(tmp);
      }
      std::vector<float> row = pitch.GetFrame(cur_frame);
      for (int32_t d = 0; d < pitch.Dim(); ++d) feats(cur_frame, d) = row[d];
    }
  }

  if (pitch_opts.simulate_first_pass_online) {
    if (cur_frame == 0) {
      KNF_LOGE("No features output since wave file too short");
      return MatrixF(0, 0);
    }
    return feats.topRows(cur_frame);
  } else {
    // want the "final" features for second pass, so get them again.
    int32_t num_frames = pitch.NumFramesReady();
    MatrixF output(num_frames, pitch.Dim());
    for (int32_t frame = 0; frame < num_frames; frame++) {
      std::vector<float> row = pitch.GetFrame(frame);
      for (int32_t d = 0; d < pitch.Dim(); ++d) output(frame, d) = row[d];
    }
    return output;
  }
}

}  // namespace knf

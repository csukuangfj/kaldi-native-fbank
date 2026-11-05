/**
 * Copyright 2013   Johns Hopkins University (author: Daniel Povey)
 *           2014   Yanqing Sun, Junjie Wang,
 *                  Daniel Povey, Korbinian Riedhammer
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * This file is copied/modified from kaldi/src/feat/online-feature.h
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
 * @file online-feature.h
 * @brief Streaming feature extraction.
 *
 * This file is copied/modified from @kfile{src/feat/online-feature.h}. It
 * provides a templated online feature extractor, OnlineGenericBaseFeature,
 * that wraps a per-frame computer such as FbankComputer or MfccComputer and
 * manages the waveform buffering for streaming use.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_ONLINE_FEATURE_H_
#define KALDI_NATIVE_FBANK_CSRC_ONLINE_FEATURE_H_

#include <cstdint>
#include <deque>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-mfcc.h"
#include "kaldi-native-fbank/csrc/feature-raw-audio-samples.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/whisper-feature.h"

namespace knf {

/// This class serves as a storage for feature vectors with an option to limit
/// the memory usage by removing old elements. The deleted frames indices are
/// "remembered" so that regardless of the MAX_ITEMS setting, the user always
/// provides the indices as if no deletion was being performed.
/// This is useful when processing very long recordings which would otherwise
/// cause the memory to eventually blow up when the features are not being
/// removed.
///
/// @par Used by:
/// OnlineGenericBaseFeature stores the computed frames in a RecyclingVector.
class RecyclingVector {
 public:
  /// By default it does not remove any elements.
  explicit RecyclingVector(int32_t items_to_hold = -1);

  /// Destructor.
  ~RecyclingVector() = default;

  /// Copy construction is not allowed.
  RecyclingVector(const RecyclingVector &) = delete;

  /// Assignment is not allowed.
  RecyclingVector &operator=(const RecyclingVector &) = delete;

  /// Return a pointer to the feature vector at the given index.
  ///
  /// The pointer is owned by RecyclingVector; users should not free it.
  const float *At(int32_t index) const;

  /// Append one feature vector to the end. If items_to_hold_ is exceeded,
  /// the oldest vector is deleted.
  void PushBack(std::vector<float> item);

  /// This method returns the size as if no "recycling" had happened,
  /// i.e. equivalent to the number of times the PushBack method has been
  /// called.
  int32_t Size() const;

  /// Discard the first n frames.
  void Pop(int32_t n);

 private:
  std::deque<std::vector<float>> items_;  ///< The stored feature vectors.
  int32_t items_to_hold_;  ///< Max number of items to keep; -1 means all.
  int32_t first_available_index_ = 0;  ///< Index of the oldest item still kept.
};

/// This is a templated class for online feature extraction;
/// it's templated on a class like MfccComputer or FbankComputer
/// that does the basic feature extraction.
///
/// @par Used by:
/// The aliases OnlineFbank, OnlineMfcc, OnlineWhisperFbank, and
/// OnlineRawAudioSamples at the bottom of this file.
///
/// @see FbankComputer, MfccComputer, WhisperFeatureComputer,
/// RawAudioSamplesComputer, RecyclingVector
template <class C>
class OnlineGenericBaseFeature {
 public:
  /// Construct from the options class of the wrapped computer C.
  explicit OnlineGenericBaseFeature(const typename C::Options &opts);

  /// Return the dimension of the output features.
  int32_t Dim() const { return computer_.Dim(); }

  /// Return the frame shift in seconds.
  float FrameShiftInSeconds() const {
    return computer_.GetFrameOptions().frame_shift_ms / 1000.0f;
  }

  /// Return the number of frames that are already computed.
  int32_t NumFramesReady() const { return features_.Size(); }

  /// Return true if `frame` is the last frame.
  ///
  /// Note: IsLastFrame() will only ever return true if you have called
  /// InputFinished() (and this frame is the last frame).
  bool IsLastFrame(int32_t frame) const {
    return input_finished_ && frame == NumFramesReady() - 1;
  }

  /// Return a pointer to the `frame`-th feature vector.
  ///
  /// The pointer is owned by this object; users should not free it.
  const float *GetFrame(int32_t frame) const { return features_.At(frame); }

  /**
   * This would be called from the application, when you get
   * more wave data.  Note: the sampling_rate is only provided so
   * the code can assert that it matches the sampling rate
   * expected in the options.
   *
   * @param sampling_rate The sampling_rate of the input waveform
   * @param waveform Pointer to a 1-D array of size n
   * @param n Number of entries in waveform
   */
  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);

  /// InputFinished() tells the class you won't be providing any
  /// more waveform.  This will help flush out the last frame or two
  /// of features, in the case where snip-edges == false; it also
  /// affects the return value of IsLastFrame().
  void InputFinished();

  /// Discard the first n frames.
  void Pop(int32_t n) { features_.Pop(n); }

 private:
  /// This function computes any additional feature frames that it is possible
  /// to compute from 'waveform_remainder_', which at this point may contain
  /// more than just a remainder-sized quantity (because AcceptWaveform()
  /// appends to waveform_remainder_ before calling this function).  It adds
  /// these feature frames to features_, and shifts off any now-unneeded
  /// samples of input from waveform_remainder_ while incrementing
  /// waveform_offset_ by the same amount.
  void ComputeFeatures();

  C computer_;  ///< Does the MFCC or fbank computation.

  FeatureWindowFunction window_function_;  ///< The windowing function.

  /// The features we have already computed.
  RecyclingVector features_;

  bool input_finished_ =
      false;  ///< True if the user has called "InputFinished()".

  /// Number of samples of waveform we have already discarded, i.e. that were
  /// prior to 'waveform_remainder_'.
  int64_t waveform_offset_ = 0;

  /// A short piece of waveform that we may need to keep after extracting all
  /// the whole frames we can (whatever length of feature will be required for
  /// the next phase of computation). It is a 1-D array.
  std::vector<float> waveform_remainder_;
};

/// Streaming raw audio samples, framed without any processing.
using OnlineRawAudioSamples = OnlineGenericBaseFeature<RawAudioSamplesComputer>;

/// Streaming filter-bank features; see FbankComputer.
using OnlineFbank = OnlineGenericBaseFeature<FbankComputer>;

/// Streaming MFCC features; see MfccComputer.
using OnlineMfcc = OnlineGenericBaseFeature<MfccComputer>;

/// Streaming Whisper log-mel features; see WhisperFeatureComputer.
using OnlineWhisperFbank = OnlineGenericBaseFeature<WhisperFeatureComputer>;

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_ONLINE_FEATURE_H_

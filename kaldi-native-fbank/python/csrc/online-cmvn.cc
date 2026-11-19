/**
 * Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/python/csrc/online-cmvn.h"

#include <pybind11/eigen.h>

#include <sstream>
#include <string>

#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/online-cmvn.h"

namespace knf {

void PybindOnlineCmvn(py::module &m) {  // NOLINT
  auto cls_Options = py::class_<OnlineCmvnOptions>(
      m, "OnlineCmvnOptions",
      R"doc(Options for :class:`OnlineCmvn`, mirroring kaldi's ``OnlineCmvnOptions``
(see ``feat/online-feature.h``).

.. seealso:: :class:`OnlineCmvn`
   For how the options affect the normalization.)doc");
  cls_Options.attr("__module__") = "kaldi_native_fbank";
  cls_Options
      .def(py::init([](int32_t cmn_window, int32_t speaker_frames,
                       int32_t global_frames, bool normalize_mean,
                       bool normalize_variance, int32_t modulus,
                       int32_t ring_buffer_size, const std::string &skip_dims) {
             OnlineCmvnOptions opts;
             opts.cmn_window = cmn_window;
             opts.speaker_frames = speaker_frames;
             opts.global_frames = global_frames;
             opts.normalize_mean = normalize_mean;
             opts.normalize_variance = normalize_variance;
             opts.modulus = modulus;
             opts.ring_buffer_size = ring_buffer_size;
             opts.skip_dims = skip_dims;
             return opts;
           }),
           py::arg("cmn_window") = 600, py::arg("speaker_frames") = 600,
           py::arg("global_frames") = 200, py::arg("normalize_mean") = true,
           py::arg("normalize_variance") = false, py::arg("modulus") = 20,
           py::arg("ring_buffer_size") = 20, py::arg("skip_dims") = "",
           R"doc(Construct with the given options.  The defaults match kaldi.

:param cmn_window: Number of frames of sliding context for cepstral mean
   normalization.
:param speaker_frames: Number of frames of previous utterance(s) from this
   speaker to use.  Must be ``<= cmn_window``.
:param global_frames: Number of frames of global-average stats to use at the
   beginning.  Must be ``<= speaker_frames``.
:param normalize_mean: If ``True`` (default), subtract the running mean.
:param normalize_variance: If ``True``, also divide by the running standard
   deviation.  Requires ``normalize_mean=True``.
:param modulus: Kept for kaldi configuration compatibility; results are not
   affected (stats are updated incrementally).
:param ring_buffer_size: Kept for kaldi configuration compatibility (see
   ``modulus``).
:param skip_dims: Colon-separated list of dimensions to skip normalization
   of, e.g. ``"13:14:15"``.)doc")
      .def_readwrite("cmn_window", &OnlineCmvnOptions::cmn_window,
                     R"doc(Number of frames of sliding context for cepstral
mean normalization.)doc")
      .def_readwrite("speaker_frames", &OnlineCmvnOptions::speaker_frames,
                     R"doc(Number of frames of previous utterance(s) from this
speaker to use in cepstral mean normalization.)doc")
      .def_readwrite("global_frames", &OnlineCmvnOptions::global_frames,
                     R"doc(Number of frames of global-average cepstral mean
normalization stats to use for the first utterance of a speaker.)doc")
      .def_readwrite("normalize_mean", &OnlineCmvnOptions::normalize_mean,
                     R"doc(If ``True``, do mean normalization (you cannot
normalize the variance but not the mean).)doc")
      .def_readwrite("normalize_variance",
                     &OnlineCmvnOptions::normalize_variance,
                     R"doc(If ``True``, do cepstral variance normalization in
addition to cepstral mean normalization.)doc")
      .def_readwrite("modulus", &OnlineCmvnOptions::modulus,
                     R"doc(Kept for kaldi configuration compatibility; results
are not affected (stats are updated incrementally).)doc")
      .def_readwrite("ring_buffer_size",
                     &OnlineCmvnOptions::ring_buffer_size,
                     R"doc(Kept for kaldi configuration compatibility (see
``modulus``).)doc")
      .def_readwrite("skip_dims", &OnlineCmvnOptions::skip_dims,
                     R"doc(Colon-separated list of dimensions to skip
normalization of, e.g. ``"13:14:15"``.)doc")
      .def("__str__", &OnlineCmvnOptions::ToString,
           "Return the options as a string.");

  auto cls_State = py::class_<OnlineCmvnState>(
      m, "OnlineCmvnState",
      R"doc(State of CMVN adaptation between utterances (not the
within-utterance computation state).

Stats matrices follow kaldi's layout: 2 x (dim+1) as
``[ sum-stats count ; sum-squared-stats 0 ]``.

Pass the state of one utterance to a new :class:`OnlineCmvn` to continue with
the next utterance of the same speaker.

.. seealso:: :meth:`OnlineCmvn.get_state`)doc");
  cls_State.attr("__module__") = "kaldi_native_fbank";
  cls_State
      .def(py::init([](const MatrixD &global_cmvn_stats) {
             OnlineCmvnState state;
             if (global_cmvn_stats.size() != 0)
               state.global_cmvn_stats = global_cmvn_stats;
             return state;
           }),
           py::arg("global_cmvn_stats") = MatrixD(),
           R"doc(Construct with global CMVN stats (required by
:class:`OnlineCmvn`), e.g. loaded from kaldi's ``global_cmvn.stats``.

:param global_cmvn_stats: Global stats of shape ``[2, dim + 1]``.)doc")
      .def_property(
          "speaker_cmvn_stats",
          [](const OnlineCmvnState &self) -> MatrixD {
            return self.speaker_cmvn_stats;
          },
          [](OnlineCmvnState &self, const MatrixD &stats) {
            self.speaker_cmvn_stats = stats;
          },
          R"doc(Total CMVN stats for this speaker up till now
``[2, dim + 1]``; may be empty at the start.)doc")
      .def_property(
          "global_cmvn_stats",
          [](const OnlineCmvnState &self) -> MatrixD {
            return self.global_cmvn_stats;
          },
          [](OnlineCmvnState &self, const MatrixD &stats) {
            self.global_cmvn_stats = stats;
          },
          R"doc(Global CMVN stats ``[2, dim + 1]`` (required).)doc")
      .def_property(
          "frozen_state",
          [](const OnlineCmvnState &self) -> MatrixD {
            return self.frozen_state;
          },
          [](OnlineCmvnState &self, const MatrixD &stats) {
            self.frozen_state = stats;
          },
          R"doc(If non-empty ``[2, dim + 1]``, the "frozen" normalization used
since :meth:`OnlineCmvn.freeze`.)doc");

  auto cls_OnlineCmvn = py::class_<OnlineCmvn>(
      m, "OnlineCmvn",
      R"doc(Online (running) cepstral mean and optional variance
normalization, a port of kaldi's ``OnlineCmvn`` (``feat/online-feature.h``).

Unlike offline CMVN this looks only at the past: in the middle of a long
utterance it accumulates stats from the previous ``cmn_window`` frames
(default 600) and uses them to normalize the current frame.  At the beginning
of an utterance the stats are augmented with up to ``speaker_frames`` frames of
speaker stats and up to ``global_frames`` frames of global stats.  Frames are
normalized using stats that include the frame itself, like kaldi.

This class is push-style: feed raw feature frames one by one and get the
normalized frames back.

Example:

    opts = knf.OnlineCmvnOptions()
    state = knf.OnlineCmvnState(global_cmvn_stats)  # [2, dim+1]
    cmvn = knf.OnlineCmvn(opts, state)
    normalized = cmvn.accept_features(raw_feats)  # [T, dim] float32
    state = cmvn.get_state()  # pass to the next utterance of this speaker

.. seealso:: :class:`OnlineCmvnOptions`, :class:`OnlineCmvnState`)doc");
  cls_OnlineCmvn.attr("__module__") = "kaldi_native_fbank";
  cls_OnlineCmvn
      .def(py::init<const OnlineCmvnOptions &, const OnlineCmvnState &>(),
           py::arg("opts"), py::arg("state"),
           R"doc(Construct with options and adaptation state.  The state
carries the required global CMVN stats, and optionally speaker stats and a
frozen state from a previous utterance.)doc")
      .def_property_readonly("dim", &OnlineCmvn::Dim,
                             "Feature dimension.")
      .def_property_readonly("num_frames_accepted",
                             &OnlineCmvn::NumFramesAccepted,
                             R"doc(Number of frames accepted so far.)doc")
      .def(
          "accept_feature",
          [](OnlineCmvn &self, const VectorF &feat) -> VectorF {
            return self.AcceptFeature(feat);
          },
          py::arg("feat"),
          R"doc(Push one raw feature frame; returns the normalized frame.

:param feat: Raw feature frame of shape ``[dim]`` (float32).
:return: Normalized frame of shape ``[dim]`` (float32).)doc")
      .def(
          "accept_features",
          [](OnlineCmvn &self, const MatrixF &feats) -> MatrixF {
            return self.AcceptFeatures(feats);
          },
          py::arg("feats"),
          R"doc(Push a batch of raw feature frames; returns the normalized
frames.

:param feats: Raw features of shape ``[T, dim]`` (float32, rows are frames).
:return: Normalized features of shape ``[T, dim]`` (float32).)doc")
      .def("freeze", &OnlineCmvn::Freeze,
           R"doc(Freeze the normalization at the current frame: from now on
frames are normalized with the current stats and the stats stop updating
(kaldi's ``OnlineCmvn::Freeze``).)doc")
      .def("get_state", &OnlineCmvn::GetState,
           R"doc(Return the adaptation state (speaker stats including all
accepted frames, plus any frozen state), like kaldi's
``OnlineCmvn::GetState``.  Pass it to a new :class:`OnlineCmvn` to continue
with the next utterance of the same speaker.)doc")
      .def_static(
          "smooth_online_cmvn_stats",
          [](const MatrixD &speaker_stats, const MatrixD &global_stats,
             const OnlineCmvnOptions &opts, const MatrixD &stats) -> MatrixD {
            MatrixD out = stats;
            OnlineCmvn::SmoothOnlineCmvnStats(speaker_stats, global_stats,
                                              opts, &out);
            return out;
          },
          py::arg("speaker_stats"), py::arg("global_stats"), py::arg("opts"),
          py::arg("stats"),
          R"doc(Smooth window stats with speaker and global stats; a port of
kaldi's static ``OnlineCmvn::SmoothOnlineCmvnStats``.

:param speaker_stats: Speaker stats ``[2, dim + 1]`` (may be empty).
:param global_stats: Global stats ``[2, dim + 1]``.
:param opts: The options in use.
:param stats: Window stats ``[2, dim + 1]`` to smooth.
:return: Smoothed stats ``[2, dim + 1]``.)doc");
}

}  // namespace knf

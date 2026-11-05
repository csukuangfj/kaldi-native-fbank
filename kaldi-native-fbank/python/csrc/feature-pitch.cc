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

#include "kaldi-native-fbank/python/csrc/feature-pitch.h"

#include <pybind11/eigen.h>

#include <cstdint>
#include <vector>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/feature-pitch.h"

namespace pybind11 {
class gil_scoped_release;
}  // namespace pybind11

namespace knf {

void PybindFeaturePitch(py::module &m) {  // NOLINT
  auto cls_PitchExtractionOptions = py::class_<PitchExtractionOptions>(
      m, "PitchExtractionOptions",
      R"doc(Options for the raw pitch extraction stage.

Mirrors the flags of kaldi's `compute-kaldi-pitch-feats
<https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-kaldi-pitch-feats.cc>`_
and kaldi's `PitchExtractionOptions
<https://github.com/kaldi-asr/kaldi/blob/master/src/feat/pitch-functions.h>`_.

.. seealso:: :class:`ProcessPitchOptions`, :class:`OnlinePitch`,
   :func:`compute_and_process_kaldi_pitch`)doc");
  cls_PitchExtractionOptions.attr("__module__") = "kaldi_native_fbank";
  cls_PitchExtractionOptions
      .def(py::init(
               [](float samp_freq, float frame_shift_ms, float frame_length_ms,
                  float preemph_coeff, float min_f0, float max_f0,
                  float soft_min_f0, float penalty_factor, float lowpass_cutoff,
                  float resample_freq, float delta_pitch, float nccf_ballast,
                  int32_t lowpass_filter_width, int32_t upsample_filter_width,
                  int32_t max_frames_latency, int32_t frames_per_chunk,
                  bool simulate_first_pass_online, int32_t recompute_frame,
                  bool nccf_ballast_online, bool snip_edges) {
                 PitchExtractionOptions opts;
                 opts.samp_freq = samp_freq;
                 opts.frame_shift_ms = frame_shift_ms;
                 opts.frame_length_ms = frame_length_ms;
                 opts.preemph_coeff = preemph_coeff;
                 opts.min_f0 = min_f0;
                 opts.max_f0 = max_f0;
                 opts.soft_min_f0 = soft_min_f0;
                 opts.penalty_factor = penalty_factor;
                 opts.lowpass_cutoff = lowpass_cutoff;
                 opts.resample_freq = resample_freq;
                 opts.delta_pitch = delta_pitch;
                 opts.nccf_ballast = nccf_ballast;
                 opts.lowpass_filter_width = lowpass_filter_width;
                 opts.upsample_filter_width = upsample_filter_width;
                 opts.max_frames_latency = max_frames_latency;
                 opts.frames_per_chunk = frames_per_chunk;
                 opts.simulate_first_pass_online = simulate_first_pass_online;
                 opts.recompute_frame = recompute_frame;
                 opts.nccf_ballast_online = nccf_ballast_online;
                 opts.snip_edges = snip_edges;
                 return opts;
               }),
           py::arg_v("samp_freq", 16000.0f, "16000"),
           py::arg_v("frame_shift_ms", 10.0f, "10.0"),
           py::arg_v("frame_length_ms", 25.0f, "25.0"),
           py::arg_v("preemph_coeff", 0.0f, "0.0"),
           py::arg_v("min_f0", 50.0f, "50"), py::arg_v("max_f0", 400.0f, "400"),
           py::arg_v("soft_min_f0", 10.0f, "10.0"),
           py::arg_v("penalty_factor", 0.1f, "0.1"),
           py::arg_v("lowpass_cutoff", 1000.0f, "1000"),
           py::arg_v("resample_freq", 4000.0f, "4000"),
           py::arg_v("delta_pitch", 0.005f, "0.005"),
           py::arg_v("nccf_ballast", 7000.0f, "7000"),
           py::arg_v("lowpass_filter_width", 1, "1"),
           py::arg_v("upsample_filter_width", 5, "5"),
           py::arg_v("max_frames_latency", 0, "0"),
           py::arg_v("frames_per_chunk", 0, "0"),
           py::arg_v("simulate_first_pass_online", false, "False"),
           py::arg_v("recompute_frame", 500, "500"),
           py::arg_v("nccf_ballast_online", false, "False"),
           py::arg_v("snip_edges", true, "True"),
           R"doc(Construct with the given options.

All fields are optional keyword arguments; the defaults are shown in the
signature (they are kaldi's defaults).

:param samp_freq: Waveform sample frequency in Hz (kaldi ``--sample-frequency``).
:param frame_shift_ms: Frame shift in milliseconds (kaldi ``--frame-shift``).
:param frame_length_ms: Frame length in milliseconds (kaldi ``--frame-length``).
:param preemph_coeff: Preemphasis coefficient; deprecated in kaldi
   (kaldi ``--preemphasis-coefficient``).
:param min_f0: Min F0 to search for, in Hz (kaldi ``--min-f0``).
:param max_f0: Max F0 to search for, in Hz (kaldi ``--max-f0``).
:param soft_min_f0: Minimum f0 applied in a soft way; must not exceed
   ``min_f0`` (kaldi ``--soft-min-f0``).
:param penalty_factor: Cost factor for F0 change (kaldi ``--penalty-factor``).
:param lowpass_cutoff: Low-pass filter cutoff in Hz (kaldi ``--lowpass-cutoff``).
:param resample_freq: Frequency the signal is downsampled to; must be more
   than twice ``lowpass_cutoff`` (kaldi ``--resample-frequency``).
:param delta_pitch: The pitch tolerance in pruning lags (kaldi ``--delta-pitch``).
:param nccf_ballast: Reduces NCCF for quiet frames when increased
   (kaldi ``--nccf-ballast``).
:param lowpass_filter_width: Low-pass filter width (kaldi ``--lowpass-filter-width``).
:param upsample_filter_width: NCCF upsampling filter width
   (kaldi ``--upsample-filter-width``).
:param max_frames_latency: Maximum frames of latency allowed for pitch
   tracking (kaldi ``--max-frames-latency``).
:param frames_per_chunk: Chunk size for :func:`compute_and_process_kaldi_pitch`
   (kaldi ``--frames-per-chunk``).
:param simulate_first_pass_online: Output first-pass online features in
   :func:`compute_and_process_kaldi_pitch`
   (kaldi ``--simulate-first-pass-online``).
:param recompute_frame: Frame at which online NCCF recomputation happens
   (kaldi ``--recompute-frame``).
:param nccf_ballast_online: Debug flag for how the NCCF ballast is computed
   (kaldi ``--nccf-ballast-online``).
:param snip_edges: If false, incomplete frames near the ending edge are not
   snipped (kaldi ``--snip-edges``).  Note: unlike other feature types,
   kaldi's pitch extractor defaults this to true.)doc")
      .def_readwrite("samp_freq", &PitchExtractionOptions::samp_freq,
                     R"doc(Waveform sample frequency in Hz.)doc")
      .def_readwrite("frame_shift_ms", &PitchExtractionOptions::frame_shift_ms,
                     R"doc(Frame shift in milliseconds.)doc")
      .def_readwrite("frame_length_ms",
                     &PitchExtractionOptions::frame_length_ms,
                     R"doc(Frame length in milliseconds.)doc")
      .def_readwrite("preemph_coeff", &PitchExtractionOptions::preemph_coeff,
                     R"doc(Preemphasis coefficient (deprecated in kaldi).)doc")
      .def_readwrite("min_f0", &PitchExtractionOptions::min_f0,
                     R"doc(Min F0 to search for, in Hz.)doc")
      .def_readwrite("max_f0", &PitchExtractionOptions::max_f0,
                     R"doc(Max F0 to search for, in Hz.)doc")
      .def_readwrite("soft_min_f0", &PitchExtractionOptions::soft_min_f0,
                     R"doc(Minimum f0 applied in a soft way.)doc")
      .def_readwrite("penalty_factor", &PitchExtractionOptions::penalty_factor,
                     R"doc(Cost factor for F0 change.)doc")
      .def_readwrite("lowpass_cutoff", &PitchExtractionOptions::lowpass_cutoff,
                     R"doc(Low-pass filter cutoff in Hz.)doc")
      .def_readwrite("resample_freq", &PitchExtractionOptions::resample_freq,
                     R"doc(Frequency the signal is downsampled to.)doc")
      .def_readwrite("delta_pitch", &PitchExtractionOptions::delta_pitch,
                     R"doc(The pitch tolerance in pruning lags.)doc")
      .def_readwrite("nccf_ballast", &PitchExtractionOptions::nccf_ballast,
                     R"doc(Reduces NCCF for quiet frames when increased.)doc")
      .def_readwrite("lowpass_filter_width",
                     &PitchExtractionOptions::lowpass_filter_width,
                     R"doc(Low-pass filter width.)doc")
      .def_readwrite("upsample_filter_width",
                     &PitchExtractionOptions::upsample_filter_width,
                     R"doc(NCCF upsampling filter width.)doc")
      .def_readwrite(
          "max_frames_latency", &PitchExtractionOptions::max_frames_latency,
          R"doc(Maximum frames of latency allowed for pitch tracking.)doc")
      .def_readwrite(
          "frames_per_chunk", &PitchExtractionOptions::frames_per_chunk,
          R"doc(Chunk size for :func:`compute_and_process_kaldi_pitch`.)doc")
      .def_readwrite(
          "simulate_first_pass_online",
          &PitchExtractionOptions::simulate_first_pass_online,
          R"doc(Output first-pass online features (batch mode only).)doc")
      .def_readwrite(
          "recompute_frame", &PitchExtractionOptions::recompute_frame,
          R"doc(Frame at which online NCCF recomputation happens.)doc")
      .def_readwrite(
          "nccf_ballast_online", &PitchExtractionOptions::nccf_ballast_online,
          R"doc(Debug flag for how the NCCF ballast is computed.)doc")
      .def_readwrite(
          "snip_edges", &PitchExtractionOptions::snip_edges,
          R"doc(If false, incomplete frames near the ending edge are not snipped.)doc")
      .def_property_readonly(
          "nccf_window_size", &PitchExtractionOptions::NccfWindowSize,
          R"doc(The NCCF window size in samples after resampling.)doc")
      .def_property_readonly(
          "nccf_window_shift", &PitchExtractionOptions::NccfWindowShift,
          R"doc(The NCCF window shift in samples after resampling.)doc");

  auto cls_ProcessPitchOptions = py::class_<ProcessPitchOptions>(
      m, "ProcessPitchOptions",
      R"doc(Options for the pitch post-processing stage.

Determines which features are produced from the raw (NCCF, pitch) pair.
With the defaults each frame is the 3-dimensional
(POV-feature, normalized-log-pitch, delta-log-pitch) output of kaldi's
`compute-and-process-kaldi-pitch-feats
<https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc>`_;
see kaldi's `ProcessPitchOptions
<https://github.com/kaldi-asr/kaldi/blob/master/src/feat/pitch-functions.h>`_.

.. seealso:: :class:`PitchExtractionOptions`, :class:`OnlinePitch`)doc");
  cls_ProcessPitchOptions.attr("__module__") = "kaldi_native_fbank";
  cls_ProcessPitchOptions
      .def(py::init([](float pitch_scale, float pov_scale, float pov_offset,
                       float delta_pitch_scale, float delta_pitch_noise_stddev,
                       int32_t normalization_left_context,
                       int32_t normalization_right_context,
                       int32_t delta_window, int32_t delay,
                       bool add_pov_feature, bool add_normalized_log_pitch,
                       bool add_delta_pitch, bool add_raw_log_pitch) {
             ProcessPitchOptions opts;
             opts.pitch_scale = pitch_scale;
             opts.pov_scale = pov_scale;
             opts.pov_offset = pov_offset;
             opts.delta_pitch_scale = delta_pitch_scale;
             opts.delta_pitch_noise_stddev = delta_pitch_noise_stddev;
             opts.normalization_left_context = normalization_left_context;
             opts.normalization_right_context = normalization_right_context;
             opts.delta_window = delta_window;
             opts.delay = delay;
             opts.add_pov_feature = add_pov_feature;
             opts.add_normalized_log_pitch = add_normalized_log_pitch;
             opts.add_delta_pitch = add_delta_pitch;
             opts.add_raw_log_pitch = add_raw_log_pitch;
             return opts;
           }),
           py::arg_v("pitch_scale", 2.0f, "2.0"),
           py::arg_v("pov_scale", 2.0f, "2.0"),
           py::arg_v("pov_offset", 0.0f, "0.0"),
           py::arg_v("delta_pitch_scale", 10.0f, "10.0"),
           py::arg_v("delta_pitch_noise_stddev", 0.005f, "0.005"),
           py::arg_v("normalization_left_context", 75, "75"),
           py::arg_v("normalization_right_context", 75, "75"),
           py::arg_v("delta_window", 2, "2"), py::arg_v("delay", 0, "0"),
           py::arg_v("add_pov_feature", true, "True"),
           py::arg_v("add_normalized_log_pitch", true, "True"),
           py::arg_v("add_delta_pitch", true, "True"),
           py::arg_v("add_raw_log_pitch", false, "False"),
           R"doc(Construct with the given options.

All fields are optional keyword arguments; the defaults are shown in the
signature (they are kaldi's defaults).

:param pitch_scale: Scaling of the normalized log-pitch feature
   (kaldi ``--pitch-scale``).
:param pov_scale: Scaling of the POV feature (kaldi ``--pov-scale``).
:param pov_offset: Offset added to the POV feature (kaldi ``--pov-offset``).
:param delta_pitch_scale: Scaling of the delta log-pitch feature
   (kaldi ``--delta-pitch-scale``).
:param delta_pitch_noise_stddev: Stddev of the noise added to the delta
   log-pitch (kaldi ``--delta-pitch-noise-stddev``).
:param normalization_left_context: Left context (frames) for log-pitch
   normalization (kaldi ``--normalization-left-context``).
:param normalization_right_context: Right context (frames) for log-pitch
   normalization (kaldi ``--normalization-right-context``).
:param delta_window: Frames on each side used for the delta window
   (kaldi ``--delta-window``).
:param delay: Frames by which the pitch information is delayed
   (kaldi ``--delay``).
:param add_pov_feature: Add the warped NCCF to the output (1 dim).
:param add_normalized_log_pitch: Add the mean-subtracted log-pitch (1 dim).
:param add_delta_pitch: Add the time derivative of log-pitch (1 dim).
:param add_raw_log_pitch: Add log(pitch) (1 dim).)doc")
      .def_readwrite("pitch_scale", &ProcessPitchOptions::pitch_scale,
                     R"doc(Scaling of the normalized log-pitch feature.)doc")
      .def_readwrite("pov_scale", &ProcessPitchOptions::pov_scale,
                     R"doc(Scaling of the POV feature.)doc")
      .def_readwrite("pov_offset", &ProcessPitchOptions::pov_offset,
                     R"doc(Offset added to the POV feature.)doc")
      .def_readwrite("delta_pitch_scale",
                     &ProcessPitchOptions::delta_pitch_scale,
                     R"doc(Scaling of the delta log-pitch feature.)doc")
      .def_readwrite("delta_pitch_noise_stddev",
                     &ProcessPitchOptions::delta_pitch_noise_stddev,
                     R"doc(Stddev of the noise added to the delta log-pitch.

The noise is drawn from a deterministic stream, so the features are
reproducible on every platform.)doc")
      .def_readwrite(
          "normalization_left_context",
          &ProcessPitchOptions::normalization_left_context,
          R"doc(Left context (frames) for log-pitch normalization.)doc")
      .def_readwrite(
          "normalization_right_context",
          &ProcessPitchOptions::normalization_right_context,
          R"doc(Right context (frames) for log-pitch normalization.)doc")
      .def_readwrite("delta_window", &ProcessPitchOptions::delta_window,
                     R"doc(Frames on each side used for the delta window.)doc")
      .def_readwrite(
          "delay", &ProcessPitchOptions::delay,
          R"doc(Frames by which the pitch information is delayed.)doc")
      .def_readwrite("add_pov_feature", &ProcessPitchOptions::add_pov_feature,
                     R"doc(Add the warped NCCF to the output (1 dim).)doc")
      .def_readwrite("add_normalized_log_pitch",
                     &ProcessPitchOptions::add_normalized_log_pitch,
                     R"doc(Add the mean-subtracted log-pitch (1 dim).)doc")
      .def_readwrite("add_delta_pitch", &ProcessPitchOptions::add_delta_pitch,
                     R"doc(Add the time derivative of log-pitch (1 dim).)doc")
      .def_readwrite("add_raw_log_pitch",
                     &ProcessPitchOptions::add_raw_log_pitch,
                     R"doc(Add log(pitch) (1 dim).)doc")
      .def_property_readonly("dim", &ProcessPitchOptions::Dim,
                             R"doc(The output feature dimension.)doc");

  auto cls_OnlinePitch = py::class_<OnlinePitch>(
      m, "OnlinePitch",
      R"doc(Streaming pitch extraction with post-processing.

This is the streaming interface of kaldi's
`compute-and-process-kaldi-pitch-feats
<https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc>`_
(kaldi's ``OnlinePitchFeature`` + ``OnlineProcessPitch``): it consumes
waveform chunks and produces the processed pitch features — by default the
3-dimensional (POV, normalized-log-pitch, delta-pitch).

Usage mirrors :class:`kaldi_native_fbank.OnlineMfcc`: feed the waveform with
:meth:`accept_waveform`, call :meth:`input_finished` when no more data is
coming, then read frames with :meth:`get_frame`.

.. seealso:: :class:`PitchExtractionOptions`, :class:`ProcessPitchOptions`,
   :func:`compute_and_process_kaldi_pitch`)doc");
  cls_OnlinePitch.attr("__module__") = "kaldi_native_fbank";
  cls_OnlinePitch
      .def(
          py::init<const PitchExtractionOptions &,
                   const ProcessPitchOptions &>(),
          py::arg("opts"),
          py::arg_v("popts", ProcessPitchOptions{}, "ProcessPitchOptions()"),
          R"doc(Construct with the given extraction and post-processing options.

:param opts: Raw pitch extraction options; see :class:`PitchExtractionOptions`.
:param popts: Post-processing options; see :class:`ProcessPitchOptions`.
   At least one of the ``add_*`` features must be enabled.)doc")
      .def_property_readonly(
          "dim", &OnlinePitch::Dim,
          R"doc(The dimension of the output features (3 by default).)doc")
      .def_property_readonly("frame_shift_in_seconds",
                             &OnlinePitch::FrameShiftInSeconds,
                             R"doc(The frame shift in seconds.)doc")
      .def_property_readonly(
          "num_frames_ready", &OnlinePitch::NumFramesReady,
          R"doc(The number of frames that have already been computed.)doc")
      .def("is_last_frame", &OnlinePitch::IsLastFrame, py::arg("frame"),
           R"doc(Return true if ``frame`` is the last frame.

Note: this can only return true after :meth:`input_finished` has been
called.

:param frame: Frame index.
:return: True if the frame is the last one.)doc")
      .def(
          "get_frame",
          [](const OnlinePitch &self, int32_t frame) {
            std::vector<float> v = self.GetFrame(frame);
            py::array_t<float> ans(static_cast<py::ssize_t>(v.size()));
            auto buf = ans.template mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); ++i) buf(i) = v[i];
            return ans;
          },
          py::arg("frame"),
          R"doc(Return the ``frame``-th feature vector.

:param frame: Frame index, must be less than ``num_frames_ready``.
:return: np.ndarray of shape ``(dim,)`` and dtype float32.)doc")
      .def(
          "accept_waveform",
          [](OnlinePitch &self, float sampling_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sampling_rate, waveform.data(),
                                waveform.size());
          },
          py::arg("sampling_rate"), py::arg("waveform"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Add more waveform samples.

:param sampling_rate: The sampling rate of ``waveform``; must match
   ``opts.samp_freq``.
:param waveform: 1-D float32 samples (kaldi semantics: raw WAV magnitudes,
   not divided by 32768).)doc")
      .def("input_finished", &OnlinePitch::InputFinished,
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Tell the extractor that no more waveform will be added.

This flushes out the last frames and fixes the features of all frames.)doc");

  m.def(
      "compute_and_process_kaldi_pitch",
      [](const PitchExtractionOptions &pitch_opts,
         const ProcessPitchOptions &process_opts,
         const std::vector<float> &wave) -> MatrixF {
        return ComputeAndProcessKaldiPitch(pitch_opts, process_opts,
                                           wave.data(), wave.size());
      },
      py::arg("pitch_opts"), py::arg("process_opts"), py::arg("wave"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Compute processed pitch features for a whole utterance.

This is kaldi's
`compute-and-process-kaldi-pitch-feats
<https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc>`_
(same arithmetic): the raw (NCCF, pitch) extraction of
`Ghahremani et al., ICASSP 2014
<https://www.danielpovey.com/files/icassp2014_pitch.pdf>`_ followed by
post-processing into the configured features (by default
POV, normalized-log-pitch, delta-pitch).  The output equals the kaldi
binary's output value-by-value for the same input and options.

It is sensitive to the variables in ``pitch_opts`` that relate to online
processing, i.e. ``max_frames_latency``, ``frames_per_chunk``,
``simulate_first_pass_online``, ``recompute_frame``.

The noise added to the delta-pitch feature (``process_opts.delta_pitch_noise_stddev``)
is drawn from a deterministic stream that reproduces the reference kaldi
realization, so the result is identical on every platform.

:param pitch_opts: Raw pitch extraction options.
:param process_opts: Post-processing options.
:param wave: 1-D float32 samples (kaldi semantics: raw WAV magnitudes, not
   divided by 32768).
:type wave: numpy.ndarray
:return: Feature matrix of shape ``(num_frames, process_opts.dim)``,
   float32.
:rtype: numpy.ndarray

Example
-------

.. code-block:: python

   import wave
   import numpy as np
   import kaldi_native_fbank as knf

   with wave.open("zh0_1s.wav") as f:
       samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
   samples = samples.astype(np.float32)  # raw magnitudes, no /32768

   pitch_opts = knf.PitchExtractionOptions(min_f0=50, max_f0=500,
                                           snip_edges=True)
   feats = knf.compute_and_process_kaldi_pitch(pitch_opts,
                                               knf.ProcessPitchOptions(),
                                               samples)
   # feats.shape == (num_frames, 3): POV, norm-log-pitch, delta-pitch

.. seealso:: :class:`OnlinePitch`, :class:`PitchExtractionOptions`,
   :class:`ProcessPitchOptions`)doc");
}

}  // namespace knf

Pitch
=====

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/feature-pitch.h` — the equivalent of
kaldi's ``compute-and-process-kaldi-pitch-feats``
(:kaldi:`pitch-functions <src/feat/pitch-functions.cc>`, the ICASSP 2014
pitch algorithm of Ghahremani et al.).

Streaming:

.. code-block:: cpp

   knf::PitchExtractionOptions pitch_opts;
   pitch_opts.min_f0 = 50;
   pitch_opts.max_f0 = 500;
   knf::ProcessPitchOptions process_opts;   // 3-dim output by default

   knf::OnlinePitch pitch(pitch_opts, process_opts);
   pitch.AcceptWaveform(pitch_opts.samp_freq, samples, n);
   pitch.InputFinished();
   for (int32_t t = 0; t != pitch.NumFramesReady(); ++t) {
     std::vector<float> feat = pitch.GetFrame(t);  // size == pitch.Dim()
   }

Batch (kaldi's ``compute-and-process-kaldi-pitch-feats``):

.. code-block:: cpp

   knf::MatrixF feats = knf::ComputeAndProcessKaldiPitch(
       pitch_opts, process_opts, samples, n);  // (num_frames, Dim())

API (doxygen)
-------------

.. doxygenstruct:: knf::PitchExtractionOptions
   :members:

.. doxygenstruct:: knf::ProcessPitchOptions
   :members:

.. doxygenclass:: knf::OnlinePitch
   :members:

.. doxygenfunction:: knf::ComputeAndProcessKaldiPitch

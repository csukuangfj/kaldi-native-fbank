MFCC
====

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/feature-mfcc.h` — kaldi-compatible MFCC
(equal to kaldi's :kaldi:`compute-mfcc-feats <src/featbin/compute-mfcc-feats.cc>` value-by-value for identical
parameters).

.. code-block:: cpp

   knf::MfccOptions opts;
   opts.frame_opts.samp_freq = 16000;
   opts.num_ceps = 13;

   knf::OnlineMfcc mfcc(opts);
   mfcc.AcceptWaveform(sample_rate, samples.data(), samples.size());
   mfcc.InputFinished();
   for (int32_t i = 0; i < mfcc.NumFramesReady(); ++i) {
     std::vector<float> frame = mfcc.GetFrame(i);
   }

See the Python page (:doc:`../python-api/mfcc`) for the full option-to-kaldi
flag mapping.  Note: feed samples at raw int16 amplitude (kaldi semantics).

Fbank
=====

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/feature-fbank.h` — kaldi-compatible
filter-bank features (equal to kaldi's
:kaldi:`compute-fbank-feats <src/featbin/compute-fbank-feats.cc>`).

.. code-block:: cpp

   knf::FbankOptions opts;
   opts.mel_opts.num_bins = 23;

   knf::OnlineFbank fbank(opts);
   fbank.AcceptWaveform(sample_rate, samples.data(), samples.size());
   fbank.InputFinished();
   std::vector<float> frame = fbank.GetFrame(0);

See :doc:`../python-api/fbank` for the kaldi flag mapping table.

API (doxygen)
-------------

.. doxygenstruct:: knf::FbankOptions
   :members:

.. doxygentypedef:: knf::OnlineFbank

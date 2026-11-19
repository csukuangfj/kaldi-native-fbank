Online CMVN
===========

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/online-cmvn.h`

Running (online) cepstral mean/variance normalization — a port of kaldi's
``OnlineCmvn`` (:kaldi:`feat/online-feature.h <src/feat/online-feature.h>`),
push-style: feed raw feature frames, get normalized frames back.  See
:doc:`../python-api/online-cmvn` for the semantics, formulas and options.

.. code-block:: cpp

   knf::OnlineCmvnOptions opts;              // kaldi defaults
   knf::OnlineCmvnState state(global_stats); // MatrixD [2, dim+1]
   knf::OnlineCmvn cmvn(opts, state);

   for (const knf::VectorF &frame : frames) {
     knf::VectorF y = cmvn.AcceptFeature(frame);   // normalized
   }
   cmvn.Freeze();                          // fix the normalization
   knf::OnlineCmvnState s = cmvn.GetState();  // hand over to next utterance

API (doxygen)
-------------

.. doxygenstruct:: knf::OnlineCmvnOptions
   :members:

.. doxygenstruct:: knf::OnlineCmvnState
   :members:

.. doxygenclass:: knf::OnlineCmvn
   :members:

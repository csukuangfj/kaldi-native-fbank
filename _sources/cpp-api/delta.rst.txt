Delta
=====

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/feature-delta.h` — the equivalent of
kaldi's ``add-deltas`` (:kaldi:`DeltaFeatures <src/feat/feature-functions.cc>`).

.. code-block:: cpp

   knf::DeltaOptions opts;              // order=2, window=2
   knf::MatrixF out = knf::ComputeDeltas(opts, feats);

API (doxygen)
-------------

.. doxygenstruct:: knf::DeltaOptions
   :members:

.. doxygenfunction:: knf::ComputeDeltas

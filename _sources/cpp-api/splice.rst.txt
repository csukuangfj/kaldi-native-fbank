Splice
======

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/feature-splice.h` — the equivalent of
kaldi's ``splice-feats`` (:kaldi:`SpliceFrames <src/feat/feature-functions.cc>`).

.. code-block:: cpp

   knf::SpliceOptions opts;       // left_context=3, right_context=3
   knf::MatrixF spliced = knf::SpliceFrames(feats, opts);

API (doxygen)
-------------

.. doxygenstruct:: knf::SpliceOptions
   :members:

.. doxygenfunction:: knf::SpliceFrames

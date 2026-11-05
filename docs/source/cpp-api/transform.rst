Transform
=========

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/feature-transform.h` — the equivalent
of kaldi's ``transform-feats``
(:kaldi:`transform-feats <src/featbin/transform-feats.cc>`).

.. code-block:: cpp

   knf::FeatureTransform lda = knf::FeatureTransform::Read("lda.mat");
   knf::MatrixF out = lda.Apply(feats);   // (T, lda.OutputDim())

API (doxygen)
-------------

.. doxygenclass:: knf::FeatureTransform
   :members:

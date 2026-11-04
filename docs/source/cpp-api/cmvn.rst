CMVN
====

Full doxygen reference (auto-generated from the headers):
`C++ API doxygen <../cpp-api-doxygen/index.html>`_

Header: :knf:`kaldi-native-fbank/csrc/feature-cmvn.h`

Semantics follow kaldi's :kaldi:`transform/cmvn.cc <src/transform/cmvn.cc>` exactly (same arithmetic and
stats layout).

.. code-block:: cpp

   knf::MatrixF feats = ...;  // rows = frames

   // full 2-row stats: row0 = [sums, count], row1 = [sumsq, (unused)]
   knf::MatrixD stats = knf::ComputeCmvnStats(feats);

   knf::CmvnOptions opts;          // norm_vars=false, reverse=false
   knf::MatrixF out = knf::ApplyCmvn(stats, opts, feats);

   // incremental
   knf::CmvnAccumulator acc(feats.cols());
   acc.Accumulate(feats);
   const knf::MatrixD &s = acc.Stats();

API (doxygen)
-------------

.. doxygenstruct:: knf::CmvnOptions
   :members:

.. doxygenclass:: knf::CmvnAccumulator
   :members:

.. doxygenfunction:: knf::InitCmvnStats
.. doxygenfunction:: knf::AccCmvnStats(const float *feat, int32_t dim, float weight, MatrixD *stats)
.. doxygenfunction:: knf::AccCmvnStats(const MatrixF &feats, MatrixD *stats)
.. doxygenfunction:: knf::ComputeCmvnStats
.. doxygenfunction:: knf::ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts, MatrixF *feats)
.. doxygenfunction:: knf::ApplyCmvn(const MatrixD &stats, const CmvnOptions &opts, const MatrixF &feats)

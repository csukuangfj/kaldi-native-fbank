CMVN
====

Cepstral mean and variance normalization, matching kaldi's
:kaldi:`compute-cmvn-stats <src/featbin/compute-cmvn-stats.cc>` /
:kaldi:`apply-cmvn <src/featbin/apply-cmvn.cc>` (see :knf:`kaldi-native-fbank/csrc/feature-cmvn.h`).

Stats layout (identical to kaldi's ``cmvn.ark``):

* **2 rows** (full stats) — ``row0 = [sum_1..sum_D, count]``,
  ``row1 = [sumsq_1..sumsq_D, (unused)]``; supports both mean and variance
  normalization.
* **1 row** (mean-only stats) — ``row0 = [sum_1..sum_D, count]``; supports mean
  normalization only (``norm_vars=True`` is an error, same as kaldi).

Usage
-----

.. code-block:: python

   import numpy as np
   import kaldi_native_fbank as knf

   feats = np.random.rand(100, 13).astype(np.float32)  # frames x dim

   stats = knf.compute_cmvn_stats(feats)   # shape: (2, dim+1), float64

   opts = knf.CmvnOptions()
   opts.norm_vars = False
   out = knf.apply_cmvn(stats, opts, feats)   # normalized copy

   # in-place variant
   knf.apply_cmvn_(stats, opts, feats)

   # reverse (undo cmvn)
   rev = knf.CmvnOptions()
   rev.reverse = True
   restored = knf.apply_cmvn(stats, rev, out)

   # incremental / speaker-level accumulation
   acc = knf.CmvnAccumulator()
   acc.accumulate(feats)
   stats2 = acc.stats          # CmvnAccumulator.stats property -> ndarray

Formulas
--------

Let :math:`x_d` be the input feature, :math:`w` the frame weight (1.0 for
plain accumulation), and :math:`N` the frame count.  The stats are:

.. math::

   \mathrm{count} = \sum_t w_t, \qquad
   \mathrm{sum}_d = \sum_t w_t x_{t,d}, \qquad
   \mathrm{sumsq}_d = \sum_t w_t x_{t,d}^2

Per-dimension mean and variance:

.. math::

   \mu_d = \frac{\mathrm{sum}_d}{\mathrm{count}}, \qquad
   \sigma_d^2 = \frac{\mathrm{sumsq}_d}{\mathrm{count}} - \mu_d^2

Mean normalization (``norm_vars=False``, kaldi ``apply-cmvn``):

.. math::

   x'_d = x_d - \mu_d

Variance normalization too (``norm_vars=True``, kaldi ``--norm-vars``):

.. math::

   x'_d = \frac{x_d - \mu_d}{\sigma_d}
       = x_d \cdot s_d + o_d,
       \quad s_d = \frac{1}{\sigma_d},\; o_d = -\mu_d s_d

with :math:`\sigma_d^2` floored at :math:`10^{-20}` (kaldi warns and floors).
The affine form :math:`x'_d = x_d s_d + o_d` is what kaldi actually computes
(multiply by scale, then add offset), and we follow the same order.

Reverse (``reverse=True``, kaldi ``--reverse``) undoes the transform:

.. math::

   x'_d = x_d \cdot \tilde{s}_d + \tilde{o}_d, \qquad
   \tilde{s}_d = \begin{cases}1 & \text{mean-only}\\ \sigma_d & \text{norm\_vars}\end{cases},
   \quad \tilde{o}_d = \mu_d

Equivalent in kaldi
-------------------

The API mirrors kaldi's :kaldi:`compute-cmvn-stats <src/featbin/compute-cmvn-stats.cc>` /
:kaldi:`apply-cmvn <src/featbin/apply-cmvn.cc>` one-to-one.
Given the same input, the outputs are value-by-value equal (see the golden
tests):

.. code-block:: bash

   # kaldi: accumulate stats
   compute-cmvn-stats ark:feats.ark ark:cmvn_stats.ark

   # kaldi: apply (mean normalization)
   apply-cmvn --utt2spk=ark:utt2spk ark:cmvn_stats.ark ark:feats.ark ark:cmvn.ark

   # kaldi: apply with variance normalization
   apply-cmvn --norm-vars --utt2spk=ark:utt2spk ark:cmvn_stats.ark \
       ark:feats.ark ark:cmvn_normvars.ark

   # kaldi: undo cmvn
   apply-cmvn --reverse --utt2spk=ark:utt2spk ark:cmvn_stats.ark \
       ark:cmvn.ark ark:restored.ark

.. code-block:: python

   # equivalent python API
   stats = knf.compute_cmvn_stats(feats)          # == compute-cmvn-stats
   out = knf.apply_cmvn(stats, knf.CmvnOptions(), feats)          # == apply-cmvn
   opts = knf.CmvnOptions(); opts.norm_vars = True                # == --norm-vars
   out = knf.apply_cmvn(stats, opts, feats)
   opts = knf.CmvnOptions(); opts.reverse = True                  # == --reverse
   restored = knf.apply_cmvn(stats, opts, out)

Mapping
^^^^^^^

================================  ==============================================
kaldi                             python API
================================  ==============================================
``compute-cmvn-stats``            ``knf.compute_cmvn_stats(feats)``
``apply-cmvn``                    ``knf.apply_cmvn(stats, opts, feats)``
``--norm-vars``                   ``CmvnOptions.norm_vars = True``
``--reverse``                     ``CmvnOptions.reverse = True``
``cmvn.ark`` (2 x dim+1 layout)   ``stats`` (same layout, ``numpy.ndarray``)
================================  ==============================================

API reference
-------------

``knf.CmvnOptions``
  Options object matching kaldi's ``apply-cmvn`` flags.

  * ``norm_vars: bool`` — variance normalization too (kaldi ``--norm-vars``)
  * ``reverse: bool`` — undo cmvn (kaldi ``--reverse``)

``knf.compute_cmvn_stats(feats) -> numpy.ndarray``
  Returns the full 2-row stats matrix (``float64``, shape ``(2, dim+1)``),
  same layout as kaldi's ``cmvn.ark``.

``knf.apply_cmvn(stats, opts, feats) -> numpy.ndarray``
  Returns the normalized copy.  Accepts 1-row (mean-only) or 2-row stats.

``knf.apply_cmvn_(stats, opts, feats) -> None``
  In-place variant.

``knf.CmvnAccumulator(dim=0)``
  Incremental accumulator (for streaming / per-speaker stats).

  * ``accumulate(feats)`` — accumulate a feature matrix
  * ``add(dim, frame, weight=1.0)`` — accumulate one frame
  * ``stats`` (property, ``numpy.ndarray``) — the accumulated stats matrix,
    layout as above; feed it to ``apply_cmvn``

Speaker-level CMVN (kaldi's utt2spk)
------------------------------------

kaldi's ``apply-cmvn --utt2spk=ark:utt2spk ...`` looks up *per-speaker* stats
for each utterance.  Here the stats are passed in explicitly, so the
utt2spk grouping is done by the caller; use ``CmvnAccumulator`` to collect
per-speaker stats first:

.. code-block:: python

   import kaldi_native_fbank as knf

   utt2feats = {"u1": feats1, "u2": feats2, "u3": feats3}
   utt2spk = {"u1": "spkA", "u2": "spkA", "u3": "spkB"}

   # 1) per-speaker stats (== compute-cmvn-stats on each speaker's utts)
   spk2acc = {}
   for utt, feats in utt2feats.items():
       spk = utt2spk[utt]
       acc = spk2acc.setdefault(spk, knf.CmvnAccumulator())
       acc.accumulate(feats)

   # 2) apply per-speaker stats (== apply-cmvn --utt2spk=...)
   opts = knf.CmvnOptions()
   utt2normed = {}
   for utt, feats in utt2feats.items():
       # CmvnAccumulator.stats is the accumulated 2x(dim+1) stats matrix
       stats = spk2acc[utt2spk[utt]].stats
       utt2normed[utt] = knf.apply_cmvn(stats, opts, feats)

Saving results as kaldi ark/scp (`kaldi-native-io <https://github.com/csukuangfj/kaldi_native_io>`_)
---------------------------------------------------------------------------------------------------------

The stats and features are plain ``numpy`` arrays and can be written in
kaldi's ark/scp formats with `kaldi-native-io
<https://github.com/csukuangfj/kaldi_native_io>`_, producing artifacts
that kaldi tools can read directly.

Install it first (it is a separate package):

.. code-block:: bash

   pip install kaldi-native-io

Usage:

.. code-block:: python

   import kaldi_native_io as knio

   # write cmvn stats (double matrix, 2 x dim+1)  == cmvn.ark
   with knio.DoubleMatrixWriter("ark,scp:cmvn_stats.ark,cmvn_stats.scp") as w:
       for spk, acc in spk2acc.items():
           w.write(spk, acc.stats)          # key = speaker, like kaldi

   # write normalized features == cmvn output ark
   with knio.FloatMatrixWriter("ark,scp:cmvn_feats.ark,cmvn_feats.scp") as w:
       for utt, feats in utt2normed.items():
           w.write(utt, feats)

   # read back with kaldi-native-io
   with knio.SequentialDoubleMatrixReader("ark:cmvn_stats.ark") as r:
       for spk, stats in r:
           ...

The resulting ``cmvn_stats.scp``/``cmvn_feats.scp`` are ordinary kaldi scp
files and can be consumed by kaldi binaries (e.g. ``apply-cmvn``, ``splice-feats``)
unchanged.


Full class/function documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: kaldi_native_fbank.CmvnOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.CmvnAccumulator
   :members:
   :special-members: __init__

.. autofunction:: kaldi_native_fbank.compute_cmvn_stats
.. autofunction:: kaldi_native_fbank.apply_cmvn
.. autofunction:: kaldi_native_fbank.apply_cmvn_

Golden tests
------------

The implementation is verified **value-by-value** against kaldi outputs in
:knf:`tests/data/feat_golden` (generated by kaldi binaries; see the README there
for the exact commands and parameter groups).  See
:knf:`kaldi-native-fbank/python/tests/test_feature_cmvn.py`.

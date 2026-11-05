Delta features
==============

Regression delta (dynamic) features — the equivalent of kaldi's
``add-deltas`` (:knf:`kaldi-native-fbank/csrc/feature-delta.h`,
:kaldi:`DeltaFeatures <src/feat/feature-functions.cc>`).

The output concatenates the input features with their delta orders:
:math:`[x, \Delta, \Delta^2]` for the default ``order=2``, i.e.
``dim * (order + 1)`` dimensions.  Edge frames are replicated.

.. code-block:: python

   import kaldi_native_fbank as knf

   opts = knf.DeltaOptions(order=2, window=2)
   out = knf.compute_deltas(opts, feats)  # (T, 3 * dim)

API reference
-------------

.. autoclass:: kaldi_native_fbank.DeltaOptions
   :members:
   :special-members: __init__

.. autofunction:: kaldi_native_fbank.compute_deltas

Splice frames
=============

Concatenate each frame with its left/right context neighbors — the
equivalent of kaldi's ``splice-feats``
(:knf:`kaldi-native-fbank/csrc/feature-splice.h`,
:kaldi:`SpliceFrames <src/feat/feature-functions.cc>`).

Edge frames are replicated (out-of-range indices clamp to the first/last
frame), so the output keeps the same number of frames and has
``dim * (1 + left + right)`` dimensions.

.. code-block:: python

   import kaldi_native_fbank as knf

   opts = knf.SpliceOptions(left_context=3, right_context=3)
   spliced = knf.splice_frames(feats, opts)  # (T, 7 * dim)

API reference
-------------

.. autoclass:: kaldi_native_fbank.SpliceOptions
   :members:
   :special-members: __init__

.. autofunction:: kaldi_native_fbank.splice_frames

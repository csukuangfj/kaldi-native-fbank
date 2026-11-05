Feature transforms (LDA)
========================

Linear/affine feature transforms — the equivalent of kaldi's
``transform-feats`` (:knf:`kaldi-native-fbank/csrc/feature-transform.h`,
:kaldi:`transform-feats <src/featbin/transform-feats.cc>`).  This is how an
LDA matrix is applied after the splice stage.

The transform is a matrix :math:`M` of shape ``(output_dim, n)``:

* :math:`n` = input dim → linear: :math:`y = x\, M^{T}`
* :math:`n` = input dim + 1 → affine: the last column of :math:`M` is the
  offset, :math:`y = x\, M_{[:,\,0:n-1]}^{T} + M_{[:,\,n-1]}`

.. code-block:: python

   import kaldi_native_fbank as knf

   lda = knf.FeatureTransform.read("lda.mat")
   out = lda.apply(feats)                       # (T, lda.output_dim)
   out = knf.transform_features(feats, lda)     # same thing

API reference
-------------

.. autoclass:: kaldi_native_fbank.FeatureTransform
   :members:
   :special-members: __init__

.. autofunction:: kaldi_native_fbank.transform_features

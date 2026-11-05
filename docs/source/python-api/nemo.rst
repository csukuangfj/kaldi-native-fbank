NeMo filter bank
================

NeMo-compatible buffered streaming filter-bank features
(:knf:`kaldi-native-fbank/csrc/online-nemo-fbank.h`).  NeMo buffered
streaming re-featurizes each requested audio window: preemphasis, then a
centered constant-zero-padded STFT, a symmetric Hann window, Slaney mel
filters and :math:`\log(\mathrm{mel} + 2^{-24})`.

Input samples are floats in ``[-1, 1]``; the class does not resample or
normalize.  There is no direct kaldi binary counterpart — this reproduces
NeMo's streaming feature extraction.

.. code-block:: python

   import numpy as np
   import kaldi_native_fbank as knf

   opts = knf.NemoFbankOptions(num_mel_bins=80)
   fbank = knf.OnlineNemoFbank(opts)

   fbank.accept_waveform(16000, samples)   # samples in [-1, 1]
   fbank.input_finished()
   frames, offset = fbank.get_frames(0, fbank.num_frames_ready, is_final=True)

API reference
-------------

.. autoclass:: kaldi_native_fbank.NemoFbankOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.OnlineNemoFbank
   :members:
   :special-members: __init__

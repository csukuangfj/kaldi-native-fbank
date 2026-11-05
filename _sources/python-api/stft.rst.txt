STFT / iSTFT
============

Short-time Fourier transform and its inverse
(:knf:`kaldi-native-fbank/csrc/stft.h`, :knf:`kaldi-native-fbank/csrc/istft.h`).

.. code-block:: python

   import kaldi_native_fbank as knf

   config = knf.StftConfig(...)
   stft = knf.Stft(config)
   spec = stft(samples)          # complex spectra
   istft = knf.IStft(config)
   wave = istft(spec)

Note: these are low-level signal-processing building blocks (as used inside
kaldi's pitch tracker etc.); there is no direct kaldi *binary* counterpart —
kaldi implements STFT inside feature extraction.

API reference
-------------

.. autoclass:: kaldi_native_fbank.StftConfig
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.StftResult
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.Stft
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.IStft
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.Rfft
   :members:
   :special-members: __init__

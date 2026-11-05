Whisper Fbank
=============

Whisper-style log-mel features (:knf:`kaldi-native-fbank/csrc/whisper-feature.h`).

.. code-block:: python

   import kaldi_native_fbank as knf

   opts = knf.WhisperFeatureOptions()
   f = knf.OnlineWhisperFbank(opts)
   f.accept_waveform(16000, samples)
   f.input_finished()
   frame = f.get_frame(0)

Note: whisper features follow OpenAI Whisper's front-end, not kaldi; there is
no kaldi binary counterpart to compare against.

API reference
-------------

.. autoclass:: kaldi_native_fbank.WhisperFeatureOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.OnlineWhisperFbank
   :members:
   :special-members: __init__

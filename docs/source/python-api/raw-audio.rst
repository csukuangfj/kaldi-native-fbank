Raw audio framing
=================

Frame raw waveform samples **without any processing** — no dithering,
preemphasis, DC offset removal or windowing
(:knf:`kaldi-native-fbank/csrc/feature-raw-audio-samples.h`).  This is useful
for models that consume raw waveform chunks and want the same streaming
interface as the feature extractors.  There is no kaldi counterpart.

.. code-block:: python

   import numpy as np
   import kaldi_native_fbank as knf

   opts = knf.RawAudioSamplesOptions()
   framer = knf.OnlineRawAudioSamples(opts)

   framer.accept_waveform(16000, samples)
   framer.input_finished()
   chunk = framer.get_frame(0)  # (dim,) raw samples of the first frame

API reference
-------------

.. autoclass:: kaldi_native_fbank.RawAudioSamplesOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.OnlineRawAudioSamples
   :members:
   :special-members: __init__

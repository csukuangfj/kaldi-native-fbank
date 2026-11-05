Window & frame extraction
=========================

Splitting a waveform into overlapping frames and pre-processing each frame
(dithering, DC offset removal, preemphasis, windowing) before the FFT
(:knf:`kaldi-native-fbank/csrc/feature-window.h`).
kaldi counterpart: :kaldi:`FrameExtractionOptions <src/feat/feature-window.h>`.
See :doc:`reference` for the full field-by-field mapping to kaldi options.

Every feature extractor accepts these settings via its ``frame_opts`` field.
The window formulas for each ``window_type`` are documented on
:py:meth:`kaldi_native_fbank.FrameExtractionOptions` (and in the C++ docs).

.. code-block:: python

   import kaldi_native_fbank as knf

   opts = knf.FrameExtractionOptions()
   opts.samp_freq = 16000
   opts.dither = 0  # deterministic output

   window = knf.FeatureWindowFunction(opts)
   frame = [0.0] * len(window.window)  # one frame of window_size
   windowed = window.apply(frame)

API reference
-------------

.. autoclass:: kaldi_native_fbank.FrameExtractionOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.FeatureWindowFunction
   :members:
   :special-members: __init__

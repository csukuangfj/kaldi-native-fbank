Mel filter banks
================

Triangular mel-frequency filter banks that project FFT bin energies onto mel
bins (:knf:`kaldi-native-fbank/csrc/mel-computations.h`).
kaldi counterpart: :kaldi:`MelBanksOptions <src/feat/mel-computations.h>`.

Both kaldi-style and librosa-style filters are supported (``is_librosa=True``).
The MFCC/fbank extractors use these internally via ``mel_opts``; you can also
use them directly:

.. code-block:: python

   import numpy as np
   import kaldi_native_fbank as knf

   mel_opts = knf.MelBanksOptions(num_bins=23, low_freq=20, high_freq=7800)
   frame_opts = knf.FrameExtractionOptions()
   banks = knf.MelBanks(mel_opts, frame_opts)

   weights = banks.get_matrix()          # (23, num_fft_bins // 2 + 1)
   mel = banks.compute(power_spectrum)   # (23,) mel energies, not log

API reference
-------------

.. autoclass:: kaldi_native_fbank.MelBanksOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.MelBanks
   :members:
   :special-members: __init__

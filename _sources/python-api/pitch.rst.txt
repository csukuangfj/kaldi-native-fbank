Pitch
=====

Pitch extraction — the equivalent of kaldi's
``compute-and-process-kaldi-pitch-feats``
(:knf:`kaldi-native-fbank/csrc/feature-pitch.h`,
:kaldi:`pitch-functions <src/feat/pitch-functions.cc>`), implementing
*"A Pitch Extraction Algorithm Tuned for Automatic Speech Recognition"*,
Ghahremani et al., ICASSP 2014.

The extractor downsamples the waveform, evaluates the normalized
cross-correlation function (NCCF) at log-spaced lags between
:math:`1/\mathrm{max\_f0}` and :math:`1/\mathrm{min\_f0}`, picks a per-frame
lag with a Viterbi search, and post-processes the raw
:math:`(\mathrm{NCCF}, \mathrm{pitch})` pair into the configured features —
by default the 3-dimensional (POV-feature, normalized-log-pitch,
delta-log-pitch) used by kaldi.

Batch usage (kaldi's ``compute-and-process-kaldi-pitch-feats``):

.. code-block:: python

   import kaldi_native_fbank as knf

   pitch_opts = knf.PitchExtractionOptions(min_f0=50, max_f0=500,
                                           snip_edges=True)
   process_opts = knf.ProcessPitchOptions()  # 3-dim output
   feats = knf.compute_and_process_kaldi_pitch(pitch_opts, process_opts,
                                               samples)  # (T, 3)

Streaming usage (same interface as :class:`kaldi_native_fbank.OnlineMfcc`):

.. code-block:: python

   pitch = knf.OnlinePitch(pitch_opts, process_opts)
   pitch.accept_waveform(16000, samples)
   pitch.input_finished()
   for t in range(pitch.num_frames_ready):
       frame = pitch.get_frame(t)  # np.ndarray of shape (3,)

**Sample scale**: feed samples at raw int16 amplitude (do not divide by
32768); kaldi's wav reader keeps raw magnitudes.

**Determinism**: the noise added to the delta-pitch feature
(``delta_pitch_noise_stddev``) is drawn from a deterministic stream that
reproduces the reference kaldi realization, so features are identical on
every platform (and match the golden test data in
``tests/data/feat_golden``).

API reference
-------------

.. autoclass:: kaldi_native_fbank.PitchExtractionOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.ProcessPitchOptions
   :members:
   :special-members: __init__

.. autoclass:: kaldi_native_fbank.OnlinePitch
   :members:
   :special-members: __init__

.. autofunction:: kaldi_native_fbank.compute_and_process_kaldi_pitch

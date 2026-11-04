Python API reference
====================

Every class and every field, with the kaldi counterpart of each item.
Option classes support ``as_dict()`` / ``from_dict()``.

FrameExtractionOptions
----------------------

Windowing/frame extraction.  kaldi counterpart:
:kaldi:`FrameExtractionOptions <src/feat/feature-window.h>`.

* ``samp_freq: float = 16000`` — sample rate of the waveform in Hz.
  *kaldi*: ``--sample-frequency`` / ``FrameExtractionOptions::samp_freq``.
* ``frame_shift_ms: float = 10`` — frame shift in milliseconds; frames are
  emitted every ``samp_freq * frame_shift_ms / 1000`` samples.
  *kaldi*: ``--frame-shift``.
* ``frame_length_ms: float = 25`` — window length in milliseconds.
  *kaldi*: ``--frame-length``.
* ``dither: float = 0.00003`` — amplitude of random dither added to samples
  before analysis (applied per call).  **0 disables dithering** — use 0 for
  deterministic output.  NOTE: knf's scale differs from kaldi's; the knf
  default 0.00003 is equivalent to kaldi's default 1.0.
  *kaldi*: ``--dither``.
* ``preemph_coeff: float = 0.97`` — preemphasis coefficient
  :math:`x[i] \leftarrow x[i] - c\,x[i-1]`.  *kaldi*: ``--preemphasis-coefficient``.
* ``remove_dc_offset: bool = True`` — subtract the mean of the frame before
  windowing.  *kaldi*: ``--remove-dc-offset``.
* ``window_type: str = "povey"`` — window shape; one of ``"povey"``,
  ``"hanning"``, ``"hamming"``, ``"sine"``, ``"blackman"``, ``"rectangular"``.
  *kaldi*: ``--window-type``.
* ``round_to_power_of_two: bool = True`` — round the FFT size up to a power of
  two (faster FFTs).  *kaldi*: ``--round-to-power-of-two``.
* ``blackman_coeff: float = 0.42`` — coefficient of the generalized Blackman
  window (only for ``window_type="blackman"``).  *kaldi*: ``--blackman-coeff``.
* ``snip_edges: bool = True`` — if True, drop frames that would extend beyond
  the waveform (num_frames = (n - window)/shift + 1); if False, pad edges and
  center windows on the whole signal.  *kaldi*: ``--snip-edges``.

MelBanksOptions
---------------

Mel filter-bank construction.  kaldi counterpart:
:kaldi:`MelBanksOptions <src/feat/mel-computations.h>`.

* ``num_bins: int = 25`` — number of mel bins (output dim of ``MelBanks``).
  *kaldi*: ``--num-mel-bins``.
* ``low_freq: float = 20`` — lower cut-off (Hz) of the mel filters.
  *kaldi*: ``--low-freq``.
* ``high_freq: float = 0`` — upper cut-off (Hz); **0 means Nyquist**
  (samp_freq/2).  *kaldi*: ``--high-freq``.
* ``vtln_low: float = 100`` / ``vtln_high: float = -500`` — VTLN warp range
  (``vtln_high <= 0`` means Nyquist).  *kaldi*: ``--vtln-low`` / ``--vtln-high``.
* ``htk_mode: bool = False`` — match HTK's filter scaling exactly (HtkMelScale).
  *kaldi*: ``--htk-mode``.
* ``use_slaney_mel_scale: bool = True`` — use the Slaney mel scale
  (kaldi's ``MelScaleSlaney``).  **knf defaults to True; kaldi defaults to
  False.**  *kaldi*: ``--use-slaney-mel-scale``.
* ``norm: str = "slaney"`` — filter area normalization: ``"none"`` or
  ``"slaney"``.  **knf defaults to "slaney"; kaldi defaults to "none".**
* ``debug_mel: bool = False`` / ``floor_to_int_bin: bool = False`` /
  ``is_librosa: bool = False`` — special modes (debug dump / integer bin
  flooring / librosa-compatible scaling).

MfccOptions
-----------

MFCC computation.  kaldi counterpart:
:kaldi:`MfccOptions <src/feat/feature-mfcc.h>`.

* ``frame_opts: FrameExtractionOptions`` / ``mel_opts: MelBanksOptions`` —
  sub-options; note ``mel_opts.num_bins`` is the number of mel filters
  (typically 23), ``num_ceps`` is the output dim.
* ``num_ceps: int = 13`` — number of cepstral coefficients (output dim).
  *kaldi*: ``--num-ceps``.
* ``use_energy: bool = True`` — replace C0 with the (log) frame energy.
  **knf defaults to True; kaldi's compute-mfcc-feats defaults to false.**
  *kaldi*: ``--use-energy``.
* ``energy_floor: float = 0`` — floor on energy before log (0 = off).
  *kaldi*: ``--energy-floor``.
* ``raw_energy: bool = True`` — compute energy from the raw (pre-preemphasis)
  frame.  *kaldi*: ``--raw-energy``.
* ``cepstral_lifter: float = 22`` — liftering coefficient applied to the
  cepstra.  *kaldi*: ``--cepstral-lifter``.
* ``htk_compat: bool = False`` — HTK-compatible scaling of the final output.
  *kaldi*: ``--htk-compat``.

FbankOptions
------------

Filter-bank features.  kaldi counterpart:
:kaldi:`FbankOptions <src/feat/feature-fbank.h>`.

* ``frame_opts`` / ``mel_opts`` — as above.
* ``use_energy: bool = False`` — use frame energy instead of mel energies.
  *kaldi*: ``--use-energy``.
* ``energy_floor: float = 0`` — energy floor (with ``use_energy``).
  *kaldi*: ``--energy-floor``.
* ``raw_energy: bool = True`` — energy from the raw (pre-preemphasis) frame.
  *kaldi*: ``--raw-energy``.
* ``use_log_fbank: bool = True`` — take log of the filter energies
  (turn off for cepstral-style pipelines).  *kaldi*: ``--use-log-fbank``.
* ``use_power: bool = True`` — use power spectrum (vs magnitude).
  *kaldi*: ``--use-power``.
* ``htk_compat: bool = False`` — HTK-compatible scaling.  *kaldi*: ``--htk-compat``.

WhisperFeatureOptions
---------------------

Whisper-style log-mel features.  No kaldi counterpart (OpenAI Whisper
front-end).

* ``frame_opts: FrameExtractionOptions`` — window settings.
* ``dim: int = 80`` — number of mel bins (output dim).

RawAudioSamplesOptions
----------------------

Raw audio frame output.  kaldi counterpart: kaldi's ``OnlineRawAudioSamples``
(used e.g. for pitch input).

* ``frame_opts: FrameExtractionOptions`` — window settings.

StftConfig
----------

STFT configuration (no direct kaldi binary counterpart; kaldi implements
STFT inside feature extraction).

* ``n_fft: int`` / ``win_length: int`` / ``hop_length: int`` — FFT / window /
  hop sizes in samples.  *kaldi*: the FFT size is
  ``frame_opts.PaddedWindowSize()``; hop = ``samp_freq*frame_shift_ms/1000``.
* ``window_type: str = ""`` — window shape (empty = rectangular default).
* ``center: bool = True`` — pad the signal so frames are centered.
* ``pad_mode: str = "reflect"`` — padding mode when ``center=True``.
* ``normalized: bool = False`` — unitary normalization.

Streaming extractors
--------------------

``OnlineMfcc`` / ``OnlineFbank`` / ``OnlineWhisperFbank`` /
``OnlineRawAudioSamples`` share one interface
(kaldi counterpart: :kaldi:`OnlineFeatureInterface <src/itf/online-feature-itf.h>`):

.. code-block:: python

   ext = knf.OnlineMfcc(opts)
   ext.accept_waveform(sample_rate, samples)  # 1-D float32, RAW int16 amplitude
   ext.input_finished()                      # signal: no more audio
   ext.num_frames_ready                      # frames available now
   frame = ext.get_frame(i)                  # list[float], len == ext.dim
   frame = ext.pop()                         # consume the oldest frame
   ext.is_last_frame(i)                      # True for the final frame
   ext.dim                                   # output dimension
   ext.frame_shift_in_seconds                # e.g. 0.01

* ``accept_waveform(sample_rate, samples)`` — push audio
  (kaldi ``OnlineFeatureItf::AcceptWaveform``).
* ``input_finished()`` — flush the tail (kaldi ``InputFinished``).
* ``get_frame(i)`` / ``pop()`` — random / sequential frame access
  (kaldi ``GetFrame``).
* ``dim`` — feature dim: MFCC ``= num_ceps``; Fbank ``= mel_opts.num_bins``
  (``+1`` with ``use_energy``); Whisper ``= dim``.

**Sample scale**: feed samples at raw int16 amplitude (do not divide by
32768); kaldi's wav reader keeps raw magnitudes.

DSP building blocks
-------------------

* ``MelBanks(opts)`` — mel filter bank; ``dim()``, ``compute(spectrum) ->
  mel_energies``, ``get_matrix()``, ``mel_scale`` / ``inverse_mel_scale`` /
  ``mel_scale_slaney`` / ``inverse_mel_scale_slaney``
  (kaldi :kaldi:`MelBanks <src/feat/mel-computations.h>`).
* ``FeatureWindowFunction(frame_opts, n)`` — window; ``apply(window, frame)``
  (kaldi :kaldi:`FeatureWindowFunction <src/feat/feature-window.h>`).
* ``Rfft(n, inverse=False)`` — in-place real FFT; ``compute(data)``
  (wraps kissfft).
* ``Stft(config)`` — ``compute(samples) -> StftResult``.
* ``IStft(config)`` — ``compute(spec) -> samples`` (overlap-add).
* ``StftResult`` — ``real``, ``imag`` (2-D arrays), ``num_frames``.

CMVN
----

See :doc:`cmvn` for formulas and kaldi command mapping
(:kaldi:`transform/cmvn.cc <src/transform/cmvn.cc>`).

``CmvnOptions``
  * ``norm_vars: bool = False`` — variance normalization.
    *kaldi*: ``apply-cmvn --norm-vars``.
  * ``reverse: bool = False`` — undo cmvn.  *kaldi*: ``apply-cmvn --reverse``.

``CmvnAccumulator(dim=0)``
  * ``accumulate(feats)`` — accumulate a ``(frames, dim)`` matrix.
  * ``add(dim, frame, weight=1.0)`` — accumulate one frame with weight
    (kaldi ``AccCmvnStats``).
  * ``stats`` — property: accumulated ``(2, dim+1)`` float64 matrix,
    kaldi ``cmvn.ark`` layout (``row0 = [sum_1..sum_D, count]``,
    ``row1 = [sumsq_1..sumsq_D, unused]``).

Module functions
----------------

* ``knf.compute_cmvn_stats(feats) -> numpy.ndarray`` — full 2-row stats
  (*kaldi*: ``compute-cmvn-stats``).
* ``knf.apply_cmvn(stats, opts, feats) -> numpy.ndarray`` — normalized copy
  (*kaldi*: ``apply-cmvn``).
* ``knf.apply_cmvn_(stats, opts, feats)`` — in-place variant.

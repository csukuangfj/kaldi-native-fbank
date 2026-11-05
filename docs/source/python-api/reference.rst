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

``SpliceOptions(left_context=3, right_context=3)``
  Contexts for :func:`splice_frames`.  kaldi counterpart:
  :kaldi:`splice-feats <src/featbin/splice-feats.cc>`.

  * ``left_context: int = 3`` — number of preceding frames to append.
    *kaldi*: ``splice-feats --left-context``.
  * ``right_context: int = 3`` — number of following frames to append.
    *kaldi*: ``splice-feats --right-context``.

``DeltaOptions(order=2, window=2)``
  Regression window for :func:`compute_deltas`.  kaldi counterpart:
  :kaldi:`add-deltas <src/featbin/add-deltas.cc>`.

  * ``order: int = 2`` — order of the delta computation.
    *kaldi*: ``add-deltas --delta-order``.
  * ``window: int = 2`` — regression window; the regression spans
    ``2 * window + 1`` frames.  *kaldi*: ``add-deltas --delta-window``.

``FeatureTransform``
  Linear/affine transforms (e.g. LDA).  kaldi counterpart:
  :kaldi:`transform-feats <src/featbin/transform-feats.cc>`.

  * ``FeatureTransform.read(filename)`` — read a kaldi matrix file
    (binary ``FM``/``DM`` or text ``[ ... ]``).
  * ``input_dim`` / ``output_dim`` — shape of the stored matrix.
  * ``is_affine(input_dim)`` — true if the matrix has one offset column
    (``n == input_dim + 1``).
  * ``apply(feats)`` — ``y = x M^T``; affine form appends the offset.

Pitch
-----

See :doc:`pitch` for the algorithm and usage.  kaldi counterpart:
:kaldi:`pitch-functions.h <src/feat/pitch-functions.h>`.

``PitchExtractionOptions``
  Raw pitch extraction (kaldi ``compute-kaldi-pitch-feats``).

  * ``samp_freq: float = 16000`` — sample rate in Hz.
    *kaldi*: ``--sample-frequency``.
  * ``frame_shift_ms: float = 10`` — frame shift in ms.  *kaldi*: ``--frame-shift``.
  * ``frame_length_ms: float = 25`` — frame length in ms.  *kaldi*: ``--frame-length``.
  * ``preemph_coeff: float = 0`` — preemphasis (deprecated in kaldi).
    *kaldi*: ``--preemphasis-coefficient``.
  * ``min_f0: float = 50`` / ``max_f0: float = 400`` — F0 search range (Hz).
    *kaldi*: ``--min-f0`` / ``--max-f0``.
  * ``soft_min_f0: float = 10`` — soft minimum f0, must not exceed ``min_f0``.
    *kaldi*: ``--soft-min-f0``.
  * ``penalty_factor: float = 0.1`` — cost factor for F0 change.
    *kaldi*: ``--penalty-factor``.
  * ``lowpass_cutoff: float = 1000`` — low-pass filter cutoff (Hz).
    *kaldi*: ``--lowpass-cutoff``.
  * ``resample_freq: float = 4000`` — downsample rate (> 2 * ``lowpass_cutoff``).
    *kaldi*: ``--resample-frequency``.
  * ``delta_pitch: float = 0.005`` — pitch tolerance in pruning lags.
    *kaldi*: ``--delta-pitch``.
  * ``nccf_ballast: float = 7000`` — reduces NCCF for quiet frames.
    *kaldi*: ``--nccf-ballast``.
  * ``lowpass_filter_width: int = 1`` / ``upsample_filter_width: int = 5`` —
    filter widths.  *kaldi*: ``--lowpass-filter-width`` /
    ``--upsample-filter-width``.
  * ``max_frames_latency: int = 0`` — max latency allowed for pitch tracking.
    *kaldi*: ``--max-frames-latency``.
  * ``frames_per_chunk: int = 0`` — batch chunk size (0 = whole utterance).
    *kaldi*: ``--frames-per-chunk``.
  * ``simulate_first_pass_online: bool = False`` — emit first-pass online
    features (batch mode).  *kaldi*: ``--simulate-first-pass-online``.
  * ``recompute_frame: int = 500`` — frame at which online NCCF
    recomputation happens.  *kaldi*: ``--recompute-frame``.
  * ``nccf_ballast_online: bool = False`` — debug flag for ballast
    computation.  *kaldi*: ``--nccf-ballast-online``.
  * ``snip_edges: bool = True`` — if False, incomplete end frames are not
    snipped.  *kaldi*: ``--snip-edges`` (kaldi's pitch default is True).
  * ``nccf_window_size`` / ``nccf_window_shift`` — read-only: NCCF window
    size/shift in resampled samples.

``ProcessPitchOptions``
  Post-processing (kaldi ``compute-and-process-kaldi-pitch-feats``); at
  least one ``add_*`` feature must be enabled.

  * ``pitch_scale: float = 2`` — scaling of the normalized log-pitch.
    *kaldi*: ``--pitch-scale``.
  * ``pov_scale: float = 2`` / ``pov_offset: float = 0`` — POV scaling and
    offset.  *kaldi*: ``--pov-scale`` / ``--pov-offset``.
  * ``delta_pitch_scale: float = 10`` — scaling of the delta log-pitch.
    *kaldi*: ``--delta-pitch-scale``.
  * ``delta_pitch_noise_stddev: float = 0.005`` — noise added to the delta
    log-pitch; drawn from a deterministic stream, so features are
    reproducible.  *kaldi*: ``--delta-pitch-noise-stddev``.
  * ``normalization_left_context: int = 75`` /
    ``normalization_right_context: int = 75`` — log-pitch normalization
    window (frames).  *kaldi*: ``--normalization-left-context`` /
    ``--normalization-right-context``.
  * ``delta_window: int = 2`` — frames on each side for the delta window.
    *kaldi*: ``--delta-window``.
  * ``delay: int = 0`` — frames by which the pitch is delayed.
    *kaldi*: ``--delay``.
  * ``add_pov_feature: bool = True`` — add the warped NCCF (1 dim).
    *kaldi*: ``--add-pov-feature``.
  * ``add_normalized_log_pitch: bool = True`` — add the POV-weighted
    mean-subtracted log-pitch (1 dim).  *kaldi*:
    ``--add-normalized-log-pitch``.
  * ``add_delta_pitch: bool = True`` — add the delta log-pitch (1 dim).
    *kaldi*: ``--add-delta-pitch``.
  * ``add_raw_log_pitch: bool = False`` — add log(pitch) (1 dim).
    *kaldi*: ``--add-raw-log-pitch``.
  * ``dim`` — read-only output dimension implied by the ``add_*`` flags.

``OnlinePitch(pitch_opts, process_opts=ProcessPitchOptions())``
  Streaming extractor with the interface of *Streaming extractors* above
  (without ``pop()``): ``accept_waveform(rate, samples)``,
  ``input_finished()``, ``num_frames_ready``, ``get_frame(i) ->
  numpy.ndarray``, ``is_last_frame(i)``, ``dim``,
  ``frame_shift_in_seconds``.  kaldi counterpart:
  :kaldi:`OnlinePitchFeature + OnlineProcessPitch <src/feat/pitch-functions.h>`.

Module functions
----------------

* ``knf.compute_cmvn_stats(feats) -> numpy.ndarray`` — full 2-row stats
  (*kaldi*: ``compute-cmvn-stats``).
* ``knf.apply_cmvn(stats, opts, feats) -> numpy.ndarray`` — normalized copy
  (*kaldi*: ``apply-cmvn``).
* ``knf.apply_cmvn_(stats, opts, feats)`` — in-place variant.
* ``knf.transform_features(feats, transform) -> numpy.ndarray`` — apply a
  linear/affine transform (shorthand for ``transform.apply(feats)``).
  *kaldi*: ``transform-feats``.
* ``knf.compute_deltas(opts, feats) -> numpy.ndarray`` — regression delta
  features ``[x, delta, delta-delta, ...]``, dimension ``dim * (order + 1)``;
  edge frames are replicated.  *kaldi*: ``add-deltas``
  (:kaldi:`DeltaFeatures <src/feat/feature-functions.cc>`).
* ``knf.splice_frames(feats, opts=SpliceOptions()) -> numpy.ndarray`` —
  concatenate each frame with its left/right context neighbors; edge frames
  are replicated, so the frame count stays the same and the dimension becomes
  ``dim * (1 + left_context + right_context)``.  *kaldi*: ``splice-feats``
  (:kaldi:`SpliceFrames <src/feat/feature-functions.cc>`).
* ``knf.compute_and_process_kaldi_pitch(pitch_opts, process_opts, wave) ->
  numpy.ndarray`` — processed pitch features of shape
  ``(num_frames, process_opts.dim)`` for a whole utterance (POV,
  normalized-log-pitch, delta-pitch by default).  *kaldi*:
  ``compute-and-process-kaldi-pitch-feats``
  (:kaldi:`ComputeAndProcessKaldiPitch <src/feat/pitch-functions.cc>`).

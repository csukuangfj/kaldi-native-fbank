"""
kaldi-native-fbank: kaldi-compatible feature extraction (C++ core, Python API).

Computes acoustic features that are numerically compatible with kaldi:
MFCC, filter bank (fbank), mel banks, STFT/ISTFT, CMVN, Whisper features,
raw-audio framing and more.

Quick start
-----------
>>> import numpy as np
>>> import kaldi_native_fbank as knf
>>> opts = knf.FbankOptions()  # defaults are visible in the signature
>>> fbank = knf.OnlineFbank(opts)
>>> samples = np.random.rand(16000).astype(np.float32)
>>> fbank.accept_waveform(16000, samples)
>>> fbank.input_finished()
>>> fbank.num_frames_ready
>>> feats = fbank.get_frame(0)  # np.ndarray of shape (dim,)

Main entry points
-----------------
Streaming extractors: OnlineFbank, OnlineMfcc, OnlineWhisperFbank,
    OnlineNemoFbank, OnlineRawAudioSamples, OnlinePitch
Options: FbankOptions, MfccOptions, FrameExtractionOptions, MelBanksOptions,
    WhisperFeatureOptions, NemoFbankOptions, RawAudioSamplesOptions,
    StftConfig, CmvnOptions, PitchExtractionOptions, ProcessPitchOptions
CMVN: compute_cmvn_stats, apply_cmvn, apply_cmvn_, CmvnAccumulator
Pitch: compute_and_process_kaldi_pitch
Transforms: Stft, IStft, Rfft, MelBanks, FeatureWindowFunction

Every option class constructor takes all fields as keyword arguments with
their defaults, so ``help(knf.FbankOptions)`` shows the full signature.
"""
from __future__ import annotations
import collections.abc
from kaldi_native_fbank.lib._kaldi_native_fbank import apply_cmvn
from kaldi_native_fbank.lib._kaldi_native_fbank import apply_cmvn_
from kaldi_native_fbank.lib._kaldi_native_fbank import compute_and_process_kaldi_pitch
from kaldi_native_fbank.lib._kaldi_native_fbank import compute_cmvn_stats
from kaldi_native_fbank.lib._kaldi_native_fbank import compute_deltas
from kaldi_native_fbank.lib._kaldi_native_fbank import splice_frames
from kaldi_native_fbank.lib._kaldi_native_fbank import transform_features
import numpy
import numpy.typing
import typing
from . import lib
__all__: list = ['CmvnAccumulator', 'CmvnOptions', 'DeltaOptions', 'FeatureTransform', 'FbankOptions', 'FeatureWindowFunction', 'FrameExtractionOptions', 'IStft', 'MelBanks', 'MelBanksOptions', 'MfccOptions', 'NemoFbankOptions', 'OnlineFbank', 'OnlineMfcc', 'OnlineNemoFbank', 'OnlinePitch', 'OnlineRawAudioSamples', 'OnlineWhisperFbank', 'PitchExtractionOptions', 'ProcessPitchOptions', 'RawAudioSamplesOptions', 'SpliceOptions', 'Rfft', 'Stft', 'StftConfig', 'StftResult', 'WhisperFeatureOptions', 'apply_cmvn', 'apply_cmvn_', 'compute_cmvn_stats', 'compute_and_process_kaldi_pitch', 'compute_deltas', 'transform_features', 'splice_frames']
class CmvnAccumulator:
    """
    Incremental accumulator for speaker-level / streaming cmvn.
    
    It always keeps the full 2-row layout (see :func:`compute_cmvn_stats`) and
    adds up weighted frames one by one, so stats can be collected over many
    utterances of the same speaker and normalized later with
    :func:`apply_cmvn`.
    
    .. seealso:: :func:`compute_cmvn_stats`, :func:`apply_cmvn`
    """
    def __init__(self, dim: typing.SupportsInt = 0) -> None:
        """
        Construct an accumulator for features of dimension ``dim``.
        
        :param dim: Feature dimension.  ``0`` (the default) creates an empty
           accumulator whose stats are grown by the first :meth:`add` or
           :meth:`accumulate` call.
        """
    def accumulate(self, feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"]) -> None:
        """
        Accumulate stats for a feature matrix (rows = frames).
        
        Equivalent to calling :meth:`add` for each row with weight 1.0.
        
        :param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
        :type feats: numpy.ndarray
        """
    def add(self, dim: typing.SupportsInt, frame: collections.abc.Sequence[typing.SupportsFloat], weight: typing.SupportsFloat = 1.0) -> None:
        """
        Accumulate stats for one frame of dimension ``dim``.
        
        Applies the same updates as kaldi's `AccCmvnStats <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_; with
        :math:`w = \\mathrm{weight}` and :math:`x_d = \\mathrm{frame}[d]`:
        
        .. math::
        
           c \\mathrel{+}= w, \\qquad
           S_d \\mathrel{+}= x_d\\, w, \\qquad
           Q_d \\mathrel{+}= x_d^2\\, w
        
        :param dim: Feature dimension; must match the dimension given to the
           constructor.
        :param frame: One frame of shape ``(dim,)``.
        :type frame: Sequence[float]
        :param weight: Weight of this frame, e.g. 1.0.
        """
    @property
    def stats(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        The accumulated 2-row stats.
        
        Row 0 is ``[sums, count]``, row 1 is ``[sumsq, (unused)]``; see
        :func:`compute_cmvn_stats` for the layout.
        
        :rtype: numpy.ndarray
        """
class CmvnOptions:
    """
    Options matching kaldi's `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_
    flags.
    
    The flags control how :func:`apply_cmvn` normalizes features.
    
    .. seealso:: :func:`apply_cmvn`
       For the normalization formulas.
    """
    def __init__(self, norm_vars: bool = False, reverse: bool = False) -> None:
        """
        Construct with the given flags.
        
        :param norm_vars: If ``True``, normalize the variances as well
           (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--norm-vars``).
        :param reverse: If ``True``, undo cmvn instead of applying it
           (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--reverse``).
        """
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    @property
    def norm_vars(self) -> bool:
        """
        If ``True``, normalize the variances as well: after
        subtracting the mean, each dimension is divided by its standard deviation.
        Requires 2-row stats (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--norm-vars``).
        """
    @norm_vars.setter
    def norm_vars(self, arg0: bool) -> None:
        ...
    @property
    def reverse(self) -> bool:
        """
        If ``True``, undo cmvn instead of applying it: the
        mean is added back and, if ``norm_vars`` is ``True``, the standard deviation is
        multiplied back (kaldi `apply-cmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/apply-cmvn.cc>`_ ``--reverse``).
        """
    @reverse.setter
    def reverse(self, arg0: bool) -> None:
        ...
class DeltaOptions:
    """
    Options for :func:`compute_deltas`.
    
    Mirrors the flags of kaldi's `add-deltas <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/add-deltas.cc>`_.
    
    .. seealso:: :func:`compute_deltas`
    """
    def __init__(self, order: typing.SupportsInt = 2, window: typing.SupportsInt = 2) -> None:
        """
        Construct with the given order and window.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param order: Order of the delta computation (kaldi ``add-deltas --delta-order``).
        :param window: Delta regression window (kaldi ``add-deltas --delta-window``).
        """
    @property
    def order(self) -> int:
        """
        Order of the delta computation.
        
        Same as kaldi ``add-deltas --delta-order``.  2 gives ``[x, delta,
        delta-delta]``.
        """
    @order.setter
    def order(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def window(self) -> int:
        """
        Delta regression window.
        
        Same as kaldi ``add-deltas --delta-window``.  The actual regression spans
        ``2 * window + 1`` frames.
        """
    @window.setter
    def window(self, arg0: typing.SupportsInt) -> None:
        ...
class FeatureTransform:
    """
    A linear (or affine) feature transform, e.g. an LDA matrix.
    
    This is kaldi's `transform-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/transform-feats.cc>`_
    (same semantics).  The transform is a matrix :math:`M` of shape
    :math:`(D_{out} \\times n)`:
    
    * if :math:`n` equals the input feature dimension, the transform is
      linear: :math:`y = x\\, M^{T}`;
    * if :math:`n` is one larger, the transform is affine and the last column
      of :math:`M` is the offset:
      :math:`y = x\\, M_{[:,\\,0:n-1]}^{T} + M_{[:,\\,n-1]}`;
    * otherwise :meth:`apply` raises an exception.
    
    Example
    -------
    
    .. code-block:: python
    
       lda = knf.FeatureTransform.read("lda.mat")
       out = lda.apply(feats)            # (T, lda.output_dim)
       out = knf.transform_features(feats, lda)
    
    .. seealso:: :func:`transform_features`
    """
    @staticmethod
    def read(filename: str) -> FeatureTransform:
        """
        Read a transform matrix from a kaldi matrix file.
        
        Supports the kaldi binary format (``"FM"``/``"DM"``, as written by kaldi's
        ``Matrix::Write``) and the kaldi text format (``[ ... ]``).
        
        :param filename: Path of the matrix file, e.g. ``lda.mat``.
        :return: The transform.
        """
    def __init__(self, mat: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"]) -> None:
        """
        Construct a transform from a matrix directly.
        
        :param mat: The transform matrix of shape ``(output_dim, input_dim)``, or
           one column wider for an affine transform (the last column is the
           offset).
        """
    def apply(self, feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]"]:
        """
        Apply the transform to a feature matrix.
        
        kaldi ``transform-feats`` semantics (rows = frames): if the transform has
        ``feats.shape[1]`` columns the output is :math:`Y = X M^{T}` (linear); if it
        has one column more, the transform is affine and the last column is the
        offset.  Otherwise an exception is raised.
        
        :param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
        :type feats: numpy.ndarray
        :return: The transformed matrix of shape ``(num_frames, output_dim)``,
           float32.
        :rtype: numpy.ndarray
        """
    def is_affine(self, input_dim: typing.SupportsInt) -> bool:
        """
        Whether the transform is affine for inputs of dimension ``input_dim``.
        
        The matrix alone cannot record the intended input dimension, so the rule
        is the one kaldi's ``transform-feats`` applies: the transform is affine iff
        it has one column more than the feature dimension (the extra column is the
        offset).
        
        :param input_dim: The input feature dimension.
        :return: True for an affine transform.
        """
    @property
    def input_dim(self) -> int:
        """
        Number of columns of the stored matrix.
        
        This is the feature dimension in the linear form; an affine transform
        consumes ``input_dim - 1`` features plus the implicit 1.0.
        """
    @property
    def output_dim(self) -> int:
        """
        Number of rows of the stored matrix (the output dimension).
        """
class FbankOptions:
    """
    Options for computing filter-bank features.
    
    This is kaldi's `FbankOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-fbank.h>`_.  It only includes things that can be done
    in a "stateless" way: it does not include energy max-normalization and it
    does not include delta computation.
    
    .. seealso::
       :class:`FrameExtractionOptions`
          Frame extraction settings (``frame_opts``).
       :class:`MelBanksOptions`
          Mel filter-bank settings (``mel_opts``).
       :class:`kaldi_native_fbank.OnlineFbank`
          The streaming extractor that consumes these options.
    """
    @staticmethod
    def from_dict(arg0: dict) -> FbankOptions:
        """
        Build options from a dict.
        """
    def __getstate__(self) -> dict:
        ...
    def __init__(self, frame_opts: FrameExtractionOptions = ..., mel_opts: MelBanksOptions = ..., use_energy: bool = False, energy_floor: typing.SupportsFloat = 0, raw_energy: bool = True, htk_compat: bool = False, use_log_fbank: bool = True, use_log_floor: bool = True, eps: typing.SupportsFloat = 1.1920929e-07, use_power: bool = True) -> None:
        """
        Construct with the given options.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param frame_opts: Frame-extraction options; see :class:`FrameExtractionOptions`.
        :param mel_opts: Mel filter-bank options (default 23 bins); see :class:`MelBanksOptions`.
        :param use_energy: Append an extra energy dimension to the filter banks.
        :param energy_floor: Absolute floor on energy (only with use_energy; needed when dither is 0).
        :param raw_energy: Compute energy before preemphasis/windowing (only with use_energy).
        :param htk_compat: Put energy last instead of first (only with use_energy).
        :param use_log_fbank: Produce log-filterbank (True) or linear (False).
        :param use_log_floor: Use kaldi's hard floor (True) or the additive NeMo guard (False).
        :param eps: Value of the log-zero guard (default FLT_EPSILON).
        :param use_power: Use power (True) or magnitude (False) in the analysis.
        """
    def __setstate__(self, arg0: dict) -> None:
        ...
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    def as_dict(self) -> dict:
        """
        Return the options as a dict.
        """
    @property
    def energy_floor(self) -> float:
        """
        Floor on energy (absolute, not relative).  Active only if ``use_energy`` is True; only necessary if ``frame_opts.dither`` is 0.  Suggested values: 0.1 or 1.0.
        """
    @energy_floor.setter
    def energy_floor(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def eps(self) -> float:
        """
        Value of the log-zero guard selected by ``use_log_floor``; defaults to ``FLT_EPSILON`` (NeMo models use ``FLT_EPSILON / 2``).
        """
    @eps.setter
    def eps(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frame_opts(self) -> FrameExtractionOptions:
        """
        Frame-extraction options; see :class:`FrameExtractionOptions`.
        """
    @frame_opts.setter
    def frame_opts(self, arg0: FrameExtractionOptions) -> None:
        ...
    @property
    def htk_compat(self) -> bool:
        """
        If True, put energy last instead of first.  Active only if ``use_energy`` is True.  Warning: not sufficient for HTK-compatible features (other parameters must change too).
        """
    @htk_compat.setter
    def htk_compat(self, arg0: bool) -> None:
        ...
    @property
    def mel_opts(self) -> MelBanksOptions:
        """
        Mel filter-bank options; see :class:`MelBanksOptions`.  The default has 23 mel bins.
        """
    @mel_opts.setter
    def mel_opts(self, arg0: MelBanksOptions) -> None:
        ...
    @property
    def raw_energy(self) -> bool:
        """
        If True, compute log energy before preemphasis and windowing; if False, after.  Active only if ``use_energy`` is True.
        """
    @raw_energy.setter
    def raw_energy(self, arg0: bool) -> None:
        ...
    @property
    def use_energy(self) -> bool:
        """
        If True, append an extra dimension with energy to the filter banks.
        """
    @use_energy.setter
    def use_energy(self, arg0: bool) -> None:
        ...
    @property
    def use_log_fbank(self) -> bool:
        """
        If True (default), produce log-filterbank; else linear.
        """
    @use_log_fbank.setter
    def use_log_fbank(self, arg0: bool) -> None:
        ...
    @property
    def use_log_floor(self) -> bool:
        """
        If True (default), use kaldi's hard floor ``log(max(e, eps))``; if False, the additive guard ``log(e + eps)`` used by NeMo models.
        """
    @use_log_floor.setter
    def use_log_floor(self, arg0: bool) -> None:
        ...
    @property
    def use_power(self) -> bool:
        """
        If True (default), use power in the filterbank analysis; else magnitude.
        """
    @use_power.setter
    def use_power(self, arg0: bool) -> None:
        ...
class FeatureWindowFunction:
    """
    Stores the windowing coefficients and applies them to a frame.
    """
    def __init__(self, opts: FrameExtractionOptions) -> None:
        """
        Construct the windowing function implied by ``opts.window_type``.
        
        The window has ``opts.WindowSize()`` samples.
        
        :param opts: Frame-extraction options selecting the window type and size.
        """
    def apply(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
        """
        Apply the window to a frame, in place.
        
        ``wave[i]`` becomes ``wave[i] * window[i]``.
        
        :param wave: One frame of ``window_size`` samples.
        :return: The windowed frame (the same list, modified in place).
        """
    @property
    def window(self) -> list[float]:
        """
        The window coefficients, a list of ``window_size`` floats.
        """
class FrameExtractionOptions:
    """
    Options for extracting frames from a waveform.
    
    This is the equivalent of kaldi's `FrameExtractionOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.h>`_.  It controls
    how a waveform is split into overlapping frames and how each frame is
    pre-processed (dithering, DC offset removal, preemphasis, windowing) before
    the FFT.
    
    .. seealso::
       :class:`FeatureWindowFunction`
          The windowing function implied by these options.
       :class:`kaldi_native_fbank.OnlineFbank`
          Streaming extractors that accept these options via ``frame_opts``.
    """
    @staticmethod
    def from_dict(arg0: dict) -> FrameExtractionOptions:
        """
        Build options from a dict.
        """
    def __getstate__(self) -> dict:
        ...
    def __init__(self, samp_freq: typing.SupportsFloat = 16000.0, frame_shift_ms: typing.SupportsFloat = 10.0, frame_length_ms: typing.SupportsFloat = 25.0, dither: typing.SupportsFloat = 0.00003, preemph_coeff: typing.SupportsFloat = 0.97, remove_dc_offset: bool = True, window_type: str = 'povey', round_to_power_of_two: bool = True, blackman_coeff: typing.SupportsFloat = 0.42, snip_edges: bool = True) -> None:
        """
        Construct with the given options.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param samp_freq: Waveform sample frequency in Hz (must match the waveform).
        :param frame_shift_ms: Frame shift in milliseconds.
        :param frame_length_ms: Frame length in milliseconds.
        :param dither: Amount of dithering; 0.0 disables it.  The default 0.00003 is equivalent to kaldi's 1.0.
        :param preemph_coeff: Preemphasis coefficient.
        :param remove_dc_offset: Subtract the mean of the waveform before the FFT.
        :param window_type: Window type, e.g. "povey"; see :attr:`window_type` for the formulas of each type.
        :param round_to_power_of_two: If True, round the FFT window size up to a power of two.
        :param blackman_coeff: Constant coefficient for the generalized Blackman window.
        :param snip_edges: If True, emit only frames that completely fit in the file; if False, reflect at the edges.
        """
    def __setstate__(self, arg0: dict) -> None:
        ...
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    def as_dict(self) -> dict:
        """
        Return the options as a dict.
        """
    @property
    def blackman_coeff(self) -> float:
        """
        Constant coefficient for the generalized Blackman window.
        """
    @blackman_coeff.setter
    def blackman_coeff(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def dither(self) -> float:
        """
        Amount of dithering; 0.0 means no dither.  The default 0.00003 is equivalent to dither=1.0 in kaldi.
        """
    @dither.setter
    def dither(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frame_length_ms(self) -> float:
        """
        Frame length in milliseconds.
        """
    @frame_length_ms.setter
    def frame_length_ms(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frame_shift_ms(self) -> float:
        """
        Frame shift in milliseconds.
        """
    @frame_shift_ms.setter
    def frame_shift_ms(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def preemph_coeff(self) -> float:
        """
        Preemphasis coefficient.
        """
    @preemph_coeff.setter
    def preemph_coeff(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def remove_dc_offset(self) -> bool:
        """
        Subtract mean of the waveform before the FFT.
        """
    @remove_dc_offset.setter
    def remove_dc_offset(self, arg0: bool) -> None:
        ...
    @property
    def round_to_power_of_two(self) -> bool:
        """
        If True, round the window size to a power of two by zero-padding the FFT input.
        """
    @round_to_power_of_two.setter
    def round_to_power_of_two(self, arg0: bool) -> None:
        ...
    @property
    def samp_freq(self) -> float:
        """
        Waveform data sample frequency (must match the waveform).
        """
    @samp_freq.setter
    def samp_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def snip_edges(self) -> bool:
        """
        If True, output only frames that completely fit in the file (the frame count depends on the frame length).  If False, the frame count depends only on the frame shift and the data is reflected at the edges.
        """
    @snip_edges.setter
    def snip_edges(self, arg0: bool) -> None:
        ...
    @property
    def window_type(self) -> str:
        """
        Window type: one of ``"hanning"``, ``"hann"``, ``"sine"``, ``"hamming"``, ``"povey"``, ``"rectangular"`` or ``"blackman"``.
        
        Let :math:`N` be the window size, :math:`n = 0, \\ldots, N-1` the sample index, and :math:`a = 2\\pi/(N-1)`.  The formulas are:
        
        * ``"hanning"``: :math:`w[n] = 0.5 - 0.5 \\cos(a n)`
        * ``"hann"``: :math:`w[n] = 0.5 - 0.5 \\cos(a n)` with :math:`a = 2\\pi/N`
          (periodic variant; see the hann_window docs of PyTorch)
        * ``"sine"``: :math:`w[n] = \\sin(0.5\\,a\\,n) = \\sin(\\pi n/(N-1))`
        * ``"hamming"``: :math:`w[n] = 0.54 - 0.46 \\cos(a n)`
        * ``"povey"``: :math:`w[n] = (0.5 - 0.5 \\cos(a n))^{0.85}`
          (like hamming but goes to zero at the edges)
        * ``"rectangular"``: :math:`w[n] = 1`
        * ``"blackman"``: :math:`w[n] = c - 0.5 \\cos(a n) + (0.5 - c) \\cos(2 a n)`
          where :math:`c` is ``blackman_coeff``.
        """
    @window_type.setter
    def window_type(self, arg0: str) -> None:
        ...
class IStft:
    """
    Reconstruct a waveform from :class:`Stft` results.
    
    This is the inverse of :class:`Stft` (compatible with librosa's istft()):
    the frames are inverse-transformed, windowed, overlap-added, and divided by
    the summed window envelope.
    
    .. seealso:: :class:`Stft`, :class:`StftConfig`
    """
    def __call__(self, stft_result: StftResult) -> list[float]:
        """
        Same as :meth:`compute`.
        """
    def __init__(self, config: StftConfig) -> None:
        """
        Construct with the same config that was used for the forward :class:`Stft`.
        
        :param config: The config used for the forward transform.
        """
    def compute(self, stft_result: StftResult) -> list[float]:
        """
        Reconstruct the waveform.
        
        :param stft_result: Output of :meth:`Stft.compute`.
        :return: 1-D list of samples.  If ``config.center`` is True, the
           ``n_fft // 2`` padding samples at both ends are removed.
        """
class MelBanks:
    """
    Triangular mel filter banks.
    
    A ``MelBanks`` object holds the weights of a set of triangular mel-frequency
    bins and can project FFT bin energies onto them.
    
    .. seealso:: :class:`MelBanksOptions`
    """
    @staticmethod
    def inverse_mel_scale(mel: typing.SupportsFloat) -> float:
        """
        HTK mel scale: convert mel frequency to Hz.
        
        .. math::
        
           \\mathrm{Hz} = 700 \\left( \\exp(\\mathrm{mel} / 1127) - 1 \\right)
        
        :param mel: Mel frequency.
        :return: The frequency in Hz.
        """
    @staticmethod
    def inverse_mel_scale_slaney(mel: typing.SupportsFloat) -> float:
        """
        Slaney mel scale: convert mel frequency to Hz.
        
        .. math::
        
           \\mathrm{Hz} = \\left\\{ \\begin{array}{ll}
           (200/3)\\,\\mathrm{mel} & \\mathrm{mel} \\le 15 \\\\
           1000 \\exp((\\mathrm{mel} - 15) \\log(6.4) / 27) & \\mathrm{otherwise}
           \\end{array} \\right.
        
        :param mel: Mel frequency (Slaney scale).
        :return: The frequency in Hz.
        """
    @staticmethod
    def mel_scale(hz: typing.SupportsFloat) -> float:
        """
        HTK mel scale: convert Hz to mel frequency.
        
        .. math::
        
           \\mathrm{mel} = 1127 \\log(1 + \\mathrm{Hz} / 700)
        
        :param hz: The frequency in Hz.
        :return: The mel frequency.
        """
    @staticmethod
    def mel_scale_slaney(hz: typing.SupportsFloat) -> float:
        """
        Slaney mel scale: convert Hz to mel frequency.
        
        .. math::
        
           \\mathrm{mel} = \\left\\{ \\begin{array}{ll}
           (3/200)\\,\\mathrm{Hz} & \\mathrm{Hz} \\le 1000 \\\\
           15 + 27 \\log(\\mathrm{Hz} / 1000) / \\log(6.4) & \\mathrm{otherwise}
           \\end{array} \\right.
        
        :param hz: The frequency in Hz.
        :return: The mel frequency (Slaney scale).
        """
    def __init__(self, opts: MelBanksOptions = ..., frame_opts: FrameExtractionOptions = ..., vtln_warp_factor: typing.SupportsFloat = 1.0) -> None:
        """
        Construct kaldi-compatible (or librosa-style) mel banks.
        
        The mel bin edges are placed uniformly on the mel scale between
        ``opts.low_freq`` and ``opts.high_freq``; each bin is a triangle in the
        frequency domain.  If ``opts.is_librosa`` is true, librosa-style filters are
        computed instead.
        
        :param opts: Mel-bin configuration; see :class:`MelBanksOptions`.
        :param frame_opts: Frame-extraction configuration (provides ``samp_freq``
           and the FFT size).
        :param vtln_warp_factor: VTLN warping factor (1.0 means no warping).
        """
    def compute(self, fft_energies: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]:
        """
        Compute mel energies (note: NOT log) from a power spectrum.
        
        For each mel bin :math:`i`, the output is the weighted sum of the FFT bin
        energies that fall into that bin:
        
        .. math::
        
           \\mathrm{mel}[i] = \\sum_{k} w[i][k] \\cdot
           \\mathrm{fft\\_energies}[\\mathrm{offset}[i] + k]
        
        where :math:`\\mathrm{offset}[i]` is the first nonzero fft bin of mel bin
        :math:`i` and :math:`w[i]` are its triangle weights.
        
        :param fft_energies: 1-D float32 array of size ``num_fft_bins // 2 + 1``
           holding the power spectrum (not log).  Must be C-contiguous.
        :return: 1-D float32 array of size ``num_mel_bins`` with the mel energies.
        """
    def get_matrix(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return the mel filter weights.
        
        :return: A float32 array of shape ``(num_mel_bins, num_fft_bins // 2 + 1)``;
           ``weights[i, k]`` is the weight of fft bin :math:`k` for mel bin :math:`i`.
        """
    @property
    def dim(self) -> int:
        ...
class MelBanksOptions:
    """
    Options for constructing triangular mel-filterbank bins.
    
    The kaldi-compatible fields correspond to kaldi's `MelBanksOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/mel-computations.h>`_; the
    ``is_librosa``-related fields select librosa-compatible filters instead.
    
    .. seealso:: :class:`MelBanks`
    """
    @staticmethod
    def from_dict(arg0: dict) -> MelBanksOptions:
        """
        Build options from a dict.
        """
    def __getstate__(self) -> dict:
        ...
    def __init__(self, num_bins: typing.SupportsInt = 25, low_freq: typing.SupportsFloat = 20, high_freq: typing.SupportsFloat = 0, vtln_low: typing.SupportsFloat = 100, vtln_high: typing.SupportsFloat = -500, debug_mel: bool = False, htk_mode: bool = False, is_librosa: bool = False, norm: str = 'slaney', use_slaney_mel_scale: bool = True, floor_to_int_bin: bool = False) -> None:
        """
        Construct with the given options.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param num_bins: Number of triangular mel-frequency bins.
        :param low_freq: Lower frequency cutoff.
        :param high_freq: Upper cutoff; 0 means none, negative values are offsets from Nyquist.
        :param vtln_low: VTLN lower cutoff of the warping function.
        :param vtln_high: VTLN upper cutoff of the warping function; negative values are offsets from Nyquist.
        :param debug_mel: Print debugging information for mel bin computation.
        :param htk_mode: Exact HTK compatibility (affects mel-energy flooring).
        :param is_librosa: Use librosa-style mel filters.
        :param norm: Librosa filter normalization ("" or "slaney").
        :param use_slaney_mel_scale: Use the Slaney mel scale (librosa mode only).
        :param floor_to_int_bin: Floor the bin edges to integers (librosa mode only).
        """
    def __setstate__(self, arg0: dict) -> None:
        ...
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    def as_dict(self) -> dict:
        """
        Return the options as a dict.
        """
    @property
    def debug_mel(self) -> bool:
        """
        Print debugging information for mel bin computation.
        """
    @debug_mel.setter
    def debug_mel(self, arg0: bool) -> None:
        ...
    @property
    def floor_to_int_bin(self) -> bool:
        """
        Floor the bin edges to integers (used only when is_librosa=True).
        """
    @floor_to_int_bin.setter
    def floor_to_int_bin(self, arg0: bool) -> None:
        ...
    @property
    def high_freq(self) -> float:
        """
        Upper frequency cutoff; 0 means no cutoff, negative values are offsets from Nyquist.
        """
    @high_freq.setter
    def high_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def htk_mode(self) -> bool:
        """
        Enable exact HTK compatibility (affects mel-energy flooring).
        """
    @htk_mode.setter
    def htk_mode(self, arg0: bool) -> None:
        ...
    @property
    def is_librosa(self) -> bool:
        """
        Use librosa-style mel filters.
        
        If True, you probably want :attr:`low_freq` to be 0.  See the librosa
        mel filter documentation.
        """
    @is_librosa.setter
    def is_librosa(self, arg0: bool) -> None:
        ...
    @property
    def low_freq(self) -> float:
        """
        Lower frequency cutoff of the mel bins.
        
        See also :attr:`high_freq`.
        """
    @low_freq.setter
    def low_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def norm(self) -> str:
        """
        Librosa filter normalization: "" or "slaney" (used only when is_librosa=True).
        """
    @norm.setter
    def norm(self, arg0: str) -> None:
        ...
    @property
    def num_bins(self) -> int:
        """
        Number of triangular mel-frequency bins.
        """
    @num_bins.setter
    def num_bins(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def use_slaney_mel_scale(self) -> bool:
        """
        Use the Slaney mel scale (used only when is_librosa=True).
        """
    @use_slaney_mel_scale.setter
    def use_slaney_mel_scale(self, arg0: bool) -> None:
        ...
    @property
    def vtln_high(self) -> float:
        """
        VTLN upper cutoff of the warping function; negative values are offsets from Nyquist.
        """
    @vtln_high.setter
    def vtln_high(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def vtln_low(self) -> float:
        """
        VTLN lower cutoff of the warping function.
        """
    @vtln_low.setter
    def vtln_low(self, arg0: typing.SupportsFloat) -> None:
        ...
class MfccOptions:
    """
    Options for computing MFCC features.
    
    This is kaldi's `MfccOptions <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-mfcc.h>`_.
    
    .. seealso::
       :class:`FrameExtractionOptions`
          Frame extraction settings (``frame_opts``).
       :class:`MelBanksOptions`
          Mel filter-bank settings (``mel_opts``).
       :class:`kaldi_native_fbank.OnlineMfcc`
          The streaming extractor that consumes these options.
    """
    @staticmethod
    def from_dict(arg0: dict) -> MfccOptions:
        """
        Build options from a dict.
        """
    def __getstate__(self) -> dict:
        ...
    def __init__(self, frame_opts: FrameExtractionOptions = ..., mel_opts: MelBanksOptions = ..., num_ceps: typing.SupportsInt = 13, use_energy: bool = True, energy_floor: typing.SupportsFloat = 0, raw_energy: bool = True, cepstral_lifter: typing.SupportsFloat = 22, htk_compat: bool = False) -> None:
        """
        Construct with the given options.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param frame_opts: Frame-extraction options; see :class:`FrameExtractionOptions`.
        :param mel_opts: Mel filter-bank options (default 23 bins); see :class:`MelBanksOptions`.
        :param num_ceps: Number of cepstra in the MFCC computation (including C0).
        :param use_energy: Use energy instead of C0.
        :param energy_floor: Absolute floor on energy (only with use_energy; needed when dither is 0).
        :param raw_energy: Compute energy before preemphasis/windowing.
        :param cepstral_lifter: Liftering constant; 0 disables liftering.
        :param htk_compat: Put energy/C0 last and use a factor of sqrt(2) on C0.
        """
    def __setstate__(self, arg0: dict) -> None:
        ...
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    def as_dict(self) -> dict:
        """
        Return the options as a dict.
        """
    @property
    def cepstral_lifter(self) -> float:
        """
        Constant that controls scaling of the MFCCs (liftering).  If 0.0, no liftering is done.
        """
    @cepstral_lifter.setter
    def cepstral_lifter(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def energy_floor(self) -> float:
        """
        Floor on energy (absolute, not relative).  Only makes a difference if ``use_energy`` is True; only necessary if ``frame_opts.dither`` is 0.  Suggested values: 0.1 or 1.0.
        """
    @energy_floor.setter
    def energy_floor(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frame_opts(self) -> FrameExtractionOptions:
        """
        Frame-extraction options; see :class:`FrameExtractionOptions`.
        """
    @frame_opts.setter
    def frame_opts(self, arg0: FrameExtractionOptions) -> None:
        ...
    @property
    def htk_compat(self) -> bool:
        """
        If True, put energy/C0 last and use a factor of sqrt(2) on C0.  Warning: not sufficient for HTK-compatible features (other parameters must change too).
        """
    @htk_compat.setter
    def htk_compat(self, arg0: bool) -> None:
        ...
    @property
    def mel_opts(self) -> MelBanksOptions:
        """
        Mel filter-bank options; see :class:`MelBanksOptions`.  The default has 23 mel bins.
        """
    @mel_opts.setter
    def mel_opts(self, arg0: MelBanksOptions) -> None:
        ...
    @property
    def num_ceps(self) -> int:
        """
        Number of cepstra in the MFCC computation (including C0).
        """
    @num_ceps.setter
    def num_ceps(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def raw_energy(self) -> bool:
        """
        If True, compute energy before preemphasis and windowing.
        """
    @raw_energy.setter
    def raw_energy(self, arg0: bool) -> None:
        ...
    @property
    def use_energy(self) -> bool:
        """
        Use energy (not C0) in the MFCC computation.
        """
    @use_energy.setter
    def use_energy(self, arg0: bool) -> None:
        ...
class NemoFbankOptions:
    """
    Options for :class:`OnlineNemoFbank`.
    
    .. seealso:: :class:`OnlineNemoFbank`
    """
    @staticmethod
    def from_dict(arg0: dict) -> NemoFbankOptions:
        """
        Build options from a dict.
        """
    def __getstate__(self) -> dict:
        ...
    def __init__(self, samp_freq: typing.SupportsFloat = 16000.0, frame_shift_ms: typing.SupportsFloat = 10.0, frame_length_ms: typing.SupportsFloat = 25.0, num_mel_bins: typing.SupportsInt = 80, low_freq: typing.SupportsFloat = 0, high_freq: typing.SupportsFloat = 0, preemph_coeff: typing.SupportsFloat = 0.97, buffered_capacity: typing.SupportsInt = 0) -> None:
        """
        Construct with the given options.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param samp_freq: Waveform sample frequency.
        :param frame_shift_ms: Frame shift in milliseconds.
        :param frame_length_ms: Frame length in milliseconds.
        :param num_mel_bins: Number of mel bins.
        :param low_freq: Lower frequency cutoff for the mel bins.
        :param high_freq: Upper cutoff; non-positive values are offsets from Nyquist.
        :param preemph_coeff: Preemphasis coefficient.
        :param buffered_capacity: Full-buffer length in frames; 0 disables the final sample-length cap.
        """
    def __setstate__(self, arg0: dict) -> None:
        ...
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    def as_dict(self) -> dict:
        """
        Return the options as a dict.
        """
    @property
    def buffered_capacity(self) -> int:
        """
        Full-buffer length in frames.  0 disables the final sample-length cap.
        """
    @buffered_capacity.setter
    def buffered_capacity(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def frame_length_ms(self) -> float:
        """
        Frame length in milliseconds.
        """
    @frame_length_ms.setter
    def frame_length_ms(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frame_shift_ms(self) -> float:
        """
        Frame shift in milliseconds.
        """
    @frame_shift_ms.setter
    def frame_shift_ms(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def high_freq(self) -> float:
        """
        Upper frequency cutoff; non-positive values are offsets from Nyquist.
        """
    @high_freq.setter
    def high_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def low_freq(self) -> float:
        """
        Lower frequency cutoff for the mel bins.
        """
    @low_freq.setter
    def low_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def num_mel_bins(self) -> int:
        """
        Number of mel bins.
        """
    @num_mel_bins.setter
    def num_mel_bins(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def preemph_coeff(self) -> float:
        """
        Preemphasis coefficient.
        """
    @preemph_coeff.setter
    def preemph_coeff(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def samp_freq(self) -> float:
        """
        Waveform sample frequency.
        """
    @samp_freq.setter
    def samp_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
class OnlineFbank:
    """
    """
    def __init__(self, opts: FbankOptions) -> None:
        """
        Construct from the options of the wrapped feature computer.
        
        :param opts: Options of the computer, e.g. :class:`FbankOptions` for
           :class:`kaldi_native_fbank.OnlineFbank`.
        """
    def accept_waveform(self, sampling_rate: typing.SupportsFloat, waveform: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Add more waveform samples.
        
        :param sampling_rate: The sampling rate of ``waveform``; must match the
           rate in the options.
        :param waveform: 1-D float32 samples.
        """
    def get_frame(self, frame: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return the ``frame``-th feature vector.
        
        :param frame: Frame index.
        :return: np.ndarray of shape ``(dim,)`` and dtype float32.  The buffer is
           owned by the extractor.
        """
    def input_finished(self) -> None:
        """
        Tell the extractor that no more waveform will be added.
        
        This flushes out the last frame or two when ``snip_edges`` is False and
        also affects the return value of :meth:`is_last_frame`.
        """
    def is_last_frame(self, frame: typing.SupportsInt) -> bool:
        """
        Return true if ``frame`` is the last frame.
        
        Note: this can only return true after :meth:`input_finished` has been
        called.
        
        :param frame: Frame index.
        :return: True if the frame is the last one.
        """
    def pop(self, n: typing.SupportsInt) -> None:
        """
        Discard the first ``n`` frames.
        
        :param n: Number of frames to discard.
        """
    @property
    def dim(self) -> int:
        """
        The dimension of the output features.
        """
    @property
    def frame_shift_in_seconds(self) -> float:
        """
        The frame shift in seconds.
        """
    @property
    def num_frames_ready(self) -> int:
        """
        The number of frames that have already been computed.
        """
class OnlineMfcc:
    """
    """
    def __init__(self, opts: MfccOptions) -> None:
        """
        Construct from the options of the wrapped feature computer.
        
        :param opts: Options of the computer, e.g. :class:`FbankOptions` for
           :class:`kaldi_native_fbank.OnlineFbank`.
        """
    def accept_waveform(self, sampling_rate: typing.SupportsFloat, waveform: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Add more waveform samples.
        
        :param sampling_rate: The sampling rate of ``waveform``; must match the
           rate in the options.
        :param waveform: 1-D float32 samples.
        """
    def get_frame(self, frame: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return the ``frame``-th feature vector.
        
        :param frame: Frame index.
        :return: np.ndarray of shape ``(dim,)`` and dtype float32.  The buffer is
           owned by the extractor.
        """
    def input_finished(self) -> None:
        """
        Tell the extractor that no more waveform will be added.
        
        This flushes out the last frame or two when ``snip_edges`` is False and
        also affects the return value of :meth:`is_last_frame`.
        """
    def is_last_frame(self, frame: typing.SupportsInt) -> bool:
        """
        Return true if ``frame`` is the last frame.
        
        Note: this can only return true after :meth:`input_finished` has been
        called.
        
        :param frame: Frame index.
        :return: True if the frame is the last one.
        """
    def pop(self, n: typing.SupportsInt) -> None:
        """
        Discard the first ``n`` frames.
        
        :param n: Number of frames to discard.
        """
    @property
    def dim(self) -> int:
        """
        The dimension of the output features.
        """
    @property
    def frame_shift_in_seconds(self) -> float:
        """
        The frame shift in seconds.
        """
    @property
    def num_frames_ready(self) -> int:
        """
        The number of frames that have already been computed.
        """
class OnlineNemoFbank:
    """
    NeMo-compatible buffered streaming filter-bank features.
    
    NeMo buffered streaming re-featurizes each requested audio window:
    preemphasis over that window, then a centered, constant-zero-padded STFT, a
    symmetric Hann window, Slaney mel filters, and
    :math:`\\log(\\mathrm{mel} + 2^{-24})`.  Input samples are floats in
    ``[-1, 1]``.  This class does not resample or normalize.
    
    .. seealso:: :class:`NemoFbankOptions`, :class:`kaldi_native_fbank.OnlineFbank`
    """
    def __init__(self, opts: NemoFbankOptions) -> None:
        """
        Construct from the given options.
        
        :param opts: Options; see :class:`NemoFbankOptions`.
        """
    def accept_waveform(self, sampling_rate: typing.SupportsFloat, waveform: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Add more waveform samples.
        
        :param sampling_rate: The sampling rate of ``waveform``; must match
           ``opts.samp_freq``.
        :param waveform: 1-D float32 samples in ``[-1, 1]``.
        """
    def get_frames(self, frame_index: typing.SupportsInt, n: typing.SupportsInt, is_final: bool = False) -> tuple:
        """
        Return ``n`` frames starting at the absolute ``frame_index``.
        
        Samples preceding the requested window are discarded; later calls cannot
        rewind.  If ``is_final`` reaches the last ready frame after
        :meth:`input_finished`, the partial hop is included.  When ``n`` equals
        ``buffered_capacity``, the window is capped at ``n`` hops by shifting its
        start by the remainder; that shift (in samples) is reported as the second
        element of the returned tuple.
        
        :param frame_index: Absolute index of the first frame to return.
        :param n: Number of frames.
        :param is_final: True if this is the final request.
        :return: A tuple ``(frames, start_sample_offset)`` where ``frames`` is a
           float32 array of shape ``(n, dim)`` and ``start_sample_offset`` is 0
           unless the buffer cap shifted the window.
        """
    def input_finished(self) -> None:
        """
        Tell the extractor that no more waveform will be added.
        """
    def is_last_frame(self, frame: typing.SupportsInt) -> bool:
        """
        Return true if frame is the last frame; only after input_finished().
        """
    @property
    def dim(self) -> int:
        """
        The feature dimension (number of mel bins).
        """
    @property
    def frame_shift_in_seconds(self) -> float:
        """
        The frame shift in seconds.
        """
    @property
    def num_frames_ready(self) -> int:
        """
        The number of whole frames that are ready.
        """
    @property
    def num_trailing_samples(self) -> int:
        """
        Buffered samples that do not make up a whole hop.
        """
class OnlinePitch:
    """
    Streaming pitch extraction with post-processing.
    
    This is the streaming interface of kaldi's
    `compute-and-process-kaldi-pitch-feats
    <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc>`_
    (kaldi's ``OnlinePitchFeature`` + ``OnlineProcessPitch``): it consumes
    waveform chunks and produces the processed pitch features — by default the
    3-dimensional (POV, normalized-log-pitch, delta-pitch).
    
    Usage mirrors :class:`kaldi_native_fbank.OnlineMfcc`: feed the waveform with
    :meth:`accept_waveform`, call :meth:`input_finished` when no more data is
    coming, then read frames with :meth:`get_frame`.
    
    .. seealso:: :class:`PitchExtractionOptions`, :class:`ProcessPitchOptions`,
       :func:`compute_and_process_kaldi_pitch`
    """
    def __init__(self, opts: PitchExtractionOptions, popts: ProcessPitchOptions = ...) -> None:
        """
        Construct with the given extraction and post-processing options.
        
        :param opts: Raw pitch extraction options; see :class:`PitchExtractionOptions`.
        :param popts: Post-processing options; see :class:`ProcessPitchOptions`.
           At least one of the ``add_*`` features must be enabled.
        """
    def accept_waveform(self, sampling_rate: typing.SupportsFloat, waveform: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Add more waveform samples.
        
        :param sampling_rate: The sampling rate of ``waveform``; must match
           ``opts.samp_freq``.
        :param waveform: 1-D float32 samples (kaldi semantics: raw WAV magnitudes,
           not divided by 32768).
        """
    def get_frame(self, frame: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return the ``frame``-th feature vector.
        
        :param frame: Frame index, must be less than ``num_frames_ready``.
        :return: np.ndarray of shape ``(dim,)`` and dtype float32.
        """
    def input_finished(self) -> None:
        """
        Tell the extractor that no more waveform will be added.
        
        This flushes out the last frames and fixes the features of all frames.
        """
    def is_last_frame(self, frame: typing.SupportsInt) -> bool:
        """
        Return true if ``frame`` is the last frame.
        
        Note: this can only return true after :meth:`input_finished` has been
        called.
        
        :param frame: Frame index.
        :return: True if the frame is the last one.
        """
    @property
    def dim(self) -> int:
        """
        The dimension of the output features (3 by default).
        """
    @property
    def frame_shift_in_seconds(self) -> float:
        """
        The frame shift in seconds.
        """
    @property
    def num_frames_ready(self) -> int:
        """
        The number of frames that have already been computed.
        """
class OnlineRawAudioSamples:
    """
    """
    def __init__(self, opts: RawAudioSamplesOptions) -> None:
        """
        Construct from the options of the wrapped feature computer.
        
        :param opts: Options of the computer, e.g. :class:`FbankOptions` for
           :class:`kaldi_native_fbank.OnlineFbank`.
        """
    def accept_waveform(self, sampling_rate: typing.SupportsFloat, waveform: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Add more waveform samples.
        
        :param sampling_rate: The sampling rate of ``waveform``; must match the
           rate in the options.
        :param waveform: 1-D float32 samples.
        """
    def get_frame(self, frame: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return the ``frame``-th feature vector.
        
        :param frame: Frame index.
        :return: np.ndarray of shape ``(dim,)`` and dtype float32.  The buffer is
           owned by the extractor.
        """
    def input_finished(self) -> None:
        """
        Tell the extractor that no more waveform will be added.
        
        This flushes out the last frame or two when ``snip_edges`` is False and
        also affects the return value of :meth:`is_last_frame`.
        """
    def is_last_frame(self, frame: typing.SupportsInt) -> bool:
        """
        Return true if ``frame`` is the last frame.
        
        Note: this can only return true after :meth:`input_finished` has been
        called.
        
        :param frame: Frame index.
        :return: True if the frame is the last one.
        """
    def pop(self, n: typing.SupportsInt) -> None:
        """
        Discard the first ``n`` frames.
        
        :param n: Number of frames to discard.
        """
    @property
    def dim(self) -> int:
        """
        The dimension of the output features.
        """
    @property
    def frame_shift_in_seconds(self) -> float:
        """
        The frame shift in seconds.
        """
    @property
    def num_frames_ready(self) -> int:
        """
        The number of frames that have already been computed.
        """
class OnlineWhisperFbank:
    """
    """
    def __init__(self, opts: WhisperFeatureOptions) -> None:
        """
        Construct from the options of the wrapped feature computer.
        
        :param opts: Options of the computer, e.g. :class:`FbankOptions` for
           :class:`kaldi_native_fbank.OnlineFbank`.
        """
    def accept_waveform(self, sampling_rate: typing.SupportsFloat, waveform: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Add more waveform samples.
        
        :param sampling_rate: The sampling rate of ``waveform``; must match the
           rate in the options.
        :param waveform: 1-D float32 samples.
        """
    def get_frame(self, frame: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float32]:
        """
        Return the ``frame``-th feature vector.
        
        :param frame: Frame index.
        :return: np.ndarray of shape ``(dim,)`` and dtype float32.  The buffer is
           owned by the extractor.
        """
    def input_finished(self) -> None:
        """
        Tell the extractor that no more waveform will be added.
        
        This flushes out the last frame or two when ``snip_edges`` is False and
        also affects the return value of :meth:`is_last_frame`.
        """
    def is_last_frame(self, frame: typing.SupportsInt) -> bool:
        """
        Return true if ``frame`` is the last frame.
        
        Note: this can only return true after :meth:`input_finished` has been
        called.
        
        :param frame: Frame index.
        :return: True if the frame is the last one.
        """
    def pop(self, n: typing.SupportsInt) -> None:
        """
        Discard the first ``n`` frames.
        
        :param n: Number of frames to discard.
        """
    @property
    def dim(self) -> int:
        """
        The dimension of the output features.
        """
    @property
    def frame_shift_in_seconds(self) -> float:
        """
        The frame shift in seconds.
        """
    @property
    def num_frames_ready(self) -> int:
        """
        The number of frames that have already been computed.
        """
class PitchExtractionOptions:
    """
    Options for the raw pitch extraction stage.
    
    Mirrors the flags of kaldi's `compute-kaldi-pitch-feats
    <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-kaldi-pitch-feats.cc>`_
    and kaldi's `PitchExtractionOptions
    <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/pitch-functions.h>`_.
    
    .. seealso:: :class:`ProcessPitchOptions`, :class:`OnlinePitch`,
       :func:`compute_and_process_kaldi_pitch`
    """
    def __init__(self, samp_freq: typing.SupportsFloat = 16000, frame_shift_ms: typing.SupportsFloat = 10.0, frame_length_ms: typing.SupportsFloat = 25.0, preemph_coeff: typing.SupportsFloat = 0.0, min_f0: typing.SupportsFloat = 50, max_f0: typing.SupportsFloat = 400, soft_min_f0: typing.SupportsFloat = 10.0, penalty_factor: typing.SupportsFloat = 0.1, lowpass_cutoff: typing.SupportsFloat = 1000, resample_freq: typing.SupportsFloat = 4000, delta_pitch: typing.SupportsFloat = 0.005, nccf_ballast: typing.SupportsFloat = 7000, lowpass_filter_width: typing.SupportsInt = 1, upsample_filter_width: typing.SupportsInt = 5, max_frames_latency: typing.SupportsInt = 0, frames_per_chunk: typing.SupportsInt = 0, simulate_first_pass_online: bool = False, recompute_frame: typing.SupportsInt = 500, nccf_ballast_online: bool = False, snip_edges: bool = True) -> None:
        """
        Construct with the given options.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature (they are kaldi's defaults).
        
        :param samp_freq: Waveform sample frequency in Hz (kaldi ``--sample-frequency``).
        :param frame_shift_ms: Frame shift in milliseconds (kaldi ``--frame-shift``).
        :param frame_length_ms: Frame length in milliseconds (kaldi ``--frame-length``).
        :param preemph_coeff: Preemphasis coefficient; deprecated in kaldi
           (kaldi ``--preemphasis-coefficient``).
        :param min_f0: Min F0 to search for, in Hz (kaldi ``--min-f0``).
        :param max_f0: Max F0 to search for, in Hz (kaldi ``--max-f0``).
        :param soft_min_f0: Minimum f0 applied in a soft way; must not exceed
           ``min_f0`` (kaldi ``--soft-min-f0``).
        :param penalty_factor: Cost factor for F0 change (kaldi ``--penalty-factor``).
        :param lowpass_cutoff: Low-pass filter cutoff in Hz (kaldi ``--lowpass-cutoff``).
        :param resample_freq: Frequency the signal is downsampled to; must be more
           than twice ``lowpass_cutoff`` (kaldi ``--resample-frequency``).
        :param delta_pitch: The pitch tolerance in pruning lags (kaldi ``--delta-pitch``).
        :param nccf_ballast: Reduces NCCF for quiet frames when increased
           (kaldi ``--nccf-ballast``).
        :param lowpass_filter_width: Low-pass filter width (kaldi ``--lowpass-filter-width``).
        :param upsample_filter_width: NCCF upsampling filter width
           (kaldi ``--upsample-filter-width``).
        :param max_frames_latency: Maximum frames of latency allowed for pitch
           tracking (kaldi ``--max-frames-latency``).
        :param frames_per_chunk: Chunk size for :func:`compute_and_process_kaldi_pitch`
           (kaldi ``--frames-per-chunk``).
        :param simulate_first_pass_online: Output first-pass online features in
           :func:`compute_and_process_kaldi_pitch`
           (kaldi ``--simulate-first-pass-online``).
        :param recompute_frame: Frame at which online NCCF recomputation happens
           (kaldi ``--recompute-frame``).
        :param nccf_ballast_online: Debug flag for how the NCCF ballast is computed
           (kaldi ``--nccf-ballast-online``).
        :param snip_edges: If false, incomplete frames near the ending edge are not
           snipped (kaldi ``--snip-edges``).  Note: unlike other feature types,
           kaldi's pitch extractor defaults this to true.
        """
    @property
    def delta_pitch(self) -> float:
        """
        The pitch tolerance in pruning lags.
        """
    @delta_pitch.setter
    def delta_pitch(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frame_length_ms(self) -> float:
        """
        Frame length in milliseconds.
        """
    @frame_length_ms.setter
    def frame_length_ms(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frame_shift_ms(self) -> float:
        """
        Frame shift in milliseconds.
        """
    @frame_shift_ms.setter
    def frame_shift_ms(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def frames_per_chunk(self) -> int:
        """
        Chunk size for :func:`compute_and_process_kaldi_pitch`.
        """
    @frames_per_chunk.setter
    def frames_per_chunk(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def lowpass_cutoff(self) -> float:
        """
        Low-pass filter cutoff in Hz.
        """
    @lowpass_cutoff.setter
    def lowpass_cutoff(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def lowpass_filter_width(self) -> int:
        """
        Low-pass filter width.
        """
    @lowpass_filter_width.setter
    def lowpass_filter_width(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_f0(self) -> float:
        """
        Max F0 to search for, in Hz.
        """
    @max_f0.setter
    def max_f0(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def max_frames_latency(self) -> int:
        """
        Maximum frames of latency allowed for pitch tracking.
        """
    @max_frames_latency.setter
    def max_frames_latency(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def min_f0(self) -> float:
        """
        Min F0 to search for, in Hz.
        """
    @min_f0.setter
    def min_f0(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def nccf_ballast(self) -> float:
        """
        Reduces NCCF for quiet frames when increased.
        """
    @nccf_ballast.setter
    def nccf_ballast(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def nccf_ballast_online(self) -> bool:
        """
        Debug flag for how the NCCF ballast is computed.
        """
    @nccf_ballast_online.setter
    def nccf_ballast_online(self, arg0: bool) -> None:
        ...
    @property
    def nccf_window_shift(self) -> int:
        """
        The NCCF window shift in samples after resampling.
        """
    @property
    def nccf_window_size(self) -> int:
        """
        The NCCF window size in samples after resampling.
        """
    @property
    def penalty_factor(self) -> float:
        """
        Cost factor for F0 change.
        """
    @penalty_factor.setter
    def penalty_factor(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def preemph_coeff(self) -> float:
        """
        Preemphasis coefficient (deprecated in kaldi).
        """
    @preemph_coeff.setter
    def preemph_coeff(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def recompute_frame(self) -> int:
        """
        Frame at which online NCCF recomputation happens.
        """
    @recompute_frame.setter
    def recompute_frame(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def resample_freq(self) -> float:
        """
        Frequency the signal is downsampled to.
        """
    @resample_freq.setter
    def resample_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def samp_freq(self) -> float:
        """
        Waveform sample frequency in Hz.
        """
    @samp_freq.setter
    def samp_freq(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def simulate_first_pass_online(self) -> bool:
        """
        Output first-pass online features (batch mode only).
        """
    @simulate_first_pass_online.setter
    def simulate_first_pass_online(self, arg0: bool) -> None:
        ...
    @property
    def snip_edges(self) -> bool:
        """
        If false, incomplete frames near the ending edge are not snipped.
        """
    @snip_edges.setter
    def snip_edges(self, arg0: bool) -> None:
        ...
    @property
    def soft_min_f0(self) -> float:
        """
        Minimum f0 applied in a soft way.
        """
    @soft_min_f0.setter
    def soft_min_f0(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def upsample_filter_width(self) -> int:
        """
        NCCF upsampling filter width.
        """
    @upsample_filter_width.setter
    def upsample_filter_width(self, arg0: typing.SupportsInt) -> None:
        ...
class ProcessPitchOptions:
    """
    Options for the pitch post-processing stage.
    
    Determines which features are produced from the raw (NCCF, pitch) pair.
    With the defaults each frame is the 3-dimensional
    (POV-feature, normalized-log-pitch, delta-log-pitch) output of kaldi's
    `compute-and-process-kaldi-pitch-feats
    <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc>`_;
    see kaldi's `ProcessPitchOptions
    <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/pitch-functions.h>`_.
    
    .. seealso:: :class:`PitchExtractionOptions`, :class:`OnlinePitch`
    """
    def __init__(self, pitch_scale: typing.SupportsFloat = 2.0, pov_scale: typing.SupportsFloat = 2.0, pov_offset: typing.SupportsFloat = 0.0, delta_pitch_scale: typing.SupportsFloat = 10.0, delta_pitch_noise_stddev: typing.SupportsFloat = 0.005, normalization_left_context: typing.SupportsInt = 75, normalization_right_context: typing.SupportsInt = 75, delta_window: typing.SupportsInt = 2, delay: typing.SupportsInt = 0, add_pov_feature: bool = True, add_normalized_log_pitch: bool = True, add_delta_pitch: bool = True, add_raw_log_pitch: bool = False) -> None:
        """
        Construct with the given options.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature (they are kaldi's defaults).
        
        :param pitch_scale: Scaling of the normalized log-pitch feature
           (kaldi ``--pitch-scale``).
        :param pov_scale: Scaling of the POV feature (kaldi ``--pov-scale``).
        :param pov_offset: Offset added to the POV feature (kaldi ``--pov-offset``).
        :param delta_pitch_scale: Scaling of the delta log-pitch feature
           (kaldi ``--delta-pitch-scale``).
        :param delta_pitch_noise_stddev: Stddev of the noise added to the delta
           log-pitch (kaldi ``--delta-pitch-noise-stddev``).
        :param normalization_left_context: Left context (frames) for log-pitch
           normalization (kaldi ``--normalization-left-context``).
        :param normalization_right_context: Right context (frames) for log-pitch
           normalization (kaldi ``--normalization-right-context``).
        :param delta_window: Frames on each side used for the delta window
           (kaldi ``--delta-window``).
        :param delay: Frames by which the pitch information is delayed
           (kaldi ``--delay``).
        :param add_pov_feature: Add the warped NCCF to the output (1 dim).
        :param add_normalized_log_pitch: Add the mean-subtracted log-pitch (1 dim).
        :param add_delta_pitch: Add the time derivative of log-pitch (1 dim).
        :param add_raw_log_pitch: Add log(pitch) (1 dim).
        """
    @property
    def add_delta_pitch(self) -> bool:
        """
        Add the time derivative of log-pitch (1 dim).
        """
    @add_delta_pitch.setter
    def add_delta_pitch(self, arg0: bool) -> None:
        ...
    @property
    def add_normalized_log_pitch(self) -> bool:
        """
        Add the mean-subtracted log-pitch (1 dim).
        """
    @add_normalized_log_pitch.setter
    def add_normalized_log_pitch(self, arg0: bool) -> None:
        ...
    @property
    def add_pov_feature(self) -> bool:
        """
        Add the warped NCCF to the output (1 dim).
        """
    @add_pov_feature.setter
    def add_pov_feature(self, arg0: bool) -> None:
        ...
    @property
    def add_raw_log_pitch(self) -> bool:
        """
        Add log(pitch) (1 dim).
        """
    @add_raw_log_pitch.setter
    def add_raw_log_pitch(self, arg0: bool) -> None:
        ...
    @property
    def delay(self) -> int:
        """
        Frames by which the pitch information is delayed.
        """
    @delay.setter
    def delay(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def delta_pitch_noise_stddev(self) -> float:
        """
        Stddev of the noise added to the delta log-pitch.
        
        The noise is drawn from a deterministic stream, so the features are
        reproducible on every platform.
        """
    @delta_pitch_noise_stddev.setter
    def delta_pitch_noise_stddev(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def delta_pitch_scale(self) -> float:
        """
        Scaling of the delta log-pitch feature.
        """
    @delta_pitch_scale.setter
    def delta_pitch_scale(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def delta_window(self) -> int:
        """
        Frames on each side used for the delta window.
        """
    @delta_window.setter
    def delta_window(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def dim(self) -> int:
        """
        The output feature dimension.
        """
    @property
    def normalization_left_context(self) -> int:
        """
        Left context (frames) for log-pitch normalization.
        """
    @normalization_left_context.setter
    def normalization_left_context(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def normalization_right_context(self) -> int:
        """
        Right context (frames) for log-pitch normalization.
        """
    @normalization_right_context.setter
    def normalization_right_context(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def pitch_scale(self) -> float:
        """
        Scaling of the normalized log-pitch feature.
        """
    @pitch_scale.setter
    def pitch_scale(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def pov_offset(self) -> float:
        """
        Offset added to the POV feature.
        """
    @pov_offset.setter
    def pov_offset(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def pov_scale(self) -> float:
        """
        Scaling of the POV feature.
        """
    @pov_scale.setter
    def pov_scale(self, arg0: typing.SupportsFloat) -> None:
        ...
class RawAudioSamplesOptions:
    """
    Options for framing raw audio samples without any processing.
    
    The default ``frame_opts`` performs no processing on the samples:
    rectangular window, no dithering, no preemphasis, no DC offset removal, no
    zero-padding.
    
    .. seealso:: :class:`kaldi_native_fbank.OnlineRawAudioSamples`
    """
    @staticmethod
    def from_dict(arg0: dict) -> RawAudioSamplesOptions:
        """
        Build options from a dict.
        """
    def __getstate__(self) -> dict:
        ...
    def __init__(self, frame_opts: FrameExtractionOptions = ...) -> None:
        """
        Construct; ``frame_opts`` defaults to un-processed framing.
        
        The default is a rectangular window with no dithering, preemphasis, DC
        offset removal or zero-padding.
        
        :param frame_opts: Frame-extraction options; see
           :class:`FrameExtractionOptions`.
        """
    def __setstate__(self, arg0: dict) -> None:
        ...
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    def as_dict(self) -> dict:
        """
        Return the options as a dict.
        """
    @property
    def frame_opts(self) -> FrameExtractionOptions:
        """
        Frame-extraction options; see :class:`FrameExtractionOptions`.
        """
    @frame_opts.setter
    def frame_opts(self, arg0: FrameExtractionOptions) -> None:
        ...
class Rfft:
    """
    n-point real discrete Fourier transform, where n is even.
    
    Forward transform:
    
    .. math::
    
       R[k] = \\sum_{j=0}^{n-1} \\mathrm{in}[j] \\cos(2\\pi j k / n),
       \\quad 0 \\le k \\le n/2
    
       I[k] = \\sum_{j=0}^{n-1} \\mathrm{in}[j] \\sin(2\\pi j k / n),
       \\quad 0 < k < n/2
    
    .. seealso:: :class:`Stft`
    """
    def __init__(self, n: typing.SupportsInt, inverse: bool = False) -> None:
        """
        Construct for an n-point transform.
        
        :param n: Number of fft points; must be even and >= 2.
        :param inverse: If True, :meth:`compute` performs the inverse transform.
        """
    def compute(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> list[float]:
        """
        Transform in place and return the packed spectrum.
        
        On return of the forward transform:
        
        * ``out[0] = R[0]``
        * ``out[1] = R[n/2]``
        * for ``1 < k < n/2``: ``out[2*k] = R[k]``, ``out[2*k+1] = I[k]``
        
        :param d: 1-D list of n floats.
        :return: The packed spectrum, a 1-D list of n floats.
        """
class SpliceOptions:
    """
    Options for :func:`splice_frames`.
    
    Mirrors the flags of kaldi's `splice-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/splice-feats.cc>`_.
    
    .. seealso:: :func:`splice_frames`
    """
    def __init__(self, left_context: typing.SupportsInt = 3, right_context: typing.SupportsInt = 3) -> None:
        """
        Construct with the given contexts.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param left_context: Number of preceding frames to append
           (kaldi ``splice-feats --left-context``).
        :param right_context: Number of following frames to append
           (kaldi ``splice-feats --right-context``).
        """
    @property
    def left_context(self) -> int:
        """
        Number of preceding frames to append.
        
        Same as kaldi ``splice-feats --left-context``.
        """
    @left_context.setter
    def left_context(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def right_context(self) -> int:
        """
        Number of following frames to append.
        
        Same as kaldi ``splice-feats --right-context``.
        """
    @right_context.setter
    def right_context(self, arg0: typing.SupportsInt) -> None:
        ...
class Stft:
    """
    Compute the short-time Fourier transform of a waveform.
    
    .. seealso::
       :class:`IStft`
          The inverse transform.
       :class:`StftConfig`
          Configuration.
    """
    def __call__(self, input: collections.abc.Sequence[typing.SupportsFloat]) -> StftResult:
        ...
    def __init__(self, config: StftConfig) -> None:
        """
        Construct with the given config.
        
        :param config: STFT configuration; see :class:`StftConfig`.
        """
    def compute(self, input: collections.abc.Sequence[typing.SupportsFloat]) -> StftResult:
        """
        Compute the STFT of a waveform.
        
        :param data: 1-D float32 samples.
        :return: A :class:`StftResult` with the complex spectra of all frames.
        """
class StftConfig:
    """
    Configuration for :class:`Stft` and :class:`IStft`.
    
    .. seealso:: :class:`Stft`, :class:`IStft`
    """
    def __init__(self, n_fft: typing.SupportsInt, hop_length: typing.SupportsInt, win_length: typing.SupportsInt, window_type: str = '', center: bool = True, pad_mode: str = 'reflect', normalized: bool = False, window: collections.abc.Sequence[typing.SupportsFloat] = []) -> None:
        """
        Construct with the given parameters.
        
        All fields are optional keyword arguments; the defaults are shown in the
        signature.
        
        :param n_fft: FFT size; should be a power of two.
        :param hop_length: Number of samples between successive frames.
        :param win_length: Window size in samples (must be <= n_fft).
        :param window_type: Window type, e.g. "hann".
        :param center: If True, pad by n_fft/2 and center the frames.
        :param pad_mode: Padding mode when center is True ("reflect").
        :param normalized: If True, scale the FFT by 1/sqrt(n_fft).
        :param window: Explicit window coefficients (overrides window_type).
        """
    def __str__(self) -> str:
        """
        Return the config as a string.
        """
    @property
    def center(self) -> bool:
        """
        If True, pad the signal by ``n_fft // 2`` on both sides (using ``pad_mode``) and center the frames.
        """
    @center.setter
    def center(self, arg0: bool) -> None:
        ...
    @property
    def hop_length(self) -> int:
        """
        Number of samples between successive frames.
        """
    @hop_length.setter
    def hop_length(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def n_fft(self) -> int:
        """
        FFT size; should be a power of two.
        """
    @n_fft.setter
    def n_fft(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def normalized(self) -> bool:
        """
        If True, scale the FFT by ``1 / sqrt(n_fft)``.
        """
    @normalized.setter
    def normalized(self, arg0: bool) -> None:
        ...
    @property
    def pad_mode(self) -> str:
        """
        Padding mode used when ``center`` is True; currently only "reflect".
        """
    @pad_mode.setter
    def pad_mode(self, arg0: str) -> None:
        ...
    @property
    def win_length(self) -> int:
        """
        Window size in samples; must be <= n_fft.  The window is zero-padded to n_fft.
        """
    @win_length.setter
    def win_length(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def window_type(self) -> str:
        """
        Window type, e.g. "hann"; ignored if explicit window coefficients are given.
        """
    @window_type.setter
    def window_type(self, arg0: str) -> None:
        ...
class StftResult:
    """
    Result of :meth:`Stft.compute`: the complex spectra of all frames.
    
    .. seealso:: :class:`Stft`, :class:`IStft`
    """
    def __init__(self, real: collections.abc.Sequence[typing.SupportsFloat], imag: collections.abc.Sequence[typing.SupportsFloat], num_frames: typing.SupportsInt) -> None:
        """
        Construct a StftResult.
        
        :param real: Real parts, shape (num_frames, n_fft // 2 + 1), flattened row-major.
        :param imag: Imaginary parts, same shape.
        :param num_frames: Number of frames in real/imag.
        """
    @property
    def imag(self) -> list[float]:
        """
        Imaginary parts of the spectra, shape (num_frames, n_fft // 2 + 1), flattened row-major.
        """
    @property
    def num_frames(self) -> int:
        """
        Number of frames in real/imag.
        """
    @property
    def real(self) -> list[float]:
        """
        Real parts of the spectra, shape (num_frames, n_fft // 2 + 1), flattened row-major.
        """
class WhisperFeatureOptions:
    """
    Options for computing Whisper mel features.
    
    Note that most fields of ``frame_opts`` are ignored and reset inside
    WhisperFeatureComputer: 16 kHz sampling rate, 25 ms window, 10 ms shift,
    "hann" window, ``snip_edges=False``, etc.
    
    .. seealso:: :class:`kaldi_native_fbank.OnlineWhisperFbank`
    """
    @staticmethod
    def from_dict(arg0: dict) -> WhisperFeatureOptions:
        """
        Build options from a dict.
        """
    def __getstate__(self) -> dict:
        ...
    def __init__(self, frame_opts: FrameExtractionOptions = ..., dim: typing.SupportsInt = 80) -> None:
        """
        Construct with the given frame options and number of mel bins.
        
        :param frame_opts: Frame options (mostly ignored and reset by the computer).
        :param dim: Number of mel bins: 80 for most Whisper models, 128 for
           Whisper large v3.
        """
    def __setstate__(self, arg0: dict) -> None:
        ...
    def __str__(self) -> str:
        """
        Return the options as a string.
        """
    def as_dict(self) -> dict:
        """
        Return the options as a dict.
        """
    @property
    def dim(self) -> int:
        """
        Number of mel bins: 80 for most Whisper models, 128 for Whisper large v3.
        """
    @dim.setter
    def dim(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def frame_opts(self) -> FrameExtractionOptions:
        """
        Frame-extraction options (mostly ignored and reset by the computer).
        """
    @frame_opts.setter
    def frame_opts(self, arg0: FrameExtractionOptions) -> None:
        ...

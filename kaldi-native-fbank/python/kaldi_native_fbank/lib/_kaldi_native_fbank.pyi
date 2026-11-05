"""
Python wrapper for kaldi native fbank
"""
from __future__ import annotations
import collections.abc
import kaldi_native_fbank
from kaldi_native_fbank import CmvnAccumulator
from kaldi_native_fbank import CmvnOptions
from kaldi_native_fbank import DeltaOptions
from kaldi_native_fbank import FbankOptions
from kaldi_native_fbank import FeatureTransform
from kaldi_native_fbank import FeatureWindowFunction
from kaldi_native_fbank import FrameExtractionOptions
from kaldi_native_fbank import IStft
from kaldi_native_fbank import MelBanks
from kaldi_native_fbank import MelBanksOptions
from kaldi_native_fbank import MfccOptions
from kaldi_native_fbank import NemoFbankOptions
from kaldi_native_fbank import OnlineFbank
from kaldi_native_fbank import OnlineMfcc
from kaldi_native_fbank import OnlineNemoFbank
from kaldi_native_fbank import OnlinePitch
from kaldi_native_fbank import OnlineRawAudioSamples
from kaldi_native_fbank import OnlineWhisperFbank
from kaldi_native_fbank import PitchExtractionOptions
from kaldi_native_fbank import ProcessPitchOptions
from kaldi_native_fbank import RawAudioSamplesOptions
from kaldi_native_fbank import Rfft
from kaldi_native_fbank import SpliceOptions
from kaldi_native_fbank import Stft
from kaldi_native_fbank import StftConfig
from kaldi_native_fbank import StftResult
from kaldi_native_fbank import WhisperFeatureOptions
import numpy
import numpy.typing
import typing
__all__: list[str] = ['CmvnAccumulator', 'CmvnOptions', 'DeltaOptions', 'FbankOptions', 'FeatureTransform', 'FeatureWindowFunction', 'FrameExtractionOptions', 'IStft', 'MelBanks', 'MelBanksOptions', 'MfccOptions', 'NemoFbankOptions', 'OnlineFbank', 'OnlineMfcc', 'OnlineNemoFbank', 'OnlinePitch', 'OnlineRawAudioSamples', 'OnlineWhisperFbank', 'PitchExtractionOptions', 'ProcessPitchOptions', 'RawAudioSamplesOptions', 'Rfft', 'SpliceOptions', 'Stft', 'StftConfig', 'StftResult', 'WhisperFeatureOptions', 'apply_cmvn', 'apply_cmvn_', 'compute_and_process_kaldi_pitch', 'compute_cmvn_stats', 'compute_deltas', 'splice_frames', 'transform_features']
def apply_cmvn(stats: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], opts: kaldi_native_fbank.CmvnOptions, feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]"]:
    """
    Apply cmvn to features and return the normalized copy.
    
    This follows kaldi's `ApplyCmvn <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_ / `ApplyCmvnReverse <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_ semantics.
    Let :math:`c` be the frame count,
    :math:`\\mu_d = \\mathrm{stats}(0, d)/c` the mean and, when ``opts.norm_vars``
    is true, :math:`\\sigma_d^2 = \\mathrm{stats}(1, d)/c - \\mu_d^2` the variance
    of dimension :math:`d`.
    
    Forward (``opts.reverse == False``), applied to every frame :math:`x`:
    
    .. math::
    
       x_d \\leftarrow x_d - \\mu_d
    
    and, if ``opts.norm_vars``:
    
    .. math::
    
       x_d \\leftarrow (x_d - \\mu_d) / \\sigma_d
    
    Reverse (``opts.reverse == True``) undoes the normalization:
    
    .. math::
    
       x_d \\leftarrow x_d + \\mu_d
    
    and, if ``opts.norm_vars``:
    
    .. math::
    
       x_d \\leftarrow x_d \\cdot \\sigma_d + \\mu_d
    
    1-row (mean-only) stats are accepted when ``norm_vars`` is false; requesting
    variance normalization with 1-row stats raises an exception (same as kaldi).
    
    :param stats: 1-row or 2-row cmvn stats, e.g. from
       :func:`compute_cmvn_stats`.
    :param opts: Options controlling the normalization; see :class:`CmvnOptions`.
    :param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
    :type feats: numpy.ndarray
    :return: A normalized copy of ``feats``.
    :rtype: numpy.ndarray
    
    Example
    -------
    
    .. code-block:: python
    
       stats = knf.compute_cmvn_stats(feats)
       opts = knf.CmvnOptions()  # norm_vars=False, reverse=False
       normed = knf.apply_cmvn(stats, opts, feats)
    
    .. seealso:: :func:`apply_cmvn_`, :func:`compute_cmvn_stats`,
       :class:`CmvnAccumulator`
    """
def apply_cmvn_(stats: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], opts: kaldi_native_fbank.CmvnOptions, feats: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]", "flags.writeable", "flags.c_contiguous"]) -> None:
    """
    In-place version of :func:`apply_cmvn`.
    
    ``feats`` is modified directly; no copy of the feature matrix is made.
    
    :param stats: 1-row or 2-row cmvn stats; see :func:`compute_cmvn_stats`.
    :param opts: Options controlling the normalization; see :class:`CmvnOptions`.
    :param feats: Feature matrix of shape ``(num_frames, dim)``.  Must be a
       writeable, C-contiguous float32 ``numpy.ndarray``; it is normalized
       in place.
    
    .. seealso:: :func:`apply_cmvn`
       For the normalization formulas.
    """
def compute_and_process_kaldi_pitch(pitch_opts: kaldi_native_fbank.PitchExtractionOptions, process_opts: kaldi_native_fbank.ProcessPitchOptions, wave: collections.abc.Sequence[typing.SupportsFloat]) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]"]:
    """
    Compute processed pitch features for a whole utterance.
    
    This is kaldi's
    `compute-and-process-kaldi-pitch-feats
    <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-and-process-kaldi-pitch-feats.cc>`_
    (same arithmetic): the raw (NCCF, pitch) extraction of
    `Ghahremani et al., ICASSP 2014
    <https://www.danielpovey.com/files/icassp2014_pitch.pdf>`_ followed by
    post-processing into the configured features (by default
    POV, normalized-log-pitch, delta-pitch).  The output equals the kaldi
    binary's output value-by-value for the same input and options.
    
    It is sensitive to the variables in ``pitch_opts`` that relate to online
    processing, i.e. ``max_frames_latency``, ``frames_per_chunk``,
    ``simulate_first_pass_online``, ``recompute_frame``.
    
    The noise added to the delta-pitch feature (``process_opts.delta_pitch_noise_stddev``)
    is drawn from a deterministic stream that reproduces the reference kaldi
    realization, so the result is identical on every platform.
    
    :param pitch_opts: Raw pitch extraction options.
    :param process_opts: Post-processing options.
    :param wave: 1-D float32 samples (kaldi semantics: raw WAV magnitudes, not
       divided by 32768).
    :type wave: numpy.ndarray
    :return: Feature matrix of shape ``(num_frames, process_opts.dim)``,
       float32.
    :rtype: numpy.ndarray
    
    Example
    -------
    
    .. code-block:: python
    
       import wave
       import numpy as np
       import kaldi_native_fbank as knf
    
       with wave.open("zh0_1s.wav") as f:
           samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
       samples = samples.astype(np.float32)  # raw magnitudes, no /32768
    
       pitch_opts = knf.PitchExtractionOptions(min_f0=50, max_f0=500,
                                               snip_edges=True)
       feats = knf.compute_and_process_kaldi_pitch(pitch_opts,
                                                   knf.ProcessPitchOptions(),
                                                   samples)
       # feats.shape == (num_frames, 3): POV, norm-log-pitch, delta-pitch
    
    .. seealso:: :class:`OnlinePitch`, :class:`PitchExtractionOptions`,
       :class:`ProcessPitchOptions`
    """
def compute_cmvn_stats(feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Compute full 2-row cmvn stats for a feature matrix.
    
    Equivalent to initializing zero stats and accumulating every row of ``feats``
    with weight 1.0 (kaldi `AccCmvnStats <https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc>`_).
    
    The returned stats matrix matches kaldi's ``cmvn.ark`` layout exactly:
    
    .. math::
    
       \\mathrm{stats} =
       \\left( \\begin{array}{cccc|c}
         S_1 & S_2 & \\cdots & S_D & c \\\\
         Q_1 & Q_2 & \\cdots & Q_D & -
       \\end{array} \\right)
    
    * Row 0, columns ``0..D-1`` hold the sums
      :math:`S_d = \\sum_t x_t(d)`; the last column :math:`c` holds the frame
      count (sum of weights).
    * Row 1, columns ``0..D-1`` hold the sums of squares
      :math:`Q_d = \\sum_t x_t(d)^2`; the last column is unused.
    
    :param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
    :type feats: numpy.ndarray
    :return: Stats matrix of shape ``(2, dim + 1)``, float64.
    :rtype: numpy.ndarray
    
    .. seealso::
       :func:`apply_cmvn`
          Apply these stats to features.
       :class:`CmvnAccumulator`
          Incremental version of this computation.
    """
def compute_deltas(opts: kaldi_native_fbank.DeltaOptions, feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]"]:
    """
    Compute delta (dynamic) features.
    
    This is kaldi's `add-deltas <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/add-deltas.cc>`_
    (same arithmetic).  The :math:`i`-th block of the output is the :math:`i`-th
    order regression delta; with :math:`N` = ``opts.window``, the first-order
    delta of frame :math:`t` is
    
    .. math::
    
       \\Delta_t = \\frac{\\sum_{j=-N}^{N} j\\, x_{\\mathrm{clamp}(t+j)}}
                       {2 \\sum_{j=1}^{N} j^2}
    
    where out-of-range frames clamp to the first/last frame (edge frames are
    replicated).  Higher orders apply the same operator recursively to the
    previous order's window.  The output rows are
    :math:`[x_t, \\Delta_t, \\Delta^2_t, \\ldots]`, i.e. the block order is
    baseline first, then increasing delta order.
    
    The regression windows themselves are computed as follows.  With
    :math:`N` = ``opts.window``, the first-order window is
    
    .. math::
    
       w^{(1)}_j = \\frac{j}{\\sum_{k=-N}^{N} k^2}
                 = \\frac{j}{2 \\sum_{k=1}^{N} k^2},
       \\qquad j = -N, \\ldots, N
    
    and each higher-order window is the discrete convolution of the previous
    window with :math:`w^{(1)}` (kaldi's ``scales_`` recursion):
    
    .. math::
    
       w^{(i)} = w^{(i-1)} * w^{(1)}, \\qquad w^{(0)} = [1]
    
    so :math:`w^{(i)}` is the :math:`i`-fold convolution of :math:`w^{(1)}`
    (its length is ``1 + 2 * i * window``).
    
    **Motivation.**  The spectrum of speech is strongly non-stationary: the
    *trajectory* of the spectrum over time carries information that the
    static spectrum does not.  Furui (1986) showed that appending the time
    derivative of the spectrum (the "delta") significantly improves
    recognition.  HTK refined this into the regression-based formulation
    used here: instead of a raw difference such as
    :math:`x_{t+1} - x_{t-1}`, the local slope of a least-squares line is
    estimated over a window of :math:`2N+1` frames, which is less sensitive
    to noise and to isolated outliers.  kaldi's ``DeltaFeatures`` implements
    the same HTK formulation, so the outputs are compatible with models
    trained on HTK- or kaldi-produced features.
    
    **Why this definition.**  The first-order window is the slope of the
    least-squares straight line fitted to the frames around :math:`t` — the
    classic HTK regression delta that kaldi's ``DeltaFeatures`` implements.
    Fitting :math:`y(j) = a + b\\, j` to :math:`x_{t+j}`,
    :math:`j = -N, \\ldots, N`, by minimizing
    :math:`\\sum_j (x_{t+j} - a - b\\, j)^2`, and using
    :math:`\\sum_j j = 0` (which decouples the intercept from the slope), the
    slope estimate is
    
    .. math::
    
       b = \\frac{\\sum_j j\\, x_{t+j}}{\\sum_j j^2}
         = \\sum_j w^{(1)}_j\\, x_{t+j}
    
    so :math:`w^{(1)}_j = j / \\sum_k k^2` is exactly the least-squares slope
    estimator: :math:`\\Delta_t = b` is the local slope of the fitted line,
    i.e. a least-squares estimate of the first derivative (in feature units
    per frame — the :math:`\\sum_j j^2` normalizer is the OLS denominator).
    
    Higher orders apply the *same* slope estimator to the previous order's
    output (:math:`\\Delta^2` is "the slope of the slope").  Since the slope
    estimator is a linear filter with kernel :math:`w^{(1)}`, filtering an
    already-filtered sequence convolves the kernels, hence
    :math:`w^{(i)} = w^{(i-1)} * w^{(1)}` with support growing to
    :math:`\\pm i N`.  Note this is *recursive* slope-of-slope — it is not
    identical to the :math:`i`-th derivative of a degree-:math:`i`
    least-squares polynomial fitted over the wider :math:`\\pm iN` window.
    
    Two further details fall out of the construction:
    
    * Normalizing by :math:`\\sum_j j^2` at every order keeps each block on
      the scale of a true :math:`i`-th derivative.
    * Edge frames replicate the first/last frame so the frame count is
      preserved and the regression "sees" a locally constant signal at the
      borders (instead of zero-padding, which would bias the deltas).
    
    **References.**
    
    * The ported implementation: kaldi's ``DeltaFeatures``
      (`feature-functions.cc <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-functions.cc>`_,
      `feature-functions.h <https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-functions.h>`_).
    * The regression-delta formulation: *The HTK Book*, S. Young et al.,
      Cambridge University Engineering Department, 2006, chapter 5
      ("Speech Input/Output"), section 5.9 ("Delta, Acceleration and Third
      Differential Coefficients"); PDF:
      https://www.danielpovey.com/files/htkbook.pdf
    * Delta features originate from S. Furui, "Speaker-independent isolated
      word recognition using dynamic features of speech spectrum", IEEE
      Trans. Acoustics, Speech, and Signal Processing, 1986,
      https://doi.org/10.1109/TASSP.1986.1164788
    
    **How the convolution is computed.**  The discrete convolution of two
    windows :math:`a` (indices :math:`-A..A`) and :math:`b` (indices
    :math:`-B..B`) is the window :math:`c = a * b` (indices
    :math:`-(A+B)..A+B`) defined by
    
    .. math::
    
       c_m = \\sum_{k} a_k \\, b_{m-k}
    
    where :math:`a_k` and :math:`b_k` are zero outside their index ranges.
    For the second-order window this is
    
    .. math::
    
       w^{(2)}_m = \\sum_{k=-N}^{N} w^{(1)}_k \\, w^{(1)}_{m-k}
    
    **Window example 1 — N = 1, orders 1 to 3.**
    :math:`\\sum_{k=-1}^{1} k^2 = 2`, so the first-order window is
    :math:`w^{(1)}_j = j/2 = [-0.5, 0, 0.5]`.  The convolution
    :math:`w^{(2)}_m = \\sum_k w^{(1)}_k w^{(1)}_{m-k}` term by term:
    
    * :math:`m = -2`: :math:`w^{(1)}_{-1} w^{(1)}_{-1} = (-0.5)(-0.5) = 0.25`
    * :math:`m = -1`: :math:`w^{(1)}_{-1} w^{(1)}_{0} + w^{(1)}_{0} w^{(1)}_{-1}
      = (-0.5)(0) + (0)(-0.5) = 0`
    * :math:`m = 0`: :math:`w^{(1)}_{-1} w^{(1)}_{1} + w^{(1)}_{0} w^{(1)}_{0}
      + w^{(1)}_{1} w^{(1)}_{-1} = (-0.5)(0.5) + 0 + (0.5)(-0.5) = -0.5`
    * :math:`m = 1`: :math:`w^{(1)}_{0} w^{(1)}_{1} + w^{(1)}_{1} w^{(1)}_{0} = 0`
    * :math:`m = 2`: :math:`w^{(1)}_{1} w^{(1)}_{1} = (0.5)(0.5) = 0.25`
    
    which gives :math:`w^{(2)} = [0.25, 0, -0.5, 0, 0.25]`.  Convolving once
    more, :math:`w^{(3)} = w^{(2)} * w^{(1)}`; e.g.
    :math:`w^{(3)}_{-3} = w^{(1)}_{-1} w^{(2)}_{-2} = (-0.5)(0.25) = -0.125` and
    :math:`w^{(3)}_{-1} = w^{(1)}_{-1} w^{(2)}_{0} + w^{(1)}_{1} w^{(2)}_{-2}
    = (-0.5)(-0.5) + (0.5)(0.25) = 0.375`.  The full chain for N = 1 is:
    
    * :math:`w^{(0)} = [1]`
    * :math:`w^{(1)} = [-0.5, 0, 0.5]`
    * :math:`w^{(2)} = [0.25, 0, -0.5, 0, 0.25]`
    * :math:`w^{(3)} = [-0.125, 0, 0.375, 0, -0.375, 0, 0.125]`
    
    **Window example 2 — N = 2.**
    :math:`\\sum_{k=-2}^{2} k^2 = 10`, so
    :math:`w^{(1)}_j = j/10 = [-0.2, -0.1, 0, 0.1, 0.2]`.  Two sample
    convolution sums for :math:`w^{(2)} = w^{(1)} * w^{(1)}`:
    
    * :math:`w^{(2)}_{0} = \\sum_k w^{(1)}_k w^{(1)}_{-k} = 2(-0.2)(0.2) +
      2(-0.1)(0.1) + (0)(0) = -0.1`
    * :math:`w^{(2)}_{4} = w^{(1)}_{2} w^{(1)}_{2} = (0.2)(0.2) = 0.04`
    
    The full 9-tap window is
    :math:`w^{(2)} = [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]`.
    
    **Example 1 — full derivation** (``x = [1, 2, 4]``, ``order=2``, ``window=1``).
    With the windows derived above,
    :math:`w^{(1)} = [-0.5, 0, 0.5]` and :math:`w^{(2)} = [0.25, 0, -0.5, 0,
    0.25]`, i.e.
    
    .. math::
    
       \\Delta_t = -0.5\\, x_{t-1} + 0.5\\, x_{t+1}
    
       \\Delta^2_t = 0.25\\, x_{t-2} - 0.5\\, x_t + 0.25\\, x_{t+2}
    
    For the 1-D input ``x = [1, 2, 4]`` (out-of-range frames clamp to the
    first/last frame):
    
    * ``t = 0``: :math:`\\Delta = -0.5 \\cdot 1 + 0.5 \\cdot 2 = 0.5`,
      :math:`\\Delta^2 = 0.25 \\cdot 1 - 0.5 \\cdot 1 + 0.25 \\cdot 4 = 0.75`
    * ``t = 1``: :math:`\\Delta = -0.5 \\cdot 1 + 0.5 \\cdot 4 = 1.5`,
      :math:`\\Delta^2 = 0.25 \\cdot 1 - 0.5 \\cdot 2 + 0.25 \\cdot 4 = 0.25`
    * ``t = 2``: :math:`\\Delta = -0.5 \\cdot 2 + 0.5 \\cdot 4 = 1.0`,
      :math:`\\Delta^2 = 0.25 \\cdot 1 - 0.5 \\cdot 4 + 0.25 \\cdot 4 = -0.75`
    
    .. code-block:: python
    
       out = knf.compute_deltas(knf.DeltaOptions(order=2, window=1), x)
       # out rows [x, delta, delta2]:
       #   [1, 0.5,  0.75]
       #   [2, 1.5,  0.25]
       #   [4, 1.0, -0.75]
    
    **Example 2 — larger window** (``x = [3, 1, 4, 1, 5]``, ``order=1``,
    ``window=2``).  With :math:`N = 2`,
    :math:`\\sum_{k=-2}^{2} k^2 = 10`, so
    :math:`w^{(1)} = [-0.2, -0.1, 0, 0.1, 0.2]` (each entry is :math:`j/10`).
    For instance at :math:`t = 2`:
    :math:`\\Delta = -0.2 \\cdot 3 - 0.1 \\cdot 1 + 0.1 \\cdot 1 + 0.2 \\cdot 5 = 0.4`.
    The output rows ``[x, delta]`` are:
    
    .. code-block:: python
    
       out = knf.compute_deltas(knf.DeltaOptions(order=1, window=2), x)
       #   [3,  0.0]
       #   [1, -0.3]
       #   [4,  0.4]
       #   [1,  0.9]
       #   [5,  0.6]
    
    **Example 3 — two feature dimensions** (``D = 2``, ``order=2``,
    ``window=1``).  Each dimension is processed independently and the blocks
    are laid out ``[x, delta, delta2]`` with the dimensions interleaved per
    block:
    
    .. code-block:: python
    
       x = [[1, 10], [2, 20], [4, 40]]
       out = knf.compute_deltas(knf.DeltaOptions(order=2, window=1), x)
       # out rows [x1, x2, d1, d2, dd1, dd2]:
       #   [1, 10,  0.5,  5.0,  0.75,  7.5]
       #   [2, 20,  1.5, 15.0,  0.25,  2.5]
       #   [4, 40,  1.0, 10.0, -0.75, -7.5]
    
    :param opts: Order/window; see :class:`DeltaOptions`.
    :param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
    :type feats: numpy.ndarray
    :return: The matrix of shape ``(num_frames, dim * (order + 1))``, float32.
    :rtype: numpy.ndarray
    
    Example
    -------
    
    .. code-block:: python
    
       opts = knf.DeltaOptions(order=2, window=2)
       out = knf.compute_deltas(opts, feats)  # (T, 3 * dim): [x, delta, d-delta]
    
    .. seealso:: :class:`DeltaOptions`, :func:`splice_frames`
    """
def splice_frames(feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"], opts: kaldi_native_fbank.SpliceOptions = ...) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]"]:
    """
    Splice each frame with its left/right context neighbors.
    
    This is kaldi's `splice-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/splice-feats.cc>`_
    (same arithmetic): for every frame index :math:`t` the output row is
    
    .. math::
    
       y_t = [ x_{\\max(0, t-L)}, \\ldots, x_{\\min(T-1, t+R)} ]
    
    where :math:`L` is ``opts.left_context``, :math:`R` is
    ``opts.right_context`` and :math:`T` is the number of input frames.  Edge
    frames are replicated: out-of-range indices clamp to the first/last frame.
    The output therefore has the same number of frames as the input and
    ``dim * (1 + L + R)`` dimensions.
    
    :param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
    :type feats: numpy.ndarray
    :param opts: Contexts; see :class:`SpliceOptions`.
    :return: The spliced matrix of shape ``(num_frames, dim * (1 + L + R))``,
       float32.
    :rtype: numpy.ndarray
    
    Example
    -------
    
    .. code-block:: python
    
       opts = knf.SpliceOptions(left_context=3, right_context=3)
       spliced = knf.splice_frames(feats, opts)  # (T, 7 * dim)
    
    .. seealso:: :class:`SpliceOptions`
    """
def transform_features(feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"], transform: kaldi_native_fbank.FeatureTransform) -> typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]"]:
    """
    Apply a transform to features and return the result.
    
    Shorthand for ``transform.apply(feats)``; matches kaldi's
    `transform-feats <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/transform-feats.cc>`_.
    
    :param feats: Feature matrix of shape ``(num_frames, dim)``, float32.
    :type feats: numpy.ndarray
    :param transform: The transform; see :class:`FeatureTransform`.
    :return: The transformed matrix of shape ``(num_frames, transform.output_dim)``.
    :rtype: numpy.ndarray
    
    .. seealso:: :class:`FeatureTransform`
    """

"""
Python wrapper for kaldi native fbank
"""
from __future__ import annotations
import kaldi_native_fbank
from kaldi_native_fbank import CmvnAccumulator
from kaldi_native_fbank import CmvnOptions
from kaldi_native_fbank import FbankOptions
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
from kaldi_native_fbank import OnlineRawAudioSamples
from kaldi_native_fbank import OnlineWhisperFbank
from kaldi_native_fbank import RawAudioSamplesOptions
from kaldi_native_fbank import Rfft
from kaldi_native_fbank import Stft
from kaldi_native_fbank import StftConfig
from kaldi_native_fbank import StftResult
from kaldi_native_fbank import WhisperFeatureOptions
import numpy
import numpy.typing
import typing
__all__: list[str] = ['CmvnAccumulator', 'CmvnOptions', 'FbankOptions', 'FeatureWindowFunction', 'FrameExtractionOptions', 'IStft', 'MelBanks', 'MelBanksOptions', 'MfccOptions', 'NemoFbankOptions', 'OnlineFbank', 'OnlineMfcc', 'OnlineNemoFbank', 'OnlineRawAudioSamples', 'OnlineWhisperFbank', 'RawAudioSamplesOptions', 'Rfft', 'Stft', 'StftConfig', 'StftResult', 'WhisperFeatureOptions', 'apply_cmvn', 'apply_cmvn_', 'compute_cmvn_stats']
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

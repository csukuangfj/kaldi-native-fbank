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
    Applies cmvn (kaldi ApplyCmvn semantics) and returns the normalized copy.
    """
def apply_cmvn_(stats: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], opts: kaldi_native_fbank.CmvnOptions, feats: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, n]", "flags.writeable", "flags.c_contiguous"]) -> None:
    """
    In-place version of apply_cmvn.
    """
def compute_cmvn_stats(feats: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Computes cmvn stats for a feature matrix (frames x dim). Returns a 2 x (dim+1) double matrix: row0=[sums, count], row1=[sumsq, (unused)] (kaldi layout).
    """

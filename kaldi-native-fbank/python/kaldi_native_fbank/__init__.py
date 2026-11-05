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
    OnlineNemoFbank, OnlineRawAudioSamples
Options: FbankOptions, MfccOptions, FrameExtractionOptions, MelBanksOptions,
    WhisperFeatureOptions, NemoFbankOptions, RawAudioSamplesOptions,
    StftConfig, CmvnOptions
CMVN: compute_cmvn_stats, apply_cmvn, apply_cmvn_, CmvnAccumulator
Transforms: Stft, IStft, Rfft, MelBanks, FeatureWindowFunction

Every option class constructor takes all fields as keyword arguments with
their defaults, so ``help(knf.FbankOptions)`` shows the full signature.
"""

from kaldi_native_fbank.lib._kaldi_native_fbank import (
    apply_cmvn,
    apply_cmvn_,
    CmvnAccumulator,
    CmvnOptions,
    DeltaOptions,
    compute_cmvn_stats,
    compute_deltas,
    splice_frames,
    FbankOptions,
    FeatureWindowFunction,
    FrameExtractionOptions,
    IStft,
    MelBanks,
    MelBanksOptions,
    MfccOptions,
    NemoFbankOptions,
    OnlineFbank,
    OnlineMfcc,
    OnlineNemoFbank,
    OnlineRawAudioSamples,
    OnlineWhisperFbank,
    RawAudioSamplesOptions,
    Rfft,
    SpliceOptions,
    Stft,
    StftConfig,
    StftResult,
    WhisperFeatureOptions,
)

__all__ = [
    "CmvnAccumulator",
    "CmvnOptions",
    "DeltaOptions",
    "FbankOptions",
    "FeatureWindowFunction",
    "FrameExtractionOptions",
    "IStft",
    "MelBanks",
    "MelBanksOptions",
    "MfccOptions",
    "NemoFbankOptions",
    "OnlineFbank",
    "OnlineMfcc",
    "OnlineNemoFbank",
    "OnlineRawAudioSamples",
    "OnlineWhisperFbank",
    "RawAudioSamplesOptions",
    "SpliceOptions",
    "Rfft",
    "Stft",
    "StftConfig",
    "StftResult",
    "WhisperFeatureOptions",
    "apply_cmvn",
    "apply_cmvn_",
    "compute_cmvn_stats",
    "compute_deltas",
    "splice_frames",
]

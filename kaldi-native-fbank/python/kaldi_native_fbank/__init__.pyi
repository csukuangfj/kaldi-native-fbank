"""
Type stubs for kaldi-native-fbank

# Copyright (c) 2025 (authors: Bangwen He)
"""

from typing import Dict, List, Tuple, Union
import numpy as np

class FbankOptions:
    """Filter bank feature extraction options."""

    def __init__(self, frame_opts: FrameExtractionOptions = ..., mel_opts: MelBanksOptions = ..., use_energy: bool = False, energy_floor: float = 0, raw_energy: bool = True, htk_compat: bool = False, use_log_fbank: bool = True, use_log_floor: bool = True, eps: float = 1.1920929e-07, use_power: bool = True) -> None: ...

    # Properties
    frame_opts: FrameExtractionOptions
    mel_opts: MelBanksOptions
    use_energy: bool
    energy_floor: float
    raw_energy: bool
    htk_compat: bool
    use_log_fbank: bool
    use_log_floor: bool
    eps: float
    use_power: bool

    def __str__(self) -> str: ...
    def as_dict(self) -> Dict[str, Union[Dict, bool, float]]: ...
    @staticmethod
    def from_dict(d: Dict[str, Union[Dict, bool, float]]) -> "FbankOptions": ...

class MfccOptions:
    """MFCC feature extraction options."""

    def __init__(self, frame_opts: FrameExtractionOptions = ..., mel_opts: MelBanksOptions = ..., num_ceps: int = 13, use_energy: bool = True, energy_floor: float = 0, raw_energy: bool = True, cepstral_lifter: float = 22, htk_compat: bool = False) -> None: ...

    # Properties
    frame_opts: FrameExtractionOptions
    mel_opts: MelBanksOptions
    num_ceps: int
    use_energy: bool
    energy_floor: float
    raw_energy: bool
    cepstral_lifter: float
    htk_compat: bool

    def __str__(self) -> str: ...
    def as_dict(self) -> Dict[str, Union[Dict, int, bool, float]]: ...
    @staticmethod
    def from_dict(d: Dict[str, Union[Dict, int, bool, float]]) -> "MfccOptions": ...

class FrameExtractionOptions:
    """Frame extraction options for audio processing."""

    def __init__(self, samp_freq: float = 16000, frame_shift_ms: float = 10, frame_length_ms: float = 25, dither: float = 0.00003, preemph_coeff: float = 0.97, remove_dc_offset: bool = True, window_type: str = "povey", round_to_power_of_two: bool = True, blackman_coeff: float = 0.42, snip_edges: bool = True) -> None: ...

    # Properties
    samp_freq: float
    frame_shift_ms: float
    frame_length_ms: float
    dither: float
    preemph_coeff: float
    remove_dc_offset: bool
    window_type: str
    round_to_power_of_two: bool
    blackman_coeff: float
    snip_edges: bool

    def __str__(self) -> str: ...
    def as_dict(self) -> Dict[str, Union[float, bool, str]]: ...
    @staticmethod
    def from_dict(d: Dict[str, Union[float, bool, str]]) -> "FrameExtractionOptions": ...

class FeatureWindowFunction:
    """Feature window function for applying windowing to audio frames."""

    def __init__(self, opts: FrameExtractionOptions) -> None: ...

    def apply(self, wave: List[float]) -> List[float]: ...

    @property
    def window(self) -> List[float]: ...

class IStft:
    """Inverse Short-Time Fourier Transform."""

    def __init__(self, config: StftConfig) -> None: ...

    def compute(self, stft_result: StftResult) -> List[float]: ...
    def __call__(self, stft_result: StftResult) -> List[float]: ...

class MelBanksOptions:
    """Mel filter bank options."""

    def __init__(self, num_bins: int = 25, low_freq: float = 20, high_freq: float = 0, vtln_low: float = 100, vtln_high: float = -500, debug_mel: bool = False, htk_mode: bool = False, is_librosa: bool = False, norm: str = "slaney", use_slaney_mel_scale: bool = True, floor_to_int_bin: bool = False) -> None: ...

    # Properties
    num_bins: int
    low_freq: float
    high_freq: float
    vtln_low: float
    vtln_high: float
    debug_mel: bool
    htk_mode: bool
    is_librosa: bool
    norm: str

    def __str__(self) -> str: ...
    def as_dict(self) -> Dict[str, Union[int, float, bool, str]]: ...
    @staticmethod
    def from_dict(d: Dict[str, Union[int, float, bool, str]]) -> "MelBanksOptions": ...

class WhisperFeatureOptions:
    """Whisper feature extraction options."""

    def __init__(self, frame_opts: FrameExtractionOptions = ..., dim: int = 80) -> None: ...

    # Properties
    frame_opts: FrameExtractionOptions
    dim: int

    def __str__(self) -> str: ...
    def as_dict(self) -> Dict[str, Union[Dict, int]]: ...
    @staticmethod
    def from_dict(d: Dict[str, Union[Dict, int]]) -> "WhisperFeatureOptions": ...

class OnlineFbank:
    """Online filter bank feature extractor."""

    def __init__(self, opts: FbankOptions) -> None: ...

    @property
    def dim(self) -> int: ...

    @property
    def frame_shift_in_seconds(self) -> float: ...

    @property
    def num_frames_ready(self) -> int: ...

    def is_last_frame(self, frame: int) -> bool: ...
    def get_frame(self, frame: int) -> np.ndarray: ...
    def accept_waveform(self, sampling_rate: float, waveform: List[float]) -> None: ...
    def input_finished(self) -> None: ...
    def pop(self, n: int) -> None: ...

class OnlineMfcc:
    """Online MFCC feature extractor."""

    def __init__(self, opts: MfccOptions) -> None: ...

    @property
    def dim(self) -> int: ...

    @property
    def frame_shift_in_seconds(self) -> float: ...

    @property
    def num_frames_ready(self) -> int: ...

    def is_last_frame(self, frame: int) -> bool: ...
    def get_frame(self, frame: int) -> np.ndarray: ...
    def accept_waveform(self, sampling_rate: float, waveform: List[float]) -> None: ...
    def input_finished(self) -> None: ...
    def pop(self, n: int) -> None: ...

class OnlineWhisperFbank:
    """Online Whisper filter bank feature extractor."""

    def __init__(self, opts: WhisperFeatureOptions) -> None: ...

    @property
    def dim(self) -> int: ...

    @property
    def frame_shift_in_seconds(self) -> float: ...

    @property
    def num_frames_ready(self) -> int: ...

    def is_last_frame(self, frame: int) -> bool: ...
    def get_frame(self, frame: int) -> np.ndarray: ...
    def accept_waveform(self, sampling_rate: float, waveform: List[float]) -> None: ...
    def input_finished(self) -> None: ...
    def pop(self, n: int) -> None: ...

class Rfft:
    """Real-valued Fast Fourier Transform."""

    def __init__(self, n: int, inverse: bool = False) -> None: ...

    def compute(self, d: List[float]) -> List[float]: ...

class StftConfig:
    """STFT configuration parameters."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_type: str = "",
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        window: List[float] = []
    ) -> None: ...

    # Properties
    n_fft: int
    hop_length: int
    win_length: int
    window_type: str
    center: bool
    pad_mode: str
    normalized: bool

    def __str__(self) -> str: ...

class StftResult:
    """STFT computation result containing real and imaginary parts."""

    def __init__(self, real: List[float], imag: List[float], num_frames: int) -> None: ...

    @property
    def real(self) -> List[float]: ...

    @property
    def imag(self) -> List[float]: ...

    @property
    def num_frames(self) -> int: ...

class Stft:
    """Short-Time Fourier Transform."""

    def __init__(self, config: StftConfig) -> None: ...

    def compute(self, input: List[float]) -> StftResult: ...
    def __call__(self, input: List[float]) -> StftResult: ...


class NemoFbankOptions:
    """Options for NeMo buffered filter banks. Frequencies are in Hz."""

    def __init__(self, samp_freq: float = 16000, frame_shift_ms: float = 10, frame_length_ms: float = 25, num_mel_bins: int = 80, low_freq: float = 0, high_freq: float = 0, preemph_coeff: float = 0.97, buffered_capacity: int = 0) -> None: ...

    samp_freq: float
    frame_shift_ms: float
    frame_length_ms: float
    num_mel_bins: int
    low_freq: float
    high_freq: float
    preemph_coeff: float
    buffered_capacity: int

    def __str__(self) -> str: ...
    def as_dict(self) -> Dict[str, Union[float, int]]: ...
    @staticmethod
    def from_dict(d: Dict[str, Union[float, int]]) -> "NemoFbankOptions": ...

class OnlineNemoFbank:
    """Re-featurize requested windows; discard raw samples before each window."""

    def __init__(self, opts: NemoFbankOptions) -> None: ...

    @property
    def dim(self) -> int: ...
    @property
    def frame_shift_in_seconds(self) -> float: ...
    @property
    def num_frames_ready(self) -> int: ...
    @property
    def num_trailing_samples(self) -> int: ...

    def is_last_frame(self, frame: int) -> bool: ...
    def accept_waveform(self, sampling_rate: float, waveform: List[float]) -> None: ...
    def input_finished(self) -> None: ...
    def get_frames(
        self, frame_index: int, n: int, is_final: bool = False
    ) -> Tuple[np.ndarray, int]:
        """Return owned (n, dim) float32 frames and the start shift in samples."""
        ...

class CmvnOptions:
    """CMVN options (kaldi apply-cmvn flags)."""

    def __init__(self, norm_vars: bool = False, reverse: bool = False) -> None: ...

    # Properties
    norm_vars: bool
    reverse: bool

    def __str__(self) -> str: ...

class CmvnAccumulator:
    """Incremental accumulator for speaker-level / streaming cmvn."""

    def __init__(self, dim: int = 0) -> None: ...

    def accumulate(self, feats: np.ndarray) -> None: ...
    def add(self, dim: int, frame: List[float], weight: float = 1.0) -> None: ...

    @property
    def stats(self) -> np.ndarray: ...

class MelBanks:
    """Triangular mel filter banks."""

    def __init__(
        self,
        opts: MelBanksOptions = ...,
        frame_opts: FrameExtractionOptions = ...,
        vtln_warp_factor: float = 1.0,
    ) -> None: ...

    def get_matrix(self) -> np.ndarray: ...
    def compute(self, fft_energies: np.ndarray) -> np.ndarray: ...

class RawAudioSamplesOptions:
    """Raw audio samples framing options."""

    def __init__(self, frame_opts: FrameExtractionOptions = ...) -> None: ...

    # Properties
    frame_opts: FrameExtractionOptions

    def __str__(self) -> str: ...
    def as_dict(self) -> Dict[str, Union[Dict, bool, float]]: ...
    @staticmethod
    def from_dict(d: Dict[str, Union[Dict, bool, float]]) -> "RawAudioSamplesOptions": ...

class OnlineRawAudioSamples:
    """Online raw audio samples framer."""

    def __init__(self, opts: RawAudioSamplesOptions) -> None: ...

    @property
    def dim(self) -> int: ...

    @property
    def frame_shift_in_seconds(self) -> float: ...

    @property
    def num_frames_ready(self) -> int: ...

    def is_last_frame(self, frame: int) -> bool: ...
    def get_frame(self, frame: int) -> np.ndarray: ...
    def accept_waveform(self, sampling_rate: float, waveform: List[float]) -> None: ...
    def input_finished(self) -> None: ...
    def pop(self, n: int) -> None: ...

def compute_cmvn_stats(feats: np.ndarray) -> np.ndarray: ...
def apply_cmvn(stats: np.ndarray, opts: CmvnOptions, feats: np.ndarray) -> np.ndarray: ...
def apply_cmvn_(stats: np.ndarray, opts: CmvnOptions, feats: np.ndarray) -> None: ...

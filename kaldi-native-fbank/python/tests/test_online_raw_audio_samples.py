#!/usr/bin/env python3

try:
    import kaldi_native_fbank as knf
except (ImportError, OSError):  # pragma: no cover
    knf = None
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


_missing = []
if knf is None:
    _missing.append("kaldi_native_fbank")
if np is None:
    _missing.append("numpy")
if _missing:
    print("WARNING: %s not installed; skipping %s"
          % (", ".join(_missing), __file__))
    raise SystemExit(77)

def main():
    sampling_rate = 1000
    samples_per_frame = 30

    opts = knf.RawAudioSamplesOptions()
    opts.frame_opts.samp_freq = sampling_rate
    opts.frame_opts.frame_length_ms = samples_per_frame
    opts.frame_opts.frame_shift_ms = samples_per_frame
    opts.frame_opts.snip_edges = True

    raw_audio_samples = knf.OnlineRawAudioSamples(opts)

    samples = np.arange(sampling_rate * 10, dtype=np.float32).tolist()

    raw_audio_samples.accept_waveform(sampling_rate, samples)

    assert raw_audio_samples.num_frames_ready == raw_audio_samples.num_frames_ready
    for i in range(raw_audio_samples.num_frames_ready):
        f = raw_audio_samples.get_frame(i)
        f = np.array(f)
        expected = (
            np.arange(samples_per_frame, dtype=np.float32) + samples_per_frame * i
        )
        assert (f - expected).sum() == 0, (f, expected)


if __name__ == "__main__":
    main()

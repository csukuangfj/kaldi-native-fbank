#!/usr/bin/env python3
# Copyright (c) 2026 Julian Pscheid

import pickle

import kaldi_native_fbank as knf
import numpy as np


def test_options():
    opts = knf.NemoFbankOptions()
    opts.num_mel_bins = 128
    opts.buffered_capacity = 8
    assert "num_mel_bins: 128" in str(opts)
    for copy in (
        pickle.loads(pickle.dumps(opts)),
        knf.NemoFbankOptions.from_dict(opts.as_dict()),
    ):
        assert copy.as_dict() == opts.as_dict()


def test_centered_features():
    opts = knf.NemoFbankOptions()
    opts.num_mel_bins = 128
    fbank = knf.OnlineNemoFbank(opts)
    i = np.arange(1281)
    audio = (0.1 * np.cos(0.031 * i) + 0.025 * np.sin(0.137 * i)).astype(np.float32)
    audio[0] += 0.3
    audio[640] -= 0.2
    fbank.accept_waveform(16000, audio)
    fbank.input_finished()
    assert fbank.dim == 128
    assert abs(fbank.frame_shift_in_seconds - 0.01) < 1e-8
    assert fbank.num_frames_ready == 8
    assert fbank.num_trailing_samples == 1
    assert fbank.is_last_frame(7)
    frames, offset = fbank.get_frames(0, 8, True)
    assert frames.shape == (8, 128) and frames.dtype == np.float32
    assert offset == 0
    # Independent NumPy centered FFT, symmetric Hann, and Slaney mel fixture.
    expected = np.array(
        [
            [-7.6648445, -7.6372275, -6.3784108, -4.1939735],
            [-7.3691812, -9.5039988, -11.0779839, -8.8404980],
            [-7.3689537, -9.3721800, -12.4265671, -10.1413660],
            [-7.2432251, -9.3878937, -16.6143456, -16.6329727],
        ]
    )
    np.testing.assert_allclose(
        frames[np.ix_([0, 1, 3, 7], [0, 17, 63, 127])], expected, atol=2e-4, rtol=0
    )
    del fbank
    assert np.isfinite(frames).all()  # The returned array owns its data.


def test_partial_hop():
    for ready, remainder in ((8, 1), (8, 79), (11, 79)):
        opts = knf.NemoFbankOptions()
        opts.buffered_capacity = 8
        fbank = knf.OnlineNemoFbank(opts)
        audio = 0.1 * np.sin(0.037 * np.arange(ready * 160 + remainder))
        audio = audio.astype(np.float32)
        fbank.accept_waveform(16000, audio[:641])
        fbank.accept_waveform(16000, audio[641:])
        fbank.get_frames(0, 8)
        fbank.input_finished()
        actual, offset = fbank.get_frames(ready - 8, 8, True)
        assert offset == remainder
        reference = knf.OnlineNemoFbank(opts)
        reference.accept_waveform(16000, audio[-1280:])
        reference.input_finished()
        expected, offset = reference.get_frames(0, 8, True)
        assert offset == 0
        np.testing.assert_array_equal(actual, expected)


def test_empty():
    for n in (0, 79):
        fbank = knf.OnlineNemoFbank(knf.NemoFbankOptions())
        fbank.accept_waveform(16000, np.zeros(n, dtype=np.float32))
        fbank.input_finished()
        assert not fbank.is_last_frame(-1)
        assert not fbank.is_last_frame(0)
        frames, offset = fbank.get_frames(0, 0, True)
        assert frames.shape == (0, 80) and offset == 0


if __name__ == "__main__":
    test_options()
    test_centered_features()
    test_partial_hop()
    test_empty()

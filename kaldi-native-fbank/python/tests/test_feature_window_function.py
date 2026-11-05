#!/usr/bin/env python3
#
# Copyright (c)  2022-2023  Xiaomi Corporation (authors: Fangjun Kuang)

try:
    import kaldi_native_fbank as knf
except (ImportError, OSError):  # pragma: no cover
    knf = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


_missing = []
if knf is None:
    _missing.append("kaldi_native_fbank")
if torch is None:
    _missing.append("torch")
if _missing:
    print("WARNING: %s not installed; skipping %s"
          % (", ".join(_missing), __file__))
    raise SystemExit(77)

def test_hann_window():
    opts = knf.FrameExtractionOptions()
    assert opts.samp_freq == 16000
    assert opts.frame_shift_ms == 10.0
    assert opts.frame_length_ms == 25.0
    opts.window_type = "hann"

    n = int((opts.samp_freq * opts.frame_length_ms) / 1000)

    window = knf.FeatureWindowFunction(opts)
    assert len(window.window) == n
    w = torch.tensor(window.window)

    t = torch.hann_window(n, periodic=True)
    assert torch.allclose(w, t, atol=1e-6), (w - t).abs().max()


def main():
    test_hann_window()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation (author: Fangjun Kuang)
#
# Smoke test for the built wheels. Run by build-wheels-win-arm64.yaml,
# build-wheels-android-termux.yaml and build-wheels-armv7l.yaml after
# installing a freshly built wheel.
#
# It extracts real fbank features so that a wheel which cannot load its
# shared libraries (bad RPATH, missing DLL, wrong architecture) fails here
# rather than on a user's machine.
#
# numpy is optional: waveforms are passed in as plain Python lists, so this
# also runs inside the Termux images where numpy may not be packaged.

import math
import platform
import sys

import kaldi_native_fbank as knf


def main() -> int:
    print("kaldi_native_fbank:", knf.__version__, knf.__file__)
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("machine:", platform.machine())

    sample_rate = 16000

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = 80

    fbank = knf.OnlineFbank(opts)

    # 0.5 s of a 440 Hz sine, as a plain list so numpy stays optional.
    samples = [
        math.sin(2 * math.pi * 440 * i / sample_rate) for i in range(sample_rate // 2)
    ]
    fbank.accept_waveform(sample_rate, samples)

    num_frames_ready = fbank.num_frames_ready
    dim = fbank.dim
    print("num_frames_ready:", num_frames_ready)
    print("dim:", dim)

    assert num_frames_ready > 0, "expected at least one frame"
    assert dim == 80, f"expected 80 mel bins, got {dim}"

    try:
        import numpy as np
    except ImportError:
        print("numpy is not installed; skipping get_frame()")
    else:
        frame = np.asarray(fbank.get_frame(0))
        print("frame shape:", frame.shape)
        assert frame.shape == (80,), f"expected a frame of 80 bins, got {frame.shape}"
        assert np.isfinite(frame).all(), "frame contains non-finite values"

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

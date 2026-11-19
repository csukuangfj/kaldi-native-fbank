#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MFCC parity: kaldi's compute-mfcc-feats vs knf.OnlineMfcc.

Both sides see identical input: mono 16 kHz wav as float32 at *int16 scale*
(like kaldi's WaveReader and vosk -- no /32768 normalization), dither=0.

Usage:
  python3 parity_mfcc.py --kaldi-dir /path/to/kaldi/src \
      --wav a.wav --wav b.wav \
      --config /path/to/vosk-model/conf/mfcc.conf
"""

import argparse
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

import kaldi_native_fbank as knf
import kaldi_native_io


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi-dir", default=None,
                   help="kaldi src dir containing featbin/ online2bin/ "
                        "(default: $KALDI_ROOT/src, /opt/kaldi/src, or the "
                        "local checkout)")
    p.add_argument("--wav", action="append", required=True,
                   help="test wav (mono, 16 kHz); may be repeated")
    p.add_argument("--config", required=True,
                   help="mfcc.conf, e.g. vosk model conf/mfcc.conf")
    p.add_argument("--tolerance", type=float, default=1e-2)
    return p.parse_args()


def load_wav_int16_scale(path):
    """Return float32 samples at int16 scale (vosk/kaldi convention)."""
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1, path
        assert f.getsampwidth() == 2, path
        assert f.getframerate() == 16000, (path, f.getframerate())
        pcm = f.readframes(f.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32)


def run_kaldi(kaldi_dir, config, wav_paths):
    """compute-mfcc-feats --dither=0 --config=... ; returns {utt: [T, D]}"""
    bin_ = Path(kaldi_dir) / "featbin" / "compute-mfcc-feats"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav_scp = tmp / "wav.scp"
        wav_scp.write_text("".join(
            f"{p.stem} {p.resolve()}\n" for p in wav_paths))
        out_ark = tmp / "feats.ark"
        cmd = [str(bin_), "--dither=0", "--config=" + str(config),
               f"scp,p:{wav_scp}", f"ark:{out_ark}"]
        subprocess.run(cmd, check=True, capture_output=True)
        with kaldi_native_io.SequentialFloatMatrixReader(
                f"ark:{out_ark}") as r:
            return {k: np.array(v, dtype=np.float32, copy=True) for k, v in r}


def run_knf(samples):
    """knf.OnlineMfcc with the same recipe as vosk's mfcc.conf."""
    opts = knf.MfccOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0  # deterministic; see plan-features.md
    opts.num_ceps = 40
    opts.use_energy = False
    opts.mel_opts.num_bins = 40
    opts.mel_opts.low_freq = 40
    opts.mel_opts.high_freq = -200
    mfcc = knf.OnlineMfcc(opts)
    mfcc.accept_waveform(16000, samples)
    mfcc.input_finished()
    return np.stack([mfcc.get_frame(i)
                     for i in range(mfcc.num_frames_ready)]).astype(np.float32)


def default_kaldi_src():
    for cand in (os.environ.get("KALDI_ROOT"), "/opt/kaldi",
                 "/Users/fangjun/open-source/kaldi"):
        if cand and os.path.isdir(os.path.join(cand, "src")):
            return os.path.join(cand, "src")
    return None


def main():
    args = get_args()
    if args.kaldi_dir is None:
        args.kaldi_dir = default_kaldi_src()
        assert args.kaldi_dir, "no kaldi found; pass --kaldi-dir or set KALDI_ROOT"
    wav_paths = [Path(w) for w in args.wav]
    ref = run_kaldi(args.kaldi_dir, args.config, wav_paths)

    failed = False
    for p in wav_paths:
        got = run_knf(load_wav_int16_scale(p))
        want = ref[p.stem].astype(np.float32)
        t = min(len(got), len(want))
        diff = np.abs(got[:t] - want[:t])
        # snip_edges=true offline gives T = 1 + (n - win)//shift; knf's online
        # path pads the tail, so it may yield up to a couple of extra frames.
        count_delta = len(got) - len(want)
        max_abs = float(diff.max()) if t else 0.0
        ok = max_abs < args.tolerance and 0 <= count_delta <= 2
        failed |= not ok
        print(f"{p.name}: kaldi {want.shape} knf {got.shape} "
              f"max_abs_diff={max_abs:.6f} tail_frames={count_delta:+d} "
              f"{'OK' if ok else 'FAIL'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

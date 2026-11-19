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
"""OnlineCmvn parity: kaldi's apply-cmvn-online vs knf.OnlineCmvn.

Both sides see identical input features (kaldi-computed MFCC) and identical
initial stats (a model's ivector/global_cmvn.stats).  Several option sets are
compared, including a two-utterance speaker case that exercises state
handover via get_state()/OnlineCmvnState.

Usage:
  python3 parity_online_cmvn.py --kaldi-dir /path/to/kaldi/src \
      --wav a.wav --wav b.wav \
      --config /path/to/vosk-model/conf/mfcc.conf \
      --global-stats /path/to/vosk-model/ivector/global_cmvn.stats
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import kaldi_native_fbank as knf
import kaldi_native_io

from parity_mfcc import run_kaldi as compute_mfcc


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi-dir", default=None,
                   help="kaldi src dir containing featbin/ online2bin/ "
                        "(default: $KALDI_ROOT/src, /opt/kaldi/src, or the "
                        "local checkout)")
    p.add_argument("--wav", action="append", required=True,
                   help="test wav (mono, 16 kHz); may be repeated")
    p.add_argument("--config", required=True, help="mfcc.conf")
    p.add_argument("--global-stats", required=True,
                   help="e.g. <model>/ivector/global_cmvn.stats")
    p.add_argument("--tolerance", type=float, default=1e-3)
    return p.parse_args()


# (name, apply-cmvn-online options, knf OnlineCmvnOptions overrides)
CASES = [
    ("default", [], {}),
    ("cmn-window=100", ["--cmn-window=100"], {"cmn_window": 100}),
    ("norm-vars", ["--norm-vars"], {"normalize_variance": True}),
    ("skip-dims=0:1", ["--skip-dims=0:1"], {"skip_dims": "0:1"}),
]


def run_kaldi_cmvn(kaldi_dir, global_stats, feats_ark, options, spk2utt=None):
    """apply-cmvn-online <global-stats> ark:feats.ark ark:out.ark"""
    bin_ = Path(kaldi_dir) / "online2bin" / "apply-cmvn-online"
    with tempfile.TemporaryDirectory() as tmp:
        out_ark = Path(tmp) / "out.ark"
        cmd = [str(bin_)]
        if spk2utt is not None:
            cmd.append(f"--spk2utt=ark:{spk2utt}")
        cmd += options
        cmd += [str(global_stats), f"ark:{feats_ark}", f"ark:{out_ark}"]
        subprocess.run(cmd, check=True, capture_output=True)
        with kaldi_native_io.SequentialFloatMatrixReader(
                f"ark:{out_ark}") as r:
            return {k: np.array(v, dtype=np.float32, copy=True) for k, v in r}


def knf_cmvn(global_stats, feats_by_utt, overrides, use_state=False):
    """Run knf.OnlineCmvn; optionally chain utterances via get_state().

    feats_by_utt: list of [T, D] float32 arrays (one per utterance).
    Returns list of normalized [T, D] float32 arrays.
    """
    opts = knf.OnlineCmvnOptions(**overrides)
    state = knf.OnlineCmvnState(
        global_cmvn_stats=np.asarray(global_stats, dtype=np.float64))
    out = []
    for feats in feats_by_utt:
        cmvn = knf.OnlineCmvn(opts, state)
        out.append(cmvn.accept_features(feats))
        if use_state:
            state = cmvn.get_state()
    return out


def compare(name, want, got, tolerance):
    t = min(len(want), len(got))
    max_abs = float(np.abs(want[:t] - got[:t]).max()) if t else 0.0
    ok = max_abs < tolerance and len(want) == len(got)
    n_nan = int(np.isnan(want[:t]).sum()) + int(np.isnan(got[:t]).sum())
    print(f"  {name}: kaldi {want.shape} knf {got.shape} "
          f"max_abs_diff={max_abs:.6f} nan={n_nan} {'OK' if ok else 'FAIL'}")
    return ok


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
    global_stats = kaldi_native_io.FloatMatrix.read(
        str(args.global_stats)).numpy().astype(np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        feats_ark = Path(tmp) / "feats.ark"
        kaldi_dir = Path(args.kaldi_dir)
        wav_scp = Path(tmp) / "wav.scp"
        wav_scp.write_text("".join(
            f"{p.stem} {p.resolve()}\n" for p in wav_paths))
        subprocess.run(
            [str(kaldi_dir / "featbin" / "compute-mfcc-feats"), "--dither=0",
             "--config=" + str(args.config), f"scp,p:{wav_scp}",
             f"ark:{feats_ark}"],
            check=True, capture_output=True)
        with kaldi_native_io.SequentialFloatMatrixReader(
                f"ark:{feats_ark}") as r:
            feats = {k: np.array(v, dtype=np.float32, copy=True) for k, v in r}

        failed = False
        for case_name, cli_opts, overrides in CASES:
            print(f"case: {case_name}")
            want = run_kaldi_cmvn(args.kaldi_dir, args.global_stats, feats_ark,
                                  cli_opts)
            got = knf_cmvn(global_stats,
                           [feats[p.stem] for p in wav_paths], overrides)
            for p, g in zip(wav_paths, got):
                failed |= not compare(p.stem, want[p.stem], g,
                                      args.tolerance)

        # Speaker case: two utterances of one speaker, state handover.
        print("case: speaker-state handover (utt2 adapts on utt1)")
        spk2utt = Path(tmp) / "spk2utt"
        spk2utt.write_text("spk " + " ".join(p.stem for p in wav_paths) + "\n")
        want = run_kaldi_cmvn(args.kaldi_dir, args.global_stats, feats_ark,
                              [], spk2utt=spk2utt)
        got = knf_cmvn(global_stats,
                       [feats[p.stem] for p in wav_paths], {},
                       use_state=True)
        for p, g in zip(wav_paths, got):
            failed |= not compare(p.stem, want[p.stem], g, args.tolerance)

        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

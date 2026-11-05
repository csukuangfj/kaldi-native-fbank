#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
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

import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import kaldi_native_fbank as knf
except (ImportError, OSError):  # pragma: no cover
    knf = None

GOLDEN = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "feat_golden" / "golden"
)


_missing = []
if knf is None:
    _missing.append("kaldi_native_fbank")
if np is None:
    _missing.append("numpy")
if _missing:
    print("WARNING: %s not installed; skipping %s"
          % (", ".join(_missing), __file__))
    raise SystemExit(77)
class TestFeatureCmvn(unittest.TestCase):
    def _load(self, *parts):
        return np.load(GOLDEN.joinpath(*parts))

    def test_stats_layout(self):
        stats = knf.compute_cmvn_stats(np.zeros((3, 5), dtype=np.float32))
        self.assertEqual(stats.shape, (2, 6))
        self.assertEqual(float(stats[0, 5]), 3.0)  # count

    def test_golden_aishell3_stats_and_cmvn(self):
        feats = self._load("aishell3_typical", "mfcc.utt.npy").astype(np.float32)
        stats = knf.compute_cmvn_stats(feats)
        ref = self._load("aishell3_typical", "cmvn_stats.utt.npy")
        self.assertEqual(stats.shape, ref.shape)
        np.testing.assert_allclose(stats, ref, rtol=1e-5, atol=1e-5)

        opts = knf.CmvnOptions()  # norm_vars=False
        out = knf.apply_cmvn(stats, opts, feats)
        ref_out = self._load("aishell3_typical", "cmvn.utt.npy")
        np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)

    def test_norm_vars_zero_mean_unit_var(self):
        feats = self._load("aishell3_typical", "mfcc.utt.npy").astype(np.float32)
        stats = knf.compute_cmvn_stats(feats)
        opts = knf.CmvnOptions()
        opts.norm_vars = True
        out = knf.apply_cmvn(stats, opts, feats)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-3)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-2)

    def test_reverse_roundtrip(self):
        feats = self._load("aishell3_typical", "mfcc.utt.npy").astype(np.float32)
        stats = knf.compute_cmvn_stats(feats)
        fwd = knf.CmvnOptions()
        rev = knf.CmvnOptions()
        rev.reverse = True
        out = knf.apply_cmvn(stats, fwd, feats)
        back = knf.apply_cmvn(stats, rev, out)
        np.testing.assert_allclose(back, feats, rtol=1e-4, atol=1e-3)

    def test_golden_edge_silence(self):
        feats = self._load(
            "edge_cases", "silence_0.5s.mfcc.utt.npy"
        ).astype(np.float32)
        stats = knf.compute_cmvn_stats(feats)
        ref = self._load("edge_cases", "silence_0.5s.cmvn_stats.utt.npy")
        np.testing.assert_allclose(stats, ref, rtol=1e-5, atol=1e-5)
        # apply-cmvn must reproduce the golden output
        ref_out = self._load("edge_cases", "silence_0.5s.cmvn.utt.npy")
        opts = knf.CmvnOptions()
        out = knf.apply_cmvn(stats, opts, feats)
        np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)

    def test_apply_cmvn_inplace(self):
        feats = self._load("aishell3_typical", "mfcc.utt.npy").astype(np.float32)
        stats = knf.compute_cmvn_stats(feats)
        opts = knf.CmvnOptions()
        expected = knf.apply_cmvn(stats, opts, feats)
        got = feats.copy()
        knf.apply_cmvn_(stats, opts, got)
        self.assertFalse(np.allclose(got, feats))
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_accumulator_matches_batch(self):
        feats = self._load("aishell3_typical", "mfcc.utt.npy").astype(np.float32)
        acc = knf.CmvnAccumulator()
        acc.accumulate(feats)
        np.testing.assert_allclose(
            acc.stats, knf.compute_cmvn_stats(feats), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()

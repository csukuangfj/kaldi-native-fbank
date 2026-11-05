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
class TestDelta(unittest.TestCase):
    def _load(self, *parts):
        return np.load(GOLDEN.joinpath(*parts))

    def _check_golden(self, group, name):
        feats = self._load(group, name.replace("deltas", "cmvn"))
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        ref = self._load(group, name)
        got = knf.compute_deltas(knf.DeltaOptions(), feats)
        self.assertEqual(got.shape, ref.shape)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_hand_case(self):
        feats = np.array([[1], [2], [3]], dtype=np.float32)
        opts = knf.DeltaOptions(order=1, window=1)
        got = knf.compute_deltas(opts, feats)
        self.assertEqual(got.shape, (3, 2))
        np.testing.assert_allclose(
            got, [[1, 0.5], [2, 1.0], [3, 0.5]], rtol=1e-6, atol=1e-6
        )

    def test_order_zero_is_baseline(self):
        feats = np.random.RandomState(0).randn(4, 3).astype(np.float32)
        out = knf.compute_deltas(knf.DeltaOptions(order=0), feats)
        np.testing.assert_array_equal(out, feats)

    def test_defaults_are_visible(self):
        doc = knf.DeltaOptions.__init__.__doc__
        self.assertIn("order", doc)
        self.assertIn("window", doc)

    def test_golden_aishell3_typical(self):
        self._check_golden("aishell3_typical", "deltas.utt.npy")

    def test_golden_english_mfa_typical(self):
        self._check_golden("english_mfa_typical", "deltas.utt.npy")

    def test_golden_edge_short_2f(self):
        self._check_golden("edge_cases", "short_2f.deltas.utt.npy")

    def test_golden_edge_silence(self):
        self._check_golden("edge_cases", "silence_0.5s.deltas.utt.npy")

    def test_errors(self):
        with self.assertRaises(ValueError):
            knf.compute_deltas(
                knf.DeltaOptions(), np.zeros((0, 3), dtype=np.float32)
            )
        with self.assertRaises(ValueError):
            knf.compute_deltas(
                knf.DeltaOptions(order=-1), np.zeros((2, 2), dtype=np.float32)
            )
        with self.assertRaises(ValueError):
            knf.compute_deltas(
                knf.DeltaOptions(window=0), np.zeros((2, 2), dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()

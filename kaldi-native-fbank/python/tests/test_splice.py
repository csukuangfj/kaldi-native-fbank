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

class TestSplice(unittest.TestCase):
    def _load(self, *parts):
        return np.load(GOLDEN.joinpath(*parts))

    def _check_golden(self, *parts):
        feats = self._load(*parts[:-1], parts[-1].replace("spliced", "cmvn"))
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        ref = self._load(*parts)
        got = knf.splice_frames(feats, knf.SpliceOptions())
        self.assertEqual(got.shape, ref.shape)
        np.testing.assert_array_equal(got, ref)

    def test_hand_case_edge_replication(self):
        feats = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        opts = knf.SpliceOptions(left_context=1, right_context=1)
        got = knf.splice_frames(feats, opts)
        self.assertEqual(got.shape, (3, 6))
        expected = np.array(
            [
                [1, 2, 1, 2, 3, 4],
                [1, 2, 3, 4, 5, 6],
                [3, 4, 5, 6, 5, 6],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(got, expected)

    def test_single_frame(self):
        feats = np.array([[7, 8]], dtype=np.float32)
        opts = knf.SpliceOptions(left_context=2, right_context=0)
        got = knf.splice_frames(feats, opts)
        np.testing.assert_array_equal(got, [[7, 8, 7, 8, 7, 8]])

    def test_defaults_are_visible(self):
        import inspect

        doc = knf.SpliceOptions.__init__.__doc__
        self.assertIn("left_context", doc)
        self.assertIn("right_context", doc)

    def test_golden_aishell3_typical(self):
        self._check_golden("aishell3_typical", "spliced.utt.npy")

    def test_golden_english_mfa_typical(self):
        self._check_golden("english_mfa_typical", "spliced.utt.npy")

    def test_golden_mandarin_mfa_pitch(self):
        self._check_golden("mandarin_mfa_pitch", "spliced.utt.npy")

    def test_golden_edge_short_2f(self):
        self._check_golden("edge_cases", "short_2f.spliced.utt.npy")

    def test_golden_edge_silence(self):
        self._check_golden("edge_cases", "silence_0.5s.spliced.utt.npy")

    def test_errors(self):
        with self.assertRaises(ValueError):
            knf.splice_frames(np.zeros((0, 3), dtype=np.float32), knf.SpliceOptions())
        with self.assertRaises(ValueError):
            knf.splice_frames(
                np.zeros((2, 2), dtype=np.float32),
                knf.SpliceOptions(left_context=-1),
            )


if __name__ == "__main__":
    unittest.main()

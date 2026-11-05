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

_missing = []
if knf is None:
    _missing.append("kaldi_native_fbank")
if np is None:
    _missing.append("numpy")
if _missing:
    print("WARNING: %s not installed; skipping %s"
          % (", ".join(_missing), __file__))
    raise SystemExit(77)

GOLDEN = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "feat_golden"
)


class TestFeatureTransform(unittest.TestCase):
    def _load(self, *parts):
        return np.load(GOLDEN.joinpath(*parts))

    def test_read_matches_npy(self):
        # The golden .mat must hold exactly the golden .npy values.
        t91 = knf.FeatureTransform.read(str(GOLDEN / "lda_test_91.mat"))
        ref = self._load("lda_test_91.npy")
        self.assertEqual(t91.output_dim, ref.shape[0])
        self.assertEqual(t91.input_dim, ref.shape[1])
        eye = np.eye(ref.shape[1], dtype=np.float32)
        out = t91.apply(eye)
        np.testing.assert_allclose(out, ref.T, rtol=1e-5, atol=1e-5)

    def test_dims_and_affine(self):
        t91 = knf.FeatureTransform.read(str(GOLDEN / "lda_test_91.mat"))
        self.assertEqual(t91.input_dim, 91)
        self.assertEqual(t91.output_dim, 40)
        self.assertFalse(t91.is_affine(91))
        self.assertTrue(t91.is_affine(90))

        t112 = knf.FeatureTransform.read(str(GOLDEN / "lda_test_112.mat"))
        self.assertEqual(t112.input_dim, 112)
        self.assertEqual(t112.output_dim, 40)
        self.assertFalse(t112.is_affine(112))

    def _check_golden(self, group, mat, spliced, lda_name):
        feats = self._load("golden", group, spliced)
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        ref = self._load("golden", group, lda_name)
        transform = knf.FeatureTransform.read(str(GOLDEN / mat))
        got = transform.apply(feats)
        self.assertEqual(got.shape, ref.shape)
        # The LDA matmul accumulates >= 91 terms; kaldi's BLAS and Eigen
        # order the sums differently, giving per-platform float noise of up
        # to a few 1e-5 (see kGoldenDotProductTol in csrc/test-golden.h).
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
        # the module-level shorthand must match
        np.testing.assert_allclose(
            knf.transform_features(feats, transform), got, rtol=1e-5, atol=1e-5
        )

    def test_golden_aishell3_typical(self):
        self._check_golden(
            "aishell3_typical", "lda_test_91.mat", "spliced.utt.npy",
            "lda.utt.npy"
        )

    def test_golden_english_mfa_typical(self):
        self._check_golden(
            "english_mfa_typical", "lda_test_91.mat", "spliced.utt.npy",
            "lda.utt.npy"
        )

    def test_golden_mandarin_mfa_pitch(self):
        self._check_golden(
            "mandarin_mfa_pitch", "lda_test_112.mat", "spliced.utt.npy",
            "lda.utt.npy"
        )

    def test_golden_edge_short_2f(self):
        self._check_golden(
            "edge_cases", "lda_test_91.mat", "short_2f.spliced.utt.npy",
            "short_2f.lda.utt.npy"
        )

    def test_hand_case_affine(self):
        mat = np.array([[1, 2, 3, 0.5], [4, 5, 6, -1.5]], dtype=np.float32)
        t = knf.FeatureTransform(mat)
        self.assertTrue(t.is_affine(3))
        feats = np.array([[1, 0, 1]], dtype=np.float32)
        out = t.apply(feats)
        np.testing.assert_allclose(out, [[4.5, 8.5]], rtol=1e-6, atol=1e-6)

    def test_errors(self):
        t91 = knf.FeatureTransform.read(str(GOLDEN / "lda_test_91.mat"))
        with self.assertRaises(ValueError):
            t91.apply(np.zeros((2, 7), dtype=np.float32))
        with self.assertRaises(RuntimeError):
            knf.FeatureTransform.read(str(GOLDEN / "no_such_file.mat"))


if __name__ == "__main__":
    unittest.main()

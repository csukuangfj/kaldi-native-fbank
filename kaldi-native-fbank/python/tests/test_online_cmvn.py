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


def make_global_stats(dim, count=1000.0):
    """Global stats with zero mean and unit variance per dim."""
    stats = np.zeros((2, dim + 1), dtype=np.float64)
    stats[0, dim] = count
    stats[1, :dim] = count  # sum-squares => variance 1
    return stats


class TestOnlineCmvn(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.dim = 6
        self.feats = (rng.standard_normal((200, self.dim)) * 3 + 1).astype(
            np.float32)

    def _make(self, **overrides):
        opts = knf.OnlineCmvnOptions(**overrides)
        state = knf.OnlineCmvnState(
            global_cmvn_stats=make_global_stats(self.dim))
        return knf.OnlineCmvn(opts, state)

    def test_options_defaults_match_kaldi(self):
        opts = knf.OnlineCmvnOptions()
        self.assertEqual(opts.cmn_window, 600)
        self.assertEqual(opts.speaker_frames, 600)
        self.assertEqual(opts.global_frames, 200)
        self.assertTrue(opts.normalize_mean)
        self.assertFalse(opts.normalize_variance)

    def test_shapes_and_count(self):
        cmvn = self._make()
        out = cmvn.accept_features(self.feats)
        self.assertEqual(out.shape, self.feats.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(cmvn.num_frames_accepted, self.feats.shape[0])

    def test_output_is_finite_and_centered(self):
        # global_frames=0: normalization uses only the running window, so the
        # tail is centered (with global backoff the mean is only reduced).
        cmvn = self._make(cmn_window=100, global_frames=0)
        out = cmvn.accept_features(self.feats)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(float(np.abs(out[-100:].mean(axis=0)).max()), 0.3)

    def test_skip_dims_untouched(self):
        cmvn = self._make(skip_dims="0:2")
        out = cmvn.accept_features(self.feats)
        np.testing.assert_array_equal(out[:, 0], self.feats[:, 0])
        np.testing.assert_array_equal(out[:, 2], self.feats[:, 2])

    def test_freeze(self):
        cmvn = self._make()
        cmvn.accept_features(self.feats)
        cmvn.freeze()
        out = cmvn.accept_features(self.feats[:10])
        # with frozen stats the normalization is a fixed offset per dim
        diffs = self.feats[:10] - out
        self.assertLess(float(np.abs(diffs - diffs[0]).max()), 1e-5)

    def test_state_handover_frozen(self):
        # After Freeze(), the state carries the frozen normalization to the
        # next utterance: chained output equals the single-instance output
        # from the freeze point on.
        one = self._make()
        one.accept_features(self.feats[:80])
        one.freeze()
        tail = one.accept_features(self.feats[80:])

        first = self._make()
        first.accept_features(self.feats[:80])
        first.freeze()
        second = knf.OnlineCmvn(knf.OnlineCmvnOptions(), first.get_state())
        tail2 = second.accept_features(self.feats[80:])
        np.testing.assert_allclose(tail2, tail, rtol=1e-5, atol=1e-5)

    def test_state_handover_uses_speaker_stats(self):
        # Speaker stats from the previous utterance change the normalization
        # of the next one (compared with a fresh, speaker-less start).
        first = self._make()
        first.accept_features(self.feats[:80])
        second = knf.OnlineCmvn(knf.OnlineCmvnOptions(), first.get_state())
        out_adapted = second.accept_features(self.feats[:80])
        fresh = self._make()
        out_fresh = fresh.accept_features(self.feats[:80])
        self.assertFalse(np.allclose(out_adapted, out_fresh))

    def test_normalize_variance_unit_scale(self):
        cmvn = self._make(cmn_window=100, normalize_variance=True)
        out = cmvn.accept_features(self.feats)
        # with variance normalization the tail has unit-ish variance
        self.assertLess(float(np.abs(out[-100:].std(axis=0) - 1).max()), 0.2)

    def test_matches_naive_reference(self):
        # Randomized comparison against a brute-force reference, following
        # kaldi's UnitTestOnlineCmvn (feat/feature-functions-test.cc) and the
        # C++ test MatchesNaiveReference.
        rng = np.random.default_rng(1234)
        for it in range(200):
            T = 1 + it % 7 * 10
            dim = 1 + it % 8
            opts = knf.OnlineCmvnOptions(
                cmn_window=5 + it % 45,
                speaker_frames=1 + it % 60,
                global_frames=1 + it % 30,
                normalize_variance=(it % 2 == 0),
                skip_dims="0" if it % 5 == 0 else "")
            with_speaker = it % 3 == 0

            feats = rng.uniform(-10, 10, (T, dim)).astype(np.float32)
            g = rng.uniform(-10, 10, (50, dim))
            glob = np.zeros((2, dim + 1), dtype=np.float64)
            glob[0, :dim] = g.sum(axis=0)
            glob[1, :dim] = (g * g).sum(axis=0)
            glob[0, dim] = 50.0
            spk = np.zeros((2, dim + 1), dtype=np.float64)
            if with_speaker:
                sp = rng.uniform(-10, 10, (30, dim))
                spk[0, :dim] = sp.sum(axis=0)
                spk[1, :dim] = (sp * sp).sum(axis=0)
                spk[0, dim] = 30.0

            state = knf.OnlineCmvnState(global_cmvn_stats=glob)
            if with_speaker:
                state.speaker_cmvn_stats = spk
            cmvn = knf.OnlineCmvn(opts, state)
            out = cmvn.accept_features(feats)

            for t in range(T):
                lo = max(0, t - opts.cmn_window + 1)
                n = t - lo + 1
                for d in range(dim):
                    if opts.skip_dims and d == 0:
                        self.assertEqual(out[t, d], feats[t, d])
                        continue
                    w = feats[lo:t + 1, d].astype(np.float64)
                    sum_, sumsq, count = float(w.sum()), float((w * w).sum()), float(n)
                    if with_speaker:
                        take = min(opts.cmn_window - count,
                                   opts.speaker_frames, spk[0, dim])
                        if take > 0:
                            sum_ += take / spk[0, dim] * spk[0, d]
                            if opts.normalize_variance:
                                sumsq += take / spk[0, dim] * spk[1, d]
                            count += take
                    if count < opts.cmn_window:
                        take = min(opts.cmn_window - count,
                                   opts.global_frames, glob[0, dim])
                        if take > 0:
                            sum_ += take / glob[0, dim] * glob[0, d]
                            if opts.normalize_variance:
                                sumsq += take / glob[0, dim] * glob[1, d]
                            count += take
                    mean = sum_ / count
                    want = feats[t, d] - mean
                    if opts.normalize_variance:
                        var = max(sumsq / count - mean * mean, 1.0e-20)
                        want = (feats[t, d] - mean) / np.sqrt(var)
                    self.assertAlmostEqual(float(out[t, d]), float(want),
                                           places=4,
                                           msg=f"it={it} t={t} d={d}")

    def test_utterance_start_global_backing(self):
        # frame 0 with too little data: normalization backs off to global stats
        glob = np.zeros((2, 2), dtype=np.float64)
        glob[0, 0] = 50.0
        glob[0, 1] = 100.0  # global mean 0.5
        cmvn = knf.OnlineCmvn(knf.OnlineCmvnOptions(),
                              knf.OnlineCmvnState(global_cmvn_stats=glob))
        y = cmvn.accept_feature(np.array([2.0], dtype=np.float32))
        # window sum=2 count=1; global takes 200 frames worth:
        # sum += 200/100*50 -> 102, count 201
        want = 2.0 - (2.0 + 200.0 / 100.0 * 50.0) / (1.0 + 200.0)
        self.assertAlmostEqual(float(y[0]), want, places=5)

    def test_utterance_start_speaker_backing(self):
        glob = np.zeros((2, 2), dtype=np.float64)
        glob[0, 0] = 50.0
        glob[0, 1] = 100.0
        spk = np.zeros((2, 2), dtype=np.float64)
        spk[0, 0] = 30.0
        spk[0, 1] = 10.0
        state = knf.OnlineCmvnState(global_cmvn_stats=glob)
        state.speaker_cmvn_stats = spk
        opts = knf.OnlineCmvnOptions(speaker_frames=5)
        cmvn = knf.OnlineCmvn(opts, state)
        y = cmvn.accept_feature(np.array([2.0], dtype=np.float32))
        # speaker takes 5 frames worth (5/10*30 = 15), global takes 200
        want = 2.0 - (2.0 + 15.0 + 100.0) / (1.0 + 5.0 + 200.0)
        self.assertAlmostEqual(float(y[0]), want, places=5)

    def test_first_frames_partial_window(self):
        # no back-off at all: frame t normalized with mean of frames 0..t
        opts = knf.OnlineCmvnOptions(cmn_window=5, global_frames=0)
        state = knf.OnlineCmvnState(
            global_cmvn_stats=make_global_stats(1))
        cmvn = knf.OnlineCmvn(opts, state)
        feats = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
        out = cmvn.accept_features(feats)
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)
        for t in range(8):
            lo = max(0, t - 5 + 1)
            want = feats[t, 0] - feats[lo:t + 1, 0].mean()
            self.assertAlmostEqual(float(out[t, 0]), float(want), places=4)

    def test_smooth_online_cmvn_stats(self):
        stats = np.zeros((2, self.dim + 1), dtype=np.float64)
        stats[0, self.dim] = 10  # window count
        smoothed = knf.OnlineCmvn.smooth_online_cmvn_stats(
            np.zeros((0, 0)), make_global_stats(self.dim),
            knf.OnlineCmvnOptions(global_frames=100), stats)
        # count is 10 (window) + 100 (global contribution)
        self.assertAlmostEqual(float(smoothed[0, self.dim]), 110.0)

    def test_invalid_options_raise(self):
        with self.assertRaises(ValueError):
            self._make(normalize_mean=False, normalize_variance=True)
        with self.assertRaises(ValueError):
            knf.OnlineCmvn(knf.OnlineCmvnOptions(),
                           knf.OnlineCmvnState())  # no global stats


if __name__ == "__main__":
    unittest.main()

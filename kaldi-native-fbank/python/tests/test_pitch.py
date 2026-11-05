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

GOLDEN = Path(__file__).resolve().parents[3] / "tests" / "data" / "feat_golden"

_missing = []
if knf is None:
    _missing.append("kaldi_native_fbank")
if np is None:
    _missing.append("numpy")
if _missing:
    print("WARNING: %s not installed; skipping %s"
          % (", ".join(_missing), __file__))
    raise SystemExit(77)


class TestPitch(unittest.TestCase):
    def _samples(self):
        # Raw int16 magnitudes as float32 (kaldi semantics: no /32768).
        s = np.load(GOLDEN / "zh0_1s.npy")
        return np.ascontiguousarray(s, dtype=np.float32)

    def _ref(self):
        return np.load(
            GOLDEN / "golden" / "mandarin_mfa_pitch" / "pitch.utt.npy"
        )

    @staticmethod
    def _opts():
        # golden group mandarin_mfa_pitch (see README.md in feat_golden):
        # compute-and-process-kaldi-pitch-feats --sample-frequency=16000
        #   --frame-shift=10 --frame-length=25 --min-f0=50 --max-f0=500
        #   --delta-pitch=0.005 --penalty-factor=0.1 --snip-edges=true
        #   --add-normalized-log-pitch=true --add-delta-pitch=true
        return knf.PitchExtractionOptions(
            samp_freq=16000,
            frame_shift_ms=10,
            frame_length_ms=25,
            min_f0=50,
            max_f0=500,
            delta_pitch=0.005,
            penalty_factor=0.1,
            snip_edges=True,
        )

    def test_batch_matches_golden(self):
        ref = self._ref()
        got = knf.compute_and_process_kaldi_pitch(
            self._opts(), knf.ProcessPitchOptions(), self._samples()
        )
        self.assertEqual(got.shape, ref.shape)
        # kGoldenDotProductTol equivalent: the NCCF dot products accumulate
        # >= 100 terms, so kaldi's BLAS and Eigen disagree by up to a few
        # 1e-5 depending on platform (see csrc/test-golden.h).
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_streaming_matches_golden(self):
        ref = self._ref()
        pitch = knf.OnlinePitch(self._opts(), knf.ProcessPitchOptions())
        self.assertEqual(pitch.dim, 3)
        pitch.accept_waveform(16000, self._samples())
        pitch.input_finished()
        self.assertEqual(pitch.num_frames_ready, ref.shape[0])
        self.assertTrue(pitch.is_last_frame(pitch.num_frames_ready - 1))

        got = np.stack(
            [pitch.get_frame(t) for t in range(pitch.num_frames_ready)]
        )
        self.assertEqual(got.shape, ref.shape)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_options_defaults(self):
        opts = knf.PitchExtractionOptions()
        self.assertEqual(opts.nccf_window_size, 100)  # 4000 * 25 / 1000
        self.assertEqual(opts.nccf_window_shift, 40)  # 4000 * 10 / 1000
        self.assertTrue(opts.snip_edges)  # kaldi pitch default

        popts = knf.ProcessPitchOptions()
        self.assertEqual(popts.dim, 3)

        popts = knf.ProcessPitchOptions(
            add_pov_feature=False, add_delta_pitch=False,
            add_raw_log_pitch=True,
        )
        self.assertEqual(popts.dim, 2)


if __name__ == "__main__":
    unittest.main()

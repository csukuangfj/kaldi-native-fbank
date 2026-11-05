# feat_golden — Golden test data for feature components

Reference outputs generated with **kaldi pre-built binaries**. Purpose: unit/golden
tests — knf's C++/Python implementations must match these goldens **value-by-value**
(tolerance <= 1e-5) with **dither=0 and identical parameters**.

**Regenerate** (needs kaldi `src/featbin` at `$KALDI_BIN`):

```bash
cd tests/data/feat_golden
KALDI_BIN=/path/to/kaldi/src/featbin bash generate.sh
```

## Test signals (raw int16 wav + same-content .npy)

| File | Content |
|---|---|
| `zh0_1s.wav` / `en0_1s.wav` | first 1.5s of real speech (16kHz) |
| `silence_0.5s.wav` | all zeros |
| `dc_0.5s.wav` | constant 500 |
| `sine_0.5s.wav` | 220Hz sine |
| `short_2f.wav` | 560 samples (exactly 2 frames with snip_edges=true; edge case) |

**Note**: kaldi's wav reading uses **raw int16 amplitude** (no /32768 normalization).
Feed samples at the same scale on the implementation side, otherwise the energy
scale differs by 32768^2.

## Parameter groups and generation commands

### 1) `aishell3_typical` — use_energy=true, snip_edges=false (aishell3 / large_pinyin)
```bash
compute-mfcc-feats --dither=0.0 --use-energy=true --snip-edges=false \
  --num-ceps=13 --num-mel-bins=23 --cepstral-lifter=22 --energy-floor=1.0 \
  --low-freq=20 --high-freq=7800 --sample-frequency=16000 --preemphasis-coefficient=0.97 \
  scp:wav.scp ark:golden/aishell3_typical/mfcc.ark
compute-cmvn-stats ark:.../mfcc.ark ark:.../cmvn_stats.ark
apply-cmvn --utt2spk=ark:utt2spk ark:.../cmvn_stats.ark ark:.../mfcc.ark ark:.../cmvn.ark
splice-feats --left-context=3 --right-context=3 ark:.../cmvn.ark ark:.../spliced.ark
add-deltas ark:.../cmvn.ark ark:.../deltas.ark
transform-feats lda_test_91.mat ark:.../spliced.ark ark:.../lda.ark
```

### 2) `english_mfa_typical` — use_energy=false, snip_edges=true (english_mfa)
Same command chain as 1) with `--use-energy=false --snip-edges=true`, signal `en0_1s.wav`.

### 3) `mandarin_mfa_pitch` — pitch enabled (mandarin_mfa-style)
```bash
compute-mfcc-feats --dither=0.0 --use-energy=false --snip-edges=true ... (as above)
compute-and-process-kaldi-pitch-feats --sample-frequency=16000 \
  --frame-shift=10 --frame-length=25 --min-f0=50 --max-f0=500 \
  --delta-pitch=0.005 --penalty-factor=0.1 --snip-edges=true \
  --add-normalized-log-pitch=true --add-delta-pitch=true \
  scp:wav.scp ark:golden/mandarin_mfa_pitch/pitch.ark   # 3 dims: POV+norm-log-pitch+delta
paste-feats ark:mfcc.ark ark:pitch.ark ark:base.ark      # 16 dims
compute-cmvn-stats / apply-cmvn / splice-feats (+-3 -> 112 dims)
transform-feats lda_test_112.mat ark:spliced.ark ark:lda.ark   # 40 dims
```

### 4) `edge_cases` — boundary cases (silence/dc/sine/short; snip_edges=true, use_energy=true)
Same chain as 1) run on each of the 4 signals; outputs named `<signal>.<component>.ark`.

## Test LDA matrices (deterministic, fixed seed)

`lda_test_91.mat` (40x91) and `lda_test_112.mat` (40x112): gaussian matrix from
`np.random.RandomState(42)` divided by sqrt(dim), stored in kaldi binary format;
matching `.npy` files for python tests.

## Passing criteria

Value-by-value agreement (tolerance <= 1e-5) with dither=0 and identical parameters.
Any larger deviation is an implementation difference and must be investigated.
(The pitch delta carries a small noise term in kaldi itself; if a comparison exceeds
tolerance, localize the component before adjusting tolerance.)

Localized deviations (investigated; tolerance adjusted):

* **LDA (`lda.utt.npy`) and pitch (`pitch.utt.npy`) comparisons use 1e-4**
  (`kGoldenDotProductTol` in `kaldi-native-fbank/csrc/test-golden.h`): their
  kaldi references are produced by matrix multiplications / long dot
  products (>= 91 terms for LDA, >= 100 for the pitch NCCF), and kaldi's
  BLAS and Eigen accumulate the sums in different orders.  This is plain
  float summation-order noise: observed maxima per platform are 1.5e-5
  (linux-arm32), 1.1e-5 (windows-x64), 2.7e-5 (windows-arm64), while the
  golden-generating platform agrees at <= 1e-5.  1e-4 leaves headroom but
  still catches real implementation differences (>= 1e-3).
* The pitch delta noise is reproduced deterministically in knf
  (see `feature-pitch.cc`), so it does not contribute deviation at all.
* Sequential-reduction pipelines (cmvn/delta/splice) stay at 1e-5.

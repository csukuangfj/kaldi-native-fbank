#!/usr/bin/env bash
# Golden test data generator for knf feature components.
# Every command below is documented in README.md.  Regenerate with:
#   cd tests/data/feat_golden && bash generate.sh
# Requires kaldi binaries in $KALDI_BIN (default: kaldi/src/featbin).
set -euo pipefail
KALDI_BIN=${KALDI_BIN:-/Users/fangjun/open-source/kaldi/src/featbin}
cd "$(dirname "$0")"

gen_group() {
  local g=$1 wav=$2; shift 2
  mkdir -p "golden/$g"
  printf "utt %s\n" "$PWD/$wav" > /tmp/g.scp
  # 1) MFCC (dither=0: must match implementation value-by-value)
  "$KALDI_BIN/compute-mfcc-feats" --dither=0.0 "$@" scp:/tmp/g.scp "ark:golden/$g/mfcc.ark"
  # 2) CMVN stats + apply (speaker = utterance itself)
  "$KALDI_BIN/compute-cmvn-stats" "ark:golden/$g/mfcc.ark" "ark:golden/$g/cmvn_stats.ark"
  printf "utt utt\n" > /tmp/g.utt2spk
  "$KALDI_BIN/apply-cmvn" --utt2spk=ark:/tmp/g.utt2spk "ark:golden/$g/cmvn_stats.ark" \
      "ark:golden/$g/mfcc.ark" "ark:golden/$g/cmvn.ark"
  # 3) splice +-3
  "$KALDI_BIN/splice-feats" --left-context=3 --right-context=3 "ark:golden/$g/cmvn.ark" "ark:golden/$g/spliced.ark"
  # 4) add-deltas (non-LDA path)
  "$KALDI_BIN/add-deltas" "ark:golden/$g/cmvn.ark" "ark:golden/$g/deltas.ark"
  # 5) LDA transform
  "$KALDI_BIN/transform-feats" "$PWD/lda_test_91.mat" "ark:golden/$g/spliced.ark" "ark:golden/$g/lda.ark"
}

# Group 1: aishell3-style (use_energy=true, snip_edges=false, no pitch)
gen_group aishell3_typical zh0_1s.wav \
  --use-energy=true --snip-edges=false --num-ceps=13 --num-mel-bins=23 \
  --cepstral-lifter=22 --energy-floor=1.0 --low-freq=20 --high-freq=7800 \
  --sample-frequency=16000 --preemphasis-coefficient=0.97

# Group 2: english_mfa-style (use_energy=false, snip_edges=true, no pitch)
gen_group english_mfa_typical en0_1s.wav \
  --use-energy=false --snip-edges=true --num-ceps=13 --num-mel-bins=23 \
  --cepstral-lifter=22 --energy-floor=1.0 --low-freq=20 --high-freq=7800 \
  --sample-frequency=16000 --preemphasis-coefficient=0.97

# Group 3: mandarin_mfa-style (pitch on) -- mfcc + pitch(3-dim) goldens
g=mandarin_mfa_pitch; mkdir -p golden/$g
printf "utt %s\n" "$PWD/zh0_1s.wav" > /tmp/g.scp
"$KALDI_BIN/compute-mfcc-feats" --dither=0.0 --use-energy=false --snip-edges=true \
  --num-ceps=13 --num-mel-bins=23 --cepstral-lifter=22 --energy-floor=1.0 \
  --low-freq=20 --high-freq=7800 --sample-frequency=16000 --preemphasis-coefficient=0.97 \
  scp:/tmp/g.scp "ark:golden/$g/mfcc.ark"
"$KALDI_BIN/compute-and-process-kaldi-pitch-feats" --sample-frequency=16000 \
  --frame-shift=10 --frame-length=25 --min-f0=50 --max-f0=500 \
  --delta-pitch=0.005 --penalty-factor=0.1 --snip-edges=true \
  --add-normalized-log-pitch=true --add-delta-pitch=true \
  scp:/tmp/g.scp "ark:golden/$g/pitch.ark"
"$KALDI_BIN/paste-feats" "ark:golden/$g/mfcc.ark" "ark:golden/$g/pitch.ark" "ark:golden/$g/base.ark"
"$KALDI_BIN/compute-cmvn-stats" "ark:golden/$g/base.ark" "ark:golden/$g/cmvn_stats.ark"
printf "utt utt\n" > /tmp/g.utt2spk
"$KALDI_BIN/apply-cmvn" --utt2spk=ark:/tmp/g.utt2spk "ark:golden/$g/cmvn_stats.ark" \
    "ark:golden/$g/base.ark" "ark:golden/$g/cmvn.ark"
"$KALDI_BIN/splice-feats" --left-context=3 --right-context=3 "ark:golden/$g/cmvn.ark" "ark:golden/$g/spliced.ark"
"$KALDI_BIN/transform-feats" "$PWD/lda_test_112.mat" "ark:golden/$g/spliced.ark" "ark:golden/$g/lda.ark"

# Group 4: edge cases (snip_edges=true, use_energy=true)
for w in silence_0.5s dc_0.5s sine_0.5s short_2f; do
  gen_group "edge_cases" "$w.wav" \
    --use-energy=true --snip-edges=true --num-ceps=13 --num-mel-bins=23 \
    --cepstral-lifter=22 --energy-floor=1.0 --low-freq=20 --high-freq=7800 \
    --sample-frequency=16000 --preemphasis-coefficient=0.97
  # keep edge-case outputs per signal name
  for f in mfcc cmvn_stats cmvn spliced deltas lda; do
    mv "golden/edge_cases/$f.ark" "golden/edge_cases/${w}.$f.ark" 2>/dev/null || true
  done
done

echo "golden generation done."

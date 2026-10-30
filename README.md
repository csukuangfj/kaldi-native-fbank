# Introduction

Kaldi-compatible online fbank feature extractor without external dependencies.

Tested on the following architectures and operating systems:

  - Linux
  - macOS
  - Windows
  - Android
  - x86
  - arm
  - aarch64

# Usage

See the following CMake-based speech recognition (i.e., text-to-speech) projects
for its usage:

- <https://github.com/k2-fsa/sherpa-ncnn>
  - Specifically, please have a look at <https://github.com/k2-fsa/sherpa-ncnn/blob/master/sherpa-ncnn/csrc/features.h>
- <https://github.com/k2-fsa/sherpa-onnx>

They use `kaldi-native-fbank` to compute fbank features for **real-time**
speech recognition.

# Python APIs

First, please install `kaldi-native-fbank` by

```bash
git clone https://github.com/csukuangfj/kaldi-native-fbank
cd kaldi-native-fbank
python3 setup.py install
```

or use

```bash
pip install kaldi-native-fbank
```

To check that you have installed `kaldi-native-fbank` successfully, please use

```
python3 -c "import kaldi_native_fbank; print(kaldi_native_fbank.__version__)"
```

which should print the version you have installed.

Please refer to

  - <https://github.com/csukuangfj/kaldi-native-fbank/blob/master/kaldi-native-fbank/python/tests/test_online_fbank.py>
  - <https://github.com/csukuangfj/kaldi-native-fbank/blob/master/kaldi-native-fbank/python/tests/test_online_mfcc.py>

for usages.

For easier reference, we post the above file below:

```python3
#!/usr/bin/env python3

import sys

try:
    import kaldifeat
except:
    print("Please install kaldifeat first")
    sys.exit(0)

import kaldi_native_fbank as knf
import torch


def main():
    sampling_rate = 16000
    samples = torch.randn(16000 * 10)

    opts = kaldifeat.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    online_fbank = kaldifeat.OnlineFbank(opts)

    online_fbank.accept_waveform(sampling_rate, samples)

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.frame_opts.snip_edges = False
    opts.mel_opts.debug_mel = False

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sampling_rate, samples.tolist())

    assert online_fbank.num_frames_ready == fbank.num_frames_ready
    for i in range(fbank.num_frames_ready):
        f1 = online_fbank.get_frame(i)
        f2 = torch.from_numpy(fbank.get_frame(i))
        assert torch.allclose(f1, f2, atol=1e-3), (i, (f1 - f2).abs().max())


if __name__ == "__main__":
    torch.manual_seed(20220825)
    main()
    print("success")
```

## NeMo buffered filter banks

`NemoFbankOptions` and `OnlineNemoFbank` support models that re-featurize a
moving audio buffer at every encoder step. They apply whole-buffer preemphasis,
a centered zero-padded STFT with a symmetric Hann window, Slaney mel filters,
and `log(mel + 2^-24)`. Samples use the range `[-1, 1]`; resampling and feature
normalization are left to the caller.

```python
import kaldi_native_fbank as knf

opts = knf.NemoFbankOptions()
opts.num_mel_bins = 128
opts.buffered_capacity = 672
fbank = knf.OnlineNemoFbank(opts)
fbank.accept_waveform(16000, samples)
fbank.input_finished()
frames, start_sample_offset = fbank.get_frames(0, fbank.num_frames_ready, True)
```

`get_frames()` returns owned NumPy data and discards raw samples before the
requested window. Use absolute frame indices and retain enough left context in
each request. A final request includes the partial hop. If its length equals
`buffered_capacity`, the window shifts by that remainder to keep its sample
length capped; `start_sample_offset` reports this shift. Shorter final windows
keep their start and include the extra samples. `num_frames_ready` counts full
hops only, and `num_trailing_samples` reports the remainder.

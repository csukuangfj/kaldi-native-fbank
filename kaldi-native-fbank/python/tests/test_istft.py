#!/usr/bin/env python3
#
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)


import torch

import kaldi_native_fbank as knf


def test_istft():
    n_fft = 16
    hop_length = 4
    center = False
    normalized = False
    window = None
    samples = torch.tensor(
        [1, 0, -2, 5, 9, -3, 2.5, 4, -1, 0.5, 3.5, -5, 6.5, 7, -2.75, 8, 12, -13]
    )

    print(samples.shape)

    torch_result = torch.stft(
        samples,
        n_fft=n_fft,
        hop_length=hop_length,
        center=center,
        return_complex=True,
        normalized=normalized,
        window=window,
    )
    # torch_result: (n_fft/2+1, num_frames), complex tensor

    print(torch_result)
    print(torch_result.shape)
    x = torch.istft(
        torch_result,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
        normalized=normalized,
    )
    print(x)
    print(x.shape)

    config = knf.StftConfig(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window_type="",
        center=center,
        pad_mode="reflect",
        normalized=normalized,
    )
    stft = knf.Stft(config)
    k = stft(samples.tolist())
    print(k.real)
    print(k.imag)
    istft = knf.IStft(config)
    print(istft(k))


def main():
    torch.manual_seed(20250308)
    test_istft()


if __name__ == "__main__":
    main()

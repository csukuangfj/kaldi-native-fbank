#!/usr/bin/env python3
#
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)


import torch

import kaldi_native_fbank as knf


def test_stft_config():
    config = knf.StftConfig(
        n_fft=512,
        hop_length=128,
        win_length=512,
        window_type="povey",
        center=True,
        pad_mode="reflect",
    )
    print(config)


def test_stft():
    samples = torch.rand(10000)
    for n_fft in [128, 512, 1024, 2048]:
        hop_length = n_fft // 4
        config = knf.StftConfig(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window_type="povey",
            center=True,
            pad_mode="reflect",
        )
        torch_result = torch.stft(
            samples,
            n_fft=n_fft,
            hop_length=hop_length,
            center=False,
            return_complex=False,
        )
        # y.shape: (n_fft/2+1, num_frames, 2)

        stft = knf.Stft(config)
        k = stft(samples.tolist())
        knf_result = torch.tensor([k.real, k.imag]).reshape(2, k.num_frames, -1)
        # now knf_result is (2, num_frames, n_fft/2+1)

        knf_result = knf_result.permute(2, 1, 0)
        assert torch.allclose(torch_result, knf_result, atol=1e-3), (
            torch_result,
            knf_result,
        )
        print("passed!")


def main():
    torch.manual_seed(20250308)
    test_stft_config()
    test_stft()


if __name__ == "__main__":
    main()

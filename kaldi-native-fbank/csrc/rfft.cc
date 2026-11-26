/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kaldi-native-fbank/csrc/rfft.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "kaldi-native-fbank/csrc/macros.h"
#include "kiss_fft.h"  // NOLINT(build/include_subdir)
#include "kiss_fftr.h"  // NOLINT(build/include_subdir)

namespace knf {

class Rfft::RfftImpl {
 public:
  RfftImpl(int32_t n, bool inverse) : n_(n), inverse_(inverse) {
    if ((n & 1) != 0) {
      KNF_LOGE("n should be even. Given: %d", n);
      throw std::invalid_argument("n should be even");
    }

    if (n < 0) {
      KNF_LOGE("n should be positive. Given: %d", n);
      throw std::invalid_argument("n should be positive");
    }

    // The kissfft plan (factorization and twiddle factors) depends only on n
    // and the direction, so build it once here instead of once per frame.
    cfg_.reset(kiss_fftr_alloc(n_, inverse_ ? 1 : 0, nullptr, nullptr));
    if (!cfg_) {
      KNF_LOGE("kiss_fftr_alloc failed for n = %d", n);
      throw std::runtime_error("kiss_fftr_alloc failed");
    }
    buf_.resize(n_ / 2 + 1);
  }

  void Compute(float *in_out) {
    if (!inverse_) {
      Forward(in_out);
    } else {
      Reverse(in_out);
    }
  }

  void Compute(double *in_out) {
    std::vector<float> f(in_out, in_out + n_);

    Compute(f.data());

    std::copy(f.begin(), f.end(), in_out);
  }

 private:
  void Forward(float *in_out) {
    std::vector<kiss_fft_cpx> &out = buf_;

    kiss_fftr(cfg_.get(), in_out, out.data());

    in_out[0] = out[0].r;
    in_out[1] = out[n_ / 2].r;

    for (int32_t i = 1; i < n_ / 2; ++i) {
      in_out[2 * i] = out[i].r;
      in_out[2 * i + 1] = out[i].i;
    }
  }

  void Reverse(float *in_out) {
    std::vector<kiss_fft_cpx> &out = buf_;
    out[0].r = in_out[0];
    out[0].i = 0;

    out[n_ / 2].r = in_out[1];
    out[n_ / 2].i = 0;

    for (int32_t i = 1; i < n_ / 2; ++i) {
      out[i].r = in_out[2 * i];
      out[i].i = in_out[2 * i + 1];
    }

    kiss_fftri(cfg_.get(), out.data(), in_out);
  }

 private:
  /// Deleter for the kissfft plan: kiss_fftr_free() is the deallocator
  /// matching kiss_fftr_alloc().
  struct CfgDeleter {
    void operator()(kiss_fftr_cfg p) const { kiss_fftr_free(p); }  // NOLINT
  };

  int32_t n_;             ///< Number of fft points.
  bool inverse_ = false;  ///< True for the inverse transform.

  /// kissfft plan for n_ points in the direction given by inverse_. It also
  /// holds kissfft's internal scratch space, so it is written by every
  /// Compute() call.
  std::unique_ptr<std::remove_pointer_t<kiss_fftr_cfg>, CfgDeleter> cfg_;

  /// Scratch buffer of n_ / 2 + 1 complex bins: the output of the forward
  /// transform, the input of the inverse transform.
  std::vector<kiss_fft_cpx> buf_;
};

Rfft::Rfft(int32_t n, bool inverse /*=false*/)
    : impl_(std::make_unique<RfftImpl>(n, inverse)) {}

Rfft::~Rfft() = default;

void Rfft::Compute(float *in_out) { impl_->Compute(in_out); }
void Rfft::Compute(double *in_out) { impl_->Compute(in_out); }

}  // namespace knf

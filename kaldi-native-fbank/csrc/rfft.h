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

/**
 * @file rfft.h
 * @brief n-point real discrete Fourier transform.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_RFFT_H_
#define KALDI_NATIVE_FBANK_CSRC_RFFT_H_

#include <cstdint>
#include <memory>

namespace knf {

/// @brief n-point real discrete Fourier transform, where n is even, n >= 2.
///
/// Forward transform:
/// \f[ R[k] = \sum_{j=0}^{n-1} \mathrm{in}[j] \cos(2\pi j k / n),
///     \qquad 0 \le k \le n/2 \f]
/// \f[ I[k] = \sum_{j=0}^{n-1} \mathrm{in}[j] \sin(2\pi j k / n),
///     \qquad 0 < k < n/2 \f]
///
/// @par Used by:
/// FbankComputer, MfccComputer, Stft and IStft.
///
/// @see ComputePowerSpectrum, Stft
class Rfft {
 public:
  /// Construct an Rfft object.
  ///
  /// @param n  Number of fft bins. It should be even.
  /// @param inverse  If true, Compute() performs the inverse transform.
  explicit Rfft(int32_t n, bool inverse = false);

  /// Destructor.
  ~Rfft();

  /** Transform in place.

   @param in_out A 1-D array of size n.
   *             On return of the forward transform:
   *               in_out[0] = R[0]
   *               in_out[1] = R[n/2]
   *               for 1 < k < n/2,
   *                 in_out[2*k] = R[k]
   *                 in_out[2*k+1] = I[k]
   *
   */
  void Compute(float *in_out);

  /// Same as above, but for double precision.
  void Compute(double *in_out);

 private:
  class RfftImpl;

  std::unique_ptr<RfftImpl> impl_;  ///< Implementation details.
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_RFFT_H_

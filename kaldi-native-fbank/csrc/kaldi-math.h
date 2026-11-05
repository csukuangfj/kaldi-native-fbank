// kaldi-native-fbank/csrc/kaldi-math.h
//
// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Yanmin Qian;
//                      Jan Silovsky;  Saarland University
// Copyright (c)  2024  Brno University of Technology (authors: Karel Vesely)
//
// This file is an excerpt from kaldi/src/base/kaldi-math.h

/**
 * @file kaldi-math.h
 * @brief Math constants and small math helpers.
 *
 * This file is an excerpt from @kfile{src/base/kaldi-math.h}.
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_KALDI_MATH_H_
#define KALDI_NATIVE_FBANK_CSRC_KALDI_MATH_H_

#include <cmath>  // logf, sqrtf, cosf
#include <cstdint>
#include <cstdlib>  // RAND_MAX

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.4142135623730950488016887
#endif

namespace knf {

/// @brief Natural logarithm of x (float overload; matches kaldi's Log).
inline float Log(float x) { return logf(x); }

/// @brief Natural logarithm of x (double overload; matches kaldi's Log).
inline double Log(double x) { return log(x); }

/// @brief Exponential of x (float overload; matches kaldi's Exp).
inline float Exp(float x) { return expf(x); }

/// @brief Exponential of x (double overload; matches kaldi's Exp).
inline double Exp(double x) { return exp(x); }

/// @brief Greatest common divisor of two integers.
///
/// Euclid's algorithm; both arguments must be >= 0 and not both zero.
/// Mirrors kaldi's `Gcd` in @kfile{src/base/kaldi-math.h}.
///
/// @param [in] m  First integer.
/// @param [in] n  Second integer.
/// @return  The greatest common divisor of `m` and `n`.
template <typename I>
inline I Gcd(I m, I n) {
  while (n != 0) {
    I tmp = m % n;
    m = n;
    n = tmp;
  }
  return m;
}

/// @brief Least common multiple of two integers.
///
/// Mirrors kaldi's `Lcm` in @kfile{src/base/kaldi-math.h}.
///
/// @param [in] m  First integer.
/// @param [in] n  Second integer.
/// @return  The least common multiple of `m` and `n` (0 if either is 0).
template <typename I>
inline I Lcm(I m, I n) {
  I tmp = Gcd(m, n);
  return tmp == 0 ? 0 : (m / tmp * n);
}

/// @brief Approximate floating-point equality test.
///
/// Returns true if `a` and `b` are equal to a relative tolerance `tol`:
/// \f[ |a - b| \le \mathrm{tol} \cdot (|a| + |b|) \f]
/// Mirrors kaldi's `ApproxEqual` in @kfile{src/base/kaldi-math.h}.
///
/// @param [in] a   First value.
/// @param [in] b   Second value.
/// @param [in] tol Relative tolerance (default 0.01).
/// @return  True if `a` and `b` are approximately equal.
template <typename Real>
inline bool ApproxEqual(Real a, Real b, Real tol = 0.01) {
  return (std::abs(a - b) <= tol * (std::abs(a) + std::abs(b)));
}

/// @brief Return a random integer between 0 and RAND_MAX, inclusive.
int Rand(struct RandomState *state = nullptr);

/// @brief State for the thread-safe random number generator.
struct RandomState {
  /// Construct; initializes seed from the current time.
  RandomState();

  unsigned seed;  ///< Seed of the random number generator.
};

/// Returns a random number strictly between 0 and 1.
///
/// @see Rand, RandGauss, RandomState
inline float RandUniform(struct RandomState *state = nullptr) {
  return static_cast<float>((Rand(state) + 1.0) / (RAND_MAX + 2.0));
}

/// @brief Return a random sample from the standard normal distribution.
///
/// Uses the Box-Muller transform:
/// \f$ \sqrt{-2 \ln U_1} \cos(2 \pi U_2) \f$
/// with \f$ U_1, U_2 \f$ drawn uniformly from (0, 1).
///
/// @see RandUniform, RandomState
inline float RandGauss(struct RandomState *state = nullptr) {
  return static_cast<float>(sqrtf(-2 * Log(RandUniform(state))) *
                            cosf(2 * M_PI * RandUniform(state)));
}

/// @brief Compute the square root of each element of an array in place.
///
/// @param in_out  Pointer to an array of size n.
/// @param n       Number of elements.
void Sqrt(float *in_out, int32_t n);

}  // namespace knf
#endif  // KALDI_NATIVE_FBANK_CSRC_KALDI_MATH_H_

// kaldi-native-fbank/csrc/kaldi-math.h
//
// Copyright (c)  2024  Brno University of Technology (authors: Karel Vesely)

// This file is an excerpt from kaldi/src/feat/kaldi-math.h

#pragma once

#include <cmath>  // logf, sqrtf, cosf
#include <cstdlib>  // RAND_MAX

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

namespace knf {

inline float Log(float x) { return logf(x); }

// Returns a random integer between 0 and RAND_MAX, inclusive
int Rand(struct RandomState* state = NULL);

// State for thread-safe random number generator
struct RandomState {
  RandomState();
  unsigned seed;
};

/// Returns a random number strictly between 0 and 1.
inline float RandUniform(struct RandomState* state = NULL) {
  return static_cast<float>((Rand(state) + 1.0) / (RAND_MAX+2.0));
}

inline float RandGauss(struct RandomState* state = NULL) {
  return static_cast<float>(sqrtf (-2 * Log(RandUniform(state)))
                            * cosf(2*M_PI*RandUniform(state)));
}

}  // namespace knf

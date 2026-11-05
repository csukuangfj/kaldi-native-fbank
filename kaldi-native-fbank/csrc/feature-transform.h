/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-transform.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/featbin/transform-feats.cc
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

/**
 * @file feature-transform.h
 * @brief Linear/affine feature transforms (e.g. LDA).
 *
 * This is kaldi's `transform-feats` (see
 * @kfile{src/featbin/transform-feats.cc}): it applies a stored matrix
 * \f$ M \f$ of shape \f$ (D_{out} \times n) \f$ to feature rows
 * \f$ x \f$ of dimension \f$ n \f$ (linear form) or \f$ n = D_{in}+1 \f$
 * (affine form, where the last column of \f$ M \f$ is an offset):
 *
 * \f[ y = x\, M^{T} \qquad\text{(linear)} \f]
 * \f[ y = x\, M_{[:,\,0:n-1]}^{T} + M_{[:,\,n-1]} \qquad\text{(affine)} \f]
 *
 * @code
 *   knf::FeatureTransform lda = knf::FeatureTransform::Read("lda.mat");
 *   knf::MatrixF out = lda.Apply(feats);   // (T, lda.OutputDim())
 * @endcode
 */

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_TRANSFORM_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_TRANSFORM_H_

#include <stdint.h>

#include <string>

#include "Eigen/Core"
#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

/**
   @brief A linear (or affine) feature transform, e.g. an LDA matrix.

   The transform is a matrix \f$ M \f$ of shape
   \f$ (D_{out} \times n) \f$; see the file description for the exact
   semantics, which follow kaldi's `transform-feats`
   (@kfile{src/featbin/transform-feats.cc}):

   * if \f$ n = D_{in} \f$ (the input feature dimension), the transform is
     linear: \f$ y = x\, M^{T} \f$;
   * if \f$ n = D_{in} + 1 \f$, the transform is affine: the input features
     get an implicit 1.0 appended, so
     \f$ y = x\, M_{[:,\,0:n-1]}^{T} + M_{[:,\,n-1]} \f$;
   * otherwise Apply() throws.

   @code
     knf::FeatureTransform lda = knf::FeatureTransform::Read("lda.mat");
     knf::MatrixF out = lda.Apply(feats);
   @endcode

   @par Used by:
   Pipelines that reduce/rotate features (mfcc -> cmvn -> splice -> lda).

   @see Read, Apply
*/
class FeatureTransform {
 public:
  /// @brief Read a transform matrix from a kaldi matrix file.
  ///
  /// Supports the kaldi binary format ("\0B" + "FM"/"DM" + dims + data,
  /// as written by kaldi's `Matrix::Write`) and the kaldi text format
  /// (`[ ... ]`).
  ///
  /// @param [in] filename  Path of the matrix file, e.g. `lda.mat`.
  /// @return  The transform.
  static FeatureTransform Read(const std::string &filename);

  /// @brief Construct a transform from a matrix directly.
  ///
  /// @param [in] mat  The transform matrix of shape
  ///                  (OutputDim() x InputDim()), or one column wider for
  ///                  an affine transform (the last column is the offset).
  explicit FeatureTransform(const MatrixF &mat);

  /// @brief Number of columns of the stored matrix.
  ///
  /// This is the feature dimension in the linear form; an affine transform
  /// consumes InputDim()-1 features plus the implicit 1.0.
  int32_t InputDim() const;

  /// @brief Number of rows of the stored matrix (the output dimension).
  int32_t OutputDim() const;

  /// @brief True if the transform is affine for inputs of dimension
  /// `input_dim`.
  ///
  /// The matrix alone cannot record the intended input dimension, so the
  /// rule is the one kaldi's `transform-feats` applies: the transform is
  /// affine iff it has one column more than the feature dimension
  /// (\f$ n = D_{in} + 1 \f$), the extra column being the offset.
  ///
  /// @param [in] input_dim  The input feature dimension.
  bool IsAffine(int32_t input_dim) const;

  /**
     @brief Apply the transform to a feature matrix.

     kaldi `transform-feats` semantics (rows = frames):

     * if the transform has `feats.cols()` columns, the output is
       \f$ Y = X M^{T} \f$ (linear);
     * if it has `feats.cols() + 1` columns, the transform is affine and
       the last column is the offset:
       \f$ Y = X M_{[:,\,0:n-1]}^{T} + \mathbf{1} b^{T} \f$;
     * otherwise an exception is thrown.

     @param [in] feats  Feature matrix of size T x D, float32.
     @return  The transformed matrix of size T x OutputDim().
  */
  MatrixF Apply(const MatrixF &feats) const;

  /**
     @brief Apply the transform to a single feature frame.

     Same semantics as the matrix version of Apply().

     @param [in] feat  Pointer to one frame of dimension `dim`.
     @param [in] dim   Dimension of the frame.
     @param [out] out  Pointer to an array of OutputDim() floats; the caller
                       must pre-allocate it.
  */
  void Apply(const float *feat, int32_t dim, float *out) const;

 private:
  MatrixF mat_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_TRANSFORM_H_

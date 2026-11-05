/// kaldi-native-fbank/kaldi-native-fbank/csrc/feature-transform.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/featbin/transform-feats.cc
//
/// This file is part of kaldi-native-fbank.
//
// See ../../LICENSE for more information

#include "kaldi-native-fbank/csrc/feature-transform.h"

#include <cstddef>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace knf {

namespace {

// kaldi binary scalars are written as: 1 byte length (signed: +sizeof,
// unsigned: -sizeof) followed by the raw value (kaldi WriteBasicType).
template <typename T>
T ReadBinaryScalar(std::istream *is) {
  char len_c = 0;
  is->read(&len_c, 1);
  int32_t len = static_cast<unsigned char>(len_c);
  if (len_c < 0) {
    len = -len;
  }
  if (len != static_cast<int32_t>(sizeof(T))) {
    throw std::runtime_error("FeatureTransform: unexpected scalar size");
  }
  T value = 0;
  is->read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!*is) {
    throw std::runtime_error("FeatureTransform: unexpected end of file");
  }
  return value;
}

// Reads a kaldi matrix in binary ("\0B" + "FM"/"DM" + dims + data) or text
// ("[ ... ]", one frame per line) format, as written by kaldi Matrix::Write.
MatrixD ReadKaldiMatrix(const std::string &filename) {
  std::ifstream is(filename, std::ios::binary);
  if (!is) {
    throw std::runtime_error("FeatureTransform: cannot open " + filename);
  }

  // Peek the first bytes to detect the format.
  char first = 0;
  is.read(&first, 1);
  bool binary = false;
  if (first == '\0') {
    char marker = 0;
    is.read(&marker, 1);
    if (marker != 'B') {
      throw std::runtime_error("FeatureTransform: " + filename +
                               " is not a kaldi binary matrix");
    }
    binary = true;
  }

  if (binary) {
    char token[3] = {0, 0, 0};
    is.read(token, 2);
    is.get();  // trailing space written by kaldi's WriteToken
    if (std::string(token) != "FM" && std::string(token) != "DM") {
      throw std::runtime_error("FeatureTransform: " + filename +
                               " is not a kaldi matrix (bad token)");
    }
    int32_t rows = ReadBinaryScalar<int32_t>(&is);
    int32_t cols = ReadBinaryScalar<int32_t>(&is);
    if (rows < 0 || cols < 0) {
      throw std::runtime_error("FeatureTransform: bad matrix shape");
    }
    MatrixD ans(rows, cols);
    if (std::string(token) == "FM") {
      std::vector<float> tmp(static_cast<size_t>(rows) * cols);
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(tmp.size() * sizeof(float)));
      for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
          ans(r, c) = tmp[static_cast<size_t>(r) * cols + c];
        }
      }
    } else {
      std::vector<double> tmp(static_cast<size_t>(rows) * cols);
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(tmp.size() * sizeof(double)));
      for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
          ans(r, c) = tmp[static_cast<size_t>(r) * cols + c];
        }
      }
    }
    if (!is) {
      throw std::runtime_error("FeatureTransform: truncated matrix in " +
                               filename);
    }
    return ans;
  }

  // Text format: " [ <row> <row> ... ]", one matrix row per line.
  is.seekg(0);
  std::string content((std::istreambuf_iterator<char>(is)),
                      std::istreambuf_iterator<char>());
  size_t open = content.find('[');
  size_t close = content.rfind(']');
  if (open == std::string::npos || close == std::string::npos ||
      close <= open) {
    throw std::runtime_error("FeatureTransform: " + filename +
                             " is not a kaldi matrix");
  }
  std::string body = content.substr(open + 1, close - open - 1);
  std::vector<std::vector<double>> rows_data;
  std::istringstream row_stream(body);
  std::string line;
  while (std::getline(row_stream, line)) {
    std::istringstream ls(line);
    std::vector<double> row;
    double v = 0.0;
    while (ls >> v) {
      row.push_back(v);
    }
    if (!row.empty()) {
      rows_data.push_back(row);
    }
  }
  if (rows_data.empty()) {
    return MatrixD(0, 0);
  }
  int32_t rows = static_cast<int32_t>(rows_data.size());
  int32_t cols = static_cast<int32_t>(rows_data[0].size());
  MatrixD ans(rows, cols);
  for (int32_t r = 0; r < rows; ++r) {
    if (static_cast<int32_t>(rows_data[r].size()) != cols) {
      throw std::runtime_error("FeatureTransform: ragged matrix in " +
                               filename);
    }
    for (int32_t c = 0; c < cols; ++c) {
      ans(r, c) = rows_data[r][c];
    }
  }
  return ans;
}

}  // namespace

FeatureTransform::FeatureTransform(const MatrixF &mat) : mat_(mat) {}

FeatureTransform FeatureTransform::Read(const std::string &filename) {
  return FeatureTransform(ReadKaldiMatrix(filename).cast<float>());
}

int32_t FeatureTransform::InputDim() const { return mat_.cols(); }

int32_t FeatureTransform::OutputDim() const { return mat_.rows(); }

bool FeatureTransform::IsAffine(int32_t input_dim) const {
  return mat_.cols() == input_dim + 1;
}

MatrixF FeatureTransform::Apply(const MatrixF &feats) const {
  int32_t dim = feats.cols();
  int32_t cols = mat_.cols();
  if (cols == dim) {
    // linear: Y = X M^T
    return feats * mat_.transpose();
  }
  if (cols == dim + 1) {
    // affine: the last column of the transform is the offset (kaldi
    // transform-feats appends an implicit 1.0 to the input features).
    // One Eigen expression, evaluated once into the return value:
    //   Y = X * M[:, 0:dim]^T + 1 * b^T
    return (feats * mat_.leftCols(dim).transpose()).rowwise() +
           mat_.col(dim).transpose();
  }
  std::ostringstream os;
  os << "FeatureTransform: transform has " << mat_.rows() << "x" << cols
     << " dims, which does not match the feature dim " << dim << " (expected "
     << cols << " or " << cols - 1 << " columns)";
  throw std::invalid_argument(os.str());
}

void FeatureTransform::Apply(const float *feat, int32_t dim, float *out) const {
  Eigen::Map<const VectorF> x(feat, dim);
  Eigen::Map<VectorF> y(out, mat_.rows());
  int32_t cols = mat_.cols();
  if (cols == dim) {
    y = mat_ * x;
    return;
  }
  if (cols == dim + 1) {
    y = mat_.leftCols(dim) * x + mat_.col(dim);
    return;
  }
  std::ostringstream os;
  os << "FeatureTransform: transform has " << mat_.rows() << "x" << cols
     << " dims, which does not match the feature dim " << dim << " (expected "
     << cols << " or " << cols - 1 << " columns)";
  throw std::invalid_argument(os.str());
}

}  // namespace knf

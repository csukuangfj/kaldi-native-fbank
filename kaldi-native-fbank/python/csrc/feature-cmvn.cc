/**
 * Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/python/csrc/feature-cmvn.h"

#include <pybind11/eigen.h>

#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-cmvn.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"

namespace knf {

void PybindFeatureCmvn(py::module &m) {  // NOLINT
  auto cls_CmvnOptions = py::class_<CmvnOptions>(m, "CmvnOptions");
  cls_CmvnOptions.attr("__module__") = "kaldi_native_fbank";
  cls_CmvnOptions
      .def(py::init([](bool norm_vars, bool reverse) {
             CmvnOptions opts;
             opts.norm_vars = norm_vars;
             opts.reverse = reverse;
             return opts;
           }),
           py::arg("norm_vars") = false, py::arg("reverse") = false)
      .def_readwrite("norm_vars", &CmvnOptions::norm_vars)
      .def_readwrite("reverse", &CmvnOptions::reverse)
      .def("__str__", [](const CmvnOptions &opts) -> std::string {
        std::ostringstream os;
        os << "CmvnOptions(norm_vars=" << (opts.norm_vars ? "True" : "False")
           << ", reverse=" << (opts.reverse ? "True" : "False") << ")";
        return os.str();
      });

  m.def("compute_cmvn_stats", &ComputeCmvnStats, py::arg("feats"),
        py::call_guard<py::gil_scoped_release>(),
        "Computes cmvn stats for a feature matrix (frames x dim). "
        "Returns a 2 x (dim+1) double matrix: row0=[sums, count], "
        "row1=[sumsq, (unused)] (kaldi layout).");

  m.def(
      "apply_cmvn",
      [](const MatrixD &stats, const CmvnOptions &opts, const MatrixF &feats)
          -> MatrixF { return ApplyCmvn(stats, opts, feats); },
      py::arg("stats"), py::arg("opts"), py::arg("feats"),
      py::call_guard<py::gil_scoped_release>(),
      "Applies cmvn (kaldi ApplyCmvn semantics) and returns the "
      "normalized copy.");

  m.def(
      "apply_cmvn_",
      [](const MatrixD &stats, const CmvnOptions &opts,
         Eigen::Ref<MatrixF> feats) { ApplyCmvn(stats, opts, feats); },
      py::arg("stats"), py::arg("opts"), py::arg("feats"),
      py::call_guard<py::gil_scoped_release>(),
      "In-place version of apply_cmvn.");

  auto cls_CmvnAccumulator = py::class_<CmvnAccumulator>(m, "CmvnAccumulator");

  cls_CmvnAccumulator.attr("__module__") = "kaldi_native_fbank";

  cls_CmvnAccumulator.def(py::init<int32_t>(), py::arg("dim") = 0)
      .def("accumulate", &CmvnAccumulator::Accumulate, py::arg("feats"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "add",
          [](CmvnAccumulator &self, int32_t dim, std::vector<float> frame,
             float weight) { self.Add(dim, frame.data(), weight); },
          py::arg("dim"), py::arg("frame"), py::arg("weight") = 1.0f)
      .def_property_readonly(
          "stats",
          [](const CmvnAccumulator &self) -> MatrixD { return self.Stats(); });
}

}  // namespace knf

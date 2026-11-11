// Copyright 2026 Universidad Politecnica de Madrid (UPM).
//
// Author: Guanliang Li
// Contributor: Pedro Espinosa Angulo, Santiago Tapia Fernandez (supervised)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// PyTorch (.pt) models through ultralytics in an embedded Python interpreter: the same
// inference as the Python sslam_tools node, CPU or CUDA. Needs python3 with ultralytics
// (and a CUDA build of torch for cuda devices) at run time.

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "sslam_tools_cpp/detectors/yolo_backend.hpp"

namespace py = pybind11;

namespace sslam
{

namespace
{

// The interpreter lives until the process exits: finalizing it with torch loaded is unsafe.
void ensure_interpreter()
{
  if (!Py_IsInitialized()) {
    py::initialize_interpreter();
  }
}

class UltralyticsBackend : public YoloBackend
{
public:
  UltralyticsBackend(const std::string & model_file, const std::string & device, double conf)
  : model_file_(model_file), device_(device), conf_(conf)
  {
    ensure_interpreter();
    try {
      const py::module_ ultralytics = py::module_::import("ultralytics");
      model_ = ultralytics.attr("YOLO")(model_file, py::arg("task") = "obb");
      for (const auto & kv : model_.attr("names").cast<py::dict>()) {
        names_[kv.first.cast<int>()] = kv.second.cast<std::string>();
      }
      if (device_.rfind("cuda", 0) == 0 &&
        !py::module_::import("torch").attr("cuda").attr("is_available")().cast<bool>())
      {
        throw std::runtime_error("device '" + device_ + "' requested but torch has no CUDA device");
      }
      torch_version_ = py::module_::import("torch").attr("__version__").cast<std::string>();
    } catch (const py::error_already_set & e) {
      throw std::runtime_error(std::string("ultralytics backend: ") + e.what());
    }
  }

  ~UltralyticsBackend() override
  {
    model_ = py::object();  // release while the interpreter is alive
  }

  std::vector<OrientedBox> detect(const cv::Mat & rgb) override
  {
    std::vector<OrientedBox> boxes;
    try {
      const cv::Mat img = rgb.isContinuous() ? rgb : rgb.clone();
      const py::array_t<uint8_t> arr({img.rows, img.cols, 3}, img.data);  // copies
      const py::object results = model_(arr, py::arg("device") = device_, py::arg("conf") = conf_,
          py::arg("verbose") = false);
      const py::object obb = results.attr("__getitem__")(0).attr("obb");
      const auto xy = obb.attr("xyxyxyxy").attr("cpu")().attr("numpy")().attr("astype")("float64")
        .cast<py::array_t<double, py::array::c_style>>();
      const auto conf = obb.attr("conf").attr("cpu")().attr("numpy")().attr("astype")("float32")
        .cast<py::array_t<float, py::array::c_style>>();
      const auto cls = obb.attr("cls").attr("cpu")().attr("numpy")().attr("astype")("int32")
        .cast<py::array_t<int32_t, py::array::c_style>>();
      const auto v = xy.unchecked<3>();
      for (py::ssize_t i = 0; i < v.shape(0); ++i) {
        OrientedBox b;
        for (int k = 0; k < 4; ++k) {b.vertices[k] = Vec2(v(i, k, 0), v(i, k, 1));}
        b.conf = conf.at(i);
        b.cls = cls.at(i);
        boxes.push_back(b);
      }
    } catch (const py::error_already_set & e) {
      throw std::runtime_error(std::string("ultralytics inference failed: ") + e.what());
    }
    return boxes;
  }

  const std::map<int, std::string> & names() const override {return names_;}

  std::string description() const override
  {
    return "ultralytics (PyTorch " + torch_version_ + ") on " + device_ + " (" + model_file_ + ")";
  }

private:
  std::string model_file_, device_, torch_version_;
  double conf_;
  py::object model_;
  std::map<int, std::string> names_;
};

}  // namespace

std::unique_ptr<YoloBackend> make_ultralytics_backend(
  const std::string & model_file, const std::string & device, double conf)
{
  return std::make_unique<UltralyticsBackend>(model_file, device, conf);
}

}  // namespace sslam

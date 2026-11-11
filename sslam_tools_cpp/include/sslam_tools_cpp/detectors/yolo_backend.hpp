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

#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "sslam_tools_cpp/geometry.hpp"

namespace sslam
{

// One oriented box in image pixels
struct OrientedBox
{
  std::array<Vec2, 4> vertices;  // ultralytics xyxyxyxy order
  float conf;
  int cls;
};

// Runs a YOLOv8-OBB model on an RGB image and returns the boxes above the confidence threshold.
class YoloBackend
{
public:
  virtual ~YoloBackend() = default;
  virtual std::vector<OrientedBox> detect(const cv::Mat & rgb) = 0;
  virtual const std::map<int, std::string> & names() const = 0;
  virtual std::string description() const = 0;
};

// Backend for models exported with NMS (yolo export ... nms=True): input [1, 3, S, S] float
// RGB in [0, 1], output [1, N, 7] rows of (cx, cy, w, h, conf, cls, angle). Does ultralytics'
// letterbox and OBB decoding; subclasses only run the tensor.
class TensorYoloBackend : public YoloBackend
{
public:
  TensorYoloBackend(int input_size, double conf_threshold, std::map<int, std::string> names);
  std::vector<OrientedBox> detect(const cv::Mat & rgb) override;
  const std::map<int, std::string> & names() const override {return names_;}

protected:
  // input: 3 * S * S floats (NCHW). Returns n_rows rows of 7 floats, valid until the next call.
  virtual const float * run(const float * input, size_t & n_rows) = 0;

  int size_;
  double conf_;
  std::map<int, std::string> names_;
  std::vector<float> input_;
};

struct YoloBackendOptions
{
  std::string model;  // model file, or an ultralytics export directory
  std::string device;  // "cpu", "gpu", "cuda", "cuda:N"
  double conf_threshold = 0.25;
  int imgsz = 0;  // input side for exported models; 0 = metadata.yaml, else fallback_imgsz
  int fallback_imgsz = 1024;
};

// Backend selection (the Python node's behaviour plus the C++ engines):
//   .pt model                 -> ultralytics (embedded Python, PyTorch), on the given device
//   device cpu                -> OpenVINO (.xml / OpenVINO export dir / .onnx)
//   device gpu, .onnx/.engine -> TensorRT (not implemented yet)
//   device gpu, OpenVINO only -> OpenVINO on the CPU (as ultralytics does), with a note
// note: set when the selection deviates from the request.
std::unique_ptr<YoloBackend> make_yolo_backend(const YoloBackendOptions & opts, std::string & note);

}  // namespace sslam

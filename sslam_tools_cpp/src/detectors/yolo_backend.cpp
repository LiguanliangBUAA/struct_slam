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

#include "sslam_tools_cpp/detectors/yolo_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace sslam
{

#ifdef SSLAM_WITH_OPENVINO
std::unique_ptr<YoloBackend> make_openvino_backend(
  const std::string & model_file, int input_size, double conf, std::map<int, std::string> names);
#endif
#ifdef SSLAM_WITH_PYTHON
std::unique_ptr<YoloBackend> make_ultralytics_backend(
  const std::string & model_file, const std::string & device, double conf);
#endif

// ---------------------------------------------------------------- TensorYoloBackend

TensorYoloBackend::TensorYoloBackend(int input_size, double conf_threshold, std::map<int, std::string> names)
: size_(input_size), conf_(conf_threshold), names_(std::move(names)),
  input_(3 * static_cast<size_t>(input_size) * input_size) {}

std::vector<OrientedBox> TensorYoloBackend::detect(const cv::Mat & rgb)
{
  const int S = size_;
  // ultralytics LetterBox (centered, pad 114)
  const int h = rgb.rows, w = rgb.cols;
  const double gain = std::min(static_cast<double>(S) / h, static_cast<double>(S) / w);
  const int unpad_w = static_cast<int>(std::round(w * gain));
  const int unpad_h = static_cast<int>(std::round(h * gain));
  const double dw = (S - unpad_w) / 2.0, dh = (S - unpad_h) / 2.0;
  cv::Mat resized = rgb;
  if (unpad_w != w || unpad_h != h) {
    cv::resize(rgb, resized, cv::Size(unpad_w, unpad_h), 0, 0, cv::INTER_LINEAR);
  }
  cv::Mat lb;
  cv::copyMakeBorder(resized, lb,
    static_cast<int>(std::round(dh - 0.1)), static_cast<int>(std::round(dh + 0.1)),
    static_cast<int>(std::round(dw - 0.1)), static_cast<int>(std::round(dw + 0.1)),
    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  // HWC uint8 -> NCHW float [0, 1]
  const size_t plane = static_cast<size_t>(S) * S;
  for (int y = 0; y < S; ++y) {
    const cv::Vec3b * row = lb.ptr<cv::Vec3b>(y);
    for (int x = 0; x < S; ++x) {
      const size_t i = static_cast<size_t>(y) * S + x;
      input_[i] = row[x][0] / 255.0f;
      input_[plane + i] = row[x][1] / 255.0f;
      input_[2 * plane + i] = row[x][2] / 255.0f;
    }
  }
  size_t n_rows = 0;
  const float * out = run(input_.data(), n_rows);

  // scale_boxes padding, as ultralytics
  const double pad_x = std::round((S - w * gain) / 2.0 - 0.1);
  const double pad_y = std::round((S - h * gain) / 2.0 - 0.1);
  std::vector<OrientedBox> boxes;
  for (size_t k = 0; k < n_rows; ++k) {
    const float * d = out + 7 * k;
    if (!(d[4] > conf_)) {continue;}
    double cx = d[0], cy = d[1], bw = d[2], bh = d[3], angle = d[6];
    // regularize_rboxes: w is the long side, angle in [0, pi)
    if (!(bw > bh)) {
      std::swap(bw, bh);
      angle += kPi / 2;
    }
    angle = pymod(angle, kPi);
    cx = (cx - pad_x) / gain;
    cy = (cy - pad_y) / gain;
    bw /= gain;
    bh /= gain;
    // xywhr2xyxyxyxy
    const double c = std::cos(angle), s = std::sin(angle);
    const Vec2 ctr(cx, cy), v1(bw / 2 * c, bw / 2 * s), v2(-bh / 2 * s, bh / 2 * c);
    boxes.push_back({{ctr + v1 + v2, ctr + v1 - v2, ctr - v1 - v2, ctr - v1 + v2}, d[4], static_cast<int>(d[5])});
  }
  return boxes;
}

// ---------------------------------------------------------------- selection

namespace
{

enum class Device { kCpu, kGpu };

Device parse_device(const std::string & device)
{
  std::string d = device;
  std::transform(d.begin(), d.end(), d.begin(), [](unsigned char c) {return std::tolower(c);});
  if (d == "cpu") {return Device::kCpu;}
  if (d == "gpu" || d.rfind("cuda", 0) == 0) {return Device::kGpu;}
  throw std::invalid_argument("Unsupported YOLO device '" + device + "' (use 'cpu', 'gpu' or 'cuda:N')");
}

// First file in dir with one of the extensions (in order of preference)
fs::path find_in(const fs::path & dir, const std::vector<std::string> & exts)
{
  for (const auto & ext : exts) {
    for (const auto & entry : fs::directory_iterator(dir)) {
      if (entry.path().extension() == ext) {return entry.path();}
    }
  }
  return {};
}

}  // namespace

std::unique_ptr<YoloBackend> make_yolo_backend(const YoloBackendOptions & opts, std::string & note)
{
  note.clear();
  const fs::path given(opts.model);
  if (!fs::exists(given)) {
    throw std::runtime_error("YOLO model not found: " + given.string());
  }
  const Device device = parse_device(opts.device);

  // PyTorch weights: ultralytics in an embedded interpreter, exactly as the Python node
  if (!fs::is_directory(given) && given.extension() == ".pt") {
#ifdef SSLAM_WITH_PYTHON
    const std::string dev = opts.device == "gpu" ? "cuda:0" : opts.device;
    return make_ultralytics_backend(given.string(), dev, opts.conf_threshold);
#else
    throw std::runtime_error("PyTorch .pt models need the ultralytics backend, which was not built "
                             "(pybind11 not found); export the model to OpenVINO/ONNX instead");
#endif
  }

  // Exported models: input size and class names from the export's metadata
  const fs::path dir = fs::is_directory(given) ? given : given.parent_path();
  std::map<int, std::string> names = {{0, "walls"}, {1, "columns"}};
  int imgsz = opts.fallback_imgsz;
  const fs::path metadata = dir / "metadata.yaml";
  if (fs::exists(metadata)) {
    const YAML::Node meta = YAML::LoadFile(metadata.string());
    if (meta["task"] && meta["task"].as<std::string>() != "obb") {
      throw std::runtime_error("YOLO model task is '" + meta["task"].as<std::string>() + "', expected 'obb'");
    }
    if (meta["imgsz"] && meta["imgsz"].IsSequence()) {imgsz = meta["imgsz"][0].as<int>();}
    if (meta["names"]) {
      names.clear();
      for (const auto & kv : meta["names"]) {names[kv.first.as<int>()] = kv.second.as<std::string>();}
    }
  }
  if (opts.imgsz > 0) {imgsz = opts.imgsz;}

  auto pick = [&](const std::vector<std::string> & exts) {
      if (fs::is_directory(given)) {return find_in(given, exts);}
      return std::find(exts.begin(), exts.end(), given.extension().string()) != exts.end() ? given : fs::path();
    };

  if (device == Device::kGpu) {
    const fs::path trt_model = pick({".engine", ".onnx"});
    if (!trt_model.empty()) {
      // TODO(TensorRT): build/cache an .engine from the .onnx on the target (e.g. the Orin NX)
      // and return a TensorYoloBackend running it here.
      throw std::runtime_error("device '" + opts.device + "' with " + trt_model.filename().string() +
              " needs the TensorRT backend, which is not implemented yet");
    }
    if (pick({".xml"}).empty()) {
      throw std::runtime_error("No model for device '" + opts.device + "' in " + given.string());
    }
    note = "OpenVINO models run on the CPU: device '" + opts.device + "' ignored";
  }

  const fs::path ov_model = pick({".xml", ".onnx"});
  if (ov_model.empty()) {
    throw std::runtime_error("No OpenVINO (.xml) or ONNX model in " + given.string() +
            " (export with: yolo export model=<model>.pt format=openvino nms=True)");
  }
#ifdef SSLAM_WITH_OPENVINO
  return make_openvino_backend(ov_model.string(), imgsz, opts.conf_threshold, names);
#else
  throw std::runtime_error("OpenVINO was not found when building sslam_tools_cpp: cannot run " +
          ov_model.string() + " (pip install openvino, then rebuild)");
#endif
}

}  // namespace sslam

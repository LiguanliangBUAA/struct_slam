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

#include <memory>
#include <string>
#include <vector>

#include "sslam_tools_cpp/detectors/image_based_detector.hpp"
#include "sslam_tools_cpp/detectors/yolo_backend.hpp"

namespace sslam
{

struct YoloDetectorConfig : ImageBasedDetectorConfig
{
  // Relative paths resolve against the package share dir (resource/yolo_model);
  // an absolute path overrides that.
  std::string yolo_model_path = "";
  // .pt (ultralytics/PyTorch), OpenVINO export dir or .xml, .onnx, .engine
  std::string yolo_model_name = "yolon1024HD-clean-AdamW_openvino_model";
  // "cpu" or "cuda:0"/"gpu" (see make_yolo_backend for which engine runs)
  std::string device = "cpu";
  bool label = true;
  // ultralytics' default predict confidence threshold
  double conf_threshold = 0.25;
  // Input side for exported models; 0 = from the export's metadata.yaml
  int yolo_imgsz = 0;

  void load(ParamReader & p);
};

// YOLOv8-OBB walls detector on the BEV image.
class YoloDetector : public ImageBasedDetector
{
public:
  YoloDetector(std::string name, ParamReader & params, const std::string & share_dir);
  const DetectorConfig & config() const override {return cfg_;}
  std::string backend_description() const {return backend_->description();}
  // Set when the backend differs from the requested device
  const std::string & backend_note() const {return backend_note_;}
  const std::vector<OrientedBox> & boxes() const {return boxes_;}

protected:
  SSCollection inference(const cv::Mat & processed_image) override;
  void build_debug_image() override;

private:
  YoloDetector(std::string name, const YoloDetectorConfig & cfg, const std::string & share_dir);
  static YoloDetectorConfig load_config(ParamReader & params);

  YoloDetectorConfig cfg_;
  std::unique_ptr<YoloBackend> backend_;
  std::string backend_note_;
  std::vector<OrientedBox> boxes_;
  cv::Mat rgb_image_;
};

}  // namespace sslam

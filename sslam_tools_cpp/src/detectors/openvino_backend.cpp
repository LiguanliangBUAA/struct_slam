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

// CPU inference with OpenVINO (OpenVINO IR .xml or ONNX model exported with NMS).

#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <openvino/openvino.hpp>

#include "sslam_tools_cpp/detectors/yolo_backend.hpp"

namespace sslam
{

namespace
{

class OpenVinoBackend : public TensorYoloBackend
{
public:
  OpenVinoBackend(const std::string & model_file, int input_size, double conf, std::map<int, std::string> names)
  : TensorYoloBackend(input_size, conf, std::move(names)), model_file_(model_file)
  {
    std::shared_ptr<ov::Model> net = core_.read_model(model_file);
    const ov::PartialShape out = net->output(0).get_partial_shape();
    if (out.rank().is_dynamic() || out.rank().get_length() != 3 || !out[2].is_static() ||
      out[2].get_length() != 7)
    {
      throw std::runtime_error("YOLO model output must be [1, N, 7] (OBB exported with nms=True), got " +
              out.to_string() + ": " + model_file);
    }
    net->reshape({1, 3, size_, size_});
    compiled_ = core_.compile_model(net, "CPU");
    request_ = compiled_.create_infer_request();
  }

  std::string description() const override
  {
    return "OpenVINO " + std::string(ov::get_openvino_version().buildNumber) + " on CPU, input " +
           std::to_string(size_) + "x" + std::to_string(size_) + " (" + model_file_ + ")";
  }

protected:
  const float * run(const float * input, size_t & n_rows) override
  {
    ov::Tensor in = request_.get_input_tensor();
    std::memcpy(in.data<float>(), input, in.get_byte_size());
    request_.infer();
    const ov::Tensor out = request_.get_output_tensor(0);
    n_rows = out.get_shape()[1];
    return out.data<const float>();
  }

private:
  std::string model_file_;
  ov::Core core_;
  ov::CompiledModel compiled_;
  ov::InferRequest request_;
};

}  // namespace

std::unique_ptr<YoloBackend> make_openvino_backend(
  const std::string & model_file, int input_size, double conf, std::map<int, std::string> names)
{
  return std::make_unique<OpenVinoBackend>(model_file, input_size, conf, std::move(names));
}

}  // namespace sslam

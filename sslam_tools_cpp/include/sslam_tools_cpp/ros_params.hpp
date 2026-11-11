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

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sslam_tools_cpp/detectors/detector.hpp"

namespace sslam
{

// Declares each requested value as a ROS parameter and records "name: value" lines for the
// startup log. Numeric parameters accept both integer and float values in the YAML files
// (10 and 10.0 alike), so the configs need not match the C++ types exactly.
class RosParamReader : public ParamReader
{
public:
  explicit RosParamReader(rclcpp::Node & node)
  : node_(node) {}

  bool get_bool(const std::string & name, bool def) override
  {
    const bool v = node_.declare_parameter<bool>(name, def);
    log(name, v ? "true" : "false");
    return v;
  }

  int64_t get_int(const std::string & name, int64_t def) override
  {
    const rclcpp::ParameterValue v = declare_number(name, rclcpp::ParameterValue(def));
    int64_t out;
    if (v.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
      out = v.get<int64_t>();
    } else {
      const double d = v.get<double>();
      if (d != std::floor(d)) {
        throw std::invalid_argument("Parameter '" + name + "' must be an integer, got " + std::to_string(d));
      }
      out = static_cast<int64_t>(d);
    }
    log(name, std::to_string(out));
    return out;
  }

  double get_double(const std::string & name, double def) override
  {
    const rclcpp::ParameterValue v = declare_number(name, rclcpp::ParameterValue(def));
    const double out = v.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER ?
      static_cast<double>(v.get<int64_t>()) : v.get<double>();
    std::ostringstream s;
    s << out;
    log(name, s.str());
    return out;
  }

  std::string get_string(const std::string & name, const std::string & def) override
  {
    const std::string v = node_.declare_parameter<std::string>(name, def);
    log(name, "'" + v + "'");
    return v;
  }

  const std::string & summary() const {return summary_;}

private:
  rclcpp::ParameterValue declare_number(const std::string & name, const rclcpp::ParameterValue & def)
  {
    rcl_interfaces::msg::ParameterDescriptor desc;
    desc.dynamic_typing = true;
    const rclcpp::ParameterValue v = node_.declare_parameter(name, def, desc);
    if (v.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER &&
      v.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE)
    {
      throw std::invalid_argument("Parameter '" + name + "' must be a number");
    }
    return v;
  }

  void log(const std::string & name, const std::string & value)
  {
    summary_ += "\t" + name + ": " + value + "\n";
  }

  rclcpp::Node & node_;
  std::string summary_;
};

}  // namespace sslam

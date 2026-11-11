### LiDAR-BEV Structural Detector

#### Overview

This algorithm performs indoor wall structure detection and floor plan  generation from 3D LiDAR point clouds represented in bird’s-eye-view  (BEV) images. It integrates traditional geometric and computer vision  methods — including RANSAC, LSD, Hough Transform, and YOLO — along with  spatiotemporal fusion to achieve robust structural recognition.

#### Packages

- [coord_conversion_cpp](coord_conversion_cpp/): LiDAR point cloud -> BEV points (`/lidar_data`)
- [sslam_tools_cpp](sslam_tools_cpp/): wall detection (`detector_node`), per-scan fusion (`local_fusion`), global wall map and SLAM observations (`global_fusion`), odometry covariance relay (`odom_covariance`)
- [msg_interfaces](msg_interfaces/), [livox_ros_driver2](livox_ros_driver2/): messages

#### Usage

1. Clone this repository into a ROS 2 workspace and build it (`colcon build`).
2. LiDAR point cloud msg should follow livox msg.
3. Parameters are in [sslam_tools_cpp/config](sslam_tools_cpp/config):
   - [lidar_config.yaml](sslam_tools_cpp/config/lidar_config.yaml): LiDAR-related settings
   - [detectors_config.yaml](sslam_tools_cpp/config/detectors_config.yaml): detection method selection and algorithm parameters
   - [fusion_config.yaml](sslam_tools_cpp/config/fusion_config.yaml): spatiotemporal fusion parameters
4. Launch:
   ```bash
   ros2 launch sslam_tools_cpp divided_launch.py use_sim_time:=true \
     extra_params_file:=<overrides.yaml> lidar_topic:=/drone0/sensor_measurements/livox/lidar
   ```

#### YOLO models

Models live in `sslam_tools_cpp/resource/yolo_model` (or an absolute `yolo_model_path`), selected with
`yolo_model_name` and `device` in the detector parameters:

| Model | `device` | Engine |
|---|---|---|
| `*.pt` (not tracked by git) | `cpu` / `cuda:0` | ultralytics (PyTorch) in an embedded Python interpreter, as before |
| OpenVINO export dir / `.xml`, `.onnx` | `cpu` | OpenVINO |
| `.onnx` / `.engine` | `gpu` | TensorRT (not implemented yet) |

Exported models must include NMS: `yolo export model=<model>.pt format=openvino nms=True`.

#### License：

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](COPYING) file for the full text.

Copyright (C) 2026 Universidad Politécnica de Madrid (UPM).

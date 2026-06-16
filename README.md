### LiDAR-BEV Structural Detector

#### Overview

This algorithm performs indoor wall structure detection and floor plan  generation from 3D LiDAR point clouds represented in bird’s-eye-view  (BEV) images. It integrates traditional geometric and computer vision  methods — including RANSAC, LSD, Hough Transform, and YOLO — along with  spatiotemporal fusion to achieve robust structural recognition.

#### Usage

1. First, clone this repository.
2. LiDAR point cloud msg should follow livox msg.
3. All other modules are launched via ROS 2 launch files. Parameters can be configured in the [config](sslam_tools/config) directory:
   - [lidar_config.yaml](sslam_tools/config/lidar_config.yaml): LiDAR-related settings
   - [detector_config.yaml](sslam_tools/config/lidar_config.yaml): Detection method selection and algorithm parameters
   - [fusion_config.yaml](sslam_tools/config/fusion_config.yaml): Spatiotemporal fusion parameters
4. After building the workspace, launch the system using the launch files provided in the [sslam_tools](sslam_tools/) package: [divided_launch.py](src/sslam_tools/launch/divided_launch.py).

#### License：

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](COPYING) file for the full text.

Copyright (C) 2026 Universidad Politécnica de Madrid (UPM).

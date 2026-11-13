# Copyright 2026 Universidad Politecnica de Madrid (UPM).
#
# Author: Guanliang Li
# Contributor: Pedro Espinosa Angulo, Santiago Tapia Fernandez (supervised)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Struct SLAM front end (C++): BEV conversion, wall detection, local and global fusion."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_nodes(context):
    config_dir = os.path.join(get_package_share_directory('sslam_tools_cpp'), 'config')
    detectors_config = os.path.join(config_dir, 'detectors_config.yaml')
    lidar_config = os.path.join(config_dir, 'lidar_config.yaml')
    fusion_config = os.path.join(config_dir, 'fusion_config.yaml')
    use_sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    # Loaded last, so its values override the package configs for whichever nodes it names.
    extra_params_file = LaunchConfiguration('extra_params_file').perform(context)
    extra = [extra_params_file] if extra_params_file else []
    # Optional overrides, applied last; empty keeps the config files' values
    odometry_topic = LaunchConfiguration('odometry_topic').perform(context)
    observations_topic = LaunchConfiguration('observations_topic').perform(context)
    sensor_frame = LaunchConfiguration('sensor_frame').perform(context)
    converter_odom = [{'odometry_topic': odometry_topic}] if odometry_topic else []
    covariance_odom = [{'input_topic': odometry_topic}] if odometry_topic else []
    fusion_frame = [{'lidar_frame': sensor_frame}] if sensor_frame else []
    observations = [('semantic_observations', observations_topic)] if observations_topic else []

    return [
        Node(
            package='coord_conversion_cpp',
            executable='converter_livox',
            name='converter',
            parameters=[lidar_config, *extra, use_sim_time, *converter_odom],
            remappings=[('/livox/points', LaunchConfiguration('lidar_topic'))]
        ),
        Node(
            package='sslam_tools_cpp',
            executable='detector_node',
            name='detector_node',
            parameters=[detectors_config, *extra, use_sim_time]
        ),
        Node(
            package='sslam_tools_cpp',
            executable='local_fusion',
            name='local_fusion_node',
            parameters=[fusion_config, *extra, use_sim_time]
        ),
        Node(
            package='sslam_tools_cpp',
            executable='global_fusion',
            name='global_fusion_node',
            parameters=[fusion_config, *extra, use_sim_time, *fusion_frame],
            remappings=observations
        ),
        # /odom -> /odom_cov with the pose covariance set from parameters; passes /odom
        # through unchanged unless the extra params file sets its std values.
        Node(
            package='sslam_tools_cpp',
            executable='odom_covariance',
            name='odom_covariance_node',
            parameters=[*extra, use_sim_time, *covariance_odom]
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation clock'),
        DeclareLaunchArgument('extra_params_file', default_value='',
                              description='Optional parameter file applied on top of the package '
                                          'configs (e.g. overrides for real-flight data)'),
        DeclareLaunchArgument('lidar_topic', default_value='/livox/points'),
        DeclareLaunchArgument('odometry_topic', default_value='',
                              description='Odometry for deskew/leveling and odom_covariance input '
                                          '(empty: from the config files)'),
        DeclareLaunchArgument('observations_topic', default_value='',
                              description='Topic of the wall observations for the SLAM backend '
                                          '(empty: semantic_observations)'),
        DeclareLaunchArgument('sensor_frame', default_value='',
                              description='TF frame of the LiDAR pose for global_fusion '
                                          '(empty: lidar_frame from the config files)'),
        OpaqueFunction(function=launch_nodes),
    ])

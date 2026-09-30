# Copyright 2026 Universidad Politecnica de Madrid (UPM).
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

"""Republish odometry with a configured pose covariance.

dps_slam weighs each odometry edge by the covariance of the odometry message (with its
calculate_odom_covariance off). The odometry's own covariance does not say how much to trust
it against the wall observations, so this node sets it from parameters, next to the wall
observation covariance of global_fusion: e.g. a tiny attitude variance to keep FAST-LIO's
heading and let walls correct position only. A std <= 0 leaves that part as received.

Start anchoring: dps_slam fixes only the first keyframe, and with a uniform per-edge covariance
a loop closure spreads its residual over the whole chain, so the walls around the start (and
everything after) shift along with it. For the first anchor_distance_m of path flown the
position covariance is set to anchor_position_std_m instead, which ties the start keyframes,
and the walls seen from them, to the fixed origin; the residual then goes to the rest of the
flight (0 disables). Attitude is better left free (anchor_attitude_std_deg <= 0 keeps
attitude_std_deg): tying it too locks the odometry's early heading error into the whole map.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry


class OdomCovarianceNode(Node):
    def __init__(self):
        super().__init__('odom_covariance_node')
        self.declare_parameter('input_topic', '/odom')
        self.declare_parameter('output_topic', '/odom_cov')
        self.declare_parameter('position_std_m', 0.0)      # x, y, z
        self.declare_parameter('attitude_std_deg', 0.0)    # roll, pitch, yaw
        self.declare_parameter('anchor_distance_m', 0.0)   # path length flown; 0 disables
        self.declare_parameter('anchor_position_std_m', 0.005)
        self.declare_parameter('anchor_attitude_std_deg', 0.0)  # <= 0: attitude_std_deg
        self.position_var = self.get_parameter('position_std_m').value ** 2
        self.attitude_var = math.radians(self.get_parameter('attitude_std_deg').value) ** 2
        self.override_position = self.get_parameter('position_std_m').value > 0.0
        self.override_attitude = self.get_parameter('attitude_std_deg').value > 0.0
        self.anchor_distance = self.get_parameter('anchor_distance_m').value
        self.anchor_position_var = self.get_parameter('anchor_position_std_m').value ** 2
        anchor_attitude_std = self.get_parameter('anchor_attitude_std_deg').value
        self.anchor_attitude_var = (math.radians(anchor_attitude_std) ** 2 if anchor_attitude_std > 0.0
                                    else self.attitude_var)
        self.travelled = 0.0
        self.last_xy = None

        output_topic = self.get_parameter('output_topic').value
        input_topic = self.get_parameter('input_topic').value
        self.publisher = self.create_publisher(Odometry, output_topic, qos_profile_sensor_data)
        self.create_subscription(Odometry, input_topic, self.callback, qos_profile_sensor_data)
        self.get_logger().info(
            f"'{input_topic}' -> '{output_topic}': position std "
            f"{'%.4f m' % math.sqrt(self.position_var) if self.override_position else 'as received'}, "
            f"attitude std "
            f"{'%.4f deg' % math.degrees(math.sqrt(self.attitude_var)) if self.override_attitude else 'as received'}"
            + (f", anchored ({math.sqrt(self.anchor_position_var):.4f} m, "
               f"{math.degrees(math.sqrt(self.anchor_attitude_var)):.4f} deg) for the first "
               f"{self.anchor_distance:.1f} m" if self.anchor_distance > 0.0 else ''))

    def callback(self, msg):
        p = msg.pose.pose.position
        if self.last_xy is not None:
            self.travelled += math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
        self.last_xy = (p.x, p.y)
        anchored = self.travelled < self.anchor_distance
        position_var = self.anchor_position_var if anchored else self.position_var
        attitude_var = self.anchor_attitude_var if anchored else self.attitude_var

        cov = list(msg.pose.covariance)
        for i in range(6):
            for j in range(6):
                block_i, block_j = i < 3, j < 3
                overridden = (self.override_position if block_i else self.override_attitude) or \
                             (self.override_position if block_j else self.override_attitude)
                if overridden:
                    cov[6 * i + j] = 0.0
        if self.override_position:
            for i in range(3):
                cov[7 * i] = position_var
        if self.override_attitude:
            for i in range(3, 6):
                cov[7 * i] = attitude_var
        msg.pose.covariance = cov
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomCovarianceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

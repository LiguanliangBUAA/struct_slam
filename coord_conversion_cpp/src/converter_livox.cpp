#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "msg_interfaces/msg/lidar_data.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "elevation_grid_filter.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <eigen3/Eigen/Geometry>

using std::placeholders::_1;

class converter : public rclcpp::Node
{
public:
  converter()
  : Node("converter"), packet_counter_(0)
  {
    this->declare_parameter("input_type", "custom_msg"); // custom_msg / pointcloud2
    this->declare_parameter("height_filter_flag", "normal"); // normal / elevation_grid_filter
    // Height filtering parameters
    this->declare_parameter("MIN_Z", 0.0); // mm
    this->declare_parameter("MAX_Z", 2000.0); // mm

    this->declare_parameter("GRID_SIZE", 100.0); // mm
    this->declare_parameter("Z_DIFF_THRESHOLD", 50.0); // mm
    this->declare_parameter("POINT_COUNT_THRESHOLD", 10); // Minimum number of points in a cell to be considered valid

    this->declare_parameter("scale_to_mm", 1000.0);
    // Each 10 packets will be published as one frame
    this->declare_parameter("accumulate_packets", 10);
    this->declare_parameter("publish_pointcloud", false);

    // Gravity/attitude compensation: level the point cloud (remove roll & pitch,
    // keep yaw) using the odometry orientation at the frame's stamp before
    // height-filtering and BEV projection, so UAV tilt doesn't get baked into the
    // wall geometry.
    this->declare_parameter("gravity_compensation", true);
    // Motion compensation: move every point, using its own timestamp (Livox per-point
    // offset_time), into the sensor frame at the frame's stamp (the latest point), so
    // an accumulated frame is one consistent snapshot even while the UAV turns.
    this->declare_parameter("deskew", true);
    this->declare_parameter("odometry_topic", "/odom");

    this->input_type_ = this->get_parameter("input_type").as_string();
    this->height_filter_flag_ = this->get_parameter("height_filter_flag").as_string();
    this->min_z_ = static_cast<float>(this->get_parameter("MIN_Z").as_double());
    this->max_z_ = static_cast<float>(this->get_parameter("MAX_Z").as_double());
    this->grid_size_ = static_cast<float>(this->get_parameter("GRID_SIZE").as_double());
    this->z_diff_threshold_ = static_cast<float>(this->get_parameter("Z_DIFF_THRESHOLD").as_double());
    this->point_count_threshold_ = this->get_parameter("POINT_COUNT_THRESHOLD").as_int();
    this->scale_to_mm_ = static_cast<float>(this->get_parameter("scale_to_mm").as_double());
    this->accumulate_packets_ = this->get_parameter("accumulate_packets").as_int();
    this->publish_pointcloud_ = this->get_parameter("publish_pointcloud").as_bool();
    this->gravity_compensation_ = this->get_parameter("gravity_compensation").as_bool();
    this->deskew_ = this->get_parameter("deskew").as_bool();
    std::string odometry_topic = this->get_parameter("odometry_topic").as_string();

    if (this->gravity_compensation_ || this->deskew_) {
      this->odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odometry_topic, rclcpp::SensorDataQoS(),
        std::bind(&converter::odom_callback, this, _1));
      RCLCPP_INFO(
        this->get_logger(), "Gravity compensation %s, deskew %s, using odometry topic: %s",
        gravity_compensation_ ? "enabled" : "disabled", deskew_ ? "enabled" : "disabled",
        odometry_topic.c_str());
    }

    if (input_type_ == "custom_msg") {
      this->subscription_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        "/livox/lidar", rclcpp::SensorDataQoS(), std::bind(&converter::topic_callback, this, _1));
    } else if (input_type_ == "pointcloud2") {
      this->pc2_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/points", rclcpp::SensorDataQoS(), std::bind(&converter::pointcloud2_callback, this, _1));
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown input_type: %s", input_type_.c_str());
    }

    this->publisher_ = this->create_publisher<msg_interfaces::msg::LidarData>("lidar_data", 10);

    if (publish_pointcloud_) {
      this->publisher_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_points", 10);
    }

    if (height_filter_flag_ == "elevation_grid_filter") {
      grid_filter_ = std::make_unique<elevation_grid_filter::ElevationGridFilter>(grid_size_, z_diff_threshold_, point_count_threshold_);
    }

    acc_points_.reserve(300000);
    acc_stamps_ns_.reserve(300000);
    raw_x_.reserve(300000);
    raw_y_.reserve(300000);
    raw_z_.reserve(300000);

    RCLCPP_INFO(this->get_logger(), "Livox Raw CustomMsg Converter initialized.");
    RCLCPP_INFO(this->get_logger(), "Accumulating %d packets per frame.", accumulate_packets_);
  }

private:
  struct PoseSample
  {
    int64_t stamp_ns;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
  };

  std::string input_type_;
  std::string height_filter_flag_;
  float min_z_; // mm
  float max_z_;
  float grid_size_; // mm
  float z_diff_threshold_; // mm
  int point_count_threshold_;
  float scale_to_mm_;
  int accumulate_packets_;
  int packet_counter_;
  bool publish_pointcloud_;

  bool gravity_compensation_;
  bool deskew_;
  // Odometry of the robot frame, assumed rigidly aligned with the sensor frame (identity
  // extrinsic rotation, translation ignored), which is how the sessions set up livox_frame.
  std::mutex odom_mutex_;
  std::deque<PoseSample> odom_buffer_;
  static constexpr int64_t kOdomBufferNs = 2'000'000'000;   // keep 2 s of odometry
  static constexpr int64_t kMaxExtrapolationNs = 50'000'000;  // use the nearest pose within 50 ms
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

  // Points of the frame being accumulated: sensor frame at their own capture time.
  std::vector<Eigen::Vector3f> acc_points_;
  std::vector<int64_t> acc_stamps_ns_;
  std::string frame_id_;

  std::vector<float> raw_x_;
  std::vector<float> raw_y_;
  std::vector<float> raw_z_;
  std::vector<float> filtered_x_;
  std::vector<float> filtered_y_;
  std::vector<float> filtered_z_;

  std::unique_ptr<elevation_grid_filter::ElevationGridFilter> grid_filter_;

  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc2_subscription_;
  rclcpp::Publisher<msg_interfaces::msg::LidarData>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_cloud_;

  static int64_t to_ns(const builtin_interfaces::msg::Time & t)
  {
    return static_cast<int64_t>(t.sec) * 1'000'000'000 + t.nanosec;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const auto & p = msg->pose.pose.position;
    const auto & q = msg->pose.pose.orientation;
    PoseSample sample{to_ns(msg->header.stamp), Eigen::Vector3d(p.x, p.y, p.z),
      Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized()};

    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!odom_buffer_.empty() && sample.stamp_ns <= odom_buffer_.back().stamp_ns) {
      odom_buffer_.clear();  // time went backwards (bag restarted / looped)
    }
    odom_buffer_.push_back(sample);
    while (odom_buffer_.front().stamp_ns < sample.stamp_ns - kOdomBufferNs) {
      odom_buffer_.pop_front();
    }
  }

  // Robot pose at stamp_ns, interpolated (linear position, slerp orientation). Outside the
  // buffered span, the nearest pose is used if within kMaxExtrapolationNs. Needs odom_mutex_.
  bool pose_at(int64_t stamp_ns, PoseSample & out) const
  {
    if (odom_buffer_.empty()) {return false;}
    if (stamp_ns <= odom_buffer_.front().stamp_ns) {
      out = odom_buffer_.front();
      return odom_buffer_.front().stamp_ns - stamp_ns <= kMaxExtrapolationNs;
    }
    if (stamp_ns >= odom_buffer_.back().stamp_ns) {
      out = odom_buffer_.back();
      return stamp_ns - odom_buffer_.back().stamp_ns <= kMaxExtrapolationNs;
    }
    auto after = std::lower_bound(
      odom_buffer_.begin(), odom_buffer_.end(), stamp_ns,
      [](const PoseSample & s, int64_t t) {return s.stamp_ns < t;});
    auto before = std::prev(after);
    double w = static_cast<double>(stamp_ns - before->stamp_ns) /
      static_cast<double>(after->stamp_ns - before->stamp_ns);
    out.stamp_ns = stamp_ns;
    out.position = before->position + w * (after->position - before->position);
    out.orientation = before->orientation.slerp(w, after->orientation);
    return true;
  }

  // Roll/pitch-only ("leveling") rotation of an orientation: applied to a sensor-frame point
  // it removes the tilt and keeps the yaw.
  static Eigen::Matrix3f leveling_rotation(const Eigen::Quaterniond & q)
  {
    double roll = std::atan2(
      2.0 * (q.w() * q.x() + q.y() * q.z()), 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));
    double sin_pitch = std::clamp(2.0 * (q.w() * q.y() - q.z() * q.x()), -1.0, 1.0);
    double pitch = std::asin(sin_pitch);
    return (Eigen::AngleAxisf(static_cast<float>(pitch), Eigen::Vector3f::UnitY()) *
           Eigen::AngleAxisf(static_cast<float>(roll), Eigen::Vector3f::UnitX())).toRotationMatrix();
  }

  void topic_callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
  {
    // Livox per-point time: packet timebase + offset_time (both ns).
    for (const auto & point : msg->points) {
      acc_points_.emplace_back(point.x, point.y, point.z);
      acc_stamps_ns_.push_back(static_cast<int64_t>(msg->timebase + point.offset_time));
    }
    frame_id_ = msg->header.frame_id;
    process_packet();
  }

  void pointcloud2_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // No per-point time: the whole message is one capture at its stamp.
    const int64_t stamp_ns = to_ns(msg->header.stamp);
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      acc_points_.emplace_back(*iter_x, *iter_y, *iter_z);
      acc_stamps_ns_.push_back(stamp_ns);
    }
    frame_id_ = msg->header.frame_id;
    process_packet();
  }

  void process_packet()
  {
    packet_counter_++;
    if (packet_counter_ < accumulate_packets_) {return;}
    packet_counter_ = 0;
    if (acc_points_.empty()) {return;}

    // The frame is stamped at its latest point; with deskew every point is moved there.
    const int64_t ref_ns = *std::max_element(acc_stamps_ns_.begin(), acc_stamps_ns_.end());
    Eigen::Matrix3f level = Eigen::Matrix3f::Identity();
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      PoseSample ref;
      const bool have_ref = pose_at(ref_ns, ref);
      if (have_ref && gravity_compensation_) {
        level = leveling_rotation(ref.orientation);
      }
      if (have_ref && deskew_) {
        const Eigen::Matrix3d ref_rot_inv = ref.orientation.toRotationMatrix().transpose();
        PoseSample at_point;
        int64_t cached_ns = -1;
        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();
        Eigen::Vector3f trans = Eigen::Vector3f::Zero();
        for (size_t i = 0; i < acc_points_.size(); ++i) {
          // Points arrive in time order; reuse the transform while the stamp repeats.
          if (acc_stamps_ns_[i] != cached_ns) {
            cached_ns = acc_stamps_ns_[i];
            if (pose_at(cached_ns, at_point)) {
              // p_ref = R_ref^T * (R_i * p + t_i - t_ref)
              rot = (ref_rot_inv * at_point.orientation.toRotationMatrix()).cast<float>();
              trans = (ref_rot_inv * (at_point.position - ref.position)).cast<float>();
            } else {
              rot.setIdentity();
              trans.setZero();
            }
          }
          acc_points_[i] = rot * acc_points_[i] + trans;
        }
      } else if (!have_ref && (deskew_ || gravity_compensation_)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "No odometry around the scan stamp: publishing it without deskew/leveling");
      }
    }

    // Level, height-filter, and project to the BEV plane (mm).
    for (const auto & p_sensor : acc_points_) {
      const Eigen::Vector3f p = level * p_sensor;
      const float z_mm = p.z() * scale_to_mm_;
      if (z_mm >= min_z_ && z_mm <= max_z_) {
        raw_x_.push_back(p.x() * scale_to_mm_);
        raw_y_.push_back(p.y() * scale_to_mm_);
        raw_z_.push_back(z_mm);
      }
    }
    if (height_filter_flag_ == "elevation_grid_filter") {
      grid_filter_->filter(raw_x_, raw_y_, raw_z_, filtered_x_, filtered_y_, filtered_z_);
    } else {
      filtered_x_ = raw_x_;
      filtered_y_ = raw_y_;
      filtered_z_ = raw_z_;
    }

    std_msgs::msg::Header header;
    header.stamp = rclcpp::Time(ref_ns, RCL_ROS_TIME);
    header.frame_id = frame_id_;  // livox_frame

    auto message = std::make_unique<msg_interfaces::msg::LidarData>();
    message->header = header;
    message->x_data = filtered_x_;
    message->y_data = filtered_y_;
    this->publisher_->publish(std::move(message));

    if (publish_pointcloud_) {
      // Publish the kept points back in the (unleveled) sensor frame at the stamp, so that
      // viewers placing them through TF, which includes the tilt, show them correctly.
      // One leveling rotation serves the whole frame, so its transpose undoes it exactly.
      const Eigen::Matrix3f unlevel = level.transpose();
      auto pc2_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
      pc2_msg->header = header;
      pc2_msg->height = 1;
      pc2_msg->width = filtered_z_.size();

      sensor_msgs::PointCloud2Modifier modifier(*pc2_msg);
      modifier.setPointCloud2FieldsByString(1, "xyz");
      modifier.resize(filtered_z_.size());

      sensor_msgs::PointCloud2Iterator<float> iter_x(*pc2_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(*pc2_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(*pc2_msg, "z");

      for (size_t i = 0; i < filtered_z_.size(); ++i) {
        const Eigen::Vector3f p = unlevel *
          Eigen::Vector3f(filtered_x_[i], filtered_y_[i], filtered_z_[i]) * 0.001f;
        *iter_x = p.x();
        *iter_y = p.y();
        *iter_z = p.z();
        ++iter_x;
        ++iter_y;
        ++iter_z;
      }

      this->publisher_cloud_->publish(std::move(pc2_msg));
    }

    acc_points_.clear();
    acc_stamps_ns_.clear();
    raw_x_.clear();
    raw_y_.clear();
    raw_z_.clear();
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<converter>());
  rclcpp::shutdown();
  return 0;
}

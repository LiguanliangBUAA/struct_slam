cmake_minimum_required(VERSION 3.8)
project(sslam_tools_cpp)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(msg_interfaces REQUIRED)
find_package(dps_slam_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(yaml-cpp REQUIRED)

# OpenVINO is the CPU backend of the YOLO-OBB detector (device: cpu). The pip wheel
# (pip install openvino) ships the C++ headers and CMake config, so look there too.
# The GPU backend (device: gpu) will be TensorRT; not implemented yet.
if(NOT OpenVINO_DIR)
  execute_process(
    COMMAND python3 -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))"
    OUTPUT_VARIABLE _ov_pip_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
find_package(OpenVINO QUIET COMPONENTS Runtime HINTS ${_ov_pip_cmake_dir})
if(OpenVINO_FOUND)
  message(STATUS "OpenVINO ${OpenVINO_VERSION} found: YOLO CPU backend enabled")
else()
  message(WARNING "OpenVINO not found: YOLO detector without CPU backend")
endif()

# ---- Core library (no ROS) ----
add_library(sslam_core SHARED
  src/core/geometry.cpp
  src/core/clustering.cpp
  src/core/assignment.cpp
  src/core/global_map.cpp
  src/detectors/detector.cpp
  src/detectors/image_based_detector.cpp
  src/detectors/lsd_detector.cpp
  src/detectors/hough_detector.cpp
  src/detectors/ransac_detector.cpp
  src/detectors/yolo_detector.cpp
  src/detectors/yolo_backend.cpp
)
target_include_directories(sslam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(sslam_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS} yaml-cpp)
target_include_directories(sslam_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

if(OpenVINO_FOUND)
  target_sources(sslam_core PRIVATE src/detectors/openvino_backend.cpp)
  target_link_libraries(sslam_core PRIVATE openvino::runtime)
  target_compile_definitions(sslam_core PRIVATE SSLAM_WITH_OPENVINO)
endif()

# PyTorch .pt models (as the Python sslam_tools, CPU or CUDA) run through ultralytics in an
# embedded Python interpreter; needs python3 + ultralytics at run time.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  message(STATUS "pybind11 ${pybind11_VERSION} found: YOLO .pt (ultralytics) backend enabled")
  target_sources(sslam_core PRIVATE src/detectors/ultralytics_backend.cpp)
  target_link_libraries(sslam_core PRIVATE pybind11::embed)
  target_compile_definitions(sslam_core PRIVATE SSLAM_WITH_PYTHON)
  # pybind11 objects held by a class: keep that file's symbols hidden, as pybind11 expects
  set_source_files_properties(src/detectors/ultralytics_backend.cpp PROPERTIES COMPILE_OPTIONS "-fvisibility=hidden")
else()
  message(WARNING "pybind11 not found: YOLO .pt models unsupported (OpenVINO/ONNX only)")
endif()

# ---- Nodes ----
set(node_deps rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros
  msg_interfaces dps_slam_msgs ament_index_cpp)

function(add_sslam_node name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} sslam_core)
  ament_target_dependencies(${name} ${node_deps})
  # The pip OpenVINO libraries live outside the system paths
  set_target_properties(${name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endfunction()

add_sslam_node(detector_node src/nodes/detector_node.cpp)
add_sslam_node(local_fusion src/nodes/local_fusion_node.cpp)
add_sslam_node(global_fusion src/nodes/global_fusion_node.cpp)
add_sslam_node(odom_covariance src/nodes/odom_covariance_node.cpp)

set_target_properties(sslam_core PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS sslam_core
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS detector_node local_fusion global_fusion odom_covariance
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config resource DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core sslam_core)
endif()

ament_package()

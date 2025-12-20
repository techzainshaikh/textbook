---
title: Perception Pipelines for Robotics AI
sidebar_position: 4
description: Building AI-powered perception systems for humanoid robots using Isaac ROS
keywords: [perception, computer vision, ai, robotics, Isaac ROS, sensor fusion]
---

# Chapter 3: Perception Pipelines for Robotics AI

## Learning Objectives

By the end of this chapter, students will be able to:
- Design and implement perception pipelines using Isaac ROS for humanoid robots
- Integrate multiple sensor modalities for robust perception
- Implement sensor fusion algorithms for enhanced environment understanding
- Apply deep learning models for object detection and scene understanding
- Validate perception system performance in simulation and real-world scenarios

## Prerequisites

Students should have:
- Understanding of computer vision fundamentals (covered in Module 2)
- Knowledge of sensor simulation (covered in Module 2, Chapter 2)
- Experience with ROS 2 messaging (covered in Module 1)
- Basic knowledge of deep learning and neural networks

## Core Concepts

Perception pipelines form the sensory foundation of AI-powered humanoid robots, enabling them to understand and interact with their environment. Isaac ROS provides GPU-accelerated perception capabilities that leverage NVIDIA hardware for real-time performance.

### Perception Pipeline Architecture

**Sensor Input Layer:**
- Multiple sensor modalities (LiDAR, cameras, IMU, GPS, etc.)
- Synchronized data acquisition and timestamp management
- Raw data preprocessing and calibration

**Feature Extraction Layer:**
- GPU-accelerated computer vision algorithms
- Deep learning inference for object detection/segmentation
- Geometric feature extraction (edges, corners, surfaces)

**Data Association Layer:**
- Feature matching and correspondence establishment
- Temporal association for tracking
- Multi-view geometric relationships

**State Estimation Layer:**
- Filtering and prediction algorithms (Kalman, particle filters)
- Sensor fusion for state estimation
- Uncertainty quantification and propagation

### Isaac ROS Perception Framework

Isaac ROS provides several key perception capabilities:
- **Isaac ROS Apriltag**: High-precision fiducial marker detection
- **Isaac ROS Stereo Dense Reconstruction**: 3D scene reconstruction
- **Isaac ROS Detection ROS**: Object detection with deep learning
- **Isaac ROS Visual Slam**: Visual simultaneous localization and mapping
- **Isaac ROS ISAAC**: Inertial sensor array conditioning

## Implementation

Let's implement a comprehensive perception pipeline using Isaac ROS:

### Isaac ROS Perception Pipeline Setup

```python
#!/usr/bin/env python3
# perception_pipeline.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import message_filters
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs

class IsaacPerceptionPipeline(Node):
    """
    Perception pipeline using Isaac ROS for humanoid robot
    """

    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # Publishers for perception outputs
        self.object_detections_pub = self.create_publisher(String, '/humanoid/perception/objects', 10)
        self.tracked_objects_pub = self.create_publisher(String, '/humanoid/perception/tracked_objects', 10)
        self.environment_map_pub = self.create_publisher(String, '/humanoid/perception/environment_map', 10)
        self.status_pub = self.create_publisher(String, '/humanoid/perception/status', 10)

        # Subscribers with message filters for synchronization
        self.rgb_sub = message_filters.Subscriber(self, Image, '/humanoid/camera/rgb/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/humanoid/camera/depth/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/humanoid/camera/rgb/camera_info')

        # Synchronize RGB, depth, and camera info
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.camera_callback)

        # LiDAR subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/humanoid/lidar/scan',
            self.lidar_callback,
            10
        )

        # IMU subscriber
        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid/imu/data',
            self.imu_callback,
            10
        )

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # Perception state
        self.objects_in_view = []
        self.tracked_objects_history = {}
        self.environment_map = {}
        self.robot_pose = None

        # Timers
        self.perception_timer = self.create_timer(0.1, self.process_perception_data)  # 10 Hz
        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info('Isaac Perception Pipeline initialized')

    def camera_callback(self, rgb_msg, depth_msg, info_msg):
        """Process synchronized camera data"""
        try:
            # Convert ROS images to OpenCV format
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, '32FC1')

            # Process perception pipeline
            self.process_camera_perception(rgb_image, depth_image, info_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing camera data: {e}')

    def lidar_callback(self, msg):
        """Process LiDAR data for environment mapping"""
        try:
            # Process LiDAR scan for obstacle detection and mapping
            self.process_lidar_perception(msg)

        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {e}')

    def imu_callback(self, msg):
        """Process IMU data for state estimation"""
        try:
            # Process IMU data for orientation and motion estimation
            self.process_imu_perception(msg)

        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def process_camera_perception(self, rgb_image, depth_image, camera_info):
        """Process camera perception pipeline"""
        # Step 1: Object detection using Isaac ROS (simulated)
        detected_objects = self.detect_objects_isaac(rgb_image)

        # Step 2: Depth-based filtering and 3D positioning
        positioned_objects = self.filter_by_depth(detected_objects, depth_image, camera_info)

        # Step 3: Track objects temporally
        tracked_objects = self.track_objects(positioned_objects)

        # Step 4: Update environment map
        self.update_environment_map(tracked_objects)

        # Publish results
        self.publish_object_detections(tracked_objects)

    def detect_objects_isaac(self, image):
        """Simulate Isaac ROS object detection"""
        # In real implementation, this would call Isaac ROS object detection nodes
        # For this example, we'll simulate object detection using OpenCV

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Simple blob detection as a proxy for object detection
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)

        # Convert to object format
        objects = []
        for kp in keypoints:
            if kp.size > 10:  # Filter small detections
                obj = {
                    'center': (int(kp.pt[0]), int(kp.pt[1])),
                    'size': kp.size,
                    'confidence': min(kp.response * 10, 1.0)  # Normalize confidence
                }
                objects.append(obj)

        return objects

    def filter_by_depth(self, objects, depth_image, camera_info):
        """Filter objects by depth and estimate 3D positions"""
        # Camera intrinsic parameters
        fx = camera_info.K[0]  # Focal length x
        fy = camera_info.K[4]  # Focal length y
        cx = camera_info.K[2]  # Principal point x
        cy = camera_info.K[5]  # Principal point y

        positioned_objects = []

        for obj in objects:
            center_x, center_y = obj['center']

            # Get depth at object center
            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]

                if depth > 0 and np.isfinite(depth):
                    # Calculate 3D position from 2D pixel + depth
                    x = (center_x - cx) * depth / fx
                    y = (center_y - cy) * depth / fy
                    z = depth

                    positioned_obj = {
                        'name': f'object_{len(positioned_objects)}',
                        'position_3d': [x, y, z],
                        'position_2d': obj['center'],
                        'size': obj['size'],
                        'confidence': obj['confidence'],
                        'timestamp': self.get_clock().now().seconds_nanoseconds()
                    }

                    positioned_objects.append(positioned_obj)

        return positioned_objects

    def track_objects(self, positioned_objects):
        """Track objects across frames"""
        current_time = self.get_clock().now()

        for obj in positioned_objects:
            obj_id = obj['name']

            if obj_id in self.tracked_objects_history:
                # Update existing track
                prev_pos = self.tracked_objects_history[obj_id]['position_3d']
                current_pos = obj['position_3d']

                # Calculate velocity
                dt = (current_time.nanoseconds - self.tracked_objects_history[obj_id]['timestamp'][1]) / 1e9
                if dt > 0:
                    velocity = [(cp - pp) / dt for cp, pp in zip(current_pos, prev_pos)]
                    obj['velocity'] = velocity
                else:
                    obj['velocity'] = [0, 0, 0]

                # Update history
                self.tracked_objects_history[obj_id].update(obj)
            else:
                # Initialize new track
                obj['velocity'] = [0, 0, 0]
                obj['track_start_time'] = current_time
                self.tracked_objects_history[obj_id] = obj

        # Clean up old tracks
        self.cleanup_old_tracks(current_time)

        return list(self.tracked_objects_history.values())

    def cleanup_old_tracks(self, current_time):
        """Remove old object tracks that haven't been updated"""
        # Remove tracks older than 5 seconds
        old_tracks = []
        for obj_id, track in self.tracked_objects_history.items():
            track_age = current_time.nanoseconds - track['timestamp'][1]
            if track_age > 5e9:  # 5 seconds in nanoseconds
                old_tracks.append(obj_id)

        for obj_id in old_tracks:
            del self.tracked_objects_history[obj_id]

    def process_lidar_perception(self, lidar_msg):
        """Process LiDAR data for environment mapping"""
        # Process LiDAR ranges to detect obstacles and map environment
        ranges = np.array(lidar_msg.ranges)
        angles = np.linspace(lidar_msg.angle_min, lidar_msg.angle_max, len(ranges))

        # Filter valid ranges
        valid_mask = np.isfinite(ranges) & (ranges > lidar_msg.range_min) & (ranges < lidar_msg.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # Convert to Cartesian coordinates
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)

        # Create occupancy grid or point cloud representation
        environment_points = list(zip(x_coords, y_coords))

        # Update environment map with LiDAR data
        self.environment_map['lidar_points'] = environment_points
        self.environment_map['timestamp'] = self.get_clock().now().seconds_nanoseconds()

    def process_imu_perception(self, imu_msg):
        """Process IMU data for state estimation"""
        # Extract orientation from IMU
        orientation = [
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ]

        # Extract angular velocity
        angular_vel = [
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ]

        # Extract linear acceleration
        linear_acc = [
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ]

        # Update internal state
        self.robot_state = {
            'orientation': orientation,
            'angular_velocity': angular_vel,
            'linear_acceleration': linear_acc,
            'timestamp': self.get_clock().now().seconds_nanoseconds()
        }

    def update_environment_map(self, tracked_objects):
        """Update environment map with perception results"""
        # Combine object detections with LiDAR environment mapping
        if 'lidar_points' in self.environment_map:
            self.environment_map['objects'] = tracked_objects
            self.environment_map['updated'] = self.get_clock().now().seconds_nanoseconds()

    def publish_object_detections(self, tracked_objects):
        """Publish object detections to ROS topic"""
        if not tracked_objects:
            return

        # Create JSON-like message with object information
        detection_msg = String()
        detection_data = {
            'timestamp': self.get_clock().now().seconds_nanoseconds(),
            'objects': [
                {
                    'id': obj['name'],
                    'position_3d': obj['position_3d'],
                    'position_2d': obj['position_2d'],
                    'velocity': obj.get('velocity', [0, 0, 0]),
                    'confidence': obj['confidence']
                } for obj in tracked_objects if obj['confidence'] > 0.5  # Filter low-confidence detections
            ]
        }

        detection_msg.data = str(detection_data)
        self.object_detections_pub.publish(detection_msg)

    def process_perception_data(self):
        """Main perception processing loop"""
        # This would coordinate all perception modules
        # In a real implementation, this would run more sophisticated processing
        pass

    def publish_status(self):
        """Publish perception system status"""
        status_msg = String()
        status_msg.data = f"Perception system active - Objects detected: {len(self.tracked_objects_history)}, " \
                         f"Environment points: {len(self.environment_map.get('lidar_points', [])) if self.environment_map else 0}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    perception_node = IsaacPerceptionPipeline()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down Isaac Perception Pipeline...')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Sensor Fusion Implementation

```python
#!/usr/bin/env python3
# sensor_fusion.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial.transform import Rotation as R
import message_filters

class IsaacSensorFusion(Node):
    """
    Sensor fusion system using Isaac ROS capabilities
    """

    def __init__(self):
        super().__init__('isaac_sensor_fusion')

        # Publishers
        self.fused_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/humanoid/perception/fused_pose', 10)
        self.fused_twist_pub = self.create_publisher(TwistStamped, '/humanoid/perception/fused_twist', 10)
        self.fusion_status_pub = self.create_publisher(Float32MultiArray, '/humanoid/perception/fusion_status', 10)

        # Subscribers with synchronization
        self.imu_sub = message_filters.Subscriber(self, Imu, '/humanoid/imu/data_raw')
        self.lidar_sub = message_filters.Subscriber(self, LaserScan, '/humanoid/lidar/scan')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/humanoid/odom')

        # Synchronize sensors
        self.sensor_sync = message_filters.ApproximateTimeSynchronizer(
            [self.imu_sub, self.lidar_sub, self.odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.sensor_sync.registerCallback(self.sensors_callback)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State estimation variables
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        # Covariance matrices
        self.position_covariance = np.eye(3) * 0.1
        self.orientation_covariance = np.eye(3) * 0.01

        # Kalman filter parameters
        self.process_noise = np.eye(6) * 0.01
        self.measurement_noise = np.eye(6) * 0.1

        # Timers
        self.fusion_timer = self.create_timer(0.05, self.run_sensor_fusion)  # 20 Hz

        self.get_logger().info('Isaac Sensor Fusion initialized')

    def sensors_callback(self, imu_msg, lidar_msg, odom_msg):
        """Process synchronized sensor data"""
        # Update internal state with new sensor readings
        self.update_imu_state(imu_msg)
        self.update_odom_state(odom_msg)
        self.update_lidar_state(lidar_msg)

    def update_imu_state(self, imu_msg):
        """Update state with IMU data"""
        # Update orientation from IMU
        self.orientation = np.array([
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ])

        # Update angular velocity
        self.angular_velocity = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])

        # Update linear acceleration (for state prediction)
        linear_acc = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        # Integrate acceleration to update velocity and position
        dt = 0.05  # Assuming 20Hz update rate
        self.velocity += linear_acc * dt
        self.position += self.velocity * dt

    def update_odom_state(self, odom_msg):
        """Update state with odometry data"""
        # Extract position from odometry
        pos = odom_msg.pose.pose.position
        self.position = np.array([pos.x, pos.y, pos.z])

        # Extract orientation from odometry
        orient = odom_msg.pose.pose.orientation
        self.orientation = np.array([orient.x, orient.y, orient.z, orient.w])

        # Extract velocity from odometry
        vel = odom_msg.twist.twist.linear
        self.velocity = np.array([vel.x, vel.y, vel.z])

    def update_lidar_state(self, lidar_msg):
        """Update state with LiDAR-based position estimates"""
        # Process LiDAR scan for position corrections
        # This would typically involve scan matching or landmark detection
        # For this example, we'll just use it to validate position estimates

        # Example: Calculate distance to nearest obstacle
        valid_ranges = [r for r in lidar_msg.ranges if np.isfinite(r)]
        if valid_ranges:
            min_range = min(valid_ranges)

            # If we're getting too close to obstacles, this might indicate
            # a problem with our position estimate
            if min_range < 0.5:  # Less than 50cm to obstacle
                self.get_logger().warn(f'Close to obstacle: {min_range:.2f}m - check position estimate')

    def run_sensor_fusion(self):
        """Execute sensor fusion algorithm"""
        try:
            # Apply Kalman filter to fuse sensor data
            self.predict_state()
            self.update_state_with_measurements()
            self.publish_fused_state()

        except Exception as e:
            self.get_logger().error(f'Error in sensor fusion: {e}')

    def predict_state(self):
        """Predict state based on motion model"""
        # Use IMU data for prediction
        dt = 0.05  # Time step

        # Predict position based on current velocity
        self.position += self.velocity * dt

        # Predict velocity based on IMU acceleration
        # (would need to transform IMU acceleration to world frame)
        # For simplicity, we'll just keep the current velocity with some decay

    def update_state_with_measurements(self):
        """Update state with sensor measurements"""
        # This is where the Kalman update step would occur
        # In a real implementation, we would fuse:
        # - Visual-inertial odometry
        # - LiDAR-based position estimates
        # - Wheel odometry
        # - IMU measurements

        # For this example, we'll just apply simple averaging
        # with different weights for different sensor types

        # Weighted fusion would happen here
        pass

    def publish_fused_state(self):
        """Publish fused state estimate"""
        # Publish fused pose with covariance
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Fill position
        pose_msg.pose.pose.position.x = self.position[0]
        pose_msg.pose.pose.position.y = self.position[1]
        pose_msg.pose.pose.position.z = self.position[2]

        # Fill orientation
        pose_msg.pose.pose.orientation.x = self.orientation[0]
        pose_msg.pose.pose.orientation.y = self.orientation[1]
        pose_msg.pose.pose.orientation.z = self.orientation[2]
        pose_msg.pose.pose.orientation.w = self.orientation[3]

        # Fill covariance (flattened 6x6 matrix)
        cov_matrix = np.zeros(36)
        cov_matrix[0] = self.position_covariance[0, 0]  # xx
        cov_matrix[7] = self.position_covariance[1, 1]  # yy
        cov_matrix[14] = self.position_covariance[2, 2]  # zz
        cov_matrix[21] = self.orientation_covariance[0, 0]  # rr
        cov_matrix[28] = self.orientation_covariance[1, 1]  # pp
        cov_matrix[35] = self.orientation_covariance[2, 2]  # yy

        pose_msg.pose.covariance = cov_matrix.tolist()

        self.fused_pose_pub.publish(pose_msg)

        # Publish fused twist
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'

        twist_msg.twist.linear.x = self.velocity[0]
        twist_msg.twist.linear.y = self.velocity[1]
        twist_msg.twist.linear.z = self.velocity[2]

        twist_msg.twist.angular.x = self.angular_velocity[0]
        twist_msg.twist.angular.y = self.angular_velocity[1]
        twist_msg.twist.angular.z = self.angular_velocity[2]

        self.fused_twist_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = IsaacSensorFusion()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        fusion_node.get_logger().info('Shutting down Isaac Sensor Fusion...')
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Examples

### Example 1: Isaac ROS Visual SLAM Integration

```python
#!/usr/bin/env python3
# visual_slam_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import cv2
from cv_bridge import CvBridge
import tf2_ros

class IsaacVisualSLAMIntegration(Node):
    """
    Integrate Isaac ROS Visual SLAM with the humanoid robot
    """

    def __init__(self):
        super().__init__('isaac_visual_slam_integration')

        # Publishers
        self.map_pub = self.create_publisher(MarkerArray, '/humanoid/vslam/map', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/humanoid/vslam/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/humanoid/vslam/odom', 10)

        # Subscribers
        self.left_image_sub = self.create_subscription(
            Image,
            '/humanoid/stereo/left/image_rect_color',
            self.left_image_callback,
            10
        )

        self.right_image_sub = self.create_subscription(
            Image,
            '/humanoid/stereo/right/image_rect_color',
            self.right_image_callback,
            10
        )

        self.left_info_sub = self.create_subscription(
            CameraInfo,
            '/humanoid/stereo/left/camera_info',
            self.camera_info_callback,
            10
        )

        # CV Bridge
        self.cv_bridge = CvBridge()

        # SLAM state
        self.left_image = None
        self.right_image = None
        self.camera_info = None
        self.feature_points = []  # 3D points in the map
        self.robot_trajectory = []  # Robot path

        # Timers
        self.processing_timer = self.create_timer(0.1, self.process_slam_step)  # 10 Hz
        self.mapping_timer = self.create_timer(1.0, self.update_map_visualization)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info('Isaac Visual SLAM Integration initialized')

    def left_image_callback(self, msg):
        """Process left stereo camera image"""
        try:
            self.left_image = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            self.get_logger().error(f'Error processing left image: {e}')

    def right_image_callback(self, msg):
        """Process right stereo camera image"""
        try:
            self.right_image = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            self.get_logger().error(f'Error processing right image: {e}')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_info = msg

    def process_slam_step(self):
        """Process one step of visual SLAM"""
        if self.left_image is None or self.right_image is None or self.camera_info is None:
            return

        try:
            # Step 1: Feature extraction
            features_left = self.extract_features(self.left_image)
            features_right = self.extract_features(self.right_image)

            # Step 2: Stereo matching to get 3D points
            points_3d = self.stereo_match(features_left, features_right)

            # Step 3: Track features across frames
            tracked_features = self.track_features(features_left)

            # Step 4: Estimate motion using 3D-2D correspondences
            motion_estimate = self.estimate_motion(tracked_features)

            # Step 5: Update map and robot pose
            self.update_map(points_3d)
            self.update_robot_pose(motion_estimate)

        except Exception as e:
            self.get_logger().error(f'Error in SLAM processing: {e}')

    def extract_features(self, image):
        """Extract features from image (simplified implementation)"""
        # In a real Isaac ROS implementation, this would use hardware-accelerated feature extraction
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Use ORB for feature detection (would be SIFT/SURF in Isaac ROS)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if keypoints is not None:
            features = [{'pt': kp.pt, 'angle': kp.angle, 'response': kp.response} for kp in keypoints]
        else:
            features = []

        return features

    def stereo_match(self, left_features, right_features):
        """Match features between stereo images to get 3D points"""
        if len(left_features) < 2 or len(right_features) < 2:
            return []

        # Simple descriptor matching
        left_pts = np.array([f['pt'] for f in left_features])
        right_pts = np.array([f['pt'] for f in right_features])

        # In a real implementation, we'd use the camera calibration to triangulate
        # For this example, we'll simulate 3D points based on disparity
        points_3d = []

        # Simple stereo matching (would be more sophisticated in Isaac ROS)
        for left_pt in left_pts:
            # Find corresponding point in right image (simplified)
            disparities = np.abs(right_pts[:, 0] - left_pt[0])
            if len(disparities) > 0:
                closest_idx = np.argmin(disparities)
                disparity = disparities[closest_idx]

                if disparity > 0:  # Valid match
                    # Simple triangulation (would use actual camera params in real implementation)
                    z = 0.1 / disparity if disparity > 0.01 else 10.0  # Depth estimate
                    x = left_pt[0] * z / 500  # Rough conversion to 3D
                    y = left_pt[1] * z / 500  # Rough conversion to 3D

                    points_3d.append([x, y, z])

        return points_3d

    def track_features(self, current_features):
        """Track features across frames"""
        # In a real implementation, this would use optical flow or feature tracking
        # to maintain correspondences between frames
        return current_features

    def estimate_motion(self, features):
        """Estimate camera/robot motion using features"""
        # This would use algorithms like PnP or iterative closest point
        # to estimate motion based on tracked features
        # For this example, we'll return a simple motion estimate
        return {
            'translation': [0.01, 0.00, 0.00],  # Small forward motion
            'rotation': [0.0, 0.0, 0.0, 1.0]   # No rotation
        }

    def update_map(self, points_3d):
        """Update the map with new 3D points"""
        for pt in points_3d:
            # Add point to map if it's not already there (simplified)
            self.feature_points.append(pt)

        # Keep only recent points to manage memory
        if len(self.feature_points) > 10000:
            self.feature_points = self.feature_points[-5000:]

    def update_robot_pose(self, motion_estimate):
        """Update robot pose based on motion estimate"""
        # Integrate motion to get new pose
        translation = motion_estimate['translation']
        rotation = motion_estimate['rotation']

        # Update pose (simplified integration)
        if self.robot_trajectory:
            last_pose = self.robot_trajectory[-1]
            new_pose = [
                last_pose[0] + translation[0],
                last_pose[1] + translation[1],
                last_pose[2] + translation[2]
            ]
        else:
            new_pose = [0.0, 0.0, 0.0]

        self.robot_trajectory.append(new_pose)

        # Publish pose estimate
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = new_pose[0]
        pose_msg.pose.position.y = new_pose[1]
        pose_msg.pose.position.z = new_pose[2]
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]

        self.pose_pub.publish(pose_msg)

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = pose_msg.header.stamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose = pose_msg.pose

        self.odom_pub.publish(odom_msg)

    def update_map_visualization(self):
        """Update map visualization markers"""
        marker_array = MarkerArray()

        # Create markers for feature points
        for i, point in enumerate(self.feature_points[-100:]):  # Show last 100 points
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.ns = 'vslam_map'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.05  # 5cm spheres
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.color.r = 0.0  # Blue points
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        # Create markers for robot trajectory
        for i, pose in enumerate(self.robot_trajectory[-50:]):  # Show last 50 poses
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.ns = 'robot_trajectory'
            marker.id = i + 1000  # Offset ID to avoid conflict
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = pose[2]
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1  # 10cm cubes
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.r = 1.0  # Red trajectory
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.map_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    slam_node = IsaacVisualSLAMIntegration()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        slam_node.get_logger().info('Shutting down Isaac Visual SLAM Integration...')
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Deep Learning Perception Pipeline

```python
#!/usr/bin/env python3
# dl_perception_pipeline.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage

class IsaacDLPerceptionPipeline(Node):
    """
    Deep learning-based perception pipeline using Isaac ROS
    """

    def __init__(self):
        super().__init__('isaac_dl_perception_pipeline')

        # Publishers
        self.detections_pub = self.create_publisher(Detection2DArray, '/humanoid/perception/detections', 10)
        self.classification_pub = self.create_publisher(String, '/humanoid/perception/classification', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/humanoid/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # CV Bridge
        self.cv_bridge = CvBridge()

        # Load pre-trained model (simulated - in real implementation would use Isaac ROS DNN nodes)
        self.model = self.load_perception_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Timers
        self.processing_timer = self.create_timer(0.1, self.process_detections)  # 10 Hz

        self.get_logger().info('Isaac Deep Learning Perception Pipeline initialized')

    def load_perception_model(self):
        """Load perception model (simulated for this example)"""
        # In a real Isaac ROS implementation, this would load a TensorRT-optimized model
        # or use Isaac ROS DNN inference nodes
        # For this example, we'll simulate the model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                # Simulate object detection output
                batch_size = x.shape[0]
                # Return mock detections: [batch, num_detections, 6] where 6 = [x, y, w, h, conf, class]
                detections = torch.tensor([[[100, 100, 50, 50, 0.9, 1],  # [x, y, width, height, confidence, class]
                                          [200, 150, 30, 30, 0.8, 2],
                                          [0, 0, 0, 0, 0, 0]]], dtype=torch.float32).repeat(batch_size, 1, 1)
                return detections

        return MockModel()

    def image_callback(self, msg):
        """Process incoming image for deep learning inference"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')

            # Store for processing
            self.current_image = cv_image
            self.current_image_header = msg.header

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_detections(self):
        """Process current image with deep learning model"""
        if not hasattr(self, 'current_image'):
            return

        try:
            # Convert OpenCV image to PIL for processing
            pil_image = PILImage.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))

            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                detections = self.model(input_tensor)

            # Process detections
            self.publish_detections(detections[0], self.current_image_header)

        except Exception as e:
            self.get_logger().error(f'Error in DL processing: {e}')

    def publish_detections(self, detections, header):
        """Publish detection results"""
        detection_array = Detection2DArray()
        detection_array.header = header

        # Process each detection
        for det in detections:
            x, y, w, h, conf, class_id = det

            # Skip invalid detections
            if w == 0 and h == 0:
                continue

            if float(conf) < 0.5:  # Confidence threshold
                continue

            detection = Detection2D()
            detection.header = header

            # Set bounding box
            detection.bbox.center.x = float(x + w/2)
            detection.bbox.center.y = float(y + h/2)
            detection.bbox.size_x = float(w)
            detection.bbox.size_y = float(h)

            # Set classification
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(float(class_id))
            hypothesis.score = float(conf)

            # Map class IDs to names (simplified)
            class_names = {1: "person", 2: "chair", 3: "table", 4: "robot"}
            class_name = class_names.get(int(float(class_id)), f"unknown_{int(float(class_id))}")

            # Create classification result
            classification_msg = String()
            classification_msg.data = f"Detected {class_name} with confidence {float(conf):.2f} at position ({float(x+w/2):.1f}, {float(y+h/2):.1f})"
            self.classification_pub.publish(classification_msg)

            detection.results.append(hypothesis)
            detection_array.detections.append(detection)

        self.detections_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    perception_node = IsaacDLPerceptionPipeline()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down Isaac DL Perception Pipeline...')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Perception pipelines form the sensory foundation of AI-powered humanoid robots. Isaac ROS provides GPU-accelerated perception capabilities that enable real-time processing of multiple sensor modalities. The key components include:

- **Multi-Sensor Integration**: Combining data from cameras, LiDAR, IMU, and other sensors
- **Deep Learning Inference**: Running neural networks for object detection, segmentation, and classification
- **State Estimation**: Using filtering and fusion algorithms to estimate robot state
- **Environment Mapping**: Creating representations of the robot's surroundings
- **Real-Time Performance**: Leveraging GPU acceleration for low-latency processing

## Exercises

### Conceptual
1. Explain the advantages of using Isaac ROS perception pipelines over traditional CPU-based computer vision approaches for humanoid robotics.

### Logical
1. Analyze the trade-offs between perception accuracy and computational performance in real-time robotics applications. When would you prioritize one over the other?

### Implementation
1. Implement a complete perception pipeline that integrates stereo vision, LiDAR, and IMU data for robust environment understanding in your humanoid robot simulation.
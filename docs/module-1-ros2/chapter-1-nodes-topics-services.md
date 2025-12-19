---
sidebar_position: 2
title: Chapter 1 - Nodes, Topics, Services, Actions
description: Understanding ROS 2 communication patterns for humanoid robotics
keywords: [ros2, nodes, topics, services, actions, communication]
---

# Chapter 1 - Nodes, Topics, Services, Actions

## Learning Objectives

By the end of this chapter, students will be able to:
1. Explain the fundamental ROS 2 communication patterns: nodes, topics, services, and actions
2. Implement basic ROS 2 nodes for humanoid robot components
3. Create publishers and subscribers for sensor and actuator data
4. Design services for synchronous robot operations
5. Compare the use cases for each communication pattern in humanoid robotics

## Prerequisites

Before starting this chapter, students should have:
- Basic Python programming knowledge
- Understanding of object-oriented programming concepts
- Familiarity with the ROS 2 installation and basic setup

## Core Concepts

### Nodes

A node is a process that performs computation. In ROS 2, nodes are the fundamental building blocks of a robotic application. Each node typically performs a specific function within the larger robot system.

In humanoid robotics, nodes might represent:
- Sensor drivers (IMU, cameras, LiDAR)
- Actuator controllers (joint position, velocity, effort)
- Perception systems (object detection, SLAM)
- Control algorithms (walking, balance, manipulation)
- High-level planning (path planning, task planning)

### Topics and Publish-Subscribe Communication

Topics enable asynchronous, many-to-many communication using a publish-subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics. This decouples the publisher from the subscriber, allowing for flexible system design.

In humanoid robotics, topics are commonly used for:
- Sensor data streams (IMU readings, camera images)
- Robot state information (joint positions, velocities)
- Control commands (desired joint positions)
- Perception results (detected objects, obstacles)

### Services and Request-Response Communication

Services enable synchronous, request-response communication between nodes. A client sends a request to a service, and the service sends back a response. This is useful for operations that require a definitive result.

In humanoid robotics, services are commonly used for:
- Robot activation/deactivation
- Parameter configuration
- Calibration procedures
- Task execution confirmation

### Actions

Actions provide a way to handle long-running tasks with feedback. They combine the features of services and topics, allowing for goal requests, feedback during execution, and final results.

In humanoid robotics, actions are commonly used for:
- Navigation to a specific location
- Manipulation tasks
- Walking or locomotion sequences
- Complex multi-step operations

## Implementation

### Creating a Basic ROS 2 Node

Let's create a simple ROS 2 node that represents a humanoid robot's head controller:

```python
# Example: Humanoid Head Controller Node
# WHAT: This code creates a ROS 2 node that controls a humanoid robot's head movements
# WHY: To demonstrate basic ROS 2 node structure and communication patterns

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

class HeadController(Node):
    def __init__(self):
        super().__init__('head_controller')

        # Publisher for head position commands
        self.head_cmd_publisher = self.create_publisher(
            Float32,
            'head_position_cmd',
            10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Timer to periodically send commands
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

        self.get_logger().info('Head Controller node initialized')

    def joint_state_callback(self, msg):
        # Process joint state messages
        head_position = None
        for name, position in zip(msg.name, msg.position):
            if name == 'head_joint':
                head_position = position
                break

        if head_position is not None:
            self.get_logger().info(f'Current head position: {head_position}')

    def timer_callback(self):
        # Send a head position command (example: move to 0.5 radians)
        cmd_msg = Float32()
        cmd_msg.data = 0.5
        self.head_cmd_publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    head_controller = HeadController()

    try:
        rclpy.spin(head_controller)
    except KeyboardInterrupt:
        pass
    finally:
        head_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `std_msgs`, `sensor_msgs`

### Creating a Publisher-Subscriber Example

Here's an example of a publisher that simulates IMU sensor data:

```python
# Example: IMU Sensor Publisher
# WHAT: This code creates a ROS 2 publisher node that simulates IMU sensor data
# WHY: To demonstrate the publish-subscribe communication pattern for sensor data

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math
import time

class ImuSensorPublisher(Node):
    def __init__(self):
        super().__init__('imu_sensor_publisher')
        self.publisher = self.create_publisher(Imu, 'imu_data', 10)

        # Timer to publish IMU data at 50Hz
        self.timer = self.create_timer(0.02, self.publish_imu_data)

        self.get_logger().info('IMU Sensor Publisher initialized')
        self.time_offset = time.time()

    def publish_imu_data(self):
        msg = Imu()

        # Set header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        # Simulate some realistic IMU data
        current_time = time.time() - self.time_offset
        msg.orientation.x = 0.0
        msg.orientation.y = 0.1 * math.sin(current_time * 0.5)
        msg.orientation.z = 0.0
        msg.orientation.w = math.sqrt(1 - msg.orientation.y**2)

        # Angular velocities (rad/s)
        msg.angular_velocity.x = 0.01 * math.cos(current_time)
        msg.angular_velocity.y = 0.02 * math.sin(current_time * 2)
        msg.angular_velocity.z = 0.005 * math.cos(current_time * 0.7)

        # Linear accelerations (m/s^2)
        msg.linear_acceleration.x = 0.1 * math.sin(current_time * 3)
        msg.linear_acceleration.y = 9.81 + 0.2 * math.cos(current_time * 2)  # gravity + movement
        msg.linear_acceleration.z = 0.05 * math.sin(current_time * 1.5)

        self.publisher.publish(msg)
        self.get_logger().debug('Published IMU data')

def main(args=None):
    rclpy.init(args=args)
    imu_publisher = ImuSensorPublisher()

    try:
        rclpy.spin(imu_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        imu_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`

And here's a corresponding subscriber:

```python
# Example: IMU Data Subscriber
# WHAT: This code creates a ROS 2 subscriber node that processes IMU sensor data
# WHY: To demonstrate the publish-subscribe communication pattern for sensor data processing

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class ImuDataProcessor(Node):
    def __init__(self):
        super().__init__('imu_data_processor')

        # Subscribe to IMU data
        self.subscription = self.create_subscription(
            Imu,
            'imu_data',
            self.imu_callback,
            10
        )

        self.get_logger().info('IMU Data Processor initialized')
        self.roll_history = []
        self.pitch_history = []

    def imu_callback(self, msg):
        # Extract orientation from quaternion
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # Convert quaternion to roll/pitch/yaw
        # Simplified conversion (assuming small yaw)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(2 * (w * y - z * x))

        # Store in history for averaging
        self.roll_history.append(roll)
        self.pitch_history.append(pitch)

        # Keep only last 10 values for averaging
        if len(self.roll_history) > 10:
            self.roll_history.pop(0)
            self.pitch_history.pop(0)

        # Calculate average
        avg_roll = sum(self.roll_history) / len(self.roll_history)
        avg_pitch = sum(self.pitch_history) / len(self.pitch_history)

        # Log processed data
        self.get_logger().info(
            f'Roll: {roll:.3f} rad, Pitch: {pitch:.3f} rad, '
            f'Avg Roll: {avg_roll:.3f} rad, Avg Pitch: {avg_pitch:.3f} rad'
        )

def main(args=None):
    rclpy.init(args=args)
    imu_processor = ImuDataProcessor()

    try:
        rclpy.spin(imu_processor)
    except KeyboardInterrupt:
        pass
    finally:
        imu_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import math  # Added import for math functions
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `sensor_msgs`, `numpy`

## Examples

### Example 1: Service for Robot Activation

Here's an example of a service that activates the humanoid robot:

```python
# Example: Robot Activation Service
# WHAT: This code creates a ROS 2 service that handles robot activation requests
# WHY: To demonstrate the request-response communication pattern for robot operations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from example_interfaces.srv import Trigger

class RobotActivationService(Node):
    def __init__(self):
        super().__init__('robot_activation_service')

        # Create service
        # WHAT: This creates a service server that listens for activation requests
        # WHY: Services are ideal for operations that need a definitive response, like activation
        self.srv = self.create_service(
            Trigger,
            'activate_robot',
            self.activate_robot_callback
        )

        self.is_active = False
        self.get_logger().info('Robot Activation Service initialized')

    def activate_robot_callback(self, request, response):
        # Process the activation request
        # WHAT: This callback handles incoming activation requests
        # WHY: The callback defines how the service responds to requests
        if not self.is_active:
            # Simulate activation process
            # WHAT: This simulates the steps needed to activate the robot
            # WHY: Real activation would involve initializing actuators and sensors
            self.get_logger().info('Activating robot...')

            # In a real implementation, this would activate actuators,
            # initialize sensors, etc.
            self.is_active = True

            # Set response values for successful activation
            # WHAT: This sets the response to indicate success
            # WHY: The service client needs to know if the operation was successful
            response.success = True
            response.message = 'Robot activated successfully'

            self.get_logger().info('Robot activated successfully')
        else:
            # Handle case where robot is already active
            # WHAT: This handles requests when the robot is already activated
            # WHY: Prevents unnecessary activation attempts
            response.success = False
            response.message = 'Robot is already active'

        return response

def main(args=None):
    """Main function to initialize and run the ROS 2 service node"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    activation_service = RobotActivationService()

    try:
        # Start spinning the node to process service requests
        # WHAT: This starts the ROS 2 event loop, processing incoming service requests
        # WHY: Without spinning, the node wouldn't execute its service callback functions
        rclpy.spin(activation_service)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        pass
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        activation_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `example_interfaces`

### Example 2: Action for Navigation

Here's an example of an action for humanoid navigation:

```python
# Example: Navigation Action
# WHAT: This code creates a ROS 2 action for humanoid navigation tasks
# WHY: To demonstrate long-running tasks with feedback for robot movement

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Point
from nav2_msgs.action import NavigateToPose
import time
import math

class HumanoidNavigationAction(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_action')

        # Create action server
        # WHAT: This creates an action server for navigation tasks
        # WHY: Actions are ideal for long-running operations that require feedback and cancellation
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info('Humanoid Navigation Action server initialized')

    def goal_callback(self, goal_request):
        # Handle incoming navigation goals
        # WHAT: This callback determines if a navigation goal should be accepted
        # WHY: Allows the server to reject goals that are not feasible
        self.get_logger().info('Received navigation goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Handle goal cancellation requests
        # WHAT: This callback determines if a goal cancellation should be accepted
        # WHY: Allows the server to control when goals can be cancelled
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the navigation goal"""
        # Log the start of navigation execution
        # WHAT: This logs that navigation has started
        # WHY: Provides feedback on the action's progress for debugging
        self.get_logger().info('Executing navigation goal...')

        # Get target position from the goal request
        # WHAT: This extracts the target coordinates from the goal
        # WHY: The navigation algorithm needs to know where to go
        target_x = goal_handle.request.pose.pose.position.x
        target_y = goal_handle.request.pose.position.y

        # Simulate navigation with feedback
        # WHAT: This creates feedback and result messages for the action
        # WHY: Actions require specific message types for feedback and results
        feedback_msg = NavigateToPose.Feedback()
        result = NavigateToPose.Result()

        # Simulate moving toward the goal
        # WHAT: This initializes the current position for the simulation
        # WHY: Navigation starts from the current position
        current_x, current_y = 0.0, 0.0  # Starting position
        step_size = 0.1  # Meters per feedback step

        # Navigation loop
        # WHAT: This is the main loop that moves the robot toward the goal
        # WHY: Continuous movement is needed until the goal is reached
        while goal_handle.is_cancel_requested is False:
            # Calculate distance to goal
            # WHAT: This calculates how far the robot is from the target
            # WHY: Needed to determine when the goal is reached
            dist_to_goal = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

            if dist_to_goal < 0.1:  # Close enough to goal
                # Goal reached successfully
                # WHAT: This indicates the navigation was successful
                # WHY: The result message needs to indicate success or failure
                result.result = True
                goal_handle.succeed()
                self.get_logger().info('Navigation completed successfully')
                return result

            # Move closer to goal
            # WHAT: This calculates the direction and moves the robot
            # WHY: The robot needs to continuously move toward the target
            direction_x = (target_x - current_x) / dist_to_goal
            direction_y = (target_y - current_y) / dist_to_goal

            current_x += direction_x * step_size
            current_y += direction_y * step_size

            # Update feedback
            # WHAT: This sends feedback to the action client about progress
            # WHY: Clients need to know the current status of the long-running action
            feedback_msg.current_pose.pose.position.x = current_x
            feedback_msg.current_pose.pose.position.y = current_y
            feedback_msg.distance_remaining = dist_to_goal

            goal_handle.publish_feedback(feedback_msg)

            # Sleep to simulate processing time
            # WHAT: This simulates the time it takes to move the robot
            # WHY: Real navigation would take actual time to execute
            time.sleep(0.5)

        # Check if goal was canceled
        # WHAT: This handles the case where the goal was canceled
        # WHY: The action needs to properly handle cancellation requests
        if goal_handle.is_cancel_requested:
            result.result = False
            goal_handle.canceled()
            self.get_logger().info('Navigation canceled')
            return result

def main(args=None):
    """Main function to initialize and run the ROS 2 action server"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)
    navigation_action = HumanoidNavigationAction()

    try:
        # Start spinning the node to process action requests
        # WHAT: This starts the ROS 2 event loop, processing incoming action requests
        # WHY: Without spinning, the node wouldn't execute its action callback functions
        rclpy.spin(navigation_action)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        pass
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        navigation_action.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `geometry_msgs`, `nav2_msgs`

## Summary

In this chapter, we've covered the fundamental communication patterns in ROS 2:
- **Nodes**: The basic computational units in ROS 2
- **Topics**: For asynchronous publish-subscribe communication
- **Services**: For synchronous request-response communication
- **Actions**: For long-running tasks with feedback

These patterns form the foundation of the ROS 2 communication system and are essential for building complex humanoid robot applications. Each pattern has specific use cases, and choosing the right one for each situation is crucial for effective robot design.

## Exercises

### Conceptual
1. Compare and contrast the four ROS 2 communication patterns (nodes, topics, services, actions). When would you use each one in a humanoid robot system?

### Logical
2. Design a communication architecture for a humanoid robot's walking controller. Which communication patterns would you use for each component (e.g., foot position tracking, balance control, step planning)?

### Implementation
3. Implement a ROS 2 node that subscribes to joint states and publishes a simplified representation of the robot's posture (e.g., standing, sitting, walking). Include proper error handling and logging.
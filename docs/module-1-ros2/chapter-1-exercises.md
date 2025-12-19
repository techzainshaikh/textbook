---
sidebar_position: 6
title: Chapter 1 - Exercises
description: Exercises for ROS 2 communication patterns in humanoid robotics
keywords: [ros2, exercises, communication, patterns, robotics]
---

# Chapter 1 - Exercises

## Conceptual Exercises

### Exercise 1: Communication Pattern Analysis
**Difficulty**: Beginner

Compare and contrast the four main ROS 2 communication patterns (nodes, topics, services, actions) in the context of humanoid robotics. For each pattern, identify at least two specific use cases in a humanoid robot system and explain why that pattern is appropriate for the use case.

**Solution**:
- **Topics**: Used for sensor data streams (IMU, camera feeds) and robot state publishing. Appropriate for asynchronous, many-to-many communication where publishers don't need responses.
- **Services**: Used for calibration requests and configuration changes. Appropriate for synchronous request-response operations requiring definitive results.
- **Actions**: Used for navigation and manipulation tasks. Appropriate for long-running operations requiring feedback and cancellation capabilities.

### Exercise 2: Node Design Principles
**Difficulty**: Intermediate

Design a node architecture for a humanoid robot's walking controller. Identify the different nodes needed, their responsibilities, and how they communicate with each other. Discuss potential issues with tight coupling and how to avoid them.

**Solution**:
A walking controller architecture might include:
- Gait generator node (publishes step targets)
- Balance controller node (publishes corrective commands)
- Joint trajectory controller node (executes commands)
- State estimator node (publishes current state)

Communication should be via topics for real-time data and services for configuration changes.

## Logical Exercises

### Exercise 3: Topic Message Design
**Difficulty**: Intermediate

Design appropriate message types for a humanoid robot's walking control system. Consider the information needed for step planning, balance feedback, and coordination between different control modules. Justify your design choices.

**Solution**:
For step planning: geometry_msgs/PoseStamped for target foot positions
For balance feedback: sensor_msgs/Imu for orientation data and geometry_msgs/Point for center of mass
For coordination: custom message with step timing, support foot, and gait phase information

### Exercise 4: Service Interface Design
**Difficulty**: Advanced

Design a service interface for a humanoid robot's emergency stop system. The service should handle different types of emergencies (balance loss, joint overheating, collision detection) and provide appropriate responses. Consider safety implications and recovery procedures.

**Solution**:
Request: emergency type, severity level, affected components
Response: action taken, recovery status, estimated time to resume
Includes provisions for different emergency types and appropriate safety responses.

## Implementation Exercises

### Exercise 5: Basic Publisher-Subscriber Pair
**Difficulty**: Beginner

Implement a publisher-subscriber pair where the publisher sends joint position commands at 10Hz and the subscriber logs the received commands. Both nodes should handle graceful shutdown when interrupted. Include proper error handling and logging.

```python
# Publisher Implementation
# WHAT: This code creates a publisher that sends joint position commands
# WHY: To demonstrate basic topic communication in ROS 2

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')

        # Create publisher for joint commands
        # WHAT: This creates a publisher that sends joint position commands
        # WHY: The robot needs to receive desired joint positions to execute movements
        self.publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)

        # Timer to send commands at 10Hz
        # WHAT: This timer callback executes every 0.1 seconds to send new commands
        # WHY: Consistent timing is important for smooth robot motion control
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Counter for generating different command values
        self.counter = 0

        self.get_logger().info('Joint Command Publisher node initialized')

    def timer_callback(self):
        """Publish joint position commands at regular intervals"""
        # Create message to hold joint commands
        # WHAT: This creates a Float64MultiArray message containing joint position commands
        # WHY: The Float64MultiArray is a flexible message type for sending multiple numerical values
        msg = Float64MultiArray()

        # Generate a pattern of joint positions using sine waves
        # WHAT: This generates smooth, oscillating joint position commands
        # WHY: Using sine waves creates natural-looking motion patterns for demonstration
        joint_positions = []
        for i in range(6):  # Example: 6 joints
            position = 0.5 * math.sin(self.counter * 0.1 + i * math.pi / 3)
            joint_positions.append(position)

        msg.data = joint_positions

        # Publish the joint command message
        # WHAT: This publishes the joint position commands to the 'joint_commands' topic
        # WHY: Other nodes (like the robot controller) subscribe to this topic to receive commands
        self.publisher.publish(msg)

        # Log the published values for debugging
        self.get_logger().info(f'Published joint commands: {msg.data}')

        # Increment counter for next iteration
        self.counter += 1

def main(args=None):
    """Main function to initialize and run the ROS 2 publisher node"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)

    # Create an instance of the JointCommandPublisher node
    # WHAT: This creates the publisher node instance
    # WHY: The node contains all the logic for publishing joint commands
    joint_publisher = JointCommandPublisher()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing timer callbacks and other events
        # WHY: Without spinning, the node wouldn't execute its timer callback to publish commands
        rclpy.spin(joint_publisher)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the publisher when needed
        joint_publisher.get_logger().info('Shutting down joint command publisher')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        joint_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `std_msgs`

### Exercise 6: Service Implementation
**Difficulty**: Intermediate

Implement a service that calibrates a humanoid robot's joint encoders. The service should simulate the calibration process, return success/failure status, and include proper error handling for different failure modes.

```python
# Service Implementation
# WHAT: This code creates a service that calibrates humanoid robot joint encoders
# WHY: To demonstrate service implementation with proper error handling and status reporting

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from example_interfaces.srv import Trigger
import time
import random

class JointCalibrationService(Node):
    def __init__(self):
        super().__init__('joint_calibration_service')

        # Create service for joint calibration
        # WHAT: This creates a service that handles joint calibration requests
        # WHY: Services are ideal for operations that require a definitive response
        self.srv = self.create_service(
            Trigger,
            'calibrate_joints',
            self.calibrate_joints_callback
        )

        # Store calibration status for each joint
        self.calibration_status = {}
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'
        ]

        # Initialize all joints as uncalibrated
        for joint_name in self.joint_names:
            self.calibration_status[joint_name] = False

        self.get_logger().info('Joint Calibration Service initialized')

    def calibrate_joints_callback(self, request, response):
        """Handle joint calibration requests"""
        # Log the calibration request
        # WHAT: This logs that a calibration request has been received
        # WHY: Logging helps with debugging and monitoring system behavior
        self.get_logger().info('Received joint calibration request')

        # Simulate the calibration process
        # WHAT: This simulates the actual calibration procedure
        # WHY: Real calibration would involve moving joints to known positions and setting encoders
        try:
            # Simulate calibration time
            time.sleep(2.0)  # Simulate actual calibration time

            # Simulate possible calibration failures
            # WHAT: This simulates potential failure conditions during calibration
            # WHY: Real systems must handle various failure modes gracefully
            if random.random() < 0.1:  # 10% chance of failure
                response.success = False
                response.message = 'Calibration failed due to encoder error'
                self.get_logger().error(response.message)
                return response

            # Mark all joints as calibrated
            # WHAT: This updates the calibration status for all joints
            # WHY: Successful calibration means joints are properly calibrated
            for joint_name in self.joint_names:
                self.calibration_status[joint_name] = True

            response.success = True
            response.message = f'Successfully calibrated {len(self.joint_names)} joints'
            self.get_logger().info(response.message)

        except Exception as e:
            # Handle any errors during calibration
            # WHAT: This catches and handles any exceptions during the calibration process
            # WHY: Error handling ensures the service doesn't crash and provides useful feedback
            response.success = False
            response.message = f'Calibration failed: {str(e)}'
            self.get_logger().error(response.message)

        return response

def main(args=None):
    """Main function to initialize and run the ROS 2 service node"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)

    # Create an instance of the JointCalibrationService node
    # WHAT: This creates the service node instance
    # WHY: The node contains all the logic for handling calibration requests
    calibration_service = JointCalibrationService()

    try:
        # Start spinning the node to process service requests
        # WHAT: This starts the ROS 2 event loop, processing incoming service requests
        # WHY: Without spinning, the node wouldn't execute its service callback functions
        rclpy.spin(calibration_service)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the service when needed
        calibration_service.get_logger().info('Shutting down joint calibration service')
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        calibration_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Dependencies**: `rclpy` version Kilted Kaiju (2025), `example_interfaces`

### Exercise 7: Action Implementation
**Difficulty**: Advanced

Implement an action for humanoid robot navigation that includes feedback during execution, result reporting, and cancellation capability. The action should simulate the navigation process with realistic timing and potential interruptions.

```python
# Action Implementation
# WHAT: This code creates an action for humanoid robot navigation tasks
# WHY: To demonstrate long-running tasks with feedback, results, and cancellation

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from nav2_msgs.action import NavigateToPose
import time
import math

class HumanoidNavigationAction(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_action')

        # Create action server for navigation
        # WHAT: This creates an action server for navigation tasks
        # WHY: Actions are ideal for long-running operations requiring feedback and cancellation
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Store current robot pose
        self.current_pose = Pose()
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0
        self.current_pose.position.z = 0.0
        self.current_pose.orientation.w = 1.0  # No rotation initially

        self.get_logger().info('Humanoid Navigation Action server initialized')

    def goal_callback(self, goal_request):
        """Accept all goals for simplicity"""
        # WHAT: This callback determines if a navigation goal should be accepted
        # WHY: Allows the server to reject goals that are not feasible
        self.get_logger().info('Received navigation goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept all cancellation requests"""
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
        target_y = goal_handle.request.pose.pose.position.y

        # Simulate navigation with feedback
        # WHAT: This creates feedback and result messages for the action
        # WHY: Actions require specific message types for feedback and results
        feedback_msg = NavigateToPose.Feedback()
        result = NavigateToPose.Result()

        # Simulate moving toward the goal
        # WHAT: This initializes the current position for the simulation
        # WHY: Navigation starts from the current position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        step_size = 0.1  # Meters per feedback step
        total_distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

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

                # Update current pose to target
                self.current_pose.position.x = target_x
                self.current_pose.position.y = target_y

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
        # WHAT: This handles the case when the user interrupts the program
        # WHY: Provides a clean way to stop the action server when needed
        navigation_action.get_logger().info('Shutting down navigation action server')
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

These exercises cover the fundamental ROS 2 communication patterns with increasing complexity from basic publisher/subscriber pairs to advanced action servers. Each exercise includes both conceptual understanding and practical implementation, reinforcing the core concepts of ROS 2 communication in humanoid robotics.
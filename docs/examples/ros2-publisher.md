# ROS 2 Publisher Example

## Complete ROS 2 Publisher Node with WHAT/WHY Comments

```python
# Example: ROS 2 Publisher Node for Humanoid Robot Joint Commands
# WHAT: This code creates a ROS 2 publisher node that publishes joint position commands
# WHY: To demonstrate the basic publish-subscribe communication pattern in ROS 2, which is fundamental for humanoid robot control systems

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import math
import time

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')

        # Create publisher for joint commands
        # WHAT: This creates a publisher that sends joint position commands to the robot's controller
        # WHY: The robot needs to receive desired joint positions to execute movements
        self.publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)

        # Timer to periodically send commands
        # WHAT: This timer callback executes every 0.1 seconds to send new commands
        # WHY: Continuous command updates are necessary for smooth robot motion control
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
        pass
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        joint_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Dependencies
- `rclpy` version Kilted Kaiju (2025)
- `std_msgs` - Standard ROS 2 message types
- `sensor_msgs` - Sensor-related message types

## How to Run
```bash
# Make sure your ROS 2 environment is sourced
source /opt/ros/kilted/setup.bash

# Run the publisher node
python3 joint_command_publisher.py
```

## Expected Output
The publisher will continuously output joint position commands to the console every 0.1 seconds, with values oscillating in a sine wave pattern.
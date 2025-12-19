# ROS 2 Subscriber Example

## Complete ROS 2 Subscriber Node with WHAT/WHY Comments

```python
# Example: ROS 2 Subscriber Node for Humanoid Robot Joint States
# WHAT: This code creates a ROS 2 subscriber node that receives joint state messages
# WHY: To demonstrate the basic publish-subscribe communication pattern in ROS 2, which is fundamental for humanoid robot sensor feedback systems

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        # Create subscriber for joint states
        # WHAT: This creates a subscriber that listens to joint state messages
        # WHY: The robot needs to receive feedback about current joint positions, velocities, and efforts
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10  # QoS history depth
        )

        # Create publisher for processed joint data
        # WHAT: This creates a publisher for processed joint information
        # WHY: Processed data might be needed by other nodes in the system
        self.processed_publisher = self.create_publisher(Float64MultiArray, 'processed_joint_data', 10)

        # Store the most recent joint state
        self.last_joint_state = None

        self.get_logger().info('Joint State Subscriber node initialized')

    def joint_state_callback(self, msg: JointState):
        """Process incoming joint state messages"""
        # Store the received joint state
        # WHAT: This saves the latest joint state message for potential later use
        # WHY: Storing the state allows other parts of the node to access the most recent data
        self.last_joint_state = msg

        # Log the received joint names and positions
        # WHAT: This logs the names and positions of all received joints
        # WHY: Logging helps with debugging and monitoring the robot's state
        if len(msg.name) == len(msg.position):
            joint_info = []
            for name, pos in zip(msg.name, msg.position):
                joint_info.append(f"{name}: {pos:.3f}")
            self.get_logger().info(f'Joint positions: {", ".join(joint_info)}')

        # Process the joint data to extract useful information
        # WHAT: This calculates derived information from the raw joint data
        # WHY: Derived information like joint velocity or acceleration might be needed for control
        if len(msg.velocity) > 0:
            avg_velocity = sum(abs(v) for v in msg.velocity) / len(msg.velocity)
            self.get_logger().info(f'Average joint velocity: {avg_velocity:.3f}')

        # Publish processed data
        # WHAT: This publishes processed joint information to another topic
        # WHY: Other nodes might need processed information rather than raw sensor data
        if len(msg.position) > 0:
            processed_msg = Float64MultiArray()
            # Calculate some processed values (e.g., average position)
            avg_position = sum(msg.position) / len(msg.position)
            processed_msg.data = [avg_position, len(msg.position), avg_velocity]
            self.processed_publisher.publish(processed_msg)

def main(args=None):
    """Main function to initialize and run the ROS 2 subscriber node"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)

    # Create an instance of the JointStateSubscriber node
    # WHAT: This creates the subscriber node instance
    # WHY: The node contains all the logic for processing joint state messages
    joint_subscriber = JointStateSubscriber()

    try:
        # Start spinning the node to process callbacks
        # WHAT: This starts the ROS 2 event loop, processing incoming messages and other events
        # WHY: Without spinning, the node wouldn't execute its callback functions to process messages
        rclpy.spin(joint_subscriber)
    except KeyboardInterrupt:
        # Handle graceful shutdown when user interrupts the program
        pass
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        joint_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Dependencies
- `rclpy` version Kilted Kaiju (2025)
- `sensor_msgs` - Sensor-related message types
- `std_msgs` - Standard ROS 2 message types

## How to Run
```bash
# Make sure your ROS 2 environment is sourced
source /opt/ros/kilted/setup.bash

# Run the subscriber node
python3 joint_state_subscriber.py
```

## Expected Output
The subscriber will continuously log joint position and velocity information as it receives messages on the 'joint_states' topic. It will also publish processed joint data to the 'processed_joint_data' topic.
# ROS 2 Service Example

## Complete ROS 2 Service Node with WHAT/WHY Comments

```python
# Example: ROS 2 Service Node for Humanoid Robot Joint Calibration
# WHAT: This code creates a ROS 2 service that handles joint calibration requests for humanoid robots
# WHY: To demonstrate the request-response communication pattern in ROS 2, which is useful for calibration and configuration tasks

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from example_interfaces.srv import Trigger
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time

class JointCalibrationService(Node):
    def __init__(self):
        super().__init__('joint_calibration_service')

        # Create service for joint calibration
        # WHAT: This creates a service that responds to calibration requests
        # WHY: Calibration is a synchronous operation that requires a definitive result, making services ideal
        self.srv = self.create_service(
            Trigger,
            'calibrate_joints',
            self.calibrate_joints_callback
        )

        # Create publisher for calibration status
        # WHAT: This publishes calibration status updates
        # WHY: Other nodes need to know when calibration is in progress or completed
        self.calibration_status_publisher = self.create_publisher(Bool, 'calibration_status', 10)

        # Store current joint states
        self.current_joint_states = None

        # Create subscriber for joint states
        # WHAT: This subscribes to joint state messages to access current positions
        # WHY: Calibration requires knowledge of current joint positions
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Joint Calibration Service initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint state"""
        # Store the most recent joint state
        # WHAT: This saves the latest joint state message
        # WHY: Calibration process needs access to current joint positions
        self.current_joint_states = msg

    def calibrate_joints_callback(self, request, response):
        """Handle joint calibration requests"""
        # Log the calibration request
        # WHAT: This logs that a calibration request has been received
        # WHY: Logging helps with debugging and monitoring system behavior
        self.get_logger().info('Received joint calibration request')

        # Publish calibration status (in progress)
        # WHAT: This publishes a status message indicating calibration has started
        # WHY: Other nodes need to know the robot is busy with calibration
        status_msg = Bool()
        status_msg.data = True
        self.calibration_status_publisher.publish(status_msg)

        # Simulate the calibration process
        # WHAT: This simulates the actual calibration procedure
        # WHY: Real calibration would involve moving joints to known positions and setting encoders
        try:
            # Check if we have joint state data
            if self.current_joint_states is None:
                response.success = False
                response.message = 'No joint state data available for calibration'
                self.get_logger().error(response.message)
                return response

            # Simulate calibration steps
            # WHAT: This simulates the process of calibrating each joint
            # WHY: Each joint needs to be moved to a known reference position during calibration
            for i, joint_name in enumerate(self.current_joint_states.name):
                self.get_logger().info(f'Calibrating joint: {joint_name} ({i+1}/{len(self.current_joint_states.name)})')

                # Simulate time needed for each joint calibration
                time.sleep(0.2)  # Simulate actual calibration time

            # Simulate final calibration steps
            # WHAT: This simulates final verification steps after all joints are calibrated
            # WHY: Verification ensures the calibration was successful
            time.sleep(0.5)

            # Set response values for successful calibration
            # WHAT: This sets the response to indicate success
            # WHY: The service client needs to know if the operation was successful
            response.success = True
            response.message = f'Successfully calibrated {len(self.current_joint_states.name)} joints'
            self.get_logger().info(response.message)

        except Exception as e:
            # Handle any errors during calibration
            # WHAT: This catches and handles any exceptions during the calibration process
            # WHY: Error handling ensures the service doesn't crash and provides useful feedback
            response.success = False
            response.message = f'Calibration failed: {str(e)}'
            self.get_logger().error(response.message)

        finally:
            # Publish calibration status (completed)
            # WHAT: This publishes a status message indicating calibration has completed
            # WHY: Other nodes need to know when the robot is no longer busy with calibration
            status_msg = Bool()
            status_msg.data = False
            self.calibration_status_publisher.publish(status_msg)

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
        pass
    finally:
        # Clean up resources
        # WHAT: This destroys the node and shuts down the rclpy library
        # WHY: Proper cleanup is important to free resources and avoid potential issues
        calibration_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Service Client Example

Here's how to call the service from a client:

```python
# Example: ROS 2 Service Client for Joint Calibration
# WHAT: This code creates a client that calls the joint calibration service
# WHY: To demonstrate how to call a ROS 2 service from another node

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

class CalibrationClient(Node):
    def __init__(self):
        super().__init__('calibration_client')

        # Create a client for the calibration service
        # WHAT: This creates a service client to call the calibration service
        # WHY: The client needs to be able to request calibration from the service
        self.cli = self.create_client(Trigger, 'calibrate_joints')

        # Wait for the service to be available
        # WHAT: This waits until the service is ready to accept requests
        # WHY: Trying to call a service before it's available would result in an error
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Calibration service not available, waiting again...')

        self.req = Trigger.Request()

    def send_request(self):
        """Send a calibration request to the service"""
        # Call the service asynchronously
        # WHAT: This sends an asynchronous request to the calibration service
        # WHY: Asynchronous calls don't block the node while waiting for a response
        self.future = self.cli.call_async(self.req)
        return self.future

def main(args=None):
    """Main function to run the service client"""
    # Initialize the ROS 2 client library
    # WHAT: This initializes the rclpy library and prepares it for node creation
    # WHY: All ROS 2 Python programs must initialize the client library before creating nodes
    rclpy.init(args=args)

    # Create the client node
    calibration_client = CalibrationClient()

    # Send the calibration request
    # WHAT: This sends a request to the calibration service
    # WHY: Initiates the calibration process
    future = calibration_client.send_request()

    try:
        # Wait for the response
        # WHAT: This waits for the service to complete and return a response
        # WHY: The client needs to know the result of the calibration operation
        rclpy.spin_until_future_complete(calibration_client, future)

        # Process the response
        # WHAT: This processes the response from the calibration service
        # WHY: The client needs to know if the calibration was successful
        if future.result() is not None:
            response = future.result()
            if response.success:
                calibration_client.get_logger().info(f'Calibration successful: {response.message}')
            else:
                calibration_client.get_logger().error(f'Calibration failed: {response.message}')
        else:
            calibration_client.get_logger().error('Failed to get calibration response')

    except KeyboardInterrupt:
        calibration_client.get_logger().info('Calibration client interrupted')
    finally:
        calibration_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Dependencies
- `rclpy` version Kilted Kaiju (2025)
- `example_interfaces` - Standard ROS 2 service interfaces
- `sensor_msgs` - Sensor-related message types
- `std_msgs` - Standard ROS 2 message types

## How to Run
```bash
# Terminal 1: Run the service
source /opt/ros/kilted/setup.bash
python3 joint_calibration_service.py

# Terminal 2: Run the client to call the service
source /opt/ros/kilted/setup.bash
python3 calibration_client.py
```

## Expected Output
The service will log when it receives calibration requests and simulate the calibration process. The client will log whether the calibration was successful or not.
# URDF Example

## Complete Humanoid Robot URDF Model with WHAT/WHY Comments

```xml
<?xml version="1.0"?>
<!-- Example: Complete Humanoid Robot URDF -->
<!-- WHAT: This URDF defines a complete humanoid robot with legs, arms, and head -->
<!-- WHY: To demonstrate a realistic humanoid model with proper kinematic chains -->

<robot name="tutorial_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Gazebo plugin configuration -->
  <!-- WHAT: This configures the gazebo_ros_control plugin for simulation -->
  <!-- WHY: The plugin enables ROS 2 to communicate with Gazebo for physics simulation -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Base link (pelvis) - the main body of the robot -->
  <!-- WHAT: This defines the base link of the robot, which serves as the root of the kinematic tree -->
  <!-- WHY: All other links are connected relative to this base link -->
  <link name="base_link">
    <inertial>
      <!-- Mass and inertia properties for physics simulation -->
      <!-- WHAT: These properties define the physical characteristics of the link for physics simulation -->
      <!-- WHY: Accurate mass and inertia properties are essential for realistic physics simulation -->
      <mass value="15.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
    <visual>
      <!-- Visual properties for rendering in simulation/visualization tools -->
      <!-- WHAT: These properties define how the link appears visually -->
      <!-- WHY: Visual representation is important for simulation visualization and debugging -->
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.2"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <!-- Collision properties for physics simulation -->
      <!-- WHAT: These properties define the collision boundaries for physics simulation -->
      <!-- WHY: Collision properties are needed for realistic interaction with the environment -->
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso link -->
  <!-- WHAT: This defines the torso link connected to the base -->
  <!-- WHY: The torso connects the legs to the arms and head -->
  <link name="torso">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.6"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting base to torso -->
  <!-- WHAT: This defines a fixed joint connecting the base to the torso -->
  <!-- WHY: Fixed joints are used when two links should not move relative to each other -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Head link -->
  <!-- WHAT: This defines the head link for the robot -->
  <!-- WHY: The head contains sensors and allows for gaze direction -->
  <link name="head">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting torso to head -->
  <!-- WHAT: This defines a revolute joint for head movement -->
  <!-- WHY: Revolute joints allow rotation around a single axis, useful for head movement -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Y-axis rotation for head nodding -->
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2.0"/>
  </joint>

  <!-- Left Arm Chain -->
  <!-- WHAT: This defines the complete kinematic chain for the left arm -->
  <!-- WHY: The arm chain allows for manipulation tasks -->
  <link name="left_shoulder">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_yaw" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.05 0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>  <!-- Z-axis rotation for shoulder yaw -->
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for shoulder pitch -->
    <limit lower="-2.0" upper="1.0" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for elbow -->
    <limit lower="-2.5" upper="0.0" effort="30" velocity="2.0"/>
  </joint>

  <link name="left_hand">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wrist" type="revolute">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Y-axis rotation for wrist -->
    <limit lower="-1.0" upper="1.0" effort="10" velocity="2.0"/>
  </joint>

  <!-- Right Arm Chain (mirror of left) -->
  <!-- WHAT: This defines the complete kinematic chain for the right arm -->
  <!-- WHY: Symmetry is important for humanoid robots to perform balanced tasks -->
  <link name="right_shoulder">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_yaw" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="0.05 -0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>  <!-- Z-axis rotation for shoulder yaw -->
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for shoulder pitch -->
    <limit lower="-1.0" upper="2.0" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.24"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for elbow -->
    <limit lower="0.0" upper="2.5" effort="30" velocity="2.0"/>
  </joint>

  <link name="right_hand">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wrist" type="revolute">
    <parent link="right_lower_arm"/>
    <child link="right_hand"/>
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Y-axis rotation for wrist -->
    <limit lower="-1.0" upper="1.0" effort="10" velocity="2.0"/>
  </joint>

  <!-- Left Leg Chain -->
  <!-- WHAT: This defines the complete kinematic chain for the left leg -->
  <!-- WHY: The leg chain enables locomotion and balance for the humanoid robot -->
  <link name="left_hip">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 -0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="0 0.08 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>  <!-- Z-axis rotation for hip yaw -->
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_hip_pitch" type="revolute">
    <parent link="left_hip"/>
    <child link="left_thigh"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for hip pitch -->
    <limit lower="-2.0" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.08"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for knee -->
    <limit lower="0.0" upper="2.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for ankle pitch -->
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

  <!-- Right Leg Chain (mirror of left) -->
  <!-- WHAT: This defines the complete kinematic chain for the right leg -->
  <!-- WHY: Symmetry is important for stable bipedal locomotion -->
  <link name="right_hip">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0.05 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip"/>
    <origin xyz="0 -0.08 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>  <!-- Z-axis rotation for hip yaw -->
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_hip_pitch" type="revolute">
    <parent link="right_hip"/>
    <child link="right_thigh"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for hip pitch -->
    <limit lower="-2.0" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.08"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.055" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for knee -->
    <limit lower="0.0" upper="2.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- X-axis rotation for ankle pitch -->
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

</robot>
```

## Xacro Example

Here's an example using Xacro for parameterized URDF:

```xml
<?xml version="1.0"?>
<!-- Example: Humanoid Robot with Xacro -->
<!-- WHAT: This URDF uses Xacro to parameterize robot dimensions and properties -->
<!-- WHY: To enable easy customization of robot models with different dimensions -->

<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="param_humanoid">

  <!-- Define robot parameters -->
  <!-- WHAT: These properties define configurable parameters for the robot -->
  <!-- WHY: Parameters allow for easy customization of robot dimensions and properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_mass" value="15.0" />
  <xacro:property name="torso_mass" value="10.0" />
  <xacro:property name="arm_mass" value="1.5" />
  <xacro:property name="leg_mass" value="3.0" />

  <!-- Define dimensions -->
  <!-- WHAT: These properties define the physical dimensions of robot components -->
  <!-- WHY: Dimensions can be easily modified to create robots of different sizes -->
  <xacro:property name="base_size_x" value="0.2" />
  <xacro:property name="base_size_y" value="0.25" />
  <xacro:property name="base_size_z" value="0.2" />

  <xacro:property name="torso_height" value="0.6" />
  <xacro:property name="torso_width" value="0.2" />
  <xacro:property name="torso_depth" value="0.2" />

  <!-- Macro for creating a simple box link -->
  <!-- WHAT: This macro defines a template for creating box-shaped links -->
  <!-- WHY: Macros reduce code duplication and make the URDF more maintainable -->
  <xacro:macro name="box_link" params="name mass x y z *inertial *visual *collision">
    <link name="${name}">
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <xacro:insert_block name="inertial"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <xacro:insert_block name="visual"/>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <xacro:insert_block name="collision"/>
      </collision>
    </link>
  </xacro:macro>

  <!-- Base link using the macro -->
  <!-- WHAT: This creates the base link using the box_link macro -->
  <!-- WHY: Using macros makes the URDF more concise and easier to maintain -->
  <xacro:box_link name="base_link" mass="${base_mass}">
    <inertial>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
    <visual>
      <geometry>
        <box size="${base_size_x} ${base_size_y} ${base_size_z}"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_size_x} ${base_size_y} ${base_size_z}"/>
      </geometry>
    </collision>
  </xacro:box_link>

  <!-- Torso using the macro -->
  <xacro:box_link name="torso" mass="${torso_mass}">
    <inertial>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${torso_depth} ${torso_width} ${torso_height}"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${torso_depth} ${torso_width} ${torso_height}"/>
      </geometry>
    </collision>
  </xacro:box_link>

  <!-- Joint connecting base to torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 ${base_size_z/2}" rpy="0 0 0"/>
  </joint>

  <!-- Macro for creating arm chain -->
  <!-- WHAT: This macro defines a template for creating complete arm kinematic chains -->
  <!-- WHY: Macros allow for consistent arm structure while enabling customization -->
  <xacro:macro name="arm_chain" params="side position_x position_y shoulder_yaw_limit shoulder_pitch_limit elbow_limit">
    <!-- Shoulder -->
    <link name="${side}_shoulder">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_shoulder_yaw" type="revolute">
      <parent link="torso"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${position_x} ${position_y} 0.3" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-shoulder_yaw_limit}" upper="${shoulder_yaw_limit}" effort="50" velocity="2.0"/>
    </joint>

    <!-- Upper arm -->
    <link name="${side}_upper_arm">
      <inertial>
        <mass value="${arm_mass}"/>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.3"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="0.3"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_shoulder_pitch" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-shoulder_pitch_limit}" upper="${shoulder_pitch_limit}" effort="50" velocity="2.0"/>
    </joint>

    <!-- Lower arm -->
    <link name="${side}_lower_arm">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 -0.12" rpy="0 0 0"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.035" length="0.24"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.035" length="0.24"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_elbow" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 -0.3" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-elbow_limit}" upper="0.0" effort="30" velocity="2.0"/>
    </joint>
  </xacro:macro>

  <!-- Create both arms using the macro -->
  <!-- WHAT: This instantiates both left and right arms using the arm_chain macro -->
  <!-- WHY: Using macros ensures consistency between left and right arms -->
  <xacro:arm_chain side="left" position_x="0.05" position_y="0.1" shoulder_yaw_limit="1.57" shoulder_pitch_limit="2.0" elbow_limit="2.5"/>
  <xacro:arm_chain side="right" position_x="0.05" position_y="-0.1" shoulder_yaw_limit="1.57" shoulder_pitch_limit="2.0" elbow_limit="2.5"/>

  <!-- Gazebo plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

</robot>
```

## Dependencies
- XML parser (built into ROS 2)
- URDF parser and tools for validation

## How to Use
```bash
# Validate the URDF file
check_urdf /path/to/your/robot.urdf

# Convert Xacro to URDF
ros2 run xacro xacro input_file.xacro > output_file.urdf

# Visualize in RViz
ros2 run rviz2 rviz2
# Then add a RobotModel display and set the robot description parameter
```

## Key Components Explained

1. **Links**: Represent rigid bodies with mass, inertia, and geometry
2. **Joints**: Define connections between links with specific movement types
3. **Inertial**: Physical properties for simulation (mass, center of mass, inertia)
4. **Visual**: How the link appears in visualization tools
5. **Collision**: How the link interacts in physics simulation
6. **Materials**: Color and appearance properties
7. **Gazebo plugins**: Configuration for physics simulation
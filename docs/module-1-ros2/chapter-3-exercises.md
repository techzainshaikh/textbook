---
sidebar_position: 8
title: Chapter 3 - Exercises
description: Exercises for URDF modeling in humanoid robotics
keywords: [urdf, modeling, exercises, humanoid, robotics, kinematics]
---

# Chapter 3 - Exercises

## Conceptual Exercises

### Exercise 1: URDF Design Principles
**Difficulty**: Intermediate

Explain the importance of proper mass and inertia properties in URDF models for humanoid robots. How do these properties affect physics simulation and robot control? What are the consequences of incorrect mass/inertia values?

**Solution**:
Mass and inertia properties are crucial for realistic physics simulation and control:
- Mass affects gravitational forces and acceleration responses
- Inertia affects rotational dynamics and stability
Incorrect values can lead to unrealistic simulation, unstable control, and potential safety issues.

### Exercise 2: Kinematic Chain Analysis
**Difficulty**: Advanced

Analyze the kinematic chain of a humanoid robot arm. Identify the degrees of freedom (DOF), joint types, and their ranges. Explain how the kinematic chain affects the workspace and dexterity of the arm. Consider the implications of different joint configurations.

**Solution**:
A typical humanoid arm has 7 DOF (shoulder: 3, elbow: 1, wrist: 2, hand: 1). The kinematic chain determines:
- Reachable workspace
- Singularity points
- Dexterity for manipulation tasks
- Redundancy for obstacle avoidance

## Logical Exercises

### Exercise 3: URDF Validation Logic
**Difficulty**: Intermediate

Design a validation system for URDF files that checks for common errors such as missing inertial properties, incorrect joint limits, and kinematic loops. Create a logical flow for the validation process and identify what should happen when different types of errors are detected.

**Solution**:
Validation flow:
1. Check XML syntax validity
2. Validate robot element and name
3. Check for required elements (inertial, visual, collision)
4. Verify joint connectivity and limits
5. Validate kinematic chain (no loops)
6. Report errors with specific locations

### Exercise 4: Collision Detection Planning
**Difficulty**: Advanced

Plan the collision geometry for a humanoid robot's foot link. Consider the trade-offs between accuracy and computational efficiency. How would you design the collision geometry to handle different terrain types and maintain balance during walking?

**Solution**:
Foot collision geometry considerations:
- Box or cylinder for main contact surface
- Rounded edges to avoid getting stuck on small obstacles
- Multiple collision primitives for complex shapes
- Balance between accuracy and simulation speed

## Implementation Exercises

### Exercise 5: Basic URDF Robot Model
**Difficulty**: Beginner

Create a basic URDF model for a simplified humanoid robot with a torso, head, 2 arms, and 2 legs. Each limb should have at least 3 joints. Include proper mass, visual, and collision properties. Validate the model using URDF tools.

```xml
<?xml version="1.0"?>
<!-- Basic URDF Robot Model -->
<!-- WHAT: This URDF defines a simplified humanoid robot with basic limbs and joints -->
<!-- WHY: To demonstrate fundamental URDF modeling concepts for humanoid robots -->

<robot name="basic_humanoid">

  <!-- Gazebo plugin configuration -->
  <!-- WHAT: This configures the gazebo_ros_control plugin for simulation -->
  <!-- WHY: The plugin enables ROS 2 to communicate with Gazebo for physics simulation -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/basic_humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Base link (pelvis) -->
  <!-- WHAT: This defines the base link of the robot, serving as the root of the kinematic tree -->
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
      <!-- Visual properties for rendering -->
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

### Exercise 6: Xacro Parameterization
**Difficulty**: Intermediate

Modify the basic robot model to use Xacro for parameterization. Add parameters for robot dimensions, masses, and joint limits. Create a macro for the arm chain that can be instantiated for both left and right arms with different positions.

```xml
<?xml version="1.0"?>
<!-- Xacro Parameterization Exercise -->
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

  <xacro:property name="arm_length" value="0.3" />
  <xacro:property name="leg_length" value="0.4" />

  <!-- Define joint limits -->
  <!-- WHAT: These properties define the motion limits for different joints -->
  <!-- WHY: Joint limits ensure safe operation and realistic movement -->
  <xacro:property name="shoulder_yaw_limit" value="1.57" />
  <xacro:property name="shoulder_pitch_limit" value="2.0" />
  <xacro:property name="elbow_limit" value="2.5" />

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

  <!-- Macro for creating cylindrical links -->
  <!-- WHAT: This macro defines a template for creating cylindrical links -->
  <!-- WHY: Cylindrical shapes are common in robot limbs -->
  <xacro:macro name="cylinder_link" params="name mass radius length *inertial *visual *collision">
    <link name="${name}">
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="0 0 ${length/2}" rpy="0 0 0"/>
        <xacro:insert_block name="inertial"/>
      </inertial>
      <visual>
        <origin xyz="0 0 ${length/2}" rpy="0 0 0"/>
        <xacro:insert_block name="visual"/>
      </visual>
      <collision>
        <origin xyz="0 0 ${length/2}" rpy="0 0 0"/>
        <xacro:insert_block name="collision"/>
      </collision>
    </link>
  </xacro:macro>

  <!-- Base link using the box macro -->
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

  <!-- Torso using the box macro -->
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
    <xacro:cylinder_link name="${side}_shoulder" mass="1.0" radius="0.05" length="0.1">
      <inertial>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
      </collision>
    </xacro:cylinder_link>

    <joint name="${side}_shoulder_yaw" type="revolute">
      <parent link="torso"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${position_x} ${position_y} 0.3" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-shoulder_yaw_limit}" upper="${shoulder_yaw_limit}" effort="50" velocity="2.0"/>
    </joint>

    <!-- Upper arm -->
    <xacro:cylinder_link name="${side}_upper_arm" mass="${arm_mass}" radius="0.04" length="${arm_length}">
      <inertial>
        <origin xyz="0 0 ${arm_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
      </inertial>
      <visual>
        <origin xyz="0 0 ${arm_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="${arm_length}"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 ${arm_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.04" length="${arm_length}"/>
        </geometry>
      </collision>
    </xacro:cylinder_link>

    <joint name="${side}_shoulder_pitch" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-shoulder_pitch_limit}" upper="${shoulder_pitch_limit}" effort="50" velocity="2.0"/>
    </joint>

    <!-- Lower arm -->
    <xacro:cylinder_link name="${side}_lower_arm" mass="1.0" radius="0.035" length="0.24">
      <inertial>
        <origin xyz="0 0 0.12" rpy="0 0 0"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.035" length="0.24"/>
        </geometry>
        <material name="blue">
          <color rgba="0.2 0.2 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0.12" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.035" length="0.24"/>
        </geometry>
      </collision>
    </xacro:cylinder_link>

    <joint name="${side}_elbow" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 ${arm_length}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-elbow_limit}" upper="0.0" effort="30" velocity="2.0"/>
    </joint>

    <!-- Hand -->
    <xacro:box_link name="${side}_hand" mass="0.5" x="0.1" y="0.08" z="0.1">
      <inertial>
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
    </xacro:box_link>

    <joint name="${side}_wrist" type="revolute">
      <parent link="${side}_lower_arm"/>
      <child link="${side}_hand"/>
      <origin xyz="0 0 0.24" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.0" upper="1.0" effort="10" velocity="2.0"/>
    </joint>
  </xacro:macro>

  <!-- Create both arms using the macro -->
  <!-- WHAT: This instantiates both left and right arms using the arm_chain macro -->
  <!-- WHY: Using macros ensures consistency between left and right arms -->
  <xacro:arm_chain side="left" position_x="0.05" position_y="0.1" shoulder_yaw_limit="${shoulder_yaw_limit}" shoulder_pitch_limit="${shoulder_pitch_limit}" elbow_limit="${elbow_limit}"/>
  <xacro:arm_chain side="right" position_x="0.05" position_y="-0.1" shoulder_yaw_limit="${shoulder_yaw_limit}" shoulder_pitch_limit="${shoulder_pitch_limit}" elbow_limit="${elbow_limit}"/>

  <!-- Macro for creating leg chain -->
  <!-- WHAT: This macro defines a template for creating complete leg kinematic chains -->
  <!-- WHY: Macros allow for consistent leg structure while enabling customization -->
  <xacro:macro name="leg_chain" params="side position_x position_y hip_yaw_limit hip_pitch_limit knee_limit ankle_limit">
    <!-- Hip -->
    <xacro:cylinder_link name="${side}_hip" mass="2.0" radius="0.06" length="0.1">
      <inertial>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.06" length="0.1"/>
        </geometry>
        <material name="green">
          <color rgba="0.2 0.8 0.2 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.06" length="0.1"/>
        </geometry>
      </collision>
    </xacro:cylinder_link>

    <joint name="${side}_hip_yaw" type="revolute">
      <parent link="base_link"/>
      <child link="${side}_hip"/>
      <origin xyz="${position_x} ${position_y} 0" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-hip_yaw_limit}" upper="${hip_yaw_limit}" effort="100" velocity="1.0"/>
    </joint>

    <!-- Thigh -->
    <xacro:cylinder_link name="${side}_thigh" mass="${leg_mass}" radius="0.06" length="${leg_length}">
      <inertial>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
      </inertial>
      <visual>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.06" length="${leg_length}"/>
        </geometry>
        <material name="green">
          <color rgba="0.2 0.8 0.2 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.06" length="${leg_length}"/>
        </geometry>
      </collision>
    </xacro:cylinder_link>

    <joint name="${side}_hip_pitch" type="revolute">
      <parent link="${side}_hip"/>
      <child link="${side}_thigh"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-hip_pitch_limit}" upper="${hip_pitch_limit}" effort="100" velocity="1.0"/>
    </joint>

    <!-- Shin -->
    <xacro:cylinder_link name="${side}_shin" mass="2.5" radius="0.055" length="${leg_length}">
      <inertial>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.08"/>
      </inertial>
      <visual>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.055" length="${leg_length}"/>
        </geometry>
        <material name="green">
          <color rgba="0.2 0.8 0.2 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.055" length="${leg_length}"/>
        </geometry>
      </collision>
    </xacro:cylinder_link>

    <joint name="${side}_knee" type="revolute">
      <parent link="${side}_thigh"/>
      <child link="${side}_shin"/>
      <origin xyz="0 0 ${leg_length}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="0.0" upper="${knee_limit}" effort="100" velocity="1.0"/>
    </joint>

    <!-- Foot -->
    <xacro:box_link name="${side}_foot" mass="1.5" x="0.2" y="0.1" z="0.1">
      <inertial>
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
    </xacro:box_link>

    <joint name="${side}_ankle" type="revolute">
      <parent link="${side}_shin"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 ${leg_length}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-ankle_limit}" upper="${ankle_limit}" effort="50" velocity="1.0"/>
    </joint>
  </xacro:macro>

  <!-- Create both legs using the macro -->
  <!-- WHAT: This instantiates both left and right legs using the leg_chain macro -->
  <!-- WHY: Using macros ensures consistency between left and right legs -->
  <xacro:leg_chain side="left" position_x="0" position_y="0.08" hip_yaw_limit="0.5" hip_pitch_limit="2.0" knee_limit="2.5" ankle_limit="0.5"/>
  <xacro:leg_chain side="right" position_x="0" position_y="-0.08" hip_yaw_limit="0.5" hip_pitch_limit="2.0" knee_limit="2.5" ankle_limit="0.5"/>

  <!-- Head -->
  <xacro:box_link name="head" mass="3.0" x="0.24" y="0.24" z="0.24">
    <inertial>
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
  </xacro:box_link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 ${torso_height}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2.0"/>
  </joint>

  <!-- Gazebo plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/param_humanoid</robotNamespace>
    </plugin>
  </gazebo>

</robot>
```

### Exercise 7: URDF Validation Script
**Difficulty**: Advanced

Create a Python script that validates URDF models for common errors such as missing inertial properties, incorrect joint limits, and kinematic loops. The script should provide detailed error messages and suggest fixes.

```python
# URDF Validation Script
# WHAT: This script validates URDF files for common errors and completeness
# WHY: To ensure URDF models are correct before using them in simulation or control

import xml.etree.ElementTree as ET
import os
from typing import Dict, List, Tuple, Optional

class URDFValidator:
    def __init__(self, urdf_path: str):
        # WHAT: Initialize the validator with the path to the URDF file
        # WHY: The validator needs to know which file to validate
        self.urdf_path = urdf_path
        self.tree = None
        self.root = None
        self.errors = []
        self.warnings = []

    def load_urdf(self) -> bool:
        """Load the URDF file and parse it"""
        # WHAT: This method loads and parses the URDF XML file
        # WHY: The validator needs to access the XML structure to perform validation
        try:
            self.tree = ET.parse(self.urdf_path)
            self.root = self.tree.getroot()
            return True
        except ET.ParseError as e:
            # WHAT: This handles XML parsing errors
            # WHY: Invalid XML syntax makes the URDF unusable
            self.errors.append(f"XML Parse Error: {str(e)}")
            return False
        except FileNotFoundError:
            # WHAT: This handles cases where the file doesn't exist
            # WHY: Missing files are a critical error that must be reported
            self.errors.append(f"File not found: {self.urdf_path}")
            return False

    def validate_robot_element(self) -> bool:
        """Validate the root robot element"""
        # WHAT: This method validates the root <robot> element and its name attribute
        # WHY: The root element must be correct for the URDF to be valid
        if self.root.tag != 'robot':
            # WHAT: Check that the root element is indeed named 'robot'
            # WHY: URDF files must have 'robot' as their root element
            self.errors.append("Root element must be 'robot'")
            return False

        robot_name = self.root.get('name')
        if not robot_name:
            # WHAT: Check that the robot element has a name attribute
            # WHY: Robot names are required for identification and namespace management
            self.errors.append("Robot element must have a 'name' attribute")
            return False

        if not robot_name.strip():
            # WHAT: Check that the robot name is not empty
            # WHY: Empty names are not meaningful and can cause issues
            self.errors.append("Robot name cannot be empty")
            return False

        return True

    def validate_links(self) -> bool:
        """Validate all link elements"""
        # WHAT: This method validates all link elements in the URDF
        # WHY: Links are fundamental components that must be correctly defined
        links = self.root.findall('link')
        if not links:
            # WHAT: Check that at least one link exists
            # WHY: A robot without links is not valid
            self.errors.append("No links found in URDF")
            return False

        link_names = set()
        for link in links:
            name = link.get('name')
            if not name:
                # WHAT: Check that each link has a name attribute
                # WHY: Link names are required for referencing in joints
                self.errors.append("Link element missing 'name' attribute")
                continue

            if name in link_names:
                # WHAT: Check for duplicate link names
                # WHY: Duplicate names cause ambiguity in the kinematic chain
                self.errors.append(f"Duplicate link name: {name}")
                continue

            link_names.add(name)

            # Check for required elements within link
            inertial = link.find('inertial')
            if inertial is None:
                # WHAT: Check if the link has an inertial element
                # WHY: While not always required, inertial properties are important for simulation
                self.warnings.append(f"Link '{name}' missing inertial element")
            else:
                self.validate_inertial(inertial, name)

            visual = link.find('visual')
            if visual is None:
                # WHAT: Check if the link has a visual element
                # WHY: Visual elements are important for visualization
                self.warnings.append(f"Link '{name}' missing visual element")

            collision = link.find('collision')
            if collision is None:
                # WHAT: Check if the link has a collision element
                # WHY: Collision elements are important for physics simulation
                self.warnings.append(f"Link '{name}' missing collision element")

        return len(self.errors) == 0

    def validate_inertial(self, inertial_element, link_name: str):
        """Validate inertial element"""
        # WHAT: This method validates the inertial properties of a link
        # WHY: Inertial properties are critical for physics simulation
        mass = inertial_element.find('mass')
        if mass is None or mass.get('value') is None:
            # WHAT: Check if the mass element and value are present
            # WHY: Mass is required for physics simulation
            self.warnings.append(f"Link '{link_name}' inertial missing mass value")
            return

        try:
            mass_value = float(mass.get('value'))
            if mass_value <= 0:
                # WHAT: Check if the mass is positive
                # WHY: Negative or zero mass is physically invalid
                self.errors.append(f"Link '{link_name}' has non-positive mass: {mass_value}")
        except ValueError:
            # WHAT: Handle invalid mass values
            # WHY: Non-numeric mass values are invalid
            self.errors.append(f"Link '{link_name}' has invalid mass value: {mass.get('value')}")

        # Check inertia values
        # WHAT: Validate the inertia matrix values
        # WHY: Inertia values must be numeric and physically plausible
        inertia = inertial_element.find('inertia')
        if inertia is not None:
            for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                value = inertia.get(attr)
                if value is not None:
                    try:
                        float(value)
                    except ValueError:
                        # WHAT: Handle invalid inertia values
                        # WHY: Non-numeric inertia values are invalid
                        self.errors.append(f"Link '{link_name}' has invalid {attr} value: {value}")

    def validate_joints(self) -> bool:
        """Validate all joint elements"""
        # WHAT: This method validates all joint elements in the URDF
        # WHY: Joints define the kinematic relationships between links
        joints = self.root.findall('joint')
        if not joints:
            # WHAT: Check if joints exist (warning, not error)
            # WHY: Some simple models might not have joints
            self.warnings.append("No joints found in URDF")
            return True  # Not necessarily an error

        link_names = {link.get('name') for link in self.root.findall('link')}
        joint_names = set()

        for joint in joints:
            name = joint.get('name')
            if not name:
                # WHAT: Check if each joint has a name
                # WHY: Joint names are required for identification
                self.errors.append("Joint element missing 'name' attribute")
                continue

            if name in joint_names:
                # WHAT: Check for duplicate joint names
                # WHY: Duplicate names cause ambiguity
                self.errors.append(f"Duplicate joint name: {name}")
                continue

            joint_names.add(name)

            joint_type = joint.get('type')
            if not joint_type:
                # WHAT: Check if each joint has a type
                # WHY: Joint type determines how the joint behaves
                self.errors.append(f"Joint '{name}' missing 'type' attribute")
                continue

            # Check for valid joint types
            # WHAT: Validate that the joint type is one of the supported types
            # WHY: Only certain joint types are supported by URDF
            valid_types = ['revolute', 'continuous', 'prismatic', 'fixed', 'floating', 'planar']
            if joint_type not in valid_types:
                self.errors.append(f"Joint '{name}' has invalid type: {joint_type}")

            # Check parent and child links exist
            # WHAT: Verify that referenced links exist in the model
            # WHY: Joints connect existing links; non-existent links are invalid
            parent = joint.find('parent')
            child = joint.find('child')

            if parent is None or parent.get('link') is None:
                self.errors.append(f"Joint '{name}' missing parent link")
            else:
                parent_link = parent.get('link')
                if parent_link not in link_names:
                    self.errors.append(f"Joint '{name}' references non-existent parent link: {parent_link}")

            if child is None or child.get('link') is None:
                self.errors.append(f"Joint '{name}' missing child link")
            else:
                child_link = child.get('link')
                if child_link not in link_names:
                    self.errors.append(f"Joint '{name}' references non-existent child link: {child_link}")

            # Check for continuous joints (they shouldn't have limits)
            # WHAT: Validate that continuous joints don't have limits
            # WHY: Continuous joints rotate infinitely and shouldn't have limits
            if joint_type == 'continuous':
                limit = joint.find('limit')
                if limit is not None:
                    self.warnings.append(f"Joint '{name}' is continuous but has limits defined")

            # Check for revolute/prismatic joints (they should have limits)
            # WHAT: Validate that revolute and prismatic joints have limits
            # WHY: These joints need limits to constrain their motion
            elif joint_type in ['revolute', 'prismatic']:
                limit = joint.find('limit')
                if limit is None:
                    self.warnings.append(f"Joint '{name}' is {joint_type} but has no limits defined")

        return len(self.errors) == 0

    def validate_geometry(self):
        """Validate geometry elements in visual and collision"""
        # WHAT: This method validates the geometry elements in visual and collision sections
        # WHY: Geometry elements must be properly defined for visualization and physics
        for link in self.root.findall('link'):
            for element_type in ['visual', 'collision']:
                for element in link.findall(element_type):
                    geometry = element.find('geometry')
                    if geometry is not None:
                        # Check for exactly one geometry type
                        # WHAT: Ensure only one geometry type is defined per element
                        # WHY: Multiple geometry types in one element are invalid
                        geometry_types = []
                        for geom_type in ['box', 'cylinder', 'sphere', 'mesh']:
                            if geometry.find(geom_type) is not None:
                                geometry_types.append(geom_type)

                        if len(geometry_types) != 1:
                            self.errors.append(
                                f"Link '{link.get('name')}' {element_type} has {len(geometry_types)} geometry types, expected exactly 1"
                            )
                        else:
                            geom_element = geometry.find(geometry_types[0])
                            if geometry_types[0] == 'mesh':
                                filename = geom_element.get('filename')
                                if filename and not os.path.isabs(filename):
                                    # Check if mesh file exists relative to URDF location
                                    # WHAT: Validate that referenced mesh files exist
                                    # WHY: Missing mesh files cause visualization and physics errors
                                    urdf_dir = os.path.dirname(self.urdf_path)
                                    mesh_path = os.path.join(urdf_dir, filename)
                                    if not os.path.exists(mesh_path):
                                        self.warnings.append(
                                            f"Mesh file not found: {mesh_path} for link '{link.get('name')}'"
                                        )

    def validate_kinematic_chain(self) -> bool:
        """Validate that the kinematic chain is valid (no loops, proper tree structure)"""
        # WHAT: This method validates the kinematic chain structure
        # WHY: URDF models must form a tree structure (no loops)
        joints = self.root.findall('joint')
        links = self.root.findall('link')

        # Create a graph of parent-child relationships
        # WHAT: Build a mapping of parent-child relationships
        # WHY: Needed to analyze the kinematic structure
        parent_to_child = {}
        child_to_parent = {}

        for joint in joints:
            parent_elem = joint.find('parent')
            child_elem = joint.find('child')

            if parent_elem is not None and child_elem is not None:
                parent_name = parent_elem.get('link')
                child_name = child_elem.get('link')

                if parent_name and child_name:
                    if child_name in child_to_parent:
                        # WHAT: Check for multiple parents (invalid in tree structure)
                        # WHY: Each link should have at most one parent in a tree
                        self.errors.append(f"Link '{child_name}' has multiple parents")
                        return False

                    child_to_parent[child_name] = parent_name
                    if parent_name not in parent_to_child:
                        parent_to_child[parent_name] = []
                    parent_to_child[parent_name].append(child_name)

        # Find the root link (one without a parent)
        # WHAT: Identify the root of the kinematic tree
        # WHY: A valid URDF tree must have exactly one root
        all_link_names = {link.get('name') for link in links}
        root_links = all_link_names - set(child_to_parent.keys())

        if len(root_links) == 0:
            self.errors.append("No root link found - all links have parents (possible loop)")
            return False
        elif len(root_links) > 1:
            self.errors.append(f"Multiple root links found: {root_links}")
            return False

        # Check for loops by traversing the tree
        # WHAT: Detect cycles in the kinematic chain
        # WHY: Loops create invalid kinematic structures
        visited = set()
        def traverse(link_name):
            if link_name in visited:
                self.errors.append(f"Loop detected in kinematic chain at link: {link_name}")
                return False
            visited.add(link_name)

            if link_name in parent_to_child:
                for child in parent_to_child[link_name]:
                    if not traverse(child):
                        return False
            return True

        root = list(root_links)[0]
        if not traverse(root):
            return False

        return True

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validations and return results"""
        # WHAT: This method runs all validation checks
        # WHY: Consolidates all validation into a single method
        self.errors = []
        self.warnings = []

        if not self.load_urdf():
            return False, self.errors, self.warnings

        # Run all validation checks
        robot_ok = self.validate_robot_element()
        links_ok = self.validate_links()
        joints_ok = self.validate_joints()
        self.validate_geometry()
        chain_ok = self.validate_kinematic_chain()

        success = robot_ok and links_ok and joints_ok and chain_ok
        return success, self.errors, self.warnings

def main(urdf_path: str = None):
    """Main function to validate a URDF file"""
    # WHAT: This is the main function that validates a URDF file
    # WHY: Provides an entry point for using the validator
    if urdf_path is None:
        # Use the example URDF file if none provided
        # WHAT: Use a default URDF file if no path is provided
        # WHY: Allows the script to run with a default example
        urdf_path = "basic_humanoid.urdf"  # This would be the path to your URDF file

    validator = URDFValidator(urdf_path)
    success, errors, warnings = validator.validate()

    print(f"URDF Validation Results for: {urdf_path}")
    print(f"Valid: {success}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    if success and not warnings:
        print("\nURDF is valid with no warnings!")
    elif success:
        print(f"\nURDF is valid but has {len(warnings)} warnings.")

if __name__ == '__main__':
    # Example usage - you would call this with the path to your URDF file
    # main("path/to/your/robot.urdf")
    print("URDFValidator class defined. Use main(urdf_path) to validate a URDF file.")
```

## Summary

These exercises cover the fundamental concepts of URDF modeling for humanoid robots, from basic model creation to advanced parameterization with Xacro and validation scripting. Each exercise reinforces the key concepts of link definition, joint constraints, and proper physical properties for realistic simulation.
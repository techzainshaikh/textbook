---
title: Unity Visualization for Human-Robot Interaction
sidebar_position: 5
description: Creating immersive visualization environments for robotics using Unity
keywords: [unity, visualization, human-robot interaction, 3d graphics, robotics simulation]
---

# Chapter 4: Unity Visualization for Human-Robot Interaction

## Learning Objectives

By the end of this chapter, students will be able to:
- Set up Unity projects for robotics visualization and human-robot interaction prototyping
- Implement realistic 3D rendering of humanoid robots with proper physics properties
- Create interactive interfaces for commanding and monitoring robot behavior
- Integrate Unity visualization with ROS 2 for real-time robot state display
- Design user experiences that facilitate effective human-robot collaboration

## Prerequisites

Students should have:
- Basic understanding of 3D graphics concepts (vertices, meshes, materials, lighting)
- Familiarity with C# programming (Unity's primary scripting language)
- Understanding of human-robot interaction principles
- Completion of previous chapters on physics simulation and sensor modeling

## Core Concepts

Unity provides high-fidelity visualization capabilities that complement Gazebo's physics simulation. While Gazebo excels at accurate physics modeling, Unity excels at photorealistic rendering and human-robot interaction design.

### Key Visualization Components

**Robot Visualization:**
- Accurate 3D models matching physical robot specifications
- Proper joint constraints and kinematic chains
- Realistic materials and textures
- Animation systems for smooth motion representation

**Scene Design:**
- Environment modeling with accurate lighting and materials
- Occlusion and collision detection for realistic interaction
- Level of detail (LOD) systems for performance optimization
- Asset optimization for real-time rendering

**Human-Robot Interface:**
- Command interfaces for robot control
- Status displays for robot state visualization
- Interactive elements for teleoperation
- Augmented reality overlays for enhanced situational awareness

### Unity-ROS Integration

Unity can connect to ROS 2 systems through several methods:
- **Unity ROS TCP Connector**: Direct TCP/IP communication
- **ROS#**: C# implementation of ROS protocols
- **WebSocket bridges**: For web-based deployments
- **Custom middleware**: For specialized applications

## Implementation

Let's implement a Unity visualization system for our humanoid robot. We'll create a basic scene structure with robot visualization and ROS 2 integration.

### Unity Project Setup

First, let's outline the basic Unity project structure for robotics visualization:

```
UnityHumanoidVisualization/
├── Assets/
│   ├── Scripts/
│   │   ├── RobotController.cs          # Handles robot state updates
│   │   ├── ROSConnectionManager.cs      # Manages ROS communication
│   │   ├── JointVisualizer.cs          # Updates joint positions
│   │   └── HRIInterface.cs             # Human-robot interaction elements
│   ├── Models/
│   │   ├── HumanoidRobot/
│   │   │   ├── Robot.prefab            # Robot prefab with joints
│   │   │   ├── Materials/              # Robot materials
│   │   │   └── Meshes/                 # Individual robot parts
│   │   └── Environments/
│   │       ├── Indoor/
│   │       └── Outdoor/
│   ├── Scenes/
│   │   └── MainScene.unity
│   ├── Materials/
│   │   ├── RobotMaterials/
│   │   └── EnvironmentMaterials/
│   ├── Plugins/
│   │   └── ROSBridgeLib/               # ROS communication library
│   └── Prefabs/
│       ├── Robot.prefab
│       └── SensorVisualizers/
├── ProjectSettings/
└── Packages/
```

### Basic Robot Controller Script

Here's a Unity C# script to handle robot state visualization:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class RobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "HumanoidRobot";
    public float visualizationScale = 1.0f;

    [Header("Joint Mapping")]
    public Dictionary<string, Transform> jointMap = new Dictionary<string, Transform>();

    [Header("State Visualization")]
    public Material activeMaterial;      // For active joints
    public Material idleMaterial;        // For inactive joints
    public Color stateColor = Color.green;

    // Robot state data received from ROS
    private Dictionary<string, float> jointPositions = new Dictionary<string, float>();
    private Vector3 robotPosition;
    private Quaternion robotRotation;

    void Start()
    {
        InitializeJointMap();
        robotPosition = transform.position;
        robotRotation = transform.rotation;
    }

    /// <summary>
    /// Initialize the mapping between ROS joint names and Unity transforms
    /// </summary>
    private void InitializeJointMap()
    {
        // Find all joint transforms under the robot hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();

        foreach (Transform child in allChildren)
        {
            if (child.name.ToLower().Contains("joint") ||
                child.name.ToLower().Contains("link"))
            {
                jointMap[child.name] = child;

                // Initialize joint positions to zero
                jointPositions[child.name] = 0.0f;
            }
        }

        Debug.Log($"Initialized {jointMap.Count} joints for robot {robotName}");
    }

    /// <summary>
    /// Update robot state from ROS data
    /// </summary>
    public void UpdateRobotState(Dictionary<string, float> positions,
                                Vector3 position, Quaternion rotation)
    {
        // Update joint positions
        foreach (var kvp in positions)
        {
            if (jointMap.ContainsKey(kvp.Key))
            {
                jointPositions[kvp.Key] = kvp.Value;
            }
        }

        // Update robot position and rotation
        robotPosition = position;
        robotRotation = rotation;

        // Apply changes in LateUpdate for smooth animation
    }

    void LateUpdate()
    {
        // Update joint positions based on stored values
        foreach (var kvp in jointPositions)
        {
            if (jointMap.ContainsKey(kvp.Key))
            {
                Transform jointTransform = jointMap[kvp.Key];

                // Apply rotation to the joint
                // Note: This is a simplified example - real implementation would depend on joint type
                jointTransform.localRotation = Quaternion.Euler(0, 0, kvp.Value * Mathf.Rad2Deg);
            }
        }

        // Update robot position and rotation
        transform.position = robotPosition * visualizationScale;
        transform.rotation = robotRotation;
    }

    /// <summary>
    /// Highlight active joints
    /// </summary>
    public void SetJointActivity(string jointName, bool isActive)
    {
        if (jointMap.ContainsKey(jointName))
        {
            Renderer renderer = jointMap[jointName].GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material = isActive ? activeMaterial : idleMaterial;
            }
        }
    }

    /// <summary>
    /// Visualize sensor data as overlays
    /// </summary>
    public void VisualizeSensorData(string sensorType, float[] sensorValues)
    {
        switch (sensorType)
        {
            case "lidar":
                VisualizeLidarData(sensorValues);
                break;
            case "imu":
                VisualizeIMUData(sensorValues);
                break;
            case "camera":
                // Camera visualization would require texture updates
                break;
        }
    }

    private void VisualizeLidarData(float[] ranges)
    {
        // Create visualization for LiDAR data
        // This could involve creating point clouds or ray visualizations
        Debug.Log($"Visualizing LiDAR data with {ranges.Length} range measurements");
    }

    private void VisualizeIMUData(float[] values)
    {
        // Visualize IMU orientation and acceleration data
        Debug.Log($"Visualizing IMU data: [{values[0]}, {values[1]}, {values[2]}]");
    }
}
```

### ROS Connection Manager

Now let's implement the ROS connection manager:

```csharp
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor;
using RosSharp.Messages.Geometry;
using RosSharp.Messages.Nav;
using System.Collections.Generic;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("Connection Settings")]
    public string rosBridgeServerUrl = "ws://127.0.0.1:9090";

    [Header("Robot Topics")]
    public string jointStatesTopic = "/humanoid/joint_states";
    public string cmdVelTopic = "/humanoid/cmd_vel";
    public string laserScanTopic = "/humanoid/scan";
    public string imuTopic = "/humanoid/imu/data";

    [Header("References")]
    public RobotController robotController;

    private RosSocket rosSocket;
    private JointStateSubscriber jointStateSubscriber;
    private TwistPublisher cmdVelPublisher;
    private LaserScanSubscriber laserScanSubscriber;
    private ImuSubscriber imuSubscriber;

    void Start()
    {
        ConnectToROSBridge();
        SubscribeToTopics();
    }

    void ConnectToROSBridge()
    {
        WebSocketProtocols webSocketProtocol = WebSocketProtocols.GetProtocol(WebSocketProtocolType.Native);

        RosBridgeClient.Protocols.IProtocol protocol = new RosBridgeClient.Protocols.WebSocketNetProtocol(
            rosBridgeServerUrl, webSocketProtocol);

        rosSocket = new RosSocket(protocol);

        Debug.Log($"Connecting to ROS Bridge at {rosBridgeServerUrl}");
    }

    void SubscribeToTopics()
    {
        // Subscribe to joint states
        jointStateSubscriber = rosSocket.Subscribe<JointState>(jointStatesTopic, JointStateCallback);

        // Subscribe to laser scan
        laserScanSubscriber = rosSocket.Subscribe<LaserScan>(laserScanTopic, LaserScanCallback);

        // Subscribe to IMU data
        imuSubscriber = rosSocket.Subscribe<Imu>(imuTopic, ImuCallback);

        Debug.Log("Subscribed to ROS topics");
    }

    void JointStateCallback(JointState jointState)
    {
        if (robotController == null) return;

        // Convert ROS joint states to Unity format
        Dictionary<string, float> jointPositions = new Dictionary<string, float>();

        for (int i = 0; i < jointState.name.Count; i++)
        {
            if (i < jointState.position.Count)
            {
                jointPositions[jointState.name[i]] = (float)jointState.position[i];
            }
        }

        // For now, use identity position and rotation
        // In a real implementation, you'd get this from odometry
        Vector3 position = Vector3.zero;
        Quaternion rotation = Quaternion.identity;

        robotController.UpdateRobotState(jointPositions, position, rotation);
    }

    void LaserScanCallback(LaserScan scan)
    {
        if (robotController == null) return;

        // Convert ROS LaserScan to float array
        float[] ranges = new float[scan.ranges.Count];
        for (int i = 0; i < scan.ranges.Count; i++)
        {
            ranges[i] = (float)scan.ranges[i];
        }

        robotController.VisualizeSensorData("lidar", ranges);
    }

    void ImuCallback(Imu imu)
    {
        if (robotController == null) return;

        // Extract orientation and angular velocity
        float[] imuValues = new float[6];
        imuValues[0] = (float)imu.orientation.x;
        imuValues[1] = (float)imu.orientation.y;
        imuValues[2] = (float)imu.orientation.z;
        imuValues[3] = (float)imu.angular_velocity.x;
        imuValues[4] = (float)imu.angular_velocity.y;
        imuValues[5] = (float)imu.angular_velocity.z;

        robotController.VisualizeSensorData("imu", imuValues);
    }

    /// <summary>
    /// Send velocity command to robot
    /// </summary>
    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (cmdVelPublisher == null)
        {
            cmdVelPublisher = rosSocket.Advertise<Twist>(cmdVelTopic);
        }

        Twist cmd = new Twist();
        cmd.linear = new Vector3Msg(linearX, 0, 0);
        cmd.angular = new Vector3Msg(0, 0, angularZ);

        cmdVelPublisher.Publish(cmd);
    }

    void OnDestroy()
    {
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}
```

### Human-Robot Interaction Interface

Let's create a UI for human-robot interaction:

```csharp
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HRIInterface : MonoBehaviour
{
    [Header("UI References")]
    public Slider linearVelocitySlider;
    public Slider angularVelocitySlider;
    public Button forwardButton;
    public Button backwardButton;
    public Button leftButton;
    public Button rightButton;
    public Button stopButton;
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI jointInfoText;

    [Header("Robot Control")]
    public ROSConnectionManager rosManager;
    public RobotController robotController;

    [Header("Control Parameters")]
    public float maxLinearVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;
    public float controlUpdateInterval = 0.1f;

    private float lastControlUpdate = 0f;

    void Start()
    {
        SetupUIEvents();
        UpdateStatusDisplay();
    }

    void Update()
    {
        // Update status display periodically
        if (Time.time - lastControlUpdate > controlUpdateInterval)
        {
            UpdateStatusDisplay();
            lastControlUpdate = Time.time;
        }
    }

    void SetupUIEvents()
    {
        // Velocity sliders
        if (linearVelocitySlider != null)
        {
            linearVelocitySlider.onValueChanged.AddListener(OnLinearVelocityChanged);
        }

        if (angularVelocitySlider != null)
        {
            angularVelocitySlider.onValueChanged.AddListener(OnAngularVelocityChanged);
        }

        // Direction buttons
        if (forwardButton != null)
        {
            forwardButton.onClick.AddListener(() => MoveRobot(1, 0));
        }

        if (backwardButton != null)
        {
            backwardButton.onClick.AddListener(() => MoveRobot(-1, 0));
        }

        if (leftButton != null)
        {
            leftButton.onClick.AddListener(() => MoveRobot(0, 1));
        }

        if (rightButton != null)
        {
            rightButton.onClick.AddListener(() => MoveRobot(0, -1));
        }

        if (stopButton != null)
        {
            stopButton.onClick.AddListener(() => MoveRobot(0, 0));
        }
    }

    void OnLinearVelocityChanged(float value)
    {
        float linearVel = value * maxLinearVelocity;
        SendVelocityCommand(linearVel, angularVelocitySlider.value * maxAngularVelocity);
    }

    void OnAngularVelocityChanged(float value)
    {
        float angularVel = value * maxAngularVelocity;
        SendVelocityCommand(linearVelocitySlider.value * maxLinearVelocity, angularVel);
    }

    void MoveRobot(int linearDir, int angularDir)
    {
        float linearVel = linearDir * maxLinearVelocity;
        float angularVel = angularDir * maxAngularVelocity;

        SendVelocityCommand(linearVel, angularVel);
    }

    void SendVelocityCommand(float linearX, float angularZ)
    {
        if (rosManager != null)
        {
            rosManager.SendVelocityCommand(linearX, angularZ);

            // Update status text
            statusText.text = $"Linear: {linearX:F2}, Angular: {angularZ:F2}";
        }
    }

    void UpdateStatusDisplay()
    {
        if (statusText != null)
        {
            statusText.text = $"ROS Connected: {(rosManager != null ? "YES" : "NO")}\n" +
                             $"Robot Active: {(robotController != null ? "YES" : "NO")}";
        }

        if (jointInfoText != null && robotController != null)
        {
            // Display information about active joints
            jointInfoText.text = "Active Joints: " + robotController.GetActiveJointCount();
        }
    }

    /// <summary>
    /// Send custom command to robot
    /// </summary>
    public void SendCustomCommand(string command)
    {
        // This would be implemented based on specific robot capabilities
        Debug.Log($"Sending custom command: {command}");
    }
}
```

## Examples

### Example 1: Robot State Visualization System

Let's create a comprehensive robot state visualization system:

```csharp
using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class RobotStateVisualizer : MonoBehaviour
{
    [Header("Visualization Settings")]
    public GameObject robotModel;
    public Material defaultMaterial;
    public Material alertMaterial;
    public Material warningMaterial;

    [Header("Sensor Visualization")]
    public GameObject lidarPointCloudPrefab;
    public GameObject cameraFrustumPrefab;
    public GameObject pathVisualizerPrefab;

    [Header("UI Overlay")]
    public GameObject sensorOverlay;
    public GameObject statusOverlay;

    private Dictionary<string, GameObject> sensorVisualizers = new Dictionary<string, GameObject>();
    private Dictionary<string, Material> originalMaterials = new Dictionary<string, Material>();

    void Start()
    {
        SetupRobotMaterials();
        InitializeSensorVisualizers();
    }

    void SetupRobotMaterials()
    {
        // Store original materials for restoration
        Renderer[] renderers = robotModel.GetComponentsInChildren<Renderer>();

        foreach (Renderer renderer in renderers)
        {
            originalMaterials[renderer.name] = renderer.sharedMaterial;
        }
    }

    void InitializeSensorVisualizers()
    {
        // Create visualizers for different sensor types
        if (lidarPointCloudPrefab != null)
        {
            GameObject lidarVis = Instantiate(lidarPointCloudPrefab, robotModel.transform);
            sensorVisualizers["lidar"] = lidarVis;
            lidarVis.SetActive(false);
        }

        if (cameraFrustumPrefab != null)
        {
            GameObject camVis = Instantiate(cameraFrustumPrefab, robotModel.transform);
            sensorVisualizers["camera"] = camVis;
            camVis.SetActive(false);
        }

        if (pathVisualizerPrefab != null)
        {
            GameObject pathVis = Instantiate(pathVisualizerPrefab, robotModel.transform);
            sensorVisualizers["path"] = pathVis;
            pathVis.SetActive(false);
        }
    }

    /// <summary>
    /// Update robot state visualization based on health indicators
    /// </summary>
    public void UpdateRobotHealthVisualization(Dictionary<string, float> healthMetrics)
    {
        foreach (var metric in healthMetrics)
        {
            string componentName = metric.Key;
            float healthLevel = metric.Value; // 0.0 to 1.0

            // Update material based on health
            if (originalMaterials.ContainsKey(componentName))
            {
                Renderer compRenderer = GetRendererByName(componentName);
                if (compRenderer != null)
                {
                    if (healthLevel < 0.3f)
                    {
                        compRenderer.sharedMaterial = alertMaterial; // Critical
                    }
                    else if (healthLevel < 0.7f)
                    {
                        compRenderer.sharedMaterial = warningMaterial; // Warning
                    }
                    else
                    {
                        compRenderer.sharedMaterial = originalMaterials[componentName]; // Normal
                    }
                }
            }
        }
    }

    /// <summary>
    /// Visualize sensor data overlay
    /// </summary>
    public void VisualizeSensorData(string sensorType, object sensorData)
    {
        if (!sensorVisualizers.ContainsKey(sensorType))
            return;

        GameObject visObject = sensorVisualizers[sensorType];
        visObject.SetActive(true);

        switch (sensorType)
        {
            case "lidar":
                UpdateLidarVisualization((float[])sensorData);
                break;
            case "camera":
                UpdateCameraVisualization((Texture2D)sensorData);
                break;
            case "imu":
                UpdateIMUVisualization((Vector3)sensorData);
                break;
        }
    }

    void UpdateLidarVisualization(float[] ranges)
    {
        if (!sensorVisualizers.ContainsKey("lidar")) return;

        GameObject lidarVis = sensorVisualizers["lidar"];

        // In a real implementation, this would update a point cloud or ray visualization
        Debug.Log($"Updating LiDAR visualization with {ranges.Length} points");

        // Example: Create simple ray visualization
        LineRenderer lineRenderer = lidarVis.GetComponent<LineRenderer>();
        if (lineRenderer != null)
        {
            lineRenderer.positionCount = ranges.Length;

            for (int i = 0; i < ranges.Length; i++)
            {
                float angle = (float)i / ranges.Length * 2 * Mathf.PI;
                float range = ranges[i];

                if (float.IsNaN(range) || float.IsInfinity(range))
                    range = 0; // Skip invalid measurements

                Vector3 point = new Vector3(
                    Mathf.Cos(angle) * range,
                    0,
                    Mathf.Sin(angle) * range
                );

                lineRenderer.SetPosition(i, point);
            }
        }
    }

    void UpdateCameraVisualization(Texture2D cameraImage)
    {
        // Update camera frustum or overlay with the camera image
        Debug.Log("Updating camera visualization");
    }

    void UpdateIMUVisualization(Vector3 orientation)
    {
        // Update IMU visualization based on orientation data
        Debug.Log($"Updating IMU visualization: {orientation}");
    }

    Renderer GetRendererByName(string name)
    {
        Renderer[] renderers = robotModel.GetComponentsInChildren<Renderer>();
        return renderers.FirstOrDefault(r => r.name == name);
    }

    /// <summary>
    /// Toggle sensor visualization
    /// </summary>
    public void ToggleSensorVisualization(string sensorType, bool visible)
    {
        if (sensorVisualizers.ContainsKey(sensorType))
        {
            sensorVisualizers[sensorType].SetActive(visible);
        }
    }
}
```

### Example 2: Scene Manager for Multi-Environment Support

```csharp
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;
using System.Collections.Generic;

public class EnvironmentSceneManager : MonoBehaviour
{
    [Header("Environment Presets")]
    public EnvironmentPreset[] environmentPresets;

    [Header("Robot Spawn Points")]
    public Transform[] spawnPoints;

    [Header("Lighting Settings")]
    public Light sunLight;
    public AnimationCurve dayNightCycle;

    [Header("Weather Effects")]
    public ParticleSystem rainEffect;
    public ParticleSystem fogEffect;

    private int currentPresetIndex = 0;
    private float dayNightTimer = 0f;
    private bool isDayNightActive = false;

    [System.Serializable]
    public class EnvironmentPreset
    {
        public string presetName;
        public Color ambientColor;
        public Color skyColor;
        public float sunIntensity;
        public Vector3 sunDirection;
        public bool enableRain;
        public bool enableFog;
        public float windSpeed;
    }

    void Start()
    {
        // Initialize with first environment preset
        if (environmentPresets.Length > 0)
        {
            ApplyEnvironmentPreset(environmentPresets[0]);
        }
    }

    /// <summary>
    /// Apply an environment preset
    /// </summary>
    public void ApplyEnvironmentPreset(EnvironmentPreset preset)
    {
        // Apply ambient lighting
        RenderSettings.ambientLight = preset.ambientColor;

        // Apply skybox colors (assuming procedural skybox)
        if (RenderSettings.skybox != null)
        {
            RenderSettings.skybox.SetColor("_SkyTint", preset.skyColor);
        }

        // Apply sun light settings
        if (sunLight != null)
        {
            sunLight.intensity = preset.sunIntensity;
            sunLight.transform.eulerAngles = preset.sunDirection;
        }

        // Enable/disable weather effects
        if (rainEffect != null)
        {
            rainEffect.gameObject.SetActive(preset.enableRain);
            var emission = rainEffect.emission;
            emission.enabled = preset.enableRain;
        }

        if (fogEffect != null)
        {
            fogEffect.gameObject.SetActive(preset.enableFog);
            var emission = fogEffect.emission;
            emission.enabled = preset.enableFog;
        }

        // Apply wind settings if using a wind system
        ApplyWindSettings(preset.windSpeed);
    }

    void ApplyWindSettings(float windSpeed)
    {
        // In a real implementation, this would affect cloth, vegetation, etc.
        Physics.Wind = new Vector3(windSpeed, 0, 0);
    }

    /// <summary>
    /// Cycle through environment presets
    /// </summary>
    public void CycleEnvironmentPreset()
    {
        if (environmentPresets.Length == 0) return;

        currentPresetIndex = (currentPresetIndex + 1) % environmentPresets.Length;
        ApplyEnvironmentPreset(environmentPresets[currentPresetIndex]);
    }

    /// <summary>
    /// Start day-night cycle
    /// </summary>
    public void StartDayNightCycle()
    {
        isDayNightActive = true;
    }

    /// <summary>
    /// Stop day-night cycle
    /// </summary>
    public void StopDayNightCycle()
    {
        isDayNightActive = false;
    }

    void Update()
    {
        if (isDayNightActive)
        {
            dayNightTimer += Time.deltaTime * 0.01f; // Slow progression

            if (sunLight != null)
            {
                // Rotate sun based on time
                float sunAngle = dayNightTimer * 360f;
                sunLight.transform.rotation = Quaternion.Euler(sunAngle, 30, 0);

                // Adjust intensity based on day/night curve
                float intensityFactor = dayNightCycle.Evaluate((dayNightTimer % 1.0f));
                sunLight.intensity = intensityFactor * 1.0f; // Base intensity

                // Adjust ambient light
                RenderSettings.ambientLight = Color.Lerp(Color.black, Color.white, intensityFactor * 0.5f);
            }
        }
    }

    /// <summary>
    /// Get random spawn point for robot
    /// </summary>
    public Transform GetRandomSpawnPoint()
    {
        if (spawnPoints.Length == 0) return null;

        int randomIndex = Random.Range(0, spawnPoints.Length);
        return spawnPoints[randomIndex];
    }

    /// <summary>
    /// Get specific spawn point by index
    /// </summary>
    public Transform GetSpawnPoint(int index)
    {
        if (index >= 0 && index < spawnPoints.Length)
        {
            return spawnPoints[index];
        }
        return null;
    }

    /// <summary>
    /// Save current environment state
    /// </summary>
    public void SaveEnvironmentState()
    {
        // In a real implementation, this would save to a file or PlayerPrefs
        Debug.Log("Environment state saved");
    }

    /// <summary>
    /// Load environment state
    /// </summary>
    public void LoadEnvironmentState()
    {
        // In a real implementation, this would load from a file or PlayerPrefs
        Debug.Log("Environment state loaded");
    }
}
```

## Summary

Unity visualization provides powerful capabilities for creating immersive human-robot interaction experiences. Key considerations for effective implementation include:

- **Performance Optimization**: Balancing visual fidelity with real-time rendering requirements
- **ROS Integration**: Ensuring reliable communication between Unity and ROS 2 systems
- **User Experience**: Designing intuitive interfaces for robot control and monitoring
- **Realism vs. Performance**: Finding the right balance for the intended application
- **Modularity**: Creating reusable components that can adapt to different robot platforms

## Exercises

### Conceptual
1. Explain the advantages and limitations of using Unity versus Gazebo for robotics visualization and human-robot interaction design.

### Logical
1. Analyze the trade-offs between visual realism and computational performance in Unity-based robot visualization. When would you prioritize one over the other in a robotics application?

### Implementation
1. Create a Unity visualization scene with a humanoid robot model that integrates with ROS 2, implements sensor data visualization, and provides a human-robot interaction interface for teleoperation and monitoring.
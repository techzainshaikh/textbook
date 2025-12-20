---
title: Synthetic Data Generation for AI Training
sidebar_position: 3
description: Creating synthetic datasets using Isaac Sim for AI model training
keywords: [synthetic data, isaac sim, data generation, ai training, computer vision]
---

# Chapter 2: Synthetic Data Generation for AI Training

## Learning Objectives

By the end of this chapter, students will be able to:
- Generate synthetic datasets using Isaac Sim for AI model training
- Configure realistic sensor noise and environmental variations in simulation
- Create diverse training scenarios with varying lighting, textures, and objects
- Implement domain randomization techniques for robust AI models
- Validate synthetic data quality against real-world datasets

## Prerequisites

Students should have:
- Understanding of computer vision and machine learning fundamentals
- Experience with Isaac Sim (covered in Chapter 1)
- Knowledge of dataset formats for training AI models
- Basic understanding of domain adaptation concepts

## Core Concepts

Synthetic data generation is a critical component of modern robotics AI development, allowing for rapid dataset creation without the time and cost of real-world data collection. Isaac Sim provides powerful capabilities for generating diverse, labeled datasets that can be used to train perception and control systems.

### Synthetic Data Benefits

**Cost and Time Efficiency:**
- Generate thousands of labeled images in minutes rather than days
- Control environmental conditions (lighting, weather, objects)
- Create rare scenarios that are difficult to capture in real life

**Safety:**
- Train in dangerous scenarios without risk to equipment or personnel
- Test edge cases without real-world consequences
- Generate failure scenarios for robustness testing

**Label Quality:**
- Perfect ground truth for segmentation, depth, poses
- Consistent labeling across large datasets
- Multiple modalities (RGB, depth, semantic segmentation) synchronized

### Domain Randomization

Domain randomization involves varying environmental parameters to create diverse training data:

**Visual Properties:**
- Lighting conditions (intensity, direction, color temperature)
- Texture variations (materials, colors, patterns)
- Camera properties (noise, blur, distortion)

**Physical Properties:**
- Object poses and configurations
- Environmental layouts
- Dynamics parameters (friction, mass, restitution)

**Sensor Properties:**
- Noise characteristics
- Resolution variations
- Distortion parameters

## Implementation

Let's implement synthetic data generation using Isaac Sim capabilities:

### Isaac Sim Synthetic Data Pipeline

```python
#!/usr/bin/env python3
# synthetic_data_generator.py

import omni
import carb
import numpy as np
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.replicator.core import Replicator
import omni.synthetic_utils as synth_utils
import PIL.Image
import json
import os
from pathlib import Path

class IsaacSyntheticDataGenerator:
    """
    Class for generating synthetic datasets using Isaac Sim
    """

    def __init__(self, output_dir="synthetic_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize replicator for synthetic data generation
        self.replicator = Replicator()

        # Data counters
        self.image_count = 0
        self.annotation_count = 0

        print(f"Synthetic data generator initialized. Output directory: {self.output_dir}")

    def setup_replication_graph(self):
        """Setup replication graph for data generation"""
        # Create a basic scene with randomizable elements
        # This would typically include:
        # - Randomizable objects with domain randomization
        # - Adjustable lighting
        # - Camera positioning
        # - Sensor noise models

        # Example: Create randomizable objects
        self.setup_randomizable_objects()

        # Example: Configure lighting randomization
        self.setup_lighting_randomization()

        # Example: Configure camera properties
        self.setup_camera_properties()

    def setup_randomizable_objects(self):
        """Setup objects that can be randomized"""
        # In a real implementation, this would use Isaac Replicator
        # to define randomizable properties for objects in the scene
        pass

    def setup_lighting_randomization(self):
        """Setup lighting that can be randomized"""
        # In a real implementation, this would define randomizable
        # lighting parameters such as intensity, color, direction
        pass

    def setup_camera_properties(self):
        """Setup camera with configurable properties"""
        # In a real implementation, this would define camera
        # properties that can be randomized during data generation
        pass

    def generate_training_data(self, num_samples=1000, data_types=["rgb", "depth", "seg"]):
        """Generate synthetic training data"""
        print(f"Generating {num_samples} synthetic samples with types: {data_types}")

        for i in range(num_samples):
            # Randomize scene parameters
            self.randomize_scene()

            # Capture data from scene
            sample_data = self.capture_sample(data_types)

            # Save sample data
            self.save_sample(sample_data, i)

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")

        print(f"Completed generation of {num_samples} synthetic samples")

    def randomize_scene(self):
        """Randomize scene properties for domain randomization"""
        # This would randomize:
        # - Object positions and orientations
        # - Lighting conditions
        # - Material properties
        # - Camera parameters
        # - Environmental settings
        pass

    def capture_sample(self, data_types):
        """Capture a single sample with specified data types"""
        sample = {}

        for data_type in data_types:
            if data_type == "rgb":
                # Capture RGB image
                sample[data_type] = self.capture_rgb_image()
            elif data_type == "depth":
                # Capture depth image
                sample[data_type] = self.capture_depth_image()
            elif data_type == "seg":
                # Capture segmentation mask
                sample[data_type] = self.capture_segmentation()
            elif data_type == "pose":
                # Capture object poses
                sample[data_type] = self.capture_poses()

        return sample

    def capture_rgb_image(self):
        """Capture RGB image from virtual camera"""
        # In Isaac Sim, this would use the rendering pipeline
        # to capture RGB data from a virtual camera
        width, height = 640, 480
        rgb_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return rgb_data

    def capture_depth_image(self):
        """Capture depth image from virtual depth sensor"""
        # In Isaac Sim, this would use the depth rendering pipeline
        width, height = 640, 480
        depth_data = np.random.uniform(0.1, 10.0, (height, width)).astype(np.float32)
        return depth_data

    def capture_segmentation(self):
        """Capture semantic segmentation mask"""
        # In Isaac Sim, this would use semantic segmentation rendering
        width, height = 640, 480
        seg_data = np.random.randint(0, 10, (height, width), dtype=np.uint8)
        return seg_data

    def capture_poses(self):
        """Capture object poses in the scene"""
        # In Isaac Sim, this would capture ground truth poses
        # of objects in the scene
        poses = {
            "object_1": {"position": [1.0, 0.0, 0.5], "rotation": [0, 0, 0, 1]},
            "object_2": {"position": [0.5, 1.0, 0.3], "rotation": [0, 0, 0, 1]}
        }
        return poses

    def save_sample(self, sample_data, sample_idx):
        """Save a single sample to disk"""
        sample_dir = self.output_dir / f"sample_{sample_idx:06d}"
        sample_dir.mkdir(exist_ok=True)

        for data_type, data in sample_data.items():
            if data_type == "rgb":
                # Save RGB image
                img = PIL.Image.fromarray(data)
                img.save(sample_dir / "rgb.png")
            elif data_type == "depth":
                # Save depth image
                depth_img = PIL.Image.fromarray((data * 256).astype(np.uint16))
                depth_img.save(sample_dir / "depth.png")
            elif data_type == "seg":
                # Save segmentation mask
                seg_img = PIL.Image.fromarray(data)
                seg_img.save(sample_dir / "segmentation.png")
            elif data_type == "pose":
                # Save pose annotations
                with open(sample_dir / "poses.json", 'w') as f:
                    json.dump(data, f)

        # Create annotation file for this sample
        annotation = {
            "sample_id": f"sample_{sample_idx:06d}",
            "data_types": list(sample_data.keys()),
            "timestamp": carb.events.acquire_events_interface().get_current_event_time(),
            "scene_parameters": self.get_current_scene_params()
        }

        with open(sample_dir / "annotation.json", 'w') as f:
            json.dump(annotation, f, indent=2)

        self.image_count += 1

    def get_current_scene_params(self):
        """Get current scene randomization parameters"""
        # Return current randomization settings for reproducibility
        return {
            "lighting_intensity": np.random.uniform(0.5, 1.5),
            "lighting_color_temp": np.random.uniform(5000, 8000),
            "camera_noise_level": np.random.uniform(0.0, 0.05),
            "object_count": np.random.randint(1, 5)
        }

    def validate_synthetic_data(self, real_dataset_stats):
        """Validate synthetic data quality against real dataset statistics"""
        # Compare statistical properties of synthetic vs real data
        # This could include:
        # - Color distribution comparison
        # - Texture complexity metrics
        # - Object size distributions
        # - Depth range distributions

        print("Validating synthetic data quality...")

        # Example validation: compare mean and std of RGB channels
        synthetic_stats = self.compute_synthetic_stats()

        # Compute similarity metrics
        similarity_score = self.compute_similarity(real_dataset_stats, synthetic_stats)

        print(f"Synthetic data quality score: {similarity_score:.3f}")

        return similarity_score > 0.7  # Return True if quality is acceptable

    def compute_synthetic_stats(self):
        """Compute statistics for synthetic dataset"""
        # This would compute various statistical measures
        # across the generated synthetic dataset
        return {
            "color_mean": [120, 115, 110],
            "color_std": [50, 45, 40],
            "depth_mean": 3.5,
            "depth_std": 1.2
        }

    def compute_similarity(self, real_stats, synthetic_stats):
        """Compute similarity between real and synthetic datasets"""
        # Simplified similarity computation
        # In practice, this would use more sophisticated statistical tests
        score = 0.0

        # Compare color statistics
        for channel in ["color_mean", "color_std"]:
            real_vals = np.array(real_stats[channel])
            synth_vals = np.array(synthetic_stats[channel])
            diff = np.mean(np.abs(real_vals - synth_vals) / (real_vals + 1e-6))
            score += (1.0 - min(diff, 1.0))

        # Compare depth statistics
        depth_diff = abs(real_stats["depth_mean"] - synthetic_stats["depth_mean"]) / real_stats["depth_mean"]
        score += (1.0 - min(depth_diff, 1.0))

        return score / 3.0  # Normalize to 0-1 range

def main():
    """Main entry point for synthetic data generation"""
    generator = IsaacSyntheticDataGenerator(output_dir="synthetic_robot_dataset")

    # Setup the replication graph
    generator.setup_replication_graph()

    # Generate training data
    generator.generate_training_data(
        num_samples=100,
        data_types=["rgb", "depth", "seg", "pose"]
    )

    print("Synthetic data generation completed!")

if __name__ == "__main__":
    main()
```

### Synthetic Dataset Validator

Let's create a validation tool to assess synthetic dataset quality:

```python
#!/usr/bin/env python3
# synthetic_dataset_validator.py

import numpy as np
import cv2
import json
import os
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

class SyntheticDatasetValidator:
    """
    Validates synthetic datasets for quality and consistency
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.samples = []
        self.stats = {}

    def load_dataset(self):
        """Load dataset samples"""
        sample_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]

        for sample_dir in sample_dirs:
            sample_data = {}

            # Load RGB image
            rgb_path = sample_dir / "rgb.png"
            if rgb_path.exists():
                sample_data["rgb"] = cv2.imread(str(rgb_path))

            # Load depth image
            depth_path = sample_dir / "depth.png"
            if depth_path.exists():
                sample_data["depth"] = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0

            # Load segmentation
            seg_path = sample_dir / "segmentation.png"
            if seg_path.exists():
                sample_data["seg"] = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)

            # Load annotations
            annot_path = sample_dir / "annotation.json"
            if annot_path.exists():
                with open(annot_path, 'r') as f:
                    sample_data["annotation"] = json.load(f)

            self.samples.append(sample_data)

        print(f"Loaded {len(self.samples)} samples from {self.dataset_path}")

    def validate_rgb_quality(self):
        """Validate RGB image quality"""
        rgb_means = []
        rgb_stds = []

        for sample in self.samples:
            if "rgb" in sample:
                img = sample["rgb"]
                # Convert BGR to RGB for proper analysis
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Calculate mean and std for each channel
                means = [np.mean(img_rgb[:,:,i]) for i in range(3)]
                stds = [np.std(img_rgb[:,:,i]) for i in range(3)]

                rgb_means.append(means)
                rgb_stds.append(stds)

        if rgb_means:
            mean_rgb_means = np.mean(rgb_means, axis=0)
            mean_rgb_stds = np.mean(rgb_stds, axis=0)

            print(f"RGB Channel Means: {mean_rgb_means}")
            print(f"RGB Channel Stds: {mean_rgb_stds}")

            # Check for reasonable ranges
            if np.any(mean_rgb_means < 20) or np.any(mean_rgb_means > 235):
                print("WARNING: Some RGB channels have extreme mean values (possible lighting issues)")

            if np.any(mean_rgb_stds < 20):
                print("WARNING: Some RGB channels have low variance (possible lack of texture)")

    def validate_depth_consistency(self):
        """Validate depth image consistency"""
        depth_values = []

        for sample in self.samples:
            if "depth" in sample:
                depth_img = sample["depth"]
                valid_depths = depth_img[depth_img > 0]  # Exclude invalid depths
                depth_values.extend(valid_depths)

        if depth_values:
            depth_array = np.array(depth_values)

            print(f"Depth Statistics:")
            print(f"  Mean: {np.mean(depth_array):.2f}m")
            print(f"  Std: {np.std(depth_array):.2f}m")
            print(f"  Min: {np.min(depth_array):.2f}m")
            print(f"  Max: {np.max(depth_array):.2f}m")
            print(f"  Median: {np.median(depth_array):.2f}m")

            # Check for reasonable depth ranges
            if np.max(depth_array) > 50.0:
                print("WARNING: Maximum depth value is unusually high")

            if np.min(depth_array) < 0.01:
                print("WARNING: Minimum depth value is unusually low (possible sensor clipping)")

    def validate_annotations(self):
        """Validate annotation quality and consistency"""
        annotation_stats = []

        for sample in self.samples:
            if "annotation" in sample:
                annot = sample["annotation"]
                annotation_stats.append(annot)

        if annotation_stats:
            print(f"Validated {len(annotation_stats)} annotations")

            # Check for consistent scene parameters
            lighting_intensities = [a["scene_parameters"]["lighting_intensity"] for a in annotation_stats if "scene_parameters" in a]
            if lighting_intensities:
                print(f"Lighting intensity range: {min(lighting_intensities):.2f} - {max(lighting_intensities):.2f}")

            object_counts = [a["scene_parameters"]["object_count"] for a in annotation_stats if "scene_parameters" in a]
            if object_counts:
                print(f"Object count range: {min(object_counts)} - {max(object_counts)} (avg: {np.mean(object_counts):.1f})")

    def validate_segmentation(self):
        """Validate segmentation masks"""
        seg_class_histograms = []

        for sample in self.samples:
            if "seg" in sample:
                seg_img = sample["seg"]
                # Calculate histogram of segmentation classes
                unique, counts = np.unique(seg_img, return_counts=True)
                hist = dict(zip(unique, counts))
                seg_class_histograms.append(hist)

        if seg_class_histograms:
            print(f"Segmentation validation for {len(seg_class_histograms)} samples")

            # Analyze class distribution
            all_classes = set()
            for hist in seg_class_histograms:
                all_classes.update(hist.keys())

            print(f"Found {len(all_classes)} unique segmentation classes: {sorted(all_classes)}")

            # Check for samples with too few distinct classes (possible rendering issues)
            for i, hist in enumerate(seg_class_histograms):
                if len(hist) < 2:
                    print(f"WARNING: Sample {i} has only {len(hist)} segmentation class(es) - may be invalid")

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n=== SYNTHETIC DATASET VALIDATION REPORT ===")

        # RGB quality validation
        print("\nRGB Quality Assessment:")
        self.validate_rgb_quality()

        # Depth consistency validation
        print("\nDepth Consistency Assessment:")
        self.validate_depth_consistency()

        # Annotation validation
        print("\nAnnotation Quality Assessment:")
        self.validate_annotations()

        # Segmentation validation
        print("\nSegmentation Quality Assessment:")
        self.validate_segmentation()

        print("\n=========================================")

def main():
    """Main validation function"""
    validator = SyntheticDatasetValidator("synthetic_robot_dataset")
    validator.load_dataset()
    validator.generate_validation_report()

if __name__ == "__main__":
    main()
```

## Examples

### Example 1: Domain Randomization Configuration

Here's an example of how to configure domain randomization in Isaac Sim:

```python
#!/usr/bin/env python3
# domain_randomization_config.py

import omni.replicator.core as rep
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

# Initialize replicator
rep.random_seed(42)

# Define randomization functions
@rep.randomizer
def randomize_lighting():
    """Randomize lighting properties"""
    lights = rep.get.light()

    with lights.randomize.light():
        # Randomize intensity between 500 and 1500
        rep.modify.attribute("inputs:intensity", rep.distribution.normal(1000, 200))

        # Randomize color temperature between 4000K and 8000K
        rep.modify.attribute("inputs:color", rep.distribution.uniform([0.8, 0.9, 1.0], [1.0, 0.9, 0.8]))

@rep.randomizer
def randomize_materials():
    """Randomize material properties"""
    prims = rep.get.prims()

    with prims.randomize.prim_type_regex(".*Material.*"):
        # Randomize roughness
        rep.modify.attribute("roughness", rep.distribution.uniform(0.1, 0.9))

        # Randomize metallic
        rep.modify.attribute("metallic", rep.distribution.uniform(0.0, 0.3))

@rep.randomizer
def randomize_objects():
    """Randomize object positions and properties"""
    cubes = rep.get.cube()

    with cubes.randomize.position():
        # Randomize positions in a certain area
        rep.modify.bbox((-2, -2, 0.1), (2, 2, 1.0))

    with cubes.randomize.rotation():
        # Randomize rotations
        rep.modify.rotation(rep.distribution.uniform((-180, -180, -180), (180, 180, 180)))

# Create trigger for randomization
trigger = rep.trigger.on_frame(num_frames=1)

# Register randomizers with trigger
with trigger:
    randomize_lighting()
    randomize_materials()
    randomize_objects()

# Create a function to capture data with randomization
def capture_randomized_data(num_samples=100):
    """Capture data with domain randomization applied"""
    print(f"Capturing {num_samples} samples with domain randomization...")

    for i in range(num_samples):
        # Trigger randomization
        rep.orchestrator.step()

        # Capture RGB, depth, segmentation
        # This would use Isaac Sim's rendering pipelines
        print(f"Captured sample {i+1}/{num_samples}")

    print("Data capture completed with domain randomization")

# Execute the randomization and capture process
capture_randomized_data(50)
```

### Example 2: Synthetic Dataset Pipeline Integration

```python
#!/usr/bin/env python3
# synthetic_pipeline_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import json
from pathlib import Path

class SyntheticPipelineIntegration(Node):
    """
    Integrate synthetic data generation with ROS 2 pipeline
    """

    def __init__(self):
        super().__init__('synthetic_pipeline_integration')

        # Publishers for synthetic data
        self.synthetic_rgb_pub = self.create_publisher(Image, '/synthetic/camera/rgb', 10)
        self.synthetic_depth_pub = self.create_publisher(Image, '/synthetic/camera/depth', 10)
        self.synthetic_status_pub = self.create_publisher(String, '/synthetic/status', 10)

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # Synthetic data parameters
        self.dataset_path = Path("synthetic_robot_dataset")
        self.current_sample_idx = 0
        self.total_samples = 0

        # Timers
        self.data_gen_timer = self.create_timer(0.1, self.publish_synthetic_data)
        self.status_timer = self.create_timer(5.0, self.publish_status)

        # Load dataset if available
        self.load_synthetic_dataset()

        self.get_logger().info('Synthetic Pipeline Integration node initialized')

    def load_synthetic_dataset(self):
        """Load synthetic dataset for streaming"""
        if self.dataset_path.exists():
            sample_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
            self.total_samples = len(sample_dirs)
            self.get_logger().info(f'Loaded {self.total_samples} synthetic samples')
        else:
            self.get_logger().warn(f'Dataset path {self.dataset_path} does not exist')

    def publish_synthetic_data(self):
        """Publish synthetic data to ROS topics"""
        if self.total_samples == 0:
            return

        # Load next sample
        sample_path = self.dataset_path / f"sample_{self.current_sample_idx:06d}"

        if not sample_path.exists():
            # Loop back to beginning if we reach the end
            self.current_sample_idx = 0
            sample_path = self.dataset_path / f"sample_{self.current_sample_idx:06d}"

        # Load RGB image
        rgb_path = sample_path / "rgb.png"
        if rgb_path.exists():
            rgb_img = cv2.imread(str(rgb_path))
            if rgb_img is not None:
                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

                # Convert to ROS Image message
                ros_img = self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
                ros_img.header.stamp = self.get_clock().now().to_msg()
                ros_img.header.frame_id = "synthetic_camera_rgb_optical_frame"

                self.synthetic_rgb_pub.publish(ros_img)

        # Load depth image
        depth_path = sample_path / "depth.png"
        if depth_path.exists():
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                # Convert to float32 meters
                depth_img_float = depth_img.astype(np.float32) / 256.0

                # Convert to ROS Image message
                ros_depth = self.cv_bridge.cv2_to_imgmsg(depth_img_float, encoding="passthrough")
                ros_depth.header.stamp = self.get_clock().now().to_msg()
                ros_depth.header.frame_id = "synthetic_camera_depth_optical_frame"

                self.synthetic_depth_pub.publish(ros_depth)

        # Move to next sample
        self.current_sample_idx = (self.current_sample_idx + 1) % self.total_samples

    def publish_status(self):
        """Publish synthetic data generation status"""
        status_msg = String()
        status_msg.data = f"Synthetic dataset streaming: {self.current_sample_idx}/{self.total_samples} samples, " \
                         f"rate: 10Hz"
        self.synthetic_status_pub.publish(status_msg)

    def validate_synthetic_data_stream(self):
        """Validate the quality of synthetic data stream"""
        # This would validate that the synthetic data stream meets quality requirements
        # For example, checking that images are properly formatted, depths are reasonable, etc.
        self.get_logger().info("Validating synthetic data stream...")

def main(args=None):
    rclpy.init(args=args)
    node = SyntheticPipelineIntegration()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Synthetic Pipeline Integration...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Synthetic data generation is a cornerstone of modern robotics AI development, enabling rapid training dataset creation with perfect ground truth annotations. Isaac Sim provides powerful capabilities for generating diverse, labeled datasets that can be used to train perception and control systems with domain randomization techniques to ensure robustness.

Key aspects include:
- **Efficiency**: Generating thousands of labeled samples in minutes rather than hours
- **Safety**: Training in dangerous scenarios without risk to equipment or personnel
- **Quality**: Perfect ground truth annotations across multiple modalities
- **Variety**: Domain randomization to create robust AI models

## Exercises

### Conceptual
1. Explain the advantages of synthetic data generation over real-world data collection for robotics AI training.

### Logical
1. Analyze the domain randomization parameters that would be most important for training a humanoid robot to operate in diverse indoor environments.

### Implementation
1. Implement a synthetic data generation pipeline that creates a dataset of 1000 images with randomized lighting, textures, and object arrangements suitable for training a humanoid robot's perception system.
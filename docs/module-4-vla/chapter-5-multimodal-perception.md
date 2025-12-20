---
title: Multimodal Perception for Vision-Language-Action Systems
sidebar_position: 6
description: Implementing multimodal perception systems that integrate vision, language, and sensor data for humanoid robotics applications
keywords: [multimodal perception, computer vision, sensor fusion, humanoid robotics, vision-language models, multimodal AI]
---

# Chapter 5: Multimodal Perception for Vision-Language-Action Systems

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement multimodal perception systems that integrate vision, language, and sensor data
- Design sensor fusion architectures for comprehensive environmental understanding
- Integrate vision-language models for object recognition and scene understanding
- Create multimodal embeddings that combine visual and textual information
- Validate and test multimodal perception systems for humanoid robotics applications

## Prerequisites

Students should have:
- Understanding of computer vision fundamentals (covered in Module 2)
- Knowledge of sensor fusion concepts (covered in Module 1)
- Familiarity with neural networks and deep learning
- Experience with Python programming and PyTorch/TensorFlow
- Basic understanding of natural language processing concepts

## Core Concepts

Multimodal perception systems combine multiple sensory inputs to create a comprehensive understanding of the environment. These systems are crucial for humanoid robots to navigate and interact effectively with complex, dynamic environments.

### Multimodal Architecture

**Vision Processing:**
- **Object Detection**: Identify and locate objects in visual scenes
- **Pose Estimation**: Determine object poses and spatial relationships
- **Scene Understanding**: Comprehend spatial layouts and semantic relationships
- **Activity Recognition**: Recognize human activities and behaviors

**Language Integration:**
- **Visual Grounding**: Link language descriptions to visual elements
- **Caption Generation**: Describe scenes in natural language
- **Query Understanding**: Interpret language queries about visual content
- **Instruction Following**: Execute language-based commands in visual contexts

**Sensor Fusion:**
- **Multi-modal Integration**: Combine visual, auditory, tactile, and proprioceptive data
- **Temporal Consistency**: Maintain coherent understanding across time
- **Uncertainty Management**: Handle noisy and incomplete sensor data
- **Real-time Processing**: Achieve timely responses for interactive applications

### Vision-Language Models

Modern vision-language models enable sophisticated understanding by combining visual and textual modalities:

- **CLIP**: Contrastive Language-Image Pretraining for zero-shot recognition
- **BLIP**: Bootstrapping Language-Image Pretraining for unified vision-language understanding
- **ViLT**: Vision-and-Language Transformer for efficient multimodal processing
- **Florence**: Foundation model for comprehensive vision-language tasks

## Implementation

Let's implement multimodal perception systems for humanoid robotics:

### Vision-Language Integration Framework

```python
#!/usr/bin/env python3
# vision_language_integration.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class PerceptionOutput:
    """Output from the multimodal perception system"""
    objects: List[Dict[str, Any]]  # Detected objects with properties
    scene_description: str         # Natural language description of scene
    spatial_relationships: List[Tuple[str, str, str]]  # Object relationships
    embeddings: Dict[str, torch.Tensor]  # Multimodal embeddings
    confidence_scores: Dict[str, float]  # Confidence scores for detections
    timestamp: float               # Processing timestamp

class VisionLanguageModel(nn.Module):
    """
    Vision-language model for multimodal perception
    """

    def __init__(self, model_type: str = "clip"):
        super().__init__()
        self.model_type = model_type

        if model_type == "clip":
            # Load pre-trained CLIP model
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif model_type == "blip":
            # Load BLIP model for image captioning
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        else:
            # Simple CNN-based approach
            self.visual_encoder = self._create_visual_encoder()
            self.text_encoder = self._create_text_encoder()

    def _create_visual_encoder(self):
        """Create a simple CNN-based visual encoder"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512)
        )

    def _create_text_encoder(self):
        """Create a simple text encoder"""
        vocab_size = 10000
        embed_dim = 512
        return nn.Embedding(vocab_size, embed_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual features"""
        if self.model_type == "clip":
            return self.clip_model.get_image_features(pixel_values=images)
        elif self.model_type == "blip":
            # For BLIP, we need to handle differently
            return self.blip_model.get_text_features(input_ids=images)  # This is a simplification
        else:
            return self.visual_encoder(images)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text into language features"""
        if self.model_type == "clip":
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            return self.clip_model.get_text_features(input_ids=inputs.input_ids)
        else:
            # Simple approach: convert text to tokens and embed
            # This is a simplified version - in practice, you'd use proper tokenization
            tokenized_texts = [[ord(c) % 10000 for c in text[:100]] for text in texts]  # Simplified tokenization
            max_len = max(len(tokens) for tokens in tokenized_texts)
            padded_texts = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized_texts]
            text_tensor = torch.tensor(padded_texts)
            embedded = self.text_encoder(text_tensor)
            return embedded.mean(dim=1)  # Average pooling

    def forward(self, images: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass through the vision-language model"""
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return {
            "image_features": image_features,
            "text_features": text_features,
            "similarity": torch.matmul(image_features, text_features.t())
        }

class ObjectDetector(nn.Module):
    """
    Object detection module for multimodal perception
    """

    def __init__(self, model_type: str = "yolo"):
        super().__init__()
        self.model_type = model_type

        if model_type == "yolo":
            # In practice, you'd load a YOLO model
            # For this example, we'll create a simple detector
            self.detector = self._create_simple_detector()
        else:
            # Use torchvision models
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
            self.detector.eval()

    def _create_simple_detector(self):
        """Create a simple object detector for demonstration"""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256)
        )

    def forward(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """Detect objects in images"""
        if self.model_type == "yolo":
            # For demonstration, return mock detections
            batch_size = images.size(0)
            detections = []

            for i in range(batch_size):
                # Create mock detections
                mock_detections = [
                    {
                        "label": "person",
                        "confidence": 0.95,
                        "bbox": [0.1, 0.1, 0.3, 0.4],  # x, y, width, height (normalized)
                        "embedding": torch.randn(512)
                    },
                    {
                        "label": "chair",
                        "confidence": 0.89,
                        "bbox": [0.4, 0.3, 0.2, 0.3],
                        "embedding": torch.randn(512)
                    }
                ]
                detections.append(mock_detections)

            return detections
        else:
            # For FasterRCNN, use the actual model
            # This is a simplified version - in practice, you'd handle the full pipeline
            return []

    def detect_objects(self, image: torch.Tensor) -> List[Dict[str, Any]]:
        """Detect objects in a single image"""
        detections = self(image.unsqueeze(0))
        return detections[0] if detections else []

class SpatialReasoner:
    """
    Spatial reasoning module for understanding object relationships
    """

    def __init__(self):
        self.spatial_relations = [
            "left of", "right of", "above", "below", "next to",
            "behind", "in front of", "inside", "outside", "on top of"
        ]

    def compute_spatial_relationships(self, objects: List[Dict[str, Any]],
                                    image_shape: Tuple[int, int]) -> List[Tuple[str, str, str]]:
        """Compute spatial relationships between objects"""
        relationships = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    rel = self._compute_relationship(obj1, obj2, image_shape)
                    if rel:
                        relationships.append(rel)

        return relationships

    def _compute_relationship(self, obj1: Dict[str, Any], obj2: Dict[str, Any],
                            image_shape: Tuple[int, int]) -> Optional[Tuple[str, str, str]]:
        """Compute the spatial relationship between two objects"""
        bbox1 = obj1["bbox"]  # [x, y, width, height] - normalized
        bbox2 = obj2["bbox"]

        # Convert normalized coordinates to pixel coordinates
        h, w = image_shape
        x1, y1 = bbox1[0] * w, bbox1[1] * h
        w1, h1 = bbox1[2] * w, bbox1[3] * h
        cx1, cy1 = x1 + w1/2, y1 + h1/2  # Center of object 1

        x2, y2 = bbox2[0] * w, bbox2[1] * h
        w2, h2 = bbox2[2] * w, bbox2[3] * h
        cx2, cy2 = x2 + w2/2, y2 + h2/2  # Center of object 2

        # Compute spatial relationship
        dx = cx2 - cx1
        dy = cy2 - cy1

        # Determine primary direction
        if abs(dx) > abs(dy):
            # Horizontal relationship
            if dx > 0:
                return (obj1["label"], "right of", obj2["label"])
            else:
                return (obj1["label"], "left of", obj2["label"])
        else:
            # Vertical relationship
            if dy > 0:
                return (obj1["label"], "below", obj2["label"])
            else:
                return (obj1["label"], "above", obj2["label"])

class MultimodalPerceptionSystem:
    """
    Complete multimodal perception system for humanoid robots
    """

    def __init__(self, vl_model_type: str = "clip", detector_type: str = "yolo"):
        self.vision_language_model = VisionLanguageModel(vl_model_type)
        self.object_detector = ObjectDetector(detector_type)
        self.spatial_reasoner = SpatialReasoner()

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.logger = logging.getLogger(__name__)

    def process_image(self, image: np.ndarray, text_queries: List[str] = None) -> PerceptionOutput:
        """Process an image with optional text queries"""
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0)

        # Detect objects
        objects = self.object_detector.detect_objects(image_tensor)

        # Generate scene description if no text queries provided
        if not text_queries:
            scene_description = self._generate_scene_description(objects)
        else:
            # Use text queries to guide perception
            scene_description = self._answer_queries(image_tensor, text_queries)

        # Compute spatial relationships
        image_h, image_w = image.shape[:2]
        spatial_relationships = self.spatial_reasoner.compute_spatial_relationships(
            objects, (image_h, image_w)
        )

        # Generate multimodal embeddings
        embeddings = self._generate_embeddings(image_tensor, objects, text_queries or [scene_description])

        # Compute confidence scores
        confidence_scores = self._compute_confidence_scores(objects)

        return PerceptionOutput(
            objects=objects,
            scene_description=scene_description,
            spatial_relationships=spatial_relationships,
            embeddings=embeddings,
            confidence_scores=confidence_scores,
            timestamp=time.time()
        )

    def _generate_scene_description(self, objects: List[Dict[str, Any]]) -> str:
        """Generate a natural language description of the scene"""
        if not objects:
            return "The scene appears to be empty."

        # Count objects by category
        obj_counts = {}
        for obj in objects:
            label = obj["label"]
            obj_counts[label] = obj_counts.get(label, 0) + 1

        # Create description
        parts = []
        for label, count in obj_counts.items():
            if count == 1:
                parts.append(f"a {label}")
            else:
                parts.append(f"{count} {label}s")

        if len(parts) == 1:
            return f"The scene contains {parts[0]}."
        elif len(parts) == 2:
            return f"The scene contains {parts[0]} and {parts[1]}."
        else:
            return f"The scene contains {', '.join(parts[:-1])}, and {parts[-1]}."

    def _answer_queries(self, image_tensor: torch.Tensor, queries: List[str]) -> str:
        """Answer text queries about the image"""
        # For simplicity, return a mock response
        # In practice, you'd use a more sophisticated approach
        if len(queries) == 1:
            return f"Based on the image, I can see objects that might relate to: {queries[0]}"
        else:
            return f"I can see objects that might relate to: {', '.join(queries)}"

    def _generate_embeddings(self, image_tensor: torch.Tensor,
                           objects: List[Dict[str, Any]],
                           texts: List[str]) -> Dict[str, torch.Tensor]:
        """Generate multimodal embeddings"""
        embeddings = {}

        # Image embedding
        with torch.no_grad():
            image_features = self.vision_language_model.encode_image(image_tensor)
            embeddings["image"] = image_features

        # Object embeddings
        for i, obj in enumerate(objects):
            embeddings[f"object_{i}"] = obj["embedding"]

        # Text embeddings
        if texts:
            text_features = self.vision_language_model.encode_text(texts)
            embeddings["text"] = text_features

        return embeddings

    def _compute_confidence_scores(self, objects: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute confidence scores for detections"""
        scores = {}
        for i, obj in enumerate(objects):
            scores[f"object_{i}_{obj['label']}"] = obj["confidence"]
        return scores

    def find_objects_by_description(self, image: np.ndarray, description: str) -> List[Dict[str, Any]]:
        """Find objects in image that match a text description"""
        # Process the image
        perception_output = self.process_image(image, [description])

        # Use the vision-language model to match description to objects
        image_tensor = self.transform(image).unsqueeze(0)
        text_features = self.vision_language_model.encode_text([description])

        matched_objects = []
        for obj in perception_output.objects:
            # Compute similarity between object embedding and text description
            obj_embedding = obj["embedding"].unsqueeze(0)
            similarity = torch.cosine_similarity(text_features, obj_embedding, dim=1)

            if similarity.item() > 0.3:  # Threshold for matching
                obj_copy = obj.copy()
                obj_copy["match_score"] = similarity.item()
                matched_objects.append(obj_copy)

        return matched_objects

def create_multimodal_perception_system(model_type: str = "clip") -> MultimodalPerceptionSystem:
    """Factory function to create a multimodal perception system"""
    return MultimodalPerceptionSystem(model_type)
```

### Sensor Fusion for Multimodal Perception

```python
#!/usr/bin/env python3
# sensor_fusion.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class SensorReading:
    """A single sensor reading"""
    sensor_type: str
    data: torch.Tensor
    timestamp: float
    confidence: float

@dataclass
class FusedPerceptionOutput:
    """Output from the sensor fusion system"""
    fused_features: torch.Tensor
    certainty_scores: Dict[str, float]
    temporal_consistency: float
    sensor_contributions: Dict[str, float]
    anomalies: List[Dict[str, Any]]

class KalmanFilter:
    """
    Kalman filter for sensor fusion and state estimation
    """

    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Initialize state and covariance matrices
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim) * 1000  # High initial uncertainty

        # Process and measurement noise
        self.Q = np.eye(state_dim) * 0.1  # Process noise
        self.R = np.eye(measurement_dim) * 1.0  # Measurement noise

        # Identity matrix
        self.I = np.eye(state_dim)

    def predict(self, F: np.ndarray, B: np.ndarray = None, u: np.ndarray = None):
        """Predict next state"""
        # State transition
        if B is not None and u is not None:
            self.state = F @ self.state + B @ u
        else:
            self.state = F @ self.state

        # Covariance prediction
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, measurement: np.ndarray, H: np.ndarray):
        """Update state with measurement"""
        # Innovation
        innovation = measurement - H @ self.state
        innovation_covariance = H @ self.covariance @ H.T + self.R

        # Kalman gain
        kalman_gain = self.covariance @ H.T @ np.linalg.inv(innovation_covariance)

        # Update state and covariance
        self.state = self.state + kalman_gain @ innovation
        self.covariance = (self.I - kalman_gain @ H) @ self.covariance

class ParticleFilter:
    """
    Particle filter for non-linear, non-Gaussian state estimation
    """

    def __init__(self, state_dim: int, num_particles: int = 1000):
        self.state_dim = state_dim
        self.num_particles = num_particles

        # Initialize particles randomly
        self.particles = np.random.randn(num_particles, state_dim) * 10
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, process_noise: float = 0.1):
        """Predict particle states"""
        noise = np.random.randn(self.num_particles, self.state_dim) * process_noise
        self.particles += noise

    def update(self, measurement: np.ndarray, measurement_function, measurement_noise: float = 1.0):
        """Update particle weights based on measurement"""
        # Compute likelihood of each particle given measurement
        for i in range(self.num_particles):
            predicted_measurement = measurement_function(self.particles[i])
            likelihood = self._gaussian_likelihood(measurement, predicted_measurement, measurement_noise)
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1e-300  # Prevent numerical issues
        self.weights /= np.sum(self.weights)

        # Resample if effective sample size is low
        effective_samples = 1.0 / np.sum(self.weights ** 2)
        if effective_samples < self.num_particles / 2:
            self._resample()

    def _gaussian_likelihood(self, measurement: np.ndarray, predicted: np.ndarray, noise: float):
        """Compute Gaussian likelihood"""
        diff = measurement - predicted
        return np.exp(-0.5 * np.sum(diff ** 2) / noise ** 2)

    def _resample(self):
        """Resample particles based on weights"""
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self) -> np.ndarray:
        """Estimate state as weighted average of particles"""
        return np.average(self.particles, axis=0, weights=self.weights)

class SensorFusionNetwork(nn.Module):
    """
    Neural network for sensor fusion
    """

    def __init__(self, sensor_configs: Dict[str, int]):
        super().__init__()
        self.sensor_configs = sensor_configs
        self.num_sensors = len(sensor_configs)

        # Sensor-specific encoders
        self.encoders = nn.ModuleDict()
        for sensor_type, input_dim in sensor_configs.items():
            self.encoders[sensor_type] = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(32 * self.num_sensors, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(64, 1)

    def forward(self, sensor_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse sensor inputs"""
        encoded_features = []

        for sensor_type, encoder in self.encoders.items():
            if sensor_type in sensor_inputs:
                encoded = encoder(sensor_inputs[sensor_type])
                encoded_features.append(encoded)
            else:
                # Use zero tensor if sensor data is missing
                batch_size = next(iter(sensor_inputs.values())).size(0)
                encoded_features.append(torch.zeros(batch_size, 32))

        # Concatenate all encoded features
        concat_features = torch.cat(encoded_features, dim=1)

        # Fuse features
        fused_features = self.fusion_layer(concat_features)

        return fused_features

class MultimodalSensorFusion:
    """
    Multimodal sensor fusion system for humanoid robotics
    """

    def __init__(self, sensor_configs: Dict[str, int]):
        self.sensor_configs = sensor_configs
        self.fusion_network = SensorFusionNetwork(sensor_configs)

        # Temporal consistency tracking
        self.temporal_buffer = {}
        self.buffer_size = 10

        # Anomaly detection
        self.anomaly_threshold = 0.8

        # State estimators
        self.kalman_filters = {}
        self.particle_filters = {}

    def register_kalman_filter(self, sensor_type: str, state_dim: int, measurement_dim: int):
        """Register a Kalman filter for a sensor"""
        self.kalman_filters[sensor_type] = KalmanFilter(state_dim, measurement_dim)

    def register_particle_filter(self, sensor_type: str, state_dim: int, num_particles: int = 1000):
        """Register a particle filter for a sensor"""
        self.particle_filters[sensor_type] = ParticleFilter(state_dim, num_particles)

    def process_sensor_readings(self, readings: List[SensorReading]) -> FusedPerceptionOutput:
        """Process multiple sensor readings and fuse them"""
        # Organize readings by sensor type
        sensor_data = {}
        for reading in readings:
            if reading.sensor_type not in sensor_data:
                sensor_data[reading.sensor_type] = []
            sensor_data[reading.sensor_type].append(reading)

        # Prepare input for fusion network
        network_inputs = {}
        for sensor_type, readings_list in sensor_data.items():
            # Take the most recent reading for each sensor type
            latest_reading = max(readings_list, key=lambda r: r.timestamp)

            if sensor_type in self.sensor_configs:
                network_inputs[sensor_type] = latest_reading.data

        # Fuse using neural network
        fused_features = self.fusion_network(network_inputs)

        # Compute certainty scores
        certainty_scores = self._compute_certainty_scores(network_inputs)

        # Update temporal consistency
        temporal_consistency = self._update_temporal_consistency(network_inputs)

        # Compute sensor contributions
        sensor_contributions = self._compute_sensor_contributions(network_inputs)

        # Detect anomalies
        anomalies = self._detect_anomalies(network_inputs)

        return FusedPerceptionOutput(
            fused_features=fused_features,
            certainty_scores=certainty_scores,
            temporal_consistency=temporal_consistency,
            sensor_contributions=sensor_contributions,
            anomalies=anomalies
        )

    def _compute_certainty_scores(self, sensor_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute certainty scores for each sensor input"""
        scores = {}
        for sensor_type, data in sensor_inputs.items():
            # Compute a simple variance-based certainty score
            variance = torch.var(data).item()
            # Lower variance means higher certainty
            certainty = 1.0 / (1.0 + variance)
            scores[sensor_type] = certainty
        return scores

    def _update_temporal_consistency(self, sensor_inputs: Dict[str, torch.Tensor]) -> float:
        """Update and compute temporal consistency"""
        current_time = time.time()

        for sensor_type, data in sensor_inputs.items():
            if sensor_type not in self.temporal_buffer:
                self.temporal_buffer[sensor_type] = []

            # Add current data to buffer
            self.temporal_buffer[sensor_type].append({
                'data': data.clone(),
                'timestamp': current_time
            })

            # Keep only recent data
            self.temporal_buffer[sensor_type] = [
                item for item in self.temporal_buffer[sensor_type]
                if current_time - item['timestamp'] < 5.0  # Keep last 5 seconds
            ]

        # Compute temporal consistency as average correlation over time
        consistency_sum = 0.0
        consistency_count = 0

        for sensor_type, buffer in self.temporal_buffer.items():
            if len(buffer) > 1:
                # Compute correlation between consecutive readings
                for i in range(1, len(buffer)):
                    prev_data = buffer[i-1]['data']
                    curr_data = buffer[i]['data']

                    # Compute cosine similarity
                    cos_sim = torch.cosine_similarity(prev_data.flatten(), curr_data.flatten(), dim=0)
                    consistency_sum += cos_sim.item()
                    consistency_count += 1

        return consistency_sum / consistency_count if consistency_count > 0 else 1.0

    def _compute_sensor_contributions(self, sensor_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute the contribution of each sensor to the fused output"""
        contributions = {}

        # For each sensor, temporarily remove it and see how much the output changes
        with torch.no_grad():
            # Get baseline fused output
            baseline_output = self.fusion_network(sensor_inputs)

            for sensor_type in sensor_inputs.keys():
                # Remove this sensor and get new output
                temp_inputs = {k: v for k, v in sensor_inputs.items() if k != sensor_type}

                if temp_inputs:  # Only if there are other sensors
                    modified_output = self.fusion_network(temp_inputs)

                    # Compute difference as contribution measure
                    diff = torch.norm(baseline_output - modified_output)
                    contributions[sensor_type] = diff.item()
                else:
                    # If removing this sensor leaves no inputs, assign high contribution
                    contributions[sensor_type] = 1.0

        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            for sensor_type in contributions:
                contributions[sensor_type] /= total

        return contributions

    def _detect_anomalies(self, sensor_inputs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Detect anomalous sensor readings"""
        anomalies = []

        for sensor_type, data in sensor_inputs.items():
            # Check for extreme values
            mean_val = torch.mean(data)
            std_val = torch.std(data)
            threshold = mean_val + 3 * std_val  # 3-sigma rule

            if torch.any(torch.abs(data) > threshold):
                anomalies.append({
                    'sensor_type': sensor_type,
                    'anomaly_type': 'outlier',
                    'timestamp': time.time(),
                    'severity': float(torch.max(torch.abs(data) / threshold))
                })

        return anomalies

def create_sensor_fusion_system(sensor_configs: Dict[str, int]) -> MultimodalSensorFusion:
    """Factory function to create a sensor fusion system"""
    return MultimodalSensorFusion(sensor_configs)
```

## Examples

### Example 1: Object Recognition and Manipulation Planning

```python
#!/usr/bin/env python3
# object_recognition_manipulation.py

import numpy as np
import torch
import cv2
from typing import Dict, Any, List
import time

class ObjectRecognitionManipulationSystem:
    """
    System that combines object recognition with manipulation planning
    """

    def __init__(self):
        self.perception_system = create_multimodal_perception_system("clip")
        self.fusion_system = create_sensor_fusion_system({
            "camera": 3 * 224 * 224,  # RGB image
            "depth": 224 * 224,      # Depth map
            "imu": 6,                # Accelerometer + gyroscope
            "force": 6               # Force/torque sensors
        })

        # Register filters for temporal consistency
        self.fusion_system.register_kalman_filter("camera", 10, 10)  # Simplified state
        self.fusion_system.register_particle_filter("force", 6)

    def process_scene_for_manipulation(self, rgb_image: np.ndarray,
                                     depth_map: np.ndarray,
                                     object_query: str) -> Dict[str, Any]:
        """Process scene to identify objects for manipulation"""

        # Use multimodal perception to find target object
        target_objects = self.perception_system.find_objects_by_description(
            rgb_image, object_query
        )

        if not target_objects:
            return {
                "success": False,
                "message": f"No objects matching '{object_query}' found",
                "candidate_objects": [],
                "manipulation_plan": None
            }

        # Get depth information for 3D positioning
        depth_info = self._extract_depth_info(depth_map, target_objects)

        # Generate manipulation plan
        manipulation_plan = self._generate_manipulation_plan(target_objects, depth_info)

        return {
            "success": True,
            "message": f"Found {len(target_objects)} objects matching '{object_query}'",
            "candidate_objects": target_objects,
            "manipulation_plan": manipulation_plan,
            "depth_info": depth_info
        }

    def _extract_depth_info(self, depth_map: np.ndarray,
                          objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract 3D information from depth map for detected objects"""
        depth_info = {}

        for obj in objects:
            bbox = obj["bbox"]  # [x, y, width, height] - normalized

            # Convert to pixel coordinates
            h, w = depth_map.shape
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int((bbox[0] + bbox[2]) * w)
            y2 = int((bbox[1] + bbox[3]) * h)

            # Ensure bounds are within image
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 > x1 and y2 > y1:
                # Extract depth region
                obj_depth_region = depth_map[y1:y2, x1:x2]

                # Compute average depth (distance to object)
                avg_depth = np.nanmedian(obj_depth_region[~np.isnan(obj_depth_region)])

                # Compute 3D center point
                center_x = x1 + (x2 - x1) / 2
                center_y = y1 + (y2 - y1) / 2

                depth_info[obj["label"]] = {
                    "distance": avg_depth,
                    "center_2d": (center_x, center_y),
                    "bbox_3d": [x1, y1, x2, y2],
                    "confidence": obj["confidence"]
                }

        return depth_info

    def _generate_manipulation_plan(self, objects: List[Dict[str, Any]],
                                  depth_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a manipulation plan for the detected objects"""
        plan = []

        for obj in objects:
            obj_label = obj["label"]
            if obj_label in depth_info:
                depth_data = depth_info[obj_label]

                # Generate approach plan
                approach_action = {
                    "action": "approach_object",
                    "target_object": obj_label,
                    "position_3d": self._pixel_to_3d(depth_data["center_2d"], depth_data["distance"]),
                    "grasp_strategy": self._select_grasp_strategy(obj_label),
                    "estimated_time": 2.0  # seconds
                }

                # Generate grasp plan
                grasp_action = {
                    "action": "grasp_object",
                    "target_object": obj_label,
                    "gripper_width": self._estimate_gripper_width(obj_label),
                    "force_limit": self._estimate_force_limit(obj_label),
                    "estimated_time": 1.5
                }

                # Generate lift plan
                lift_action = {
                    "action": "lift_object",
                    "height_offset": 0.1,  # Lift 10cm
                    "estimated_time": 1.0
                }

                plan.extend([approach_action, grasp_action, lift_action])

        return plan

    def _pixel_to_3d(self, pixel_coords: Tuple[float, float], depth: float) -> List[float]:
        """Convert 2D pixel coordinates to 3D world coordinates (simplified)"""
        # This is a simplified transformation
        # In practice, you'd use camera intrinsic parameters
        px, py = pixel_coords
        fx, fy = 500.0, 500.0  # Focal lengths (typical values)
        cx, cy = 320.0, 240.0  # Principal point (typical for 640x480)

        # Convert to 3D assuming pinhole camera model
        x = (px - cx) * depth / fx
        y = (py - cy) * depth / fy
        z = depth

        return [x, y, z]

    def _select_grasp_strategy(self, object_label: str) -> str:
        """Select appropriate grasp strategy based on object type"""
        grasp_strategies = {
            "cup": "top_grasp",
            "bottle": "side_grasp",
            "box": "corner_grasp",
            "sphere": "enclosing_grasp",
            "cylinder": "side_grasp"
        }

        return grasp_strategies.get(object_label, "power_grasp")

    def _estimate_gripper_width(self, object_label: str) -> float:
        """Estimate appropriate gripper width for object"""
        width_estimates = {
            "cup": 0.08,      # 8cm
            "bottle": 0.06,   # 6cm
            "box": 0.10,      # 10cm
            "sphere": 0.07,   # 7cm
            "cylinder": 0.05  # 5cm
        }

        return width_estimates.get(object_label, 0.05)  # Default 5cm

    def _estimate_force_limit(self, object_label: str) -> float:
        """Estimate appropriate force limit for grasping"""
        force_limits = {
            "cup": 10.0,      # Light object
            "bottle": 15.0,   # Medium weight
            "box": 20.0,      # Heavier
            "sphere": 8.0,    # Fragile
            "cylinder": 12.0  # Medium weight
        }

        return force_limits.get(object_label, 10.0)  # Default 10N

    def integrate_sensor_data(self, sensor_readings: List[SensorReading]) -> Dict[str, Any]:
        """Integrate multiple sensor readings for robust perception"""
        fusion_output = self.fusion_system.process_sensor_readings(sensor_readings)

        # Combine with visual perception
        result = {
            "fused_features": fusion_output.fused_features,
            "certainty_scores": fusion_output.certainty_scores,
            "temporal_consistency": fusion_output.temporal_consistency,
            "sensor_contributions": fusion_output.sensor_contributions,
            "anomalies": fusion_output.anomalies
        }

        return result

def simulate_sensor_readings() -> List[SensorReading]:
    """Simulate sensor readings for testing"""
    readings = []

    # Simulate camera reading
    camera_data = torch.randn(3, 224, 224)  # RGB image
    readings.append(SensorReading(
        sensor_type="camera",
        data=camera_data,
        timestamp=time.time(),
        confidence=0.9
    ))

    # Simulate depth reading
    depth_data = torch.randn(224, 224)  # Depth map
    readings.append(SensorReading(
        sensor_type="depth",
        data=depth_data,
        timestamp=time.time(),
        confidence=0.85
    ))

    # Simulate IMU reading
    imu_data = torch.randn(6)  # Accelerometer + gyroscope
    readings.append(SensorReading(
        sensor_type="imu",
        data=imu_data,
        timestamp=time.time(),
        confidence=0.95
    ))

    # Simulate force sensor reading
    force_data = torch.randn(6)  # Force/torque
    readings.append(SensorReading(
        sensor_type="force",
        data=force_data,
        timestamp=time.time(),
        confidence=0.9
    ))

    return readings

def main():
    """Main function to demonstrate multimodal perception"""
    print("Initializing Multimodal Perception System...")

    # Create the system
    system = ObjectRecognitionManipulationSystem()

    # Simulate an RGB image (in practice, this would come from a camera)
    dummy_rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Simulate a depth map (in practice, this would come from a depth sensor)
    dummy_depth_map = np.random.rand(480, 640).astype(np.float32) * 3.0  # 0-3 meters

    # Process a scene to find a specific object
    query = "red cup"
    result = system.process_scene_for_manipulation(dummy_rgb_image, dummy_depth_map, query)

    print(f"Object recognition result: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Number of candidate objects: {len(result['candidate_objects'])}")
    print(f"Manipulation plan steps: {len(result['manipulation_plan']) if result['manipulation_plan'] else 0}")

    # Simulate sensor fusion
    sensor_readings = simulate_sensor_readings()
    fusion_result = system.integrate_sensor_data(sensor_readings)

    print(f"Sensor fusion completed with {len(fusion_result['certainty_scores'])} sensor types")
    print(f"Temporal consistency: {fusion_result['temporal_consistency']:.3f}")
    print(f"Detected anomalies: {len(fusion_result['anomalies'])}")

    print("\nMultimodal perception system demonstration completed!")
    print("In a real implementation, this would connect to actual robot sensors and perception systems.")

if __name__ == "__main__":
    main()
```

### Example 2: Scene Understanding and Navigation Planning

```python
#!/usr/bin/env python3
# scene_understanding_navigation.py

import numpy as np
import torch
from typing import Dict, Any, List, Tuple
import time

class SceneUnderstandingNavigationSystem:
    """
    System that combines scene understanding with navigation planning
    """

    def __init__(self):
        self.perception_system = create_multimodal_perception_system("clip")
        self.fusion_system = create_sensor_fusion_system({
            "camera": 3 * 224 * 224,
            "lidar": 360,  # Simplified LIDAR data
            "imu": 6,
            "odometry": 3
        })

    def analyze_scene_for_navigation(self, rgb_image: np.ndarray,
                                   lidar_scan: np.ndarray,
                                   navigation_goal: str) -> Dict[str, Any]:
        """Analyze scene to plan navigation to a goal location"""

        # Use multimodal perception to understand the scene
        perception_output = self.perception_system.process_image(rgb_image, [navigation_goal])

        # Integrate with LIDAR data for obstacle detection
        obstacles = self._process_lidar_data(lidar_scan)

        # Combine visual and LIDAR information
        combined_analysis = self._combine_visual_lidar(perception_output, obstacles)

        # Generate navigation plan
        navigation_plan = self._generate_navigation_plan(combined_analysis, navigation_goal)

        return {
            "success": True,
            "scene_analysis": combined_analysis,
            "navigation_plan": navigation_plan,
            "obstacles": obstacles,
            "spatial_relationships": perception_output.spatial_relationships
        }

    def _process_lidar_data(self, lidar_scan: np.ndarray) -> List[Dict[str, Any]]:
        """Process LIDAR scan to detect obstacles"""
        obstacles = []

        # Simple threshold-based obstacle detection
        distance_threshold = 1.0  # meters
        min_points = 3  # Minimum points to consider as obstacle

        current_cluster = []
        for i, distance in enumerate(lidar_scan):
            if distance < distance_threshold and not np.isnan(distance):
                # Convert polar to Cartesian coordinates
                angle = (i / len(lidar_scan)) * 2 * np.pi
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)

                current_cluster.append((x, y))
            else:
                if len(current_cluster) >= min_points:
                    # Calculate cluster center and size
                    cluster_array = np.array(current_cluster)
                    center_x = np.mean(cluster_array[:, 0])
                    center_y = np.mean(cluster_array[:, 1])
                    size = np.std(cluster_array)

                    obstacles.append({
                        "type": "obstacle",
                        "center": (center_x, center_y),
                        "size": size,
                        "points": current_cluster.copy()
                    })

                current_cluster = []

        # Handle the last cluster if it exists
        if len(current_cluster) >= min_points:
            cluster_array = np.array(current_cluster)
            center_x = np.mean(cluster_array[:, 0])
            center_y = np.mean(cluster_array[:, 1])
            size = np.std(cluster_array)

            obstacles.append({
                "type": "obstacle",
                "center": (center_x, center_y),
                "size": size,
                "points": current_cluster
            })

        return obstacles

    def _combine_visual_lidar(self, perception_output: PerceptionOutput,
                            lidar_obstacles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine visual perception with LIDAR data"""
        combined = {
            "visual_objects": perception_output.objects,
            "lidar_obstacles": lidar_obstacles,
            "spatial_relationships": perception_output.spatial_relationships,
            "scene_description": perception_output.scene_description,
            "combined_map": self._create_combined_map(perception_output, lidar_obstacles)
        }

        return combined

    def _create_combined_map(self, perception_output: PerceptionOutput,
                           lidar_obstacles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a combined occupancy map from visual and LIDAR data"""
        # This would typically create a 2D or 3D occupancy grid
        # For this example, we'll return a simplified representation

        map_data = {
            "free_space": [],
            "occupied_space": [],
            "object_locations": {},
            "obstacle_locations": []
        }

        # Add object locations from visual perception
        for obj in perception_output.objects:
            # This would convert from image coordinates to world coordinates
            # For now, we'll just store the label and confidence
            map_data["object_locations"][obj["label"]] = {
                "confidence": obj["confidence"],
                "bbox": obj["bbox"]
            }

        # Add obstacle locations from LIDAR
        for obstacle in lidar_obstacles:
            map_data["obstacle_locations"].append(obstacle["center"])

        return map_data

    def _generate_navigation_plan(self, scene_analysis: Dict[str, Any],
                                goal_description: str) -> List[Dict[str, Any]]:
        """Generate a navigation plan based on scene analysis"""
        plan = []

        # Analyze the goal description to find relevant objects/locations
        relevant_objects = self._find_relevant_objects(scene_analysis, goal_description)

        if relevant_objects:
            # Create navigation steps to approach relevant objects
            for obj_info in relevant_objects:
                approach_step = {
                    "action": "navigate_to_object",
                    "target": obj_info["label"],
                    "position": self._estimate_object_position(obj_info),
                    "estimated_distance": obj_info.get("estimated_distance", 1.0),
                    "estimated_time": 5.0  # seconds
                }
                plan.append(approach_step)

                # Add inspection step
                inspect_step = {
                    "action": "inspect_object",
                    "target": obj_info["label"],
                    "estimated_time": 2.0
                }
                plan.append(inspect_step)
        else:
            # If no specific objects found, create a general exploration plan
            exploration_step = {
                "action": "explore_towards_goal",
                "goal_description": goal_description,
                "estimated_time": 10.0
            }
            plan.append(exploration_step)

        # Add safety checks
        safety_check = {
            "action": "check_surroundings",
            "estimated_time": 1.0
        }
        plan.append(safety_check)

        return plan

    def _find_relevant_objects(self, scene_analysis: Dict[str, Any],
                             goal_description: str) -> List[Dict[str, Any]]:
        """Find objects that are relevant to the navigation goal"""
        relevant_objects = []

        # Simple keyword matching approach
        goal_lower = goal_description.lower()

        for obj in scene_analysis["visual_objects"]:
            obj_label = obj["label"].lower()
            if obj_label in goal_lower or any(keyword in goal_lower for keyword in [obj_label, "near", "by", "at"]):
                relevant_objects.append({
                    "label": obj["label"],
                    "confidence": obj["confidence"],
                    "bbox": obj["bbox"]
                })

        return relevant_objects

    def _estimate_object_position(self, obj_info: Dict[str, Any]) -> List[float]:
        """Estimate 3D position of an object (simplified)"""
        # This would normally use depth information
        # For now, return a placeholder position
        return [1.0, 0.5, 0.0]  # x, y, z in meters

    def update_navigation_map(self, sensor_readings: List[SensorReading]) -> Dict[str, Any]:
        """Update navigation map with new sensor data"""
        fusion_output = self.fusion_system.process_sensor_readings(sensor_readings)

        # Update the navigation map based on fused sensor data
        updated_map = {
            "timestamp": time.time(),
            "fused_features": fusion_output.fused_features,
            "updated_obstacles": self._detect_new_obstacles(fusion_output),
            "certainty_map": fusion_output.certainty_scores,
            "temporal_consistency": fusion_output.temporal_consistency
        }

        return updated_map

    def _detect_new_obstacles(self, fusion_output: FusedPerceptionOutput) -> List[Dict[str, Any]]:
        """Detect new obstacles from fused sensor data"""
        # This would analyze the fused features to identify new obstacles
        # For this example, we'll return mock data
        return [
            {"position": [2.0, 1.5, 0.0], "size": 0.3, "confidence": 0.9},
            {"position": [0.5, 2.0, 0.0], "size": 0.5, "confidence": 0.8}
        ]

def simulate_navigation_scenario():
    """Simulate a navigation scenario"""
    system = SceneUnderstandingNavigationSystem()

    # Simulate sensor data
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_lidar = np.random.rand(360).astype(np.float32) * 5.0  # 0-5 meters

    # Analyze scene for navigation to a specific goal
    goal = "navigate to the kitchen counter"
    result = system.analyze_scene_for_navigation(dummy_image, dummy_lidar, goal)

    print(f"Navigation analysis completed:")
    print(f"- Objects detected: {len(result['scene_analysis']['visual_objects'])}")
    print(f"- LIDAR obstacles: {len(result['obstacles'])}")
    print(f"- Navigation plan steps: {len(result['navigation_plan'])}")
    print(f"- Spatial relationships: {len(result['spatial_relationships'])}")

    # Simulate updating navigation map with new sensor readings
    sensor_readings = simulate_sensor_readings()  # From previous example
    map_update = system.update_navigation_map(sensor_readings)

    print(f"\nNavigation map updated:")
    print(f"- New obstacles detected: {len(map_update['updated_obstacles'])}")
    print(f"- Certainty scores: {list(map_update['certainty_map'].keys())}")
    print(f"- Temporal consistency: {map_update['temporal_consistency']:.3f}")

def main():
    """Main function for scene understanding and navigation"""
    print("Initializing Scene Understanding and Navigation System...")

    simulate_navigation_scenario()

    print("\nScene understanding and navigation system demonstrated!")
    print("This system can analyze scenes and generate navigation plans based on multimodal perception.")

if __name__ == "__main__":
    main()
```

## Summary

Multimodal perception systems integrate multiple sensory inputs to create a comprehensive understanding of the environment for humanoid robots. Key components include:

- **Vision-Language Integration**: Combining visual and textual information for object recognition and scene understanding
- **Sensor Fusion**: Integrating data from multiple sensors (cameras, LIDAR, IMU, etc.) for robust perception
- **Spatial Reasoning**: Understanding spatial relationships between objects and the environment
- **Temporal Consistency**: Maintaining coherent understanding across time
- **Anomaly Detection**: Identifying unexpected or anomalous sensor readings

The effectiveness of these systems depends on proper integration of different modalities and robust handling of sensor noise and uncertainty.

## Exercises

### Conceptual
1. Compare and contrast different sensor fusion approaches (Kalman filtering, particle filtering, neural network fusion). What are the trade-offs of each approach for humanoid robotics?

### Logical
1. Design a multimodal perception system that can handle sensor failures gracefully. How would your system maintain functionality when one or more sensors become unavailable?

### Implementation
1. Implement a complete multimodal perception pipeline that integrates vision, LIDAR, and IMU data for object recognition and navigation planning in a humanoid robot simulation.
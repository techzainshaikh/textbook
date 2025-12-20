---
title: Speech Recognition for Robotics with OpenAI Whisper
sidebar_position: 3
description: Implementing speech recognition systems for humanoid robots using OpenAI Whisper and other speech-to-text technologies
keywords: [speech recognition, OpenAI Whisper, speech-to-text, robotics, natural language processing, humanoid AI]
---

# Chapter 2: Speech Recognition for Robotics with OpenAI Whisper

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement speech recognition systems for humanoid robot command input
- Integrate OpenAI Whisper for accurate speech-to-text conversion
- Process and validate speech commands for robotic applications
- Handle speech recognition errors and uncertainties in robot systems
- Design robust speech interfaces for noisy environments

## Prerequisites

Students should have:
- Understanding of basic signal processing concepts
- Knowledge of Python programming and audio processing libraries
- Familiarity with natural language processing fundamentals
- Basic understanding of robotics command interfaces (covered in Module 1)
- Experience with machine learning frameworks (PyTorch/TensorFlow)

## Core Concepts

Speech recognition systems enable humanoid robots to understand natural language commands through spoken input. Modern approaches leverage deep learning models to convert audio signals into text, which can then be processed by language understanding systems.

### Speech Recognition Pipeline

**Audio Input:**
- **Microphone Array**: Multiple microphones for noise reduction and directionality
- **Audio Preprocessing**: Filtering, noise reduction, and signal enhancement
- **Feature Extraction**: Mel-spectrograms, MFCCs, or raw waveform processing

**Model Processing:**
- **Acoustic Model**: Maps audio features to phonemes or subword units
- **Language Model**: Provides linguistic context for word prediction
- **Decoder**: Combines acoustic and language models to generate text

**Post-Processing:**
- **Text Refinement**: Grammar correction and context-based validation
- **Command Extraction**: Identifying actionable commands from speech
- **Confidence Scoring**: Assessing recognition quality and reliability

### OpenAI Whisper Architecture

Whisper represents a significant advancement in speech recognition, using a large-scale transformer-based approach:

- **Encoder**: Processes audio spectrograms using transformer layers
- **Decoder**: Generates text tokens conditioned on audio context
- **Multilingual Capability**: Trained on multiple languages simultaneously
- **Robustness**: Performs well across diverse accents and audio conditions

## Implementation

Let's implement speech recognition for humanoid robotics using OpenAI Whisper:

### Whisper-Based Speech Recognition System

```python
#!/usr/bin/env python3
# whisper_speech_recognition.py

import torch
import whisper
import numpy as np
import librosa
import pyaudio
import wave
import threading
import queue
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import asyncio

@dataclass
class SpeechRecognitionResult:
    """Result from speech recognition"""
    text: str
    confidence: float
    language: str
    timestamp: float
    audio_duration: float
    raw_transcription: Dict[str, Any]  # Raw Whisper output

class AudioCapture:
    """
    Audio capture system for speech recognition
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.audio_queue = queue.Queue()

    def start_recording(self):
        """Start audio recording"""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        self.recording = True

        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()

    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False
        if self.record_thread:
            self.record_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _record_audio(self):
        """Internal recording loop"""
        while self.recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_chunk)
            except Exception as e:
                print(f"Audio recording error: {e}")
                break

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get an audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_audio_segment(self, duration: float = 3.0) -> np.ndarray:
        """Record a specific duration of audio"""
        frames = []
        samples_to_record = int(self.sample_rate * duration)
        chunks_to_record = int(samples_to_record / self.chunk_size)

        for _ in range(chunks_to_record):
            chunk = self.get_audio_chunk()
            if chunk is not None:
                frames.append(chunk)

        if frames:
            return np.concatenate(frames)
        else:
            return np.array([])

class WhisperSpeechRecognizer:
    """
    Speech recognition using OpenAI Whisper
    """

    def __init__(self, model_size: str = "base", device: str = None):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Whisper model
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size).to(self.device)

        # Audio capture system
        self.audio_capture = AudioCapture()

    def preprocess_audio(self, audio: np.ndarray, target_sr: int = 16000) -> np.ndarray:
        """Preprocess audio for Whisper"""
        # Ensure audio is in the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed
        if target_sr != 16000:  # Whisper expects 16kHz
            audio = librosa.resample(audio, orig_sr=target_sr, target_sr=16000)

        # Normalize audio
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

        return audio

    def transcribe_audio(self, audio: np.ndarray, language: str = "en") -> SpeechRecognitionResult:
        """Transcribe audio using Whisper"""
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio)

        # Transcribe using Whisper
        result = self.model.transcribe(
            processed_audio,
            language=language,
            temperature=0.0,  # Deterministic output
            best_of=1,
            fp16=(self.device == "cuda")
        )

        # Calculate confidence (Whisper doesn't provide confidence, so we estimate)
        confidence = self._estimate_confidence(result)

        return SpeechRecognitionResult(
            text=result["text"],
            confidence=confidence,
            language=language,
            timestamp=time.time(),
            audio_duration=len(processed_audio) / 16000,  # 16kHz sample rate
            raw_transcription=result
        )

    def _estimate_confidence(self, transcription: Dict[str, Any]) -> float:
        """Estimate confidence from Whisper output"""
        # Whisper doesn't provide confidence scores directly
        # We can estimate based on the length and complexity of the transcription
        text = transcription.get("text", "")

        if not text.strip():
            return 0.0

        # Simple heuristic: longer, more complex text is more likely to be confident
        # In practice, you might use other metrics or implement custom confidence estimation
        return min(0.9, len(text.strip().split()) * 0.1 + 0.5)

    def continuous_recognition(self, callback_func, silence_threshold: float = 0.01,
                             silence_duration: float = 2.0) -> None:
        """Continuous speech recognition with silence detection"""
        self.audio_capture.start_recording()

        audio_buffer = np.array([])
        silence_start = None
        sample_rate = self.audio_capture.sample_rate

        try:
            while True:
                chunk = self.audio_capture.get_audio_chunk(timeout=0.1)
                if chunk is not None:
                    audio_buffer = np.concatenate([audio_buffer, chunk])

                    # Check for silence
                    rms = np.sqrt(np.mean(chunk ** 2))

                    if rms < silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > silence_duration and len(audio_buffer) > sample_rate:
                            # Process the speech segment
                            if len(audio_buffer) > sample_rate:  # At least 1 second of audio
                                result = self.transcribe_audio(audio_buffer)
                                if result.confidence > 0.3:  # Minimum confidence threshold
                                    callback_func(result)

                            # Reset for next segment
                            audio_buffer = np.array([])
                            silence_start = None
                    else:
                        silence_start = None

        except KeyboardInterrupt:
            print("Stopping continuous recognition...")
        finally:
            self.audio_capture.stop_recording()

    def recognize_from_file(self, audio_file_path: str, language: str = "en") -> SpeechRecognitionResult:
        """Recognize speech from an audio file"""
        # Load audio file
        audio, sr = librosa.load(audio_file_path, sr=16000)  # Whisper expects 16kHz

        # Transcribe
        return self.transcribe_audio(audio, language)

    def recognize_from_microphone(self, duration: float = 5.0, language: str = "en") -> SpeechRecognitionResult:
        """Recognize speech from microphone input"""
        self.audio_capture.start_recording()
        time.sleep(0.1)  # Brief delay to ensure recording starts

        # Record audio
        audio = self.audio_capture.record_audio_segment(duration)

        self.audio_capture.stop_recording()

        if len(audio) == 0:
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                language=language,
                timestamp=time.time(),
                audio_duration=0.0,
                raw_transcription={}
            )

        # Transcribe
        return self.transcribe_audio(audio, language)

class SpeechCommandValidator:
    """
    Validate and process recognized speech commands
    """

    def __init__(self):
        # Define valid robot commands and their patterns
        self.valid_commands = {
            "navigation": [
                "go to", "move to", "navigate to", "walk to", "go", "move", "navigate"
            ],
            "manipulation": [
                "pick up", "grasp", "take", "lift", "place", "put", "drop", "release"
            ],
            "interaction": [
                "hello", "hi", "greet", "introduce", "help", "stop", "wait", "continue"
            ]
        }

        # Object categories for manipulation
        self.object_categories = [
            "cube", "ball", "box", "bottle", "cup", "object", "item"
        ]

    def validate_command(self, text: str) -> Dict[str, Any]:
        """Validate if the recognized text is a valid robot command"""
        text_lower = text.lower().strip()

        validation_result = {
            "is_valid": False,
            "command_type": None,
            "extracted_entities": [],
            "intent": None,
            "confidence": 0.0
        }

        # Check for valid command patterns
        for cmd_type, patterns in self.valid_commands.items():
            for pattern in patterns:
                if pattern in text_lower:
                    validation_result["is_valid"] = True
                    validation_result["command_type"] = cmd_type
                    validation_result["intent"] = pattern

                    # Extract entities (objects, locations)
                    entities = self._extract_entities(text_lower, pattern)
                    validation_result["extracted_entities"] = entities

                    # Set confidence based on command clarity
                    validation_result["confidence"] = self._calculate_command_confidence(
                        text_lower, entities
                    )

                    break

        return validation_result

    def _extract_entities(self, text: str, command_pattern: str) -> List[Dict[str, str]]:
        """Extract entities (objects, locations) from command text"""
        entities = []

        # Remove the command pattern to get the object/location part
        remaining_text = text.replace(command_pattern, "").strip()

        # Extract object names
        for obj_cat in self.object_categories:
            if obj_cat in remaining_text:
                entities.append({
                    "type": "object",
                    "value": obj_cat,
                    "confidence": 0.8
                })

        # Extract location names
        location_keywords = ["table", "shelf", "desk", "floor", "box", "container"]
        for loc in location_keywords:
            if loc in remaining_text:
                entities.append({
                    "type": "location",
                    "value": loc,
                    "confidence": 0.8
                })

        return entities

    def _calculate_command_confidence(self, text: str, entities: List[Dict[str, str]]) -> float:
        """Calculate confidence in the command validity"""
        confidence = 0.5  # Base confidence

        # Increase confidence if entities are present
        if entities:
            confidence += 0.3

        # Increase confidence if command is specific
        if len(text.split()) > 2:
            confidence += 0.2

        return min(1.0, confidence)

class RobotSpeechInterface:
    """
    Complete speech interface for humanoid robots
    """

    def __init__(self, whisper_model_size: str = "base"):
        self.speech_recognizer = WhisperSpeechRecognizer(whisper_model_size)
        self.command_validator = SpeechCommandValidator()
        self.command_history = []
        self.is_listening = False

    def process_speech_command(self, audio_input: np.ndarray = None,
                             audio_file: str = None,
                             microphone_duration: float = 5.0) -> Dict[str, Any]:
        """Process speech command from various input sources"""

        # Get speech recognition result
        if audio_file:
            result = self.speech_recognizer.recognize_from_file(audio_file)
        elif audio_input is not None:
            result = self.speech_recognizer.transcribe_audio(audio_input)
        else:
            # Use microphone
            result = self.speech_recognizer.recognize_from_microphone(microphone_duration)

        # Validate the recognized command
        validation = self.command_validator.validate_command(result.text)

        # Create complete response
        response = {
            "recognition_result": result,
            "validation": validation,
            "can_execute": validation["is_valid"] and result.confidence > 0.5,
            "timestamp": time.time()
        }

        # Add to command history
        self.command_history.append(response)

        return response

    def start_continuous_listening(self, command_callback):
        """Start continuous listening for speech commands"""
        self.is_listening = True

        def callback_wrapper(result):
            if self.is_listening:
                response = self.process_speech_command(audio_input=result)
                if response["can_execute"]:
                    command_callback(response)

        # Start continuous recognition
        self.speech_recognizer.continuous_recognition(callback_wrapper)

    def stop_listening(self):
        """Stop continuous listening"""
        self.is_listening = False

    def get_recent_commands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent speech commands"""
        return self.command_history[-limit:]

def create_robot_speech_interface(model_size: str = "base") -> RobotSpeechInterface:
    """Factory function to create a robot speech interface"""
    return RobotSpeechInterface(model_size)

# Example usage and testing
def test_speech_recognition():
    """Test function for speech recognition"""
    print("Testing Whisper-based speech recognition...")

    # Create speech interface
    speech_interface = create_robot_speech_interface("base")

    # Test with sample text (in practice, you'd record from microphone or load audio file)
    # For demonstration, we'll simulate audio processing

    print("Speech recognition system initialized successfully!")
    print("You can now use this system to process speech commands for your humanoid robot.")

if __name__ == "__main__":
    test_speech_recognition()
```

### Advanced Speech Recognition Features

```python
#!/usr/bin/env python3
# advanced_speech_features.py

import torch
import whisper
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
import threading
from dataclasses import dataclass
import time

@dataclass
class EnhancedRecognitionResult:
    """Enhanced result with additional metadata"""
    text: str
    confidence: float
    language: str
    timestamp: float
    audio_duration: float
    word_timestamps: List[Dict[str, float]]  # Timestamps for each word
    detected_language: str
    temperature: float
    no_speech_prob: float

class MultilingualSpeechRecognizer:
    """
    Multilingual speech recognition with language detection
    """

    def __init__(self, model_size: str = "medium"):
        self.model = whisper.load_model(model_size)
        self.supported_languages = {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "zh": "chinese",
            "ja": "japanese",
            "ko": "korean"
        }

    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of the audio"""
        # Pad/trim audio to fit in memory
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        return detected_lang

    def transcribe_multilingual(self, audio: np.ndarray,
                              detect_language: bool = True) -> EnhancedRecognitionResult:
        """Transcribe audio with automatic language detection"""
        if detect_language:
            detected_lang = self.detect_language(audio)
        else:
            detected_lang = "en"  # Default to English

        # Transcribe with detected language
        result = self.model.transcribe(
            audio,
            language=detected_lang,
            temperature=0.0,
            word_timestamps=True
        )

        # Extract word-level timestamps if available
        word_timestamps = []
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        word_timestamps.append({
                            "text": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0)
                        })

        # Estimate confidence based on no_speech_prob
        no_speech_prob = result.get("no_speech_prob", 0.0)
        confidence = 1.0 - no_speech_prob

        return EnhancedRecognitionResult(
            text=result["text"],
            confidence=confidence,
            language=result.get("language", "unknown"),
            timestamp=time.time(),
            audio_duration=len(audio) / 16000,
            word_timestamps=word_timestamps,
            detected_language=detected_lang,
            temperature=0.0,
            no_speech_prob=no_speech_prob
        )

class NoiseRobustRecognizer:
    """
    Speech recognition optimized for noisy environments
    """

    def __init__(self, base_model_size: str = "base"):
        self.model = whisper.load_model(base_model_size)
        self.noise_threshold = 0.01  # RMS threshold for speech detection

    def preprocess_for_noise(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio to handle noise"""
        import librosa

        # Apply noise reduction
        # This is a simplified approach - in practice, you might use more sophisticated techniques
        reduced_audio = librosa.effects.percussive(audio)

        # Normalize
        reduced_audio = reduced_audio / max(np.max(reduced_audio), abs(np.min(reduced_audio))) if np.max(reduced_audio) != 0 else reduced_audio

        return reduced_audio

    def vad_segmentation(self, audio: np.ndarray, sample_rate: int = 16000) -> List[np.ndarray]:
        """Voice activity detection to segment speech from noise"""
        import librosa

        # Simple VAD based on energy
        frame_length = 2048
        hop_length = 512

        # Calculate frame energy
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)

        # Normalize energy
        normalized_energy = frame_energy / np.max(frame_energy) if np.max(frame_energy) > 0 else frame_energy

        # Identify speech frames (above threshold)
        speech_frames = normalized_energy > self.noise_threshold

        # Group consecutive speech frames into segments
        segments = []
        current_segment = []

        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                current_segment.append(i)
            else:
                if len(current_segment) > 0:
                    # Convert frame indices back to audio samples
                    start_frame = current_segment[0] * hop_length
                    end_frame = current_segment[-1] * hop_length + frame_length
                    segment_audio = audio[start_frame:end_frame]
                    segments.append(segment_audio)
                    current_segment = []

        # Add final segment if exists
        if len(current_segment) > 0:
            start_frame = current_segment[0] * hop_length
            end_frame = current_segment[-1] * hop_length + frame_length
            segment_audio = audio[start_frame:end_frame]
            segments.append(segment_audio)

        return segments

    def recognize_in_noise(self, audio: np.ndarray) -> List[EnhancedRecognitionResult]:
        """Recognize speech in noisy conditions"""
        # Preprocess audio
        clean_audio = self.preprocess_for_noise(audio)

        # Perform VAD segmentation
        segments = self.vad_segmentation(clean_audio)

        results = []
        for segment in segments:
            # Transcribe each segment
            result = self.model.transcribe(segment, temperature=0.0)

            # Create result object
            enhanced_result = EnhancedRecognitionResult(
                text=result["text"],
                confidence=0.8,  # Placeholder confidence
                language=result.get("language", "en"),
                timestamp=time.time(),
                audio_duration=len(segment) / 16000,
                word_timestamps=[],
                detected_language=result.get("language", "en"),
                temperature=0.0,
                no_speech_prob=result.get("no_speech_prob", 0.0)
            )

            results.append(enhanced_result)

        return results

class KeywordSpotting:
    """
    Keyword spotting for robot wake word detection
    """

    def __init__(self, wake_words: List[str] = None):
        self.wake_words = wake_words or ["robot", "hey robot", "hello robot", "assistant"]
        self.activation_threshold = 0.7

    def detect_wake_word(self, text: str) -> bool:
        """Check if text contains a wake word"""
        text_lower = text.lower().strip()

        for wake_word in self.wake_words:
            if wake_word in text_lower:
                return True

        return False

    def extract_command_after_wake(self, text: str) -> str:
        """Extract the command after the wake word"""
        text_lower = text.lower().strip()

        for wake_word in self.wake_words:
            if wake_word in text_lower:
                # Remove the wake word and return the rest
                command = text_lower.replace(wake_word, "", 1).strip()
                return command

        # If no wake word found, return original text
        return text

class SpeechRecognitionPipeline:
    """
    Complete pipeline for speech recognition in robotics
    """

    def __init__(self, model_size: str = "base"):
        self.multilingual_recognizer = MultilingualSpeechRecognizer(model_size)
        self.noise_robust_recognizer = NoiseRobustRecognizer(model_size)
        self.keyword_spotter = KeywordSpotting()
        self.is_active = False

    def process_audio_stream(self, audio_chunk: np.ndarray, is_noisy: bool = False) -> Dict[str, Any]:
        """Process a chunk of audio in the pipeline"""
        start_time = time.time()

        # Determine if we need noise-robust processing
        if is_noisy:
            results = self.noise_robust_recognizer.recognize_in_noise(audio_chunk)
        else:
            result = self.multilingual_recognizer.transcribe_multilingual(audio_chunk)
            results = [result]

        # Process each result
        processed_results = []
        for result in results:
            # Check for wake word
            has_wake_word = self.keyword_spotter.detect_wake_word(result.text)
            command = self.keyword_spotter.extract_command_after_wake(result.text)

            processed_result = {
                "original_text": result.text,
                "command": command,
                "has_wake_word": has_wake_word,
                "confidence": result.confidence,
                "language": result.detected_language,
                "is_robot_command": has_wake_word and len(command.strip()) > 0
            }

            processed_results.append(processed_result)

        processing_time = time.time() - start_time

        return {
            "results": processed_results,
            "processing_time": processing_time,
            "timestamp": time.time()
        }

def create_advanced_speech_pipeline(model_size: str = "base") -> SpeechRecognitionPipeline:
    """Factory function to create an advanced speech recognition pipeline"""
    return SpeechRecognitionPipeline(model_size)
```

## Examples

### Example 1: Robot Command Recognition System

```python
#!/usr/bin/env python3
# robot_command_recognition.py

import numpy as np
import asyncio
from typing import Dict, Any, Callable
import time

class RobotCommandRecognitionSystem:
    """
    Complete system for recognizing and executing robot commands from speech
    """

    def __init__(self):
        self.speech_pipeline = create_advanced_speech_pipeline("base")
        self.robot_interface = None  # This would connect to actual robot
        self.command_queue = []
        self.is_running = False

    def set_robot_interface(self, robot_interface):
        """Set the robot interface for command execution"""
        self.robot_interface = robot_interface

    async def process_speech_command(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process speech command and prepare for execution"""
        # Process through the speech pipeline
        pipeline_result = self.speech_pipeline.process_audio_stream(audio_data)

        # Extract the highest confidence robot command
        robot_commands = [
            result for result in pipeline_result["results"]
            if result["is_robot_command"] and result["confidence"] > 0.6
        ]

        if robot_commands:
            # Sort by confidence and get the best one
            best_command = max(robot_commands, key=lambda x: x["confidence"])

            # Parse the command for robot execution
            parsed_command = self._parse_robot_command(best_command["command"])

            return {
                "success": True,
                "command": parsed_command,
                "confidence": best_command["confidence"],
                "raw_text": best_command["original_text"],
                "processing_time": pipeline_result["processing_time"]
            }
        else:
            return {
                "success": False,
                "command": None,
                "confidence": 0.0,
                "raw_text": "",
                "processing_time": pipeline_result["processing_time"],
                "error": "No valid robot command detected"
            }

    def _parse_robot_command(self, command_text: str) -> Dict[str, Any]:
        """Parse natural language command into robot actions"""
        command_text = command_text.lower().strip()

        # Define command patterns
        if any(word in command_text for word in ["move", "go", "navigate", "walk"]):
            # Navigation command
            target_location = self._extract_location(command_text)
            return {
                "action": "navigation",
                "target": target_location,
                "parameters": {"speed": 0.5}
            }
        elif any(word in command_text for word in ["pick", "grasp", "take", "lift"]):
            # Manipulation command
            target_object = self._extract_object(command_text)
            return {
                "action": "manipulation",
                "target": target_object,
                "parameters": {"gripper_position": 0.8}
            }
        elif any(word in command_text for word in ["place", "put", "drop", "release"]):
            # Placement command
            target_location = self._extract_location(command_text)
            return {
                "action": "placement",
                "target": target_location,
                "parameters": {"gripper_position": 0.0}
            }
        else:
            # Unknown command
            return {
                "action": "unknown",
                "target": command_text,
                "parameters": {}
            }

    def _extract_location(self, command: str) -> str:
        """Extract location from command"""
        # Simple keyword-based extraction
        locations = ["table", "shelf", "desk", "floor", "box", "cabinet"]

        for location in locations:
            if location in command:
                return location

        return "unknown_location"

    def _extract_object(self, command: str) -> str:
        """Extract object from command"""
        # Simple keyword-based extraction
        objects = ["cube", "ball", "bottle", "cup", "object", "item", "box"]

        for obj in objects:
            if obj in command:
                return obj

        return "unknown_object"

    async def execute_robot_command(self, command_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the parsed robot command"""
        if not command_result["success"]:
            return {
                "success": False,
                "error": command_result.get("error", "No command to execute")
            }

        command = command_result["command"]

        # In a real system, this would interface with the robot
        # For simulation, we'll just return success
        execution_result = {
            "action": command["action"],
            "target": command["target"],
            "parameters": command["parameters"],
            "status": "executed",  # In real system, this would come from robot feedback
            "timestamp": time.time()
        }

        return {
            "success": True,
            "execution_result": execution_result,
            "original_command": command_result["raw_text"]
        }

    async def continuous_command_processing(self, audio_source_func: Callable,
                                          command_callback: Callable = None):
        """Continuously process commands from an audio source"""
        self.is_running = True

        while self.is_running:
            try:
                # Get audio data from source
                audio_data = await audio_source_func()

                if audio_data is not None and len(audio_data) > 0:
                    # Process the speech command
                    command_result = await self.process_speech_command(audio_data)

                    if command_result["success"]:
                        # Execute the command
                        execution_result = await self.execute_robot_command(command_result)

                        # Call the callback if provided
                        if command_callback:
                            await command_callback(execution_result)

                        # Add to command queue
                        self.command_queue.append(execution_result)

                        print(f"Executed command: {command_result['raw_text']}")
                    else:
                        print(f"Command not recognized: {command_result.get('error', 'Unknown error')}")

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error in command processing: {e}")
                await asyncio.sleep(0.5)  # Brief pause before continuing

    def stop_processing(self):
        """Stop continuous command processing"""
        self.is_running = False

    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history"""
        return self.command_queue[-limit:]

def simulate_audio_source():
    """Simulate an audio source for testing"""
    # In a real implementation, this would capture from microphone
    # For simulation, return empty array
    return np.array([])

async def command_execution_callback(execution_result: Dict[str, Any]):
    """Callback function for executed commands"""
    print(f"Command executed: {execution_result['execution_result']['action']} "
          f"to {execution_result['execution_result']['target']}")

async def main():
    """Main function to demonstrate robot command recognition"""
    print("Initializing Robot Command Recognition System...")

    # Create the system
    robot_system = RobotCommandRecognitionSystem()

    print("System initialized. Ready to process speech commands.")
    print("In a real implementation, this would connect to a robot interface.")

    # Example of processing a single command
    # In practice, you would get this from a microphone or audio file
    sample_audio = np.random.random(16000 * 3)  # 3 seconds of random audio for demo

    result = await robot_system.process_speech_command(sample_audio)
    print(f"Processing result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Speech Recognition with Error Handling and Validation

```python
#!/usr/bin/env python3
# speech_error_handling.py

import numpy as np
import asyncio
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

@dataclass
class RecognitionError:
    """Information about recognition errors"""
    error_type: str
    error_message: str
    confidence: float
    timestamp: float
    audio_context: Optional[np.ndarray] = None

class SpeechRecognitionValidator:
    """
    Validate and handle errors in speech recognition
    """

    def __init__(self):
        self.error_log = []
        self.confidence_threshold = 0.6
        self.min_audio_length = 0.5  # seconds
        self.max_audio_length = 30.0  # seconds

    def validate_audio_input(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Validate audio input before processing"""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "suggested_action": "proceed"
        }

        # Check audio length
        audio_duration = len(audio) / sample_rate
        if audio_duration < self.min_audio_length:
            validation_result["is_valid"] = False
            validation_result["issues"].append({
                "type": "too_short",
                "message": f"Audio too short: {audio_duration:.2f}s (min: {self.min_audio_length}s)",
                "severity": "high"
            })
            validation_result["suggested_action"] = "request_longer_audio"
        elif audio_duration > self.max_audio_length:
            validation_result["is_valid"] = False
            validation_result["issues"].append({
                "type": "too_long",
                "message": f"Audio too long: {audio_duration:.2f}s (max: {self.max_audio_length}s)",
                "severity": "medium"
            })
            validation_result["suggested_action"] = "truncate_or_split"

        # Check for silence
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < 0.001:  # Very low energy indicates silence
            validation_result["is_valid"] = False
            validation_result["issues"].append({
                "type": "silence",
                "message": f"Audio appears to be silent (RMS: {rms_energy:.6f})",
                "severity": "high"
            })
            validation_result["suggested_action"] = "request_new_input"

        # Check for clipping
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.9:  # Close to maximum range indicates potential clipping
            validation_result["issues"].append({
                "type": "clipping",
                "message": f"Audio may be clipped (max amplitude: {max_amplitude:.3f})",
                "severity": "low"
            })

        return validation_result

    def validate_recognition_result(self, result: Dict[str, Any],
                                  original_audio: np.ndarray) -> Dict[str, Any]:
        """Validate the recognition result"""
        validation = {
            "is_valid": True,
            "confidence_score": result.get("confidence", 0.0),
            "issues": [],
            "suggested_action": "accept"
        }

        # Check confidence
        if result.get("confidence", 0.0) < self.confidence_threshold:
            validation["is_valid"] = False
            validation["issues"].append({
                "type": "low_confidence",
                "message": f"Recognition confidence too low: {result.get('confidence', 0.0):.3f}",
                "severity": "high"
            })
            validation["suggested_action"] = "request_confirmation"

        # Check for empty result
        text = result.get("text", "").strip()
        if not text:
            validation["is_valid"] = False
            validation["issues"].append({
                "type": "empty_result",
                "message": "No text recognized",
                "severity": "high"
            })
            validation["suggested_action"] = "request_new_input"

        # Check for potential errors in the recognized text
        if self._detect_recognition_errors(text):
            validation["issues"].append({
                "type": "potential_error",
                "message": "Recognition may contain errors",
                "severity": "medium"
            })
            validation["suggested_action"] = "request_confirmation"

        return validation

    def _detect_recognition_errors(self, text: str) -> bool:
        """Detect potential recognition errors in text"""
        # Check for repeated words (common recognition error)
        words = text.split()
        if len(words) >= 4:
            for i in range(len(words) - 3):
                if words[i] == words[i+1] == words[i+2]:
                    return True

        # Check for non-sensical sequences
        non_sensical_patterns = [
            "um um um", "uh uh uh", "ah ah ah",  # Repeated filler words
            "the the the", "and and and"  # Repeated common words
        ]

        text_lower = text.lower()
        for pattern in non_sensical_patterns:
            if pattern in text_lower:
                return True

        return False

    def log_error(self, error: RecognitionError):
        """Log recognition errors for analysis"""
        self.error_log.append(error)

        # Keep only recent errors (last 1000)
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-1000:]

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about recognition errors"""
        if not self.error_log:
            return {"total_errors": 0}

        error_types = {}
        for error in self.error_log:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1

        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "recent_errors": len([e for e in self.error_log if time.time() - e.timestamp < 3600])  # Last hour
        }

class RobustSpeechRecognitionSystem:
    """
    Robust speech recognition with comprehensive error handling
    """

    def __init__(self):
        self.speech_pipeline = create_advanced_speech_pipeline("base")
        self.validator = SpeechRecognitionValidator()
        self.error_handlers = self._initialize_error_handlers()
        self.recovery_strategies = self._initialize_recovery_strategies()

    def _initialize_error_handlers(self) -> Dict[str, Callable]:
        """Initialize error handling strategies"""
        return {
            "low_confidence": self._handle_low_confidence,
            "empty_result": self._handle_empty_result,
            "silence": self._handle_silence,
            "too_short": self._handle_too_short,
            "too_long": self._handle_too_long
        }

    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies"""
        return {
            "request_confirmation": self._request_confirmation,
            "request_new_input": self._request_new_input,
            "truncate_or_split": self._truncate_audio,
            "request_longer_audio": self._request_longer_audio
        }

    async def robust_recognition(self, audio: np.ndarray) -> Dict[str, Any]:
        """Perform robust speech recognition with error handling"""
        # Step 1: Validate audio input
        audio_validation = self.validator.validate_audio_input(audio)

        if not audio_validation["is_valid"]:
            # Handle audio validation errors
            error_type = audio_validation["issues"][0]["type"]
            if error_type in self.error_handlers:
                return await self.error_handlers[error_type](audio, audio_validation)

        # Step 2: Perform recognition
        try:
            pipeline_result = self.speech_pipeline.process_audio_stream(audio)

            # Get the best result
            if pipeline_result["results"]:
                best_result = max(pipeline_result["results"],
                                key=lambda x: x.get("confidence", 0.0))

                # Step 3: Validate recognition result
                result_validation = self.validator.validate_recognition_result(
                    best_result, audio
                )

                if result_validation["is_valid"]:
                    return {
                        "success": True,
                        "text": best_result["original_text"],
                        "confidence": best_result["confidence"],
                        "command": best_result["command"] if "command" in best_result else "",
                        "validation": result_validation
                    }
                else:
                    # Handle recognition validation errors
                    action = result_validation["suggested_action"]
                    if action in self.recovery_strategies:
                        return await self.recovery_strategies[action](
                            best_result, result_validation
                        )
                    else:
                        return {
                            "success": False,
                            "error": f"Unrecognized validation issue: {action}",
                            "confidence": best_result.get("confidence", 0.0)
                        }
            else:
                return {
                    "success": False,
                    "error": "No results from speech pipeline",
                    "confidence": 0.0
                }

        except Exception as e:
            # Log the error
            error_info = RecognitionError(
                error_type="recognition_exception",
                error_message=str(e),
                confidence=0.0,
                timestamp=time.time()
            )
            self.validator.log_error(error_info)

            return {
                "success": False,
                "error": f"Recognition exception: {str(e)}",
                "confidence": 0.0
            }

    async def _handle_low_confidence(self, audio: np.ndarray, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle low confidence recognition"""
        # Try with different model settings
        try:
            # For demonstration, we'll just return a low confidence result
            # In practice, you might try different temperature settings or models
            result = self.speech_pipeline.noise_robust_recognizer.recognize_in_noise(audio)

            if result:
                best_result = max(result, key=lambda x: x.confidence)
                return {
                    "success": True,
                    "text": best_result.text,
                    "confidence": best_result.confidence,
                    "validation": validation,
                    "warning": "Low confidence result - consider requesting confirmation"
                }
        except:
            pass

        return {
            "success": False,
            "error": "Low confidence recognition failed",
            "confidence": 0.0
        }

    async def _handle_empty_result(self, audio: np.ndarray, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle empty recognition result"""
        return {
            "success": False,
            "error": "No speech detected in audio",
            "confidence": 0.0,
            "suggested_action": "request_new_input"
        }

    async def _request_confirmation(self, result: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """Request user confirmation for uncertain recognition"""
        return {
            "success": True,
            "text": result.get("original_text", ""),
            "confidence": result.get("confidence", 0.0),
            "requires_confirmation": True,
            "suggested_confirmation_text": f"Did you say: '{result.get('original_text', '')}'?",
            "validation": validation
        }

    async def _request_new_input(self, result: Dict[str, Any] = None,
                                validation: Dict[str, Any] = None) -> Dict[str, Any]:
        """Request new audio input"""
        return {
            "success": False,
            "error": "Invalid input detected",
            "requires_new_input": True,
            "suggested_action": "Please speak again more clearly"
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get the status of the recognition system"""
        return {
            "error_statistics": self.validator.get_error_statistics(),
            "confidence_threshold": self.validator.confidence_threshold,
            "validation_rules": {
                "min_audio_length": self.validator.min_audio_length,
                "max_audio_length": self.validator.max_audio_length
            }
        }

def main():
    """Main function for speech recognition error handling"""
    print("Initializing Robust Speech Recognition System...")

    # Create the robust system
    robust_system = RobustSpeechRecognitionSystem()

    print("System initialized with error handling capabilities.")
    print("Status:", robust_system.get_system_status())

    print("\nIn a real implementation, this system would handle various recognition errors")
    print("and provide appropriate recovery strategies for robotic applications.")

if __name__ == "__main__":
    main()
```

## Summary

Speech recognition is a critical component of Vision-Language-Action systems, enabling natural human-robot interaction through spoken commands. Key aspects include:

- **OpenAI Whisper**: State-of-the-art speech recognition model that provides high accuracy across multiple languages
- **Audio Processing**: Proper handling of audio input, preprocessing, and noise reduction for robotics applications
- **Error Handling**: Robust validation and error recovery mechanisms for reliable operation
- **Integration**: Seamless connection between speech recognition and robot command execution systems
- **Real-time Performance**: Efficient processing for responsive robot interaction

## Exercises

### Conceptual
1. Compare and contrast different speech recognition approaches (traditional HMM-based vs. modern transformer-based like Whisper). What are the advantages and disadvantages of each for robotics applications?

### Logical
1. Design a speech recognition pipeline that can handle multiple speakers in a household robotics scenario. How would you incorporate speaker identification and personalized recognition for different family members?

### Implementation
1. Implement a complete speech recognition system with Wake Word detection, Whisper-based transcription, and command validation for a humanoid robot, including proper error handling and confidence scoring.
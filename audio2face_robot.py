"""
audio2face_robot.py
====================

This script demonstrates how to build a voice‑driven animated face using
NVIDIA's open‑source Audio2Face‑3D microservice together with basic
voice activity detection (VAD), speech‑to‑text (STT) via OpenAI, and a
simple 2D parametric face renderer.  The resulting pipeline is intended
to be used as a reference by an agent tasked with assembling a more
complete assistant robot.  It shows how to:

  * Capture audio from a microphone in short frames using `sounddevice`.
  * Detect the end of a user's utterance with `webrtcvad` and a silence
    threshold.
  * Optionally transcribe the utterance with OpenAI's whisper models.
  * Send the raw audio to the Audio2Face‑3D microservice and receive
    blendshape weights and an inferred emotion.
  * Map the inferred emotion to a set of simple pose parameters (brows,
    mouth, eyes, head tilt) and animate a face drawn with `pygame`.

This script is not a complete production system.  It relies on an
Audio2Face‑3D microservice running locally (see the README in the
Audio2Face‑3D repository for instructions on starting the service) and
requires a valid OpenAI API key if transcription is desired.  Use it as
a starting point for your own integration.

Dependencies:
  pip install sounddevice soundfile numpy webrtcvad requests pygame openai

Environment variables:
  OPENAI_API_KEY – your OpenAI API key (if transcription is enabled).
  AUDIO2FACE_SERVER_URL – base URL for the Audio2Face microservice
    (default: http://localhost:5000/infer).

Note:
  Running Audio2Face inference requires a supported NVIDIA GPU and the
  pre‑trained models from the Audio2Face‑3D repository.  The
  microservice can be run inside a Docker container provided by
  NVIDIA.

"""

import io
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import requests
import sounddevice as sd
import webrtcvad

try:
    import pygame
except ImportError:
    pygame = None  # Optional; the renderer is disabled if pygame isn't available.

try:
    import openai
except ImportError:
    openai = None  # STT is optional.

# -----------------------------------------------------------------------------
# Voice Activity Detection and Audio Capture
# -----------------------------------------------------------------------------

class AudioStream:
    """Continuously records audio from the default microphone in small frames.

    Each frame is 20 ms long (160 samples at 8 kHz) which is the required
    granularity for WebRTC VAD.  The frames are pushed into a thread‑safe
    queue for processing by the main loop.  The stream runs in a background
    thread and stops when `stop()` is called.
    """

    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 20):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        threading.Thread(target=self._run, daemon=True).start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if not self._running:
            return
        # Copy to avoid referencing memory after callback returns
        self._queue.put(bytes(indata))

    def _run(self) -> None:
        # Keep the thread alive until stopped
        while self._running:
            time.sleep(0.1)

    def read_frame(self) -> bytes:
        """Blocking read of the next audio frame."""
        return self._queue.get()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

# -----------------------------------------------------------------------------
# Parametric Face Renderer
# -----------------------------------------------------------------------------

@dataclass
class FacePose:
    """Represents a set of facial pose parameters for a simple 2D face."""

    brow_l: float = 0.0
    brow_r: float = 0.0
    mouth_curve: float = 0.0
    mouth_open: float = 0.0
    eye_open: float = 1.0
    head_tilt: float = 0.0


class FaceRenderer:
    """Draws a simple face using pygame given the current pose."""

    def __init__(self, width: int = 640, height: int = 480):
        if pygame is None:
            raise RuntimeError(
                "pygame is not installed; install it or set up an alternative renderer"
            )
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Audio2Face Robot")
        self.clock = pygame.time.Clock()

    def draw_face(self, pose: FacePose) -> None:
        # Background
        self.screen.fill((30, 30, 30))
        cx, cy = self.width // 2, self.height // 2
        face_radius = min(self.width, self.height) * 0.35
        # Apply head tilt by rotating the coordinate system
        tilt_deg = pose.head_tilt * 57.2958  # radians to degrees
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Draw head
        pygame.draw.circle(surface, (240, 220, 210), (cx, cy), int(face_radius))
        # Mouth
        mouth_y = cy + int(face_radius * 0.4)
        mouth_width = face_radius * 0.8
        mouth_height = pose.mouth_open * 30 + 5
        curve = pose.mouth_curve
        start_angle = np.pi + curve * np.pi / 4
        end_angle = 2 * np.pi - curve * np.pi / 4
        rect = pygame.Rect(
            cx - mouth_width / 2, mouth_y - mouth_height / 2, mouth_width, mouth_height
        )
        pygame.draw.arc(surface, (50, 10, 10), rect, start_angle, end_angle, 4)
        # Eyes
        eye_offset_x = face_radius * 0.35
        eye_offset_y = face_radius * 0.15
        eye_width = face_radius * 0.25
        eye_height = face_radius * 0.15 * pose.eye_open
        for sign in [-1, 1]:
            ex = cx + sign * eye_offset_x
            ey = cy - eye_offset_y
            pygame.draw.ellipse(
                surface,
                (20, 20, 20),
                (ex - eye_width / 2, ey - eye_height / 2, eye_width, eye_height),
            )
            # Brows
            brow_y = ey - eye_height * 1.2
            brow_width = eye_width
            brow_height = face_radius * 0.03
            brow_curve = pose.brow_l if sign == -1 else pose.brow_r
            brow_rect = pygame.Rect(
                ex - brow_width / 2, brow_y - brow_height / 2, brow_width, brow_height
            )
            # Tilt brow by adjusting start/end angles
            brow_start = np.pi * (1.1 + brow_curve * 0.2)
            brow_end = np.pi * (1.4 + brow_curve * 0.2)
            pygame.draw.arc(surface, (0, 0, 0), brow_rect, brow_start, brow_end, 3)
        # Rotate the surface for head tilt
        rotated = pygame.transform.rotate(surface, tilt_deg)
        rect = rotated.get_rect(center=(cx, cy))
        self.screen.blit(rotated, rect)
        pygame.display.flip()
        self.clock.tick(30)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def transcribe_audio(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Transcribes an utterance using OpenAI whisper if available.

    Returns an empty string if OpenAI is not configured.
    """
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        return ""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        import soundfile as sf

        # Write to in‑memory buffer as WAV
        with io.BytesIO() as buf:
            # Convert int16 bytes to float32 for whisper
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            sf.write(buf, audio_np, sample_rate, format="WAV")
            buf.seek(0)
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                response_format="json",
            )
        return response.get("text", "")
    except Exception as e:
        print("STT error:", e)
        return ""


def send_to_audio2face(audio_data: bytes, sample_rate: int = 16000) -> Dict:
    """Send audio to the Audio2Face microservice and return the JSON result.

    The microservice must accept multipart/form‑data with an 'audio' file.
    See the Audio2Face‑3D microservice sample for the expected API.
    """
    server_url = os.getenv("AUDIO2FACE_SERVER_URL", "http://localhost:8000/infer")
    try:
        files = {
            "audio": ("utterance.wav", audio_data, "application/octet-stream"),
            "sample_rate": (None, str(sample_rate)),
        }
        resp = requests.post(server_url, files=files, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("Audio2Face request error:", e)
        return {}


def map_emotion_to_pose(emotion: str) -> FacePose:
    """Map the Audio2Face emotion label to a FacePose.

    The Audio2Face microservice may return emotions like 'neutral', 'happy',
    'sad', 'angry', 'surprised', etc.  Adjust these values to taste.
    """
    # Base neutral pose
    pose = FacePose()
    if emotion == "happy":
        pose.mouth_curve = 0.8
        pose.mouth_open = 0.3
        pose.brow_l = pose.brow_r = -0.2
        pose.eye_open = 0.95
        pose.head_tilt = 0.05
    elif emotion == "sad":
        pose.mouth_curve = -0.6
        pose.mouth_open = 0.2
        pose.brow_l = 0.4
        pose.brow_r = 0.4
        pose.eye_open = 0.9
        pose.head_tilt = -0.05
    elif emotion == "angry":
        pose.mouth_curve = -0.3
        pose.mouth_open = 0.2
        pose.brow_l = 0.6
        pose.brow_r = 0.6
        pose.eye_open = 0.9
        pose.head_tilt = -0.02
    elif emotion == "surprised":
        pose.mouth_curve = 0.2
        pose.mouth_open = 0.6
        pose.brow_l = -0.5
        pose.brow_r = -0.5
        pose.eye_open = 1.0
        pose.head_tilt = 0.0
    elif emotion == "neutral":
        pose.mouth_curve = 0.0
        pose.mouth_open = 0.1
        pose.brow_l = pose.brow_r = 0.0
        pose.eye_open = 0.95
        pose.head_tilt = 0.0
    # Add other mappings as necessary
    return pose


def main() -> None:
    # Initialize audio stream and VAD
    vad = webrtcvad.Vad(2)  # Aggressiveness (0–3). 2 is a good default.
    stream = AudioStream(sample_rate=16000, frame_duration_ms=20)
    stream.start()
    audio_frames: list[bytes] = []
    speech_started = False
    silence_ms = 0
    silence_threshold_ms = 800  # end of speech if this much silence accumulated

    # Initialize renderer if available
    renderer = None
    pose = FacePose()
    if pygame is not None:
        renderer = FaceRenderer()

    try:
        while True:
            frame = stream.read_frame()
            is_speech = vad.is_speech(frame, sample_rate=16000)
            if is_speech:
                audio_frames.append(frame)
                speech_started = True
                silence_ms = 0
            elif speech_started:
                silence_ms += 20
                if silence_ms >= silence_threshold_ms:
                    # End of utterance
                    utterance_data = b"".join(audio_frames)
                    audio_frames = []
                    speech_started = False
                    silence_ms = 0
                    # Process utterance in a worker thread to avoid blocking audio
                    threading.Thread(
                        target=process_utterance, args=(utterance_data, pose, renderer)
                    ).start()
            # Draw current pose
            if renderer is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                renderer.draw_face(pose)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        if pygame is not None:
            pygame.quit()


def process_utterance(audio_data: bytes, pose: FacePose, renderer: FaceRenderer | None) -> None:
    """Handle a completed utterance: transcribe, query Audio2Face, update pose."""
    # Optionally transcribe speech (useful for logging)
    transcript = transcribe_audio(audio_data)
    if transcript:
        print("You said:", transcript)
    # Send to Audio2Face microservice
    result = send_to_audio2face(audio_data)
    emotion = result.get("emotion", "neutral")
    print("Detected emotion:", emotion)
    # Map emotion to pose
    new_pose = map_emotion_to_pose(emotion)
    # Update pose values atomically
    pose.brow_l = new_pose.brow_l
    pose.brow_r = new_pose.brow_r
    pose.mouth_curve = new_pose.mouth_curve
    pose.mouth_open = new_pose.mouth_open
    pose.eye_open = new_pose.eye_open
    pose.head_tilt = new_pose.head_tilt


if __name__ == "__main__":
    main()

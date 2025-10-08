"""
Robot Assistant — Desktop (Python) with END‑OF‑SPEECH detection
- Realtime microphone stream with VAD (voice activity detection)
- Detects when the speaker STOPS talking (endpointing) and then triggers
  transcription + emotion classification for the whole utterance
- Animates a face in a PyGame window to reflect emotion

Setup
-----
1) Python 3.10+
2) pip install -r requirements.txt   (see inline list below)
3) Set your key:  export OPENAI_API_KEY=sk-...
4) Run:  python main.py

Requirements
------------
pygame
sounddevice
soundfile
numpy
webrtcvad
openai

Notes
-----
- VAD uses WebRTC (webrtcvad) at 20ms frames; it buffers audio until
  there is sustained silence (e.g., 800 ms) after speech → utterance end.
- You can tune thresholds under VADConfig.
- Swap SimpleFace with your robot’s renderer when ready.
"""

import io
import os
import time
import json
import queue
import threading
from dataclasses import dataclass
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
from pygame import gfxdraw
import webrtcvad
from dotenv import load_dotenv

from openai import OpenAI

# --------------- Config -----------------
load_dotenv()

SAMPLE_RATE = 16000          # VAD requires 8/16/32/48k; we use 16k
CHANNELS = 1
FRAME_MS = 20                # 10, 20, or 30ms only
SILENCE_TAIL_MS = 800        # how much trailing silence = end of utterance
PRE_ROLL_MS = 200            # audio kept before first speech
VAD_AGGRESSIVENESS = 2       # 0-3 (3 most aggressive)
EMOTION_HYSTERESIS_FRAMES = 2  # Utterances to see emotion before change

STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
CLS_MODEL = os.getenv("CLS_MODEL", "gpt-4-turbo")
LABELS = ["listening","happy","laughing","sad","angry","surprised","confused"]

# --------------- OpenAI client ----------
client = OpenAI()

# --------------- Helpers ----------------
def float32_to_int16(audio_f32: np.ndarray) -> bytes:
    # Expect shape (n, 1) float32 in [-1,1]
    a = np.clip(audio_f32.squeeze(), -1.0, 1.0)
    a_i16 = (a * 32767.0).astype(np.int16)
    return a_i16.tobytes()

@dataclass
class Utterance:
    pcm_bytes: bytes  # 16-bit mono PCM
    wav_bytes: bytes  # WAV container for STT
    start_ts: float
    end_ts: float

# --------------- VAD Stream -------------
class VADConfig:
    samplerate = SAMPLE_RATE
    frame_ms = FRAME_MS
    aggressiveness = VAD_AGGRESSIVENESS
    pre_roll_ms = PRE_ROLL_MS
    silence_tail_ms = SILENCE_TAIL_MS

class VADStream:
    """Turn continuous mic audio into utterances using WebRTC VAD.
    Emits an Utterance when speech ends (detected by trailing silence).
    """
    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.frame_size = int(cfg.samplerate * cfg.frame_ms / 1000)
        self.bytes_per_frame = self.frame_size * 2  # int16 mono
        self.vad = webrtcvad.Vad(cfg.aggressiveness)
        self.q_frames = queue.Queue()  # raw PCM16 frames in bytes
        self.q_utts = queue.Queue()    # Utterance objects
        self._running = False
        self._stream = None
        self._th = None

        # state
        self.preroll = deque(maxlen=int(cfg.pre_roll_ms / cfg.frame_ms))
        self.active_buf = bytearray()
        self.in_speech = False
        self.last_voice_ts = 0.0
        self.latest_rms = 0.0

    def _audio_cb(self, indata, frames, time_info, status):
        # indata float32; convert to int16 PCM expected by VAD
        self.latest_rms = np.sqrt(np.mean(np.square(indata)))
        pcm = float32_to_int16(indata)
        # break into exact FRAME_MS chunks
        for off in range(0, len(pcm), self.bytes_per_frame):
            chunk = pcm[off:off+self.bytes_per_frame]
            if len(chunk) == self.bytes_per_frame:
                self.q_frames.put(chunk)

    def start(self):
        if self._running: return
        self._running = True
        self._stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=self.cfg.samplerate,
            blocksize=self.frame_size,  # pulls about FRAME_MS frames
            dtype='float32',
            callback=self._audio_cb
        )
        self._stream.start()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop(); self._stream.close()
        if self._th:
            self._th.join(timeout=1.0)

    def _emit_utterance(self, end_ts: float):
        if not self.active_buf and not self.preroll:
            return
        self.in_speech = False
        pcm_bytes = b''.join(self.preroll) + bytes(self.active_buf)
        # pack as WAV for STT
        bio = io.BytesIO()
        # convert back to np.int16 for soundfile
        pcm_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.int16)
        sf.write(bio, pcm_np, self.cfg.samplerate, format='WAV', subtype='PCM_16')
        wav_bytes = bio.getvalue()
        start_ts = end_ts - (len(pcm_bytes) / 2) / self.cfg.samplerate
        utt = Utterance(pcm_bytes=pcm_bytes, wav_bytes=wav_bytes, start_ts=start_ts, end_ts=end_ts)
        self.q_utts.put(utt)
        # reset
        self.active_buf.clear()
        self.preroll.clear()


    def _process_frame(self):
        """Processes a single frame of audio from the queue."""
        try:
            frame = self.q_frames.get(timeout=0.01)
            now = time.time()
            is_voiced = False
            try:
                is_voiced = self.vad.is_speech(frame, self.cfg.samplerate)
            except Exception:
                is_voiced = False

            if is_voiced:
                self.last_voice_ts = now
                if not self.in_speech:
                    self.in_speech = True
                    self.active_buf.extend(b''.join(self.preroll))
                    self.preroll.clear()
                self.active_buf.extend(frame)
            else:
                if self.in_speech:
                    self.active_buf.extend(frame)
                    if (now - self.last_voice_ts) >= self.cfg.silence_tail_ms / 1000.0:
                        self._emit_utterance(now)
                else:
                    self.preroll.append(frame)

        except queue.Empty:
            if self.in_speech and (time.time() - self.last_voice_ts) >= self.cfg.silence_tail_ms / 1000.0:
                self._emit_utterance(time.time())
            return

    def _loop(self):
        while self._running:
            self._process_frame()

# --------------- NLP worker -------------
class NLPWorker:
    def __init__(self, hysteresis_threshold=2):
        self.label = "listening"
        self.confidence = 0.0
        self._running = False
        self._th = None
        self._in_q = queue.Queue()

        # Hysteresis state
        self.hysteresis_threshold = hysteresis_threshold
        self.candidate_label = "listening"
        self.candidate_confidence = 0.0
        self.candidate_count = 0

    def push_utterance(self, utt: Utterance):
        self._in_q.put(utt)

    def start(self):
        if self._running: return
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False
        if self._th:
            self._th.join(timeout=1.0)

    def _transcribe(self, wav_bytes: bytes) -> str:
        file_like = io.BytesIO(wav_bytes)
        file_like.name = "utt.wav"
        try:
            t = client.audio.transcriptions.create(
                file=file_like,
                model=STT_MODEL,
                language='en'
            )
            return getattr(t, 'text', '') or ''
        except Exception as e:
            try:
                t = client.audio.transcriptions.create(
                    file=file_like,
                    model='whisper-1',
                    language='en'
                )
                return getattr(t, 'text', '') or ''
            except Exception:
                print('Transcription error:', e)
                return ''

    def _classify(self, text: str):
        try:
            system = (
                "You output ONLY strict JSON: {\"label\": string, \"confidence\": number}. "
                f"Choose label from this set: {', '.join(LABELS)}. "
                "If neutral/no emotion, use 'listening'."
            )
            resp = client.chat.completions.create(
                model=CLS_MODEL,
                temperature=0,
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":f"""Text: {text}
Respond with JSON only."""}
                ]
            )
            data = json.loads(resp.choices[0].message.content)
            label = data.get('label', 'listening')
            conf = float(data.get('confidence', 0.5))
            if label not in LABELS:
                label = 'listening'
            return label, conf
        except Exception as e:
            print('Classification error:', e)
            return 'listening', 0.0

    def _process_utterance(self):
        """Processes a single utterance from the queue."""
        try:
            utt = self._in_q.get(timeout=0.01)
        except queue.Empty:
            return

        text = self._transcribe(utt.wav_bytes)
        if not text:
            if self.label != "listening":
                print("No text from STT, returning to listening state.")
            self.label, self.confidence = "listening", 0.0
            self.candidate_label, self.candidate_count = "listening", 0
            return

        new_label, new_conf = self._classify(text)

        if new_label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = new_label
            self.candidate_confidence = new_conf
            self.candidate_count = 1

        if self.candidate_count >= self.hysteresis_threshold:
            if self.label != self.candidate_label:
                print(f"Emotion state changed to: {self.candidate_label}")
            self.label = self.candidate_label
            self.confidence = self.candidate_confidence

    def _loop(self):
        while self._running:
            self._process_utterance()

# --------------- Face UI (PyGame) -------
@dataclass
class FacePose:
    """Represents a set of facial pose parameters for a simple 2D face."""
    brow_l: float = 0.0
    brow_r: float = 0.0
    mouth_curve: float = 0.0
    mouth_open: float = 0.0
    eye_open: float = 1.0
    pupil_x: float = 0.0
    pupil_y: float = 0.0
    head_tilt: float = 0.0

class FaceRenderer:
    """Draws a simple face using pygame given the current pose."""

    def __init__(self, width: int = 640, height: int = 480):
        pygame.init()
        pygame.display.set_caption("Robot Face")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.bg_color = (10, 16, 40)
        self.face_color = (240, 220, 210)
        self.stroke_color = (20, 20, 20)
        self.hud_emotion = "listening"

    def draw_face(self, pose: FacePose, status: str) -> None:
        self.screen.fill(self.bg_color)
        cx, cy = self.width // 2, self.height // 2
        face_radius = min(self.width, self.height) * 0.35

        face_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(face_surface, self.face_color, (cx, cy), int(face_radius))

        eye_offset_x = face_radius * 0.35
        eye_offset_y = face_radius * 0.15
        eye_width = face_radius * 0.3
        eye_height = face_radius * 0.3 * pose.eye_open
        for sign in [-1, 1]:
            ex = cx + sign * eye_offset_x
            ey = cy - eye_offset_y
            pygame.draw.ellipse(face_surface, (255, 255, 255), (ex - eye_width / 2, ey - eye_height / 2, eye_width, eye_height))
            pygame.draw.ellipse(face_surface, self.stroke_color, (ex - eye_width / 2, ey - eye_height / 2, eye_width, eye_height), 2)
            pupil_radius = eye_width * 0.2
            px = ex + pose.pupil_x * (eye_width / 2 - pupil_radius)
            py = ey + pose.pupil_y * (eye_height / 2 - pupil_radius)
            pygame.draw.circle(face_surface, self.stroke_color, (int(px), int(py)), int(pupil_radius))

            brow_y = ey - eye_height * 1.5
            brow_width = eye_width * 1.2
            brow_height = face_radius * 0.05
            brow_curve = pose.brow_l if sign == -1 else pose.brow_r
            brow_rect = pygame.Rect(ex - brow_width / 2, brow_y - brow_height, brow_width, brow_height * 2)
            start_angle = np.pi + (np.pi/4 * brow_curve * sign)
            end_angle = 2*np.pi - (np.pi/4 * brow_curve * sign)
            pygame.draw.arc(face_surface, self.stroke_color, brow_rect, start_angle, end_angle, 5)

        mouth_y = cy + int(face_radius * 0.4)
        mouth_width = face_radius * 0.8
        mouth_height = pose.mouth_open * 40 + 5
        curve = pose.mouth_curve
        if mouth_height > 5:
            mouth_rect = pygame.Rect(cx - mouth_width/2, mouth_y - mouth_height/2, mouth_width, mouth_height)
            pygame.draw.ellipse(face_surface, self.stroke_color, mouth_rect)
        else:
            start_angle = np.pi + curve * np.pi / 2.5
            end_angle = 2 * np.pi - curve * np.pi / 2.5
            mouth_rect = pygame.Rect(cx - mouth_width / 2, mouth_y - 20, mouth_width, 40)
            pygame.draw.arc(face_surface, self.stroke_color, mouth_rect, start_angle, end_angle, 4)

        tilt_deg = pose.head_tilt * -30
        rotated_surface = pygame.transform.rotate(face_surface, tilt_deg)
        rotated_rect = rotated_surface.get_rect(center=(cx, cy))
        self.screen.blit(rotated_surface, rotated_rect.topleft)

        font = pygame.font.SysFont(None, 22)
        hud_emotion = font.render(f"emotion: {self.hud_emotion}", True, (180,200,255))
        hud_status = font.render(f"status: {status}", True, (150,170,220))
        self.screen.blit(hud_emotion, (16, 16))
        self.screen.blit(hud_status, (16, 40))

        pygame.display.flip()

    def loop(self, state_getter):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pose, status, emotion = state_getter()
            self.hud_emotion = emotion
            self.draw_face(pose, status)
            self.clock.tick(60)
        pygame.quit()


def map_emotion_to_pose(emotion: str) -> FacePose:
    """Map the emotion label to a target FacePose."""
    pose = FacePose() # Start with neutral
    if emotion == "happy":
        pose.mouth_curve = 0.8
        pose.mouth_open = 0.3
        pose.brow_l = pose.brow_r = -0.3
    elif emotion == "sad":
        pose.mouth_curve = -0.6
        pose.brow_l = 0.3
        pose.brow_r = -0.3
        pose.eye_open = 0.8
    elif emotion == "angry":
        pose.mouth_curve = -0.4
        pose.brow_l = 0.6
        pose.brow_r = 0.6
        pose.head_tilt = -0.1
    elif emotion == "surprised":
        pose.mouth_open = 0.7
        pose.brow_l = -0.7
        pose.brow_r = -0.7
        pose.eye_open = 1.0
    elif emotion == "laughing":
        pose.mouth_curve = 0.5
        pose.mouth_open = 0.8
    elif emotion == "confused":
        pose.brow_r = 0.4
        pose.mouth_curve = -0.1
        pose.head_tilt = 0.1
    return pose


# --------------- App wiring -------------
class App:
    def __init__(self):
        self.vad = VADStream(VADConfig())
        self.nlp = NLPWorker(hysteresis_threshold=EMOTION_HYSTERESIS_FRAMES)
        self.face = FaceRenderer(width=640, height=480)
        self.status = "listening"
        self.current_pose = FacePose()
        self.target_pose = FacePose()
        self.last_blink_time = time.time()
        self.next_blink_delay = np.random.uniform(2.0, 5.0)
        self.is_blinking = False
        self.blink_end_time = 0.0
        self.last_emotion_label = "listening"

    def start(self):
        print("Starting VAD & NLP…")
        self.vad.start()
        self.nlp.start()
        threading.Thread(target=self._pump, daemon=True).start()
        self.face.loop(self._get_state)
        self.shutdown()

    def _pump(self):
        while True:
            try:
                utt = self.vad.q_utts.get(timeout=0.01)
                self.nlp.push_utterance(utt)
            except queue.Empty:
                continue

    def _get_state(self):
        now = time.time()
        dt = self.face.clock.get_time() / 1000.0

        if self.vad.in_speech:
            self.status = "speaking"
        elif self.status == "speaking":
            self.status = "processing"

        if self.nlp.label != self.last_emotion_label:
            if self.status == "processing":
                self.status = "listening"
            self.last_emotion_label = self.nlp.label
            self.target_pose = map_emotion_to_pose(self.nlp.label)

        tween_speed = 4.0
        for field in self.current_pose.__dataclass_fields__:
            current_val = getattr(self.current_pose, field)
            target_val = getattr(self.target_pose, field)
            new_val = current_val + (target_val - current_val) * min(dt * tween_speed, 1.0)
            setattr(self.current_pose, field, new_val)

        final_pose = FacePose(**self.current_pose.__dict__)

        blink_duration = 0.12
        if self.is_blinking and now > self.blink_end_time:
            self.is_blinking = False
            self.last_blink_time = now
            self.next_blink_delay = np.random.uniform(2.5, 5.5)

        if not self.is_blinking and now > self.last_blink_time + self.next_blink_delay:
            self.is_blinking = True
            self.blink_end_time = now + blink_duration

        if self.is_blinking:
            final_pose.eye_open = 0.0
        else:
            breath_cycle = np.sin(now * np.pi)
            final_pose.head_tilt += breath_cycle * 0.02
            final_pose.eye_open -= abs(breath_cycle) * 0.03

        if self.vad.in_speech:
            rms = self.vad.latest_rms
            final_pose.mouth_open += rms * 4.0
            final_pose.eye_open -= rms * 2.0

        final_pose.eye_open = np.clip(final_pose.eye_open, 0, 1)
        final_pose.mouth_open = np.clip(final_pose.mouth_open, 0, 1)

        return final_pose, self.status, self.nlp.label

    def shutdown(self):
        print("Shutting down…")
        self.vad.stop()
        self.nlp.stop()


if __name__ == "__main__":
    App().start()

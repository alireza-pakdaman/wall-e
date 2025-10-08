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

from openai import OpenAI

# --------------- Config -----------------
SAMPLE_RATE = 16000          # VAD requires 8/16/32/48k; we use 16k
CHANNELS = 1
FRAME_MS = 20                # 10, 20, or 30ms only
SILENCE_TAIL_MS = 800        # how much trailing silence = end of utterance
PRE_ROLL_MS = 200            # audio kept before first speech
VAD_AGGRESSIVENESS = 2       # 0-3 (3 most aggressive)

STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")  # fallback to whisper-1 if needed
CLS_MODEL = os.getenv("CLS_MODEL", "gpt-4.1-mini")
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

    def _audio_cb(self, indata, frames, time_info, status):
        # indata float32; convert to int16 PCM expected by VAD
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
        self.in_speech = False

    def _loop(self):
        frame_dur = self.cfg.frame_ms / 1000.0
        silence_needed = self.cfg.silence_tail_ms / 1000.0
        while self._running:
            try:
                frame = self.q_frames.get(timeout=0.1)
            except queue.Empty:
                continue
            now = time.time()
            is_voiced = False
            try:
                is_voiced = self.vad.is_speech(frame, self.cfg.samplerate)
            except Exception:
                is_voiced = False

            if is_voiced:
                self.last_voice_ts = now
                if not self.in_speech:
                    # move preroll into active on first speech
                    self.in_speech = True
                    self.active_buf.extend(b''.join(self.preroll))
                    self.preroll.clear()
                self.active_buf.extend(frame)
            else:
                if self.in_speech:
                    # still inside an utterance, but this frame is silent
                    self.active_buf.extend(frame)
                    if (now - self.last_voice_ts) >= silence_needed:
                        self._emit_utterance(now)
                else:
                    # idle, keep a preroll buffer so we don’t clip starts
                    self.preroll.append(frame)

# --------------- NLP worker -------------
class NLPWorker:
    def __init__(self):
        self.rolling_text = ""
        self.label = "listening"
        self.confidence = 0.0
        self._running = False
        self._th = None
        self._in_q = queue.Queue()  # Utterance

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
                    {"role":"user","content":f"Text: {text}
Respond with JSON only."}
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

    def _loop(self):
        while self._running:
            try:
                utt = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            text = self._transcribe(utt.wav_bytes)
            if text:
                self.rolling_text = (self.rolling_text + ' ' + text).strip()
                label, conf = self._classify(self.rolling_text)
                self.label, self.confidence = label, conf
            else:
                # no text, keep listening state
                self.label, self.confidence = 'listening', 0.0

# --------------- Face UI (PyGame) -------
class SimpleFace:
    def __init__(self, width=520, height=520):
        pygame.init()
        pygame.display.set_caption("Robot Face — Emotion + Endpointing")
        self.w, self.h = width, height
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.bg = (10, 16, 40)
        self.face_col = (27, 39, 94)
        self.stroke = (210, 224, 255)
        self.t0 = time.time()
        self.status = "listening"

    def draw(self, label: str, blink: bool, status: str):
        s = self.screen
        s.fill(self.bg)
        cx, cy, r = self.w//2, self.h//2, min(self.w, self.h)//2 - 40

        # face circle
        gfxdraw.filled_circle(s, cx, cy, r, self.face_col)
        gfxdraw.aacircle(s, cx, cy, r, (80, 98, 160))

        # eyes
        eye_rx, eye_ry = 28, 16
        ex_off = 80
        ey = cy - 40
        if blink:
            eye_ry = 2
        pygame.draw.ellipse(s, self.stroke, (cx-ex_off-eye_rx, ey-eye_ry, eye_rx*2, eye_ry*2), 0)
        pygame.draw.ellipse(s, self.stroke, (cx+ex_off-eye_rx, ey-eye_ry, eye_rx*2, eye_ry*2), 0)

        # brows & mouth based on label
        def line(a,b,th=6):
            pygame.draw.line(s, self.stroke, a, b, th)

        by = ey - 40
        if label == "happy":
            line((cx-120, by+5), (cx-40, by))
            line((cx+40, by), (cx+120, by+5))
        elif label == "sad":
            line((cx-120, by+10), (cx-40, by+20))
            line((cx+40, by+20), (cx+120, by+10))
        elif label == "angry":
            line((cx-120, by), (cx-40, by+20))
            line((cx+40, by+20), (cx+120, by))
        elif label == "laughing":
            line((cx-120, by-5), (cx-40, by-10))
            line((cx+40, by-10), (cx+120, by-5))
        elif label == "surprised":
            line((cx-120, by-6), (cx-40, by-6))
            line((cx+40, by-6), (cx+120, by-6))
        elif label == "confused":
            line((cx-120, by), (cx-40, by+6))
            line((cx+40, by-6), (cx+120, by))
        else:
            line((cx-120, by), (cx-40, by))
            line((cx+40, by), (cx+120, by))

        my = cy + 60
        if label == "happy":
            pygame.draw.arc(s, self.stroke, (cx-100, my-40, 200, 100), np.pi*0.1, np.pi-0.1, 8)
        elif label == "laughing":
            pygame.draw.arc(s, self.stroke, (cx-100, my-60, 200, 140), 0, np.pi, 10)
        elif label == "sad":
            pygame.draw.arc(s, self.stroke, (cx-100, my-10, 200, 100), np.pi, 2*np.pi, 8)
        elif label == "angry":
            pygame.draw.line(s, self.stroke, (cx-90, my), (cx+90, my-6), 8)
        elif label == "surprised":
            pygame.draw.circle(s, self.stroke, (cx, my), 16, 6)
        elif label == "confused":
            pygame.draw.arc(s, self.stroke, (cx-90, my-10, 180, 60), np.pi*0.1, np.pi*0.8, 8)
        else:
            pygame.draw.arc(s, self.stroke, (cx-100, my-30, 200, 80), np.pi*0.1, np.pi-0.1, 6)

        # HUD
        font = pygame.font.SysFont(None, 22)
        hud1 = font.render(f"emotion: {label}", True, (180,200,255))
        hud2 = font.render(f"status: {status}", True, (150,170,220))
        s.blit(hud1, (16, 16))
        s.blit(hud2, (16, 40))

    def loop(self, state_getter):
        blink_t = 0
        blink_dur = 0.12
        next_blink = time.time() + np.random.uniform(2.0, 5.0)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            now = time.time()
            blink = False
            if now >= next_blink:
                blink_t = now + blink_dur
                next_blink = now + np.random.uniform(2.5, 5.5)
            if now <= blink_t:
                blink = True

            label, status = state_getter()
            self.draw(label, blink, status)
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()

# --------------- App wiring -------------
class App:
    def __init__(self):
        self.vad = VADStream(VADConfig())
        self.nlp = NLPWorker()
        self.face = SimpleFace()
        self.status = "listening"  # listening | speaking | processing

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
                utt = self.vad.q_utts.get(timeout=0.1)
            except queue.Empty:
                continue
            # got a full utterance → process
            self.status = "processing"
            self.nlp.push_utterance(utt)
            # small delay so UI can show processing briefly
            time.sleep(0.01)
            self.status = "listening"

    def _get_state(self):
        return self.nlp.label, self.status

    def shutdown(self):
        print("Shutting down…")
        self.vad.stop()
        self.nlp.stop()


if __name__ == "__main__":
    App().start()

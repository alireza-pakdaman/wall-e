import unittest
from unittest.mock import Mock, patch, MagicMock
import queue
import time

# Mock the necessary modules before importing the main script
import sys
sys.modules['sounddevice'] = Mock()
sys.modules['webrtcvad'] = Mock()
sys.modules['openai'] = Mock()
sys.modules['pygame'] = Mock()
sys.modules['soundfile'] = Mock()

from main import VADStream, VADConfig, Utterance

class TestVADStream(unittest.TestCase):

    def setUp(self):
        """Set up a VADStream instance for testing."""
        self.config = VADConfig()
        self.config.frame_ms = 20
        self.config.samplerate = 16000
        self.config.pre_roll_ms = 100  # 5 frames
        self.config.silence_tail_ms = 200 # 10 frames

        # Mock webrtcvad.Vad
        self.mock_vad_instance = MagicMock()
        self.vad_patcher = patch('main.webrtcvad.Vad')
        self.mock_webrtc_vad = self.vad_patcher.start()
        self.mock_webrtc_vad.return_value = self.mock_vad_instance

        self.vad_stream = VADStream(self.config)
        self.vad_stream.q_frames = queue.Queue() # Use a real queue for frames

    def tearDown(self):
        """Stop the patcher."""
        self.vad_patcher.stop()

    def _feed_frames(self, is_speech_list):
        """Helper to feed frames into the VAD stream's queue."""
        for is_speech in is_speech_list:
            frame_data = b'\x00' * (self.config.samplerate * self.config.frame_ms // 1000 * 2)
            self.vad_stream.q_frames.put(frame_data)
            self.mock_vad_instance.is_speech.return_value = is_speech

    @patch('main.time.time')
    def test_utterance_detection(self, mock_time):
        """Test that a full utterance is detected correctly."""
        mock_time.return_value = 0.0

        def run_vad_loop(is_speech_list):
            for is_speech in is_speech_list:
                mock_time.return_value += self.config.frame_ms / 1000.0
                frame_data = b'\x00' * self.vad_stream.bytes_per_frame
                self.vad_stream.q_frames.put(frame_data)
                self.mock_vad_instance.is_speech.return_value = is_speech
                self.vad_stream._process_frame()

        # 1. Initial silence (should be buffered as pre-roll)
        run_vad_loop([False] * 5)
        self.assertEqual(len(self.vad_stream.preroll), 5)
        self.assertFalse(self.vad_stream.in_speech)

        # 2. Speech starts
        run_vad_loop([True] * 10)
        self.assertTrue(self.vad_stream.in_speech)
        self.assertEqual(len(self.vad_stream.preroll), 0)
        # active_buf should contain pre-roll + speech frames
        self.assertEqual(len(self.vad_stream.active_buf), (5 + 10) * self.vad_stream.bytes_per_frame)

        # 3. Trailing silence (less than threshold)
        run_vad_loop([False] * 9)
        self.assertTrue(self.vad_stream.in_speech) # Still in speech
        self.assertEqual(self.vad_stream.q_utts.qsize(), 0) # No utterance emitted yet

        # 4. One more silent frame to hit the threshold
        run_vad_loop([False] * 1)

        # Now an utterance should be emitted
        self.assertFalse(self.vad_stream.in_speech)
        self.assertEqual(self.vad_stream.q_utts.qsize(), 1)

        # Verify the emitted utterance
        utt = self.vad_stream.q_utts.get()
        self.assertIsInstance(utt, Utterance)
        # Total frames = 5 (pre-roll) + 10 (speech) + 10 (silence) = 25
        expected_bytes = 25 * self.vad_stream.bytes_per_frame
        self.assertEqual(len(utt.pcm_bytes), expected_bytes)

if __name__ == '__main__':
    unittest.main()
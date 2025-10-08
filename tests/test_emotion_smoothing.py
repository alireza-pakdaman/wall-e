import unittest
from unittest.mock import Mock, patch

# Mock the necessary modules before importing the main script
import sys
sys.modules['sounddevice'] = Mock()
sys.modules['webrtcvad'] = Mock()
sys.modules['openai'] = Mock()
sys.modules['pygame'] = Mock()
sys.modules['soundfile'] = Mock()

from main import NLPWorker, Utterance

class TestEmotionSmoothing(unittest.TestCase):

    def setUp(self):
        """Set up an NLPWorker instance for testing."""
        self.hysteresis_threshold = 2
        self.nlp_worker = NLPWorker(hysteresis_threshold=self.hysteresis_threshold)

        # We don't need the background thread for this test
        self.nlp_worker.start = Mock()
        self.nlp_worker.stop = Mock()

    def _create_mock_utterance(self):
        """Creates a mock Utterance object."""
        return Utterance(pcm_bytes=b'', wav_bytes=b'', start_ts=0, end_ts=0)

    def test_emotion_hysteresis(self):
        """Test that the emotion label only changes after a threshold is met."""

        # Patch the internal methods that call out to OpenAI
        with patch.object(self.nlp_worker, '_transcribe') as mock_transcribe, \
             patch.object(self.nlp_worker, '_classify') as mock_classify:

            # Initial state should be "listening"
            self.assertEqual(self.nlp_worker.label, "listening")

            # --- Frame 1: Detect "happy" ---
            mock_transcribe.return_value = "This is happy."
            mock_classify.return_value = ("happy", 0.9)
            self.nlp_worker._in_q.put(self._create_mock_utterance())
            self.nlp_worker._process_utterance()

            # Label should NOT have changed yet
            self.assertEqual(self.nlp_worker.label, "listening")
            self.assertEqual(self.nlp_worker.candidate_label, "happy")
            self.assertEqual(self.nlp_worker.candidate_count, 1)

            # --- Frame 2: Detect "happy" again ---
            self.nlp_worker._in_q.put(self._create_mock_utterance())
            self.nlp_worker._process_utterance()

            # Now the label SHOULD change to "happy"
            self.assertEqual(self.nlp_worker.label, "happy")
            self.assertEqual(self.nlp_worker.candidate_label, "happy")
            self.assertEqual(self.nlp_worker.candidate_count, 2)

            # --- Frame 3: Detect "sad" ---
            mock_transcribe.return_value = "This is sad."
            mock_classify.return_value = ("sad", 0.95)
            self.nlp_worker._in_q.put(self._create_mock_utterance())
            self.nlp_worker._process_utterance()

            # Label should NOT change back immediately
            self.assertEqual(self.nlp_worker.label, "happy")
            self.assertEqual(self.nlp_worker.candidate_label, "sad")
            self.assertEqual(self.nlp_worker.candidate_count, 1)

            # --- Frame 4: No text from STT ---
            mock_transcribe.return_value = "" # Simulate STT failure
            self.nlp_worker._in_q.put(self._create_mock_utterance())
            self.nlp_worker._process_utterance()

            # Label should reset to "listening"
            self.assertEqual(self.nlp_worker.label, "listening")
            self.assertEqual(self.nlp_worker.candidate_label, "listening")
            self.assertEqual(self.nlp_worker.candidate_count, 0)


if __name__ == '__main__':
    # To run these tests, you would typically use a test runner like pytest
    # or `python -m unittest discover tests` from the root directory.
    # We add a dummy queue to the worker to allow the loop to run once.
    NLPWorker._in_q = queue.Queue()
    NLPWorker._in_q.put(_create_mock_utterance())
    unittest.main()
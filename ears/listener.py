import logging
import time
import collections
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceListener")

class VoiceListener:
    def __init__(self, model_size="tiny.en", device="cpu", wake_word="ghost"):
        """
        Initialize the 'Ears' of Ghost-1.
        
        Args:
            model_size: Whisper model size (tiny.en is fastest)
            device: 'cpu' or 'cuda'
            wake_word: Word to trigger command mode
        """
        self.wake_word = wake_word.lower()
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.chunk_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        logger.info(f"Loading Whisper model ({model_size}) on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type="int8")
            logger.info("Whisper loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.model = None

        self.vad = webrtcvad.Vad(3)  # Aggressiveness: 3 (high)
        
    def listen_for_command(self, timeout=None):
        """
        Continuously listen for the wake word and command.
        Returns the command string when detected.
        """
        if not self.model:
            logger.warning("Voice disabled - Model not loaded.")
            return None

        logger.info(f"Listening for '{self.wake_word}'...")
        
        buffer = collections.deque(maxlen=100) # Ring buffer for pre-speech
        recording = []
        triggered = False
        silence_frames = 0
        MAX_SILENCE = 30  # ~1 second of silence to stop
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(status)
            # Convert to 16-bit PCM for VAD
            audio_data = (indata * 32767).astype(np.int16).flatten()
            recording.append(audio_data)

        # Start stream logic involves blocking read usually, 
        # but sounddevice callback is async. 
        # For simplicity in this synchronous agent step, we'll use blocking read loop.
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=None, dtype="int16") as stream:
            start_time = time.time()
            
            audio_buffer = []
            speaking = False
            silence_counter = 0
            
            while True:
                if timeout and (time.time() - start_time > timeout):
                    return None
                
                # Read chunk
                audio_chunk, overflow = stream.read(self.chunk_size)
                audio_bytes = audio_chunk.flatten().tobytes()
                
                # VAD Check
                is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
                
                if is_speech:
                    if not speaking:
                        logger.debug("Speech detected...")
                        speaking = True
                    audio_buffer.append(audio_chunk.flatten())
                    silence_counter = 0
                else:
                    if speaking:
                        audio_buffer.append(audio_chunk.flatten())
                        silence_counter += 1
                        
                        if silence_counter > MAX_SILENCE:
                            # Speech ended
                            logger.info("Processing speech...")
                            return self._transcribe(np.concatenate(audio_buffer))
                    else:
                        # Keep a small buffer of pre-speech?
                        pass
                        
    def _transcribe(self, audio_data):
        """Transcribe audio data using Faster-Whisper."""
        # Normalize to float32 -1..1
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        segments, info = self.model.transcribe(audio_float, beam_size=5)
        
        full_text = " ".join([segment.text for segment in segments]).strip().lower()
        logger.info(f"Heard: '{full_text}'")
        
        if self.wake_word in full_text:
            # Extract command after wake word
            command = full_text.split(self.wake_word, 1)[1].strip()
            if command:
                logger.info(f"Command recognized: {command}")
                return command
            else:
                return "Wake word detected, but no command."
        
        return None

if __name__ == "__main__":
    # Test
    listener = VoiceListener()
    print("Say 'Ghost, open browser'")
    cmd = listener.listen_for_command()
    print(f"Result: {cmd}")

import sys
import os
import time

import edge_tts
from loguru import logger
from .tts_interface import TTSInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# Check out doc at https://github.com/rany2/edge-tts
# Use `edge-tts --list-voices` to list all available voices


class TTSEngine(TTSInterface):
    # After the main voice returns no audio, the fallback voice goes first for
    # this many seconds — an outage of one voice usually lasts a while, and
    # every failed request costs about 3 seconds.
    VOICE_COOLDOWN = 300

    def __init__(self, voice="en-US-AvaMultilingualNeural", fallback_voice=None):
        self.voice = voice
        # A second voice (same language) used when `voice` returns no audio.
        self.fallback_voice = fallback_voice if fallback_voice != voice else None
        self._voice_failed_at = None

        self.temp_audio_file = "temp"
        self.file_extension = "mp3"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

    def _voices_to_try(self):
        """Voices for one sentence, in order; a voice may be listed twice."""
        if not self.fallback_voice:
            return [self.voice] * 3
        if (
            self._voice_failed_at is not None
            and time.monotonic() - self._voice_failed_at < self.VOICE_COOLDOWN
        ):
            return [self.fallback_voice, self.fallback_voice, self.voice, self.voice]
        return [self.voice, self.fallback_voice, self.fallback_voice, self.voice]

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext: str
            name of the file without extension


        Returns:
        str: the path to the generated audio file

        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        # edge-tts intermittently fails with "No audio was received" (service
        # hiccup, or one voice being down), which would otherwise leave the
        # character silently displaying text. Try again, and switch to the
        # fallback voice when there is one.
        voices = self._voices_to_try()
        for attempt, voice in enumerate(voices):
            try:
                communicate = edge_tts.Communicate(text, voice)
                communicate.save_sync(file_name)
                if voice != self.voice:
                    logger.info(f"edge-tts spoke with the fallback voice {voice}.")
                return file_name
            except Exception as e:
                logger.warning(
                    f"edge-tts attempt {attempt + 1}/{len(voices)} with {voice} failed: {e}"
                )
                if voice == self.voice and self.fallback_voice:
                    self._voice_failed_at = time.monotonic()
                elif attempt + 1 < len(voices) and voices[attempt + 1] == voice:
                    time.sleep(1)

        logger.critical(f"\nError: edge-tts unable to generate audio for: {text}")
        logger.critical("It's possible that edge-tts is blocked in your region.")
        return None


# en-US-AvaMultilingualNeural
# en-US-EmmaMultilingualNeural
# en-US-JennyNeural

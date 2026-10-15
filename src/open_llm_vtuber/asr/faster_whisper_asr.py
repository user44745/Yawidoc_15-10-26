import numpy as np
from loguru import logger
from faster_whisper import WhisperModel
from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    BEAM_SEARCH = True
    # SAMPLE_RATE # Defined in asr_interface.py

    def __init__(
        self,
        model_path: str = "distil-medium.en",
        download_root: str = None,
        language: str = "en",
        device: str = "auto",
        compute_type: str = "int8",
        prompt: str = None,
        vad_filter: bool = False,
    ) -> None:
        self.MODEL_PATH = model_path
        self.LANG = language
        self.prompt = prompt
        # Drop non-speech before decoding: Whisper invents sentences for
        # silence and noise ("Thanks for watching!").
        self.vad_filter = vad_filter

        if device != "cpu":
            # CTranslate2 needs the CUDA libraries (cuBLAS, cuDNN) that ship
            # with torch; importing torch puts them on the DLL search path.
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
        try:
            self.model = WhisperModel(
                model_size_or_path=model_path,
                download_root=download_root,
                device=device,
                compute_type=compute_type,
            )
            if device != "cpu":
                self._warm_up()
        except Exception as e:
            if device == "cpu":
                raise
            # No usable GPU (missing CUDA libraries, out of memory ...):
            # recognition still works on the CPU, only much slower.
            logger.error(
                f"faster-whisper could not run '{model_path}' on {device}: {e}. "
                "Falling back to the CPU, which is much slower."
            )
            self.model = WhisperModel(
                model_size_or_path=model_path,
                download_root=download_root,
                device="cpu",
                compute_type="int8",
            )

    def _warm_up(self) -> None:
        """Run one tiny decode so the first real utterance is not the slow one."""
        segments, _ = self.model.transcribe(
            np.zeros(self.SAMPLE_RATE, dtype=np.float32),
            beam_size=1,
            language=self.LANG if self.LANG else "en",
            max_new_tokens=1,
            without_timestamps=True,
        )
        list(segments)

    def transcribe_np(self, audio: np.ndarray) -> str:
        if self.prompt:
            segments, info = self.model.transcribe(
                audio,
                beam_size=5 if self.BEAM_SEARCH else 1,
                language=self.LANG if self.LANG else None,
                condition_on_previous_text=False,
                initial_prompt=self.prompt,
                vad_filter=self.vad_filter,
            )
        else:
            segments, info = self.model.transcribe(
                audio,
                beam_size=5 if self.BEAM_SEARCH else 1,
                language=self.LANG if self.LANG else None,
                condition_on_previous_text=False,
                vad_filter=self.vad_filter,
            )
        text = [segment.text for segment in segments]

        if not text:
            return ""
        else:
            return "".join(text)

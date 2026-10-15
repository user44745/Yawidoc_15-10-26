# src/open_llm_vtuber/interpreter/service.py
"""Engines behind the interpreter page (/interpreter), and the quick-phrase
bar of the avatar page.

Doctor speaking:  Thai text or speech -> standard sentences to choose from ->
                  the chosen sentence's Pattani text and audio. No translation.
                  The quick bar offers the sentences chosen most often.
Patient speaking: Pattani / Malay speech -> machine translation into Thai.
                  In test mode the page rates each translation; the ratings
                  are appended to logs/interpreter_ratings.jsonl.

Speech recognition and translation run on this machine. Only the voice needs
the internet: without it the page shows text and plays nothing.

Speech recognition, translation and voices are those of the two translator
characters, so both are configured in one place (characters/*.yaml). They are
loaded the first time they are needed.
"""

import base64
import io
import json
import socket
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from ..asr.asr_factory import ASRFactory
from ..config_manager import read_yaml
from ..config_manager.character import CharacterConfig
from ..translate.thai_romanize import thai_to_latin
from ..translate.translate_factory import TranslateFactory
from ..tts.tts_factory import TTSFactory
from .sentence_bank import NOT_FOUND, SentenceBank

DOCTOR_CHARACTER = "characters/translator_thai_pattani.yaml"
PATIENT_CHARACTER = "characters/translator_pattani_thai.yaml"
SAMPLE_RATE = 16000
# The voice service (edge-tts). Reaching it is what "online" means here.
VOICE_HOST = ("speech.platform.bing.com", 443)
NETWORK_CHECK_SECONDS = 5  # how long one check is trusted
RATINGS_FILE = Path("logs") / "interpreter_ratings.jsonl"
RATINGS = ("correct", "minor_error", "critical_error")
RATEABLE = 200  # translations kept in memory so that they can be rated
# Quick bar: the sentences chosen most often, then the starters. There is one
# bar for each way of translating: Thai sentences that are said in Pattani
# (the doctor's), and Pattani sentences that are said in Thai (the patient's).
THAI_PATTANI = "thai_pattani"
PATTANI_THAI = "pattani_thai"
DIRECTIONS = (THAI_PATTANI, PATTANI_THAI)
QUICK_FILE = Path("pattani_dataset") / "quick_sentences.json"  # starters (Thai text)
USAGE_FILE = Path("logs") / "interpreter_usage.json"  # times each was chosen
QUICK_COUNT = 12


def decode_wav(audio_base64: str) -> np.ndarray:
    """A base64 WAV recording as 16 kHz mono float32 samples."""
    samples, rate = sf.read(io.BytesIO(base64.b64decode(audio_base64)), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise ValueError(f"recording must be {SAMPLE_RATE} Hz, got {rate} Hz")
    return samples


class InterpreterService:
    def __init__(self):
        self.bank = SentenceBank()
        self._loading = threading.Lock()
        self._characters: dict[str, CharacterConfig] = {}
        self._asr: dict[str, object] = {}
        self._tts = None
        self._patient_translator = None
        self._online = (0.0, False)  # (time of the check, result)
        self._recent: OrderedDict[str, dict] = OrderedDict()
        self._rating = threading.Lock()
        # Sentences are kept by their Thai text: ids change when the bank is rebuilt.
        self._by_thai = {s.thai: s for s in self.bank.sentences.values()}
        self._starters = self._by_direction(self._read_json(QUICK_FILE), list)
        missing = [
            thai
            for starters in self._starters.values()
            for thai in starters
            if thai not in self._by_thai
        ]
        if missing:
            logger.warning(
                f"{QUICK_FILE}: not in the sentence bank, skipped: {missing}"
            )
        self._usage = self._by_direction(self._read_json(USAGE_FILE), dict)
        self._counting = threading.Lock()

    @staticmethod
    def _read_json(path: Path):
        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.warning(f"Interpreter: could not read {path}: {e}")
            return None

    @staticmethod
    def _by_direction(value, kind: type) -> dict:
        """What a quick-bar file holds, for each way of translating. A file
        written before there were two holds the Thai -> Pattani part only."""
        if isinstance(value, kind) and not (
            kind is dict and any(key in value for key in DIRECTIONS)
        ):
            value = {THAI_PATTANI: value}
        if not isinstance(value, dict):
            value = {}
        return {
            direction: value[direction]
            if isinstance(value.get(direction), kind)
            else kind()
            for direction in DIRECTIONS
        }

    def online(self) -> bool:
        """Whether the voice service can be reached."""
        checked, result = self._online
        if time.monotonic() - checked < NETWORK_CHECK_SECONDS and checked:
            return result
        try:
            socket.create_connection(VOICE_HOST, timeout=2).close()
            result = True
        except OSError:
            result = False
        self._online = (time.monotonic(), result)
        return result

    def network_status(self) -> dict:
        return {"type": "network_status", "online": self.online()}

    def _character(self, path: str) -> CharacterConfig:
        if path not in self._characters:
            self._characters[path] = CharacterConfig.model_validate(
                read_yaml(path)["character_config"]
            )
        return self._characters[path]

    def _recognizer(self, path: str):
        with self._loading:
            if path not in self._asr:
                config = self._character(path).asr_config
                logger.info(f"Interpreter: loading speech recognition of {path}")
                self._asr[path] = ASRFactory.get_asr_system(
                    config.asr_model, **getattr(config, config.asr_model).model_dump()
                )
        return self._asr[path]

    def _voice(self):
        with self._loading:
            if self._tts is None:
                config = self._character(DOCTOR_CHARACTER).tts_config
                self._tts = TTSFactory.get_tts_engine(
                    config.tts_model,
                    **getattr(config, config.tts_model.lower()).model_dump(),
                )
        return self._tts

    def voice(self):
        """A voice that reads Pattani written in Thai script: the one of the
        Thai -> Pattani translator. For characters whose own voice cannot."""
        return self._voice()

    def _speak(self, text: str, name: str) -> str | None:
        """URL of an audio file of the text, or None when no voice answered."""
        path = Path("cache") / f"{name}.mp3"
        if not (path.exists() and path.stat().st_size > 0):
            voice = self._voice()
            needs_internet = (
                self._character(DOCTOR_CHARACTER).tts_config.tts_model == "edge_tts"
            )
            if needs_internet and not self.online():
                return None  # text only, without waiting for the voice to fail
            made = voice.generate_audio(text, name)
            if not made or Path(made).stat().st_size == 0:
                return None
            path = Path(made)
        return "/cache/" + path.name

    # ---------- doctor speaking ----------

    def catalog(self) -> dict:
        return {"type": "categories", "categories": self.bank.categories()}

    def sentences(self, category: str) -> dict:
        return {
            "type": "sentences",
            "category": category,
            "sentences": self.bank.in_category(category),
        }

    def doctor_text(self, text: str, heard: bool = False) -> dict:
        """Standard sentences the doctor may mean. Nothing is played here: the
        doctor reads the candidates and picks one."""
        candidates = self.bank.match(text)
        reply = {
            "type": "candidates",
            "query": text,
            "candidates": [c.to_dict() for c in candidates],
        }
        if heard:
            reply["heard"] = text
        if not candidates:
            reply["message"] = NOT_FOUND
        return reply

    def doctor_audio(self, audio_base64: str) -> dict:
        text = self._recognizer(DOCTOR_CHARACTER).transcribe_np(
            decode_wav(audio_base64)
        )
        text = text.strip()
        logger.info(f"Interpreter: doctor said {text!r}")
        return self.doctor_text(text, heard=True)

    def quick(self, direction: str = THAI_PATTANI) -> dict:
        """The quick bar of one way of translating: the sentences chosen most
        often, most chosen first, filled up with the starter sentences.
        `label` is the sentence in the language it is translated from: Thai
        for Thai -> Pattani, Pattani for Pattani -> Thai."""
        usage, starters = self._usage[direction], self._starters[direction]
        used = sorted(
            (thai for thai in usage if thai in self._by_thai),
            key=lambda thai: -usage[thai],
        )
        starters = [thai for thai in starters if thai in self._by_thai]
        chosen = list(dict.fromkeys([*used, *starters]))[:QUICK_COUNT]
        sentences = []
        for thai in chosen:
            sentence = self._by_thai[thai]
            sentences.append(
                {
                    "id": sentence.id,
                    "thai": thai,
                    "label": thai
                    if direction == THAI_PATTANI
                    else sentence.pattani.replace("-", ""),
                    "check_numbers": sentence.check_numbers,
                    "count": usage.get(thai, 0),
                }
            )
        return {"type": "quick", "direction": direction, "sentences": sentences}

    def _count(self, thai: str, direction: str = THAI_PATTANI) -> None:
        """Remember that this sentence was chosen."""
        with self._counting:
            usage = self._usage[direction]
            usage[thai] = usage.get(thai, 0) + 1
            try:
                USAGE_FILE.parent.mkdir(exist_ok=True)
                draft = USAGE_FILE.with_suffix(".tmp")
                with draft.open("w", encoding="utf-8") as f:
                    json.dump(self._usage, f, ensure_ascii=False, indent=1)
                draft.replace(USAGE_FILE)
            except OSError as e:
                logger.warning(f"Interpreter: could not save {USAGE_FILE}: {e}")

    def quick_phrase(
        self, sentence_id: str, direction: str = THAI_PATTANI
    ) -> dict | None:
        """A quick-bar sentence for the avatar to say, taken from the sentence
        bank as it is; nothing is translated.

        Thai -> Pattani: the Thai is what the user said, the Pattani is said.
        Pattani -> Thai: the Pattani is what the user said, the Thai is said.
        """
        sentence = self.bank.sentences.get(sentence_id)
        if sentence is None:
            return None
        self._count(sentence.thai, direction)
        # hyphens in a reading mark syllables; a voice would pause on them
        pattani = sentence.pattani.replace("-", "")
        latin = thai_to_latin(pattani)
        phrase = {
            "id": sentence.id,
            "direction": direction,
            "thai": sentence.thai,
            "pattani": sentence.pattani,
            "latin": latin,
        }
        if direction == PATTANI_THAI:
            phrase.update(said=pattani, spoken=sentence.thai, display=sentence.thai)
        else:
            phrase.update(
                said=sentence.thai,
                spoken=pattani,
                display=f"{sentence.pattani} \n{latin}",
            )
        return phrase

    def play(self, sentence_id: str) -> dict:
        sentence = self.bank.sentences.get(sentence_id)
        if sentence is None:
            return {"type": "error", "message": f"unknown sentence id: {sentence_id}"}
        self._count(sentence.thai)
        # hyphens in a reading mark syllables; a voice would pause on them
        url = self._speak(
            sentence.pattani.replace("-", ""), f"interpreter_{sentence.id}"
        )
        return {
            "type": "audio_url",
            "id": sentence.id,
            "thai": sentence.thai,
            "pattani": sentence.pattani,
            "latin": thai_to_latin(sentence.pattani.replace("-", "")),
            "check_numbers": sentence.check_numbers,
            "audio_url": url,
        }

    # ---------- patient speaking ----------

    def _translator(self):
        with self._loading:
            if self._patient_translator is None:
                config = self._character(
                    PATIENT_CHARACTER
                ).tts_preprocessor_config.translator_config
                logger.info("Interpreter: loading the Pattani -> Thai translator")
                self._patient_translator = TranslateFactory.get_translator(
                    config.translate_provider,
                    getattr(config, config.translate_provider).model_dump(),
                )
        return self._patient_translator

    def patient_text(self, text: str, source: str = "typed") -> dict:
        thai = self._translator().translate(text) if text else ""
        translation_id = uuid.uuid4().hex
        url = (
            self._speak(thai, f"interpreter_patient_{translation_id}") if thai else None
        )
        self._recent[translation_id] = {"heard": text, "thai": thai, "source": source}
        while len(self._recent) > RATEABLE:
            self._recent.popitem(last=False)
        return {
            "type": "patient_translation",
            "id": translation_id,
            "heard": text,
            "thai": thai,
            "audio_url": url,
            "online": self.online(),
        }

    def patient_audio(self, audio_base64: str) -> dict:
        text = self._recognizer(PATIENT_CHARACTER).transcribe_np(
            decode_wav(audio_base64)
        )
        logger.info(f"Interpreter: patient said {text.strip()!r}")
        return self.patient_text(text.strip(), source="speech")

    def rate(self, translation_id: str, rating: str) -> dict:
        """Test mode: record how good a translation was. Rating the same
        translation again adds a line; the last line for an id counts."""
        translation = self._recent.get(translation_id)
        if translation is None:
            return {"type": "error", "message": "unknown translation id"}
        if rating not in RATINGS:
            return {"type": "error", "message": f"unknown rating: {rating}"}
        row = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "id": translation_id,
            "rating": rating,
            **translation,
        }
        with self._rating:
            RATINGS_FILE.parent.mkdir(exist_ok=True)
            with RATINGS_FILE.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Interpreter: translation {translation_id} rated {rating}")
        return {"type": "rated", "id": translation_id, "rating": rating}


def speaks_pattani(context) -> bool:
    """Whether the character of this session is one of the Thai / Pattani
    translators (its own voice can then read Pattani in Thai script)."""
    character = context.character_config
    provider = character.tts_preprocessor_config.translator_config.translate_provider
    return (
        character.use_translation
        and context.translate_engine is not None
        and provider in ("pattani", "pattani_thai")
    )


def quick_direction(context) -> str:
    """Which quick bar the character of this session gets: the Pattani -> Thai
    translator the Pattani sentences that are said in Thai, every other
    character the Thai sentences that are said in Pattani."""
    character = context.character_config
    provider = character.tts_preprocessor_config.translator_config.translate_provider
    if speaks_pattani(context) and provider == "pattani_thai":
        return PATTANI_THAI
    return THAI_PATTANI


_shared: InterpreterService | None = None
_sharing = threading.Lock()


def shared_service() -> InterpreterService:
    """The service of this server. The interpreter page and the avatar page
    use the same sentence bank and the same quick bar."""
    global _shared
    with _sharing:
        if _shared is None:
            _shared = InterpreterService()
    return _shared

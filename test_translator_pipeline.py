#!/usr/bin/env python
"""
test_translator_pipeline.py — headless end-to-end test of the
"pattani_thai_translator" character's translation pipeline:

  edge-tts (Malay speech) -> faster-whisper ASR (language "ms")
  -> Malay text -> "pattani_thai" translator (hospital phrase table,
     NLLB zsm_Latn -> tha_Thai, LLM review) -> Thai text

Prerequisite: Ollama running with the configured model (for the LLM review).

It validates the character yaml through the project's pydantic config
manager, then initializes the exact engines the server would use.

Run:  uv run python test_translator_pipeline.py
"""

import asyncio
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
# Match run_server.py: model caches live under ./models
os.environ.setdefault("HF_HOME", str(ROOT / "models"))


MEDICAL_CASES = [
    ("Encik pun rasa mahkamah mana hari ini", "คุณรู้สึกเป็นยังไงบ้างวันนี้"),
    ("Tekan di sini sebental", "กดแผลตรงนี้ไว้สักครู่"),
    ("Rasa loya.", "คลื่นไส้"),
    ("Hulurkan tangan kiri", "ยื่นแขนซ้ายมา"),
    # a negation must survive
    ("Saya tak ada demam", lambda thai: "ไม่" in thai and "ไข้" in thai),
    # what Whisper invents for unclear audio is dropped, not translated
    ("Terima kasih kerana menonton!", ""),
]


def main() -> int:
    # ---------- 1. validate the character config through the config manager ----------
    from src.open_llm_vtuber.config_manager import read_yaml
    from src.open_llm_vtuber.config_manager.character import CharacterConfig

    print("[1/5] Validating characters/translator_pattani_thai.yaml ...")
    raw = read_yaml(str(ROOT / "characters" / "translator_pattani_thai.yaml"))
    cfg = CharacterConfig.model_validate(raw["character_config"])
    print(f"      OK — conf_uid={cfg.conf_uid}, use_translation={cfg.use_translation}, "
          f"asr={cfg.asr_config.asr_model}, provider="
          f"{cfg.tts_preprocessor_config.translator_config.translate_provider}")

    # ---------- 2. generate a Malay speech sample with edge-tts ----------
    import edge_tts

    print("[2/5] Generating Malay speech sample with edge-tts (ms-MY-OsmanNeural) ...")
    sample = ROOT / "cache" / "test_malay_sample.mp3"
    sample.parent.mkdir(parents=True, exist_ok=True)

    async def synth() -> None:
        communicate = edge_tts.Communicate(
            "Selamat pagi, saya nak pergi ke pasar untuk membeli ikan segar.",
            "ms-MY-OsmanNeural",
        )
        await communicate.save(str(sample))

    asyncio.run(synth())
    print(f"      OK — {sample} ({sample.stat().st_size / 1024:.0f} KiB)")

    # ---------- 3. init ASR (same call the server makes) and transcribe ----------
    from faster_whisper.audio import decode_audio
    from src.open_llm_vtuber.asr.asr_factory import ASRFactory

    print(f"[3/5] Loading faster-whisper '{cfg.asr_config.faster_whisper.model_path}' ...")
    asr_engine = ASRFactory.get_asr_system(
        cfg.asr_config.asr_model,
        **getattr(cfg.asr_config, cfg.asr_config.asr_model).model_dump(),
    )
    audio = decode_audio(str(sample), sampling_rate=16000)
    malay_text = asr_engine.transcribe_np(audio)
    print(f"      OK — ASR output (Malay): {malay_text!r}")
    if not malay_text.strip():
        print("❌ ASR returned empty text", file=sys.stderr)
        return 1

    # ---------- 4. init the translator (same call the server makes) and translate ----------
    from src.open_llm_vtuber.translate.translate_factory import TranslateFactory

    tcfg = cfg.tts_preprocessor_config.translator_config
    print(f"[4/5] Loading '{tcfg.translate_provider}' translator ...")
    translator = TranslateFactory.get_translator(
        tcfg.translate_provider, getattr(tcfg, tcfg.translate_provider).model_dump()
    )
    thai_text = translator.translate(malay_text)
    print(f"      OK — translator output (Thai): {thai_text!r}")
    if not thai_text.strip():
        print("❌ translator returned no Thai", file=sys.stderr)
        return 1

    # things the recognizer really wrote for hospital phrases, and what must come out
    for heard, wanted in MEDICAL_CASES:
        got = translator.translate(heard)
        ok = got == wanted if isinstance(wanted, str) else wanted(got)
        print(f"      {'OK' if ok else 'WRONG'} — {heard!r} => {got!r}")
        if not ok:
            print("❌ a hospital phrase was not translated as expected", file=sys.stderr)
            return 1

    # ---------- 5. synthesize the Thai translation with the configured TTS voice ----------
    tts_voice = cfg.tts_config.edge_tts.voice
    print(f"[5/5] Synthesizing Thai output with edge-tts ({tts_voice}) ...")

    async def synth_thai() -> None:
        communicate = edge_tts.Communicate(thai_text, tts_voice)
        await communicate.save(str(sample.with_suffix(".thai.mp3")))

    asyncio.run(synth_thai())
    out = sample.with_suffix(".thai.mp3")
    print(f"      OK — {out} ({out.stat().st_size / 1024:.0f} KiB)")

    print("\n==================== END-TO-END TEST PASSED ====================")
    print("Malay in : Selamat pagi, saya nak pergi ke pasar untuk membeli ikan segar.")
    print(f"ASR heard: {malay_text.strip()}")
    print(f"Thai out : {thai_text.strip()}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)

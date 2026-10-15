#!/usr/bin/env python
"""
test_thai_pattani.py — headless end-to-end test of the
"thai_pattani_translator" character's translation pipeline:

  edge-tts (Thai speech) -> faster-whisper ASR (language "th") -> Thai text
  -> "pattani" translator (phrasebook / NLLB draft / sound rules / LLM consult)
  -> Pattani Malay in Thai script -> edge-tts (Thai voice)

It validates the character yaml through the project's pydantic config
manager, then initializes the exact engines the server would use.

Prerequisites: Ollama running with the configured model (for the LLM
consult step) and pattani_dataset/phrasebook.jsonl (build_phrasebook.py).

Run:  uv run python test_thai_pattani.py
"""

import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
# Match run_server.py: model caches live under ./models
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

# Windows consoles default to a legacy codepage (cp874) that can't encode
# every character printed here — force UTF-8 so results print cleanly.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

SPOKEN_THAI = "วันนี้ปวดท้องไหมคะ"
# typed inputs: the first one is a phrasebook sentence, the rest are unseen
TEXT_INPUTS = [
    "มาอยู่โรงพยาบาลกี่วันแล้วคะ",
    "สวัสดี",
    "ปวดหัวไหมคะ",
    "กินยาหลังอาหารนะคะ",
]
# Standard Malay word -> spelling found in pattani_dataset/pairs.jsonl
ATTESTED_WORDS = {
    "makan": "มาแก",
    "ada": "อาดอ",
    "sakit": "ซาเกะ",
    "perut": "ปือโระ",
    "demam": "ดือแม",
    "batuk": "บาโตะ",
    "rumah": "รูเมาะห์",
    "lepas": "ลือปะห์",
    "belum": "บือลง",
    "lebih": "ลือเบะห์",
    "puluh": "ปูโละห์",
    "tidur": "ตีโด",
    "anak": "อาเนาะ",
    "rokok": "รอเกาะ",
    "tinggi": "ตีงี",
    "sampai": "ซาปา",
    "jantung": "ยาตง",
    "nanti": "นาตี",
    "pening": "ปือนิง",
    "dua": "ดูวอ",
}
LATIN = re.compile(r"[A-Za-z]")
THAI = re.compile(r"[฀-๿]")


def main() -> int:
    # ---------- 0. sound rules reproduce the spellings in the transcripts ----------
    from src.open_llm_vtuber.translate.pattani_script import word_to_pattani

    wrong = {
        malay: word_to_pattani(malay)
        for malay, pattani in ATTESTED_WORDS.items()
        if word_to_pattani(malay) != pattani
    }
    print(
        f"[0/5] Sound rules: {len(ATTESTED_WORDS) - len(wrong)}/{len(ATTESTED_WORDS)} "
        "attested words reproduced"
    )
    if wrong:
        print(f"❌ sound rules disagree with the transcripts: {wrong}", file=sys.stderr)
        return 1

    # ---------- 1. validate the character config through the config manager ----------
    from src.open_llm_vtuber.config_manager import read_yaml
    from src.open_llm_vtuber.config_manager.character import CharacterConfig

    print("[1/5] Validating characters/translator_thai_pattani.yaml ...")
    raw = read_yaml(str(ROOT / "characters" / "translator_thai_pattani.yaml"))
    cfg = CharacterConfig.model_validate(raw["character_config"])
    tcfg = cfg.tts_preprocessor_config.translator_config
    print(
        f"      OK — conf_uid={cfg.conf_uid}, use_translation={cfg.use_translation}, "
        f"asr={cfg.asr_config.asr_model}, provider={tcfg.translate_provider}"
    )

    # ---------- 2. generate a Thai speech sample with the configured TTS ----------
    from src.open_llm_vtuber.tts.tts_factory import TTSFactory

    tts_voice = cfg.tts_config.edge_tts.voice
    print(f"[2/5] Generating Thai speech sample with edge-tts ({tts_voice}) ...")
    tts_engine = TTSFactory.get_tts_engine(
        cfg.tts_config.tts_model,
        **getattr(cfg.tts_config, cfg.tts_config.tts_model.lower()).model_dump(),
    )
    sample = tts_engine.generate_audio(SPOKEN_THAI, "test_thai_sample")
    if not sample:
        print("❌ TTS produced no audio for the speech sample", file=sys.stderr)
        return 1
    sample = Path(sample)
    print(f"      OK — {sample} ({sample.stat().st_size / 1024:.0f} KiB)")

    # ---------- 3. init ASR (same call the server makes) and transcribe ----------
    from faster_whisper.audio import decode_audio
    from src.open_llm_vtuber.asr.asr_factory import ASRFactory

    fw = cfg.asr_config.faster_whisper
    print(
        f"[3/5] Loading faster-whisper '{fw.model_path}' (language={fw.language}) ..."
    )
    asr_engine = ASRFactory.get_asr_system(
        cfg.asr_config.asr_model,
        **getattr(cfg.asr_config, cfg.asr_config.asr_model).model_dump(),
    )
    heard = asr_engine.transcribe_np(decode_audio(str(sample), sampling_rate=16000))
    print(f"      OK — ASR output (Thai): {heard!r}")
    if not heard.strip():
        print("❌ ASR returned empty text", file=sys.stderr)
        return 1

    # ---------- 4. init the translator (same call the server makes) and translate ----------
    from src.open_llm_vtuber.translate.translate_factory import TranslateFactory

    print(
        f"[4/5] Loading '{tcfg.translate_provider}' translator "
        f"(NLLB {tcfg.pattani.nllb_model}, LLM consult="
        f"{tcfg.pattani.llm_model if tcfg.pattani.llm_consult else 'off'}) ..."
    )
    translator = TranslateFactory.get_translator(
        tcfg.translate_provider, getattr(tcfg, tcfg.translate_provider).model_dump()
    )
    results = [(text, translator.translate(text)) for text in [heard] + TEXT_INPUTS]
    # what the chat shows: the spoken Thai-script line, then Latin letters
    shown = [translator.display_text(pattani).split("\n") for _, pattani in results]
    for (text, pattani), lines in zip(results, shown):
        print(f"      {text.strip()}  =>  {pattani}  |  {lines[-1].strip()}")
    # every output must be Pattani in Thai script, never the Thai input handed back
    not_translated = [
        text
        for text, pattani in results
        if not THAI.search(pattani)
        or LATIN.search(pattani)
        or pattani.strip() == text.strip()
    ]
    # ... and be shown a second time in Latin letters only
    not_translated += [
        text
        for (text, pattani), lines in zip(results, shown)
        if len(lines) != 2
        or lines[0].strip() != pattani
        or THAI.search(lines[1])
        or not LATIN.search(lines[1])
    ]

    # ---------- 5. every translation must be spoken by the configured TTS ----------
    print(
        f"[5/5] Synthesizing all {len(results)} Pattani outputs with "
        f"{cfg.tts_config.tts_model} ({tts_voice}) ..."
    )
    silent = []
    for n, (_, pattani) in enumerate(results):
        out = tts_engine.generate_audio(pattani, f"test_thai_sample.pattani{n}")
        if out and Path(out).stat().st_size > 0:
            print(f"      OK — {out} ({Path(out).stat().st_size / 1024:.0f} KiB)")
        else:
            silent.append(pattani)
    if silent:
        print(f"❌ TTS produced no audio for: {silent}", file=sys.stderr)
        return 1

    if not_translated:
        print(f"\n❌ not translated to Pattani: {not_translated}", file=sys.stderr)
        return 1
    print("\n==================== END-TO-END TEST PASSED ====================")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)

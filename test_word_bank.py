#!/usr/bin/env python
"""
test_word_bank.py — run every row of the word bank
(data_thai_to_pattani/*.csv) through both translator characters and list
what does not come out as the word bank says.

  Thai -> Pattani:  the Thai meaning must give the reading of the row
                    (rows whose reading is Standard Malay pronunciation only
                    have to give Pattani in Thai script)
  Pattani -> Thai:  the Malay / Yawi word must give one of the Thai meanings

Run:  uv run python test_word_bank.py            typed text
      uv run python test_word_bank.py --speech   spoken: edge-tts voice ->
                                                 speech recognition -> translator
Prerequisites: uv run python build_phrasebook.py, and Ollama running.
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
# Match run_server.py: model caches live under ./models
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

# Windows consoles default to a legacy codepage (cp874) that can't encode
# every character printed here — force UTF-8 so results print cleanly.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from build_phrasebook import (  # noqa: E402
    DIALECT_CATEGORIES,
    DIALECT_CATEGORY_NAMES,
    SKIPPED_MALAY,
    clean_reading,
    loose,
    malay_choices,
    read_word_bank,
    thai_choices,
)
from src.open_llm_vtuber.translate.pattani_script import thai_key  # noqa: E402

THAI = re.compile(r"[฀-๿]")
LATIN = re.compile(r"[A-Za-z]")


def load(preset: str):
    """The recognizer and translator of a character, built as the server does."""
    from src.open_llm_vtuber.asr.asr_factory import ASRFactory
    from src.open_llm_vtuber.config_manager import read_yaml
    from src.open_llm_vtuber.config_manager.character import CharacterConfig
    from src.open_llm_vtuber.translate.translate_factory import TranslateFactory

    cfg = CharacterConfig.model_validate(
        read_yaml(str(ROOT / "characters" / preset))["character_config"]
    )
    tcfg = cfg.tts_preprocessor_config.translator_config
    translator = TranslateFactory.get_translator(
        tcfg.translate_provider, getattr(tcfg, tcfg.translate_provider).model_dump()
    )

    def recognizer():
        return ASRFactory.get_asr_system(
            cfg.asr_config.asr_model,
            **getattr(cfg.asr_config, cfg.asr_config.asr_model).model_dump(),
        )

    return translator, recognizer


def speak(text: str, voice: str, path: Path) -> bool:
    """A recording of the text by an edge-tts voice, kept between runs."""
    import asyncio

    import edge_tts

    if path.exists() and path.stat().st_size > 1500:
        return True
    for _ in range(5):
        try:
            asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))
            if path.stat().st_size > 1500:
                return True
        except Exception:
            pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--speech", action="store_true", help="go through speech recognition"
    )
    args = ap.parse_args()

    banks = sorted((ROOT / "data_thai_to_pattani").glob("*.csv"))
    if not banks:
        print("❌ no word bank CSV in data_thai_to_pattani/", file=sys.stderr)
        return 1
    rows = read_word_bank(banks[0])
    print(f"Word bank: {banks[0].name} ({len(rows)} rows)")
    clips = ROOT / "cache" / "word_bank_clips"
    if args.speech:
        from faster_whisper.audio import decode_audio

        clips.mkdir(parents=True, exist_ok=True)

    # ---------- Thai -> Pattani ----------
    translator, recognizer = load("translator_thai_pattani.yaml")
    hear = recognizer().transcribe_np if args.speech else None
    total = right = 0
    wrong = []
    done = set()
    for n, row in enumerate(rows):
        dialect = row["category"] in DIALECT_CATEGORY_NAMES or any(
            mark in row["category"] for mark in DIALECT_CATEGORIES
        )
        reading = clean_reading(row["reading"])
        for k, thai in enumerate(thai_choices(row["thai"])):
            if thai.isdigit() or "..." in thai or thai_key(thai) in done:
                continue
            if any(
                m.lower().strip("?") in SKIPPED_MALAY
                for m in malay_choices(row["malay"])
            ):
                continue
            done.add(thai_key(thai))
            said = thai
            if hear:
                clip = clips / f"th_{n:03d}_{k}.mp3"
                if not speak(thai, "th-TH-NiwatNeural", clip):
                    continue
                said = hear(decode_audio(str(clip), sampling_rate=16000)).strip()
            out = translator.translate(said) if said else ""
            total += 1
            is_pattani = (
                bool(THAI.search(out)) and not LATIN.search(out) and out != said
            )
            if dialect and "..." not in reading:
                ok = loose(out) == loose(reading)
            else:
                ok = is_pattani
            right += ok
            if not ok:
                heard = f" (heard: {said})" if hear else ""
                wrong.append(
                    f"   {thai}{heard} -> {out!r}   word bank: {reading} [{row['malay']}]"
                )
    print(f"\nThai -> Pattani: {right}/{total} as in the word bank")
    print("\n".join(wrong))
    thai_ok = right == total
    del translator, hear

    # ---------- Pattani -> Thai ----------
    translator, recognizer = load("translator_pattani_thai.yaml")
    hear = recognizer().transcribe_np if args.speech else None
    # every Thai meaning the word bank gives for a Malay phrase
    meanings: dict[str, set[str]] = {}
    for row in rows:
        for malay in malay_choices(row["malay"]):
            key = re.sub(r"[^a-z0-9]+", "", malay.lower())
            meanings.setdefault(key, set()).update(
                thai_key(t) for t in thai_choices(row["thai"])
            )
    total = right = 0
    wrong = []
    done = set()
    for n, row in enumerate(rows):
        for k, malay in enumerate(malay_choices(row["malay"])):
            key = re.sub(r"[^a-z0-9]+", "", malay.lower())
            if (
                "..." in malay
                or key in done
                or malay.lower().strip("?") in SKIPPED_MALAY
            ):
                continue
            done.add(key)
            said = malay
            if hear:
                clip = clips / f"ms_{n:03d}_{k}.mp3"
                if not speak(malay, "ms-MY-YasminNeural", clip):
                    continue
                said = hear(decode_audio(str(clip), sampling_rate=16000)).strip()
            out = translator.translate(said) if said else ""
            total += 1
            ok = thai_key(out) in meanings[key]
            right += ok
            if not ok:
                heard = f" (heard: {said})" if hear else ""
                wrong.append(f"   {malay}{heard} -> {out!r}   word bank: {row['thai']}")
    print(f"\nPattani -> Thai: {right}/{total} as in the word bank")
    print("\n".join(wrong))

    if not args.speech and not (thai_ok and right == total):
        print(
            "\n❌ typed word-bank entries must all come out as listed", file=sys.stderr
        )
        return 1
    print("\n==================== WORD BANK TEST DONE ====================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

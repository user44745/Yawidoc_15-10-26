#!/usr/bin/env python
"""
calibrate_matcher.py — set the doctor-mode matching threshold from test data.

Below the threshold the interpreter page answers "sentence not found; please
use an interpreter" and plays nothing. The threshold is not chosen by hand:
it is the score that best separates, in pattani_dataset/matcher_testset.jsonl,

  - Thai that means a standard sentence (typed with a polite ending, reworded,
    or as speech recognition heard it), from
  - Thai that is not in the sentence bank at all.

    uv run python calibrate_matcher.py                  calibrate
    uv run python calibrate_matcher.py --collect        rebuild the test set first
                                                        (edge-tts voice + speech
                                                        recognition; needs the GPU)

Output: pattani_dataset/matcher_threshold.json
Run it again whenever the sentence bank changes (after build_phrasebook.py).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "pattani_dataset"
TESTSET = DATA / "matcher_testset.jsonl"

# Thai a doctor or nurse may say that the sentence bank does not contain.
ABSENT = [
    "ห้องน้ำอยู่ทางไหน", "ค่ารักษาทั้งหมดเท่าไหร่", "ต้องผ่าตัดพรุ่งนี้เช้า", "ญาติเข้าเยี่ยมได้ตอนบ่าย",
    "กรุณาลงชื่อตรงนี้", "ต้องเอกซเรย์ปอดก่อน", "งดน้ำงดอาหารหลังเที่ยงคืน", "ขอดูบัตรประชาชนหน่อย",
    "วันนี้อากาศดีมาก", "รถติดมากเลยวันนี้", "เดี๋ยวจะวัดความดันให้", "ถอดรองเท้าก่อนขึ้นเตียง",
    "ต้องใส่สายน้ำเกลือ", "ผลตรวจเลือดปกติดี", "ห้ามลุกจากเตียงเอง", "กดปุ่มนี้ถ้าต้องการความช่วยเหลือ",
    "ต้องตรวจคลื่นหัวใจ", "มีประกันสุขภาพไหม", "โทรศัพท์ของคุณดังอยู่", "พรุ่งนี้กลับบ้านได้แล้ว",
    "ต้องทำแผลทุกวัน", "อย่าให้แผลโดนน้ำ", "น้ำหนักเท่าไหร่", "ส่วนสูงเท่าไหร่",
    "ตั้งครรภ์อยู่หรือเปล่า", "ประจำเดือนมาครั้งสุดท้ายเมื่อไหร่", "ใส่ฟันปลอมหรือเปล่า",
    "ต้องเปลี่ยนเป็นชุดผู้ป่วย", "ร้านอาหารอยู่ชั้นล่าง", "ที่จอดรถเต็มแล้ว", "ฝนตกหนักมากเมื่อเช้า",
    "ช่วยปิดโทรศัพท์ด้วย", "กรอกแบบฟอร์มนี้ให้ครบ", "ต้องฉีดวัคซีนบาดทะยัก", "จะส่งตัวไปโรงพยาบาลศูนย์",
    "ต้องเจาะน้ำไขสันหลัง", "มองตามนิ้วมือของหมอ", "ลองเดินให้ดูหน่อย", "กลืนน้ำลายได้ไหม",
    "ได้ยินเสียงชัดไหม", "มองเห็นตัวหนังสือนี้ไหม", "ขยับนิ้วเท้าให้ดูหน่อย", "ทำงานอะไร",
    "แต่งงานหรือยัง", "มีลูกกี่คน", "นอนตะแคงซ้าย", "กลั้นหายใจไว้", "เบ่งเหมือนถ่ายอุจจาระ",
    "รอเรียกชื่อที่หน้าห้อง", "เชิญรับบัตรคิว", "ชำระเงินที่ช่องสาม", "กินอาหารอ่อนๆ ก่อนนะ",
    "ห้ามกินของหมักดอง", "ต้องออกกำลังกายสม่ำเสมอ", "ลดอาหารเค็ม", "พักผ่อนให้เพียงพอ",
    "ล้างมือบ่อยๆ", "ใส่หน้ากากอนามัยด้วย", "ขอเบอร์โทรศัพท์ญาติ", "แพทย์เวรจะมาดูอาการคืนนี้",
]  # fmt: skip
# The same sentence with another number: must be flagged, whatever its score.
NUMBER_VARIANTS = [
    "วันละสองครั้ง", "วันละสี่ครั้ง", "ทานยานี้วันละ 2 ครั้ง เช้า กลางวัน เย็น", "วันละห้าครั้ง",
]  # fmt: skip
POLITE = ["ครับ", "ค่ะ", "นะคะ", "คะ"]
# categories whose sentences are instructions to the patient
INSTRUCTIONS = ("คำสั่ง", "เจาะเลือด", "ห้องยา")


def collect() -> None:
    """Rebuild the test set from the sentence bank."""
    import asyncio

    import edge_tts
    from faster_whisper.audio import decode_audio

    from src.open_llm_vtuber.asr.asr_factory import ASRFactory
    from src.open_llm_vtuber.config_manager import read_yaml
    from src.open_llm_vtuber.config_manager.character import CharacterConfig
    from src.open_llm_vtuber.interpreter.sentence_bank import SentenceBank

    bank = SentenceBank()
    cfg = CharacterConfig.model_validate(
        read_yaml(str(ROOT / "characters" / "translator_thai_pattani.yaml"))[
            "character_config"
        ]
    )
    asr = ASRFactory.get_asr_system(
        cfg.asr_config.asr_model,
        **getattr(cfg.asr_config, cfg.asr_config.asr_model).model_dump(),
    )
    clips = ROOT / "cache" / "matcher_clips"
    clips.mkdir(parents=True, exist_ok=True)

    def heard(text: str, name: str) -> str | None:
        clip = clips / f"{name}.mp3"
        for _ in range(5):
            if clip.exists() and clip.stat().st_size > 1500:
                break
            try:
                asyncio.run(
                    edge_tts.Communicate(text, "th-TH-NiwatNeural").save(str(clip))
                )
            except Exception:
                pass
        else:
            return None
        return asr.transcribe_np(decode_audio(str(clip), sampling_rate=16000)).strip()

    rows = []
    for n, sentence in enumerate(bank.sentences.values()):
        for k, wording in enumerate([sentence.thai, *sentence.variants]):
            polite = wording.rstrip("?") + POLITE[(n + k) % len(POLITE)]
            rows.append({"kind": "typed", "query": polite, "expected": sentence.id})
            reworded = []
            if "ไหม" in wording:
                reworded.append(wording.replace("ไหม", "มั้ย"))
            if any(mark in sentence.category for mark in INSTRUCTIONS):
                # the same instruction, asked politely
                reworded.append(f"ช่วย{wording}หน่อยครับ")
                reworded.append(f"กรุณา{wording}ด้วยค่ะ")
            rows += [
                {"kind": "reworded", "query": text, "expected": sentence.id}
                for text in reworded
            ]
            said = heard(wording, f"{sentence.id}_{k}")
            if said:
                rows.append({"kind": "spoken", "query": said, "expected": sentence.id})
        if n % 50 == 0:
            print(f"  {n}/{len(bank.sentences)} sentences", flush=True)
    for n, text in enumerate(ABSENT):
        rows.append({"kind": "absent", "query": text, "expected": None})
        said = heard(text, f"absent_{n}")
        if said:
            rows.append({"kind": "absent spoken", "query": said, "expected": None})
    rows += [
        {"kind": "number variant", "query": t, "expected": None}
        for t in NUMBER_VARIANTS
    ]
    with TESTSET.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"Wrote {len(rows)} test sentences -> {TESTSET}")


def calibrate() -> int:
    from src.open_llm_vtuber.interpreter.sentence_bank import SentenceBank

    if not TESTSET.exists():
        print(f"❌ {TESTSET} not found; run with --collect first", file=sys.stderr)
        return 1
    bank = SentenceBank()
    with TESTSET.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    present, absent, unflagged = [], [], []
    for row in rows:
        ranked = bank.ranked(row["query"])[:3]
        if row["kind"] == "number variant":
            # (c): every candidate for it must be marked as a different number
            unflagged += [row["query"] for c in ranked[:1] if not c.number_mismatch]
        elif row["expected"] is None:
            absent.append(ranked[0].score if ranked else 0.0)
        else:
            # the score at which the right sentence is among the three offered
            score = next((c.score for c in ranked if c.id == row["expected"]), 0.0)
            present.append(score)

    best = None
    for threshold in sorted({round(s, 3) for s in present + absent if s > 0}):
        found = sum(s >= threshold for s in present) / len(present)
        refused = sum(s < threshold for s in absent) / len(absent)
        # both kinds of mistake count the same; on a tie the stricter threshold
        if best is None or (found + refused, threshold) >= (best[1] + best[2], best[0]):
            best = (threshold, found, refused)
    threshold, found, refused = best
    result = {
        "threshold": threshold,
        "method": "score that maximizes (standard sentences found + absent sentences refused)",
        "test_set": TESTSET.name,
        "standard_sentence_queries": len(present),
        "found_at_threshold": round(found, 4),
        "absent_queries": len(absent),
        "refused_at_threshold": round(refused, 4),
        "number_variants_not_flagged": unflagged,
        "sentences_in_bank": len(bank.sentences),
    }
    out = DATA / "matcher_threshold.json"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    kinds = {}
    for row in rows:
        if row["expected"]:
            ranked = bank.ranked(row["query"])[:3]
            ok = any(c.id == row["expected"] and c.score >= threshold for c in ranked)
            kinds.setdefault(row["kind"], []).append(ok)
    for kind, oks in kinds.items():
        print(f"  {kind}: right sentence offered for {sum(oks)}/{len(oks)}")
    print(f"Wrote {out}")
    return 1 if unflagged else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="rebuild the test set first")
    args = ap.parse_args()
    if args.collect:
        collect()
    sys.exit(calibrate())

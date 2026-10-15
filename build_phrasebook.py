#!/usr/bin/env python
"""
build_phrasebook.py — build the reference data of the two translator characters
from the human-made sources.

Inputs:
    pattani_dataset/pairs.jsonl     interview pairs (from parse_transcripts.py)
    pattani_dataset/glossary.jsonl  hand-maintained terms, {"thai", "pattani"};
                                    "/" in the Thai side separates synonyms
    data_thai_to_pattani/*.csv      word bank: category, Malay / Yawi word,
                                    reading in Thai letters, Thai meaning

Outputs (one JSON object per line):
    pattani_dataset/phrasebook.jsonl        Thai -> Pattani (Thai script)
        {"thai", "pattani", "kind": "sentence" | "term", "source"}
    pattani_dataset/medical_phrases.jsonl   Malay / Pattani (Latin) -> Thai
        {"malay", "thai"}
    pattani_dataset/lexicon.json            Malay word -> Pattani (Thai script),
        only words whose dialect form the sound rules cannot derive
    pattani_dataset/standard_sentences.jsonl  the doctor's sentence bank
        {"id", "category", "thai", "variants", "pattani", "source"}
        Only Pattani text written by a person: word-bank readings, glossary
        and the interviewer's lines. Nothing here comes from the sound rules.

To correct a word or a sentence, edit the CSV (or glossary.jsonl) and run:
    uv run python build_phrasebook.py
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.open_llm_vtuber.translate import pattani_script  # noqa: E402
from src.open_llm_vtuber.translate.pattani_script import (  # noqa: E402
    BUILTIN_LEXICON,
    malay_to_pattani,
    thai_key,
)

# Word-bank categories whose reading column is Pattani dialect. In the other
# categories the reading is Standard Malay pronunciation, so the Pattani form
# is derived from the Malay word by the sound rules instead.
DIALECT_CATEGORIES = ("ถิ่น", "บุคคลและครอบครัว", "การกินและการใช้ชีวิต", "ตัวเลข", "บทสนทนา")
DIALECT_CATEGORY_NAMES = {"คำสั่งและการปฏิบัติ"}

# Rows of the word bank that are not used, with the reason. They are printed
# at the end of every run so they can be checked and corrected in the CSV.
SKIPPED_MALAY = {
    "anak lembu": "means a young cow; the calf of the leg is 'betis'",
    "limpa": "means the spleen, not the liver",
    "berapa jubu lah": "reading is garbled",
}
# Corrections applied on top of the word bank.
THAI_FOR_MALAY = {"hati": "ตับ"}  # listed under both heart and liver; it is the liver
READING_FIXES = {"ซาเกะ มเอนอ": "ซาเกะ มานอ"}  # typo; the interviews say ซาเกะมานอ

# Thai meanings with a choice written inside a word, and what each choice is.
THAI_CHOICES = {
    "ยื่นแขนขวา/ซ้ายมา": ["ยื่นแขนขวามา", "ยื่นแขนซ้ายมา"],
    "สวัสดีตอนบ่าย/เที่ยง": ["สวัสดีตอนบ่าย", "สวัสดีตอนเที่ยง"],
    "ปวด/เจ็บอะไร (เป็นอะไรมา)": ["เป็นอะไรมา", "ปวดอะไร", "เจ็บอะไร"],
}
POLITE = re.compile(r"\s*(ครับ/ค่ะ|ครับ/คะ|ครับ|ค่ะ|คะ)$")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_word_bank(path: Path) -> list[dict]:
    """Rows of the word bank; a comma inside the Malay text shifts the columns."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for fields in list(csv.reader(f))[1:]:
            fields = [x.strip() for x in fields]
            while fields and not fields[-1]:
                fields.pop()
            if len(fields) < 4:
                continue
            rows.append(
                {
                    "category": fields[0],
                    "malay": ", ".join(fields[1:-2]),
                    "reading": fields[-2],
                    "thai": fields[-1],
                }
            )
    return rows


def thai_choices(thai: str) -> list[str]:
    """The separate Thai meanings written in one cell."""
    thai = POLITE.sub("", thai.strip())
    if thai in THAI_CHOICES:
        return THAI_CHOICES[thai]
    number = re.fullmatch(r"(\d+) \((.+)\)", thai)
    if number:  # "4 (สี่)"
        return [number.group(2), number.group(1)]
    choices = []
    for part in re.split(r"\s+/\s+|,\s+", thai):
        extra = re.fullmatch(r"(.+?)\s*\((.+)\)", part)  # "มีไข้ (ตัวร้อน)"
        choices += [extra.group(1), extra.group(2)] if extra else [part]
    choices = [POLITE.sub("", c).strip() for c in choices if c.strip()]
    if choices and choices[-1].endswith("ไหม"):  # "เวียนหัว / มึนหัวไหม" asks twice
        choices = [c if c.endswith("ไหม") else c + "ไหม" for c in choices]
    return choices


def malay_choices(malay: str) -> list[str]:
    """The separate Malay phrases written in one cell."""
    choices = []
    for part in re.split(r"\s+/\s+", malay):
        words = part.split()
        slashed = [i for i, w in enumerate(words) if "/" in w.strip("/")]
        if len(slashed) == 1:  # "Hulurkan tangan kanan/kiri"
            i = slashed[0]
            choices += [
                " ".join(words[:i] + [w] + words[i + 1 :]) for w in words[i].split("/")
            ]
        else:
            choices.append(part)
    return [c.strip() for c in choices if c.strip() and c.strip() != "-"]


def clean_reading(reading: str) -> str:
    reading = reading.split(" / ")[0].replace("_", "").strip(" ?")
    return READING_FIXES.get(reading, reading)


def loose(text: str) -> str:
    """Spelling habits that do not change the sound: ฮ / ห์ at the end, ฆ, ญ, tone marks."""
    text = re.sub("[่-๋็์ฺ\\s\\-_.?]", "", text)
    text = text.replace("ฆ", "ก").replace("ญ", "ย")
    return re.sub("[ฮห]$", "", text)


def build(pairs_path: Path, glossary_path: Path, bank_path: Path | None):
    phrasebook: list[dict] = []
    seen: set[str] = set()
    notes: list[str] = []

    def add(thai: str, pattani: str, kind: str, source: str) -> None:
        thai, pattani = " ".join(thai.split()), " ".join(pattani.split())
        key = thai_key(thai)
        # skip empty sides, lines that were already Thai, and a Thai sentence
        # that an earlier (more trusted) source already translates
        if not thai or not pattani or thai == pattani or key in seen:
            return
        seen.add(key)
        phrasebook.append(
            {"thai": thai, "pattani": pattani, "kind": kind, "source": source}
        )

    for row in read_jsonl(pairs_path):
        if row.get("has_parallel"):
            add(
                row.get("thai") or "",
                row.get("malay_thai") or "",
                "sentence",
                f"interview p{row['person']}t{row['tape']}:{row['line_no']}",
            )
    if glossary_path.exists():
        for row in read_jsonl(glossary_path):
            for thai in row["thai"].split("/"):
                add(thai, row["pattani"], "term", "glossary")

    phrases: list[dict] = []
    malay_seen: set[str] = set()
    lexicon: dict[str, str] = {}
    bank = []
    for row in read_word_bank(bank_path) if bank_path else []:
        malays = [m for m in malay_choices(row["malay"]) if "..." not in m]
        thais = [t for t in thai_choices(row["thai"]) if "..." not in t]
        for m in [m for m in malays if m.lower().strip("?") in SKIPPED_MALAY]:
            notes.append(
                f"not used: '{m}' = '{row['thai']}' ({SKIPPED_MALAY[m.lower().strip('?')]})"
            )
            malays.remove(m)
        if not thais or (not malays and row["malay"] != "-"):
            continue
        # a cell with as many Malay as Thai choices pairs them in order
        # (Laut / Lauk = ทะเล / กับข้าว); otherwise they are synonyms
        paired = len(malays) == len(thais)
        meaning = {
            m: THAI_FOR_MALAY.get(
                re.sub(r"[^a-z0-9]+", "", m.lower()), thais[i] if paired else thais[0]
            )
            for i, m in enumerate(malays)
        }
        dialect = row["category"] in DIALECT_CATEGORY_NAMES or any(
            mark in row["category"] for mark in DIALECT_CATEGORIES
        )
        bank.append((row, malays, thais, meaning, dialect))

    # Malay -> Thai
    for _row, malays, _thais, meaning, _dialect in bank:
        for malay in malays:
            key = re.sub(r"[^a-z0-9]+", "", malay.lower())
            if key and key not in malay_seen and not meaning[malay].isdigit():
                malay_seen.add(key)
                phrases.append({"malay": malay.strip(" ?"), "thai": meaning[malay]})

    # dialect words the sound rules cannot derive from the Malay word; they are
    # found first so that the rules below already use them
    pattani_script.LEXICON = dict(BUILTIN_LEXICON)
    for row, malays, _thais, _meaning, dialect in bank:
        if dialect and len(malays) == 1 and " / " not in row["reading"]:
            word, reading = malays[0].lower().strip(" ?"), clean_reading(row["reading"])
            if (
                re.fullmatch(r"[a-z]+( [a-z]+)?", word)
                and word not in BUILTIN_LEXICON
                and not re.search("[A-Za-z]", reading)
                and loose(malay_to_pattani(word)) != loose(reading)
            ):
                lexicon[word] = reading
    pattani_script.LEXICON.update(lexicon)

    # Thai -> Pattani: readings that are dialect first, then the Standard Malay
    # rows turned into Pattani by the sound rules
    for want_dialect in (True, False):
        for row, malays, thais, meaning, dialect in bank:
            if dialect != want_dialect:
                continue
            for thai in thais:
                if thai.isdigit():
                    continue
                if dialect:
                    pattani = clean_reading(row["reading"])
                elif malays:
                    # the Malay phrase that carries this meaning
                    malay = next((m for m in malays if meaning[m] == thai), malays[0])
                    pattani = malay_to_pattani(malay)
                else:
                    continue
                if re.search("[A-Za-z]", pattani):
                    notes.append(
                        f"not used for Thai -> Pattani: '{row['malay']}' (no Thai-script form)"
                    )
                    continue
                add(thai, pattani, "term", "word bank")
    return phrasebook, phrases, lexicon, notes


def reading_choices(reading: str) -> list[str]:
    """The separate readings written in one cell, like malay_choices()."""
    return [c.replace("_", "").strip(" ?") for c in malay_choices(reading)]


def build_standard_sentences(
    pairs_path: Path, glossary_path: Path, bank_path: Path | None
) -> list[dict]:
    """The sentences a doctor can play: Thai, its other wordings, and the
    Pattani text exactly as a person wrote it."""
    sentences: list[dict] = []
    taken: set[str] = set()

    def add(ident: str, category: str, thais: list[str], pattani: str, source: str):
        thais = [t for t in thais if t and thai_key(t) not in taken]
        pattani = " ".join(READING_FIXES.get(pattani, pattani).split())
        if not thais or not pattani or re.search("[A-Za-z]", pattani):
            return
        taken.update(thai_key(t) for t in thais)
        sentences.append(
            {
                "id": ident,
                "category": category,
                "thai": thais[0],
                "variants": thais[1:],
                "pattani": pattani,
                "source": source,
            }
        )

    # the word bank; where it lists a dialect reading for a meaning, that one
    # is taken before a Standard Malay reading of the same meaning
    rows = read_word_bank(bank_path) if bank_path else []
    for want_dialect in (True, False):
        for n, row in enumerate(rows, 2):  # 2 = the row's line in the CSV
            dialect = row["category"] in DIALECT_CATEGORY_NAMES or any(
                mark in row["category"] for mark in DIALECT_CATEGORIES
            )
            if dialect != want_dialect or "..." in row["thai"] + row["reading"]:
                continue
            malays = [m for m in malay_choices(row["malay"]) if "..." not in m]
            readings = reading_choices(row["reading"])
            aligned = len(readings) == len(malays)  # one reading per Malay phrase
            keep = [
                k
                for k, m in enumerate(malays)
                if m.lower().strip("?") not in SKIPPED_MALAY
            ]
            if malays and not keep:
                continue
            if aligned:
                readings = [readings[k] for k in keep]
            malays = [malays[k] for k in keep]
            thais = [t for t in thai_choices(row["thai"]) if not t.isdigit()]
            if not thais or not readings:
                continue

            if len(thais) == len(malays) > 1:
                # one cell holding several separate entries: kanan / kiri
                for k, thai in enumerate(thais):
                    reading = readings[k] if aligned else readings[0]
                    add(
                        f"wb{n:03d}{'abcdef'[k]}",
                        row["category"],
                        [thai],
                        reading,
                        "word bank",
                    )
                continue
            # several Thai wordings of one meaning; its reading is that of the
            # Malay phrase which carries this meaning (Jantung, not Hati, for หัวใจ)
            reading = readings[0]
            if aligned:
                for k, malay in enumerate(malays):
                    key = re.sub(r"[^a-z0-9]+", "", malay.lower())
                    if THAI_FOR_MALAY.get(key, thais[0]) == thais[0]:
                        reading = readings[k]
                        break
            add(f"wb{n:03d}", row["category"], thais, reading, "word bank")

    if glossary_path.exists():
        for n, row in enumerate(read_jsonl(glossary_path), 1):
            thais = [t.strip() for t in row["thai"].split("/")]
            add(
                f"gl{n:02d}",
                "คำศัพท์อาการเจ็บป่วย (ภาษามลายูถิ่น)",
                thais,
                row["pattani"],
                "glossary",
            )

    # what the interviewer asked the patients
    for row in read_jsonl(pairs_path):
        if row.get("has_parallel") and row.get("speaker") == "ถาม":
            add(
                f"iv{row['person']}{row['tape']}-{row['line_no']:03d}",
                "คำถามซักประวัติ (จากบทสัมภาษณ์)",
                [" ".join((row.get("thai") or "").split())],
                row.get("malay_thai") or "",
                "interview",
            )
    return sentences


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    data = ROOT / "pattani_dataset"
    banks = sorted((ROOT / "data_thai_to_pattani").glob("*.csv"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(data / "pairs.jsonl"))
    ap.add_argument("--glossary", default=str(data / "glossary.jsonl"))
    ap.add_argument("--word-bank", default=str(banks[0]) if banks else None)
    ap.add_argument("--out", default=str(data / "phrasebook.jsonl"))
    ap.add_argument("--phrases-out", default=str(data / "medical_phrases.jsonl"))
    ap.add_argument("--lexicon-out", default=str(data / "lexicon.json"))
    ap.add_argument("--sentences-out", default=str(data / "standard_sentences.jsonl"))
    args = ap.parse_args()

    bank = Path(args.word_bank) if args.word_bank else None
    phrasebook, phrases, lexicon, notes = build(
        Path(args.pairs), Path(args.glossary), bank
    )

    with Path(args.out).open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in phrasebook)
    n_terms = sum(e["kind"] == "term" for e in phrasebook)
    print(
        f"Wrote {len(phrasebook)} Thai -> Pattani entries "
        f"({len(phrasebook) - n_terms} sentences, {n_terms} terms) -> {args.out}"
    )
    if bank:
        with Path(args.phrases_out).open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in phrases)
        print(f"Wrote {len(phrases)} Malay -> Thai phrases -> {args.phrases_out}")
        with Path(args.lexicon_out).open("w", encoding="utf-8", newline="\n") as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"Wrote {len(lexicon)} dialect words -> {args.lexicon_out}")
    sentences = build_standard_sentences(Path(args.pairs), Path(args.glossary), bank)
    with Path(args.sentences_out).open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in sentences)
    print(f"Wrote {len(sentences)} standard sentences -> {args.sentences_out}")
    for note in notes:
        print("  note:", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())

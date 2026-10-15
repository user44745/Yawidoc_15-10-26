#!/usr/bin/env python
"""
parse_transcripts.py — Stage 1 of the Pattani fine-tune pipeline.

Parse the interview transcripts in `สัมภาษ/` (UTF-8 .md files in the format
`speaker: <Malay-dialect-in-Thai-script> = <Central Thai>`) into a clean
JSONL file `pattani_dataset/pairs.jsonl` used by the later stages.

Row schema (one JSON object per line):
    {
      "person": 1 | 2 | 3,          # which interviewee ("คนที่ N")
      "tape": 1 | 2,                # tape index within the person's audio
      "line_no": int,               # original line number in the .md
      "speaker": str,               # normalized speaker label
      "malay_thai": str,            # Malay-dialect text in Thai script
      "thai": str | null,           # Central Thai translation ("=" right side)
      "note": str | null,           # parenthetical annotations extracted
      "has_parallel": bool,          # True when the line contains an "=" pair
    }

Audio-file mapping (for align_audio.py):
    person 1, tape 1 -> "คนที่ 1.m4a"      (~250 s)
    person 1, tape 2 -> "คนที่1.m4a"      (~116 s)
    person 2, tape 1 -> "คนที่2.m4a"      (~378 s)
    person 3, tape 1 -> "คนที่3.m4a"      (~436 s)

Usage:
    uv run python parse_transcripts.py
    uv run python parse_transcripts.py --src "สัมภาษ" --out pattani_dataset
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Speaker labels seen across the three transcripts (all follow "label:").
# Everything is normalized to these canonical names.
SPEAKER_ALIASES = {
    "ถาม": "ถาม",
    "ผู้ถาม": "ถาม",
    "ตอบ": "ตอบ",
    "ผู้สัม": "ตอบ",
    "ผู้ให้สัม": "ตอบ",
    "เมาะ": "ตอบ",
    "ลูกผู้ให้สัม": "ตอบ",
    "พี่เจน": "ตอบ",
    "ก้ะ": "ตอบ",
    "บอก": "ตอบ",
}
SPEAKER_RE = re.compile(
    r"^\s*("
    + "|".join(sorted(SPEAKER_ALIASES, key=len, reverse=True))
    + r")\s*[:：]\s*(.*)$",
    re.UNICODE,
)
TAPE_RE = re.compile(r"^\s*เทป\s*(\d+)\s*$", re.UNICODE)
NOTE_RE = re.compile(r"\(([^)]*)\)", re.UNICODE)


def strip_notes(text: str) -> tuple[str, str | None]:
    """Pull parenthetical annotations out of `text`.

    Returns (clean_text, joined_notes). Keeps Thai-script text usable for
    romanization; the notes go to the `note` field instead of being lost.
    """
    notes = NOTE_RE.findall(text)
    clean = NOTE_RE.sub(" ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    note = "; ".join(n.strip() for n in notes if n.strip()) or None
    return clean, note


def parse_line(raw: str) -> dict | None:
    """Parse one transcript line into a partial row (no person/tape yet).

    Returns None for lines that carry no usable utterance (headers, blank
    lines, pure-Thai summary lines without a speaker label).
    """
    line = raw.strip()
    if not line or line.startswith("#"):
        return None

    m = SPEAKER_RE.match(line)
    if not m:
        # Lines without a speaker label are interviewer summaries /
        # clarifications in Central Thai — keep them for traceability
        # but mark them as non-parallel, non-utterance rows.
        body, note = strip_notes(line)
        if not body:
            return None
        return {
            "speaker": "สรุป",
            "malay_thai": "",
            "thai": body,
            "note": note,
            "has_parallel": False,
        }

    label, body = SPEAKER_ALIASES[m.group(1)], m.group(2).strip()
    body, note = strip_notes(body)
    if not body:
        return None

    if "=" in body:
        malay, _, thai = body.partition("=")
        malay, thai = malay.strip(), thai.lstrip("|").strip()
        if not malay:  # e.g. "ตอบ: = ไม่มี" — nothing spoken on the left
            return None
        return {
            "speaker": label,
            "malay_thai": malay,
            "thai": thai or None,
            "note": note,
            "has_parallel": True,
        }
    # No "=": either pure dialect or a pure-Thai answer. Keep as-is;
    # align_audio.py decides which of these are usable ASR labels.
    return {
        "speaker": label,
        "malay_thai": body,
        "thai": None,
        "note": note,
        "has_parallel": False,
    }


def parse_file(path: Path, person: int) -> list[dict]:
    rows: list[dict] = []
    tape = 1
    text = path.read_text(encoding="utf-8-sig")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        tape_m = TAPE_RE.match(raw)
        if tape_m:
            tape = int(tape_m.group(1))
            continue
        row = parse_line(raw)
        if row is None:
            continue
        row.update(person=person, tape=tape, line_no=line_no)
        rows.append(row)
    return rows


def find_transcripts(src: Path) -> dict[int, Path]:
    """Map person number -> transcript .md path.

    Header conventions in the files: person 1 -> "# คนที่1",
    person 2 -> "คนที่สอง", person 3 -> "คนที่3". Inner folder names use
    Thai script (mojibake in some shells); Python rglob handles them fine.
    """
    headers = {1: ("คนที่1",), 2: ("คนที่สอง", "คนที่2"), 3: ("คนที่3",)}
    found: dict[int, Path] = {}
    for path in sorted(src.rglob("*.md")):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        head = next((l for l in text.splitlines() if l.strip()), "")
        for person, patterns in headers.items():
            if person not in found and any(p in head for p in patterns):
                found[person] = path
    return found


def main() -> None:
    # Windows consoles default to a legacy code page (cp874 on Thai systems)
    # that cannot encode every character; force UTF-8 so printing Thai file
    # names never crashes with UnicodeEncodeError (same pattern as gen_jawi.py).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Parse สัมภาษ transcripts into pattani_dataset/pairs.jsonl"
    )
    parser.add_argument("--src", default=str(ROOT / "สัมภาษ"))
    parser.add_argument("--out", default=str(ROOT / "pattani_dataset"))
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    transcripts = find_transcripts(src)
    if not transcripts:
        sys.exit(f"No transcripts found under {src}")

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for person in sorted(transcripts):
        path = transcripts[person]
        rows.extend(parse_file(path, person))
        print(f"person {person}: {path.name} -> {len(rows)} rows (cumulative)")

    dest = out / "pairs.jsonl"
    with dest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parallel = sum(r["has_parallel"] for r in rows)
    print(f"wrote {len(rows)} rows to {dest} ({parallel} with '=' parallel pairs)")
    by_speaker = {}
    for r in rows:
        by_speaker[r["speaker"]] = by_speaker.get(r["speaker"], 0) + 1
    print("speakers:", ", ".join(f"{k}={v}" for k, v in sorted(by_speaker.items())))


if __name__ == "__main__":
    main()

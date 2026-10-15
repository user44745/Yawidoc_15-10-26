# src/open_llm_vtuber/translate/pattani_script.py
"""Standard Malay (Latin script) -> Pattani Malay written in Thai script.

Rule-based and deterministic. Applies the regular sound correspondences
between Standard Malay and the Kelantan-Pattani dialect, spelled the way the
interview transcripts in pattani_dataset/pairs.jsonl spell them, e.g.

    makan -> มาแก      ada -> อาดอ       sakit -> ซาเกะ     perut -> ปือโระ
    rumah -> รูเมาะห์    anak -> อาเนาะ    lepas -> ลือปะห์    tidur -> ตีโด
    sampai -> ซาปา     tinggi -> ตีงี     jantung -> ยาตง    dua -> ดูวอ

Words whose dialect form is not predictable from the sound rules come from
LEXICON: forms attested in the transcripts (BUILTIN_LEXICON) plus the dialect
words of the word bank (pattani_dataset/lexicon.json, written by
build_phrasebook.py).
"""

import json
import re
from pathlib import Path

VOWELS = "aeiou"
DIGRAPHS = ("ng", "ny", "sy", "kh", "gh")

ONSET = {
    "b": "บ", "c": "จ", "d": "ด", "f": "ป", "g": "ก", "h": "ฮ", "j": "ย",
    "k": "ก", "l": "ล", "m": "ม", "n": "น", "p": "ป", "q": "ก", "r": "ร",
    "s": "ซ", "t": "ต", "v": "ว", "w": "ว", "x": "ซ", "y": "ย", "z": "ซ",
    "ng": "ง", "ny": "ย", "sy": "ช", "kh": "ค", "gh": "ก",
}  # fmt: skip
CODA = {
    "b": "บ", "c": "จ", "d": "ด", "f": "ฟ", "g": "ก", "j": "จ", "k": "ก",
    "l": "ล", "m": "ม", "n": "น", "p": "บ", "q": "ก", "s": "ส", "t": "ด",
    "v": "ฟ", "w": "ว", "x": "ก", "y": "ย", "z": "ส",
    "ng": "ง", "ny": "น", "sy": "ช", "kh": "ค", "gh": "ก",
}  # fmt: skip

# The dialect simplifies a nasal + stop made at the same place of articulation:
# before a voiceless stop the nasal is lost (sampai -> ซาปา, nanti -> นาตี),
# before a voiced stop the stop is lost (tinggi -> ตีงี, sembilan -> มีแล).
NASAL_LOST = {("m", "p"), ("n", "t"), ("ng", "k"), ("n", "c"), ("n", "j")}
STOP_LOST = {("m", "b"), ("n", "d"), ("ng", "g")}

NASALS = {"m", "n", "ng", "ny"}
VELAR_STOPS = {"k", "g", "q", "x"}
STOPS = {"p", "t", "b", "d", "c", "j"} | VELAR_STOPS
SIBILANTS = {"s", "z", "sy"}
LIQUIDS = {"r", "l"}

# Dialect words that the sound rules cannot derive. Every value is spelled as
# it appears in pattani_dataset/pairs.jsonl.
BUILTIN_LEXICON = {
    "tidak ada": "ตะเดาะ",
    "tak ada": "ตะเดาะ",
    "tiada": "ตะเดาะ",
    "rumah sakit": "เมาะห์ซาเกะ",
    "hospital": "เมาะห์ซาเกะ",
    "tidak": "เดาะ",
    "tak": "เดาะ",
    "sudah": "เดาะห์",
    "telah": "เดาะห์",
    "apa": "กาปอ",
    "berapa": "วาปอ",
    "kepala": "ปาลอ",
    "doktor": "บอมอ",
    "ubat": "ฮูบะ",
    "datang": "มารี",
    "sekarang": "ลอนิง",
    "kini": "ลอนิง",
    "pernah": "แบซอ",
    "selalu": "เซาะมอ",
    "sentiasa": "เซาะมอ",
    "kalau": "กาแด",
    "jika": "กาแด",
    "dia": "ยอ",
    "mereka": "ยอ",
    "yang": "เฮาะ",
    "dan": "ดืองา",
    "dengan": "ดืองา",
    "ini": "นิง",
    "itu": "ตู",
    "sini": "ซีนิง",
    "rasa": "วาซอ",
    "berasa": "วาซอ",
    "merasa": "วาซอ",
    "umur": "ออมอ",
    "orang": "ออแร",
    "keluar": "ตูเบะ",
    "boleh": "บูเละห์",
    "semalam": "มาแรง",
    "kelmarin": "มาแรง",
    "sedikit": "ซีกิ",
    "sikit": "ซีกิ",
    "kena": "คือนอ",
    "perlu": "คือนอ",
    "mesti": "คือนอ",
    "harus": "คือนอ",
    "tengok": "แตเงาะ",
    "lihat": "แตเงาะ",
    "beri": "วี",
    "bagi": "วี",
    "hendak": "เนาะ",
    "mahu": "เนาะ",
    "tetapi": "ตาปี",
    "juga": "ยูเฆาะ",
    "lagi": "ลาฆี",
}

# The word bank adds to the transcripts' words and never replaces them.
LEXICON_FILE = Path(__file__).parents[3] / "pattani_dataset" / "lexicon.json"
LEXICON = dict(BUILTIN_LEXICON)
if LEXICON_FILE.exists():
    with LEXICON_FILE.open(encoding="utf-8") as _f:
        LEXICON.update(
            {k: v for k, v in json.load(_f).items() if k not in BUILTIN_LEXICON}
        )

# "-kah" on these makes a content question, not a yes/no question.
WH_WORDS = {"apa", "siapa", "bila", "mana", "berapa", "bagaimana", "kenapa", "mengapa"}
# Sentence-final yes/no question particle, as in อาดอซาเกะปาลอเดาะ.
QUESTION_PARTICLE = "เดาะ"
# Standard Malay says "already" before the verb, the dialect says it last:
# วาปอฮารีเดาะห์ (berapa hari doh), ลามอเดาะห์ (lama doh).
PERFECTIVE = {"sudah", "telah", "dah"}
PERFECTIVE_PARTICLE = "เดาะห์"
# ... but only within one clause, so sentences with these are left alone.
CONJUNCTIONS = {"dan", "tetapi", "tapi", "kemudian", "lalu", "kerana", "sebab", "atau"}
# The transcripts address a patient by a kin term or not at all, never with
# a Standard Malay "you".
SECOND_PERSON = {"anda", "awak", "kamu", "engkau", "kau"}

WORD = re.compile(r"[A-Za-z']+")

# Sentence-final Thai polite particles; Pattani Malay has no equivalent.
POLITE_TAIL = re.compile(r"(นะ|น่ะ)?(ครับผม|ครับ|คับ|ค่ะ|คะ|ค่า|จ้ะ|จ๊ะ|ฮะ)$")
# Spaces and punctuation. Thai vowel signs and tone marks are letters here:
# \W would drop them and make หัก and หก, or ไข้ and ไข่, the same text.
NOT_TEXT = re.compile(r"[^0-9A-Za-z\u0E01-\u0E3A\u0E40-\u0E4E]+")


def collapse_repeats(text: str) -> str:
    """What speech recognition repeats on short recordings, once: one word
    written three or more times (จุด จุด จุด) or a whole phrase written twice
    (Sakit perut, sakit perut)."""
    words = [w.strip(".,!?").lower() for w in text.split()]
    if len(words) >= 3 and len(set(words)) == 1:
        return text.split()[0].strip(".,!?")
    half = len(words) // 2
    if half >= 2 and len(words) % 2 == 0 and words[:half] == words[half:]:
        return " ".join(text.split()[:half]).strip(".,!?")
    return text


def thai_key(text: str) -> str:
    """What identifies a Thai sentence: its letters, without spaces,
    punctuation, the repeat mark and a polite ending."""
    text = NOT_TEXT.sub("", text).replace("ๆ", "")
    while True:
        stripped = POLITE_TAIL.sub("", text)
        if stripped == text or not stripped:
            return text
        text = stripped


SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def _units(word: str) -> list[str]:
    """Split a lowercase Malay word into vowels and consonants (with digraphs)."""
    units, i = [], 0
    while i < len(word):
        if word[i : i + 2] in DIGRAPHS:
            units.append(word[i : i + 2])
            i += 2
        else:
            units.append(word[i])
            i += 1
    return units


def _syllables(units: list[str]) -> list[dict]:
    """Group units into syllables: {"onset": [...], "vowel": str, "coda": str | None}."""
    syllables: list[dict] = []
    pending: list[str] = []
    i = 0
    while i < len(units):
        unit = units[i]
        if unit not in VOWELS:
            pending.append(unit)
            i += 1
            continue

        vowel = unit
        # ai / au / oi are diphthongs only at the very end of a word
        if i == len(units) - 2 and unit + units[i + 1] in ("ai", "au", "oi"):
            vowel = unit + units[i + 1]
            i += 1

        if not syllables:
            onset = pending
        elif not pending:
            onset = [_glide(syllables[-1]["vowel"], vowel)]
        elif len(pending) == 1:
            onset = pending
        elif (pending[0], pending[1]) in NASAL_LOST:
            onset = pending[1:]
        elif (pending[0], pending[1]) in STOP_LOST:
            onset = [pending[0]] + pending[2:]
        else:
            syllables[-1]["coda"] = pending[0]
            onset = pending[1:]

        syllables.append({"onset": onset, "vowel": vowel, "coda": None})
        pending = []
        i += 1

    if syllables and pending:
        syllables[-1]["coda"] = pending[0]
    return syllables


def _glide(previous: str, vowel: str) -> str:
    """Consonant that links two adjacent vowels (dua -> ดูวอ, siap -> ซียะ)."""
    if previous == "i":
        return "y"
    if previous == "u":
        return "w"
    if vowel[0] in "ie":
        return "y"
    if vowel[0] in "uo":
        return "w"
    return ""  # glottal stop, written with อ


def _onset(onset: list[str]) -> str:
    return "".join(ONSET.get(c, "") for c in onset) or "อ"


def _medial(o: str, vowel: str, coda: str | None) -> str:
    """Spell a non-final syllable."""
    if coda is None or coda in ("r", "h"):
        return {"a": o + "า", "i": o + "ี", "u": o + "ู", "o": o + "อ"}.get(
            vowel, o + "ือ"
        )
    x = CODA.get(coda, "")
    return {
        "a": o + "ั" + x,
        "i": o + "ิ" + x,
        "u": o + "ุ" + x,
        "o": o + "อ" + x,
    }.get(vowel, o + "ึ" + x)


def _final(o: str, vowel: str, coda: str | None) -> str:
    """Spell the last syllable of a word, where the dialect differs most."""
    if vowel in ("ai", "au"):  # sampai -> ซาปา
        return o + "า"
    if vowel == "oi":
        return o + "อย"

    if coda is None:
        return {"a": o + "อ", "i": o + "ี", "u": o + "ู", "o": o + "อ"}.get(
            vowel, o + "ือ"
        )
    if coda in NASALS:
        return {
            "a": "แ" + o,  # makan -> มาแก
            "i": o + "ิง",  # pening -> ปือนิง
            "u": o + "ง",  # turun -> ตูรง
            "o": o + "อง",
        }.get(vowel, "เ" + o + "ง")
    if coda == "h":
        return {
            "a": "เ" + o + "าะห์",  # rumah -> รูเมาะห์
            "o": "เ" + o + "าะห์",
            "u": "โ" + o + "ะห์",  # puluh -> ปูโละห์
        }.get(vowel, "เ" + o + "ะห์")  # lebih -> ลือเบะห์
    if coda in SIBILANTS:
        return {
            "a": o + "ะห์",  # lepas -> ลือปะห์
            "i": o + "ิห์",
            "u": o + "ุห์",
            "o": "เ" + o + "าะห์",
        }.get(vowel, "เ" + o + "ะห์")
    if coda in STOPS:
        if vowel == "a":
            # anak -> อาเนาะ, but ubat -> ฮูบะ
            return "เ" + o + "าะ" if coda in VELAR_STOPS else o + "ะ"
        return {
            "o": "เ" + o + "าะ",  # rokok -> รอเกาะ
            "u": "โ" + o + "ะ",  # batuk -> บาโตะ
        }.get(vowel, "เ" + o + "ะ")  # sakit -> ซาเกะ
    if coda in LIQUIDS:
        return {
            "a": o + "า",  # ketar -> กือตา
            "i": "เ" + o,
            "u": "โ" + o,  # tidur -> ตีโด
            "o": o + "อ",
        }.get(vowel, "เ" + o + "อ")
    # any other final consonant (f, v, w, y) is kept
    x = CODA.get(coda, "")
    return {
        "a": o + "ั" + x,
        "i": o + "ิ" + x,
        "u": o + "ุ" + x,
        "o": o + "อ" + x,
    }.get(vowel, "เ" + o + x)


def word_to_pattani(word: str) -> str:
    """Spell one Standard Malay word as Pattani Malay in Thai script."""
    word = word.lower().replace("'", "")
    if word in LEXICON:
        return LEXICON[word]
    syllables = _syllables(_units(word))
    if not syllables:  # no vowel: an abbreviation, leave it alone
        return word
    last = len(syllables) - 1
    return "".join(
        (_final if i == last else _medial)(_onset(s["onset"]), s["vowel"], s["coda"])
        for i, s in enumerate(syllables)
    )


def _sentence_to_pattani(sentence: str) -> str:
    is_question = sentence.rstrip().endswith("?")
    # words, plus anything else worth keeping (numbers, Thai text)
    tokens = re.findall(r"[A-Za-z']+|[^\sA-Za-z'.,;:!?\"()\-]+", sentence)
    words = [t.lower() for t in tokens]
    spoken = sum(1 for t in tokens if WORD.fullmatch(t))
    one_clause = not re.search(r"[,;]", sentence) and not CONJUNCTIONS & set(words)

    out: list[str] = []
    yes_no = False
    perfective_at = None  # where "already" stood, once it is taken out
    i = 0
    while i < len(words):
        word = words[i]
        if not WORD.fullmatch(tokens[i]):
            out.append(tokens[i])
            i += 1
            continue
        if word in SECOND_PERSON and spoken >= 3:
            i += 1
            continue
        if word in PERFECTIVE and one_clause and perfective_at is None:
            perfective_at = len(out)
            i += 1
            continue
        pair = f"{word} {words[i + 1]}" if i + 1 < len(words) else ""
        if pair in LEXICON:
            out.append(LEXICON[pair])
            i += 2
            continue
        # adakah / bolehkah ... mark a yes/no question; the dialect marks it
        # with a sentence-final particle instead
        if word.endswith("kah") and len(word) > 4:
            word = word[:-3]
            yes_no = yes_no or word not in WH_WORDS
        elif word == "ke" and is_question and i == len(words) - 1:
            yes_no = True
            i += 1
            continue
        out.append(word_to_pattani(word))
        i += 1

    if perfective_at is not None:
        # a yes/no question keeps its own word order; elsewhere it goes last
        out.insert(perfective_at if yes_no else len(out), PERFECTIVE_PARTICLE)
    if yes_no and out:
        out.append(QUESTION_PARTICLE)
    return " ".join(out)


def malay_to_pattani(text: str) -> str:
    """Convert Standard Malay text to Pattani Malay in Thai script."""
    sentences = SENTENCE_END.split(text.strip().lstrip("-–—• ").strip())
    return " ".join(filter(None, (_sentence_to_pattani(s) for s in sentences)))

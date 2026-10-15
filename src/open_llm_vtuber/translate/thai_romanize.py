# src/open_llm_vtuber/translate/thai_romanize.py
"""Thai script -> Latin letters, the way the Thai text is read aloud.

Used to show Pattani Malay written in Thai script in Latin letters as well,
e.g. อาดอ ซาเกะ ปาลอ เดาะ -> a-do sa-ke pa-lo do. Spelling follows the Royal
Thai General System (ue = ือ, ae = แ, ch = จ/ช ...); syllables are joined
with hyphens because Thai script does not mark word boundaries.

One addition for Pattani text: a final ห์ / ฮฺ / ห is written "h"
(เดาะห์ -> doh), which is what those spellings stand for.
"""

import re

CONSONANTS = {chr(c) for c in range(0x0E01, 0x0E2F)}  # ก .. ฮ
LEAD_VOWELS = set("เแโไใ")
VOWEL_SIGNS = set("ะาำัิีึืุู็")
SONORANTS = set("งญนมยรลว")
FINAL_H = "\x01"  # stands for ห์ / ฮฺ after preprocessing
THAI_CLUSTERS = {
    ("ก", "ร"), ("ก", "ล"), ("ก", "ว"), ("ข", "ร"), ("ข", "ล"), ("ข", "ว"),
    ("ค", "ร"), ("ค", "ล"), ("ค", "ว"), ("ต", "ร"), ("ป", "ร"), ("ป", "ล"),
    ("พ", "ร"), ("พ", "ล"), ("ผ", "ล"),
}  # fmt: skip

ONSET = {
    "ก": "k", "ข": "kh", "ฃ": "kh", "ค": "kh", "ฅ": "kh", "ฆ": "kh", "ง": "ng",
    "จ": "ch", "ฉ": "ch", "ช": "ch", "ซ": "s", "ฌ": "ch", "ญ": "y", "ฎ": "d",
    "ฏ": "t", "ฐ": "th", "ฑ": "th", "ฒ": "th", "ณ": "n", "ด": "d", "ต": "t",
    "ถ": "th", "ท": "th", "ธ": "th", "น": "n", "บ": "b", "ป": "p", "ผ": "ph",
    "ฝ": "f", "พ": "ph", "ฟ": "f", "ภ": "ph", "ม": "m", "ย": "y", "ร": "r",
    "ฤ": "rue", "ล": "l", "ฦ": "lue", "ว": "w", "ศ": "s", "ษ": "s", "ส": "s",
    "ห": "h", "ฬ": "l", "อ": "", "ฮ": "h",
}  # fmt: skip
FINAL = {
    "ก": "k", "ข": "k", "ค": "k", "ฆ": "k", "ง": "ng", "จ": "t", "ช": "t",
    "ซ": "t", "ฌ": "t", "ญ": "n", "ฎ": "t", "ฏ": "t", "ฐ": "t", "ฑ": "t",
    "ฒ": "t", "ณ": "n", "ด": "t", "ต": "t", "ถ": "t", "ท": "t", "ธ": "t",
    "น": "n", "บ": "p", "ป": "p", "พ": "p", "ฟ": "p", "ภ": "p", "ม": "m",
    "ย": "i", "ร": "n", "ล": "n", "ฬ": "n", "ว": "o", "ศ": "t", "ษ": "t",
    "ส": "t", "ห": "h", FINAL_H: "h",
}  # fmt: skip

TONE_MARKS = re.compile("[่-๋]")
# ห์ / ฮฺ spell a final h; any other consonant under ์ is silent
WRITTEN_H = re.compile("ห์|ฮฺ")
SILENT = re.compile("[ก-ฮ][ิุ]?์|[์ฺ]")
THAI_RUN = re.compile("[ก-๎\x01]+")


class _Reader:
    """Reads one run of Thai characters syllable by syllable."""

    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def at(self, k: int) -> str:
        return self.s[k] if 0 <= k < len(self.s) else ""

    def is_vowel_sign(self, k: int) -> bool:
        return self.at(k) in VOWEL_SIGNS and self.at(k) != ""

    def o_is_vowel(self, k: int) -> bool:
        """Is the อ at k the vowel of the consonant before it?

        อ is both a vowel (ดอ) and a silent initial (อา, ออ), so a row of them
        is read from the right: อาดอออแร is อา-ดอ-ออ-แร.
        """
        end = k
        while self.at(end) == "อ":
            end += 1
        run = end - k
        if self.is_vowel_sign(end):  # the last อ carries that vowel sign
            run -= 1
        elif self.at(end) == "" and run > 1:  # a trailing อ only lengthens: ฮออ
            run -= 1
        return run % 2 == 1

    def has_own_vowel(self, k: int) -> bool:
        """Is the consonant at k followed by a vowel written for it?"""
        nxt = self.at(k + 1)
        if self.is_vowel_sign(k + 1):
            return True
        if nxt == "อ":
            return self.o_is_vowel(k + 1)
        if nxt == "ว":  # บวม: ว between two consonants is the vowel "ua"
            return self.at(k + 2) in FINAL and not self.is_vowel_sign(k + 3)
        return False

    def takes_final(self, j: int) -> bool:
        """Does the consonant at j close the current syllable?"""
        f, g = self.at(j), self.at(j + 1)
        if f == FINAL_H:
            return True
        if f not in FINAL:
            return False
        if f == "ห":  # a plain ห closes a syllable only at the very end (แดห)
            return g == ""
        if g == "" or g in LEAD_VOWELS or g in (FINAL_H, "ๆ"):
            return True
        if g in VOWEL_SIGNS:
            return False
        if g == "อ":  # final unless that อ is this consonant's vowel
            return not self.o_is_vowel(j + 1)
        if g == "ว":
            return self.has_own_vowel(j + 1) and (f, g) not in THAI_CLUSTERS
        if (f, g) in THAI_CLUSTERS and self.has_own_vowel(j + 1):
            return False
        if self.has_own_vowel(j + 1):
            return True
        # g has no vowel of its own: "f + g" is a syllable like ลง / ตง / คน
        return g not in "งนม"

    def final(self, required: bool = False) -> str:
        if self.takes_final(self.i) or (required and self.at(self.i) in FINAL):
            self.i += 1
            return FINAL[self.at(self.i - 1)]
        return ""

    def final_h(self) -> str:
        """After a short vowel written with ะ only a final h can follow."""
        f = self.at(self.i)
        if f == FINAL_H or (f == "ห" and self.at(self.i + 1) == ""):
            self.i += 1
            return "h"
        return ""

    def onset(self, lead: str) -> str:
        first = self.at(self.i)
        self.i += 1
        nxt = self.at(self.i)
        # silent ห (หมอ, หยุด) and อ (อยู่, อย่าง) in front of the real initial
        if (first == "ห" and nxt in SONORANTS and nxt != "") or (
            first == "อ" and nxt == "ย" and not lead and self.is_vowel_sign(self.i + 1)
        ):
            self.i += 1
            return ONSET[nxt]
        if (first, nxt) in THAI_CLUSTERS:
            if lead in ("เ", "ไ", "ใ"):
                cluster = nxt != "ว" or self.is_vowel_sign(self.i + 1)
            elif lead:  # แ / โ take no vowel sign after the initial except ะ ็
                after = self.at(self.i + 1)
                cluster = after in ("", "ะ", "็") or (
                    nxt != "ว" and after in FINAL and not self.has_own_vowel(self.i + 1)
                )
            else:  # กลี, กลอง, and ตรง with its unwritten vowel
                after = self.at(self.i + 1)
                cluster = self.is_vowel_sign(self.i + 1) or (
                    nxt != "ว"
                    and (
                        self.has_own_vowel(self.i)
                        or (after in FINAL and not self.has_own_vowel(self.i + 1))
                    )
                )
            if cluster:
                self.i += 1
                return ONSET[first] + ONSET[nxt]
        return ONSET[first]

    def rime(self, lead: str) -> str:
        """Vowel plus final consonant, starting right after the initial."""
        v = self.at(self.i)
        if lead == "เ":
            if v == "า":
                self.i += 1
                if self.at(self.i) == "ะ":
                    self.i += 1
                    return "o" + self.final_h()
                return "ao"
            if v == "ะ":
                self.i += 1
                return "e" + self.final_h()
            if v == "ี" and self.at(self.i + 1) == "ย":
                self.i += 2
                return "ia" + self.final()
            if v == "ื" and self.at(self.i + 1) == "อ":
                self.i += 2
                return "uea" + self.final()
            if v == "ิ":
                self.i += 1
                return "oe" + self.final(required=True)
            if v == "็":
                self.i += 1
                return "e" + self.final(required=True)
            if v == "อ" and self.at(self.i + 1) == "ะ":
                self.i += 2
                return "oe"
            if v == "อ" and not self.is_vowel_sign(self.i + 1):
                self.i += 1
                return "oe" + self.final()
            if v == "ย" and not self.has_own_vowel(self.i):
                self.i += 1
                return "oei"
            return "e" + self.final()
        if lead == "แ":
            if v == "็":
                self.i += 1
                return "ae" + self.final(required=True)
            if v == "ะ":
                self.i += 1
                return "ae" + self.final_h()
            return "ae" + self.final()
        if lead == "โ":
            if v == "ะ":
                self.i += 1
                return "o" + self.final_h()
            return "o" + self.final()
        if lead:  # ไ / ใ
            if v == "ย" and not self.has_own_vowel(self.i):
                self.i += 1
            return "ai"

        if v == "็":  # ก็
            self.i += 1
            return "o"
        if v == "ั":
            self.i += 1
            if self.at(self.i) == "ว" and not self.has_own_vowel(self.i):
                self.i += 1
                return "ua"
            return "a" + self.final(required=True)
        if v in ("ิ", "ี"):
            self.i += 1
            return "i" + self.final()
        if v in ("ึ", "ื"):
            self.i += 1
            if v == "ื" and self.at(self.i) == "อ":
                self.i += 1
            return "ue" + self.final()
        if v in ("ุ", "ู"):
            self.i += 1
            return "u" + self.final()
        if v == "า":
            self.i += 1
            return "a" + self.final()
        if v == "ะ":
            self.i += 1
            return "a" + self.final_h()
        if v == "ำ":
            self.i += 1
            return "am"
        if v == "อ" and self.o_is_vowel(self.i):
            self.i += 1
            return "o" + self.final()
        if (
            v == "ว"
            and self.at(self.i + 1) in FINAL
            and not self.is_vowel_sign(self.i + 2)
        ):
            self.i += 1
            return "ua" + self.final(required=True)
        if v in CONSONANTS and not self.has_own_vowel(self.i):
            self.i += 1  # two bare consonants: ลง, ตง, คน
            return "o" + FINAL.get(v, "")
        if v == FINAL_H:
            self.i += 1
            return "oh"
        return "a"  # a bare consonant before another syllable: สบาย

    def syllables(self) -> list[str]:
        out: list[str] = []
        while self.i < len(self.s):
            ch = self.at(self.i)
            if ch == "ๆ":  # repeat mark
                self.i += 1
                if out:
                    out.append(out[-1])
                continue
            lead = ""
            if ch in LEAD_VOWELS:
                lead = ch
                self.i += 1
            if self.at(self.i) not in CONSONANTS:
                if not lead:
                    self.i += 1  # a stray mark
                continue
            start = self.i
            onset = self.onset(lead)
            # a lone อ with nothing to carry (ฮออ) is not a syllable
            if (
                not lead
                and self.s[start] == "อ"
                and self.i == start + 1
                and self.at(self.i) not in VOWEL_SIGNS | {"อ", "ว"} | CONSONANTS
            ):
                continue
            out.append(onset + self.rime(lead))
        return out


def thai_to_latin(text: str) -> str:
    """Spell Thai-script text in Latin letters; other characters are kept."""
    text = SILENT.sub("", WRITTEN_H.sub(FINAL_H, TONE_MARKS.sub("", text)))
    return THAI_RUN.sub(lambda m: "-".join(_Reader(m.group()).syllables()), text)
